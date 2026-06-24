"""Learned 4-class combiner over per-cut probabilities (idea #4, version 1).

Instead of routing a sample through a fixed hierarchy of hard 0.5 thresholds
(where one early wrong turn is unrecoverable), we treat each binary cut's
P(class=1) as a feature and let a single 4-class classifier learn the mapping
to stage 1..4. This replaces brittle routing with a soft, learned combiner.

Feature sets (scalar = P(class=1) per cut)
------------------------------------------
* "topology3" : the 3 cuts of the OOF-selected topology.
* "all"       : every available cut (10-dim) — more signal, still low-dim.
* "ordinal3"  : the 3 cumulative cuts [1_vs_234, 12_vs_34, 123_vs_4] = the
                ordinal encoding [P(>=2), P(>=3), P(>=4)].

Feature set (embedding = penultimate-layer features per cut)
------------------------------------------------------------
* "embed_ordinal3" : concat the penultimate embeddings of the 3 ordinal cuts
                (idea #4 "version 3"). High-dim (~3000-5000) on only ~229
                samples, so it is run through StandardScaler (+ optional PCA)
                inside the CV pipeline to fight the curse of dimensionality.
                NOT wired into the web app (too heavy; usually overfits).

Honesty (no leakage)
--------------------
* Train features  = OOF (each sample scored/embedded by a cut model that never
  trained on it). Scaler/PCA are fit inside each CV fold (sklearn Pipeline).
* Model selection = 5-fold CV macro-F1 on those OOF features.
* Test features   = soft-vote (scalar) / fold-averaged (embedding) on the test.
The outer test set is used only for the final report, never for selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .cuts_registry import CUTS
from .data import (
    filter_indices_for_cut,
    get_4class_labels,
    make_4class_eval_loader,
    make_cv_folds,
    make_outer_split,
)
from .eval import (
    bootstrap_classification_ci,
    binary_metrics,
    dump_json,
    save_classification_report,
    save_confusion_matrix,
)
from .hierarchy import CutPredictor, compute_oof_cut_p1
from .models import create_model
from .seed import SEED, set_seed

STAGE_NAMES = ["stage_1", "stage_2", "stage_3", "stage_4"]
ORDINAL_CUTS = ["1_vs_234", "12_vs_34", "123_vs_4"]  # [P(>=2), P(>=3), P(>=4)]


class _LabelOffsetClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a classifier that requires labels 0..K-1 (e.g. XGBoost) so it accepts
    our 1-based stage labels. Exposes classes_/predict/predict_proba in the
    original label space; clone-safe for cross_val_predict."""

    def __init__(self, base):
        self.base = base

    def fit(self, X, y):
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self.base_ = clone(self.base)
        self.base_.fit(X, y_enc)
        self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        return self._le.inverse_transform(self.base_.predict(X))

    def predict_proba(self, X):
        # base_.classes_ are 0..K-1 in the LabelEncoder order, so columns align
        # with self.classes_ (sorted original labels).
        return self.base_.predict_proba(X)


def candidate_models(seed: int = SEED) -> dict:
    """The user's wish list: boosting (HistGB / XGBoost / LightGBM), tree
    (RandomForest), and SVM with two kernels (+ logreg baseline).

    XGBoost / LightGBM are added only if installed."""
    models: dict = {
        "logreg": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed
        ),
        "svm_linear": SVC(
            kernel="linear", class_weight="balanced", random_state=seed
        ),
        "svm_rbf": SVC(
            kernel="rbf", class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample", random_state=seed
        ),
        "hist_gboost": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=seed
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = _LabelOffsetClassifier(XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
            random_state=seed, verbosity=0,
        ))
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            class_weight="balanced", random_state=seed, verbose=-1,
        )
    except Exception:
        pass
    return models


def _load_predictors(cut_names, out_root: Path, device: str) -> dict[str, CutPredictor]:
    predictors: dict[str, CutPredictor] = {}
    for cn in cut_names:
        cut_dir = out_root / "cuts" / cn
        if (cut_dir / "ensemble" / "winner.json").exists():
            predictors[cn] = CutPredictor(cut_dir, device)
        else:
            print(f"  [skip] {cn}: no ensemble/winner.json", flush=True)
    return predictors


def build_features(
    cut_names: list[str],
    out_root: Path,
    data_root: Path,
    splits_dir: Path,
    device: str,
    batch_size: int = 16,
):
    """Return (X_oof, y_tv, X_test, y_test, used_cuts) with 1-based labels."""
    set_seed(SEED)
    predictors = _load_predictors(cut_names, out_root, device)
    used = [cn for cn in cut_names if cn in predictors]
    if not used:
        raise FileNotFoundError("No usable cuts (run 05_build_ensemble first).")

    outer = make_outer_split(data_root, splits_dir / "outer_split.json")
    tv = sorted(int(i) for i in outer["train_val_idx"])
    labels = get_4class_labels(data_root).numpy()
    y_tv = labels[tv] + 1

    # OOF features for train_val (leak-free)
    X_oof = np.zeros((len(tv), len(used)), dtype=np.float64)
    for j, cn in enumerate(used):
        print(f"  [feat-oof] {cn}", flush=True)
        X_oof[:, j] = compute_oof_cut_p1(
            CUTS[cn], predictors[cn], data_root, splits_dir, tv, batch_size
        )

    # Test features = soft-vote of the 5 fold winners per cut
    test_loader = make_4class_eval_loader(data_root, outer["test_idx"], batch_size=batch_size)
    y_test: list[int] = []
    per_cut_test: dict[str, list[np.ndarray]] = {cn: [] for cn in used}
    for imgs, lbls in test_loader:
        for cn in used:
            per_cut_test[cn].append(predictors[cn].prob_class1(imgs))
        y_test.extend(int(l) + 1 for l in lbls.numpy().tolist())
    X_test = np.column_stack([np.concatenate(per_cut_test[cn]) for cn in used])
    y_test = np.asarray(y_test, dtype=np.int64)

    return X_oof, y_tv, X_test, y_test, used


# ---------------------------------------------------------------------------
# Embedding features (idea #4 "version 3"): penultimate-layer vectors per cut
# ---------------------------------------------------------------------------

def _final_linear(model, backbone: str):
    """The final classifier Linear of each backbone — its INPUT is the
    penultimate (pooled) embedding we want."""
    name = backbone.lower()
    if name.startswith("efficientnet"):
        return model.classifier[1]
    if name == "resnet50":
        return model.fc
    if name.startswith("convnext"):
        return model.classifier[2]
    if name.startswith("densenet"):
        return model.classifier
    raise ValueError(f"No embedding linear rule for {backbone}")


@torch.no_grad()
def _embeddings_on_loader(model, linear, loader, device: str) -> np.ndarray:
    """Capture the input to the final Linear (the pooled embedding) for a loader."""
    store: dict = {}

    def pre_hook(_m, inp):
        store["e"] = inp[0].detach().cpu().numpy().astype(np.float64)

    h = linear.register_forward_pre_hook(pre_hook)
    model.eval()
    feats = []
    for imgs, _ in loader:
        model(imgs.to(device))
        feats.append(store["e"])
    h.remove()
    return np.concatenate(feats) if feats else np.zeros((0, 1))


def _best_backbone_for_cut(out_root: Path, cut_name: str) -> str:
    """Pick the cut's backbone with the highest CV-mean macro-F1 (from
    summary.csv). One fixed backbone per cut keeps embedding dims consistent."""
    sp = out_root / "cuts" / cut_name / "summary.csv"
    df = pd.read_csv(sp)
    col = "cv_mean_macro_f1"
    df = df.dropna(subset=[col])
    return str(df.sort_values(col, ascending=False).iloc[0]["backbone"])


def build_embedding_features(
    cut_names: list[str],
    out_root: Path,
    data_root: Path,
    splits_dir: Path,
    device: str,
    batch_size: int = 16,
):
    """OOF + test penultimate embeddings for the given (full-coverage) cuts.

    For each cut: one fixed backbone, its 5 fold checkpoints.
      OOF  : each train_val sample embedded by the fold model that held it out.
      Test : average of the 5 fold models' embeddings (same backbone -> same dim).
    Returns (X_oof, y_tv, X_test, y_test, used_backbones)."""
    set_seed(SEED)
    outer = make_outer_split(data_root, splits_dir / "outer_split.json")
    tv = sorted(int(i) for i in outer["train_val_idx"])
    test_list = sorted(int(i) for i in outer["test_idx"])
    pos = {g: i for i, g in enumerate(tv)}
    labels = get_4class_labels(data_root).numpy()
    y_tv = labels[tv] + 1
    y_test = labels[test_list] + 1

    oof_blocks, test_blocks, used = [], [], []
    for cn in cut_names:
        cut = CUTS[cn]
        bb = _best_backbone_for_cut(out_root, cn)
        used.append(f"{cn}:{bb}")
        cut_tv = filter_indices_for_cut(data_root, outer["train_val_idx"], cut)
        folds = make_cv_folds(data_root, cut_tv)

        oof_rows: dict[int, np.ndarray] = {}
        test_sum = None
        n_test_models = 0
        for fi, (_tr, va) in enumerate(folds):
            ckpt = out_root / "cuts" / cn / "cv" / f"fold_{fi}" / bb / f"best_{bb}.pth"
            if not ckpt.exists():
                continue
            m = create_model(bb, num_classes=2).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            lin = _final_linear(m, bb)

            va = [int(g) for g in va]
            va_loader = make_4class_eval_loader(data_root, va, batch_size=batch_size)
            emb_va = _embeddings_on_loader(m, lin, va_loader, device)
            for j, g in enumerate(va):
                oof_rows[g] = emb_va[j]

            test_loader = make_4class_eval_loader(data_root, test_list, batch_size=batch_size)
            emb_te = _embeddings_on_loader(m, lin, test_loader, device)
            test_sum = emb_te if test_sum is None else test_sum + emb_te
            n_test_models += 1
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        D = next(iter(oof_rows.values())).shape[0]
        X_cut_oof = np.zeros((len(tv), D), dtype=np.float64)
        for g, vec in oof_rows.items():
            X_cut_oof[pos[g]] = vec
        X_cut_test = test_sum / max(n_test_models, 1)
        print(f"  [embed] {cn} bb={bb} dim={D}", flush=True)
        oof_blocks.append(X_cut_oof)
        test_blocks.append(X_cut_test)

    X_oof = np.concatenate(oof_blocks, axis=1)
    X_test = np.concatenate(test_blocks, axis=1)
    return X_oof, y_tv, X_test, y_test, used


def _wrap_pipeline(clf, pca_n: int):
    steps = [("scaler", StandardScaler())]
    if pca_n and pca_n > 0:
        steps.append(("pca", PCA(n_components=pca_n, random_state=SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def run_embedding_combiner(
    featureset: str,
    cut_names: list[str],
    out_root: Path,
    data_root: Path,
    splits_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
    seed: int = SEED,
    n_boot: int = 2000,
    pca_n: int = 50,
) -> dict:
    print(f"\n[combiner] === featureset='{featureset}' (embeddings, pca={pca_n}) ===", flush=True)
    X_oof, y_tv, X_test, y_test, used = build_embedding_features(
        cut_names, out_root, data_root, splits_dir, device, batch_size
    )
    print(f"[combiner] X_oof={X_oof.shape} X_test={X_test.shape}", flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    eff_pca = min(pca_n, X_oof.shape[1], 150) if pca_n and pca_n > 0 else 0
    rows = []
    for name, clf in candidate_models(seed).items():
        pipe = _wrap_pipeline(clf, eff_pca)
        oof_pred = cross_val_predict(pipe, X_oof, y_tv, cv=cv)
        oof_macro = float(f1_score(y_tv, oof_pred, average="macro", zero_division=0))
        rows.append({"model": name, "oof_macro_f1": oof_macro})
        print(f"  [combiner] {name:14s} oof_macro_f1={oof_macro:.4f}", flush=True)

    rows.sort(key=lambda r: r["oof_macro_f1"], reverse=True)
    best_name = rows[0]["model"]
    best_pipe = _wrap_pipeline(candidate_models(seed)[best_name], eff_pca)
    best_pipe.fit(X_oof, y_tv)
    y_pred = best_pipe.predict(X_test)

    m = binary_metrics(y_test, y_pred)
    m["macro_f1"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    ci = bootstrap_classification_ci(y_test, y_pred, n_boot=n_boot, seed=seed)

    comb_dir = out_root / "combiner" / featureset
    comb_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(comb_dir / "model_selection.csv", index=False, encoding="utf-8-sig")
    save_confusion_matrix(
        y_test - 1, y_pred - 1, STAGE_NAMES,
        comb_dir / "confusion_matrix_test.png", title=f"{featureset} {best_name}",
    )
    save_classification_report(
        y_test - 1, y_pred - 1, STAGE_NAMES, comb_dir / "classification_report_test.txt"
    )
    result = {
        "featureset": featureset,
        "backbones": used,
        "n_features": int(X_oof.shape[1]),
        "pca_n": int(eff_pca),
        "best_model": best_name,
        "oof_macro_f1": rows[0]["oof_macro_f1"],
        "test_macro_f1": m["macro_f1"],
        "test_accuracy": m["accuracy"],
        "test_macro_f1_ci_low": ci["macro_f1_ci_low"],
        "test_macro_f1_ci_high": ci["macro_f1_ci_high"],
        "all_models": rows,
    }
    dump_json(result, comb_dir / "best.json")
    print(
        f"[combiner] {featureset}: best={best_name} oof={result['oof_macro_f1']:.4f} "
        f"test_macro_f1={result['test_macro_f1']:.4f} "
        f"(95% CI {ci['macro_f1_ci_low']:.3f}-{ci['macro_f1_ci_high']:.3f})",
        flush=True,
    )
    return result


def run_combiner(
    featureset: str,
    cut_names: list[str],
    out_root: Path,
    data_root: Path,
    splits_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
    seed: int = SEED,
    n_boot: int = 2000,
) -> dict:
    print(f"\n[combiner] === featureset='{featureset}' cuts={cut_names} ===", flush=True)
    X_oof, y_tv, X_test, y_test, used = build_features(
        cut_names, out_root, data_root, splits_dir, device, batch_size
    )
    print(f"[combiner] X_oof={X_oof.shape} X_test={X_test.shape} cuts={used}", flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for name, model in candidate_models(seed).items():
        oof_pred = cross_val_predict(model, X_oof, y_tv, cv=cv)
        oof_macro = float(f1_score(y_tv, oof_pred, average="macro", zero_division=0))
        rows.append({"model": name, "oof_macro_f1": oof_macro})
        print(f"  [combiner] {name:14s} oof_macro_f1={oof_macro:.4f}", flush=True)

    rows.sort(key=lambda r: r["oof_macro_f1"], reverse=True)
    best_name = rows[0]["model"]
    best_model = candidate_models(seed)[best_name]
    # SVC has no predict_proba unless probability=True; the web needs per-stage
    # probabilities, so enable it for the persisted SVM model.
    if best_name.startswith("svm"):
        best_model.set_params(probability=True)
    best_model.fit(X_oof, y_tv)
    y_pred = best_model.predict(X_test)

    m = binary_metrics(y_test, y_pred)
    m["macro_f1"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    ci = bootstrap_classification_ci(y_test, y_pred, n_boot=n_boot, seed=seed)

    comb_dir = out_root / "combiner" / featureset
    comb_dir.mkdir(parents=True, exist_ok=True)
    # Persist the fitted model + feature (cut) order so the web app can load it.
    model_path = comb_dir / "combiner_model.joblib"
    joblib.dump({"model": best_model, "cuts": used, "classes": [int(c) for c in best_model.classes_]}, model_path)
    pd.DataFrame(rows).to_csv(comb_dir / "model_selection.csv", index=False, encoding="utf-8-sig")
    save_confusion_matrix(
        y_test - 1, y_pred - 1, STAGE_NAMES,
        comb_dir / "confusion_matrix_test.png",
        title=f"combiner[{featureset}] {best_name}",
    )
    save_classification_report(
        y_test - 1, y_pred - 1, STAGE_NAMES, comb_dir / "classification_report_test.txt"
    )
    result = {
        "featureset": featureset,
        "cuts_used": used,
        "n_features": len(used),
        "best_model": best_name,
        "model_path": str(model_path.resolve()),
        "oof_macro_f1": rows[0]["oof_macro_f1"],
        "test_macro_f1": m["macro_f1"],
        "test_accuracy": m["accuracy"],
        "test_macro_f1_ci_low": ci["macro_f1_ci_low"],
        "test_macro_f1_ci_high": ci["macro_f1_ci_high"],
        "all_models": rows,
    }
    dump_json(result, comb_dir / "best.json")
    print(
        f"[combiner] {featureset}: best={best_name} "
        f"oof={result['oof_macro_f1']:.4f}  test_macro_f1={result['test_macro_f1']:.4f} "
        f"(95% CI {ci['macro_f1_ci_low']:.3f}-{ci['macro_f1_ci_high']:.3f})",
        flush=True,
    )
    return result


def compare_with_hierarchy(
    out_root: Path,
    data_root: Path,
    splits_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
    seed: int = SEED,
    n_boot: int = 2000,
    embed: bool = False,
    pca_n: int = 50,
) -> pd.DataFrame:
    """Tabulate combiner variants against the hierarchy hard routing.

    Scalar feature sets: topology3, all, ordinal3.
    With embed=True, also runs embed_ordinal3 (raw + PCA) — the high-dim
    penultimate-embedding variant (idea #4 v3)."""
    best_path = out_root / "hierarchy" / "best_topology.json"
    topo3 = None
    hier_row = None
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        topo3 = list(best["cuts_used"])
        hier_row = {
            "method": f"hierarchy[{best['name']}]",
            "n_features": 3,
            "best_model": "hard_routing",
            "oof_macro_f1": float(best.get("oof_macro_f1", float("nan"))),
            "test_macro_f1": float(best.get("test_macro_f1", float("nan"))),
            "test_accuracy": float(best.get("test_accuracy", float("nan"))),
        }

    results = []
    if topo3:
        results.append(run_combiner(
            "topology3", topo3, out_root, data_root, splits_dir, device, batch_size, seed, n_boot
        ))
    results.append(run_combiner(
        "all", sorted(CUTS.keys()), out_root, data_root, splits_dir, device, batch_size, seed, n_boot
    ))
    results.append(run_combiner(
        "ordinal3", ORDINAL_CUTS, out_root, data_root, splits_dir, device, batch_size, seed, n_boot
    ))
    if embed:
        results.append(run_embedding_combiner(
            "embed_ordinal3_raw", ORDINAL_CUTS, out_root, data_root, splits_dir,
            device, batch_size, seed, n_boot, pca_n=0,
        ))
        results.append(run_embedding_combiner(
            f"embed_ordinal3_pca{pca_n}", ORDINAL_CUTS, out_root, data_root, splits_dir,
            device, batch_size, seed, n_boot, pca_n=pca_n,
        ))

    table = []
    if hier_row is not None:
        table.append(hier_row)
    for r in results:
        table.append({
            "method": f"combiner[{r['featureset']}]",
            "n_features": r["n_features"],
            "best_model": r["best_model"],
            "oof_macro_f1": r["oof_macro_f1"],
            "test_macro_f1": r["test_macro_f1"],
            "test_accuracy": r["test_accuracy"],
        })
    df = pd.DataFrame(table)
    comb_root = out_root / "combiner"
    comb_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(comb_root / "comparison.csv", index=False, encoding="utf-8-sig")
    print("\n[combiner] === comparison (selection by OOF; test = report only) ===", flush=True)
    print(df.to_string(index=False), flush=True)
    return df
