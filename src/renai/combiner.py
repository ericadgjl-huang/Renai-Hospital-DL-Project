"""Learned 4-class combiner over per-cut probabilities (idea #4, version 1).

Instead of routing a sample through a fixed hierarchy of hard 0.5 thresholds
(where one early wrong turn is unrecoverable), we treat each binary cut's
P(class=1) as a feature and let a single 4-class classifier learn the mapping
to stage 1..4. This replaces brittle routing with a soft, learned combiner.

Feature sets
------------
* "topology3" : the 3 cuts of the OOF-selected topology (matches the original
                idea: three cut-probabilities per sample).
* "all"       : every available cut (10-dim) — more signal, still low-dim.

Honesty (no leakage)
--------------------
* Train features  = OOF per-cut probabilities (each sample scored by a cut
  model that never trained on it) — reuses `hierarchy.compute_oof_cut_p1`.
* Model selection = 5-fold CV macro-F1 on those OOF features.
* Test features   = the cuts' soft-vote probabilities on the held-out test set.
The outer test set is used only for the final report, never for selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

from .cuts_registry import CUTS
from .data import get_4class_labels, make_4class_eval_loader, make_outer_split
from .eval import (
    bootstrap_classification_ci,
    binary_metrics,
    dump_json,
    save_classification_report,
    save_confusion_matrix,
)
from .hierarchy import CutPredictor, compute_oof_cut_p1
from .seed import SEED, set_seed

STAGE_NAMES = ["stage_1", "stage_2", "stage_3", "stage_4"]


def candidate_models(seed: int = SEED) -> dict:
    """The user's wish list: boosting, tree, and SVM with two kernels."""
    return {
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
    best_model.fit(X_oof, y_tv)
    y_pred = best_model.predict(X_test)

    m = binary_metrics(y_test, y_pred)
    m["macro_f1"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    ci = bootstrap_classification_ci(y_test, y_pred, n_boot=n_boot, seed=seed)

    comb_dir = out_root / "combiner" / featureset
    comb_dir.mkdir(parents=True, exist_ok=True)
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
) -> pd.DataFrame:
    """Run the combiner for 'topology3' and 'all', tabulated against hierarchy."""
    # topology3 cuts come from the OOF-selected topology, if present.
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
    all_cuts = sorted(CUTS.keys())
    results.append(run_combiner(
        "all", all_cuts, out_root, data_root, splits_dir, device, batch_size, seed, n_boot
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
