"""OOF-based fold ensemble for one cut — *best backbone per fold* edition.

Training (`renai.cv`) still produces 5 backbones x 5 folds = 25 ckpts per cut.
At ensemble time we pick **the single best backbone per fold** (by that fold's
validation macro-F1) — yielding **5 models per cut**, one per fold.  These 5
models form the ensemble; the other 20 ckpts stay on disk but are unused.

Two strategies are compared *purely on out-of-fold predictions* over the
train+val pool (the outer 20% test set is never used to pick the winner):

1. fold_voting
     OOF: for every sample s in V_k, just use the fold-k winner's softmax
          (only that model is guaranteed not to have seen s).
     Test: mean softmax of the 5 fold-winner models on the test sample.

2. stacking
     OOF features X_oof[s] = the fold-k winner's softmax on s. Shape (N, 2).
     Meta classifier: LogisticRegression(class_weight="balanced").
     OOF score: cross_val_predict(meta, X_oof, y_oof, cv=5) — unbiased.
     Test features: mean softmax of the 5 fold winners on the test sample,
       shape (N_test, 2).  Meta is retrained on all of X_oof and applied.
     Note: with only 2 input dims (and p[:,0]+p[:,1]=1), stacking essentially
     learns a calibrated threshold; it rarely beats voting here, but we keep
     it so the comparison is on record.

Decision: chosen = argmax over {fold_voting, stacking} of OOF macro-F1.
test_metrics are reported only — they never influence the decision."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import (
    Cut,
    filter_indices_for_cut,
    make_cv_folds,
    make_eval_loader_for_cut,
    make_outer_split,
)
from .eval import (
    binary_metrics,
    dump_json,
    save_classification_report,
    save_confusion_matrix,
)
from .models import DEFAULT_BACKBONES, create_model
from .seed import SEED, set_seed


@dataclass
class EnsembleDecision:
    cut: str
    chosen: str                            # "fold_voting" | "stacking"
    members: list[dict]                    # 5 picked: [{backbone, fold, ckpt, val_macro_f1}]
    oof_macro_f1_voting: float
    oof_macro_f1_stacking: float
    test_macro_f1: float
    test_accuracy: float
    test_auc: float
    artifacts_dir: str


def discover_fold_ckpts(
    cut_root: Path,
    backbones: Sequence[str] = DEFAULT_BACKBONES,
) -> list[dict]:
    """Return every (backbone, fold, ckpt) tuple that has been trained so far."""
    cv_root = cut_root / "cv"
    members: list[dict] = []
    if not cv_root.exists():
        return members
    for fold_dir in sorted(cv_root.glob("fold_*")):
        try:
            fi = int(fold_dir.name.split("_")[1])
        except ValueError:
            continue
        for bb in backbones:
            ckpt = fold_dir / bb / f"best_{bb}.pth"
            if ckpt.exists():
                members.append({"backbone": bb, "fold": fi, "ckpt": str(ckpt)})
    return members


def _pick_best_backbone_per_fold(
    cut_root: Path,
    all_members: list[dict],
) -> list[dict]:
    """For each fold, pick the backbone with highest val_macro_f1.

    Falls back to the cv summary if a val_metrics.json is missing.  Returns
    [{backbone, fold, ckpt, val_macro_f1}] sorted by fold."""
    by_fold: dict[int, list[dict]] = {}
    for m in all_members:
        by_fold.setdefault(int(m["fold"]), []).append(m)

    winners: list[dict] = []
    for fi in sorted(by_fold):
        scored: list[tuple[float, dict]] = []
        for entry in by_fold[fi]:
            bb = entry["backbone"]
            metrics_path = cut_root / "cv" / f"fold_{fi}" / bb / "val_metrics.json"
            score = float("-inf")
            if metrics_path.exists():
                try:
                    j = json.loads(metrics_path.read_text(encoding="utf-8"))
                    score = float(j.get("macro_f1", float("-inf")))
                except Exception:
                    pass
            scored.append((score, entry))
        # max by score; if all -inf the first ckpt wins (deterministic given sort).
        best_score, best_entry = max(scored, key=lambda t: t[0])
        winners.append({
            "backbone": best_entry["backbone"],
            "fold": int(best_entry["fold"]),
            "ckpt": best_entry["ckpt"],
            "val_macro_f1": float(best_score) if best_score != float("-inf") else float("nan"),
        })
    return winners


@torch.no_grad()
def _softmax_on_loader(model: torch.nn.Module, loader, device: str) -> np.ndarray:
    out = []
    for imgs, _labels in loader:
        p = F.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
        out.append(p)
    return np.concatenate(out) if out else np.zeros((0, 2))


@torch.no_grad()
def _collect_labels(loader) -> np.ndarray:
    ys = []
    for _imgs, labels in loader:
        ys.append(np.asarray(labels))
    return np.concatenate(ys) if ys else np.array([], dtype=np.int64)


def _load_cached_val_probs(fold_dir: Path, bb: str) -> np.ndarray | None:
    p = fold_dir / bb / "val_probs.npy"
    if not p.exists():
        return None
    try:
        arr = np.load(p)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
    except Exception:
        pass
    return None


def build_ensemble_for_cut(
    cut: Cut,
    data_root: Path,
    out_root: Path,
    splits_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
    backbones: Sequence[str] = DEFAULT_BACKBONES,
) -> EnsembleDecision:
    """Pick the best backbone per fold, then OOF-decide voting vs stacking."""
    set_seed(SEED)
    cut_root = out_root / "cuts" / cut.name
    ens_dir = cut_root / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)

    all_members = discover_fold_ckpts(cut_root, backbones=backbones)
    if not all_members:
        raise FileNotFoundError(
            f"No per-fold ckpts under {cut_root / 'cv'} — run train_cv first."
        )

    members = _pick_best_backbone_per_fold(cut_root, all_members)
    print(f"  [ensemble] cut={cut.name}: {len(members)} fold winners", flush=True)
    for m in members:
        print(
            f"    fold {m['fold']}: best={m['backbone']}  "
            f"val_macro_f1={m['val_macro_f1']:.4f}",
            flush=True,
        )

    outer = make_outer_split(data_root, splits_dir / "outer_split.json")
    cut_train_val_idx = filter_indices_for_cut(data_root, outer["train_val_idx"], cut)
    cut_test_idx      = filter_indices_for_cut(data_root, outer["test_idx"],      cut)
    folds = make_cv_folds(data_root, cut_train_val_idx)
    fold_to_va: dict[int, list[int]] = {fi: list(va) for fi, (_tr, va) in enumerate(folds)}
    fold_to_winner: dict[int, dict] = {int(m["fold"]): m for m in members}

    # --- OOF features (just one fold-winner softmax per sample) -------------
    print(f"  [ensemble] extracting OOF features (5 winners x their val folds)", flush=True)
    X_oof_blocks: list[np.ndarray] = []
    y_oof_blocks: list[np.ndarray] = []
    for fi in sorted(fold_to_winner):
        va_idx = fold_to_va.get(fi)
        if va_idx is None:
            print(f"    [warn] fold {fi}: no va split (mismatched cv folds), skipped", flush=True)
            continue
        va_loader = make_eval_loader_for_cut(data_root, cut, va_idx, batch_size=batch_size)
        y_oof_blocks.append(_collect_labels(va_loader))

        winner = fold_to_winner[fi]
        bb = winner["backbone"]
        cached = _load_cached_val_probs(cut_root / "cv" / f"fold_{fi}", bb)
        if cached is not None and cached.shape[0] == len(va_idx):
            probs = cached
            tag = "(cached)"
        else:
            m = create_model(bb, num_classes=2).to(device)
            m.load_state_dict(torch.load(winner["ckpt"], map_location=device))
            m.eval()
            probs = _softmax_on_loader(m, va_loader, device)
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tag = "(inferred)"
        X_oof_blocks.append(probs)
        print(f"    .. fold {fi} bb={bb} probs={probs.shape} {tag}", flush=True)

    X_oof = np.concatenate(X_oof_blocks, axis=0) if X_oof_blocks else np.zeros((0, 2))
    y_oof = np.concatenate(y_oof_blocks) if y_oof_blocks else np.array([], dtype=np.int64)
    print(f"  [ensemble] OOF matrix X_oof={X_oof.shape}  y_oof={y_oof.shape}", flush=True)

    # --- OOF voting (single-model softmax per sample; argmax is voting too) ---
    voting_oof_probs = X_oof
    voting_oof_pred = voting_oof_probs.argmax(axis=1)
    voting_oof_metrics = binary_metrics(y_oof, voting_oof_pred, voting_oof_probs)

    # --- OOF stacking (cross_val_predict over the meta) ---------------------
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    meta = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    stacking_oof_probs = cross_val_predict(meta, X_oof, y_oof, cv=cv5, method="predict_proba")
    stacking_oof_pred = stacking_oof_probs.argmax(axis=1)
    stacking_oof_metrics = binary_metrics(y_oof, stacking_oof_pred, stacking_oof_probs)

    print(
        f"  [ensemble] OOF macro_f1  voting={voting_oof_metrics['macro_f1']:.4f}  "
        f"stacking={stacking_oof_metrics['macro_f1']:.4f}",
        flush=True,
    )

    chosen = (
        "stacking"
        if stacking_oof_metrics["macro_f1"] > voting_oof_metrics["macro_f1"]
        else "fold_voting"
    )
    print(f"  [ensemble] chosen={chosen} (by OOF macro_f1)", flush=True)

    # --- Test inference -----------------------------------------------------
    test_loader = make_eval_loader_for_cut(
        data_root, cut, cut_test_idx, batch_size=batch_size,
    )
    y_test = _collect_labels(test_loader)

    per_fold_test: list[np.ndarray] = []
    for member in members:
        bb = member["backbone"]
        m = create_model(bb, num_classes=2).to(device)
        m.load_state_dict(torch.load(member["ckpt"], map_location=device))
        m.eval()
        per_fold_test.append(_softmax_on_loader(m, test_loader, device).astype(np.float64))
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_test_softmax = (
        np.mean(np.stack(per_fold_test, axis=0), axis=0)
        if per_fold_test else np.zeros((len(y_test), 2))
    )

    meta_path: str | None = None
    if chosen == "fold_voting":
        test_probs_out = avg_test_softmax.astype(np.float32)
    else:  # stacking
        meta_final = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        )
        meta_final.fit(X_oof, y_oof)
        meta_path = str(ens_dir / "meta_logreg.pkl")
        joblib.dump(meta_final, meta_path)
        test_probs_out = meta_final.predict_proba(avg_test_softmax).astype(np.float32)
        np.save(ens_dir / "X_test_features.npy", avg_test_softmax)

    y_pred = test_probs_out.argmax(axis=1) if len(test_probs_out) else np.array([], dtype=np.int64)
    test_metrics = binary_metrics(y_test, y_pred, test_probs_out)

    np.save(ens_dir / "test_probs.npy", test_probs_out)
    np.save(ens_dir / "test_y_true.npy", y_test)
    np.save(ens_dir / "X_oof.npy", X_oof)
    np.save(ens_dir / "y_oof.npy", y_oof)
    save_confusion_matrix(
        y_test, y_pred, list(cut.class_names),
        ens_dir / "confusion_matrix_test.png",
        title=f"{cut.name} | {chosen} (5 fold winners)",
    )
    save_classification_report(
        y_test, y_pred, list(cut.class_names),
        ens_dir / "classification_report_test.txt",
    )

    decision = EnsembleDecision(
        cut=cut.name,
        chosen=chosen,
        members=members,
        oof_macro_f1_voting=float(voting_oof_metrics["macro_f1"]),
        oof_macro_f1_stacking=float(stacking_oof_metrics["macro_f1"]),
        test_macro_f1=float(test_metrics["macro_f1"]),
        test_accuracy=float(test_metrics["accuracy"]),
        test_auc=float(test_metrics.get("auc", float("nan"))),
        artifacts_dir=str(ens_dir),
    )
    dump_json(
        {
            "decision": asdict(decision),
            "oof_metrics": {
                "fold_voting": voting_oof_metrics,
                "stacking": stacking_oof_metrics,
            },
            "test_metrics": test_metrics,
            "n_members": len(members),
            "meta_path": meta_path,
            "all_trained_ckpts": all_members,   # for traceability
        },
        ens_dir / "winner.json",
    )
    return decision


def summarize_all_cuts(out_root: Path) -> pd.DataFrame:
    rows = []
    for winner_path in (out_root / "cuts").glob("*/ensemble/winner.json"):
        info = json.loads(winner_path.read_text(encoding="utf-8"))
        d = info["decision"]
        rows.append({
            "cut": d["cut"],
            "chosen": d["chosen"],
            "oof_voting": d["oof_macro_f1_voting"],
            "oof_stacking": d["oof_macro_f1_stacking"],
            "n_members": info.get("n_members", len(d.get("members", []))),
            "test_macro_f1": d["test_macro_f1"],
            "test_accuracy": d["test_accuracy"],
            "test_auc": d["test_auc"],
            "fold_winners": ", ".join(
                f"f{m['fold']}={m['backbone']}" for m in d["members"]
            ),
        })
    return pd.DataFrame(rows).sort_values("cut") if rows else pd.DataFrame()
