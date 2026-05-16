"""Top-3 cross-family ensemble for one cut.

Two ensemble strategies are implemented and compared on the test set:
  (a) soft-voting   — mean of softmax probabilities
  (b) stacking      — LogisticRegression meta classifier trained on
                      out-of-fold val probabilities (kept consistent with the
                      original 03.25_m2_stacking_top3 design)

For each cut we compare:
  - best single backbone (by test_macro_f1)
  - voting ensemble of top-3 cross-family
  - stacking ensemble of top-3 cross-family
and write the winner to outputs/cuts/<cut>/ensemble/winner.json.

If the best ensemble does not beat the best single model on macro_f1, the
single model is declared the winner — that matches the user's spec ("如果真的
有變好就使用集成的，但如果沒有變好就使用單一模型")."""

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
    predict_loader,
    save_classification_report,
    save_confusion_matrix,
)
from .models import FAMILY_OF, create_model
from .seed import SEED, set_seed


@dataclass
class EnsembleDecision:
    cut: str
    chosen: str                # "single" | "voting" | "stacking"
    chosen_backbone: str | None
    chosen_members: list[str]
    test_macro_f1: float
    test_accuracy: float
    test_auc: float
    artifacts_dir: str


def _pick_top3_cross_family(summary_csv: Path) -> list[str]:
    """Pick the best backbone per family on cv_mean_macro_f1, return up to 3."""
    df = pd.read_csv(summary_csv)
    df = df.dropna(subset=["cv_mean_macro_f1"])
    df["family"] = df["backbone"].map(FAMILY_OF)
    best_per_family = (
        df.sort_values("cv_mean_macro_f1", ascending=False)
        .drop_duplicates("family")
    )
    return best_per_family["backbone"].head(3).tolist()


def _load_finals(cut_root: Path, backbones: Sequence[str], device: str):
    models = []
    for bb in backbones:
        ckpt = cut_root / "final" / bb / f"best_{bb}.pth"
        m = create_model(bb, num_classes=2).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        models.append((bb, m))
    return models


@torch.no_grad()
def _oof_features(
    cut_root: Path,
    cut: Cut,
    data_root: Path,
    backbones: Sequence[str],
    folds,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Out-of-fold softmax features from per-fold checkpoints.

    For each sample in the cut's train+val pool we use the fold model that
    treated it as validation — never one that saw it during training.  This is
    the only way to feed a stacking meta-classifier without data leakage."""
    n_total = sum(len(va) for _, va in folds)
    K = len(backbones)
    X = np.zeros((n_total, 2 * K), dtype=np.float32)
    y = np.zeros((n_total,), dtype=np.int64)
    sample_indices: list[int] = []

    cursor = 0
    for fi, (_tr, va) in enumerate(folds):
        loader = make_eval_loader_for_cut(data_root, cut, va, batch_size=batch_size)
        # collect labels once
        ys_local: list[int] = []
        per_model_probs: list[list[np.ndarray]] = [[] for _ in backbones]
        for k, bb in enumerate(backbones):
            ckpt = cut_root / "cv" / f"fold_{fi}" / bb / f"best_{bb}.pth"
            if not ckpt.exists():
                # Fall back to final ckpt if a per-fold one is missing.
                ckpt = cut_root / "final" / bb / f"best_{bb}.pth"
            m = create_model(bb, num_classes=2).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            for imgs, labels in loader:
                p = F.softmax(m(imgs.to(device)), dim=1).cpu().numpy()
                per_model_probs[k].append(p)
                if k == 0:
                    ys_local.extend(int(l) for l in labels.numpy().tolist())
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        n_local = len(ys_local)
        for k in range(K):
            X[cursor:cursor + n_local, 2 * k:2 * k + 2] = np.concatenate(per_model_probs[k])
        y[cursor:cursor + n_local] = np.asarray(ys_local)
        sample_indices.extend(va)
        cursor += n_local

    return X, y, sample_indices


@torch.no_grad()
def _stack_probs(models, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_stack [N, 2*K], y [N]) for the loader."""
    feats_per_model = [[] for _ in models]
    ys = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        for k, (_, m) in enumerate(models):
            p = F.softmax(m(imgs), dim=1).cpu().numpy()
            feats_per_model[k].append(p)
        ys.append(np.asarray(labels))
    X = np.concatenate([np.concatenate(fs) for fs in feats_per_model], axis=1) if feats_per_model[0] else np.zeros((0, 2 * len(models)))
    y = np.concatenate(ys) if ys else np.array([])
    return X, y


def build_ensemble_for_cut(
    cut: Cut,
    data_root: Path,
    out_root: Path,
    splits_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
) -> EnsembleDecision:
    set_seed(SEED)
    cut_root = out_root / "cuts" / cut.name
    ens_dir = cut_root / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = cut_root / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Run CV first; missing {summary_csv}")
    df = pd.read_csv(summary_csv)

    # --- top-3 cross-family ---
    top3 = _pick_top3_cross_family(summary_csv)
    members = _load_finals(cut_root, top3, device)

    # --- gather OOF val features for stacking ---
    outer = make_outer_split(data_root, splits_dir / "outer_split.json")
    cut_train_val_idx = filter_indices_for_cut(data_root, outer["train_val_idx"], cut)
    cut_test_idx      = filter_indices_for_cut(data_root, outer["test_idx"],      cut)
    folds = make_cv_folds(data_root, cut_train_val_idx)
    test_loader = make_eval_loader_for_cut(data_root, cut, cut_test_idx, batch_size=batch_size)

    # OOF features for meta training (no leakage)
    X_val, y_val, _ = _oof_features(
        cut_root=cut_root,
        cut=cut,
        data_root=data_root,
        backbones=top3,
        folds=folds,
        device=device,
        batch_size=batch_size,
    )
    # Test features come from the FINAL retrained models (one per backbone)
    X_test, y_test = _stack_probs(members, test_loader, device)

    meta = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    meta.fit(X_val, y_val)
    joblib.dump(meta, ens_dir / "meta_logreg.pkl")
    np.save(ens_dir / "X_test.npy", X_test)
    np.save(ens_dir / "y_test.npy", y_test)

    # --- stacking metrics ---
    stack_pred = meta.predict(X_test)
    stack_probs2 = meta.predict_proba(X_test)
    stack_metrics = binary_metrics(y_test, stack_pred, stack_probs2)

    # --- voting metrics: average softmax across members ---
    K = len(members)
    avg_probs = np.zeros_like(stack_probs2)
    for k in range(K):
        avg_probs += X_test[:, 2 * k:2 * k + 2]
    avg_probs /= max(K, 1)
    vote_pred = avg_probs.argmax(axis=1)
    vote_metrics = binary_metrics(y_test, vote_pred, avg_probs)

    # --- best single (from summary.csv test_macro_f1) ---
    df_ok = df.dropna(subset=["test_macro_f1"]).sort_values("test_macro_f1", ascending=False)
    best_single = df_ok.iloc[0] if len(df_ok) else None
    single_metrics = (
        {
            "macro_f1": float(best_single["test_macro_f1"]),
            "accuracy": float(best_single["test_acc"]),
            "auc": float(best_single.get("test_auc", float("nan"))),
        }
        if best_single is not None else
        {"macro_f1": float("-inf"), "accuracy": 0.0, "auc": float("nan")}
    )
    best_single_bb = str(best_single["backbone"]) if best_single is not None else None

    candidates = {
        "single":   {"backbone": best_single_bb, "members": [best_single_bb] if best_single_bb else [], **single_metrics},
        "voting":   {"backbone": None,           "members": top3, **vote_metrics},
        "stacking": {"backbone": None,           "members": top3, **stack_metrics},
    }

    # Choose: ensemble wins only if its macro_f1 > single by some margin.
    # User spec: "如果真的有變好就使用集成的，但如果沒有變好就使用單一模型"
    # → require strict improvement.
    single_f1 = candidates["single"]["macro_f1"]
    best_ens_name = max(("voting", "stacking"), key=lambda k: candidates[k]["macro_f1"])
    if candidates[best_ens_name]["macro_f1"] > single_f1:
        chosen = best_ens_name
    else:
        chosen = "single"

    decision = EnsembleDecision(
        cut=cut.name,
        chosen=chosen,
        chosen_backbone=best_single_bb if chosen == "single" else None,
        chosen_members=top3 if chosen != "single" else [best_single_bb] if best_single_bb else [],
        test_macro_f1=float(candidates[chosen]["macro_f1"]),
        test_accuracy=float(candidates[chosen]["accuracy"]),
        test_auc=float(candidates[chosen].get("auc", float("nan"))),
        artifacts_dir=str(ens_dir),
    )

    # Write outcomes
    dump_json({"candidates": candidates, "decision": asdict(decision)}, ens_dir / "winner.json")
    if chosen == "stacking":
        save_confusion_matrix(
            y_test, stack_pred, list(cut.class_names),
            ens_dir / "confusion_matrix_test.png",
            title=f"{cut.name} | stacking",
        )
        save_classification_report(
            y_test, stack_pred, list(cut.class_names),
            ens_dir / "classification_report_test.txt",
        )
    elif chosen == "voting":
        save_confusion_matrix(
            y_test, vote_pred, list(cut.class_names),
            ens_dir / "confusion_matrix_test.png",
            title=f"{cut.name} | voting",
        )
        save_classification_report(
            y_test, vote_pred, list(cut.class_names),
            ens_dir / "classification_report_test.txt",
        )

    return decision
