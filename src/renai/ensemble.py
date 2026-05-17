"""Fold-level soft-voting ensemble for one cut.

For each cut we take the 25 validation-best checkpoints (5 backbones x 5 folds)
saved by `renai.cv` and average their softmax probabilities on the outer 20%
test set.  This is a textbook cross-validation / bagging ensemble — every
checkpoint was the best one on its own fold's validation, so no per-fold model
ever sees the outer test split during training.

No more "single vs ensemble" decision: the soft-vote of all available fold
models is always the winner.  No stacking either — fold-level stacking would
require a held-out set that we don't have."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import (
    Cut,
    filter_indices_for_cut,
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
    chosen: str                            # always "fold_voting" in the new pipeline
    members: list[dict]                    # [{"backbone": ..., "fold": ..., "ckpt": ...}]
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


@torch.no_grad()
def _avg_softmax_on_loader(members: list[dict], loader, device: str) -> np.ndarray:
    """Mean softmax over all members for every sample yielded by `loader`."""
    sums: np.ndarray | None = None
    n_seen = 0
    for entry in members:
        bb = entry["backbone"]
        ckpt = entry["ckpt"]
        m = create_model(bb, num_classes=2).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        per_model = []
        for imgs, _labels in loader:
            p = F.softmax(m(imgs.to(device)), dim=1).cpu().numpy()
            per_model.append(p)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        per_model_np = np.concatenate(per_model) if per_model else np.zeros((0, 2))
        if sums is None:
            sums = np.zeros_like(per_model_np)
            n_seen = 0
        sums += per_model_np
        n_seen += 1
    if sums is None or n_seen == 0:
        return np.zeros((0, 2))
    return sums / float(n_seen)


@torch.no_grad()
def _collect_labels(loader) -> np.ndarray:
    ys = []
    for _imgs, labels in loader:
        ys.append(np.asarray(labels))
    return np.concatenate(ys) if ys else np.array([])


def build_ensemble_for_cut(
    cut: Cut,
    data_root: Path,
    out_root: Path,
    splits_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
    backbones: Sequence[str] = DEFAULT_BACKBONES,
) -> EnsembleDecision:
    """Soft-vote every available per-fold best ckpt for this cut and evaluate
    the average on the 20% outer test set."""
    set_seed(SEED)
    cut_root = out_root / "cuts" / cut.name
    ens_dir = cut_root / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)

    members = discover_fold_ckpts(cut_root, backbones=backbones)
    if not members:
        raise FileNotFoundError(
            f"No per-fold ckpts under {cut_root / 'cv'} — run train_cv first."
        )

    outer = make_outer_split(data_root, splits_dir / "outer_split.json")
    cut_test_idx = filter_indices_for_cut(data_root, outer["test_idx"], cut)
    test_loader = make_eval_loader_for_cut(
        data_root, cut, cut_test_idx, batch_size=batch_size,
    )

    y_test = _collect_labels(test_loader)
    avg_probs = _avg_softmax_on_loader(members, test_loader, device)
    y_pred = avg_probs.argmax(axis=1) if len(avg_probs) else np.array([], dtype=np.int64)
    test_metrics = binary_metrics(y_test, y_pred, avg_probs)

    np.save(ens_dir / "test_probs.npy", avg_probs)
    np.save(ens_dir / "test_y_true.npy", y_test)
    save_confusion_matrix(
        y_test, y_pred, list(cut.class_names),
        ens_dir / "confusion_matrix_test.png",
        title=f"{cut.name} | fold soft-vote ({len(members)} models)",
    )
    save_classification_report(
        y_test, y_pred, list(cut.class_names),
        ens_dir / "classification_report_test.txt",
    )

    decision = EnsembleDecision(
        cut=cut.name,
        chosen="fold_voting",
        members=members,
        test_macro_f1=float(test_metrics["macro_f1"]),
        test_accuracy=float(test_metrics["accuracy"]),
        test_auc=float(test_metrics.get("auc", float("nan"))),
        artifacts_dir=str(ens_dir),
    )
    dump_json(
        {
            "decision": asdict(decision),
            "test_metrics": test_metrics,
            "n_members": len(members),
        },
        ens_dir / "winner.json",
    )
    return decision


def summarize_all_cuts(out_root: Path) -> pd.DataFrame:
    """Roll every cut's ensemble winner.json into a single table."""
    rows = []
    for winner_path in (out_root / "cuts").glob("*/ensemble/winner.json"):
        import json
        info = json.loads(winner_path.read_text(encoding="utf-8"))
        d = info["decision"]
        rows.append({
            "cut": d["cut"],
            "n_members": info.get("n_members", len(d.get("members", []))),
            "test_macro_f1": d["test_macro_f1"],
            "test_accuracy": d["test_accuracy"],
            "test_auc": d["test_auc"],
        })
    return pd.DataFrame(rows).sort_values("cut") if rows else pd.DataFrame()
