"""Evaluation helpers — metrics, confusion matrix, classification report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model on loader.

    Returns (y_true, y_pred, probs) where probs is shape (N, num_classes)."""
    model.eval()
    ys, preds, probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        p = F.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(axis=1))
        ys.append(np.asarray(labels))
    if not ys:
        return np.array([]), np.array([]), np.zeros((0, 2))
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None) -> dict:
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if probs is not None and probs.shape[1] == 2 and len(np.unique(y_true)) == 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        except ValueError:
            out["auc"] = float("nan")
    return out


def bootstrap_classification_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap (percentile) confidence intervals for accuracy and macro-F1.

    With a tiny test set (here n≈58) a single point estimate is misleading;
    resampling the test set with replacement gives an honest interval that a
    thesis reviewer expects to see reported."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    point_acc = float(accuracy_score(y_true, y_pred)) if n else float("nan")
    point_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if n else float("nan")

    rng = np.random.default_rng(seed)
    accs = np.empty(n_boot, dtype=np.float64)
    f1s = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        accs[b] = accuracy_score(yt, yp)
        f1s[b] = f1_score(yt, yp, average="macro", zero_division=0)

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "n": int(n),
        "n_boot": int(n_boot),
        "accuracy": point_acc,
        "accuracy_ci_low": float(np.percentile(accs, lo_q)),
        "accuracy_ci_high": float(np.percentile(accs, hi_q)),
        "macro_f1": point_f1,
        "macro_f1_ci_low": float(np.percentile(f1s, lo_q)),
        "macro_f1_ci_high": float(np.percentile(f1s, hi_q)),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
    title: str = "",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(max(4, len(class_names)), max(4, len(class_names))))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    thr = cm.max() / 2 if cm.size else 0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )
    out_path.write_text(report, encoding="utf-8")
    return out_path


def dump_json(payload: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
