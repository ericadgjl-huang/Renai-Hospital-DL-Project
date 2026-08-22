"""Stage 19 — Paper-ready figures for the final leak-free CORN ensemble.

Produces (into <out-root>/paper_figures/):
  fig1_confusion_matrix.png  — counts + row-normalised confusion matrix (test)
  fig2_confusion_matrix_oof.png — same on the OOF (n=363, the stable estimate)
  fig3_per_stage_f1.png      — per-stage precision / recall / F1 bars (OOF)
  fig4_method_progression.png — QWK across the pipeline milestones vs human kappa

All use the SAME final ensemble as scripts/17 (equal weight is fine for figures;
selection was done on OOF elsewhere).

Usage:
    python scripts/19_paper_figures.py --out-root outputs_merged_patient_full \
        --data-root stage_cls_merged --groups-csv outputs_merged/patient_groups.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torchvision.datasets import ImageFolder

from renai.data import get_4class_labels, make_outer_split, patient_groups
from renai.ordinal import OrdinalLogisticMeta, decode_rank_count, ordinal_metrics

STAGES = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
BLUE = "#3b6fb0"


def load_stacks(out_root, kind):
    P = []
    for s in glob.glob(f"{out_root}/corn*"):
        if os.path.isdir(s):
            for bb in sorted(os.listdir(s)):
                f = f"{s}/{bb}/{kind}_cumulative.npy"
                if os.path.exists(f):
                    P.append(np.load(f))
    return P


def load_ensemble(out_root, kind):
    return np.mean(load_stacks(out_root, kind), axis=0)


def stacking_preds(out_root, y_oof):
    """Best stacking meta (ordinal logistic C=3) predictions on OOF + test."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    X_oof = np.nan_to_num(np.hstack(load_stacks(out_root, "oof")))
    X_test = np.nan_to_num(np.hstack(load_stacks(out_root, "test")))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof_pred = cross_val_predict(OrdinalLogisticMeta(C=3.0), X_oof, y_oof, cv=cv)
    meta = OrdinalLogisticMeta(C=3.0).fit(X_oof, y_oof)
    return oof_pred, meta.predict(X_test)


def cm_figure(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(STAGES, rotation=20, ha="right"); ax.set_yticklabels(STAGES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]*100:.0f}%)", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "#222", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


def per_stage_figure(y_true, y_pred, title, path):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1, 2, 3, 4],
                                                 zero_division=0)
    x = np.arange(4); w = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x - w, p, w, label="Precision", color="#9ec3e6")
    ax.bar(x,     r, w, label="Recall",    color=BLUE)
    ax.bar(x + w, f, w, label="F1",        color="#1f3f66")
    ax.set_xticks(x); ax.set_xticklabels(STAGES)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title(title, fontsize=11)
    for xi, val in zip(x, r):
        ax.text(xi, val + 0.02, f"{val:.2f}", ha="center", fontsize=8)
    ax.legend(loc="lower left", fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


def progression_figure(path):
    # QWK across pipeline milestones (see docs). Grouped by evaluation regime.
    labels = ["RandomForest\ncombiner", "Hard\nhierarchy", "Ordinal\ncombiner",
              "Class-wt\nCORN", "Merged 5-bb\n(leaky)", "Leak-free 9-bb\nvoting (OOF)",
              "Leak-free 9-bb\nstacking (OOF)"]
    qwk = [0.589, 0.625, 0.669, 0.700, 0.833, 0.785, 0.798]
    colors = ["#c0504d", "#c0504d", "#4f81bd", "#4f81bd", "#9ec3e6", "#4f81bd", "#1f3f66"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.bar(range(len(labels)), qwk, color=colors)
    ax.axhspan(0.39, 0.46, color="#f2c14e", alpha=0.35)
    ax.text(len(labels) - 0.5, 0.425, "human inter-observer κ (0.39–0.46)",
            ha="right", va="center", fontsize=8, color="#7a5c00")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Quadratic weighted kappa"); ax.set_ylim(0, 0.95)
    ax.set_title("QWK across pipeline milestones (see §0 of the methods doc)", fontsize=11)
    for i, v in enumerate(qwk):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("outputs_merged_patient_full"))
    ap.add_argument("--data-root", type=Path, default=Path("stage_cls_merged"))
    ap.add_argument("--groups-csv", default="outputs_merged/patient_groups.csv")
    ap.add_argument("--splits-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.splits_dir is None:
        args.splits_dir = args.out_root / "splits"
    fig_dir = args.out_root / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    groups = patient_groups(args.data_root, args.groups_csv) if args.groups_csv else None
    outer = make_outer_split(args.data_root, args.splits_dir / "outer_split.json", groups=groups)
    tv = sorted(int(i) for i in outer["train_val_idx"])
    test_idx = sorted(int(i) for i in outer["test_idx"])
    labels = get_4class_labels(args.data_root).numpy()
    y_oof, y_test = labels[tv] + 1, labels[test_idx] + 1

    # Best model = stacking (ordinal logistic); see §0.5 / §2.2 of the docs.
    oof_pred, test_pred = stacking_preds(args.out_root, y_oof)
    mo, mt = ordinal_metrics(y_oof, oof_pred), ordinal_metrics(y_test, test_pred)

    cm_figure(y_test, test_pred,
              f"Final stacking ensemble — test (n={mt['n']})\nQWK={mt['qwk']:.3f}  acc={mt['accuracy']:.3f}",
              fig_dir / "fig1_confusion_matrix_test.png")
    cm_figure(y_oof, oof_pred,
              f"Final stacking ensemble — cross-validated OOF (n={mo['n']})\nQWK={mo['qwk']:.3f}  acc={mo['accuracy']:.3f}",
              fig_dir / "fig2_confusion_matrix_oof.png")
    per_stage_figure(y_oof, oof_pred,
                     "Per-stage precision / recall / F1 (stacking, OOF n=363)",
                     fig_dir / "fig3_per_stage_f1_oof.png")
    progression_figure(fig_dir / "fig4_qwk_progression.png")
    print(f"[fig] all figures in {fig_dir}")


if __name__ == "__main__":
    main()
