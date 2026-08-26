"""Stage 25 — Merge Ficat stage 2 & 3 into one class (professor's request).

Turns the 4-class problem {1,2,3,4} into an ordinal 3-class problem
{1, 2+3(merged), 4}. Stage 2 vs 3 is the hardest adjacent boundary, so a
clinician may only need {early / mid / collapse}.

No CNN retraining needed: the trained CORN ensemble already outputs cumulative
probs P(>=2), P(>=3), P(>=4). Merging 2&3 just DROPS the >=3 threshold — the
3-class cumulative probs are exactly [P(>=2), P(>=4)]. We report both:
  * voting   : average [P>=2, P>=4] over backbones, rank-count decode.
  * stacking : a low-capacity ordinal meta over the 9x3 features (OOF-selected).

Reuses the saved oof/test cumulative .npy (CPU-only, fast).

Usage:
    python scripts/25_merge23_experiment.py --out-root outputs_merged_patient_rebox \
        --data-root stage_cls_merged_rebox
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from torchvision.datasets import ImageFolder

from renai.data import get_4class_labels, make_outer_split, patient_groups
from renai.seed import SEED, set_seed

# 4-class stage -> merged 3-class label: 1->1, 2->2, 3->2, 4->3
MERGE = {1: 1, 2: 2, 3: 2, 4: 3}
NAMES3 = ["stage_1", "stage_2+3", "stage_4"]


def merge3(y4):
    return np.array([MERGE[int(v)] for v in y4])


class OrdinalMeta3(BaseEstimator, ClassifierMixin):
    """3-class all-threshold ordinal meta (thresholds: class>=2, class>=3)."""
    def __init__(self, C=1.0):
        self.C = C
    def fit(self, X, y):
        self.models = []
        for thr in (2, 3):
            t = (y >= thr).astype(int)
            lr = LogisticRegression(class_weight="balanced", C=self.C, max_iter=3000).fit(X, t)
            self.models.append((lr, list(lr.classes_).index(1)))
        self.classes_ = np.array([1, 2, 3])
        return self
    def _cum(self, X):
        Q = np.column_stack([m.predict_proba(X)[:, p] for m, p in self.models])
        for k in range(1, Q.shape[1]):
            Q[:, k] = np.minimum(Q[:, k], Q[:, k - 1])
        return Q
    def predict(self, X):
        return np.clip(np.rint(1 + self._cum(X).sum(1)), 1, 3).astype(int)
    def predict_proba(self, X):
        Q = self._cum(X)
        P = np.clip(np.column_stack([1 - Q[:, 0], Q[:, 0] - Q[:, 1], Q[:, 1]]), 0, None)
        return P / P.sum(1, keepdims=True)


def metrics3(y, p):
    return {"n": int(len(y)), "accuracy": round(float(accuracy_score(y, p)), 3),
            "qwk": round(float(cohen_kappa_score(y, p, labels=[1, 2, 3], weights="quadratic")), 3),
            "macro_f1": round(float(f1_score(y, p, labels=[1, 2, 3], average="macro", zero_division=0)), 3),
            "recall": [round(float(r), 3) for r in
                       recall_score(y, p, labels=[1, 2, 3], average=None, zero_division=0)]}


def cm_fig(y, p, title, path):
    cm = confusion_matrix(y, p, labels=[1, 2, 3])
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(NAMES3, rotation=20, ha="right"); ax.set_yticklabels(NAMES3)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]*100:.0f}%)", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "#222", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


def load_stacks(out_root, kind):
    return [np.load(f) for s in glob.glob(f"{out_root}/corn*") if os.path.isdir(s)
            for bb in sorted(os.listdir(s))
            if os.path.exists(f := f"{s}/{bb}/{kind}_cumulative.npy")]


def main():
    ap = argparse.ArgumentParser(description="Merge stage 2&3; 3-class ordinal result.")
    ap.add_argument("--out-root", type=Path, default=Path("outputs_merged_patient_rebox"))
    ap.add_argument("--data-root", type=Path, default=Path("stage_cls_merged_rebox"))
    ap.add_argument("--groups-csv", default="outputs_merged/patient_groups.csv")
    args = ap.parse_args()
    set_seed(SEED)

    groups = patient_groups(args.data_root, args.groups_csv) if args.groups_csv else None
    outer = make_outer_split(args.data_root, args.out_root / "splits" / "outer_split.json", groups=groups)
    tv = sorted(int(i) for i in outer["train_val_idx"])
    te = sorted(int(i) for i in outer["test_idx"])
    lab = get_4class_labels(args.data_root).numpy()
    y_oof, y_test = merge3(lab[tv] + 1), merge3(lab[te] + 1)

    oof = load_stacks(args.out_root, "oof")            # list of (N,3) per backbone
    test = load_stacks(args.out_root, "test")
    B = len(oof)
    # per-backbone [P>=2, P>=4] (drop >=3), then averaged for voting
    oof_v = np.mean([o[:, [0, 2]] for o in oof], axis=0)
    test_v = np.mean([t[:, [0, 2]] for t in test], axis=0)

    def decode(P2):   # rank-count with [>=2, >=4]
        return 1 + (P2 >= 0.5).sum(1).astype(int)

    results = {}
    # --- voting ---
    results["voting"] = {"oof": metrics3(y_oof, decode(oof_v)),
                         "test": metrics3(y_test, decode(test_v))}
    vote_test_pred = decode(test_v)

    # --- stacking: 9x3=27 features, OOF-select meta by QWK ---
    Xo = np.nan_to_num(np.hstack(oof)); Xt = np.nan_to_num(np.hstack(test))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    metas = {"logreg": LogisticRegression(class_weight="balanced", max_iter=3000),
             "svm_linear": SVC(kernel="linear", class_weight="balanced")}
    for C in (0.3, 1.0, 3.0):
        metas[f"ordinal(C={C})"] = OrdinalMeta3(C=C)
    best = None
    for name, clf in metas.items():
        q = cohen_kappa_score(y_oof, cross_val_predict(clf, Xo, y_oof, cv=cv),
                              labels=[1, 2, 3], weights="quadratic")
        print(f"  [stacking] {name:14s} OOF QWK={q:.4f}")
        if best is None or q > best[0]:
            best = (q, name, clf)
    _, best_name, best_clf = best
    oof_pred = cross_val_predict(best_clf, Xo, y_oof, cv=cv)
    best_clf.fit(Xo, y_oof)
    stack_test_pred = best_clf.predict(Xt)
    results["stacking"] = {"meta": best_name,
                           "oof": metrics3(y_oof, oof_pred),
                           "test": metrics3(y_test, stack_test_pred)}

    # figures + save
    fig_dir = args.out_root / "merge23"; fig_dir.mkdir(parents=True, exist_ok=True)
    cm_fig(y_oof, oof_pred, f"3-class (2+3 merged) stacking OOF\nacc={results['stacking']['oof']['accuracy']} "
           f"QWK={results['stacking']['oof']['qwk']}", fig_dir / "confusion_oof_stacking.png")
    cm_fig(y_test, stack_test_pred, f"3-class stacking test (n={len(y_test)})",
           fig_dir / "confusion_test_stacking.png")
    (fig_dir / "result.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ 合併 2&3 → 3 分類 {1, 2+3, 4} ================")
    for m in ("voting", "stacking"):
        o, t = results[m]["oof"], results[m]["test"]
        tag = m + (f"[{results[m]['meta']}]" if m == "stacking" else "")
        print(f"  {tag:26s} OOF: acc={o['accuracy']} QWK={o['qwk']} mf1={o['macro_f1']} "
              f"recall(1/2+3/4)={o['recall']}")
        print(f"  {'':26s} test: acc={t['accuracy']} QWK={t['qwk']} mf1={t['macro_f1']}")
    print(f"\n  (對照 4 分類 stacking OOF: acc 0.675 / QWK 0.799)")
    print(f"[merge23] wrote {fig_dir}/result.json + confusion matrices")


if __name__ == "__main__":
    main()
