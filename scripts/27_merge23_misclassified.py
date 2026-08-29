"""Stage 27 — Export misclassified crops for the 3-class (2&3 merged) models.

The professor wants to SEE the X-rays where the merged class "stage 2+3" is
predicted as "stage 1" (and the other error types). scripts/25 & 26 only saved
confusion matrices; this exports the actual crops, foldered by true->pred, for
BOTH the re-derive (no retrain, scripts/25) and the from-scratch (retrained,
scripts/26) 3-class models.

Predictions are the cross-validated OOF stacking predictions (each image scored
by a fold model that did not train on it) — the same predictions behind the
reported confusion matrices, and the largest set of examples.

Output (per model):
    <dst>/<model>/true_{T}__pred_{P}/*.jpg   (T,P in {1, 2+3, 4})
    <dst>/<model>/misclassified.csv
where <dst> = outputs_merged_patient_rebox/merge23_misclassified/

Usage:
    python scripts/27_merge23_misclassified.py
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from torchvision.datasets import ImageFolder

from renai.data import get_4class_labels, make_outer_split, patient_groups
from renai.seed import SEED, set_seed

NAME3 = {1: "1", 2: "2+3", 3: "4"}
LABELDIR = {1: "1", 2: "2and3", 3: "4"}       # folder-safe


class OrdinalMeta3(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
    def fit(self, X, y):
        self.models = []
        for thr in (2, 3):
            lr = LogisticRegression(class_weight="balanced", C=self.C, max_iter=3000).fit(X, (y >= thr).astype(int))
            self.models.append((lr, list(lr.classes_).index(1)))
        self.classes_ = np.array([1, 2, 3]); return self
    def _cum(self, X):
        Q = np.column_stack([m.predict_proba(X)[:, p] for m, p in self.models])
        Q[:, 1] = np.minimum(Q[:, 1], Q[:, 0]); return Q
    def predict(self, X):
        return np.clip(np.rint(1 + self._cum(X).sum(1)), 1, 3).astype(int)


def load_cum(out_root, kind, cols=None):
    P = []
    for s in glob.glob(f"{out_root}/corn*"):
        if os.path.isdir(s):
            for bb in sorted(os.listdir(s)):
                f = f"{s}/{bb}/{kind}_cumulative.npy"
                if os.path.exists(f):
                    a = np.load(f)
                    P.append(a[:, cols] if cols is not None else a)
    return P


def oof_preds(out_root, data_root, splits_dir, merged_labels, cols):
    """Cross-validated OOF stacking predictions + true 3-class labels + image paths."""
    groups = patient_groups(data_root, "outputs_merged/patient_groups.csv")
    outer = make_outer_split(data_root, splits_dir / "outer_split.json", groups=groups)
    tv = sorted(int(i) for i in outer["train_val_idx"])
    y = merged_labels[tv]
    X = np.nan_to_num(np.hstack(load_cum(out_root, "oof", cols)))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    metas = {"logreg": LogisticRegression(class_weight="balanced", max_iter=3000),
             "svm_linear": SVC(kernel="linear", class_weight="balanced"),
             "ordinal(C=1)": OrdinalMeta3(1.0), "ordinal(C=3)": OrdinalMeta3(3.0)}
    best = max(((cohen_kappa_score(y, cross_val_predict(c, X, y, cv=cv), labels=[1, 2, 3],
                                   weights="quadratic"), n, c) for n, c in metas.items()),
               key=lambda t: t[0])
    _, best_name, best_clf = best
    pred = cross_val_predict(best_clf, X, y, cv=cv)
    paths = [ImageFolder(data_root).samples[i][0] for i in tv]
    return y, pred, paths, best_name


def export(model_tag, y, pred, paths, dst: Path):
    root = dst / model_tag
    if root.exists():
        shutil.rmtree(root)
    rows = []
    for t, p, path in zip(y, pred, paths):
        if t == p:
            continue
        folder = root / f"true_{LABELDIR[int(t)]}__pred_{LABELDIR[int(p)]}"
        folder.mkdir(parents=True, exist_ok=True)
        src = Path(path)
        shutil.copy2(src, folder / f"true{LABELDIR[int(t)]}_as_pred{LABELDIR[int(p)]}__{src.name}")
        rows.append({"crop_filename": src.name, "true": NAME3[int(t)], "pred": NAME3[int(p)],
                     "note": f"真實 {NAME3[int(t)]} → 誤判為 {NAME3[int(p)]}"})
    df = pd.DataFrame(rows).sort_values(["true", "pred", "crop_filename"]) if rows else pd.DataFrame()
    df.to_csv(root / "misclassified.csv", index=False, encoding="utf-8-sig")
    n_23_1 = int(((np.asarray(y) == 2) & (np.asarray(pred) == 1)).sum())
    print(f"[{model_tag}] 共 {len(rows)} 張分錯（OOF）；其中『2+3 → 1』= {n_23_1} 張 -> {root}")
    return len(rows), n_23_1


def main():
    set_seed(SEED)
    dst = ROOT / "outputs_merged_patient_rebox" / "merge23_misclassified"
    dst.mkdir(parents=True, exist_ok=True)

    # --- (A) re-derive (no retrain): 4-class rebox model, labels merged 2&3 ---
    dr4 = ROOT / "stage_cls_merged_rebox"
    lab4 = get_4class_labels(dr4).numpy() + 1
    merged4 = np.array([{1: 1, 2: 2, 3: 2, 4: 3}[int(v)] for v in lab4])
    yA, pA, pathsA, mA = oof_preds(ROOT / "outputs_merged_patient_rebox", dr4,
                                   ROOT / "outputs_merged_patient_rebox" / "splits",
                                   merged4, cols=[0, 2])   # [P>=2, P>=4]
    export(f"沒重訓_改解碼[{mA}]", yA, pA, pathsA, dst)

    # --- (B) from-scratch (retrained 3-class): 3-class dataset labels ---
    dr3 = ROOT / "stage_cls_merged_rebox_3cls"
    lab3 = get_4class_labels(dr3).numpy() + 1   # folder idx 0,1,2 -> 1,2,3
    yB, pB, pathsB, mB = oof_preds(ROOT / "outputs_corn3_rebox", dr3,
                                   ROOT / "outputs_corn3_rebox" / "splits",
                                   lab3, cols=None)        # already 2-dim
    export(f"有重訓_從頭[{mB}]", yB, pB, pathsB, dst)

    print(f"\n[done] 兩個版本的分錯圖都在 {dst}")
    print("  每個資料夾是一種錯誤類型（true_X__pred_Y），"
          "『2+3 誤判為 1』= 資料夾 true_2and3__pred_1")


if __name__ == "__main__":
    main()
