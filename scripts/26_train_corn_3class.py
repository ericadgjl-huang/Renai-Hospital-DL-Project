"""Stage 26 — Train a proper 3-threshold CORN from scratch (stage 2&3 merged).

The professor's "merge 2&3" request, done rigorously: instead of re-decoding the
4-class model (scripts/25), this trains dedicated CORN models on the 3-class
ordinal problem {1, 2+3, 4} — so each backbone allocates its K-1 = 2 thresholds
(>=class2, >=class3) directly to the boundaries that matter.

Same patient-level split (same groups CSV + seed) as the 4-class rebox run, so
results are comparable. Trains the 9-backbone ensemble (class-weighted), saves
per-backbone OOF/test cumulative probs, then finalises voting + stacking.

Usage (conda env unet_labeling):
    python scripts/26_train_corn_3class.py --device 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from renai.corn import (corn_cumulative_probs, corn_loss, corn_pos_weights, create_corn_model)
from renai.data import (get_4class_labels, make_cv_folds, make_eval_transform,
                        make_outer_split, make_train_transform, patient_groups)
from renai.seed import SEED, set_seed, torch_generator

K = 3                      # merged classes: 1, 2+3, 4
NAMES = ["stage_1", "stage_2+3", "stage_4"]
IMGSZ = 384


def decode3(cum: np.ndarray) -> np.ndarray:
    """rank-count on the 2 cumulative probs [P>=2, P>=3] (monotone-enforced)."""
    Q = cum.copy()
    Q[:, 1] = np.minimum(Q[:, 1], Q[:, 0])
    return 1 + (Q >= 0.5).sum(1).astype(int)


def metrics3(y, p):
    return {"n": int(len(y)), "accuracy": round(float(accuracy_score(y, p)), 3),
            "qwk": round(float(cohen_kappa_score(y, p, labels=[1, 2, 3], weights="quadratic")), 3),
            "macro_f1": round(float(f1_score(y, p, labels=[1, 2, 3], average="macro", zero_division=0)), 3),
            "recall": [round(float(r), 3) for r in
                       recall_score(y, p, labels=[1, 2, 3], average=None, zero_division=0)]}


def qwk3(y, p):
    if len(np.unique(np.concatenate([y, p]))) < 2:
        return 0.0
    return float(cohen_kappa_score(y, p, labels=[1, 2, 3], weights="quadratic"))


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


def train_loader(data_root, idx, bs):
    base = ImageFolder(data_root, transform=make_train_transform(IMGSZ))
    return DataLoader(Subset(base, list(idx)), batch_size=bs, shuffle=True, num_workers=0,
                      generator=torch_generator())


@torch.no_grad()
def cum_on(model, data_root, idx, device, bs=16):
    base = ImageFolder(data_root, transform=make_eval_transform(IMGSZ))  # 3 folders OK (no 4-class assert)
    loader = DataLoader(Subset(base, list(idx)), batch_size=bs, shuffle=False, num_workers=0)
    P, ys = [], []
    for imgs, lbls in loader:
        P.append(corn_cumulative_probs(model(imgs.to(device))).cpu().numpy())
        ys.extend(int(l) + 1 for l in lbls.numpy())
    return (np.concatenate(P) if P else np.zeros((0, K - 1))), np.asarray(ys)


def train_fold(bb, tr, va, data_root, device, epochs, lr, wd, patience, bs, ckpt, rin, class_weight):
    model = create_corn_model(bb, radimagenet_dir=rin, num_classes=K).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    tl = train_loader(data_root, tr, bs)
    pw = None
    if class_weight:
        st = get_4class_labels(data_root).numpy()[list(tr)] + 1
        pw = corn_pos_weights(st, device, num_classes=K)
    best, best_state, since = -2.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for imgs, lbls in tl:
            stages = (lbls + 1).to(device)
            opt.zero_grad()
            corn_loss(model(imgs.to(device)), stages, pos_weights=pw, num_classes=K).backward()
            opt.step()
        Pv, yv = cum_on(model, data_root, [int(g) for g in va], device)
        q = qwk3(yv, decode3(Pv))
        if q > best:
            best, since = q, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if patience and since >= patience:
                break
    torch.save(best_state or model.state_dict(), ckpt)
    return best


def run_backbone(bb, rin, args, tv, test_idx, groups):
    sub = "corn_cw_radimagenet" if rin else "corn_cw"
    bdir = args.out_root / sub / bb
    bdir.mkdir(parents=True, exist_ok=True)
    pos = {g: i for i, g in enumerate(tv)}
    oof = np.full((len(tv), K - 1), np.nan)
    test_sum, n = None, 0
    lr = 1e-3 if rin else 1e-4
    ep = 40 if rin else 30
    pat = 10 if rin else 8
    for fi, (tr, va) in enumerate(make_cv_folds(args.data_root, tv, groups=groups)):
        t0 = time.time()
        ck = bdir / f"fold_{fi}.pth"
        train_fold(bb, tr, va, args.data_root, args.device, ep, lr, 1e-4, pat, args.batch,
                   ck, str(args.radimagenet) if rin else None, True)
        m = create_corn_model(bb, num_classes=K).to(args.device)
        m.load_state_dict(torch.load(ck, map_location=args.device)); m.eval()
        Pv, _ = cum_on(m, args.data_root, [int(g) for g in va], args.device)
        for j, g in enumerate(va):
            oof[pos[int(g)]] = Pv[j]
        Pt, _ = cum_on(m, args.data_root, test_idx, args.device)
        test_sum = Pt if test_sum is None else test_sum + Pt
        n += 1
        print(f"  [{bb}{'(R)' if rin else ''}] fold {fi} done {time.time()-t0:.0f}s", flush=True)
        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    np.save(bdir / "oof_cumulative.npy", oof)
    np.save(bdir / "test_cumulative.npy", test_sum / max(n, 1))


def cm_fig(y, p, title, path):
    cm = confusion_matrix(y, p, labels=[1, 2, 3]); cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(NAMES, rotation=20, ha="right"); ax.set_yticklabels(NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]*100:.0f}%)", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "#222", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="From-scratch 3-class CORN (stage 2&3 merged).")
    ap.add_argument("--data-root", type=Path, default=Path("stage_cls_merged_rebox_3cls"))
    ap.add_argument("--out-root", type=Path, default=Path("outputs_corn3_rebox"))
    ap.add_argument("--groups-csv", default="outputs_merged/patient_groups.csv")
    ap.add_argument("--radimagenet", type=Path, default=Path("weights/radimagenet"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"
    set_seed(SEED)

    groups = patient_groups(args.data_root, args.groups_csv) if args.groups_csv else None
    outer = make_outer_split(args.data_root, args.out_root / "splits" / "outer_split.json", groups=groups)
    tv = sorted(int(i) for i in outer["train_val_idx"])
    test_idx = sorted(int(i) for i in outer["test_idx"])
    lab = get_4class_labels(args.data_root).numpy()
    y_oof, y_test = lab[tv] + 1, lab[test_idx] + 1
    print(f"[corn3] train_val={len(tv)} test={len(test_idx)} | classes: {NAMES}", flush=True)

    IMAGENET = ["efficientnet_b0", "efficientnet_b1", "convnext_tiny", "convnext_small",
                "densenet121", "densenet169", "resnet50"]
    RIN = ["resnet50", "densenet121"]
    for bb in IMAGENET:
        print(f"\n[corn3] === {bb} (ImageNet) ===", flush=True)
        run_backbone(bb, False, args, tv, test_idx, groups)
    for bb in RIN:
        print(f"\n[corn3] === {bb} (RadImageNet) ===", flush=True)
        run_backbone(bb, True, args, tv, test_idx, groups)

    # ---- finalize: voting + stacking ----
    def load(kind):
        return [np.load(f) for s in glob.glob(f"{args.out_root}/corn*") if os.path.isdir(s)
                for bb in sorted(os.listdir(s)) if os.path.exists(f := f"{s}/{bb}/{kind}_cumulative.npy")]
    oof, test = load("oof"), load("test")
    oof_v, test_v = np.mean(oof, axis=0), np.mean(test, axis=0)
    res = {"voting": {"oof": metrics3(y_oof, decode3(oof_v)), "test": metrics3(y_test, decode3(test_v))}}

    Xo, Xt = np.nan_to_num(np.hstack(oof)), np.nan_to_num(np.hstack(test))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    metas = {"logreg": LogisticRegression(class_weight="balanced", max_iter=3000),
             "svm_linear": SVC(kernel="linear", class_weight="balanced"),
             "ordinal(C=1)": OrdinalMeta3(1.0), "ordinal(C=3)": OrdinalMeta3(3.0)}
    best = max(((qwk3(y_oof, cross_val_predict(c, Xo, y_oof, cv=cv)), n, c) for n, c in metas.items()),
               key=lambda t: t[0])
    _, best_name, best_clf = best
    oof_pred = cross_val_predict(best_clf, Xo, y_oof, cv=cv)
    best_clf.fit(Xo, y_oof)
    stack_test = best_clf.predict(Xt)
    res["stacking"] = {"meta": best_name, "oof": metrics3(y_oof, oof_pred), "test": metrics3(y_test, stack_test)}

    fdir = args.out_root / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    cm_fig(y_oof, oof_pred, f"3-class CORN (from scratch) stacking OOF\n"
           f"acc={res['stacking']['oof']['accuracy']} QWK={res['stacking']['oof']['qwk']}",
           fdir / "confusion_oof_stacking.png")
    cm_fig(y_test, stack_test, f"3-class CORN stacking test (n={len(y_test)})",
           fdir / "confusion_test_stacking.png")
    (args.out_root / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ 從頭訓練 3-threshold CORN (2&3 合併) ================")
    for m in ("voting", "stacking"):
        o, t = res[m]["oof"], res[m]["test"]
        tag = m + (f"[{res[m].get('meta','')}]" if m == "stacking" else "")
        print(f"  {tag:24s} OOF: acc={o['accuracy']} QWK={o['qwk']} mf1={o['macro_f1']} recall={o['recall']}")
        print(f"  {'':24s} test: acc={t['accuracy']} QWK={t['qwk']} mf1={t['macro_f1']}")
    print("  (對照 scripts/25 re-derive: voting OOF acc 0.747 / stacking OOF acc 0.739)")
    print(f"[corn3] wrote {args.out_root}/result.json + figures/")


if __name__ == "__main__":
    main()
