"""Stage 11 (experiment) — NO cross-validation, single train/val/test split.

Mirrors the older "調整8" setup: one model per cut (no 5-fold CV), so each
image's embedding comes from a SINGLE model (never fold-averaged). Tests the
hypothesis that, without fold-averaging, the high-dim embeddings behave better.

Everything writes to outputs_nocv/ — the CV pipeline under outputs/ is untouched.

Split (reuses the frozen outer test so numbers stay comparable):
    (train+val) : test = 8:2   (test = the existing 58-image outer test)
    train : val          = 8:2 (within the 229 train_val pool)

Key honesty notes:
  * t-SNE is drawn for TRAIN (seen by the model) AND TEST (unseen) separately.
    A clean train t-SNE but messy test t-SNE = the model memorized train.
  * The combiner meta-classifier is fit on VAL features (the CNN never trained
    on val) and reported on TEST. val is small (~46), so treat embedding
    numbers as illustrative.

Usage:
    python scripts/11_nocv_experiment.py
    python scripts/11_nocv_experiment.py --backbones efficientnet_b0 resnet50 densenet121
    python scripts/11_nocv_experiment.py --pca 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from renai.combiner import ORDINAL_CUTS, STAGE_NAMES, _final_linear  # noqa: E402
from renai.cuts_registry import CUTS  # noqa: E402
from renai.data import (  # noqa: E402
    get_4class_labels,
    make_4class_eval_loader,
    make_loaders_for_cut,
    make_outer_split,
)
from renai.eval import (  # noqa: E402
    bootstrap_classification_ci,
    binary_metrics,
    dump_json,
    save_classification_report,
    save_confusion_matrix,
)
from renai.models import create_model  # noqa: E402
from renai.seed import SEED, set_seed  # noqa: E402
from renai.train import train_one  # noqa: E402


def _balanced_weights(data_root, cut, tr):
    labels4 = get_4class_labels(data_root).numpy()
    b = [cut.relabel(int(labels4[i])) for i in tr]
    n = len(b)
    c = [b.count(0), b.count(1)]
    return [(n / (2.0 * x)) if x > 0 else 0.0 for x in c]


@torch.no_grad()
def _extract(model, linear, loader, device):
    """Return (embeddings (N,D), P(class=1) (N,)) for one model on a loader."""
    store = {}

    def pre(_m, inp):
        store["e"] = inp[0].detach().cpu().numpy().astype(np.float64)

    h = linear.register_forward_pre_hook(pre)
    model.eval()
    embs, probs = [], []
    for imgs, _ in loader:
        out = model(imgs.to(device))
        probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
        embs.append(store["e"])
    h.remove()
    return np.concatenate(embs), np.concatenate(probs)


def _tsne_scatter(X, y, title, path, seed=SEED):
    Xs = StandardScaler().fit_transform(X)
    n = len(Xs)
    perp = float(min(30, max(5, (n - 1) // 3)))
    emb = TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    for s, color in zip([1, 2, 3, 4], ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]):
        m = y == s
        ax.scatter(emb[m, 0], emb[m, 1], s=18, c=color, label=f"stage {s}",
                   alpha=0.7, edgecolors="none")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  [tsne] wrote {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="No-CV single-split experiment.")
    p.add_argument("--data-root", type=Path, default=Path("stage_cls_dataset"))
    p.add_argument("--out-root", type=Path, default=Path("outputs_nocv"))
    p.add_argument("--splits-dir", type=Path, default=Path("outputs/splits"))
    p.add_argument("--backbones", nargs="+",
                   default=["efficientnet_b0", "resnet50", "densenet121"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pca", type=int, default=0, help="PCA for embedding combiner (0=off)")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    set_seed(SEED)
    device = args.device
    out = args.out_root
    (out).mkdir(parents=True, exist_ok=True)

    # --- single split (reuse frozen outer test) ---------------------------
    outer = make_outer_split(args.data_root, args.splits_dir / "outer_split.json")
    labels = get_4class_labels(args.data_root).numpy()
    tv = list(outer["train_val_idx"])
    test = sorted(int(i) for i in outer["test_idx"])
    tr, va = train_test_split(tv, test_size=0.2, random_state=SEED, stratify=labels[tv])
    tr, va = sorted(int(i) for i in tr), sorted(int(i) for i in va)
    print(f"[nocv] split: train={len(tr)} val={len(va)} test={len(test)}", flush=True)
    y_tr, y_va, y_te = labels[tr] + 1, labels[va] + 1, labels[test] + 1

    # --- one model per ordinal cut (pick best backbone by val) ------------
    blocks = {"tr": [], "va": [], "te": []}
    probs = {"tr": [], "va": [], "te": []}
    chosen = []
    for cn in ORDINAL_CUTS:
        cut = CUTS[cn]
        cw = _balanced_weights(args.data_root, cut, tr)
        best = None
        for bb in args.backbones:
            trl, val = make_loaders_for_cut(args.data_root, cut, tr, va,
                                            batch_size=args.batch_size)
            res = train_one(bb, trl, val, out / "cuts" / cn / bb, device,
                            epochs=args.epochs, lr=args.lr,
                            weight_decay=args.weight_decay, patience=args.patience,
                            class_weights=cw)
            print(f"  [nocv] {cn} {bb} val_macro_f1={res.best_val_macro_f1:.4f}", flush=True)
            if best is None or res.best_val_macro_f1 > best[2]:
                best = (bb, res.ckpt_path, res.best_val_macro_f1)
        bb, ckpt, sc = best
        chosen.append(f"{cn}:{bb}(val={sc:.3f})")
        m = create_model(bb, num_classes=2).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
        lin = _final_linear(m, bb)
        for key, idx in (("tr", tr), ("va", va), ("te", test)):
            e, pr = _extract(m, lin, make_4class_eval_loader(args.data_root, idx,
                              batch_size=args.batch_size), device)
            blocks[key].append(e); probs[key].append(pr)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    E = {k: np.concatenate(v, axis=1) for k, v in blocks.items()}      # embeddings
    P = {k: np.column_stack(v) for k, v in probs.items()}              # scalar [P>=2,>=3,>=4]
    print(f"[nocv] chosen backbones: {chosen}", flush=True)
    print(f"[nocv] embed dims: train{E['tr'].shape} test{E['te'].shape}", flush=True)

    # --- t-SNE: TRAIN (seen) vs TEST (unseen) -----------------------------
    _tsne_scatter(E["tr"], y_tr, f"NO-CV embeddings: TRAIN (seen)  {E['tr'].shape[1]}-d",
                  out / "tsne_train_seen.png")
    _tsne_scatter(E["te"], y_te, f"NO-CV embeddings: TEST (unseen)  {E['te'].shape[1]}-d",
                  out / "tsne_test_unseen.png")

    # --- combiner (RandomForest), meta fit on VAL, reported on TEST -------
    def _eval(Xva, yva, Xte, yte, name, pca_n):
        steps = [("scaler", StandardScaler())]
        if pca_n and pca_n > 0:
            steps.append(("pca", PCA(n_components=min(pca_n, Xva.shape[1], len(yva) - 1),
                                     random_state=SEED)))
        steps.append(("clf", RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample", random_state=SEED)))
        pipe = Pipeline(steps)
        pipe.fit(Xva, yva)
        pred = pipe.predict(Xte)
        macro = float(f1_score(yte, pred, average="macro", zero_division=0))
        ci = bootstrap_classification_ci(yte, pred)
        save_confusion_matrix(yte - 1, pred - 1, STAGE_NAMES,
                              out / f"confusion_{name}.png", title=f"no-CV {name}")
        save_classification_report(yte - 1, pred - 1, STAGE_NAMES,
                                   out / f"report_{name}.txt")
        print(f"[nocv] {name:22s} test_macro_f1={macro:.4f} "
              f"(95% CI {ci['macro_f1_ci_low']:.3f}-{ci['macro_f1_ci_high']:.3f})", flush=True)
        return {"method": name, "n_features": int(Xva.shape[1]),
                "test_macro_f1": macro, "test_accuracy": ci["accuracy"],
                "ci_low": ci["macro_f1_ci_low"], "ci_high": ci["macro_f1_ci_high"]}

    rows = [
        _eval(P["va"], y_va, P["te"], y_te, "scalar_ordinal3", 0),
        _eval(E["va"], y_va, E["te"], y_te, "embed_ordinal3_raw", 0),
        _eval(E["va"], y_va, E["te"], y_te, f"embed_ordinal3_pca{args.pca}", args.pca),
    ]
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "comparison_nocv.csv", index=False, encoding="utf-8-sig")
    dump_json({"split": {"train": len(tr), "val": len(va), "test": len(test)},
               "chosen_backbones": chosen, "results": rows},
              out / "summary_nocv.json")
    print("\n[nocv] === done. outputs in outputs_nocv/ ===", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
