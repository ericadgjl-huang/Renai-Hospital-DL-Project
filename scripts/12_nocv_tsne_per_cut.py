"""Stage 12 (experiment) — per-cut t-SNE for the no-CV models, TRAIN vs TEST.

Reuses the checkpoints trained by 11_nocv_experiment.py (NO retraining). For
each of the 3 ordinal cuts it draws a t-SNE of that cut's single-model
embedding, coloured by 4-class stage, for TRAIN (seen) and TEST (unseen).

Each cut is a BINARY classifier, so its embedding is optimized to separate its
own two sides (e.g. 1_vs_234 -> {1} vs {2,3,4}); do not expect a clean 4-class
split from a single cut. Writes to outputs_nocv/per_cut_tsne/.

Usage:
    python scripts/12_nocv_tsne_per_cut.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from renai.combiner import _final_linear  # noqa: E402
from renai.data import get_4class_labels, make_4class_eval_loader, make_outer_split  # noqa: E402
from renai.models import create_model  # noqa: E402
from renai.seed import SEED  # noqa: E402


@torch.no_grad()
def _embed(model, linear, loader, device):
    store = {}

    def pre(_m, inp):
        store["e"] = inp[0].detach().cpu().numpy().astype(np.float64)

    h = linear.register_forward_pre_hook(pre)
    model.eval()
    out = []
    for imgs, _ in loader:
        model(imgs.to(device))
        out.append(store["e"])
    h.remove()
    return np.concatenate(out)


def _scatter(X, y, title, path, seed=SEED):
    Xs = StandardScaler().fit_transform(X)
    n = len(Xs)
    perp = float(min(30, max(5, (n - 1) // 3)))
    emb = TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    for s, c in zip([1, 2, 3, 4], ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]):
        m = y == s
        ax.scatter(emb[m, 0], emb[m, 1], s=20, c=c, label=f"stage {s}", alpha=0.75, edgecolors="none")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  [tsne] wrote {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Per-cut no-CV t-SNE (train vs test).")
    p.add_argument("--data-root", type=Path, default=Path("stage_cls_dataset"))
    p.add_argument("--out-root", type=Path, default=Path("outputs_nocv"))
    p.add_argument("--splits-dir", type=Path, default=Path("outputs/splits"))
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = args.device

    summary = json.loads((args.out_root / "summary_nocv.json").read_text(encoding="utf-8"))
    chosen = {}  # cut -> backbone
    for item in summary["chosen_backbones"]:
        m = re.match(r"([^:]+):([^(]+)", item)
        chosen[m.group(1)] = m.group(2)
    print(f"[per-cut] chosen backbones: {chosen}", flush=True)

    outer = make_outer_split(args.data_root, args.splits_dir / "outer_split.json")
    labels = get_4class_labels(args.data_root).numpy()
    tv = list(outer["train_val_idx"])
    test = sorted(int(i) for i in outer["test_idx"])
    tr, _va = train_test_split(tv, test_size=0.2, random_state=SEED, stratify=labels[tv])
    tr = sorted(int(i) for i in tr)
    y_tr, y_te = labels[tr] + 1, labels[test] + 1

    tdir = args.out_root / "per_cut_tsne"
    tdir.mkdir(parents=True, exist_ok=True)

    for cn, bb in chosen.items():
        ckpt = args.out_root / "cuts" / cn / bb / f"best_{bb}.pth"
        m = create_model(bb, num_classes=2).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
        lin = _final_linear(m, bb)
        e_tr = _embed(m, lin, make_4class_eval_loader(args.data_root, tr, batch_size=16), device)
        e_te = _embed(m, lin, make_4class_eval_loader(args.data_root, test, batch_size=16), device)
        _scatter(e_tr, y_tr, f"{cn} [{bb}] TRAIN seen ({e_tr.shape[1]}-d)", tdir / f"{cn}_train.png")
        _scatter(e_te, y_te, f"{cn} [{bb}] TEST unseen ({e_te.shape[1]}-d)", tdir / f"{cn}_test.png")
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[per-cut] done -> {tdir}", flush=True)


if __name__ == "__main__":
    main()
