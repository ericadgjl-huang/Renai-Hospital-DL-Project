"""Stage 10 — t-SNE visualization of the per-cut penultimate embeddings.

VISUALIZATION ONLY. t-SNE has no transform and must not be used as classifier
features (see renai.combiner). This just shows whether the high-dim embeddings
separate the 4 stages. Reuses existing checkpoints (no training).

Usage:
    python scripts/10_tsne_embeddings.py
    python scripts/10_tsne_embeddings.py --perplexity 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from renai.combiner import visualize_embeddings_tsne  # noqa: E402
from renai.cli.common import add_common_args, resolve_splits_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="t-SNE of per-cut embeddings (visualization).")
    add_common_args(p)
    p.add_argument("--perplexity", type=float, default=30.0)
    args = p.parse_args()
    visualize_embeddings_tsne(
        out_root=args.out_root,
        data_root=args.data_root,
        splits_dir=resolve_splits_dir(args),
        device=args.device,
        perplexity=args.perplexity,
    )


if __name__ == "__main__":
    main()
