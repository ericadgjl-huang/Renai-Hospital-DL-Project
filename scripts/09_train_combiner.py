"""Stage 09 — Learned 4-class combiner over per-cut probabilities (idea #4 v1).

Compares a soft, learned combiner (boosting / tree / SVM kernels / logreg over
cut-probability features) against the fixed-hierarchy hard routing. Selection is
by OOF macro-F1; the test set is report-only.

Run after 05_build_ensemble.py (and 06 if you want the 'topology3' feature set).

Usage:
    python scripts/09_train_combiner.py
    python scripts/09_train_combiner.py --n-boot 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from renai.combiner import compare_with_hierarchy  # noqa: E402
from renai.cli.common import add_common_args, resolve_splits_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Train learned combiner; compare to hierarchy.")
    add_common_args(p)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    compare_with_hierarchy(
        out_root=args.out_root,
        data_root=args.data_root,
        splits_dir=resolve_splits_dir(args),
        device=args.device,
        batch_size=args.batch_size,
        n_boot=args.n_boot,
    )


if __name__ == "__main__":
    main()
