"""Search the 5 binary-tree topologies, pick the best by 4-class macro-F1."""

from __future__ import annotations

import argparse

from ..cuts_registry import CUTS
from ..hierarchy import search_best_topology
from .common import add_common_args, resolve_splits_dir


def main():
    p = argparse.ArgumentParser(description="Search best hierarchy topology.")
    add_common_args(p)
    args = p.parse_args()

    winner = search_best_topology(
        cuts=CUTS,
        data_root=args.data_root,
        out_root=args.out_root,
        splits_dir=resolve_splits_dir(args),
        device=args.device,
        batch_size=args.batch_size,
    )
    print("\n=== Hierarchy search result ===")
    print(winner)


if __name__ == "__main__":
    main()
