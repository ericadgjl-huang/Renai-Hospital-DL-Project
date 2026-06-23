"""Run train_cv across every cut needed by the 5 topologies."""

from __future__ import annotations

import argparse

from ..cuts_registry import CUTS
from ..cv import run_cv_for_cut
from ..models import DEFAULT_BACKBONES
from .common import add_common_args, resolve_splits_dir


def main():
    p = argparse.ArgumentParser(description="Train 5-fold CV for ALL 10 cuts.")
    add_common_args(p)
    p.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    p.add_argument("--only", nargs="*", default=None,
                   help="Optional whitelist of cut names; defaults to all 10")
    args = p.parse_args()

    targets = list(CUTS.keys()) if not args.only else args.only
    for i, name in enumerate(targets, 1):
        print(f"\n############ [{i}/{len(targets)}] CUT={name} ############", flush=True)
        run_cv_for_cut(
            cut=CUTS[name],
            data_root=args.data_root,
            out_root=args.out_root,
            splits_dir=resolve_splits_dir(args),
            backbones=args.backbones,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            weight_decay=args.weight_decay,
            patience=args.patience,
            smoke=args.smoke,
        )


if __name__ == "__main__":
    main()
