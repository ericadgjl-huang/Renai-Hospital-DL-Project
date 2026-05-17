"""Build top-3 cross-family ensemble for one cut, decide vs single best."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from ..cuts_registry import CUTS
from ..ensemble import build_ensemble_for_cut
from .common import add_common_args, cut_choices, resolve_splits_dir


def main():
    p = argparse.ArgumentParser(description="Build ensemble & pick winner for a cut.")
    add_common_args(p)
    p.add_argument("--cut", choices=cut_choices(),
                   help="If omitted, runs over every cut found in outputs/cuts/")
    args = p.parse_args()

    targets = [args.cut] if args.cut else list(CUTS.keys())
    for name in targets:
        cut_dir = args.out_root / "cuts" / name
        if not (cut_dir / "summary.csv").exists():
            print(f"  skipping {name}: no summary.csv yet (run train_cv first)")
            continue
        print(f"\n=== ensemble for cut={name} ===", flush=True)
        decision = build_ensemble_for_cut(
            cut=CUTS[name],
            data_root=args.data_root,
            out_root=args.out_root,
            splits_dir=resolve_splits_dir(args),
            device=args.device,
            batch_size=args.batch_size,
        )
        print(f"  -> winner: {decision.chosen}  oof_macro_f1={decision.oof_macro_f1:.4f}")


if __name__ == "__main__":
    main()
