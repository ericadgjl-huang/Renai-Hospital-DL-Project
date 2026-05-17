"""Build the fold-voting ensemble for one (or every) cut."""

from __future__ import annotations

import argparse

from ..cuts_registry import CUTS
from ..ensemble import build_ensemble_for_cut, summarize_all_cuts
from .common import add_common_args, cut_choices, resolve_splits_dir


def main():
    p = argparse.ArgumentParser(description="Soft-vote per-fold ckpts for each cut.")
    add_common_args(p)
    p.add_argument("--cut", choices=cut_choices(),
                   help="If omitted, runs over every cut found in outputs/cuts/")
    args = p.parse_args()

    targets = [args.cut] if args.cut else list(CUTS.keys())
    for name in targets:
        cut_dir = args.out_root / "cuts" / name
        if not (cut_dir / "cv").exists():
            print(f"  skipping {name}: no cv/ directory yet (run train_cv first)")
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
        print(
            f"  -> chosen={decision.chosen}  n_members={len(decision.members)}\n"
            f"     OOF voting={decision.oof_macro_f1_voting:.4f}  "
            f"OOF stacking={decision.oof_macro_f1_stacking:.4f}\n"
            f"     test_macro_f1={decision.test_macro_f1:.4f}  "
            f"test_acc={decision.test_accuracy:.4f}  "
            f"test_auc={decision.test_auc:.4f}"
        )

    df = summarize_all_cuts(args.out_root)
    if not df.empty:
        roll_csv = args.out_root / "ensemble_summary.csv"
        df.to_csv(roll_csv, index=False, encoding="utf-8-sig")
        print(f"\nrolled-up summary -> {roll_csv}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
