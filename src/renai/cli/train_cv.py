"""Run 5-fold CV for a single cut (all 5 backbones)."""

from __future__ import annotations

import argparse

from ..cv import run_cv_for_cut
from ..models import DEFAULT_BACKBONES
from .common import add_common_args, cut_choices, get_cut, resolve_splits_dir


def main():
    p = argparse.ArgumentParser(description="Train 5-fold CV for one cut.")
    add_common_args(p)
    p.add_argument("--cut", required=True, choices=cut_choices())
    p.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    args = p.parse_args()

    cut = get_cut(args.cut)
    res = run_cv_for_cut(
        cut=cut,
        data_root=args.data_root,
        out_root=args.out_root,
        splits_dir=resolve_splits_dir(args),
        backbones=args.backbones,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        smoke=args.smoke,
    )
    print("\n=== summary ===")
    print(f"  written -> {res.summary_csv}")
    for bb, stats in res.per_backbone.items():
        print(
            f"  {bb:18s}  cv_macro_f1={stats['cv_mean_macro_f1']}+/-{stats['cv_std_macro_f1']}  "
            f"acc={stats['cv_mean_acc']}  auc={stats['cv_mean_auc']}  folds={stats['n_folds']}"
        )


if __name__ == "__main__":
    main()
