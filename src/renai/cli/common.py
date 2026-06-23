"""Shared CLI helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..cuts_registry import CUTS

DEFAULT_DATA_ROOT = Path("stage_cls_dataset")
DEFAULT_OUT_ROOT = Path("outputs")
DEFAULT_SPLITS_DIR = DEFAULT_OUT_ROOT / "splits"


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                   help="ImageFolder with stage_1..stage_4 subfolders")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                   help="Where to write outputs/cuts/<name>/...")
    p.add_argument("--splits-dir", type=Path, default=None,
                   help="Where outer_split.json is cached (default: <out-root>/splits)")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW decoupled L2 (regularization for small data)")
    p.add_argument("--patience", type=int, default=8,
                   help="Early-stop after N epochs without val-macro-F1 gain (0=off)")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run: 1 fold, ≤2 epochs — for sanity checking")


def resolve_splits_dir(args) -> Path:
    return args.splits_dir if args.splits_dir is not None else (args.out_root / "splits")


def cut_choices() -> list[str]:
    return sorted(CUTS.keys())


def get_cut(name: str):
    if name not in CUTS:
        raise SystemExit(f"Unknown cut '{name}'. Choices: {cut_choices()}")
    return CUTS[name]
