"""Stage 08 — Bootstrap 95% CIs for the final test metrics.

Reads predictions already written by 05_build_ensemble.py and
06_search_hierarchy.py; trains nothing. Run it after 06.

Usage:
    python scripts/08_report_ci.py
    python scripts/08_report_ci.py --n-boot 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from renai.report import build_ci_report  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Bootstrap CI report.")
    p.add_argument("--out-root", type=Path, default=Path("outputs"))
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_ci_report(args.out_root, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
