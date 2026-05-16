"""Stage 04 — Run 5-fold CV for every cut needed by all 5 topologies.

This is the long one — 10 cuts × 5 backbones × 5 folds + 10×5 final retrains.
Plan for ~hours on a single GPU.

Usage:
    python scripts/04_train_all_cuts.py
    python scripts/04_train_all_cuts.py --smoke
    python scripts/04_train_all_cuts.py --only 1_vs_234 2_vs_3
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from renai.cli.train_all_cuts import main  # noqa: E402

if __name__ == "__main__":
    main()
