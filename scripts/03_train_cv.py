"""Stage 03 — 5-fold CV for one cut. Saves the validation-best ckpt per fold.

Usage:
    python scripts/03_train_cv.py --cut 1_vs_234 --epochs 30
    python scripts/03_train_cv.py --cut 2_vs_3 --smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly without `pip install -e .`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from renai.cli.train_cv import main  # noqa: E402

if __name__ == "__main__":
    main()
