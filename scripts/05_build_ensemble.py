"""Stage 05 — Build top-3 ensemble + decide vs single best, per cut.

Usage:
    python scripts/05_build_ensemble.py            # all cuts that have CV done
    python scripts/05_build_ensemble.py --cut 2_vs_3
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from renai.cli.build_ensemble import main  # noqa: E402

if __name__ == "__main__":
    main()
