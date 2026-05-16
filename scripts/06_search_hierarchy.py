"""Stage 06 — Exhaustive 5-topology search; pick best by 4-class macro-F1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from renai.cli.search_hierarchy import main  # noqa: E402

if __name__ == "__main__":
    main()
