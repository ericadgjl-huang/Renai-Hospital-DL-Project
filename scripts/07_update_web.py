"""Stage 07 — Refresh web_app/_runtime.json after a hierarchy search."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from renai.cli.update_web import main  # noqa: E402

if __name__ == "__main__":
    main()
