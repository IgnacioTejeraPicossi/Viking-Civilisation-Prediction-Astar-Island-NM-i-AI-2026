#!/usr/bin/env python3
"""Run full round (simulate + predict + submit). Set AINM_TOKEN."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without install: repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

from astar_island.cli import main

if __name__ == "__main__":
    load_dotenv(_ROOT / ".env")
    raise SystemExit(main())
