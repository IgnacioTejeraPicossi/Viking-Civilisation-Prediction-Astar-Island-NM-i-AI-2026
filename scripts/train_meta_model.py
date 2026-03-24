#!/usr/bin/env python3
"""Train LightGBM meta-model from data/rounds/*.json (after backfill)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def main() -> None:
    rounds_dir = _ROOT / "data" / "rounds"
    files = list(rounds_dir.glob("*.json"))
    if not files:
        print("No JSON in data/rounds — run scripts/backfill_completed_rounds.py first")
        sys.exit(1)
    # Placeholder: load ground_truth tensors and build X/y for LightGBM
    _ = [json.loads(p.read_text(encoding="utf-8")) for p in files[:1]]
    print("Stub: implement feature extraction from analysis JSON + LGBM training.")


if __name__ == "__main__":
    main()
