#!/usr/bin/env python3
"""Fetch /analysis for completed rounds and persist JSON for meta-model training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from astar_island.api_client import AstarClient
from astar_island import config


def main() -> None:
    config.require_token()
    client = AstarClient()
    out_dir = _ROOT / "data" / "rounds"
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in client.my_rounds():
        if r.get("status") != "completed":
            continue
        rid = r["id"]
        n = int(r.get("seeds_count", 5))
        for seed_index in range(n):
            try:
                data = client.analysis(rid, seed_index)
            except Exception as e:  # noqa: BLE001
                print(rid, seed_index, e)
                continue
            path = out_dir / f"{rid}_seed{seed_index}.json"
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print("wrote", path)


if __name__ == "__main__":
    main()
