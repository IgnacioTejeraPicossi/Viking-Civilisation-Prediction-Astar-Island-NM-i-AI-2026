"""Write structured JSON run logs under data/observations/ for post-hoc analysis."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astar_island import __version__
from astar_island.observation_store import ObservationStore
from astar_island.query_planner import WindowPlan


def _repo_root() -> Path:
    """src/astar_island/run_log.py -> parents[2] = project root (editable installs)."""
    return Path(__file__).resolve().parents[2]


def default_observations_dir() -> Path:
    root = _repo_root()
    if "site-packages" in str(root).lower():
        return Path.cwd() / "data" / "observations"
    return root / "data" / "observations"


def serialize_plans(plans: list[WindowPlan]) -> list[dict[str, Any]]:
    return [asdict(p) for p in plans]


def store_summary(store: ObservationStore) -> dict[str, Any]:
    return {
        "settlement_sample_locations": len(store.settlement_samples),
        "observed_fraction_per_seed": [
            round(store.observed_fraction(s), 6) for s in range(store.seeds_count)
        ],
    }


def prediction_log_summary(preds: list[object]) -> list[dict[str, Any]]:
    """
    Cheap per-seed stats for logs (no full H×W×6 tensor): entropy and confidence.
    Useful for tuning without huge JSON files.
    """
    import numpy as np

    eps = 1e-12
    out: list[dict[str, Any]] = []
    for i, p in enumerate(preds):
        arr = np.asarray(p, dtype=np.float64)
        ent = -(arr * np.log(arr + eps)).sum(axis=-1)
        conf = arr.max(axis=-1)
        out.append(
            {
                "seed_index": i,
                "mean_cell_entropy": round(float(ent.mean()), 6),
                "p95_cell_entropy": round(float(np.percentile(ent, 95)), 6),
                "mean_max_prob": round(float(conf.mean()), 6),
            }
        )
    return out


def write_run_log_file(
    payload: dict[str, Any],
    *,
    log_dir: Path | None = None,
    prefix: str = "run",
) -> Path:
    """
    Atomically write JSON to data/observations/{prefix}_YYYYMMDDTHHMMSSZ.json.

    Returns the path written.
    """
    log_dir = log_dir or default_observations_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = log_dir / f"{prefix}_{ts}.json"
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    path.write_text(text, encoding="utf-8")
    return path
