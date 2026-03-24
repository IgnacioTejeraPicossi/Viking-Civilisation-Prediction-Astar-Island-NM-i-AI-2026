from __future__ import annotations

import json
from pathlib import Path

from astar_island.query_planner import WindowPlan
import numpy as np

from astar_island.run_log import prediction_log_summary, serialize_plans, write_run_log_file


def test_serialize_plans():
    plans = [WindowPlan(0, 1, 2, 15, 15, "anchor", 3)]
    assert serialize_plans(plans)[0]["tag"] == "anchor"


def test_prediction_log_summary():
    p = np.full((2, 2, 6), 1.0 / 6.0)
    s = prediction_log_summary([p])
    assert len(s) == 1
    assert "mean_cell_entropy" in s[0]


def test_write_run_log_file(tmp_path: Path):
    p = write_run_log_file({"meta": {"test": True}}, log_dir=tmp_path, prefix="test")
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["meta"]["test"] is True
