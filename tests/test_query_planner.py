from __future__ import annotations

import numpy as np

from astar_island import config
from astar_island.features import extract_map_features
from astar_island.query_planner import initial_query_plan, total_planned_queries


def _tiny_map():
    grid = [[10, 11, 11], [11, 1, 4], [11, 4, 5]]
    settlements = [{"x": 1, "y": 1, "has_port": False, "alive": True}]
    return extract_map_features(grid, settlements)


def test_plan_respects_budget():
    feats = _tiny_map()
    plans = initial_query_plan([feats], 3, 3, max_queries=config.MAX_QUERIES_PER_ROUND)
    assert total_planned_queries(plans) <= config.MAX_QUERIES_PER_ROUND
    assert total_planned_queries(plans) >= 1


def test_low_max_queries_finishes_no_infinite_loop():
    """Regression: trim must drop windows when all repeats==1 but sum > max_q."""
    from astar_island.constants import TERRAIN_PLAINS
    from astar_island.features import extract_map_features

    grid = [[TERRAIN_PLAINS] * 40 for _ in range(40)]
    settlements = [{"x": 5, "y": 5, "has_port": False, "alive": True}]
    feats = extract_map_features(grid, settlements)
    all_feats = [feats] * 5
    plans = initial_query_plan(all_feats, 40, 40, max_queries=5)
    assert total_planned_queries(plans) <= 5
    assert total_planned_queries(plans) >= 1
