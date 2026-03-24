from __future__ import annotations

import numpy as np

from astar_island.constants import TERRAIN_TO_CLASS
from astar_island.features import extract_map_features
from astar_island.models import SimResult, SimSettlement, ViewportInfo
from astar_island.observation_store import ObservationStore
from astar_island.predictor_baseline import build_prediction
from astar_island.regime_estimator import RoundRegimeEstimator


def test_build_prediction_sums_to_one():
    grid = [[10, 11], [11, 1]]
    feats = extract_map_features(grid, [{"x": 1, "y": 1, "has_port": True, "alive": True}])
    store = ObservationStore(2, 2, 1)
    regime = RoundRegimeEstimator().estimate(store)
    pred = build_prediction(feats, regime, store, 0)
    assert pred.shape == (2, 2, 6)
    assert np.allclose(pred.sum(axis=-1), 1.0)


def test_empirical_blend():
    grid = [[11, 11], [11, 1]]
    feats = extract_map_features(grid, [{"x": 1, "y": 1, "has_port": False, "alive": True}])
    store = ObservationStore(2, 2, 1)
    sim = SimResult(
        grid=[[1, 1], [1, 1]],
        settlements=[
            SimSettlement(
                x=1, y=1, population=1.0, food=0.5, wealth=0.5, defense=0.5,
                has_port=False, alive=True, owner_id=1,
            )
        ],
        viewport=ViewportInfo(x=0, y=0, w=2, h=2),
        width=2,
        height=2,
        queries_used=1,
        queries_max=50,
    )
    store.add_sim_result(0, sim, TERRAIN_TO_CLASS)
    regime = {"expansion": 0.5, "aggression": 0.5, "trade": 0.5, "harsh_winter": 0.5, "reclamation": 0.5}
    pred = build_prediction(feats, regime, store, 0)
    assert pred.shape == (2, 2, 6)
