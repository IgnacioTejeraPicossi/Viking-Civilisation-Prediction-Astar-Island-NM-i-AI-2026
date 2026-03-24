from __future__ import annotations

import numpy as np

from astar_island.constants import TERRAIN_FOREST, TERRAIN_MOUNTAIN, TERRAIN_OCEAN, TERRAIN_PLAINS
from astar_island.features import extract_map_features, in_bounds, is_coast, multi_source_distance


def test_in_bounds():
    assert in_bounds(3, 3, 0, 0)
    assert not in_bounds(3, 3, 3, 0)


def test_coast_detection():
    g = np.array(
        [
            [TERRAIN_OCEAN, TERRAIN_PLAINS, TERRAIN_PLAINS],
            [TERRAIN_OCEAN, TERRAIN_PLAINS, TERRAIN_MOUNTAIN],
            [TERRAIN_OCEAN, TERRAIN_PLAINS, TERRAIN_PLAINS],
        ],
        dtype=np.int16,
    )
    assert is_coast(g, 1, 1)
    assert not is_coast(g, 1, 2)


def test_multi_source_distance():
    m = np.zeros((3, 3), dtype=bool)
    m[1, 1] = True
    d = multi_source_distance(m)
    assert d[1, 1] == 0
    assert d[0, 1] == 1


def test_extract_map_features_shapes():
    grid = [[TERRAIN_OCEAN, TERRAIN_PLAINS], [TERRAIN_PLAINS, TERRAIN_FOREST]]
    feats = extract_map_features(grid, [{"x": 0, "y": 1, "has_port": False, "alive": True}])
    assert feats["terrain"].shape == (2, 2)
    assert feats["settlement_mask"][1, 0]
    assert feats["adjacent_forests"].shape == (2, 2)
