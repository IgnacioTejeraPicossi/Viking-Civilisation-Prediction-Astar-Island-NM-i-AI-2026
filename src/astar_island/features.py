"""Static features from initial map + settlement positions (no API queries)."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from astar_island.constants import NEI4, NEI8, TERRAIN_FOREST, TERRAIN_MOUNTAIN, TERRAIN_OCEAN

# Chebyshev distance threshold to flag a cell as "near a coastal settlement"
NEAR_COASTAL_SETTLEMENT_DIST = 4


def in_bounds(h: int, w: int, y: int, x: int) -> bool:
    return 0 <= y < h and 0 <= x < w


def is_land(v: int) -> bool:
    """Land cell for coast detection (excludes ocean and mountain)."""
    return v not in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN)


def is_coast(grid: np.ndarray, y: int, x: int) -> bool:
    if not is_land(int(grid[y, x])):
        return False
    hh, ww = grid.shape
    for dx, dy in NEI4:
        yy, xx = y + dy, x + dx
        if in_bounds(hh, ww, yy, xx) and grid[yy, xx] == TERRAIN_OCEAN:
            return True
    return False


def multi_source_distance(mask: np.ndarray) -> np.ndarray:
    """Manhattan (4-connected) BFS distance from True cells; unreachable stays large."""
    h, w = mask.shape
    dist = np.full((h, w), 10**9, dtype=np.int32)
    q: deque[tuple[int, int]] = deque()
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        dist[y, x] = 0
        q.append((y, x))
    while q:
        y, x = q.popleft()
        for dx, dy in NEI4:
            yy, xx = y + dy, x + dx
            if in_bounds(h, w, yy, xx) and dist[yy, xx] > dist[y, x] + 1:
                dist[yy, xx] = dist[y, x] + 1
                q.append((yy, xx))
    return dist


def chebyshev_multi_source_distance(mask: np.ndarray) -> np.ndarray:
    """Chebyshev (8-connected / chessboard) BFS distance from True cells; unreachable stays large.

    Chebyshev distance = max(|dy|, |dx|), equivalent to 8-directional unit-cost BFS.
    Used for distance-binned priors (matches competitor's regime-conditional model).
    """
    h, w = mask.shape
    dist = np.full((h, w), 10**9, dtype=np.int32)
    q: deque[tuple[int, int]] = deque()
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        dist[y, x] = 0
        q.append((y, x))
    while q:
        y, x = q.popleft()
        d = dist[y, x]
        for dx, dy in NEI8:
            yy, xx = y + dy, x + dx
            if in_bounds(h, w, yy, xx) and dist[yy, xx] > d + 1:
                dist[yy, xx] = d + 1
                q.append((yy, xx))
    return dist


def extract_map_features(grid: list[list[int]], settlements: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """
    Build numpy feature layers for one seed's initial state.

    Ocean and mountain are non-transitable for coast / distance semantics.
    """
    arr = np.array(grid, dtype=np.int16)
    h, w = arr.shape

    ocean_mask = arr == TERRAIN_OCEAN
    mountain_mask = arr == TERRAIN_MOUNTAIN
    forest_mask = arr == TERRAIN_FOREST

    coast_mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            coast_mask[y, x] = is_coast(arr, y, x)

    settlement_mask = np.zeros((h, w), dtype=bool)
    port_mask = np.zeros((h, w), dtype=bool)
    for s in settlements:
        settlement_mask[s["y"], s["x"]] = True
        if s.get("has_port"):
            port_mask[s["y"], s["x"]] = True

    dist_settlement = (
        multi_source_distance(settlement_mask) if settlement_mask.any() else np.full((h, w), 999, dtype=np.int32)
    )
    dist_coast = multi_source_distance(coast_mask) if coast_mask.any() else np.full((h, w), 999, dtype=np.int32)

    # Chebyshev (chessboard) distances — used for distance-binned priors
    coastal_settlement_mask = settlement_mask & coast_mask
    cheb_dist_settlement = (
        chebyshev_multi_source_distance(settlement_mask)
        if settlement_mask.any()
        else np.full((h, w), 999, dtype=np.int32)
    )
    cheb_dist_coast_settlement = (
        chebyshev_multi_source_distance(coastal_settlement_mask)
        if coastal_settlement_mask.any()
        else np.full((h, w), 999, dtype=np.int32)
    )

    adjacent_forests = np.zeros((h, w), dtype=np.int16)
    adjacent_mountains = np.zeros((h, w), dtype=np.int16)
    for y in range(h):
        for x in range(w):
            cf, cm = 0, 0
            for dx, dy in NEI8:
                yy, xx = y + dy, x + dx
                if not in_bounds(h, w, yy, xx):
                    continue
                v = int(arr[yy, xx])
                if v == TERRAIN_FOREST:
                    cf += 1
                if v == TERRAIN_MOUNTAIN:
                    cm += 1
            adjacent_forests[y, x] = cf
            adjacent_mountains[y, x] = cm

    frontier_score = np.exp(-dist_settlement / 3.0) * (~mountain_mask) * (~ocean_mask)

    return {
        "terrain": arr,
        "ocean_mask": ocean_mask,
        "mountain_mask": mountain_mask,
        "forest_mask": forest_mask,
        "coast_mask": coast_mask,
        "settlement_mask": settlement_mask,
        "port_mask": port_mask,
        "coastal_settlement_mask": coastal_settlement_mask,
        "dist_settlement": dist_settlement,
        "dist_coast": dist_coast,
        "adjacent_forests": adjacent_forests,
        "adjacent_mountains": adjacent_mountains,
        "frontier_score": frontier_score.astype(np.float64),
        # Chebyshev distances for distance-binned priors
        "cheb_dist_settlement": cheb_dist_settlement,
        "cheb_dist_coast_settlement": cheb_dist_coast_settlement,
    }
