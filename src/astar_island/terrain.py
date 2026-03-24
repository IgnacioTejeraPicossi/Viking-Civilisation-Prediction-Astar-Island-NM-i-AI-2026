"""Terrain helpers shared across features and predictors."""

from __future__ import annotations

from astar_island.constants import TERRAIN_MOUNTAIN, TERRAIN_OCEAN


def is_passable_terrain(code: int) -> bool:
    """False for ocean and mountain (no movement through)."""
    return code not in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN)
