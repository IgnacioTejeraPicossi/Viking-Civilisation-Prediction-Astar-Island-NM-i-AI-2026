"""Terrain codes, prediction class indices, and small geometry helpers."""

from __future__ import annotations

# --- API terrain codes (initial grid and simulate viewport) ---
TERRAIN_OCEAN = 10
TERRAIN_PLAINS = 11
TERRAIN_EMPTY = 0
TERRAIN_SETTLEMENT = 1
TERRAIN_PORT = 2
TERRAIN_RUIN = 3
TERRAIN_FOREST = 4
TERRAIN_MOUNTAIN = 5

# --- Prediction classes (6-way distribution per cell) ---
CLASS_EMPTY = 0
CLASS_SETTLEMENT = 1
CLASS_PORT = 2
CLASS_RUIN = 3
CLASS_FOREST = 4
CLASS_MOUNTAIN = 5

CLASS_NAMES = ("Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain")

TERRAIN_NAMES: dict[int, str] = {
    TERRAIN_OCEAN: "Ocean",
    TERRAIN_PLAINS: "Plains",
    TERRAIN_EMPTY: "Empty",
    TERRAIN_SETTLEMENT: "Settlement",
    TERRAIN_PORT: "Port",
    TERRAIN_RUIN: "Ruin",
    TERRAIN_FOREST: "Forest",
    TERRAIN_MOUNTAIN: "Mountain",
}

# Ocean, plains, and empty collapse to prediction class Empty.
TERRAIN_TO_CLASS: dict[int, int] = {
    TERRAIN_OCEAN: CLASS_EMPTY,
    TERRAIN_PLAINS: CLASS_EMPTY,
    TERRAIN_EMPTY: CLASS_EMPTY,
    TERRAIN_SETTLEMENT: CLASS_SETTLEMENT,
    TERRAIN_PORT: CLASS_PORT,
    TERRAIN_RUIN: CLASS_RUIN,
    TERRAIN_FOREST: CLASS_FOREST,
    TERRAIN_MOUNTAIN: CLASS_MOUNTAIN,
}

NEI4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEI8 = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)


def terrain_to_class(value: int) -> int:
    """Map a terrain code to a 6-class prediction index."""
    if value not in TERRAIN_TO_CLASS:
        raise ValueError(f"Unknown terrain code: {value}")
    return TERRAIN_TO_CLASS[value]


def is_static_terrain_code(value: int) -> bool:
    """True for cells treated as mostly static for baseline (ocean, mountain, forest)."""
    return value in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN, TERRAIN_FOREST)


def is_land_code(value: int) -> bool:
    """True for cells considered walkable/transitable land (exclude ocean and mountain)."""
    return value not in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN)
