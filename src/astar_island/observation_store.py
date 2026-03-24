"""Aggregate stochastic viewport samples per cell and settlement stats."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from astar_island.models import SimResult


@dataclass
class CellObs:
    counts: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int32))
    n: int = 0

    def update(self, cls_idx: int) -> None:
        self.counts[cls_idx] += 1
        self.n += 1

    def probs(self, alpha: float = 0.5) -> np.ndarray:
        p = self.counts.astype(np.float64) + alpha
        return p / p.sum()


class ObservationStore:
    def __init__(self, width: int, height: int, seeds_count: int) -> None:
        self.width = width
        self.height = height
        self.seeds_count = seeds_count
        self.cell_obs: list[list[list[CellObs]]] = [
            [[CellObs() for _ in range(width)] for _ in range(height)] for _ in range(seeds_count)
        ]
        self.settlement_samples: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)

    def add_sim_result(
        self,
        seed_index: int,
        sim_result: SimResult,
        terrain_to_class_map: dict[int, int],
    ) -> None:
        vx = sim_result.viewport.x
        vy = sim_result.viewport.y
        grid = sim_result.grid

        for dy, row in enumerate(grid):
            for dx, v in enumerate(row):
                x = vx + dx
                y = vy + dy
                cls_idx = terrain_to_class_map[int(v)]
                self.cell_obs[seed_index][y][x].update(cls_idx)

        for s in sim_result.settlements:
            self.settlement_samples[(seed_index, s.x, s.y)].append(
                {
                    "population": s.population,
                    "food": s.food,
                    "wealth": s.wealth,
                    "defense": s.defense,
                    "has_port": s.has_port,
                    "alive": s.alive,
                    "owner_id": s.owner_id,
                }
            )

    def empirical_cell_probs(self, seed_index: int, y: int, x: int) -> np.ndarray | None:
        obs = self.cell_obs[seed_index][y][x]
        return obs.probs() if obs.n > 0 else None

    def cell_count(self, seed_index: int, y: int, x: int) -> int:
        return self.cell_obs[seed_index][y][x].n

    def observed_fraction(self, seed_index: int) -> float:
        """Fraction of map cells with at least one observation for this seed."""
        total = self.width * self.height
        if total == 0:
            return 0.0
        n_obs = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.cell_obs[seed_index][y][x].n > 0:
                    n_obs += 1
        return n_obs / total
