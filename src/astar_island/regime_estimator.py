"""Bayesian regime estimation from stochastic viewport observations.

The hidden round regime controls settlement survival, spawning, ruin formation,
forest reclamation, spatial spread, and coastal affinity.  All five seeds share
the same regime, so observations are pooled across seeds for better estimates.

Each parameter is estimated via a Beta-posterior mean (Bayesian shrinkage):

    estimate = (prior_n * prior_mean + n_successes) / (prior_n + n_total)

Prior strengths (prior_n) reflect how much data is needed to move away from
the prior; they were inspired by the top NM i AI 2026 solution (MIT licence).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from astar_island.constants import (
    CLASS_EMPTY,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_SETTLEMENT,
    TERRAIN_MOUNTAIN,
    TERRAIN_OCEAN,
)
from astar_island.observation_store import ObservationStore

# ---------------------------------------------------------------------------
# Bayesian prior hyper-parameters (prior_n, prior_mean)
# Inspired by the top NM i AI 2026 Rust solution (MIT licence)
# ---------------------------------------------------------------------------
_PRIORS: dict[str, tuple[float, float]] = {
    "survival_rate":       (20.0, 0.60),
    "spawn_rate":          (8.0,  0.05),
    "ruin_rate":           (40.0, 0.05),   # strong prior — ruin is rare
    "forest_reclamation":  (15.0, 0.08),
    "spatial_decay":       (10.0, 0.30),
    "coastal_spawn_boost": (10.0, 1.50),
}

CHAOTIC_THRESHOLD = 0.95  # spatial_decay above this → chaotic round flag


# ---------------------------------------------------------------------------
# RegimeEstimate
# ---------------------------------------------------------------------------

@dataclass
class RegimeEstimate:
    """Estimated hidden round parameters."""

    survival_rate: float = 0.60        # fraction of initial settlements that survive
    spawn_rate: float = 0.05           # new-settlement density among non-initial land cells
    ruin_rate: float = 0.05            # fraction of observed land cells that are ruins
    forest_reclamation: float = 0.08   # fraction of observed land cells that are forest
    spatial_decay: float = 0.30        # clustering of spawns around initial settlements
    coastal_spawn_boost: float = 1.50  # ratio coastal vs inland new-settlement rate
    is_chaotic: bool = False           # spatial_decay > CHAOTIC_THRESHOLD

    def as_vector(self) -> np.ndarray:
        """Return the 6-dimensional regime feature vector (for Ridge model)."""
        return np.array([
            self.survival_rate,
            self.spawn_rate,
            self.ruin_rate,
            self.forest_reclamation,
            self.spatial_decay,
            self.coastal_spawn_boost,
        ], dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "survival_rate": self.survival_rate,
            "spawn_rate": self.spawn_rate,
            "ruin_rate": self.ruin_rate,
            "forest_reclamation": self.forest_reclamation,
            "spatial_decay": self.spatial_decay,
            "coastal_spawn_boost": self.coastal_spawn_boost,
            "is_chaotic": self.is_chaotic,
        }

    # backward-compat shim so old code that consumed a plain dict still works
    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)


# ---------------------------------------------------------------------------
# Bayesian helper
# ---------------------------------------------------------------------------

def _bayes(n_success: float, n_total: float, param: str) -> float:
    prior_n, prior_mean = _PRIORS[param]
    return (prior_n * prior_mean + n_success) / (prior_n + n_total)


# ---------------------------------------------------------------------------
# BayesianRegimeEstimator
# ---------------------------------------------------------------------------

class BayesianRegimeEstimator:
    """Estimate round regime from pooled viewport observations across all seeds."""

    def estimate(
        self,
        store: ObservationStore,
        all_features: list[dict[str, Any]],
        initial_states: list[Any],  # list[InitialState] from Pydantic model
    ) -> RegimeEstimate:
        """
        Parameters
        ----------
        store:
            Populated ObservationStore after all simulate calls.
        all_features:
            Output of ``extract_map_features`` for each seed — must include
            ``coast_mask``, ``ocean_mask``, ``mountain_mask``,
            ``cheb_dist_settlement``.
        initial_states:
            ``detail.initial_states`` — list of ``InitialState`` Pydantic models
            with ``.settlements`` (list of ``InitialSettlement``).
        """
        # ------------------------------------------------------------------
        # Build initial-position look-ups
        # ------------------------------------------------------------------
        # init_positions[seed_idx] = set of (y, x) tuples
        init_positions: list[set[tuple[int, int]]] = []
        init_coastal: list[set[tuple[int, int]]] = []
        for s_idx, state in enumerate(initial_states):
            pos: set[tuple[int, int]] = set()
            c_pos: set[tuple[int, int]] = set()
            feats = all_features[s_idx] if s_idx < len(all_features) else {}
            coast = feats.get("coast_mask")
            for s in state.settlements:
                pos.add((s.y, s.x))
                if coast is not None and bool(coast[s.y, s.x]):
                    c_pos.add((s.y, s.x))
            init_positions.append(pos)
            init_coastal.append(c_pos)

        # ------------------------------------------------------------------
        # survival_rate  —  from settlement_samples on INITIAL positions
        # ------------------------------------------------------------------
        alive_count = 0.0
        total_surv = 0.0
        for (seed_idx, x, y), samples in store.settlement_samples.items():
            if (y, x) in init_positions[seed_idx]:
                for s in samples:
                    total_surv += 1.0
                    if s["alive"]:
                        alive_count += 1.0
        survival_rate = _bayes(alive_count, total_surv, "survival_rate")

        # ------------------------------------------------------------------
        # Pool all cell observations across seeds (non-static land cells only)
        # ------------------------------------------------------------------
        total_land_obs = 0.0
        ruin_obs = 0.0
        forest_obs = 0.0
        spawn_obs = 0.0          # settlement/port on NON-initial land cells
        spawn_total = 0.0        # total obs on non-initial land cells
        coastal_spawn_obs = 0.0  # settlement/port on coastal non-initial cells
        coastal_spawn_total = 0.0
        inland_spawn_obs = 0.0
        inland_spawn_total = 0.0

        # For spatial_decay: weighted mean Chebyshev distance of spawn observations
        decay_dist_weighted = 0.0
        decay_weight_sum = 0.0

        for seed_idx in range(store.seeds_count):
            feats = all_features[seed_idx] if seed_idx < len(all_features) else {}
            ocean_mask = feats.get("ocean_mask")
            mountain_mask = feats.get("mountain_mask")
            coast_mask = feats.get("coast_mask")
            cheb_dist = feats.get("cheb_dist_settlement")
            init_pos = init_positions[seed_idx]

            for y in range(store.height):
                for x in range(store.width):
                    obs = store.cell_obs[seed_idx][y][x]
                    if obs.n == 0:
                        continue
                    # Skip static cells
                    if ocean_mask is not None and bool(ocean_mask[y, x]):
                        continue
                    if mountain_mask is not None and bool(mountain_mask[y, x]):
                        continue

                    n = float(obs.n)
                    total_land_obs += n
                    ruin_obs += float(obs.counts[CLASS_RUIN])
                    forest_obs += float(obs.counts[CLASS_FOREST])

                    # Spawn on non-initial positions
                    if (y, x) not in init_pos:
                        n_spawn_obs = float(obs.counts[CLASS_SETTLEMENT] + obs.counts[CLASS_PORT])
                        spawn_obs += n_spawn_obs
                        spawn_total += n
                        is_coast = coast_mask is not None and bool(coast_mask[y, x])
                        if is_coast:
                            coastal_spawn_obs += n_spawn_obs
                            coastal_spawn_total += n
                        else:
                            inland_spawn_obs += n_spawn_obs
                            inland_spawn_total += n
                        # Spatial decay: weight by spawn probability
                        if cheb_dist is not None:
                            d = float(cheb_dist[y, x])
                            decay_dist_weighted += n_spawn_obs * d
                            decay_weight_sum += n_spawn_obs

        ruin_rate = _bayes(ruin_obs, total_land_obs, "ruin_rate")
        forest_reclamation = _bayes(forest_obs, total_land_obs, "forest_reclamation")
        spawn_rate = _bayes(spawn_obs, spawn_total, "spawn_rate")

        # ------------------------------------------------------------------
        # spatial_decay  —  how concentrated are new spawns around initial settlements?
        # High decay (~1) = spawns very close to initial positions.
        # ------------------------------------------------------------------
        if decay_weight_sum > 1e-6:
            mean_spawn_dist = decay_dist_weighted / decay_weight_sum
            # exp(-mean_dist / 5) maps dist=0 → 1.0, dist=5 → 0.37, dist=13+ → ≈0.07
            spatial_decay_emp = float(np.exp(-mean_spawn_dist / 5.0))
        else:
            spatial_decay_emp = _PRIORS["spatial_decay"][1]

        # Bayesian shrinkage (treat as proportion with prior)
        prior_n_decay, prior_mean_decay = _PRIORS["spatial_decay"]
        eff_obs_decay = max(decay_weight_sum, 0.0)
        spatial_decay = (prior_n_decay * prior_mean_decay + eff_obs_decay * spatial_decay_emp) / (
            prior_n_decay + eff_obs_decay
        )

        # ------------------------------------------------------------------
        # coastal_spawn_boost
        # ------------------------------------------------------------------
        if coastal_spawn_total > 0 and inland_spawn_total > 0:
            coastal_rate = coastal_spawn_obs / coastal_spawn_total
            inland_rate = inland_spawn_obs / inland_spawn_total
            if inland_rate > 1e-6:
                boost_emp = float(np.clip(coastal_rate / inland_rate, 0.5, 5.0))
            else:
                boost_emp = _PRIORS["coastal_spawn_boost"][1]
            n_boost = min(coastal_spawn_total + inland_spawn_total, 50.0)
            prior_n_b, prior_mean_b = _PRIORS["coastal_spawn_boost"]
            coastal_spawn_boost = (prior_n_b * prior_mean_b + n_boost * boost_emp) / (
                prior_n_b + n_boost
            )
        else:
            coastal_spawn_boost = _PRIORS["coastal_spawn_boost"][1]

        # Clip everything to valid ranges
        spatial_decay = float(np.clip(spatial_decay, 0.01, 0.99))
        is_chaotic = spatial_decay > CHAOTIC_THRESHOLD

        return RegimeEstimate(
            survival_rate=float(np.clip(survival_rate, 0.05, 0.99)),
            spawn_rate=float(np.clip(spawn_rate, 0.001, 0.50)),
            ruin_rate=float(np.clip(ruin_rate, 0.001, 0.50)),
            forest_reclamation=float(np.clip(forest_reclamation, 0.001, 0.80)),
            spatial_decay=spatial_decay,
            coastal_spawn_boost=float(np.clip(coastal_spawn_boost, 0.5, 4.0)),
            is_chaotic=is_chaotic,
        )


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

class RoundRegimeEstimator:
    """Legacy shim — delegates to BayesianRegimeEstimator with empty feature/state lists."""

    def estimate(self, observation_store: ObservationStore) -> RegimeEstimate:
        return BayesianRegimeEstimator().estimate(observation_store, [], [])
