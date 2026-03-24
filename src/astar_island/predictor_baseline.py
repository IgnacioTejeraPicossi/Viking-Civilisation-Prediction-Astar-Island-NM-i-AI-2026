"""Heuristic probabilistic predictor using distance-binned priors + empirical blend.

Replaces the old flat terrain-type prior table with Chebyshev distance-binned,
coastal-aware priors from ``gt_priors.GTPriorStore``.  Regime parameters from
``BayesianRegimeEstimator`` are used to scale the priors before blending.

For static cells (ocean → Empty, mountain → Mountain) a deterministic high-
confidence prior is applied regardless of distance bin.
"""

from __future__ import annotations

import numpy as np

from astar_island.constants import (
    CLASS_EMPTY,
    CLASS_MOUNTAIN,
    TERRAIN_MOUNTAIN,
    TERRAIN_OCEAN,
)
from astar_island.gt_priors import GTPriorStore, get_cell_category
from astar_island.observation_store import ObservationStore
from astar_island.regime_estimator import RegimeEstimate

# High-confidence priors for static terrain types
_OCEAN_PRIOR = np.array([0.97, 0.005, 0.005, 0.005, 0.01, 0.005])
_MOUNTAIN_PRIOR = np.array([0.01, 0.005, 0.005, 0.005, 0.005, 0.97])


def normalize_probs(p: np.ndarray, floor: float = 0.001) -> np.ndarray:
    p = np.maximum(np.asarray(p, dtype=np.float64), floor)
    return p / p.sum()


def build_prediction(
    seed_features: dict[str, np.ndarray],
    regime: RegimeEstimate | dict,
    observation_store: ObservationStore,
    seed_index: int,
    prior_store: GTPriorStore | None = None,
) -> np.ndarray:
    """Build an H×W×6 prediction tensor for one seed.

    Parameters
    ----------
    seed_features:
        Feature dict from ``extract_map_features`` (must include
        ``cheb_dist_settlement`` and ``cheb_dist_coast_settlement``).
    regime:
        Current ``RegimeEstimate`` (or legacy plain dict — both supported).
    observation_store:
        Populated observation store.
    seed_index:
        Which seed index to predict.
    prior_store:
        Optional ``GTPriorStore``; a default (no fitted priors) store is used
        when ``None``.
    """
    # Support both the new RegimeEstimate dataclass and the old plain dict
    if isinstance(regime, dict):
        from astar_island.regime_estimator import RegimeEstimate as RE
        regime = RE(
            survival_rate=regime.get("expansion", 0.60),
            spawn_rate=0.05,
            ruin_rate=regime.get("aggression", 0.05),
            forest_reclamation=regime.get("reclamation", 0.08),
            spatial_decay=0.30,
            coastal_spawn_boost=1.5 if regime.get("trade", 0.5) > 0.5 else 1.0,
        )

    if prior_store is None:
        prior_store = GTPriorStore()

    h, w = seed_features["terrain"].shape
    pred = np.zeros((h, w, 6), dtype=np.float64)

    terrain = seed_features["terrain"]
    ocean_mask = seed_features["ocean_mask"]
    mountain_mask = seed_features["mountain_mask"]

    for y in range(h):
        for x in range(w):
            t = int(terrain[y, x])

            # ----- Static cells — deterministic priors -----
            if t == TERRAIN_OCEAN or bool(ocean_mask[y, x]):
                prior = normalize_probs(_OCEAN_PRIOR.copy())
            elif t == TERRAIN_MOUNTAIN or bool(mountain_mask[y, x]):
                prior = normalize_probs(_MOUNTAIN_PRIOR.copy())
            else:
                # ----- Dynamic land cell — distance-binned prior -----
                cat = get_cell_category(seed_features, y, x)
                prior = prior_store.get_prior(cat, regime)

            # ----- Blend with empirical observations -----
            empirical = observation_store.empirical_cell_probs(seed_index, y, x)
            if empirical is None:
                pred[y, x] = prior
            else:
                n = observation_store.cell_count(seed_index, y, x)
                # λ increases with observation count; cap at 0.80
                lam = min(0.80, 0.20 + 0.10 * n)
                pred[y, x] = normalize_probs((1.0 - lam) * prior + lam * empirical)

    return pred
