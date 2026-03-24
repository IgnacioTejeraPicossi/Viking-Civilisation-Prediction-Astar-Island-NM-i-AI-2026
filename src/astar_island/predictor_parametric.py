"""Regime-conditional parametric predictor.

Implements the full three-component blend from the top NM i AI 2026 solution
(MIT licence), adapted to Python:

    80 % parametric  — Ridge-regression model: regime_vector → per-bin class probs
    15 % kernel      — Gaussian-weighted average of past-round GT priors
     5 % scaled avg  — training-set mean priors scaled by regime ratios

When ``data/regime_priors.json`` does not yet exist (no training data) the
model transparently falls back to ``predictor_baseline.build_prediction``, which
already uses distance-binned priors + regime scaling.  Results improve
progressively as more completed rounds are backfilled.

Post-processing applied unconditionally:
    • Temperature scaling (T = 1.04) — mild softening for calibration
    • Port constraint   — zero port probability on non-coastal cells
    • Mountain constraint — zero mountain probability on non-mountain initial cells
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from astar_island import config
from astar_island.constants import (
    TERRAIN_MOUNTAIN,
    TERRAIN_OCEAN,
)
from astar_island.gt_priors import GTPriorStore, get_cell_category
from astar_island.observation_store import ObservationStore
from astar_island.predictor_baseline import normalize_probs
from astar_island.regime_estimator import RegimeEstimate


# ---------------------------------------------------------------------------
# Module-level shared GTPriorStore (loaded once per process)
# ---------------------------------------------------------------------------

_STORE_CACHE: dict[str, GTPriorStore] = {}


def _get_store(priors_path: Path | None = None) -> GTPriorStore:
    """Return a cached GTPriorStore, loading from JSON if the file exists."""
    path = priors_path or config.get_regime_priors_path()
    key = str(path)
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = GTPriorStore(path if Path(path).exists() else None)
    return _STORE_CACHE[key]


def invalidate_store_cache() -> None:
    """Force reload of regime priors on the next prediction call."""
    _STORE_CACHE.clear()


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def apply_temperature_scaling(pred: np.ndarray, T: float = 1.04) -> np.ndarray:
    """Temperature-scale a probability tensor (H × W × 6).

    Divides the log-probabilities by T then re-normalises.  T > 1 softens the
    distribution slightly, improving calibration.
    """
    eps = 1e-12
    log_p = np.log(pred + eps) / T
    log_p -= log_p.max(axis=-1, keepdims=True)
    exp_p = np.exp(log_p)
    return exp_p / exp_p.sum(axis=-1, keepdims=True)


def enforce_constraints(pred: np.ndarray, feats: dict[str, Any]) -> np.ndarray:
    """Enforce hard physical constraints on the prediction tensor.

    Rules (inspired by top NM i AI 2026 solution, MIT licence):
      • Port (class 2) is impossible on non-coastal cells.
      • Mountain (class 5) is impossible on cells that are not mountains in
        the initial grid (mountains are static — they cannot appear or vanish).

    After zeroing, rows are renormalised with a small floor to avoid division
    by zero and infinite KL.
    """
    pred = pred.copy()
    coast: np.ndarray = feats["coast_mask"].astype(np.float64)        # 1.0 coastal, 0.0 inland
    mountain: np.ndarray = feats["mountain_mask"].astype(np.float64)  # 1.0 mountain

    # Port impossible inland
    pred[:, :, 2] *= coast[:, :, np.newaxis].reshape(pred.shape[0], pred.shape[1])
    # Mountain impossible on dynamic cells
    pred[:, :, 5] *= mountain[:, :, np.newaxis].reshape(pred.shape[0], pred.shape[1])

    # Renormalise with floor
    pred = np.maximum(pred, 1e-6)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------

class RegimeConditionalPredictor:
    """Full parametric predictor with post-processing.

    Usage
    -----
    predictor = RegimeConditionalPredictor()          # loads JSON if present
    preds = predictor.build_all_seeds(all_features, regime, store)
    """

    def __init__(self, priors_path: Path | None = None) -> None:
        self._store = _get_store(priors_path)

    # ------------------------------------------------------------------
    # Per-cell prior
    # ------------------------------------------------------------------

    def _cell_prior(self, feats: dict, y: int, x: int, regime: RegimeEstimate) -> np.ndarray:
        """Return regime-conditioned prior for one cell."""
        t = int(feats["terrain"][y, x])
        if bool(feats["ocean_mask"][y, x]):
            return normalize_probs(np.array([0.97, 0.005, 0.005, 0.005, 0.01, 0.005]))
        if t == TERRAIN_MOUNTAIN or bool(feats["mountain_mask"][y, x]):
            return normalize_probs(np.array([0.01, 0.005, 0.005, 0.005, 0.005, 0.97]))
        cat = get_cell_category(feats, y, x)
        return self._store.get_prior(cat, regime)

    # ------------------------------------------------------------------
    # Single-seed prediction
    # ------------------------------------------------------------------

    def build_prediction(
        self,
        seed_features: dict[str, Any],
        regime: RegimeEstimate,
        observation_store: ObservationStore,
        seed_index: int,
        *,
        temperature: float = config.TEMPERATURE_SCALING,
        apply_constraint: bool = True,
    ) -> np.ndarray:
        """Return an H×W×6 prediction tensor for one seed."""
        h, w = seed_features["terrain"].shape
        pred = np.zeros((h, w, 6), dtype=np.float64)

        for y in range(h):
            for x in range(w):
                prior = self._cell_prior(seed_features, y, x, regime)
                empirical = observation_store.empirical_cell_probs(seed_index, y, x)

                if empirical is None:
                    pred[y, x] = prior
                else:
                    n = observation_store.cell_count(seed_index, y, x)
                    # Slightly more weight on empirical when parametric priors are fitted
                    max_lam = 0.85 if self._store._has_parametric else 0.80
                    lam = min(max_lam, 0.20 + 0.10 * n)
                    pred[y, x] = normalize_probs((1.0 - lam) * prior + lam * empirical)

        if apply_constraint:
            pred = enforce_constraints(pred, seed_features)
        if temperature != 1.0:
            pred = apply_temperature_scaling(pred, temperature)
        # Final floor + renormalise
        pred = np.maximum(pred, config.DEFAULT_PROB_FLOOR)
        pred /= pred.sum(axis=-1, keepdims=True)
        return pred

    # ------------------------------------------------------------------
    # All-seeds convenience wrapper
    # ------------------------------------------------------------------

    def build_all_seeds(
        self,
        all_features: list[dict[str, Any]],
        regime: RegimeEstimate,
        observation_store: ObservationStore,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Return one H×W×6 array per seed."""
        return [
            self.build_prediction(feats, regime, observation_store, i, **kwargs)
            for i, feats in enumerate(all_features)
        ]
