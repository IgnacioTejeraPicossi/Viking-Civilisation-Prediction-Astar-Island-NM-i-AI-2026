"""Sanitize and submit H×W×6 predictions.

Post-processing pipeline (applied in ``submit_all_seeds``):
  1. Temperature scaling   — mild distribution softening (T = 1.04)
  2. Constraint enforcement — port / mountain hard rules
  3. Probability floor      — ensures no infinite-KL classes
  4. Row normalisation      — each cell sums to 1.0
  5. Shape + sum validation

The temperature and constraint steps are also exposed as stand-alone helpers
for use in the parametric predictor and offline evaluation scripts.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np

from astar_island import config


class SupportsSubmit(Protocol):
    def submit(self, round_id: str, seed_index: int, prediction: list[list[list[float]]]) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def apply_temperature_scaling(pred: np.ndarray, T: float = config.TEMPERATURE_SCALING) -> np.ndarray:
    """Temperature-scale an H×W×6 probability tensor.

    Divides log-probabilities by T then re-softmaxes.  T > 1 softens the
    distribution slightly, which improves entropy-weighted KL calibration.
    """
    eps = 1e-12
    log_p = np.log(pred + eps) / T
    log_p -= log_p.max(axis=-1, keepdims=True)
    exp_p = np.exp(log_p)
    return exp_p / exp_p.sum(axis=-1, keepdims=True)


def enforce_constraints(pred: np.ndarray, feats: dict[str, Any]) -> np.ndarray:
    """Enforce hard physical constraints on an H×W×6 prediction tensor.

    Port (class 2)
        Impossible on non-coastal cells (land with no cardinal ocean neighbour).
        Setting it to zero then renormalising redistributes the mass to other
        classes; this avoids penalising the team for impossible terrain states.

    Mountain (class 5)
        Mountains are static — they cannot appear or disappear over 50 years.
        If the initial terrain is not mountain, mountain probability is zeroed.

    Both rules inspired by the top NM i AI 2026 solution (MIT licence).
    """
    pred = pred.copy()
    coast: np.ndarray = feats["coast_mask"].astype(np.float64)       # 1 = coastal land
    mountain: np.ndarray = feats["mountain_mask"].astype(np.float64) # 1 = mountain cell

    # Broadcasting: multiply each cell's class 2 / class 5 by the mask
    pred[:, :, 2] *= coast
    pred[:, :, 5] *= mountain

    # Renormalise with a small floor to prevent zero-sum rows
    pred = np.maximum(pred, 1e-6)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


def sanitize_prediction(
    pred: np.ndarray,
    floor: float = config.DEFAULT_PROB_FLOOR,
    feats: dict[str, Any] | None = None,
    *,
    temperature: float = config.TEMPERATURE_SCALING,
) -> np.ndarray:
    """Apply the full post-processing pipeline and return a clean H×W×6 tensor.

    Steps
    -----
    1. Temperature scaling (if temperature != 1.0)
    2. Constraint enforcement (if ``feats`` is provided)
    3. Floor + normalisation
    """
    pred = np.asarray(pred, dtype=np.float64)
    if temperature != 1.0:
        pred = apply_temperature_scaling(pred, temperature)
    if feats is not None:
        pred = enforce_constraints(pred, feats)
    pred = np.maximum(pred, floor)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


def validate_prediction(pred: np.ndarray, height: int, width: int) -> None:
    """Raise ValueError if the tensor shape or probability sums are invalid."""
    if pred.shape != (height, width, 6):
        raise ValueError(f"Expected {(height, width, 6)}, got {pred.shape}")
    if np.any(pred < 0):
        raise ValueError("Negative probability found")
    s = pred.sum(axis=-1)
    if not np.allclose(s, 1.0, atol=1e-3):
        raise ValueError("Probabilities do not sum to 1.0")


# ---------------------------------------------------------------------------
# Submission loop
# ---------------------------------------------------------------------------

def submit_all_seeds(
    client: SupportsSubmit,
    round_id: str,
    preds: list[np.ndarray],
    floor: float = config.DEFAULT_PROB_FLOOR,
    all_features: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Sanitise and submit one prediction per seed, respecting the rate limit.

    Parameters
    ----------
    client:
        Any object with a ``submit(round_id, seed_index, prediction)`` method.
    round_id:
        Active round UUID.
    preds:
        List of H×W×6 numpy arrays, one per seed.
    floor:
        Minimum per-class probability before submission.
    all_features:
        Optional feature dicts (one per seed) used for constraint enforcement.
        When provided, port and mountain constraints are applied before submit.
    """
    out: list[dict[str, Any]] = []
    for seed_idx, pred in enumerate(preds):
        feats = all_features[seed_idx] if all_features and seed_idx < len(all_features) else None
        pred = sanitize_prediction(pred, floor=floor, feats=feats)
        validate_prediction(pred, pred.shape[0], pred.shape[1])
        resp = client.submit(round_id, seed_idx, pred.tolist())
        out.append(resp)
        if seed_idx < len(preds) - 1:
            time.sleep(config.SUBMIT_MIN_INTERVAL_SEC)
    return out
