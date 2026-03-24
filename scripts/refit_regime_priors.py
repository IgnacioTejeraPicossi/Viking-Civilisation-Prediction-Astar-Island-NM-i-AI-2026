#!/usr/bin/env python3
"""Refit regime-conditional priors from ground-truth /analysis data.

Run backfill first to fetch GT data:
    python scripts/backfill_completed_rounds.py

Then refit:
    python scripts/refit_regime_priors.py
    # or via CLI:
    astar-island refit

What this script does
---------------------
1. Load all ``data/rounds/{round_id}_seed{n}.json`` files.
2. For each round + seed: parse the initial state and GT probability tensor.
3. Compute per-cell CellCategory from Chebyshev distances and coast flags.
4. Accumulate per-category GT probability distributions (distance-binned priors).
5. Compute the oracle regime for each round from the GT tensor.
6. Fit a Ridge regression model (sklearn, α=0.001) mapping regime vectors
   to per-category class probabilities.
7. Build kernel-interpolation and scaled-average data structures.
8. Save everything to ``data/regime_priors.json`` for use by
   ``RegimeConditionalPredictor``.

Inspired by the top NM i AI 2026 Rust solution (MIT licence).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

from astar_island import config
from astar_island.constants import TERRAIN_MOUNTAIN, TERRAIN_OCEAN
from astar_island.features import extract_map_features
from astar_island.gt_priors import (
    CellCategory,
    GTPriorStore,
    N_CLASSES,
    get_cell_category,
    get_default_prior,
)
from astar_island.regime_estimator import RegimeEstimate


# ---------------------------------------------------------------------------
# Oracle regime from GT tensor
# ---------------------------------------------------------------------------

def compute_oracle_regime(
    gt_tensor: np.ndarray,
    initial_grid: list[list[int]],
    initial_settlements: list[dict[str, Any]],
    feats: dict[str, Any],
) -> RegimeEstimate:
    """Estimate the hidden regime directly from ground-truth probabilities.

    Parameters
    ----------
    gt_tensor:
        H × W × 6 ground-truth probability tensor.
    initial_grid:
        H × W integer terrain codes from the initial state.
    initial_settlements:
        List of ``{x, y, has_port, alive}`` dicts.
    feats:
        Pre-computed feature dict from ``extract_map_features``.
    """
    init_pos = {(s["y"], s["x"]) for s in initial_settlements}
    ocean_mask: np.ndarray = feats["ocean_mask"]
    mountain_mask: np.ndarray = feats["mountain_mask"]
    coast_mask: np.ndarray = feats["coast_mask"]
    cheb_dist: np.ndarray = feats["cheb_dist_settlement"]

    is_land = (~ocean_mask) & (~mountain_mask)

    # survival_rate: mean P(Settlement | Port) for initial settlement cells
    surv_probs: list[float] = []
    for s in initial_settlements:
        y, x = s["y"], s["x"]
        surv_probs.append(float(gt_tensor[y, x, 1] + gt_tensor[y, x, 2]))
    survival_rate = float(np.mean(surv_probs)) if surv_probs else 0.60

    # ruin_rate / forest_reclamation from all land cells
    land_gt = gt_tensor[is_land]
    ruin_rate = float(np.mean(land_gt[:, 3])) if len(land_gt) > 0 else 0.05
    forest_reclamation = float(np.mean(land_gt[:, 4])) if len(land_gt) > 0 else 0.08

    # spawn_rate: mean P(Settlement | Port) for non-initial land cells
    non_init_mask = is_land.copy()
    for y_s, x_s in init_pos:
        if 0 <= y_s < non_init_mask.shape[0] and 0 <= x_s < non_init_mask.shape[1]:
            non_init_mask[y_s, x_s] = False
    spawn_gt = gt_tensor[non_init_mask]
    spawn_rate = float(np.mean(spawn_gt[:, 1] + spawn_gt[:, 2])) if len(spawn_gt) > 0 else 0.05

    # spatial_decay: distance-weighted centre of spawn mass
    if non_init_mask.any():
        spawn_prob = (gt_tensor[:, :, 1] + gt_tensor[:, :, 2]) * non_init_mask
        total_mass = float(spawn_prob.sum())
        if total_mass > 1e-6:
            mean_dist = float((spawn_prob * cheb_dist.astype(float)).sum() / total_mass)
        else:
            mean_dist = 8.0
        spatial_decay = float(np.exp(-mean_dist / 5.0))
    else:
        spatial_decay = 0.30

    # coastal_spawn_boost
    coastal_ni = coast_mask & non_init_mask
    inland_ni = (~coast_mask) & non_init_mask & is_land
    if coastal_ni.any() and inland_ni.any():
        c_rate = float(np.mean(gt_tensor[coastal_ni, 1] + gt_tensor[coastal_ni, 2]))
        i_rate = float(np.mean(gt_tensor[inland_ni, 1] + gt_tensor[inland_ni, 2]))
        coastal_spawn_boost = float(np.clip(c_rate / i_rate if i_rate > 1e-6 else 1.5, 0.5, 4.0))
    else:
        coastal_spawn_boost = 1.5

    is_chaotic = spatial_decay > config.CHAOTIC_SPATIAL_DECAY_THRESHOLD

    return RegimeEstimate(
        survival_rate=float(np.clip(survival_rate, 0.05, 0.99)),
        spawn_rate=float(np.clip(spawn_rate, 0.001, 0.50)),
        ruin_rate=float(np.clip(ruin_rate, 0.001, 0.50)),
        forest_reclamation=float(np.clip(forest_reclamation, 0.001, 0.80)),
        spatial_decay=float(np.clip(spatial_decay, 0.01, 0.99)),
        coastal_spawn_boost=coastal_spawn_boost,
        is_chaotic=is_chaotic,
    )


# ---------------------------------------------------------------------------
# Per-category GT prior accumulation
# ---------------------------------------------------------------------------

def accumulate_category_priors(
    gt_tensor: np.ndarray,
    feats: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return mean GT class distribution per CellCategory key for one seed."""
    h, w = gt_tensor.shape[:2]
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    ocean = feats["ocean_mask"]
    mountain = feats["mountain_mask"]

    for y in range(h):
        for x in range(w):
            if bool(ocean[y, x]) or bool(mountain[y, x]):
                continue
            cat = get_cell_category(feats, y, x)
            key = cat.to_key()
            if key not in sums:
                sums[key] = np.zeros(N_CLASSES, dtype=np.float64)
                counts[key] = 0
            sums[key] += gt_tensor[y, x]
            counts[key] += 1

    result: dict[str, np.ndarray] = {}
    for key, arr in sums.items():
        n = counts[key]
        if n > 0:
            p = arr / n
            p = np.maximum(p, 1e-6)
            result[key] = p / p.sum()
    return result


# ---------------------------------------------------------------------------
# Ridge regression fitting
# ---------------------------------------------------------------------------

def fit_parametric_model(
    regime_vectors: list[np.ndarray],
    category_priors: list[dict[str, np.ndarray]],
    alpha: float = 0.001,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Fit one Ridge regression per category that maps regime → class probs.

    Returns
    -------
    cat_keys : list[str]
        Ordered list of category key strings.
    W : np.ndarray
        Shape (n_cats, N_CLASSES, n_regime_features).
    b : np.ndarray
        Shape (n_cats, N_CLASSES).
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        print("[refit] scikit-learn not installed — skipping parametric model")
        return [], np.empty((0,)), np.empty((0,))

    # Collect all category keys present in at least 3 rounds
    key_counts: dict[str, int] = {}
    for cp in category_priors:
        for k in cp:
            key_counts[k] = key_counts.get(k, 0) + 1

    cat_keys = sorted(k for k, c in key_counts.items() if c >= 3)
    n_cats = len(cat_keys)
    n_regime = len(regime_vectors[0]) if regime_vectors else 6
    n_rounds = len(regime_vectors)

    if n_cats == 0 or n_rounds < 3:
        print(f"[refit] not enough data for parametric model (rounds={n_rounds}, cats={n_cats})")
        return [], np.empty((0,)), np.empty((0,))

    X = np.array(regime_vectors, dtype=np.float64)          # (n_rounds, n_regime)
    W = np.zeros((n_cats, N_CLASSES, n_regime), dtype=np.float64)
    b = np.zeros((n_cats, N_CLASSES), dtype=np.float64)

    ridge = Ridge(alpha=alpha, fit_intercept=True)

    for i, key in enumerate(cat_keys):
        # y = target class probabilities for each round (using default where missing)
        default = get_default_prior(CellCategory.from_key(key))
        y = np.array([
            cp.get(key, default) for cp in category_priors
        ], dtype=np.float64)   # (n_rounds, N_CLASSES)

        for cls in range(N_CLASSES):
            ridge.fit(X, y[:, cls])
            W[i, cls, :] = ridge.coef_
            b[i, cls] = ridge.intercept_

    print(f"[refit] fitted parametric model for {n_cats} categories, {n_rounds} rounds")
    return cat_keys, W, b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(exclude_chaotic: bool = False) -> None:
    rounds_dir = _ROOT / "data" / "rounds"
    if not rounds_dir.exists() or not list(rounds_dir.glob("*.json")):
        print(
            "No data in data/rounds/ — run scripts/backfill_completed_rounds.py first",
            file=sys.stderr,
        )
        sys.exit(1)

    files = sorted(rounds_dir.glob("*.json"))
    print(f"[refit] found {len(files)} round files in {rounds_dir}")

    regime_vectors: list[np.ndarray] = []
    category_priors: list[dict[str, np.ndarray]] = []
    kernel_regimes: list[np.ndarray] = []
    kernel_priors: list[dict[str, np.ndarray]] = []
    scaled_acc: dict[str, list[np.ndarray]] = {}
    skipped = 0

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[refit] skip {path.name}: {e}")
            skipped += 1
            continue

        # Parse fields -------------------------------------------------------
        initial_state = data.get("initial_state") or data.get("round", {}).get("initial_state")
        gt_raw = data.get("ground_truth") or data.get("gt_tensor") or data.get("prediction")
        if initial_state is None or gt_raw is None:
            print(f"[refit] skip {path.name}: missing initial_state or gt_tensor")
            skipped += 1
            continue

        grid = initial_state.get("grid")
        settlements_raw = initial_state.get("settlements", [])
        if grid is None:
            print(f"[refit] skip {path.name}: missing grid")
            skipped += 1
            continue

        gt_tensor = np.array(gt_raw, dtype=np.float64)  # H × W × 6
        if gt_tensor.ndim != 3 or gt_tensor.shape[2] != N_CLASSES:
            print(f"[refit] skip {path.name}: unexpected gt shape {gt_tensor.shape}")
            skipped += 1
            continue

        # Normalise settlements field
        settlements: list[dict[str, Any]] = []
        for s in settlements_raw:
            settlements.append({
                "x": int(s.get("x", 0)),
                "y": int(s.get("y", 0)),
                "has_port": bool(s.get("has_port", False)),
                "alive": bool(s.get("alive", True)),
            })

        # Features -----------------------------------------------------------
        try:
            feats = extract_map_features(grid, settlements)
        except Exception as e:
            print(f"[refit] skip {path.name}: feature error: {e}")
            skipped += 1
            continue

        # Oracle regime ------------------------------------------------------
        regime = compute_oracle_regime(gt_tensor, grid, settlements, feats)

        if exclude_chaotic and regime.is_chaotic:
            print(f"[refit] skip {path.name}: chaotic (decay={regime.spatial_decay:.2f})")
            skipped += 1
            # Still include in kernel priors (helps edge-case rounds)
        else:
            regime_vectors.append(regime.as_vector())
            category_priors.append(accumulate_category_priors(gt_tensor, feats))

        # Always add to kernel (chaotic rounds count for similarity lookup)
        kernel_regimes.append(regime.as_vector())
        kernel_priors.append(accumulate_category_priors(gt_tensor, feats))

        # Accumulate scaled avg
        cat_p = accumulate_category_priors(gt_tensor, feats)
        for key, p in cat_p.items():
            scaled_acc.setdefault(key, []).append(p)

    n_rounds = len(regime_vectors)
    print(f"[refit] usable rounds: {n_rounds}  skipped: {skipped}")

    if n_rounds == 0:
        print("[refit] no usable rounds — aborting", file=sys.stderr)
        sys.exit(1)

    # Scaled average ---------------------------------------------------------
    scaled_avg: dict[str, np.ndarray] = {}
    for key, arr_list in scaled_acc.items():
        avg = np.mean(arr_list, axis=0)
        avg = np.maximum(avg, 1e-6)
        scaled_avg[key] = avg / avg.sum()

    # Parametric model -------------------------------------------------------
    cat_keys, W, b = fit_parametric_model(regime_vectors, category_priors)
    has_parametric = len(cat_keys) > 0

    # Build GTPriorStore and save --------------------------------------------
    store = GTPriorStore()
    store._has_parametric = has_parametric
    store._cat_keys = cat_keys
    if has_parametric:
        store._param_W = W
        store._param_b = b
    store._kernel_regimes = kernel_regimes
    store._kernel_priors = kernel_priors
    store._scaled_avg = scaled_avg

    out_path = config.get_regime_priors_path()
    store.save(out_path)
    print(f"[refit] saved → {out_path}  (parametric={has_parametric}, kernel_rounds={len(kernel_regimes)})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Refit regime priors from GT data")
    ap.add_argument("--no-chaotic", action="store_true", help="Exclude chaotic rounds from training")
    ns = ap.parse_args()
    main(exclude_chaotic=ns.no_chaotic)
