#!/usr/bin/env python3
"""Leave-one-out cross-validation over completed rounds.

Measures how well the regime-conditional predictor performs when trained on all
rounds except the one being evaluated.  Uses the same entropy-weighted KL
metric as the competition.

Usage
-----
    python scripts/evaluate_loo.py
    # or via CLI:
    astar-island eval

Prerequisites
-------------
    python scripts/backfill_completed_rounds.py   # fetch GT tensors
    python scripts/refit_regime_priors.py          # fit priors (optional but recommended)

Output
------
    Per-round scores and average score printed to stdout.
    Also writes ``data/loo_results.json``.

Scoring formula (competition)
------------------------------
    entropy_weight(cell) = -sum_k gt[k] * log(gt[k])
    kl(cell)             = sum_k gt[k] * log(gt[k] / pred[k])
    weighted_kl          = sum(entropy_weight * kl) / sum(entropy_weight)
    score                = 100 * exp(-3 * weighted_kl)

Only dynamic cells (not pure ocean / mountain) contribute.
Inspired by the top NM i AI 2026 solution (MIT licence).
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
from astar_island.gt_priors import GTPriorStore, N_CLASSES, accumulate_category_priors_fn
from astar_island.observation_store import ObservationStore
from astar_island.predictor_parametric import RegimeConditionalPredictor
from astar_island.regime_estimator import RegimeEstimate
from astar_island.submission import enforce_constraints, apply_temperature_scaling

# We import accumulate_category_priors from the refit script to avoid duplication
from refit_regime_priors import (
    accumulate_category_priors,
    compute_oracle_regime,
    fit_parametric_model,
)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def entropy_weighted_kl(
    gt: np.ndarray,
    pred: np.ndarray,
    feats: dict[str, Any],
    eps: float = 1e-12,
) -> float:
    """Compute entropy-weighted KL divergence, ignoring ocean/mountain cells."""
    ocean = feats["ocean_mask"]
    mountain = feats["mountain_mask"]
    dynamic = (~ocean) & (~mountain)   # (H, W)

    gt_d = gt[dynamic]     # (N, 6)
    pred_d = pred[dynamic] # (N, 6)

    # Entropy weights
    w = -(gt_d * np.log(gt_d + eps)).sum(axis=-1)   # (N,)
    # Per-cell KL(gt || pred)
    kl = (gt_d * np.log((gt_d + eps) / (pred_d + eps))).sum(axis=-1)  # (N,)

    w_sum = float(w.sum())
    if w_sum < 1e-10:
        return 0.0
    return float((w * kl).sum() / w_sum)


def score_from_wkl(weighted_kl: float) -> float:
    return 100.0 * float(np.exp(-3.0 * weighted_kl))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_round_file(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON or None on error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[eval] skip {path.name}: {e}")
        return None


def parse_round_data(
    data: dict[str, Any],
) -> tuple[list[list[int]], list[dict], np.ndarray] | None:
    """Extract (grid, settlements, gt_tensor) from a round JSON file."""
    initial_state = data.get("initial_state") or data.get("round", {}).get("initial_state")
    gt_raw = data.get("ground_truth") or data.get("gt_tensor") or data.get("prediction")
    if initial_state is None or gt_raw is None:
        return None

    grid = initial_state.get("grid")
    settlements_raw = initial_state.get("settlements", [])
    if grid is None:
        return None

    gt_tensor = np.array(gt_raw, dtype=np.float64)
    if gt_tensor.ndim != 3 or gt_tensor.shape[2] != N_CLASSES:
        return None

    settlements = [
        {
            "x": int(s.get("x", 0)),
            "y": int(s.get("y", 0)),
            "has_port": bool(s.get("has_port", False)),
            "alive": bool(s.get("alive", True)),
        }
        for s in settlements_raw
    ]
    return grid, settlements, gt_tensor


# ---------------------------------------------------------------------------
# LOO fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(
    test_grid: list[list[int]],
    test_settlements: list[dict],
    test_gt: np.ndarray,
    train_regime_vectors: list[np.ndarray],
    train_category_priors: list[dict[str, np.ndarray]],
    train_scaled_avg: dict[str, np.ndarray],
    exclude_chaotic: bool = False,
) -> float:
    """Evaluate one LOO fold.  Returns the entropy-weighted KL score (0-100)."""
    # Build features for the test seed
    feats = extract_map_features(test_grid, test_settlements)

    # Compute oracle regime for the test round (used to query the model)
    test_regime = compute_oracle_regime(test_gt, test_grid, test_settlements, feats)

    # Build a GTPriorStore fitted on training rounds only
    cat_keys, W, b = fit_parametric_model(train_regime_vectors, train_category_priors)
    has_param = len(cat_keys) > 0

    store = GTPriorStore()
    store._has_parametric = has_param
    store._cat_keys = cat_keys
    if has_param:
        store._param_W = W
        store._param_b = b
    store._kernel_regimes = list(train_regime_vectors)
    store._kernel_priors = list(train_category_priors)
    store._scaled_avg = train_scaled_avg

    # Predict (prior-only — no API observations in LOO)
    predictor = RegimeConditionalPredictor.__new__(RegimeConditionalPredictor)
    predictor._store = store

    obs_store = ObservationStore(
        width=test_gt.shape[1],
        height=test_gt.shape[0],
        seeds_count=1,
    )
    pred = predictor.build_prediction(feats, test_regime, obs_store, 0)

    kl = entropy_weighted_kl(test_gt, pred, feats)
    return score_from_wkl(kl)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(rounds_dir: str | None = None) -> None:
    rdir = Path(rounds_dir) if rounds_dir else _ROOT / "data" / "rounds"
    if not rdir.exists():
        print(f"[eval] rounds directory not found: {rdir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(rdir.glob("*.json"))
    if not files:
        print(
            "[eval] no round files found — run scripts/backfill_completed_rounds.py first",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[eval] loading {len(files)} round files…")

    # Load all rounds
    rounds: list[tuple[str, list[list[int]], list[dict], np.ndarray]] = []
    for path in files:
        data = load_round_file(path)
        if data is None:
            continue
        parsed = parse_round_data(data)
        if parsed is None:
            print(f"[eval] skip {path.name}: could not parse")
            continue
        grid, settlements, gt = parsed
        rounds.append((path.stem, grid, settlements, gt))

    if len(rounds) < 2:
        print(f"[eval] need at least 2 rounds for LOO (have {len(rounds)})", file=sys.stderr)
        sys.exit(1)

    print(f"[eval] running LOO-CV over {len(rounds)} rounds…\n")

    # Precompute features and oracle regimes
    all_feats: list[dict] = []
    all_regimes: list[RegimeEstimate] = []
    all_cat_priors: list[dict[str, np.ndarray]] = []

    for name, grid, settlements, gt in rounds:
        feats = extract_map_features(grid, settlements)
        regime = compute_oracle_regime(gt, grid, settlements, feats)
        cat_p = accumulate_category_priors(gt, feats)
        all_feats.append(feats)
        all_regimes.append(regime)
        all_cat_priors.append(cat_p)

    # LOO loop
    results: list[dict[str, Any]] = []
    scores: list[float] = []

    for i, (name, grid, settlements, gt) in enumerate(rounds):
        # Training set = all rounds except i
        train_indices = [j for j in range(len(rounds)) if j != i]
        train_regime_vectors = [all_regimes[j].as_vector() for j in train_indices]
        train_cat_priors = [all_cat_priors[j] for j in train_indices]

        # Scaled average from training rounds
        scaled_acc: dict[str, list[np.ndarray]] = {}
        for j in train_indices:
            for key, p in all_cat_priors[j].items():
                scaled_acc.setdefault(key, []).append(p)
        scaled_avg = {
            k: (lambda a: a / a.sum())(np.maximum(np.mean(v, axis=0), 1e-6))
            for k, v in scaled_acc.items()
        }

        score = evaluate_fold(
            grid,
            settlements,
            gt,
            train_regime_vectors,
            train_cat_priors,
            scaled_avg,
        )
        scores.append(score)
        chaotic_flag = " [chaotic]" if all_regimes[i].is_chaotic else ""
        print(f"  [{i+1:>2}/{len(rounds)}] {name:<40}  score={score:.2f}{chaotic_flag}")
        results.append({"name": name, "score": round(score, 4), "chaotic": all_regimes[i].is_chaotic})

    avg_score = float(np.mean(scores))
    print(f"\n{'='*60}")
    print(f"  LOO-CV average score : {avg_score:.2f}  (n={len(scores)})")
    non_chaotic = [s for s, r in zip(scores, results) if not r["chaotic"]]
    if non_chaotic:
        print(f"  Non-chaotic average  : {np.mean(non_chaotic):.2f}  (n={len(non_chaotic)})")
    print(f"{'='*60}\n")

    # Save results
    out_path = _ROOT / "data" / "loo_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"avg_score": avg_score, "rounds": results}, indent=2),
        encoding="utf-8",
    )
    print(f"[eval] results saved → {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LOO cross-validation")
    ap.add_argument("--rounds-dir", type=str, default=None, help="Path to rounds directory")
    ns = ap.parse_args()
    main(rounds_dir=ns.rounds_dir)
