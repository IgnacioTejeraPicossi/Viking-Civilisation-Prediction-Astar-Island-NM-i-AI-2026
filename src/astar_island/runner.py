"""End-to-end adaptive round: features → simulate (entropy-driven) → regime → predict → submit.

Pipeline overview
-----------------
1. Fetch active round and map features.
2. Check budget; abort if plan would over-spend.
3. Build prior-only predictions (initial entropy map for planner).
4. Adaptive simulation loop:
   a. Pick next viewport = window with highest mean Shannon entropy.
   b. POST /simulate → update ObservationStore.
   c. Update prediction for the affected seed.
   d. Every REGIME_UPDATE_INTERVAL queries: re-estimate regime (Bayesian shrinkage).
5. Final full regime estimation on all observations.
6. Final prediction pass (all seeds) with post-processing.
7. Optional POST /submit for all seeds.
8. Optional JSON log.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astar_island import config
from astar_island.api_client import AstarClient
from astar_island.constants import TERRAIN_TO_CLASS
from astar_island.features import extract_map_features
from astar_island.observation_store import ObservationStore
from astar_island.predictor_parametric import RegimeConditionalPredictor
from astar_island.query_planner import (
    WindowPlan,
    adaptive_pick_next_window,
    total_planned_queries,
)
from astar_island.regime_estimator import BayesianRegimeEstimator, RegimeEstimate
from astar_island.run_log import (
    prediction_log_summary,
    serialize_plans,
    store_summary,
    write_run_log_file,
)
from astar_island.submission import submit_all_seeds

# Re-estimate regime every this many simulate calls
REGIME_UPDATE_INTERVAL = 5


def _err(msg: str) -> None:
    print(f"[astar-island] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_active_round(
    client: AstarClient | None = None,
    *,
    submit: bool = True,
    max_queries: int | None = None,
    round_id: str | None = None,
    write_log: bool = True,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Fetch the active round, run adaptive simulates, predict, optionally submit.

    Returns a summary dict (round_id, regime, queries_run, submissions, …).
    """
    started_at = datetime.now(timezone.utc).isoformat()
    _err("creating API client…")
    cli = client or AstarClient()

    # ------------------------------------------------------------------
    # Resolve round
    # ------------------------------------------------------------------
    if round_id is None:
        _err("GET /rounds (looking for active round)…")
        active = cli.get_active_round()
    else:
        active = {"id": round_id}

    if not active:
        raise RuntimeError("No active round (and no round_id provided)")

    rid = active["id"]
    _err(f"GET /rounds/{{id}} for round {rid}…")
    detail = cli.get_round_detail(rid)

    # ------------------------------------------------------------------
    # Budget check
    # ------------------------------------------------------------------
    budget_before: dict[str, Any] | None = None
    try:
        _err("GET /budget …")
        budget_before = cli.get_budget().model_dump()
    except Exception as e:  # noqa: BLE001
        budget_before = {"error": str(e)}

    budget_left: int = max_queries or config.MAX_QUERIES_PER_ROUND
    if isinstance(budget_before, dict) and "error" not in budget_before:
        qu_used = int(budget_before.get("queries_used", 0))
        qu_max = int(budget_before.get("queries_max", 50))
        available = qu_max - qu_used
        if budget_left > available:
            if max_queries is not None:
                raise RuntimeError(
                    f"Query budget insufficient: requested {max_queries} but only "
                    f"{available} left ({qu_used}/{qu_max} used)."
                )
            budget_left = available
            _err(f"budget_left adjusted to {budget_left} (used {qu_used}/{qu_max})")

    if budget_left <= 0:
        raise RuntimeError(
            "No query budget remaining. "
            "Check `astar-island budget` or wait for a new round."
        )

    # ------------------------------------------------------------------
    # Build map features for all seeds
    # ------------------------------------------------------------------
    _err("building map features…")
    all_features: list[dict] = []
    for i, state in enumerate(detail.initial_states):
        _err(f"  seed {i + 1}/{detail.seeds_count} …")
        settlements = [s.model_dump() for s in state.settlements]
        feats = extract_map_features(state.grid, settlements)
        all_features.append(feats)

    # ------------------------------------------------------------------
    # Initialise observation store + predictor
    # ------------------------------------------------------------------
    store = ObservationStore(detail.map_width, detail.map_height, detail.seeds_count)
    predictor = RegimeConditionalPredictor()

    # Default regime (used until enough observations accumulate)
    regime = RegimeEstimate()

    # ------------------------------------------------------------------
    # Prior-only initial predictions (seed the entropy map)
    # ------------------------------------------------------------------
    _err("building initial prior predictions for entropy planner…")
    preds: list[Any] = predictor.build_all_seeds(all_features, regime, store)

    # ------------------------------------------------------------------
    # Adaptive simulation loop
    # ------------------------------------------------------------------
    queries_run = 0
    executed_plans: list[WindowPlan] = []

    _err(f"starting adaptive simulation loop ({budget_left} queries)…")
    for q_idx in range(budget_left):
        # Pick the window with the highest mean prediction entropy
        plan = adaptive_pick_next_window(
            preds,
            all_features,
            detail.map_width,
            detail.map_height,
        )
        _err(
            f"  simulate {q_idx + 1}/{budget_left} "
            f"seed={plan.seed_index} @{plan.x},{plan.y} (entropy)"
        )
        sim = cli.simulate(rid, plan.seed_index, plan.x, plan.y, plan.w, plan.h)
        store.add_sim_result(plan.seed_index, sim, TERRAIN_TO_CLASS)
        executed_plans.append(plan)
        queries_run += 1

        # Update prediction for the affected seed immediately
        preds[plan.seed_index] = predictor.build_prediction(
            all_features[plan.seed_index], regime, store, plan.seed_index
        )

        # Periodically re-estimate regime with accumulated observations
        if queries_run % REGIME_UPDATE_INTERVAL == 0:
            regime = BayesianRegimeEstimator().estimate(
                store, all_features, detail.initial_states
            )
            _err(
                f"  regime update @ q={queries_run}: "
                f"surv={regime.survival_rate:.2f} "
                f"ruin={regime.ruin_rate:.3f} "
                f"decay={regime.spatial_decay:.2f}"
                + (" [CHAOTIC]" if regime.is_chaotic else "")
            )
            # Rebuild all predictions with updated regime
            preds = predictor.build_all_seeds(all_features, regime, store)

        if queries_run < budget_left:
            time.sleep(config.SIMULATE_MIN_INTERVAL_SEC)

    # ------------------------------------------------------------------
    # Final regime + prediction
    # ------------------------------------------------------------------
    _err("final regime estimation…")
    regime = BayesianRegimeEstimator().estimate(
        store, all_features, detail.initial_states
    )
    _err(
        f"final regime: surv={regime.survival_rate:.2f} "
        f"spawn={regime.spawn_rate:.3f} "
        f"ruin={regime.ruin_rate:.3f} "
        f"forest={regime.forest_reclamation:.3f} "
        f"decay={regime.spatial_decay:.2f} "
        f"coast_boost={regime.coastal_spawn_boost:.2f}"
        + (" [CHAOTIC]" if regime.is_chaotic else "")
    )

    _err("building final predictions…")
    preds = predictor.build_all_seeds(all_features, regime, store)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------
    submissions: list[dict[str, Any]] | None = None
    if submit:
        _err("submitting predictions…")
        submissions = submit_all_seeds(cli, rid, preds, all_features=all_features)

    # ------------------------------------------------------------------
    # Budget after
    # ------------------------------------------------------------------
    budget_after: dict[str, Any] | None = None
    try:
        budget_after = cli.get_budget().model_dump()
    except Exception as e:  # noqa: BLE001
        budget_after = {"error": str(e)}

    finished_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Optional run log
    # ------------------------------------------------------------------
    log_path: str | None = None
    if write_log:
        from astar_island import __version__
        _err("writing run log…")
        payload: dict[str, Any] = {
            "meta": {
                "package_version": __version__,
                "started_at": started_at,
                "finished_at": finished_at,
                "submit": submit,
                "max_queries": max_queries,
                "adaptive": True,
            },
            "round": {
                "round_id": rid,
                "round_number": detail.round_number,
                "map_width": detail.map_width,
                "map_height": detail.map_height,
                "seeds_count": detail.seeds_count,
            },
            "budget_before": budget_before,
            "budget_after": budget_after,
            "query_plan": {
                "planned_total": budget_left,
                "queries_run": queries_run,
                "windows": serialize_plans(executed_plans),
            },
            "regime": regime.as_dict(),
            "observation_store": store_summary(store),
            "prediction_summary": prediction_log_summary(preds),
            "submissions": submissions,
        }
        path = write_run_log_file(payload, log_dir=log_dir)
        log_path = str(path)
        _err(f"log written → {log_path}")

    return {
        "round_id": rid,
        "round_number": detail.round_number,
        "map_width": detail.map_width,
        "map_height": detail.map_height,
        "seeds_count": detail.seeds_count,
        "queries_run": queries_run,
        "regime": regime.as_dict(),
        "submissions": submissions,
        "predictions": preds,
        "log_path": log_path,
    }


def solve_active_round(token: str) -> dict[str, Any]:
    """Thin entrypoint for FastAPI: token string → full run with submit."""
    return run_active_round(AstarClient(token=token), submit=True)
