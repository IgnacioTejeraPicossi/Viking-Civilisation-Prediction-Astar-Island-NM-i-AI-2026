"""Adaptive viewport placement using Shannon entropy of current predictions.

Two planning modes are provided:

Static (``initial_query_plan``)
    Score windows upfront from map features alone (anchor + frontier + coastal).
    Used as a cold-start before any observations are available and kept for
    backward-compatibility.

Adaptive (``adaptive_pick_next_window``)
    After each simulate call the caller re-invokes this function with the
    latest per-seed predictions.  Windows are scored by the mean Shannon
    entropy of the current predictions in that window — high entropy means
    the model is uncertain there, so observing it gives the most information.
    This is the strategy used by the top NM i AI 2026 solution (MIT licence).

Typical adaptive runner loop
-----------------------------
    preds = predictor.build_all_seeds(...)   # prior-only initial predictions
    for _ in range(budget):
        plan = adaptive_pick_next_window(preds, all_features, width, height)
        sim  = client.simulate(rid, plan.seed_index, plan.x, plan.y, plan.w, plan.h)
        store.add_sim_result(plan.seed_index, sim, TERRAIN_TO_CLASS)
        preds[plan.seed_index] = predictor.build_prediction(
            all_features[plan.seed_index], regime, store, plan.seed_index
        )
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from astar_island import config

# ---------------------------------------------------------------------------
# Window plan
# ---------------------------------------------------------------------------

@dataclass
class WindowPlan:
    seed_index: int
    x: int
    y: int
    w: int
    h: int
    tag: str
    repeats: int


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def clamp_window(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int]:
    x = max(0, min(x, width - w))
    y = max(0, min(y, height - h))
    return x, y


def total_planned_queries(plans: list[WindowPlan]) -> int:
    return sum(p.repeats for p in plans)


# ---------------------------------------------------------------------------
# Entropy-based adaptive scoring (primary strategy)
# ---------------------------------------------------------------------------

def entropy_score_window(
    pred: np.ndarray,
    x: int,
    y: int,
    w: int = 15,
    h: int = 15,
) -> float:
    """Mean Shannon entropy of current predictions inside the viewport.

    Higher entropy → the model is more uncertain → the window carries more
    expected information gain.

    Parameters
    ----------
    pred : np.ndarray
        H × W × 6 prediction tensor (probabilities, must sum to 1 per cell).
    x, y : int
        Top-left corner of the candidate viewport.
    w, h : int
        Viewport dimensions.
    """
    eps = 1e-12
    sl: np.ndarray = pred[y : y + h, x : x + w]   # (h_clip, w_clip, 6)
    ent: np.ndarray = -(sl * np.log(sl + eps)).sum(axis=-1)  # (h_clip, w_clip)
    return float(ent.mean())


def adaptive_pick_next_window(
    preds: list[np.ndarray],
    all_features: list[dict],
    width: int,
    height: int,
    stride: int = 5,
    vp_w: int = config.DEFAULT_VIEWPORT_W,
    vp_h: int = config.DEFAULT_VIEWPORT_H,
) -> WindowPlan:
    """Pick the single viewport with the highest mean entropy across all seeds.

    If no predictions are available for a seed (empty list slot), that seed is
    skipped.  The function always returns a valid ``WindowPlan``.

    Parameters
    ----------
    preds:
        Current per-seed prediction tensors (H × W × 6).  May contain
        ``None`` entries for seeds not yet initialised.
    all_features:
        Feature dicts from ``extract_map_features``, one per seed.
    width, height:
        Full map dimensions.
    stride:
        Grid stride when enumerating candidate window origins.
    vp_w, vp_h:
        Viewport dimensions (default 15 × 15).
    """
    best_score = -1.0
    best_plan = WindowPlan(0, 0, 0, vp_w, vp_h, "adaptive", 1)  # safe default

    for seed_idx, pred in enumerate(preds):
        if pred is None:
            continue
        for ry in range(0, max(1, height - vp_h + 1), stride):
            for rx in range(0, max(1, width - vp_w + 1), stride):
                cx, cy = clamp_window(rx, ry, vp_w, vp_h, width, height)
                score = entropy_score_window(pred, cx, cy, vp_w, vp_h)
                if score > best_score:
                    best_score = score
                    best_plan = WindowPlan(seed_idx, cx, cy, vp_w, vp_h, "adaptive", 1)

    return best_plan


# ---------------------------------------------------------------------------
# Static feature-based window scoring (used for initial cold-start plan)
# ---------------------------------------------------------------------------

def score_window(feats: dict, x: int, y: int, w: int = 15, h: int = 15) -> float:
    sl = np.s_[y : y + h, x : x + w]
    s_density = feats["settlement_mask"][sl].mean()
    p_density = feats["port_mask"][sl].mean()
    coast = feats["coast_mask"][sl].mean()
    frontier = feats["frontier_score"][sl].mean()
    forest = feats["forest_mask"][sl].mean()
    mountain = feats["mountain_mask"][sl].mean()
    return (
        3.0 * s_density
        + 2.0 * p_density
        + 1.8 * coast
        + 2.5 * frontier
        + 0.7 * forest
        - 2.2 * mountain
    )


def score_coast_window(feats: dict, x: int, y: int, w: int = 15, h: int = 15) -> float:
    """Emphasise coastal / trade-relevant tiles."""
    sl = np.s_[y : y + h, x : x + w]
    coast = feats["coast_mask"][sl].mean()
    s_density = feats["settlement_mask"][sl].mean()
    p_density = feats["port_mask"][sl].mean()
    mountain = feats["mountain_mask"][sl].mean()
    return 2.5 * coast + 1.5 * s_density + 2.0 * p_density - 1.5 * mountain


def best_windows_for_seed(
    feats: dict, width: int, height: int, stride: int = 5
) -> list[tuple[float, int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for y in range(0, max(1, height - 14), stride):
        for x in range(0, max(1, width - 14), stride):
            candidates.append((score_window(feats, x, y), x, y))
    return sorted(candidates, reverse=True)


def best_coastal_windows_for_seed(
    feats: dict, width: int, height: int, stride: int = 5
) -> list[tuple[float, int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for y in range(0, max(1, height - 14), stride):
        for x in range(0, max(1, width - 14), stride):
            candidates.append((score_coast_window(feats, x, y), x, y))
    return sorted(candidates, reverse=True)


def _pick_window(
    ranked: list[tuple[float, int, int]],
    avoid: set[tuple[int, int]],
    min_index: int = 0,
) -> tuple[int, int] | None:
    for i, (_, x, y) in enumerate(ranked):
        if i < min_index:
            continue
        if (x, y) not in avoid:
            return x, y
    for _, x, y in ranked:
        if (x, y) not in avoid:
            return x, y
    return None


# ---------------------------------------------------------------------------
# Static upfront planner (backward-compatible; used for cold-start only)
# ---------------------------------------------------------------------------

def initial_query_plan(
    all_seed_features: list[dict[str, np.ndarray]],
    width: int,
    height: int,
    max_queries: int | None = None,
) -> list[WindowPlan]:
    """Build a static anchor + frontier + coastal window plan for all seeds.

    This plan is used only as a *cold start* before the adaptive runner takes
    over.  When ``max_queries`` is small (≤ n_seeds * 3) the plan is trimmed.
    """
    max_q = max_queries if max_queries is not None else config.MAX_QUERIES_PER_ROUND
    plans: list[WindowPlan] = []

    for seed_idx, feats in enumerate(all_seed_features):
        ranked = best_windows_for_seed(feats, width, height)
        coastal_ranked = best_coastal_windows_for_seed(feats, width, height)
        used: set[tuple[int, int]] = set()

        if ranked:
            _, x1, y1 = ranked[0]
            x1, y1 = clamp_window(x1, y1, 15, 15, width, height)
            plans.append(WindowPlan(seed_idx, x1, y1, 15, 15, "anchor", 3))
            used.add((x1, y1))

        if len(ranked) > 1:
            pick = _pick_window(ranked, used, min_index=1)
            if pick:
                x2, y2 = clamp_window(pick[0], pick[1], 15, 15, width, height)
                plans.append(WindowPlan(seed_idx, x2, y2, 15, 15, "frontier", 3))
                used.add((x2, y2))

        if coastal_ranked:
            pick_c = _pick_window(coastal_ranked, used, min_index=0) or (coastal_ranked[0][1], coastal_ranked[0][2])
            xc, yc = clamp_window(pick_c[0], pick_c[1], 15, 15, width, height)
            plans.append(WindowPlan(seed_idx, xc, yc, 15, 15, "coastal", 4))
            used.add((xc, yc))

    # Top up with extra repeats on anchor windows until budget is reached
    remaining = max_q - total_planned_queries(plans)
    seed_anchor: dict[int, WindowPlan] = {p.seed_index: p for p in plans if p.tag == "anchor"}
    step = 0
    while remaining > 0 and seed_anchor:
        keys = sorted(seed_anchor.keys())
        seed_anchor[keys[step % len(keys)]].repeats += 1
        remaining -= 1
        step += 1

    # Fill remaining budget with extra ranked windows
    remaining = max_q - total_planned_queries(plans)
    extra_idx = 0
    while remaining > 0 and extra_idx < 50:
        for seed_idx, feats in enumerate(all_seed_features):
            if remaining <= 0:
                break
            ranked = best_windows_for_seed(feats, width, height)
            pos = 2 + (extra_idx // len(all_seed_features))
            if pos < len(ranked):
                _, xe, ye = ranked[pos]
                xe, ye = clamp_window(xe, ye, 15, 15, width, height)
                plans.append(WindowPlan(seed_idx, xe, ye, 15, 15, "extra", 1))
                remaining -= 1
        extra_idx += 1

    # Hard cap: trim to max_q
    drop_priority = {"extra": 0, "coastal": 1, "frontier": 2, "anchor": 3}
    while total_planned_queries(plans) > max_q:
        changed = False
        for p in reversed(plans):
            if p.repeats > 1 and p.tag != "anchor":
                p.repeats -= 1
                changed = True
                break
        if not changed:
            for p in reversed(plans):
                if p.repeats > 1:
                    p.repeats -= 1
                    changed = True
                    break
        if not changed:
            if not plans:
                break
            idx = min(range(len(plans)), key=lambda i: drop_priority.get(plans[i].tag, 99))
            plans.pop(idx)

    return plans
