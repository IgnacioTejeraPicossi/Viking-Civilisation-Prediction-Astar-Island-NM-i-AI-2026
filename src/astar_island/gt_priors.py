"""Distance-binned, coastal-aware GT probability priors.

Inspired by the top-scoring NM i AI 2026 solution (MIT).

Cells are categorised by three static spatial features:
  - dist_bin         : Chebyshev distance to the nearest initial settlement (8 bins)
  - is_coastal       : land cell with at least one cardinal-direction ocean neighbour
  - near_coastal_settlement : Chebyshev distance ≤ 4 to a coastal initial settlement

This gives up to 8 × 2 × 2 = 32 categories.  Per-category class distributions
are either loaded from a fitted JSON file (after ``refit_regime_priors.py`` has
run) or fall back to hard-coded domain-knowledge defaults that are still much
better than the old flat terrain-type priors.

Classes (index order for the 6-probability prediction tensor):
  0 Empty  1 Settlement  2 Port  3 Ruin  4 Forest  5 Mountain
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from astar_island.regime_estimator import RegimeEstimate

# ---------------------------------------------------------------------------
# Distance bins
# ---------------------------------------------------------------------------

# Upper boundary (inclusive) for each of the 8 Chebyshev distance bins.
# Bin 0: d=0  (the cell IS an initial settlement)
# Bin 1: d=1  Bin 2: d=2  Bin 3: d=3
# Bin 4: d=4-5  Bin 5: d=6-8  Bin 6: d=9-12  Bin 7: d=13+
DIST_BIN_UPPER: list[int] = [0, 1, 2, 3, 5, 8, 12, 9_999]
N_DIST_BINS: int = len(DIST_BIN_UPPER)  # 8
N_CLASSES: int = 6

# How close to a coastal settlement a cell must be to be flagged "near_cs"
NEAR_CS_THRESHOLD: int = 4


def dist_to_bin(d: int) -> int:
    """Map a Chebyshev distance value to its bin index (0-7)."""
    for i, upper in enumerate(DIST_BIN_UPPER):
        if d <= upper:
            return i
    return N_DIST_BINS - 1


# ---------------------------------------------------------------------------
# Cell category
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellCategory:
    """Immutable, hashable spatial category for a single map cell."""

    dist_bin: int            # 0..7
    is_coastal: bool         # land adjacent to ocean (cardinal)
    near_coastal_settlement: bool  # Chebyshev dist ≤ NEAR_CS_THRESHOLD to coastal settlement

    def to_key(self) -> str:
        return f"{self.dist_bin}_{int(self.is_coastal)}_{int(self.near_coastal_settlement)}"

    @staticmethod
    def from_key(key: str) -> "CellCategory":
        parts = key.split("_")
        return CellCategory(int(parts[0]), bool(int(parts[1])), bool(int(parts[2])))


def get_cell_category(feats: dict[str, Any], y: int, x: int) -> CellCategory:
    """Return the CellCategory for cell (y, x) given pre-computed feature maps."""
    d = int(feats["cheb_dist_settlement"][y, x])
    dist_bin = dist_to_bin(d)
    is_coastal = bool(feats["coast_mask"][y, x])
    near_cs = int(feats["cheb_dist_coast_settlement"][y, x]) <= NEAR_CS_THRESHOLD
    return CellCategory(dist_bin, is_coastal, near_cs)


# ---------------------------------------------------------------------------
# Default prior table
# ---------------------------------------------------------------------------
# Layout per entry: [Empty, Settlement, Port, Ruin, Forest, Mountain]
# Keyed by (dist_bin, is_coastal).  The near_coastal_settlement flag applies
# a multiplicative boost to Port probability on top of these defaults.
# These values encode domain knowledge and are refined by refit_regime_priors.py.

_DEFAULT_PRIOR_TABLE: dict[tuple[int, bool], list[float]] = {
    # --- dist_bin=0: the cell WAS itself an initial settlement ---
    (0, False): [0.04, 0.54, 0.04, 0.24, 0.09, 0.05],
    (0, True):  [0.04, 0.30, 0.34, 0.18, 0.07, 0.07],
    # --- dist_bin=1: 1 Chebyshev step from a settlement ---
    (1, False): [0.30, 0.28, 0.02, 0.17, 0.17, 0.06],
    (1, True):  [0.20, 0.18, 0.26, 0.14, 0.13, 0.09],
    # --- dist_bin=2 ---
    (2, False): [0.47, 0.18, 0.01, 0.14, 0.15, 0.05],
    (2, True):  [0.32, 0.12, 0.22, 0.12, 0.14, 0.08],
    # --- dist_bin=3 ---
    (3, False): [0.57, 0.12, 0.01, 0.11, 0.14, 0.05],
    (3, True):  [0.42, 0.09, 0.19, 0.10, 0.13, 0.07],
    # --- dist_bin=4: d=4-5 ---
    (4, False): [0.65, 0.08, 0.01, 0.09, 0.13, 0.04],
    (4, True):  [0.50, 0.06, 0.17, 0.08, 0.13, 0.06],
    # --- dist_bin=5: d=6-8 ---
    (5, False): [0.71, 0.05, 0.01, 0.07, 0.12, 0.04],
    (5, True):  [0.56, 0.04, 0.14, 0.07, 0.13, 0.06],
    # --- dist_bin=6: d=9-12 ---
    (6, False): [0.76, 0.03, 0.01, 0.05, 0.12, 0.03],
    (6, True):  [0.61, 0.03, 0.12, 0.05, 0.13, 0.06],
    # --- dist_bin=7: d=13+, far from any settlement ---
    (7, False): [0.80, 0.02, 0.01, 0.04, 0.10, 0.03],
    (7, True):  [0.65, 0.02, 0.09, 0.04, 0.12, 0.08],
}

_FALLBACK_PRIOR: list[float] = [0.70, 0.08, 0.05, 0.07, 0.07, 0.03]


def _normalize(v: np.ndarray | list[float], floor: float = 1e-6) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    a = np.maximum(a, floor)
    return a / a.sum()


def get_default_prior(cat: CellCategory) -> np.ndarray:
    """Return the normalized default prior for *cat* from the hard-coded table."""
    raw = _DEFAULT_PRIOR_TABLE.get((cat.dist_bin, cat.is_coastal), _FALLBACK_PRIOR)
    p = _normalize(raw)
    # Extra port boost for coastal cells near a coastal settlement
    if cat.near_coastal_settlement and cat.is_coastal:
        p[2] = min(p[2] * 1.5, 0.60)  # class 2 = Port
        p = _normalize(p)
    return p


# ---------------------------------------------------------------------------
# GTPriorStore
# ---------------------------------------------------------------------------

class GTPriorStore:
    """Per-category prior store with regime scaling and optional fitted model.

    Without ``data/regime_priors.json``:
        Returns ``get_default_prior(cat)`` scaled by the current ``RegimeEstimate``.

    With fitted priors (after ``scripts/refit_regime_priors.py``):
        Blends three components:
          80 % parametric Ridge model (regime_vector → class_probs per category)
          15 % kernel interpolation (Gaussian-weighted past-round GT priors)
           5 % scaled-average (training-set mean priors × regime ratio)
    """

    # Default regime values used as the scaling baseline
    _DEFAULT_REGIME = {
        "survival_rate": 0.60,
        "spawn_rate": 0.05,
        "ruin_rate": 0.05,
        "forest_reclamation": 0.08,
        "spatial_decay": 0.30,
        "coastal_spawn_boost": 1.50,
    }

    def __init__(self, priors_path: Path | str | None = None) -> None:
        self._has_parametric: bool = False
        # Ridge model: W shape (n_cats, N_CLASSES, N_REGIME_FEATURES), b shape (n_cats, N_CLASSES)
        self._cat_keys: list[str] = []
        self._param_W: np.ndarray | None = None
        self._param_b: np.ndarray | None = None
        # Kernel interpolation
        self._kernel_regimes: list[np.ndarray] = []
        self._kernel_priors: list[dict[str, np.ndarray]] = []
        # Scaled average
        self._scaled_avg: dict[str, np.ndarray] = {}

        if priors_path is not None:
            path = Path(priors_path)
            if path.exists():
                self.load(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prior(self, cat: CellCategory, regime: "RegimeEstimate | None" = None) -> np.ndarray:
        """Return the best available prior for *cat*, optionally regime-conditioned."""
        if not self._has_parametric or regime is None:
            base = get_default_prior(cat)
            if regime is not None:
                base = self._apply_regime_scaling(base, cat, regime)
            return base

        p_param = self._predict_parametric(cat, regime)
        p_kernel = self._predict_kernel(cat, regime)
        p_scaled = self._predict_scaled(cat, regime)
        blended = 0.80 * p_param + 0.15 * p_kernel + 0.05 * p_scaled
        return _normalize(blended)

    # ------------------------------------------------------------------
    # Regime scaling (used without fitted model)
    # ------------------------------------------------------------------

    def _apply_regime_scaling(
        self,
        base: np.ndarray,
        cat: CellCategory,
        regime: "RegimeEstimate",
    ) -> np.ndarray:
        """Multiplicatively scale class probabilities by regime ratio vs defaults."""
        d = self._DEFAULT_REGIME
        p = base.copy()
        # Settlement probability scales with survival_rate
        surv_scale = regime.survival_rate / d["survival_rate"]
        p[1] = p[1] * surv_scale
        # Port probability scales with survival + coastal_spawn_boost (coastal cells only)
        port_boost = (regime.coastal_spawn_boost / d["coastal_spawn_boost"]) if cat.is_coastal else 1.0
        p[2] = p[2] * surv_scale * port_boost
        # Ruin scales with ruin_rate
        p[3] = p[3] * (regime.ruin_rate / d["ruin_rate"])
        # Forest scales with forest_reclamation
        p[4] = p[4] * max(0.4, regime.forest_reclamation / d["forest_reclamation"])
        return _normalize(p)

    # ------------------------------------------------------------------
    # Parametric (Ridge)
    # ------------------------------------------------------------------

    def _predict_parametric(self, cat: CellCategory, regime: "RegimeEstimate") -> np.ndarray:
        key = cat.to_key()
        if key not in self._cat_keys or self._param_W is None:
            return self._apply_regime_scaling(get_default_prior(cat), cat, regime)
        idx = self._cat_keys.index(key)
        rv = regime.as_vector()
        # Linear model output, then softmax
        logits = self._param_W[idx] @ rv + self._param_b[idx]  # (N_CLASSES,)
        logits -= logits.max()
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum()

    # ------------------------------------------------------------------
    # Kernel interpolation
    # ------------------------------------------------------------------

    def _predict_kernel(self, cat: CellCategory, regime: "RegimeEstimate") -> np.ndarray:
        if not self._kernel_priors:
            return self._apply_regime_scaling(get_default_prior(cat), cat, regime)
        key = cat.to_key()
        rv = regime.as_vector()
        weights: list[float] = []
        for kr in self._kernel_regimes:
            diff = rv - kr
            weights.append(float(np.exp(-0.5 * float(np.dot(diff, diff)) / (0.20 ** 2))))
        wt = np.array(weights, dtype=np.float64)
        if wt.sum() < 1e-10:
            wt = np.ones(len(weights), dtype=np.float64)
        wt /= wt.sum()
        p = np.zeros(N_CLASSES, dtype=np.float64)
        for w, kp in zip(wt, self._kernel_priors):
            prior = kp.get(key, get_default_prior(cat))
            p += w * prior
        return _normalize(p)

    # ------------------------------------------------------------------
    # Scaled average
    # ------------------------------------------------------------------

    def _predict_scaled(self, cat: CellCategory, regime: "RegimeEstimate") -> np.ndarray:
        key = cat.to_key()
        base = self._scaled_avg.get(key, get_default_prior(cat)).copy()
        return self._apply_regime_scaling(base, cat, regime)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save fitted priors to *path* (JSON)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "has_parametric": self._has_parametric,
            "cat_keys": self._cat_keys,
            "scaled_avg": {k: v.tolist() for k, v in self._scaled_avg.items()},
            "kernel_regimes": [v.tolist() for v in self._kernel_regimes],
            "kernel_priors": [
                {k: v.tolist() for k, v in d.items()} for d in self._kernel_priors
            ],
        }
        if self._param_W is not None:
            data["param_W"] = self._param_W.tolist()
            data["param_b"] = self._param_b.tolist()
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path | str) -> None:
        """Load fitted priors from *path* (JSON)."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self._has_parametric = raw.get("has_parametric", False)
        self._cat_keys = raw.get("cat_keys", [])
        self._scaled_avg = {k: np.array(v) for k, v in raw.get("scaled_avg", {}).items()}
        self._kernel_regimes = [np.array(v) for v in raw.get("kernel_regimes", [])]
        self._kernel_priors = [
            {k: np.array(v) for k, v in d.items()} for d in raw.get("kernel_priors", [])
        ]
        if "param_W" in raw:
            self._param_W = np.array(raw["param_W"])
            self._param_b = np.array(raw["param_b"])
