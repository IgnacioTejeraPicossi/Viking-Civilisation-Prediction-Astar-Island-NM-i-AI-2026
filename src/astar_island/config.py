"""Environment, hyperparameters, and path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def get_token() -> str:
    """Current AINM_TOKEN from the environment (after optional .env load)."""
    _load_dotenv_if_available()
    return os.getenv("AINM_TOKEN", "").strip()


# ---------------------------------------------------------------------------
# API / viewport defaults
# ---------------------------------------------------------------------------

DEFAULT_VIEWPORT_W: int = 15
DEFAULT_VIEWPORT_H: int = 15
MAX_QUERIES_PER_ROUND: int = 50

# requests timeout: (connect, read) tuple avoids hanging on dead TCP
HTTP_CONNECT_TIMEOUT: int = 15
DEFAULT_REQUEST_TIMEOUT: int = 30
SIMULATE_TIMEOUT: int = 60

# API rate limits (official): POST /simulate 5/s, POST /submit 2/s per team
SIMULATE_MIN_INTERVAL_SEC: float = 0.25
SUBMIT_MIN_INTERVAL_SEC: float = 0.55

# ---------------------------------------------------------------------------
# Prediction hyperparameters
# ---------------------------------------------------------------------------

# Minimum per-class probability applied before every submit (avoids infinite KL)
DEFAULT_PROB_FLOOR: float = 0.001

# Dirichlet smoothing alpha for empirical cell probabilities
DEFAULT_ALPHA_SMOOTHING: float = 0.5

# Temperature scaling (T > 1 softens predictions → better KL calibration)
TEMPERATURE_SCALING: float = 1.04

# ---------------------------------------------------------------------------
# Regime estimator
# ---------------------------------------------------------------------------

# Rounds with spatial_decay above this are flagged as chaotic and may be
# excluded from averaged priors in the refit pipeline
CHAOTIC_SPATIAL_DECAY_THRESHOLD: float = 0.95


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """src/astar_island/config.py → parents[2] = project root (editable installs)."""
    return Path(__file__).resolve().parents[2]


def get_regime_priors_path() -> Path:
    """Path to ``data/regime_priors.json`` relative to the project root."""
    root = _repo_root()
    if "site-packages" in str(root).lower():
        return Path.cwd() / "data" / "regime_priors.json"
    return root / "data" / "regime_priors.json"


# ---------------------------------------------------------------------------
# Token / URL helpers
# ---------------------------------------------------------------------------

def require_token() -> str:
    """Return JWT from AINM_TOKEN or raise ValueError."""
    t = get_token()
    if not t:
        raise ValueError("AINM_TOKEN is not set")
    return t


def get_base_url() -> str:
    """Return API base URL without trailing slash."""
    _load_dotenv_if_available()
    return os.getenv("AINM_BASE_URL", "https://api.ainm.no").rstrip("/")


def get_default_viewport() -> tuple[int, int]:
    """Default (width, height) for simulate viewport."""
    return DEFAULT_VIEWPORT_W, DEFAULT_VIEWPORT_H
