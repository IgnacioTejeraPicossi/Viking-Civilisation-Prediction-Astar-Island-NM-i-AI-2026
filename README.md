# NM i AI 2026 — Astar Island (Exercise 3-testing)

Python workspace for **Astar Island**: predict the final **terrain probability
distribution** (tensor **H×W×6**) for a black-box Norse simulator.
Each round gives **50 simulator queries** shared across **5 map seeds**;
each query returns a **15×15** viewport (see `docs/` for the official spec).

This repository is a **complete reference implementation** developed for the
NM i AI 2026 competition and subsequently improved using techniques from the
top-scoring solution (MIT licence).

**Short status / pointers:** **[STATUS.md](STATUS.md)**

---

## Algorithmic improvements over the competition version

The following upgrades were applied post-competition, inspired by the top
NM i AI 2026 entry (MIT licence):

| Technique | Where | Impact |
|-----------|-------|--------|
| **Chebyshev distance bins** (8 bins) | `gt_priors.py`, `features.py` | Much better spatial priors than flat terrain-type lookup |
| **Coastal / near-coastal-settlement stratification** | `gt_priors.py` | Correctly distinguishes port-eligible vs inland cells |
| **Bayesian regime shrinkage** | `regime_estimator.py` | 6 parameters (survival, spawn, ruin, forest, spatial decay, coastal boost) estimated with Beta-posterior shrinkage |
| **Entropy-driven adaptive queries** | `query_planner.py`, `runner.py` | Each viewport is chosen to maximise mean Shannon entropy of current predictions — beats static window scoring |
| **Regime-conditional parametric model** | `predictor_parametric.py`, `gt_priors.py` | Ridge (λ=0.001) regression mapping regime features → per-bin class probabilities; blends 80 % parametric + 15 % kernel + 5 % scaled avg when training data is available |
| **Temperature scaling** (T=1.04) | `submission.py`, `predictor_parametric.py` | Mild calibration softening that improves entropy-weighted KL |
| **Hard physical constraints** | `submission.py` | Port impossible inland; mountain impossible on dynamic cells |
| **Probability floor** tightened to 0.001 | `config.py` | Less over-smoothing while still avoiding infinite KL |
| **Oracle regime + LOO-CV pipeline** | `scripts/refit_regime_priors.py`, `scripts/evaluate_loo.py` | Offline measurement of model quality; enables iterative calibration |

---

## Technical stack

| Layer | Technology |
|-------|------------|
| Language | **Python ≥ 3.11** |
| Build | **setuptools** (editable install `pip install -e '.[dev]'`) |
| HTTP | **requests** — session with `Authorization: Bearer <JWT>` |
| Validation / IO | **Pydantic v2** |
| Numerics | **NumPy**, **SciPy**, **pandas** |
| ML | **scikit-learn** `Ridge` (refit pipeline); **LightGBM**, **CatBoost** (available) |
| HTTP service | **FastAPI** + **uvicorn** (`astar_island.service`) |
| Config | **`python-dotenv`** |
| Tests | **pytest** |

---

## Configuration (`src/astar_island/config.py`)

| Symbol | Value | Role |
|--------|-------|------|
| `AINM_TOKEN` | env | JWT (also loaded from `.env`) |
| `AINM_BASE_URL` | `https://api.ainm.no` | API host |
| `DEFAULT_VIEWPORT_W/H` | 15 | Simulate viewport size |
| `MAX_QUERIES_PER_ROUND` | 50 | Query budget |
| `DEFAULT_PROB_FLOOR` | **0.001** | Minimum class probability (tightened from 0.01) |
| `TEMPERATURE_SCALING` | **1.04** | Post-processing temperature |
| `CHAOTIC_SPATIAL_DECAY_THRESHOLD` | **0.95** | Flag for chaotic rounds |
| `HTTP_CONNECT_TIMEOUT` | 15 s | |
| `DEFAULT_REQUEST_TIMEOUT` | 30 s | |
| `SIMULATE_TIMEOUT` | 60 s | |
| `SIMULATE_MIN_INTERVAL_SEC` | 0.25 | Rate-limit guard |
| `SUBMIT_MIN_INTERVAL_SEC` | 0.55 | Rate-limit guard |

---

## Domain encoding (`src/astar_island/constants.py`)

**API terrain codes**: `0` empty, `1` settlement, `2` port, `3` ruin,
`4` forest, `5` mountain, `10` ocean, `11` plains.

**Prediction classes** (6-way per cell):
`0` Empty · `1` Settlement · `2` Port · `3` Ruin · `4` Forest · `5` Mountain

---

## Package layout (`src/astar_island/`)

| Module | Responsibility |
|--------|----------------|
| `config.py` | Env, timeouts, floors, temperature, paths |
| `constants.py` | Terrain codes, class indices, neighbourhoods |
| `models.py` | Pydantic: `RoundDetail`, `SimResult`, `BudgetInfo`, … |
| `api_client.py` | `AstarClient` — Bearer auth, 429 retry |
| `features.py` | `extract_map_features` — coast, BFS/Chebyshev distances, adjacency |
| **`gt_priors.py`** | `CellCategory`, 8-bin distance table, `GTPriorStore` (regime scaling + fitted model) |
| **`regime_estimator.py`** | `RegimeEstimate` dataclass, `BayesianRegimeEstimator` (6-param Bayesian shrinkage) |
| `predictor_baseline.py` | Distance-binned prior + empirical blend (no trained model needed) |
| **`predictor_parametric.py`** | `RegimeConditionalPredictor` — Ridge + kernel + scaled avg + post-processing |
| `observation_store.py` | Per-cell class counts + settlement samples |
| **`query_planner.py`** | Entropy-based `adaptive_pick_next_window`; static `initial_query_plan` |
| **`submission.py`** | Temperature scaling, port/mountain constraints, floor, validate, submit |
| **`runner.py`** | Adaptive loop: entropy pick → simulate → update pred → refit regime |
| `run_log.py` | JSON run log writer |
| `cli.py` | argparse: `run-round`, `budget`, `active`, `my-rounds`, `refit`, `eval` |
| `service.py` | FastAPI `/health`, `/solve` |
| `predictor_round_model.py` | Stub for future LightGBM model |

---

## Adaptive execution pipeline (`run_active_round`)

1. Resolve active round and fetch map features for all 5 seeds.
2. Check query budget; abort if insufficient.
3. Build **prior-only predictions** — seeds the entropy map for the planner.
4. **Adaptive simulation loop** (up to 50 queries):
   - Pick next viewport = window with highest **mean Shannon entropy** across all seeds' current predictions.
   - `POST /simulate` → update `ObservationStore`.
   - Rebuild prediction for the affected seed immediately.
   - Every 5 queries: re-estimate regime with `BayesianRegimeEstimator`.
5. Final regime estimation (all observations pooled across all seeds).
6. Final prediction pass with post-processing:
   - Temperature scaling (T=1.04)
   - Port constraint (zero inland port probability)
   - Mountain constraint (zero mountain probability on dynamic cells)
   - Floor 0.001 + renormalise.
7. Optional `POST /submit` for all seeds.
8. Optional JSON log to `data/observations/`.

---

## Regime parameters

| Parameter | Prior mean | Prior n | Measures |
|-----------|-----------|---------|---------|
| `survival_rate` | 0.60 | 20 | Fraction of initial settlements still alive |
| `spawn_rate` | 0.05 | 8 | New-settlement density on non-initial land cells |
| `ruin_rate` | 0.05 | 40 | Fraction of land cells that are ruins |
| `forest_reclamation` | 0.08 | 15 | Fraction of land cells that are forest |
| `spatial_decay` | 0.30 | 10 | Clustering of spawns near initial settlements |
| `coastal_spawn_boost` | 1.50 | 10 | Coastal vs inland settlement rate ratio |

Rounds with `spatial_decay > 0.95` are flagged **chaotic** and can be excluded
from the training set via `astar-island refit --no-chaotic`.

---

## Scoring (competition rules)

Ground truth is a **distribution** per cell.
Metric: **entropy-weighted KL divergence**:

```
score = 100 × exp(−3 × weighted_KL)
weighted_KL = Σ_cells entropy_weight(cell) × KL(gt ∥ pred) / Σ entropy_weight
```

Ocean and mountain cells (near-zero entropy) contribute negligibly.
Predictions must be non-negative, sum to 1 per cell, with floor ≥ 0.001.

---

## Requirements

- **Python 3.11+**
- Account on [app.ainm.no](https://app.ainm.no) and a JWT token

---

## Setup

1. Copy `.env.example` to `.env` and set:

```env
AINM_TOKEN=your_jwt_here
AINM_BASE_URL=https://api.ainm.no
```

2. Editable install (from repo root):

```powershell
pip install -e '.[dev]'
```

On Windows PowerShell use single quotes around `'.[dev]'`.

---

## Usage

### Check active round and budget

```powershell
astar-island active
astar-island budget
```

### Full adaptive run (simulate + predict + submit all seeds)

```powershell
astar-island run-round
```

### Simulate only (no submit)

```powershell
astar-island run-round --no-submit
```

### Flags for `run-round`

| Flag | Effect |
|------|--------|
| `--max-queries N` | Cap query budget |
| `--no-log` | Skip JSON log |
| `--round-id UUID` | Target a specific round |
| `--no-submit` | Skip submission |

### Backfill completed rounds (fetch GT data)

```powershell
python scripts/backfill_completed_rounds.py
```

Writes `data/rounds/{round_id}_seed{n}.json` for every completed round.

### Refit regime priors (after backfill)

```powershell
astar-island refit
# or exclude chaotic rounds:
astar-island refit --no-chaotic
```

Writes `data/regime_priors.json`.  The parametric predictor loads this
automatically on the next `run-round`.

### LOO cross-validation

```powershell
astar-island eval
```

Prints per-round scores and an average.  Results saved to `data/loo_results.json`.

### Tests

```powershell
pytest -q
```

### Optional HTTP server

```powershell
uvicorn astar_island.service:app --host 0.0.0.0 --port 8080
```

---

## Troubleshooting

- **HTTP 429:** rate limit or no queries left — check `astar-island budget`.
- **`refit` says "no data":** run `scripts/backfill_completed_rounds.py` first.
- **Slow first run:** Chebyshev BFS runs per seed at startup (~ms on 40×40).
- **Module not found:** reinstall with `pip install -e '.[dev]'`.

---

## API limits (official)

- **`POST /astar-island/simulate`:** max **5 req/s** per team.
- **`POST /astar-island/submit`:** max **2 req/s** per team.

---

## Repository layout

```
src/astar_island/     # Main package
scripts/              # run_round, backfill, refit_regime_priors, evaluate_loo, train_meta_model
tests/
data/                 # raw, observations, rounds, models, loo_results.json
docs/                 # Competition and planning docs
STATUS.md
```

---

## Entry point

`[project.scripts]` in `pyproject.toml`: **`astar-island`** → `astar_island.cli:main`.
