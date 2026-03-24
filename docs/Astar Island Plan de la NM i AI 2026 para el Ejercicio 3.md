Plan de la NM i AI 2026 para el Ejercicio 3: Astar Island
...
Ignacio: NM i AI 2026, ejercicio 'Astar Island'. Buenos dias, he abierto un nuevo chat para este tercer ejercicio de la competicion poruqe en el otro que habiamos hecho los planes para los dos primers ejercicios estaba saturado y muy lento. El objetivo es realizar un plan de implentacion paso por paso y con ejemplos de codigo para que yo lo lleve a Cursor AI y lo implementemos. Esta operacion la has hecho ya con exito en los dos ejercicios anteriores. Aqui arriba te colocare todos los documentos necesarios , con la informacio necesaria para que crees ese plan ambicioso. Adelante mi heroe con el plan, pideme si ves que necesitas alguna informacion mas que no este clara, aunque en esos documentos esta todo lo aportado por los organizadores Astar.Los documentos de Google som informacion del entorn en que se llevara a cabo, y los documentos de Astar son la descripcion del propio ejercicio, por ultimo el 'Geting Starting' es el primero que debes leer para ponerte en la situacion, gracias. La simulacion te la coloco aqui para ver si puedes verla: https://app.ainm.no/docs/astar-island/mechanics
ChatGPT:
Voy a consolidar la mecánica del ejercicio, el scoring y el flujo de API, y a partir de eso te preparo un plan de implementación agresivo pero ejecutable en Cursor AI. También voy a cruzarlo con la documentación del entorno para que el plan ya salga orientado a deploy y submission.
Ya tengo lo esencial: el problema no es “ver más mapa”, sino gastar las 50 queries para inferir los parámetros ocultos que controlan crecimiento, conflicto, puertos, ruina y recuperación. Eso cambia el plan: primero necesitamos un baseline sólido y después una capa de inferencia de “régimen del mundo”.
Thought for 2m 37s
Sí, pude leer la página de mechanics y con eso ya se puede montar un plan serio. Detecté además una ambigüedad importante en la documentación oficial: Astar Island tiene un flujo documentado de cliente directo contra GET /rounds, POST /simulate y POST /submit, pero la página general de Cloud Run también menciona Astar como tarea con endpoint HTTPS /solve. Para no bloquearos por esa inconsistencia, el plan de abajo va con arquitectura dual: core solver local/CLI primero y wrapper FastAPI opcional después. 
La idea ganadora aquí no es “ver más mapa”, sino inferir el régimen oculto del mundo. El round tiene 5 seeds, 50 queries totales compartidas, viewport máximo 15×15, el mapa inicial sí es visible, los parámetros ocultos son comunes al round, y cada llamada a simulate usa un sim_seed distinto. Eso significa que una misma ventana repetida varias veces sirve para estimar distribución local, mientras que observar varias seeds sirve para inferir expansión, conflicto, comercio, inviernos duros y recuperación de ruinas. Además, el scoring penaliza mucho los ceros y pondera más las celdas dinámicas/entrópicas. 
Plan ambicioso de implementación
Fase 0 — Estructura del proyecto
Objetivo: dejar un esqueleto limpio para que Cursor AI pueda implementar por módulos.
astar-island/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ .env.example
├─ src/
│  └─ astar_island/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ constants.py
│     ├─ models.py
│     ├─ api_client.py
│     ├─ terrain.py
│     ├─ features.py
│     ├─ query_planner.py
│     ├─ observation_store.py
│     ├─ regime_estimator.py
│     ├─ predictor_baseline.py
│     ├─ predictor_round_model.py
│     ├─ submission.py
│     ├─ service.py          # opcional FastAPI
│     └─ cli.py
├─ scripts/
│  ├─ run_round.py
│  ├─ backfill_completed_rounds.py
│  └─ train_meta_model.py
├─ data/
│  ├─ raw/
│  ├─ observations/
│  ├─ rounds/
│  └─ models/
└─ tests/
   ├─ test_api_client.py
   ├─ test_features.py
   ├─ test_query_planner.py
   ├─ test_predictor.py
   └─ test_submission.py
requirements.txt inicial:
numpy
pandas
pydantic
requests
scipy
scikit-learn
lightgbm
catboost
fastapi
uvicorn[standard]
python-dotenv
pytest
Fase 1 — Cliente API y modelos de dominio
Objetivo: tener acceso robusto al round activo, budget, initial states, simulate y submit. La especificación oficial documenta autenticación con cookie o Bearer, GET /rounds, GET /rounds/{round_id}, POST /simulate, POST /submit, /my-rounds, /my-predictions/{round_id} y /analysis/{round_id}/{seed_index}. 
constants.py
CLASS_EMPTY = 0
CLASS_SETTLEMENT = 1
CLASS_PORT = 2
CLASS_RUIN = 3
CLASS_FOREST = 4
CLASS_MOUNTAIN = 5

TERRAIN_OCEAN = 10
TERRAIN_PLAINS = 11
TERRAIN_EMPTY = 0
TERRAIN_SETTLEMENT = 1
TERRAIN_PORT = 2
TERRAIN_RUIN = 3
TERRAIN_FOREST = 4
TERRAIN_MOUNTAIN = 5

TERRAIN_TO_CLASS = {
    TERRAIN_OCEAN: CLASS_EMPTY,
    TERRAIN_PLAINS: CLASS_EMPTY,
    TERRAIN_EMPTY: CLASS_EMPTY,
    TERRAIN_SETTLEMENT: CLASS_SETTLEMENT,
    TERRAIN_PORT: CLASS_PORT,
    TERRAIN_RUIN: CLASS_RUIN,
    TERRAIN_FOREST: CLASS_FOREST,
    TERRAIN_MOUNTAIN: CLASS_MOUNTAIN,
}
models.py
from pydantic import BaseModel
from typing import List, Optional

class InitialSettlement(BaseModel):
    x: int
    y: int
    has_port: bool
    alive: bool

class InitialState(BaseModel):
    grid: List[List[int]]
    settlements: List[InitialSettlement]

class RoundDetail(BaseModel):
    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: List[InitialState]

class SimSettlement(BaseModel):
    x: int
    y: int
    population: float
    food: float
    wealth: float
    defense: float
    has_port: bool
    alive: bool
    owner_id: int

class SimResult(BaseModel):
    grid: List[List[int]]
    settlements: List[SimSettlement]
    viewport: dict
    width: int
    height: int
    queries_used: int
    queries_max: int
api_client.py
import requests
from typing import Any, Dict
from .models import RoundDetail, SimResult

class AstarClient:
    def __init__(self, token: str, base_url: str = "https://api.ainm.no"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def get_rounds(self) -> list[dict]:
        r = self.session.get(f"{self.base_url}/astar-island/rounds", timeout=30)
        r.raise_for_status()
        return r.json()

    def get_active_round(self) -> dict | None:
        rounds = self.get_rounds()
        return next((r for r in rounds if r["status"] == "active"), None)

    def get_round_detail(self, round_id: str) -> RoundDetail:
        r = self.session.get(f"{self.base_url}/astar-island/rounds/{round_id}", timeout=30)
        r.raise_for_status()
        return RoundDetail.model_validate(r.json())

    def get_budget(self) -> dict:
        r = self.session.get(f"{self.base_url}/astar-island/budget", timeout=30)
        r.raise_for_status()
        return r.json()

    def simulate(self, round_id: str, seed_index: int, x: int, y: int, w: int = 15, h: int = 15) -> SimResult:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": x,
            "viewport_y": y,
            "viewport_w": w,
            "viewport_h": h,
        }
        r = self.session.post(f"{self.base_url}/astar-island/simulate", json=payload, timeout=60)
        r.raise_for_status()
        return SimResult.model_validate(r.json())

    def submit(self, round_id: str, seed_index: int, prediction: list[list[list[float]]]) -> dict:
        payload = {"round_id": round_id, "seed_index": seed_index, "prediction": prediction}
        r = self.session.post(f"{self.base_url}/astar-island/submit", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def my_rounds(self) -> list[dict]:
        r = self.session.get(f"{self.base_url}/astar-island/my-rounds", timeout=30)
        r.raise_for_status()
        return r.json()

    def analysis(self, round_id: str, seed_index: int) -> dict:
        r = self.session.get(f"{self.base_url}/astar-island/analysis/{round_id}/{seed_index}", timeout=60)
        r.raise_for_status()
        return r.json()
Fase 2 — Reconstrucción local y features del mapa inicial
La mecánica oficial dice que el mapa inicial es reconstruible localmente, que océano/plains/empty colapsan a clase 0, que mountain es estático, que forest es mayormente estático pero puede recuperar ruinas, y que lo interesante son Settlement, Port y Ruin. También sabemos que el ciclo anual pasa por growth, conflict, trade, winter y environment. 
Eso nos permite calcular features sin gastar queries:
•	is_ocean_border
•	is_coast
•	is_mountain
•	is_forest
•	dist_to_initial_settlement
•	dist_to_coast
•	adjacent_forests
•	adjacent_mountains
•	frontier_score
•	port_potential
•	ruin_reclaim_potential
features.py
import numpy as np
from collections import deque
from .constants import *

NEI4 = [(1,0),(-1,0),(0,1),(0,-1)]
NEI8 = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1) if dx or dy]

def in_bounds(h: int, w: int, y: int, x: int) -> bool:
    return 0 <= y < h and 0 <= x < w

def is_land(v: int) -> bool:
    return v not in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN)

def is_coast(grid: np.ndarray, y: int, x: int) -> bool:
    if not is_land(grid[y, x]):
        return False
    h, w = grid.shape
    for dx, dy in NEI4:
        yy, xx = y + dy, x + dx
        if in_bounds(h, w, yy, xx) and grid[yy, xx] == TERRAIN_OCEAN:
            return True
    return False

def multi_source_distance(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    dist = np.full((h, w), 10**9, dtype=np.int32)
    q = deque()
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        dist[y, x] = 0
        q.append((y, x))
    while q:
        y, x = q.popleft()
        for dx, dy in NEI4:
            yy, xx = y + dy, x + dx
            if in_bounds(h, w, yy, xx) and dist[yy, xx] > dist[y, x] + 1:
                dist[yy, xx] = dist[y, x] + 1
                q.append((yy, xx))
    return dist

def extract_map_features(grid: list[list[int]], settlements: list[dict]) -> dict[str, np.ndarray]:
    arr = np.array(grid, dtype=np.int16)
    h, w = arr.shape

    ocean_mask = arr == TERRAIN_OCEAN
    mountain_mask = arr == TERRAIN_MOUNTAIN
    forest_mask = arr == TERRAIN_FOREST
    coast_mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            coast_mask[y, x] = is_coast(arr, y, x)

    settlement_mask = np.zeros((h, w), dtype=bool)
    port_mask = np.zeros((h, w), dtype=bool)
    for s in settlements:
        settlement_mask[s["y"], s["x"]] = True
        if s["has_port"]:
            port_mask[s["y"], s["x"]] = True

    dist_settlement = multi_source_distance(settlement_mask) if settlement_mask.any() else np.full((h, w), 999)
    dist_coast = multi_source_distance(coast_mask) if coast_mask.any() else np.full((h, w), 999)

    adjacent_forests = np.zeros((h, w), dtype=np.int16)
    for y in range(h):
        for x in range(w):
            c = 0
            for dx, dy in NEI8:
                yy, xx = y + dy, x + dx
                if in_bounds(h, w, yy, xx) and arr[yy, xx] == TERRAIN_FOREST:
                    c += 1
            adjacent_forests[y, x] = c

    frontier_score = np.exp(-dist_settlement / 3.0) * (~mountain_mask) * (~ocean_mask)

    return {
        "terrain": arr,
        "ocean_mask": ocean_mask,
        "mountain_mask": mountain_mask,
        "forest_mask": forest_mask,
        "coast_mask": coast_mask,
        "settlement_mask": settlement_mask,
        "port_mask": port_mask,
        "dist_settlement": dist_settlement,
        "dist_coast": dist_coast,
        "adjacent_forests": adjacent_forests,
        "frontier_score": frontier_score,
    }
Fase 3 — Query planner: gastar las 50 queries con intención
La documentación oficial deja claro que las 50 queries se comparten entre las 5 seeds y que cada simulate devuelve solo una ventana 5–15. Repetir una ventana no es desperdicio: cada llamada usa un sim_seed distinto, así que obtienes una muestra nueva de la distribución local. 
Estrategia de budget recomendada
•	15 queries: 1 ventana “anchor” por seed, repetida 3 veces.
•	15 queries: 1 ventana “frontier expansion” por seed, repetida 3 veces.
•	10 queries: 1 ventana “coastal/trade/port” en las seeds donde exista mayor costa útil.
•	10 queries: adaptivas según lo observado.
•	Reserva 0–5: solo para revalidación final si todavía quedan.
Qué debe maximizar una ventana:
•	densidad de asentamientos iniciales
•	mezcla costa + asentamientos
•	frontera entre tierra libre y asentamientos
•	zonas donde una ruina o expansión futura sea plausible
•	no gastar muchas queries en montañas ni océano
query_planner.py
from dataclasses import dataclass
import numpy as np

@dataclass
class WindowPlan:
    seed_index: int
    x: int
    y: int
    w: int
    h: int
    tag: str
    repeats: int

def clamp_window(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int]:
    x = max(0, min(x, width - w))
    y = max(0, min(y, height - h))
    return x, y

def score_window(feats: dict[str, np.ndarray], x: int, y: int, w: int = 15, h: int = 15) -> float:
    sl = np.s_[y:y+h, x:x+w]
    s_density = feats["settlement_mask"][sl].mean()
    p_density = feats["port_mask"][sl].mean()
    coast = feats["coast_mask"][sl].mean()
    frontier = feats["frontier_score"][sl].mean()
    forest = feats["forest_mask"][sl].mean()
    mountain = feats["mountain_mask"][sl].mean()
    return (
        3.0 * s_density +
        2.0 * p_density +
        1.8 * coast +
        2.5 * frontier +
        0.7 * forest -
        2.2 * mountain
    )

def best_windows_for_seed(feats: dict[str, np.ndarray], width: int, height: int, stride: int = 5) -> list[tuple[float,int,int]]:
    candidates = []
    for y in range(0, height - 14, stride):
        for x in range(0, width - 14, stride):
            candidates.append((score_window(feats, x, y), x, y))
    return sorted(candidates, reverse=True)

def initial_query_plan(all_seed_features: list[dict[str, np.ndarray]], width: int, height: int) -> list[WindowPlan]:
    plans: list[WindowPlan] = []
    for seed_idx, feats in enumerate(all_seed_features):
        ranked = best_windows_for_seed(feats, width, height)
        if ranked:
            _, x1, y1 = ranked[0]
            plans.append(WindowPlan(seed_idx, x1, y1, 15, 15, "anchor", 3))
        if len(ranked) > 1:
            _, x2, y2 = ranked[1]
            plans.append(WindowPlan(seed_idx, x2, y2, 15, 15, "frontier", 3))
    return plans
Fase 4 — Observation store y muestras repetidas
Objetivo: guardar cada simulate como muestra probabilística, no como snapshot único.
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

@dataclass
class CellObs:
    counts: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int32))
    n: int = 0

    def update(self, cls_idx: int):
        self.counts[cls_idx] += 1
        self.n += 1

    def probs(self, alpha: float = 0.5) -> np.ndarray:
        p = self.counts.astype(float) + alpha
        return p / p.sum()

class ObservationStore:
    def __init__(self, width: int, height: int, seeds_count: int):
        self.width = width
        self.height = height
        self.cell_obs = [
            [[CellObs() for _ in range(width)] for _ in range(height)]
            for _ in range(seeds_count)
        ]
        self.settlement_samples = defaultdict(list)

    def add_sim_result(self, seed_index: int, sim_result, terrain_to_class: dict[int, int]):
        vx = sim_result.viewport["x"]
        vy = sim_result.viewport["y"]
        grid = sim_result.grid

        for dy, row in enumerate(grid):
            for dx, v in enumerate(row):
                x = vx + dx
                y = vy + dy
                cls_idx = terrain_to_class[v]
                self.cell_obs[seed_index][y][x].update(cls_idx)

        for s in sim_result.settlements:
            self.settlement_samples[(seed_index, s.x, s.y)].append({
                "population": s.population,
                "food": s.food,
                "wealth": s.wealth,
                "defense": s.defense,
                "has_port": s.has_port,
                "alive": s.alive,
                "owner_id": s.owner_id,
            })

    def empirical_cell_probs(self, seed_index: int, y: int, x: int) -> np.ndarray | None:
        obs = self.cell_obs[seed_index][y][x]
        return obs.probs() if obs.n > 0 else None
Fase 5 — Estimador de régimen del round
Las fases de simulación oficiales permiten inferir latentes útiles: crecimiento/expansión, conflicto/raids, comercio/puertos, dureza del invierno y recuperación ambiental. Además, simulate devuelve stats internas solo para asentamientos dentro del viewport: population, food, wealth, defense, owner_id, port y alive. Eso es oro para un estimador de régimen. 
Propuesta práctica: no intentes primero un modelo bayesiano complejo; empieza con un regime estimator por scores en [0,1].
import numpy as np

def safe_mean(xs):
    return float(np.mean(xs)) if xs else 0.0

class RoundRegimeEstimator:
    def estimate(self, observation_store) -> dict[str, float]:
        foods, wealths, defenses, pops = [], [], [], []
        owner_switch_proxy = []
        ruin_mass = []
        port_alive = []

        for samples in observation_store.settlement_samples.values():
            owners = [s["owner_id"] for s in samples if s["alive"]]
            foods.extend([s["food"] for s in samples if s["alive"]])
            wealths.extend([s["wealth"] for s in samples if s["alive"]])
            defenses.extend([s["defense"] for s in samples if s["alive"]])
            pops.extend([s["population"] for s in samples if s["alive"]])

            if owners:
                owner_switch_proxy.append(len(set(owners)) / max(1, len(owners)))
            port_alive.extend([1.0 if s["has_port"] and s["alive"] else 0.0 for s in samples])

        # proxies simples; luego se refinan
        expansion = np.clip(0.45 * safe_mean(pops) + 0.35 * safe_mean(foods) + 0.20 * safe_mean(wealths), 0, 1)
        aggression = np.clip(0.60 * safe_mean(owner_switch_proxy) + 0.20 * (1 - safe_mean(foods)) + 0.20 * safe_mean(defenses), 0, 1)
        trade = np.clip(0.50 * safe_mean(wealths) + 0.30 * safe_mean(port_alive) + 0.20 * safe_mean(foods), 0, 1)
        harsh_winter = np.clip(1 - safe_mean(foods), 0, 1)
        reclamation = np.clip(0.55 * safe_mean(foods) + 0.25 * safe_mean(pops) + 0.20 * safe_mean(port_alive), 0, 1)

        return {
            "expansion": float(expansion),
            "aggression": float(aggression),
            "trade": float(trade),
            "harsh_winter": float(harsh_winter),
            "reclamation": float(reclamation),
        }
Fase 6 — Predictor baseline serio
Aquí está el primer baseline competitivo de verdad.
Reglas del baseline
1.	Mountain: casi one-hot a mountain.
2.	Ocean: casi one-hot a empty.
3.	Forest: prior alto a forest, pequeño a ruin/empty.
4.	Initial settlement:
o	si era puerto inicial o costa fuerte: masa en settlement/port/ruin
o	si inland y rodeado de bosque: más settlement
o	si conflicto/agresión alta o invierno duro: más ruin
5.	Tierra vacía/plains:
o	lejos de asentamientos: empty dominante
o	frontera cercana a settlements + expansión alta: subir settlement
o	costa cercana + trade alto: subir port un poco
6.	Ruin candidates:
o	alrededor de settlements iniciales o de celdas observadas como settlement/port
o	si harsh_winter o aggression altos: subir ruin
o	si reclamation alto y bosque cerca: repartir ruin/forest/empty
7.	Si una celda fue observada varias veces, mezclar prior con frecuencia empírica local.
predictor_baseline.py
import numpy as np
from .constants import *

def normalize_probs(p: np.ndarray, floor: float = 0.01) -> np.ndarray:
    p = np.maximum(p, floor)
    return p / p.sum()

def base_cell_prior(feats: dict[str, np.ndarray], y: int, x: int, regime: dict[str, float]) -> np.ndarray:
    terrain = feats["terrain"][y, x]
    coast = feats["coast_mask"][y, x]
    forest = feats["forest_mask"][y, x]
    mountain = feats["mountain_mask"][y, x]
    dist_set = feats["dist_settlement"][y, x]
    frontier = feats["frontier_score"][y, x]
    adj_forest = feats["adjacent_forests"][y, x]

    p = np.array([0.70, 0.08, 0.05, 0.07, 0.07, 0.03], dtype=float)

    if terrain == TERRAIN_MOUNTAIN:
        return normalize_probs(np.array([0.01, 0.005, 0.005, 0.005, 0.005, 0.97]))
    if terrain == TERRAIN_OCEAN:
        return normalize_probs(np.array([0.97, 0.005, 0.005, 0.005, 0.005, 0.01]))
    if terrain == TERRAIN_FOREST:
        return normalize_probs(np.array([0.10, 0.03, 0.01, 0.08, 0.75, 0.03]))
    if terrain == TERRAIN_SETTLEMENT:
        return normalize_probs(np.array([
            0.03,
            0.55 - 0.15 * regime["harsh_winter"],
            0.12 + 0.10 * coast * regime["trade"],
            0.22 + 0.12 * regime["aggression"] + 0.08 * regime["harsh_winter"],
            0.05,
            0.03,
        ]))
    if terrain == TERRAIN_PORT:
        return normalize_probs(np.array([
            0.03,
            0.18,
            0.54 + 0.12 * regime["trade"],
            0.17 + 0.08 * regime["aggression"],
            0.04,
            0.04,
        ]))
    if terrain == TERRAIN_RUIN:
        return normalize_probs(np.array([
            0.15,
            0.12 * regime["reclamation"],
            0.05 * coast * regime["reclamation"],
            0.45 + 0.10 * regime["aggression"],
            0.18 + 0.15 * (adj_forest / 8.0),
            0.05,
        ]))

    # plains / empty land
    p_empty = 0.82 - 0.35 * frontier
    p_settlement = 0.08 + 0.28 * frontier * regime["expansion"] * np.exp(-dist_set / 4.0)
    p_port = 0.02 + 0.16 * coast * regime["trade"] * np.exp(-dist_set / 5.0)
    p_ruin = 0.03 + 0.12 * frontier * regime["aggression"]
    p_forest = 0.03 + 0.10 * (adj_forest / 8.0) * (1 - regime["expansion"])
    p_mountain = 0.02

    return normalize_probs(np.array([p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]))

def build_prediction(seed_features: dict[str, np.ndarray], regime: dict[str, float], observation_store, seed_index: int) -> np.ndarray:
    h, w = seed_features["terrain"].shape
    pred = np.zeros((h, w, 6), dtype=float)

    for y in range(h):
        for x in range(w):
            prior = base_cell_prior(seed_features, y, x, regime)
            empirical = observation_store.empirical_cell_probs(seed_index, y, x)

            if empirical is None:
                pred[y, x] = prior
            else:
                # más peso a la observación si se repitió varias veces
                n = observation_store.cell_obs[seed_index][y][x].n
                lam = min(0.75, 0.20 + 0.10 * n)
                pred[y, x] = normalize_probs((1 - lam) * prior + lam * empirical)

    return pred
Fase 7 — Learning layer entre rounds
Aquí es donde podéis despegar de verdad.
El endpoint oficial /analysis/{round_id}/{seed_index} devuelve prediction + ground truth + score + initial grid una vez terminado el round. Y /my-rounds devuelve score por round y por seed. Eso permite construir un dataset propio de rounds históricos y entrenar un meta-modelo. 
Dataset por celda
Por cada celda y seed:
•	terrain inicial
•	coast, dist_to_settlement, adj_forest, frontier_score
•	stats agregadas de la ventana si fue observada
•	probs empíricas locales si la celda fue observada
•	régimen estimado del round
•	target = ground_truth[y][x][class]
Modelos recomendados
Orden de implementación:
1.	baseline heurístico
2.	LightGBMRegressor 6 modelos one-vs-rest
3.	calibración + renormalización
4.	opcional CatBoost
train_meta_model.py idea:
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

model = MultiOutputRegressor(
    LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
)

model.fit(X_train, Y_train)  # Y_train shape = [n_samples, 6]
pred = model.predict(X_valid)
pred = np.maximum(pred, 0.01)
pred = pred / pred.sum(axis=1, keepdims=True)
Punto importante: como el scoring es KL ponderado por entropía, no optimicéis solo accuracy del argmax. Hay que optimizar calibración de probabilidades. 
Fase 8 — Validación y envío seguro
La especificación oficial exige tensor H×W×6, no negativos, suma 1.0 por celda, y la última submission por seed sobrescribe la anterior. El scoring oficial recomienda floor mínimo 0.01 para evitar KL infinita. 
submission.py
import numpy as np

def sanitize_prediction(pred: np.ndarray, floor: float = 0.01) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    pred = np.maximum(pred, floor)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred

def validate_prediction(pred: np.ndarray, height: int, width: int):
    if pred.shape != (height, width, 6):
        raise ValueError(f"Expected {(height, width, 6)}, got {pred.shape}")
    if np.any(pred < 0):
        raise ValueError("Negative probability found")
    s = pred.sum(axis=-1)
    if not np.allclose(s, 1.0, atol=1e-3):
        raise ValueError("Probabilities do not sum to 1.0")

def submit_all_seeds(client, round_id: str, preds: list[np.ndarray]):
    for seed_idx, pred in enumerate(preds):
        pred = sanitize_prediction(pred)
        validate_prediction(pred, pred.shape[0], pred.shape[1])
        resp = client.submit(round_id, seed_idx, pred.tolist())
        print(seed_idx, resp)
Fase 9 — Runner completo del round
scripts/run_round.py
from astar_island.api_client import AstarClient
from astar_island.features import extract_map_features
from astar_island.query_planner import initial_query_plan
from astar_island.observation_store import ObservationStore
from astar_island.regime_estimator import RoundRegimeEstimator
from astar_island.predictor_baseline import build_prediction
from astar_island.submission import submit_all_seeds
from astar_island.constants import TERRAIN_TO_CLASS
import os

def main():
    token = os.environ["AINM_TOKEN"]
    client = AstarClient(token)
    active = client.get_active_round()
    if not active:
        raise RuntimeError("No active round")

    detail = client.get_round_detail(active["id"])
    all_features = []

    for state in detail.initial_states:
        settlements = [s.model_dump() for s in state.settlements]
        feats = extract_map_features(state.grid, settlements)
        all_features.append(feats)

    store = ObservationStore(detail.map_width, detail.map_height, detail.seeds_count)
    plans = initial_query_plan(all_features, detail.map_width, detail.map_height)

    for p in plans:
        for _ in range(p.repeats):
            sim = client.simulate(detail.id, p.seed_index, p.x, p.y, p.w, p.h)
            store.add_sim_result(p.seed_index, sim, TERRAIN_TO_CLASS)

    regime = RoundRegimeEstimator().estimate(store)

    preds = []
    for seed_idx, feats in enumerate(all_features):
        pred = build_prediction(feats, regime, store, seed_idx)
        preds.append(pred)

    submit_all_seeds(client, detail.id, preds)

if __name__ == "__main__":
    main()
Fase 10 — Wrapper FastAPI opcional
Yo no empezaría por aquí, pero como la documentación general de Cloud Run menciona Astar con endpoint HTTPS y además recomienda europe-north1, min-instances=1 y revisar logs, conviene dejarlo preparado si decidís desplegarlo así. 
service.py
from fastapi import FastAPI
from pydantic import BaseModel
from .runner import solve_active_round

app = FastAPI()

class SolveRequest(BaseModel):
    token: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/solve")
def solve(req: SolveRequest):
    result = solve_active_round(req.token)
    return {"status": "completed", "result": result}
Orden realista para Cursor AI
Este es el orden exacto que yo seguiría:
1.	implementar constants.py, models.py, api_client.py
2.	probar get_active_round, get_round_detail, get_budget
3.	implementar features.py
4.	generar un notebook o script que pinte el mapa inicial por seed
5.	implementar query_planner.py
6.	implementar observation_store.py
7.	ejecutar 5–10 queries manuales y persistir resultados
8.	implementar regime_estimator.py
9.	implementar predictor_baseline.py
10.	implementar submission.py
11.	correr una submission completa
12.	implementar /my-rounds + /analysis backfill
13.	entrenar meta-modelo entre rounds
14.	añadir service.py y Docker solo si os aporta valor operativo
Qué probar en la primera versión
Primera submission útil:
•	mountains casi one-hot
•	ocean casi one-hot empty
•	forest prior fuerte
•	settlement/port/ruin con priors dependientes de costa, frontier y régimen
•	floor 0.01
•	queries repetidas en ventanas informativas
Eso ya debería superar claramente el uniforme, porque aprovecha:
•	estructura del mapa inicial,
•	estocasticidad muestral por repetición,
•	y régimen compartido del round. 
Qué haría yo para ir a por score alto
Tres capas, en este orden:
•	Capa 1: baseline heurístico probabilístico estable
•	Capa 2: estimator de régimen del round con repeated windows
•	Capa 3: meta-modelo entrenado con /analysis de rounds anteriores
La tercera capa es la que más upside tiene, porque transforma cada round pasado en datos supervisados reales del organizador. 
Recomendación táctica final
No intentéis “cubrir el mapa” con las 50 queries. Intentad medir incertidumbre y régimen. El scoring recompensa acertar distribuciones en celdas dinámicas, no acertar montañas y océano. Repetir la misma ventana 2–4 veces en sitios clave vale más que abrir ventanas aleatorias por todo el grid. 
•	
5f6f0582-0748-42a3-8613-fbda971b9a6e.md
autenticación, flujo base contra la API y formato del tensor H×W×6

•	
c84eaf65-5918-47c5-8d60-51a09d1f2eac.md
fórmula de scoring, entropía, KL y floor de probabilidad

•	
97e41340-30c0-4708-996f-5a80a09501a2.md
contexto general del campeonato y encaje de la tarea

•	
09ba217c-5459-4323-a834-aa15ff2a2cd7.md
notas de despliegue en Cloud Run, región y logs

En el siguiente paso te conviene pedir que te lo convierta en un pack listo para pegar en Cursor AI, dividido por archivos concretos y prompts de implementación uno a uno.

Ignacio: Puedes convertirlo en un pack listo para pegar en Cursor AI, dividido por archivos concretos y prompts de implementación uno a uno.
...
Sí. Aquí tienes un pack listo para pegar en Cursor AI, dividido por archivos y por prompts individuales, en un orden de implementación práctico.
La idea es que pegues un prompt cada vez en Cursor, revises el resultado, hagas build/tests, y luego pases al siguiente. Así evitas que Cursor mezcle demasiadas responsabilidades en una sola iteración.
________________________________________
Pack Cursor AI — Astar Island
Modo de uso recomendado en Cursor
1.	Crea la estructura de carpetas.
2.	Pega el Prompt 0 para el scaffolding.
3.	Luego pega los prompts 1, 2, 3... en orden.
4.	Después de cada prompt:
o	revisa diff,
o	corrige imports,
o	ejecuta tests,
o	confirma que no rompió nada.
5.	No le pidas todavía optimizaciones complejas hasta tener el baseline funcionando end-to-end.
________________________________________
Prompt 0 — Scaffolding inicial del proyecto
Pega esto en Cursor:
Quiero que prepares el scaffolding inicial para un proyecto Python llamado `astar-island` orientado al ejercicio "Astar Island" de NM i AI 2026.

Crea esta estructura:

astar-island/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ .env.example
├─ src/
│  └─ astar_island/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ constants.py
│     ├─ models.py
│     ├─ api_client.py
│     ├─ terrain.py
│     ├─ features.py
│     ├─ query_planner.py
│     ├─ observation_store.py
│     ├─ regime_estimator.py
│     ├─ predictor_baseline.py
│     ├─ predictor_round_model.py
│     ├─ submission.py
│     ├─ cli.py
│     └─ service.py
├─ scripts/
│  ├─ run_round.py
│  ├─ backfill_completed_rounds.py
│  └─ train_meta_model.py
├─ data/
│  ├─ raw/
│  ├─ observations/
│  ├─ rounds/
│  └─ models/
└─ tests/
   ├─ test_api_client.py
   ├─ test_features.py
   ├─ test_query_planner.py
   ├─ test_predictor.py
   └─ test_submission.py

Requisitos:
- Usa Python 3.11+.
- Añade `requirements.txt` con:
  numpy
  pandas
  pydantic
  requests
  scipy
  scikit-learn
  lightgbm
  catboost
  fastapi
  uvicorn[standard]
  python-dotenv
  pytest
- Añade `pyproject.toml` mínimo para proyecto Python instalable.
- Añade `.env.example` con:
  AINM_TOKEN=
  AINM_BASE_URL=https://api.ainm.no
- Añade un README corto explicando el propósito del proyecto.
- No implementes todavía lógica compleja. Solo scaffolding limpio y consistente.
- Usa imports absolutos dentro del paquete `astar_island`.

Devuélveme el contenido completo de los archivos creados o modificados.
________________________________________
Prompt 1 — src/astar_island/constants.py
Implementa el archivo `src/astar_island/constants.py`.

Contexto:
Astar Island tiene terrain codes:
10 Ocean
11 Plains
0 Empty
1 Settlement
2 Port
3 Ruin
4 Forest
5 Mountain

Pero la predicción final usa 6 clases:
0 Empty
1 Settlement
2 Port
3 Ruin
4 Forest
5 Mountain

Necesito:
1. Constantes con nombres claros para terrain y class ids.
2. Diccionario `TERRAIN_TO_CLASS`.
3. Lista o tupla `CLASS_NAMES`.
4. Lista o tupla `TERRAIN_NAMES`.
5. Constantes de vecinos `NEI4` y `NEI8`.
6. Un pequeño helper `is_static_terrain_code(value: int) -> bool` que devuelva True para mountain, ocean y forest.
7. Un helper `is_land_code(value: int) -> bool` que considere no transitable solo ocean y mountain si tiene sentido según el baseline inicial.
8. Un helper `terrain_to_class(value: int) -> int`.

Requisitos:
- Código limpio, tipado y con docstrings breves.
- Sin dependencias externas.
- Devuélveme el contenido completo del archivo.
________________________________________
Prompt 2 — src/astar_island/config.py
Implementa `src/astar_island/config.py`.

Quiero una configuración simple basada en variables de entorno usando `os.getenv`, sin meter todavía pydantic-settings.

Debe incluir:
- `AINM_TOKEN`
- `AINM_BASE_URL`
- `DEFAULT_VIEWPORT_W = 15`
- `DEFAULT_VIEWPORT_H = 15`
- `MAX_QUERIES_PER_ROUND = 50`
- `DEFAULT_ALPHA_SMOOTHING = 0.5`
- `DEFAULT_PROB_FLOOR = 0.01`
- `DEFAULT_REQUEST_TIMEOUT = 30`
- `SIMULATE_TIMEOUT = 60`

También:
- función `require_token() -> str` que lance `ValueError` si no existe token
- función `get_base_url() -> str`
- función `get_default_viewport() -> tuple[int, int]`

Requisitos:
- Código muy simple y robusto.
- Tipado completo.
- Docstrings breves.
- Devuélveme el archivo completo.
________________________________________
Prompt 3 — src/astar_island/models.py
Implementa `src/astar_island/models.py` con modelos Pydantic para la API de Astar Island.

Quiero modelos suficientemente prácticos para:
- detalle de round
- initial states
- settlements iniciales
- resultados de simulate
- settlements observados en simulate
- submit response básico
- budget response básico

Incluye al menos:
- `InitialSettlement`
- `InitialState`
- `RoundSummary`
- `RoundDetail`
- `SimSettlement`
- `ViewportInfo`
- `SimResult`
- `BudgetInfo`
- `SubmitResponse`

Detalles:
- `InitialState` debe contener `grid` y `settlements`
- `SimResult` debe contener `grid`, `settlements`, `viewport`, `width`, `height`, `queries_used`, `queries_max`
- `SimSettlement` debe incluir `x`, `y`, `population`, `food`, `wealth`, `defense`, `has_port`, `alive`, `owner_id`

Requisitos:
- Pydantic v2 style (`model_validate` compatible)
- Tipado correcto
- Nada de lógica compleja todavía
- Añade helpers pequeños si aportan claridad, por ejemplo métodos `shape()` o similares, pero sin sobreingeniería
- Devuélveme el archivo completo
________________________________________
Prompt 4 — src/astar_island/api_client.py
Implementa `src/astar_island/api_client.py` con un cliente HTTP para la API de Astar Island.

Quiero una clase `AstarClient` con:
- constructor `__init__(token: str, base_url: str | None = None)`
- autenticación Bearer
- sesión `requests.Session()`
- timeouts razonables
- manejo básico de errores con `raise_for_status()`

Métodos:
- `get_rounds() -> list[dict]`
- `get_active_round() -> dict | None`
- `get_round_detail(round_id: str) -> RoundDetail`
- `get_budget() -> BudgetInfo | dict`
- `simulate(round_id: str, seed_index: int, x: int, y: int, w: int = 15, h: int = 15) -> SimResult`
- `submit(round_id: str, seed_index: int, prediction: list[list[list[float]]]) -> dict`
- `my_rounds() -> list[dict]`
- `analysis(round_id: str, seed_index: int) -> dict`

Endpoints esperados:
- /astar-island/rounds
- /astar-island/rounds/{round_id}
- /astar-island/budget
- /astar-island/simulate
- /astar-island/submit
- /astar-island/my-rounds
- /astar-island/analysis/{round_id}/{seed_index}

Requisitos:
- Importa configuración desde `config.py`.
- Importa modelos desde `models.py`.
- Añade helper privado `_url(path: str) -> str`.
- Añade helper privado `_post_json` y `_get_json` si simplifica el código.
- Código limpio, práctico y sin abstracciones innecesarias.
- Devuélveme el archivo completo.
________________________________________
Prompt 5 — src/astar_island/features.py
Implementa `src/astar_island/features.py`.

Objetivo:
extraer features del mapa inicial de cada seed sin gastar queries.

Necesito:
1. helper `in_bounds(h, w, y, x) -> bool`
2. helper `is_coast(grid: np.ndarray, y: int, x: int) -> bool`
3. helper `multi_source_distance(mask: np.ndarray) -> np.ndarray` usando BFS
4. función principal:
   `extract_map_features(grid: list[list[int]], settlements: list[dict]) -> dict[str, np.ndarray]`

La función principal debe calcular al menos:
- `terrain`
- `ocean_mask`
- `mountain_mask`
- `forest_mask`
- `coast_mask`
- `settlement_mask`
- `port_mask`
- `dist_settlement`
- `dist_coast`
- `adjacent_forests`
- `adjacent_mountains`
- `frontier_score`

Reglas del baseline:
- Ocean y mountain son no transitables
- Coast = celda de tierra con vecino oceánico cardinal
- `frontier_score` debe favorecer tierra no montañosa/no oceánica cercana a asentamientos pero aún abierta a expansión

Requisitos:
- Usa numpy
- Usa vecinos desde `constants.py`
- Código muy legible
- Añade docstrings breves
- Devuélveme el archivo completo
________________________________________
Prompt 6 — src/astar_island/observation_store.py
Implementa `src/astar_island/observation_store.py`.

Objetivo:
guardar observaciones repetidas por seed y por celda, porque varias llamadas a `simulate` sobre una misma ventana generan muestras estocásticas útiles.

Quiero:
1. una dataclass `CellObs`
   - `counts: np.ndarray` de tamaño 6
   - `n: int`
   - método `update(cls_idx: int) -> None`
   - método `probs(alpha: float = 0.5) -> np.ndarray`

2. una clase `ObservationStore`
   - constructor `(width: int, height: int, seeds_count: int)`
   - estructura para observaciones por celda y por seed
   - estructura para muestras de settlements observados

Métodos:
- `add_sim_result(seed_index: int, sim_result, terrain_to_class_map: dict[int, int]) -> None`
- `empirical_cell_probs(seed_index: int, y: int, x: int) -> np.ndarray | None`
- `cell_count(seed_index: int, y: int, x: int) -> int`
- `observed_fraction(seed_index: int) -> float`

Además:
- guarda samples de settlements en algo como `self.settlement_samples[(seed_index, x, y)]`
- cada sample debe guardar al menos:
  population, food, wealth, defense, has_port, alive, owner_id

Requisitos:
- usa dataclasses
- usa numpy
- robusto y simple
- compatible con `SimResult` ya definido
- Devuélveme el archivo completo
________________________________________
Prompt 7 — src/astar_island/query_planner.py
Implementa `src/astar_island/query_planner.py`.

Objetivo:
planificar el uso de las 50 queries de forma estratégica, priorizando ventanas informativas, no cobertura uniforme.

Quiero:
1. dataclass `WindowPlan` con:
   - seed_index
   - x
   - y
   - w
   - h
   - tag
   - repeats

2. helper `clamp_window(...)`
3. función `score_window(feats, x, y, w=15, h=15) -> float`
4. función `best_windows_for_seed(feats, width, height, stride=5) -> list[tuple[float, int, int]]`
5. función principal `initial_query_plan(all_seed_features, width, height) -> list[WindowPlan]`

La score de una ventana debe favorecer:
- densidad de settlements iniciales
- puertos iniciales
- costa
- frontier_score
- algo de bosque
y penalizar:
- exceso de montaña

Estrategia inicial:
- por cada seed selecciona al menos:
  - una ventana `anchor`
  - una ventana `frontier`
- ambas con `repeats=3` por defecto si caben en el presupuesto

Requisitos:
- evita coordenadas fuera del mapa
- código claro
- no optimices todavía presupuesto dinámico avanzado
- Devuélveme el archivo completo
________________________________________
Prompt 8 — src/astar_island/regime_estimator.py
Implementa `src/astar_island/regime_estimator.py`.

Objetivo:
estimar un "régimen del round" a partir de observaciones agregadas de settlements, porque los parámetros ocultos son comunes a todas las seeds del round.

Quiero una clase `RoundRegimeEstimator` con método:
- `estimate(observation_store) -> dict[str, float]`

Debe devolver un dict con scores en [0, 1]:
- `expansion`
- `aggression`
- `trade`
- `harsh_winter`
- `reclamation`

Hazlo primero con proxies heurísticos basados en muestras de settlements observados:
- population
- food
- wealth
- defense
- owner_id variability
- has_port
- alive

Incluye helpers internos simples, por ejemplo `safe_mean`.

No quiero un modelo bayesiano ni ML todavía. Solo heurística razonable, estable y fácil de ajustar.

Requisitos:
- siempre clamp a [0, 1]
- que no rompa si hay pocas muestras
- código limpio y fácil de tunear
- Devuélveme el archivo completo
________________________________________
Prompt 9 — src/astar_island/predictor_baseline.py
Implementa `src/astar_island/predictor_baseline.py`.

Objetivo:
construir un baseline probabilístico serio para producir el tensor HxWx6 por seed.

Necesito:
1. helper `normalize_probs(p: np.ndarray, floor: float = 0.01) -> np.ndarray`
2. función `base_cell_prior(feats: dict[str, np.ndarray], y: int, x: int, regime: dict[str, float]) -> np.ndarray`
3. función `build_prediction(seed_features: dict[str, np.ndarray], regime: dict[str, float], observation_store, seed_index: int) -> np.ndarray`

Reglas del baseline:
- Mountain casi one-hot a clase mountain
- Ocean casi one-hot a clase empty
- Forest con prior alto a forest
- Initial settlement con masa en settlement / port / ruin según coast, trade, aggression y harsh_winter
- Initial port con prior alto a port
- Initial ruin con mezcla ruin / forest / settlement / port según reclamation y coast
- Plains / empty:
  - lejos de settlements => mostly empty
  - cerca de frontier y con expansion alta => más settlement
  - costa + trade alto => algo de port
  - aggression alta => algo de ruin

Mezcla final:
- si una celda fue observada, mezcla prior + empirical probs
- cuanto mayor sea `n` observaciones en la celda, mayor peso de empirical probs
- asegura normalización final

Requisitos:
- usa numpy
- floor mínimo configurable en normalize
- código claro
- docstrings breves
- Devuélveme el archivo completo
________________________________________
Prompt 10 — src/astar_island/submission.py
Implementa `src/astar_island/submission.py`.

Quiero:
1. `sanitize_prediction(pred: np.ndarray, floor: float = 0.01) -> np.ndarray`
2. `validate_prediction(pred: np.ndarray, height: int, width: int) -> None`
3. `submit_all_seeds(client, round_id: str, preds: list[np.ndarray]) -> list[dict]`

Reglas:
- shape exacto `(height, width, 6)`
- no negativos
- suma 1 por celda
- aplicar floor antes de normalizar
- `submit_all_seeds` debe enviar cada seed y devolver lista de respuestas

Además:
- añade helper pequeño opcional para convertir a lista JSON-safe
- mensajes de error claros

Requisitos:
- robusto
- sencillo
- Devuélveme el archivo completo
________________________________________
Prompt 11 — src/astar_island/cli.py
Implementa `src/astar_island/cli.py`.

Quiero una CLI mínima con `argparse` para:
- `active-round`
- `budget`
- `simulate`
- `solve-round`

Comportamiento:
- `active-round` imprime round activo
- `budget` imprime presupuesto
- `simulate` acepta:
  --round-id
  --seed-index
  --x
  --y
  --w
  --h
- `solve-round` ejecuta el pipeline baseline completo del round activo:
  - obtener round activo
  - cargar initial states
  - extraer features por seed
  - crear plan inicial de queries
  - ejecutar queries
  - llenar ObservationStore
  - estimar régimen
  - construir predicciones
  - validar y enviar submissions

Requisitos:
- usa `AstarClient`
- usa `require_token()`
- no sobrecargues el archivo con demasiada lógica; si hace falta, crea funciones auxiliares dentro del mismo archivo
- salida por consola clara
- Devuélveme el archivo completo
________________________________________
Prompt 12 — scripts/run_round.py
Implementa `scripts/run_round.py` como script ejecutable simple que invoque el pipeline baseline completo.

Quiero:
- obtener token desde entorno
- obtener round activo
- cargar round detail
- extraer features
- generar plan de queries
- ejecutar `simulate` según `repeats`
- acumular observaciones
- estimar régimen
- construir predicciones para todas las seeds
- validar
- enviar submissions

Además:
- imprime progreso claro por consola
- imprime resumen final con round_id y seeds enviadas
- maneja caso `no active round`

Requisitos:
- reutiliza las funciones ya implementadas
- no dupliques lógica innecesaria si puedes importar desde `cli.py`
- Devuélveme el archivo completo
________________________________________
Prompt 13 — tests/test_features.py
Implementa `tests/test_features.py`.

Quiero tests unitarios básicos para:
- `in_bounds`
- `is_coast`
- `multi_source_distance`
- `extract_map_features`

Crea grids pequeños artificiales y fáciles de entender.

Valida al menos:
- que coast detecta tierra junto a ocean
- que mountain_mask y forest_mask se calculan bien
- que dist_settlement tiene 0 en un settlement inicial
- que frontier_score devuelve array con shape correcto

Requisitos:
- usa pytest
- tests pequeños, claros y deterministas
- Devuélveme el archivo completo
________________________________________
Prompt 14 — tests/test_submission.py
Implementa `tests/test_submission.py`.

Quiero tests para:
- `sanitize_prediction`
- `validate_prediction`

Valida:
- floor mínimo aplicado
- renormalización correcta
- shape incorrecto lanza ValueError
- probabilidades negativas lanzan ValueError
- sumas distintas de 1 lanzan ValueError

Usa arrays pequeños creados con numpy.

Requisitos:
- pytest
- tests concisos y claros
- Devuélveme el archivo completo
________________________________________
Prompt 15 — tests/test_query_planner.py
Implementa `tests/test_query_planner.py`.

Quiero tests básicos para:
- `clamp_window`
- `score_window`
- `best_windows_for_seed`
- `initial_query_plan`

Crea features sintéticos pequeños usando numpy para comprobar:
- que las ventanas no se salen del mapa
- que una ventana con settlements/costa puntúa mejor que una vacía
- que `initial_query_plan` devuelve al menos una ventana anchor y una frontier por seed cuando sea posible

Requisitos:
- pytest
- tests simples
- Devuélveme el archivo completo
________________________________________
Prompt 16 — tests/test_predictor.py
Implementa `tests/test_predictor.py`.

Quiero tests básicos para:
- `normalize_probs`
- `base_cell_prior`
- `build_prediction`

Valida al menos:
- que `normalize_probs` devuelve suma 1
- que mountain produce probabilidad dominante en class mountain
- que ocean produce masa dominante en class empty
- que `build_prediction` devuelve array `(H, W, 6)`
- que cuando hay observación empírica en una celda, la predicción sigue estando normalizada

Puedes usar un `ObservationStore` pequeño y fake data sencilla.

Requisitos:
- pytest
- tests cortos y robustos
- Devuélveme el archivo completo
________________________________________
Prompt 17 — Refactor final del pipeline
Este prompt es para usarlo cuando ya tengas los archivos anteriores implementados.
Revisa el pipeline completo de Astar Island que acabamos de implementar.

Objetivo:
hacer una pasada de refactor ligero y saneamiento sin cambiar la arquitectura.

Quiero que:
1. revises imports rotos o redundantes
2. unifiques nombres inconsistentes
3. añadas type hints faltantes
4. elimines duplicación obvia
5. mejores mensajes de error
6. compruebes que `scripts/run_round.py` y `src/astar_island/cli.py` funcionan con la estructura actual
7. no introduzcas dependencias nuevas
8. no cambies el comportamiento funcional principal

Además:
- si detectas funciones demasiado largas, divídelas con criterio
- mantén el código orientado a baseline competitivo, no sobreingenierizado
- devuelve los archivos modificados completos
________________________________________
Prompt extra — capa 2 de mejora competitiva
Cuando el baseline ya funcione end-to-end, pega esto:
Ahora quiero una mejora competitiva de segunda capa para Astar Island sin romper el baseline actual.

Objetivo:
mejorar el uso de queries y la mezcla probabilística usando observaciones repetidas y un query planner adaptativo.

Haz lo siguiente:
1. mejora `query_planner.py` para permitir una segunda fase adaptativa basada en:
   - observed_fraction por seed
   - incertidumbre local
   - ventanas con alta mezcla entre settlement/port/ruin
2. añade en `observation_store.py` helpers para estimar incertidumbre por ventana
3. mejora `predictor_baseline.py` para:
   - reforzar celdas observadas repetidamente
   - suavizar mejor la transición settlement/port/ruin
   - aprovechar mejor coast + trade + aggression + reclamation
4. mantén compatibilidad con la CLI y el runner
5. no metas todavía LightGBM ni entrenamiento histórico

Quiero cambios prácticos, medibles y fáciles de testear.
Devuélveme los archivos modificados completos.
________________________________________
Prompt extra — capa 3 con aprendizaje de rounds anteriores
Cuando ya tengáis submissions funcionales y acceso a /analysis, pega esto:
Quiero añadir una capa de aprendizaje supervisado para Astar Island usando rounds históricos ya terminados.

Implementa una primera versión simple y robusta de esta funcionalidad:

1. `scripts/backfill_completed_rounds.py`
   - leer `my_rounds`
   - identificar rounds completed
   - para cada round y seed, llamar a `analysis(round_id, seed_index)`
   - guardar JSONs crudos en `data/rounds/`

2. `src/astar_island/predictor_round_model.py`
   - utilities para construir dataset tabular por celda
   - features:
     terrain inicial, coast, forest, mountain, dist_settlement, dist_coast, adjacent_forests, frontier_score, regime scores, empirical probs si existen
   - target:
     vector de 6 probabilidades ground truth por celda

3. `scripts/train_meta_model.py`
   - cargar dataset histórico
   - entrenar un modelo base tipo `MultiOutputRegressor(LGBMRegressor(...))`
   - guardar modelo en `data/models/`

4. deja preparado un hook en el pipeline para combinar:
   baseline heuristic + historical model

Requisitos:
- primera versión simple
- sin sobreingeniería
- con código claro y comentarios breves
- devuelve los archivos completos creados o modificados
________________________________________
Orden exacto que te recomiendo usar hoy
Pega estos prompts en este orden:
1.	Prompt 0
2.	Prompt 1
3.	Prompt 2
4.	Prompt 3
5.	Prompt 4
6.	Prompt 5
7.	Prompt 6
8.	Prompt 7
9.	Prompt 8
10.	Prompt 9
11.	Prompt 10
12.	Prompt 11
13.	Prompt 12
14.	Prompt 13–16
15.	Prompt 17
Y después, cuando eso funcione, pasas a los dos prompts extra.
________________________________________
Consejo operativo para Cursor AI
Cuando pegues cada prompt, añade esta línea al final para que Cursor no se disperse:
No cambies archivos no relacionados salvo que sea estrictamente necesario para que compile.
Y cuando quieras máxima precisión en una sola pieza, añade:
Antes de escribir código, inspecciona primero los archivos existentes relacionados y respeta su estilo e imports.
Si quieres, en el siguiente mensaje te lo convierto en un Pack 2 todavía más práctico, con:
•	prompts ya agrupados por tandas de trabajo de 20–30 minutos,
•	comandos de terminal para probar cada fase,
•	y checklist de validación para cada bloque antes de pasar al siguiente.
Ignacio: puedes convertirlo en un Pack 2 todavía más práctico, con:
•	prompts ya agrupados por tandas de trabajo de 20–30 minutos,
•	comandos de terminal para probar cada fase,
y checklist de validación para cada bloque antes de pasar al siguiente
...
Sí. Aquí tienes el Pack 2, pensado para trabajar en bloques de 20–30 minutos en Cursor AI, con un flujo muy operativo.
La lógica es esta:
•	cada tanda tiene un objetivo pequeño y cerrable,
•	en cada tanda pegas solo los prompts indicados del Pack 1,
•	luego ejecutas unos comandos concretos en terminal,
•	y no pasas a la siguiente hasta cumplir la checklist de validación.
Voy a asumir Windows + PowerShell, que es lo que sueles usar.
________________________________________
Pack 2 — Astar Island en tandas de 20–30 minutos
Regla general para todas las tandas
Añade siempre al final del prompt en Cursor estas dos líneas:
Antes de escribir código, inspecciona primero los archivos existentes relacionados y respeta su estilo e imports.
No cambies archivos no relacionados salvo que sea estrictamente necesario para que compile.
________________________________________
Tanda 0 — Preparación del entorno (15–20 min)
Objetivo
Dejar el proyecto listo para iterar rápido.
Prompts a pegar en Cursor
Todavía ninguno, salvo que quieras que Cursor te cree el scaffold directamente con el Prompt 0.
Terminal PowerShell
Desde la carpeta raíz del proyecto:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
Si ya existe requirements.txt:
pip install -r requirements.txt
Si todavía no existe, no pasa nada: lo instalarás después de la Tanda 1.
Crea .env a partir del ejemplo cuando exista:
Copy-Item .env.example .env
Checklist de validación
•	existe la carpeta raíz del proyecto
•	puedes activar .venv
•	python --version funciona
•	pip --version funciona
No avanzar si
•	la venv no activa
•	PowerShell bloquea scripts y no puedes activar .venv
Si PowerShell bloquea activación:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
________________________________________
Tanda 1 — Scaffolding + constantes + config (20–30 min)
Objetivo
Tener la base del proyecto y los primeros archivos estables.
Prompts a pegar en Cursor
En este orden:
1.	Prompt 0 — Scaffolding inicial
2.	Prompt 1 — constants.py
3.	Prompt 2 — config.py
Terminal PowerShell
Instala dependencias si el scaffold ya creó requirements.txt:
pip install -r requirements.txt
Comprueba estructura e imports básicos:
Get-ChildItem -Recurse
python -c "from astar_island.constants import TERRAIN_TO_CLASS; print(TERRAIN_TO_CLASS)"
python -c "from astar_island.config import get_base_url, get_default_viewport; print(get_base_url(), get_default_viewport())"
Si el paquete no se resuelve, desde la raíz del proyecto:
$env:PYTHONPATH = "src"
python -c "from astar_island.constants import CLASS_NAMES; print(CLASS_NAMES)"
Checklist de validación
•	existe src/astar_island/
•	requirements.txt existe
•	constants.py importa sin error
•	config.py importa sin error
•	TERRAIN_TO_CLASS tiene los mappings correctos
•	get_default_viewport() devuelve (15, 15)
No avanzar si
•	hay imports rotos
•	src no es visible en terminal
•	constants.py mezcla terrain ids y class ids incorrectamente
________________________________________
Tanda 2 — Modelos y cliente API (20–30 min)
Objetivo
Poder hablar con la API y modelar round/simulate de forma limpia.
Prompts a pegar en Cursor
En este orden:
1.	Prompt 3 — models.py
2.	Prompt 4 — api_client.py
Terminal PowerShell
Primero prueba imports:
$env:PYTHONPATH = "src"
python -c "from astar_island.models import RoundDetail, SimResult; print('models ok')"
python -c "from astar_island.api_client import AstarClient; print('client ok')"
Si ya tienes token en .env, puedes probar algo mínimo desde terminal o luego desde CLI:
$env:AINM_TOKEN = "TU_TOKEN_AQUI"
$env:AINM_BASE_URL = "https://api.ainm.no"
python -c "from astar_island.api_client import AstarClient; c=AstarClient(token='$env:AINM_TOKEN'); print(type(c.get_rounds()).__name__)"
Checklist de validación
•	RoundDetail, SimResult, BudgetInfo y modelos relacionados importan bien
•	AstarClient se instancia sin error
•	get_rounds() devuelve lista si el token es válido
•	los endpoints están bien construidos bajo /astar-island/...
No avanzar si
•	model_validate falla por estructura rara
•	el cliente usa rutas incorrectas
•	la autenticación Bearer no está puesta
________________________________________
Tanda 3 — Features del mapa inicial (20–30 min)
Objetivo
Extraer información útil del mapa sin gastar queries.
Prompts a pegar en Cursor
Solo:
1.	Prompt 5 — features.py
Terminal PowerShell
Prueba imports:
$env:PYTHONPATH = "src"
python -c "from astar_island.features import in_bounds, multi_source_distance, extract_map_features; print('features ok')"
Si quieres una prueba rápida manual:
python - <<'PY'
import numpy as np
from astar_island.features import extract_map_features

grid = [
    [10,10,10,10],
    [10,11, 1,10],
    [10, 4,11,10],
    [10,10,10,10],
]
settlements = [{"x":2,"y":1,"has_port":False}]
f = extract_map_features(grid, settlements)
print(f["terrain"].shape)
print(f["coast_mask"])
print(f["dist_settlement"])
PY
Checklist de validación
•	extract_map_features devuelve un dict[str, np.ndarray]
•	están presentes:
o	terrain
o	coast_mask
o	dist_settlement
o	dist_coast
o	frontier_score
•	dist_settlement vale 0 en un settlement inicial
•	coast_mask detecta tierra junto a océano
No avanzar si
•	multi_source_distance no funciona
•	frontier_score sale con shape incorrecto
•	is_coast marca océano o montaña como costa
________________________________________
Tanda 4 — Observation store + query planner (20–30 min)
Objetivo
Poder almacenar observaciones repetidas y planificar queries útiles.
Prompts a pegar en Cursor
En este orden:
1.	Prompt 6 — observation_store.py
2.	Prompt 7 — query_planner.py
Terminal PowerShell
$env:PYTHONPATH = "src"
python -c "from astar_island.observation_store import ObservationStore; print('store ok')"
python -c "from astar_island.query_planner import WindowPlan, initial_query_plan; print('planner ok')"
Prueba rápida de planner:
python - <<'PY'
import numpy as np
from astar_island.query_planner import score_window

feats = {
    "settlement_mask": np.zeros((20,20), dtype=bool),
    "port_mask": np.zeros((20,20), dtype=bool),
    "coast_mask": np.zeros((20,20), dtype=bool),
    "forest_mask": np.zeros((20,20), dtype=bool),
    "mountain_mask": np.zeros((20,20), dtype=bool),
    "frontier_score": np.zeros((20,20), dtype=float),
}
feats["settlement_mask"][5:7,5:7] = True
feats["coast_mask"][5:10,5:10] = True
feats["frontier_score"][5:10,5:10] = 0.8
print(score_window(feats, 0, 0, 15, 15))
PY
Checklist de validación
•	CellObs.probs() devuelve vector de tamaño 6
•	ObservationStore puede guardar muestras sin error
•	initial_query_plan() devuelve ventanas con tag
•	aparecen al menos ventanas anchor y frontier
•	las ventanas no se salen del mapa
No avanzar si
•	add_sim_result no sabe mapear viewport
•	clamp_window falla en bordes
•	el planner genera coordenadas negativas o fuera de rango
________________________________________
Tanda 5 — Estimador de régimen + predictor baseline (25–30 min)
Objetivo
Construir el primer motor real de predicción probabilística.
Prompts a pegar en Cursor
En este orden:
1.	Prompt 8 — regime_estimator.py
2.	Prompt 9 — predictor_baseline.py
Terminal PowerShell
$env:PYTHONPATH = "src"
python -c "from astar_island.regime_estimator import RoundRegimeEstimator; print('regime ok')"
python -c "from astar_island.predictor_baseline import build_prediction; print('predictor ok')"
Prueba mínima de forma:
python - <<'PY'
import numpy as np
from astar_island.predictor_baseline import normalize_probs

p = np.array([0,0,0,1,0,0], dtype=float)
print(normalize_probs(p))
PY
Checklist de validación
•	estimate() devuelve exactamente:
o	expansion
o	aggression
o	trade
o	harsh_winter
o	reclamation
•	todos los valores están en [0,1]
•	build_prediction() devuelve tensor (H, W, 6)
•	cada celda sale normalizada
•	montaña tiene probabilidad dominante en mountain
•	océano tiene probabilidad dominante en empty
No avanzar si
•	aparecen NaN
•	el predictor genera probabilidades negativas
•	no se normaliza al final
________________________________________
Tanda 6 — Submission + CLI + runner (25–30 min)
Objetivo
Cerrar el flujo end-to-end: round activo → queries → predicción → submit.
Prompts a pegar en Cursor
En este orden:
1.	Prompt 10 — submission.py
2.	Prompt 11 — cli.py
3.	Prompt 12 — scripts/run_round.py
Terminal PowerShell
Prueba imports:
$env:PYTHONPATH = "src"
python -c "from astar_island.submission import validate_prediction; print('submission ok')"
python -c "from astar_island.cli import main; print('cli ok')"
Pruebas funcionales si ya tienes token:
$env:AINM_TOKEN = "TU_TOKEN_AQUI"
$env:AINM_BASE_URL = "https://api.ainm.no"
python -m astar_island.cli active-round
python -m astar_island.cli budget
Si solve-round ya está listo:
python -m astar_island.cli solve-round
O el runner:
python scripts/run_round.py
Checklist de validación
•	validate_prediction rechaza shape incorrecto
•	sanitize_prediction aplica floor y renormaliza
•	active-round funciona
•	budget funciona
•	simulate puede invocarse con parámetros
•	solve-round recorre el pipeline sin romper imports
No avanzar si
•	la CLI tiene demasiada lógica duplicada
•	run_round.py y cli.py hacen cosas distintas
•	submit_all_seeds() no devuelve respuestas por seed
________________________________________
Tanda 7 — Tests mínimos obligatorios (20–30 min)
Objetivo
Asegurar que el baseline no se rompe mientras iteráis.
Prompts a pegar en Cursor
En este orden:
1.	Prompt 13 — tests/test_features.py
2.	Prompt 14 — tests/test_submission.py
3.	Prompt 15 — tests/test_query_planner.py
4.	Prompt 16 — tests/test_predictor.py
Terminal PowerShell
Ejecuta primero todo:
$env:PYTHONPATH = "src"
pytest -q
Si falla, ve por bloques:
pytest tests/test_features.py -q
pytest tests/test_submission.py -q
pytest tests/test_query_planner.py -q
pytest tests/test_predictor.py -q
Checklist de validación
•	todos los tests pasan
•	no hay tests frágiles ni dependientes de red
•	los tests usan grids sintéticos simples
•	el predictor sigue devolviendo shape (H,W,6)
No avanzar si
•	los tests llaman a la API real
•	los tests fallan por imports o por path
•	los tests validan cosas ambiguas
________________________________________
Tanda 8 — Refactor corto y saneamiento (20–25 min)
Objetivo
Limpiar el baseline antes de meter mejoras competitivas.
Prompts a pegar en Cursor
Solo:
1.	Prompt 17 — Refactor final del pipeline
Terminal PowerShell
$env:PYTHONPATH = "src"
pytest -q
python -m astar_island.cli active-round
python -m astar_island.cli budget
Si quieres probar el solve otra vez:
python -m astar_island.cli solve-round
Checklist de validación
•	imports limpios
•	sin duplicación obvia
•	tipos coherentes
•	CLI y runner alineados
•	tests siguen pasando
No avanzar si
•	el refactor cambió comportamiento
•	el baseline ya no reproduce el flujo anterior
•	aparecen “fixes” no relacionados
________________________________________
Tanda 9 — Mejora competitiva capa 2 (20–30 min)
Objetivo
Subir nivel sin meter todavía ML histórico.
Prompt a pegar en Cursor
Solo:
1.	Prompt extra — capa 2 de mejora competitiva
Qué debe mejorar
•	planner adaptativo
•	mejor uso de repeated windows
•	mezcla probabilística más inteligente
•	más peso a celdas realmente observadas
Terminal PowerShell
$env:PYTHONPATH = "src"
pytest -q
python -m astar_island.cli solve-round
Checklist de validación
•	sigue funcionando el pipeline completo
•	el planner ya no es puramente estático
•	una celda observada varias veces pesa más
•	no se ha roto submission.py
•	no se ha introducido ML innecesario
________________________________________
Tanda 10 — Aprendizaje histórico capa 3 (25–30 min)
Objetivo
Empezar a capturar rounds terminados para crear ventaja acumulativa.
Prompt a pegar en Cursor
Solo:
1.	Prompt extra — capa 3 con aprendizaje de rounds anteriores
Terminal PowerShell
Backfill:
$env:PYTHONPATH = "src"
python scripts/backfill_completed_rounds.py
Entrenamiento:
python scripts/train_meta_model.py
Checklist de validación
•	se guardan JSONs crudos en data/rounds/
•	existe dataset tabular usable
•	el modelo se entrena y se guarda
•	el hook de combinación con baseline queda preparado
•	no se ha reemplazado aún el baseline heurístico, solo complementado
________________________________________
Comandos de terminal por fase, resumidos
Instalar entorno
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Hacer visible src
$env:PYTHONPATH = "src"
Ejecutar tests
pytest -q
pytest tests/test_features.py -q
pytest tests/test_submission.py -q
pytest tests/test_query_planner.py -q
pytest tests/test_predictor.py -q
Probar CLI
python -m astar_island.cli active-round
python -m astar_island.cli budget
python -m astar_island.cli simulate --round-id ROUND_ID --seed-index 0 --x 0 --y 0 --w 15 --h 15
python -m astar_island.cli solve-round
Ejecutar runner
python scripts/run_round.py
________________________________________
Checklist maestra antes de pasar del baseline a mejoras competitivas
No pases a la capa 2 ni a la capa 3 hasta que todo esto sea verdad:
•	el proyecto instala dependencias sin errores
•	constants.py, config.py, models.py, api_client.py importan bien
•	extract_map_features() funciona
•	ObservationStore acumula repeticiones
•	initial_query_plan() devuelve ventanas válidas
•	RoundRegimeEstimator().estimate() devuelve 5 scores en [0,1]
•	build_prediction() devuelve (H, W, 6)
•	validate_prediction() protege shape, negativos y sumas
•	solve-round recorre todo el pipeline
•	los tests básicos pasan
________________________________________
Estrategia práctica de trabajo en una tarde
Si hoy quieres avanzar rápido, yo haría este orden real:
Bloque A
Tandas 1 y 2
Resultado esperado: proyecto creado, cliente API funcional.
Bloque B
Tandas 3 y 4
Resultado esperado: features + planner + store funcionando.
Bloque C
Tandas 5 y 6
Resultado esperado: baseline completo con submit.
Bloque D
Tandas 7 y 8
Resultado esperado: estabilidad y limpieza.
Bloque E
Tandas 9 y 10
Resultado esperado: optimización competitiva y acumulación de datos históricos.
________________________________________
Mini checklist antes de pegar cada prompt en Cursor
Antes de cada prompt, haz esta comprobación mental:
•	este prompt toca solo 1–3 archivos
•	ya tengo la fase anterior estable
•	ya ejecuté los tests o comandos mínimos de la fase anterior
•	no estoy pidiendo a Cursor demasiadas cosas de golpe
________________________________________
Recomendación táctica para no perder tiempo
Cuando Cursor te devuelva demasiado código o mezcle cosas, dile esto:
Reduce el cambio al mínimo necesario. Mantén el diseño actual. No reestructures otros módulos.
Y cuando quieras que te lo deje más ejecutable, añade:
Al final, incluye también un breve bloque con “qué probar ahora en terminal”.
En el siguiente mensaje te puedo preparar un Pack 3 ultra operativo, con formato de plantilla diaria de ejecución, tipo:
•	tanda,
•	prompt usado,
•	comando lanzado,
•	resultado esperado,
•	resultado real,
•	siguiente decisión.
Ignacio:si gracias,  me puedes preparar un Pack 3 ultra operativo, con formato de plantilla diaria de ejecución, tipo:
•	tanda,
•	prompt usado,
•	comando lanzado,
•	resultado esperado,
•	resultado real,
•	siguiente decisión.
Sí. Aquí tienes el Pack 3 ultra operativo, pensado como plantilla diaria de ejecución para trabajar con Cursor AI sobre Astar Island.
La idea es que lo uses como bitácora de trabajo real: pegas el prompt en Cursor, ejecutas el comando en terminal, anotas qué esperabas, qué pasó de verdad y decides el siguiente paso sin perder contexto.
________________________________________
Pack 3 — Plantilla diaria de ejecución
Cómo usar este pack
Cada tanda debe ocupar 20–30 minutos y terminar con una decisión clara:
•	seguir
•	corregir
•	hacer rollback parcial
•	refactorizar antes de continuar
No avances a la siguiente tanda si el bloque actual no quedó estable.
________________________________________
1. Plantilla maestra diaria
Copia esto tal cual en tu documento de trabajo diario:
# Astar Island — Diario de ejecución

**Fecha:**  
**Objetivo del día:**  
**Tiempo disponible:**  
**Estado inicial del proyecto:**  
**Branch / commit inicial:**  

---

## Tanda 1
**Duración prevista:** 20–30 min  
**Objetivo:**  

**Prompt usado en Cursor:**  
```text
[Pegar aquí el prompt exacto]
Comando lanzado en terminal:
[Pegar aquí el comando exacto]
Resultado esperado:
•	
•	
Resultado real:
•	
•	
Problemas detectados:
•	
Siguiente decisión:
•	Continuar a la siguiente tanda
•	Corregir en esta misma tanda
•	Hacer mini-refactor
•	Volver al estado anterior
Notas rápidas:
•	
________________________________________
Tanda 2
Duración prevista: 20–30 min
Objetivo:
Prompt usado en Cursor:
[Pegar aquí el prompt exacto]
Comando lanzado en terminal:
[Pegar aquí el comando exacto]
Resultado esperado:
•	
•	
Resultado real:
•	
•	
Problemas detectados:
•	
Siguiente decisión:
•	Continuar a la siguiente tanda
•	Corregir en esta misma tanda
•	Hacer mini-refactor
•	Volver al estado anterior
Notas rápidas:
•	
________________________________________
Tanda 3
Duración prevista: 20–30 min
Objetivo:
Prompt usado en Cursor:
[Pegar aquí el prompt exacto]
Comando lanzado en terminal:
[Pegar aquí el comando exacto]
Resultado esperado:
•	
•	
Resultado real:
•	
•	
Problemas detectados:
•	
Siguiente decisión:
•	Continuar a la siguiente tanda
•	Corregir en esta misma tanda
•	Hacer mini-refactor
•	Volver al estado anterior
Notas rápidas:
•	
________________________________________
Resumen del día
Qué quedó terminado:
•	
•	
Qué quedó a medias:
•	
Bloqueos:
•	
Siguiente paso recomendado para mañana:
•	
Branch / commit final:

---

# 2. Plantilla compacta por tanda

Esto sirve cuando quieres llevar un control más rápido y no tan largo.

```markdown
## Tanda X
**Objetivo:**  
**Prompt usado:**  
**Comando lanzado:**  
**Resultado esperado:**  
**Resultado real:**  
**Siguiente decisión:**  
________________________________________
3. Plantilla diaria recomendada para Astar Island
Te la dejo ya adaptada al flujo que definimos.
________________________________________
Día 1 — Scaffolding + base técnica
Meta del día
Dejar listo el proyecto con:
•	scaffolding
•	constants
•	config
•	models
•	api_client
________________________________________
Tanda 1
## Tanda 1
**Objetivo:** Crear el scaffolding inicial del proyecto.

**Prompt usado:** Prompt 0 — Scaffolding inicial del proyecto

**Comando lanzado:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Resultado esperado:
•	Existe la estructura base del proyecto
•	Existe requirements.txt
•	Existe src/astar_island/
•	La venv funciona
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar con constants/config
•	Corregir estructura
•	Repetir scaffolding parcial

---

### Tanda 2
```markdown
## Tanda 2
**Objetivo:** Implementar `constants.py` y `config.py`.

**Prompt usado:** Prompt 1 + Prompt 2

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
python -c "from astar_island.constants import TERRAIN_TO_CLASS; print(TERRAIN_TO_CLASS)"
python -c "from astar_island.config import get_base_url, get_default_viewport; print(get_base_url(), get_default_viewport())"
Resultado esperado:
•	constants.py importa sin error
•	config.py importa sin error
•	viewport por defecto = (15, 15)
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar con models/api client
•	Corregir imports
•	Ajustar mappings

---

### Tanda 3
```markdown
## Tanda 3
**Objetivo:** Implementar `models.py` y `api_client.py`.

**Prompt usado:** Prompt 3 + Prompt 4

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
python -c "from astar_island.models import RoundDetail, SimResult; print('models ok')"
python -c "from astar_island.api_client import AstarClient; print('client ok')"
Resultado esperado:
•	Modelos Pydantic correctos
•	Cliente API importable
•	Endpoints bien construidos
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar al Día 2
•	Corregir modelos
•	Revisar autenticación o rutas

---

## Día 2 — Features + observation store + query planner

### Meta del día
Tener la capa de observación e inferencia espacial.

---

### Tanda 1
```markdown
## Tanda 1
**Objetivo:** Implementar `features.py`.

**Prompt usado:** Prompt 5

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
python -c "from astar_island.features import in_bounds, multi_source_distance, extract_map_features; print('features ok')"
Resultado esperado:
•	extract_map_features() funciona
•	coast_mask se calcula bien
•	dist_settlement y dist_coast se generan
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar con observation store
•	Corregir BFS/distancias
•	Revisar masks

---

### Tanda 2
```markdown
## Tanda 2
**Objetivo:** Implementar `observation_store.py`.

**Prompt usado:** Prompt 6

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
python -c "from astar_island.observation_store import ObservationStore; print('store ok')"
Resultado esperado:
•	CellObs guarda counts
•	ObservationStore inicializa bien
•	se pueden mezclar observaciones por seed y celda
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar con planner
•	Corregir estructura interna
•	Simplificar almacenamiento

---

### Tanda 3
```markdown
## Tanda 3
**Objetivo:** Implementar `query_planner.py`.

**Prompt usado:** Prompt 7

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
python -c "from astar_island.query_planner import WindowPlan, initial_query_plan; print('planner ok')"
Resultado esperado:
•	genera ventanas válidas
•	incluye anchor y frontier
•	no se sale del mapa
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar al Día 3
•	Ajustar scoring
•	Corregir clamp_window

---

## Día 3 — Régimen + predictor baseline

### Meta del día
Tener el primer motor predictivo completo.

---

### Tanda 1
```markdown
## Tanda 1
**Objetivo:** Implementar `regime_estimator.py`.

**Prompt usado:** Prompt 8

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
python -c "from astar_island.regime_estimator import RoundRegimeEstimator; print('regime ok')"
Resultado esperado:
•	devuelve 5 scores
•	todos están en [0,1]
•	no rompe con pocas muestras
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar con predictor
•	Ajustar proxies
•	Añadir clamps o defaults

---

### Tanda 2
```markdown
## Tanda 2
**Objetivo:** Implementar `predictor_baseline.py`.

**Prompt usado:** Prompt 9

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
python -c "from astar_island.predictor_baseline import build_prediction; print('predictor ok')"
Resultado esperado:
•	build_prediction() devuelve (H, W, 6)
•	cada celda queda normalizada
•	montaña y océano tienen priors coherentes
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar con tests rápidos
•	Corregir normalización
•	Ajustar mezcla prior/empirical

---

### Tanda 3
```markdown
## Tanda 3
**Objetivo:** Hacer una comprobación rápida de forma del predictor.

**Prompt usado:** Sin prompt nuevo, solo validación manual.

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
python - <<'PY'
import numpy as np
from astar_island.predictor_baseline import normalize_probs
p = np.array([0,0,0,1,0,0], dtype=float)
print(normalize_probs(p))
PY
Resultado esperado:
•	vector normalizado
•	sin NaN
•	sin negativos
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar al Día 4
•	Ajustar floor
•	Revisar estabilidad numérica

---

## Día 4 — Submission + CLI + runner

### Meta del día
Cerrar el pipeline end-to-end.

---

### Tanda 1
```markdown
## Tanda 1
**Objetivo:** Implementar `submission.py`.

**Prompt usado:** Prompt 10

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
python -c "from astar_island.submission import sanitize_prediction, validate_prediction; print('submission ok')"
Resultado esperado:
•	valida shape
•	aplica floor
•	renormaliza correctamente
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar con CLI
•	Corregir validación
•	Revisar manejo de errores

---

### Tanda 2
```markdown
## Tanda 2
**Objetivo:** Implementar `cli.py`.

**Prompt usado:** Prompt 11

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
python -m astar_island.cli active-round
python -m astar_island.cli budget
Resultado esperado:
•	CLI responde
•	active-round funciona
•	budget funciona
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar con runner
•	Corregir argparse/imports
•	Revisar token/base_url

---

### Tanda 3
```markdown
## Tanda 3
**Objetivo:** Implementar `scripts/run_round.py`.

**Prompt usado:** Prompt 12

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
python scripts/run_round.py
Resultado esperado:
•	encuentra round activo
•	ejecuta queries
•	genera predicciones
•	intenta enviar submissions
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar al Día 5
•	Corregir pipeline
•	Revisar flujo solve-round

---

## Día 5 — Tests + refactor corto

### Meta del día
Estabilizar el baseline.

---

### Tanda 1
```markdown
## Tanda 1
**Objetivo:** Crear tests básicos.

**Prompt usado:** Prompt 13 + Prompt 14 + Prompt 15 + Prompt 16

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
pytest -q
Resultado esperado:
•	tests pasan
•	no llaman a la API real
•	cubren features, submission, planner y predictor
Resultado real:
•	
•	
Siguiente decisión:
•	Continuar con refactor
•	Corregir tests
•	Corregir código productivo

---

### Tanda 2
```markdown
## Tanda 2
**Objetivo:** Refactor ligero y saneamiento.

**Prompt usado:** Prompt 17

**Comando lanzado:**
```powershell
$env:PYTHONPATH = "src"
pytest -q
python -m astar_island.cli active-round
python -m astar_island.cli budget
Resultado esperado:
•	imports limpios
•	sin duplicación obvia
•	tests siguen verdes
Resultado real:
•	
•	
Siguiente decisión:
•	Pasar a mejora competitiva capa 2
•	Hacer un último ajuste baseline
•	Congelar versión estable

---

# 4. Plantilla de ejecución rápida para una sola sesión

Si un día solo tienes 1 hora, usa esta versión:

```markdown
# Astar Island — Sesión rápida

**Fecha:**  
**Tiempo disponible:** 60 min  
**Objetivo único de la sesión:**  

## Tanda 1
**Prompt usado:**  
**Comando lanzado:**  
**Resultado esperado:**  
**Resultado real:**  
**Siguiente decisión:**  

## Tanda 2
**Prompt usado:**  
**Comando lanzado:**  
**Resultado esperado:**  
**Resultado real:**  
**Siguiente decisión:**  

## Tanda 3
**Prompt usado:**  
**Comando lanzado:**  
**Resultado esperado:**  
**Resultado real:**  
**Siguiente decisión:**  

## Cierre
**Qué quedó listo:**  
**Qué quedó pendiente:**  
**Primer paso de la próxima sesión:**  
________________________________________
5. Reglas de decisión para no perder tiempo
Usa esta tabla mental al final de cada tanda:
Continúa a la siguiente tanda si:
•	compila
•	importa
•	el comando principal funciona
•	no introdujo errores nuevos obvios
Corrige en la misma tanda si:
•	fallan imports
•	hay errores de shape
•	hay errores de normalización
•	el archivo nuevo rompe el flujo actual
Haz mini-refactor si:
•	Cursor duplicó lógica
•	hay nombres inconsistentes
•	la misma función hace demasiadas cosas
Haz rollback parcial si:
•	el cambio abrió demasiados archivos
•	mezcló arquitectura nueva innecesaria
•	rompió una parte ya estable
________________________________________
6. Bloque final de cierre diario
Úsalo siempre al terminar el día:
## Cierre diario

**Estado del baseline:**
- [ ] No arranca
- [ ] Arranca parcialmente
- [ ] Arranca y predice
- [ ] Arranca, predice y envía
- [ ] Arranca, predice, envía y está cubierto con tests

**Mayor avance del día:**
-  

**Mayor problema del día:**
-  

**Decisión para la próxima sesión:**
- [ ] Seguir implementación
- [ ] Corregir bugs
- [ ] Reforzar tests
- [ ] Empezar mejora competitiva capa 2
- [ ] Empezar capa 3 con aprendizaje histórico

**Primer prompt de la próxima sesión:**
```text
[Pegar aquí el siguiente prompt exacto]
Primer comando de la próxima sesión:
[Pegar aquí el primer comando exacto]

---

# 7. Recomendación práctica de uso con Cursor

En cada tanda, pega el prompt técnico y añade al final esto:

```text
Antes de escribir código, inspecciona primero los archivos existentes relacionados y respeta su estilo e imports.
No cambies archivos no relacionados salvo que sea estrictamente necesario para que compile.
Al final, incluye también un breve bloque con “qué probar ahora en terminal”.
Eso suele mejorar bastante el resultado de Cursor y reduce cambios innecesarios.
________________________________________
8. Pack 3, versión ultra compacta
Por si quieres una plantilla mínima para copiar muchas veces:
## Tanda X
**Objetivo:**  
**Prompt usado:**  
**Comando lanzado:**  
**Resultado esperado:**  
**Resultado real:**  
**Siguiente decisión:**  
________________________________________
Si quieres, te lo convierto en el siguiente mensaje en una versión ya maquetada para pegar directamente en Notion o Google Docs, con títulos limpios, separadores y casillas listas para uso diario.

Ignacio: si gracias,convierte en el siguiente mensaje en una versión ya maquetada para pegar directamente en Notion o Google Docs, con títulos limpios, separadores y casillas listas para uso diario.
Claro. Aquí tienes una versión maquetada y limpia, lista para copiar y pegar en Notion o Google Docs. Está pensada para uso diario real, con títulos claros, separadores y casillas listas para marcar.
________________________________________
ASTAR ISLAND — PLANTILLA DIARIA DE EJECUCIÓN
Datos de la sesión
Fecha: ______________________________________
Objetivo del día: ______________________________________
Tiempo disponible: ______________________________________
Estado inicial del proyecto: ______________________________________
Branch / commit inicial: ______________________________________
________________________________________
Resumen rápido del día
Meta principal de hoy
•	[ ]
Meta secundaria de hoy
•	[ ]
Quiero terminar hoy
•	[ ]
•	[ ]
•	[ ]
________________________________________
________________________________________
TANDA 1
Duración prevista: 20–30 min
Objetivo de la tanda:
________________________________________
________________________________________
Prompt usado en Cursor
[Pegar aquí el prompt exacto]
Comando lanzado en terminal
[Pegar aquí el comando exacto]
Resultado esperado
•	Compila
•	Importa correctamente
•	El archivo principal quedó creado/modificado
•	El comportamiento esperado aparece
•	No rompe módulos anteriores
Resultado real
•	[ ]
•	[ ]
•	[ ]
Problemas detectados
•	[ ]
•	[ ]
•	[ ]
Siguiente decisión
•	Continuar a la siguiente tanda
•	Corregir en esta misma tanda
•	Hacer mini-refactor
•	Hacer rollback parcial
•	Parar y revisar arquitectura
Notas rápidas
________________________________________
________________________________________
________________________________________
________________________________________
TANDA 2
Duración prevista: 20–30 min
Objetivo de la tanda:
________________________________________
________________________________________
Prompt usado en Cursor
[Pegar aquí el prompt exacto]
Comando lanzado en terminal
[Pegar aquí el comando exacto]
Resultado esperado
•	Compila
•	Importa correctamente
•	El archivo principal quedó creado/modificado
•	El comportamiento esperado aparece
•	No rompe módulos anteriores
Resultado real
•	[ ]
•	[ ]
•	[ ]
Problemas detectados
•	[ ]
•	[ ]
•	[ ]
Siguiente decisión
•	Continuar a la siguiente tanda
•	Corregir en esta misma tanda
•	Hacer mini-refactor
•	Hacer rollback parcial
•	Parar y revisar arquitectura
Notas rápidas
________________________________________
________________________________________
________________________________________
________________________________________
TANDA 3
Duración prevista: 20–30 min
Objetivo de la tanda:
________________________________________
________________________________________
Prompt usado en Cursor
[Pegar aquí el prompt exacto]
Comando lanzado en terminal
[Pegar aquí el comando exacto]
Resultado esperado
•	Compila
•	Importa correctamente
•	El archivo principal quedó creado/modificado
•	El comportamiento esperado aparece
•	No rompe módulos anteriores
Resultado real
•	[ ]
•	[ ]
•	[ ]
Problemas detectados
•	[ ]
•	[ ]
•	[ ]
Siguiente decisión
•	Continuar a la siguiente tanda
•	Corregir en esta misma tanda
•	Hacer mini-refactor
•	Hacer rollback parcial
•	Parar y revisar arquitectura
Notas rápidas
________________________________________
________________________________________
________________________________________
________________________________________
TANDA 4
Duración prevista: 20–30 min
Objetivo de la tanda:
________________________________________
________________________________________
Prompt usado en Cursor
[Pegar aquí el prompt exacto]
Comando lanzado en terminal
[Pegar aquí el comando exacto]
Resultado esperado
•	Compila
•	Importa correctamente
•	El archivo principal quedó creado/modificado
•	El comportamiento esperado aparece
•	No rompe módulos anteriores
Resultado real
•	[ ]
•	[ ]
•	[ ]
Problemas detectados
•	[ ]
•	[ ]
•	[ ]
Siguiente decisión
•	Continuar a la siguiente tanda
•	Corregir en esta misma tanda
•	Hacer mini-refactor
•	Hacer rollback parcial
•	Parar y revisar arquitectura
Notas rápidas
________________________________________
________________________________________
________________________________________
________________________________________
CHECKLIST GENERAL DEL DÍA
Estado técnico actual
•	La venv funciona
•	requirements.txt está actualizado
•	src es visible desde terminal
•	Los imports principales funcionan
•	No hay errores obvios de path
•	No hay errores obvios de typing
•	No hay errores obvios de shape
•	No hay errores obvios de normalización
•	El pipeline sigue coherente
Estado del baseline
•	Aún no arranca
•	Arranca parcialmente
•	Arranca y carga datos
•	Arranca y genera features
•	Arranca y planifica queries
•	Arranca y acumula observaciones
•	Arranca y predice
•	Arranca, predice y valida
•	Arranca, predice, valida y envía
•	Arranca, predice, envía y tiene tests básicos
________________________________________
RESUMEN DEL DÍA
Qué quedó terminado
•	[ ]
•	[ ]
•	[ ]
Qué quedó a medias
•	[ ]
•	[ ]
•	[ ]
Bloqueos encontrados
•	[ ]
•	[ ]
•	[ ]
Mayor avance del día
________________________________________
________________________________________
Mayor problema del día
________________________________________
________________________________________
Lección útil de hoy
________________________________________
________________________________________
________________________________________
SIGUIENTE PASO RECOMENDADO
Próxima acción
________________________________________
________________________________________
Primer prompt de la próxima sesión
[Pegar aquí el siguiente prompt exacto]
Primer comando de la próxima sesión
[Pegar aquí el siguiente comando exacto]
Decisión para la próxima sesión
•	Seguir implementación
•	Corregir bugs
•	Reforzar tests
•	Refactorizar
•	Empezar mejora competitiva capa 2
•	Empezar capa 3 con aprendizaje histórico
________________________________________
________________________________________
ASTAR ISLAND — PLANTILLA RÁPIDA DE UNA SOLA TANDA
Úsala cuando solo quieras registrar un bloque corto.
Tanda X
Objetivo:
________________________________________
Prompt usado en Cursor
[Pegar aquí el prompt exacto]
Comando lanzado en terminal
[Pegar aquí el comando exacto]
Resultado esperado
•	[ ]
•	[ ]
•	[ ]
Resultado real
•	[ ]
•	[ ]
•	[ ]
Siguiente decisión
•	Continuar
•	Corregir
•	Refactorizar
•	Rollback parcial
Notas
________________________________________
________________________________________
________________________________________
ASTAR ISLAND — PLANTILLA DE SESIÓN RÁPIDA DE 60 MINUTOS
Datos de la sesión
Fecha: ______________________________________
Tiempo disponible: 60 min
Objetivo único de la sesión:
________________________________________
________________________________________
Tanda 1
Prompt usado
[Pegar aquí el prompt exacto]
Comando lanzado
[Pegar aquí el comando exacto]
Resultado esperado
•	[ ]
•	[ ]
Resultado real
•	[ ]
•	[ ]
Siguiente decisión
•	Continuar
•	Corregir
________________________________________
Tanda 2
Prompt usado
[Pegar aquí el prompt exacto]
Comando lanzado
[Pegar aquí el comando exacto]
Resultado esperado
•	[ ]
•	[ ]
Resultado real
•	[ ]
•	[ ]
Siguiente decisión
•	Continuar
•	Corregir
________________________________________
Tanda 3
Prompt usado
[Pegar aquí el prompt exacto]
Comando lanzado
[Pegar aquí el comando exacto]
Resultado esperado
•	[ ]
•	[ ]
Resultado real
•	[ ]
•	[ ]
Siguiente decisión
•	Continuar
•	Corregir
________________________________________
Cierre de la sesión
Qué quedó listo
•	[ ]
•	[ ]
Qué quedó pendiente
•	[ ]
•	[ ]
Primer paso de la próxima sesión
________________________________________
________________________________________
REGLAS DE DECISIÓN RÁPIDA
Continúa si
•	Compila
•	Importa
•	El comando funciona
•	No rompió nada anterior
Corrige en la misma tanda si
•	Fallan imports
•	Fallan shapes
•	Fallan probabilidades
•	Hay errores de rutas
•	Se rompió el flujo anterior
Haz mini-refactor si
•	Cursor duplicó lógica
•	Los nombres quedaron inconsistentes
•	Una función hace demasiadas cosas
•	El archivo quedó difícil de mantener
Haz rollback parcial si
•	Se tocaron demasiados archivos
•	Apareció sobreingeniería innecesaria
•	Se rompió una parte ya estable
________________________________________
BLOQUE FIJO PARA PEGAR AL FINAL DE CADA PROMPT EN CURSOR
Antes de escribir código, inspecciona primero los archivos existentes relacionados y respeta su estilo e imports.
No cambies archivos no relacionados salvo que sea estrictamente necesario para que compile.
Al final, incluye también un breve bloque con “qué probar ahora en terminal”.
Si quieres, el siguiente paso útil es que te lo convierta también en una versión con las tandas de Día 1 a Día 5 ya pre-rellenadas, para que solo tengas que ir completando “resultado real” y “siguiente decisión”.
....





Ignacio Tejera
Mars 2026

