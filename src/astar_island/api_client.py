"""HTTP client for Astar Island API."""

from __future__ import annotations

import time
from typing import Any

import requests

from astar_island import config
from astar_island.models import BudgetInfo, RoundDetail, SimResult


def _timeout(read_seconds: int) -> tuple[int, int]:
    return (config.HTTP_CONNECT_TIMEOUT, read_seconds)


class AstarClient:
    """Bearer-authenticated session to api.ainm.no/astar-island."""

    def __init__(self, token: str | None = None, base_url: str | None = None) -> None:
        tok = (token or config.get_token()).strip()
        if not tok:
            raise ValueError("JWT token is required (AINM_TOKEN or constructor argument)")
        self.base_url = (base_url or config.get_base_url()).rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {tok}"

    def _url(self, path: str) -> str:
        p = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{p}"

    def _get_json(self, path: str, read_timeout: int | None = None) -> Any:
        rt = read_timeout if read_timeout is not None else config.DEFAULT_REQUEST_TIMEOUT
        r = self.session.get(self._url(path), timeout=_timeout(rt))
        r.raise_for_status()
        return r.json()

    def _post_json(self, path: str, payload: dict[str, Any], read_timeout: int | None = None) -> Any:
        rt = read_timeout if read_timeout is not None else config.DEFAULT_REQUEST_TIMEOUT
        r = self.session.post(self._url(path), json=payload, timeout=_timeout(rt))
        r.raise_for_status()
        return r.json()

    def _post_json_retry_429(
        self,
        path: str,
        payload: dict[str, Any],
        read_timeout: int,
        *,
        max_attempts: int = 25,
    ) -> Any:
        """
        POST with retries on HTTP 429.

        Note: API docs say 429 can mean **rate limit** OR **query budget exhausted** for /simulate.
        """
        rt = read_timeout
        last: requests.Response | None = None
        for attempt in range(max_attempts):
            r = self.session.post(self._url(path), json=payload, timeout=_timeout(rt))
            last = r
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                try:
                    wait = float(ra) if ra is not None else min(2.0 + attempt * 0.4, 30.0)
                except ValueError:
                    wait = min(2.0 + attempt * 0.4, 30.0)
                wait = min(max(wait, 0.6), 60.0)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()

        detail = (last.text[:300] + "…") if last and last.text and len(last.text) > 300 else (last.text if last else "")
        hint = (
            " For /simulate, HTTP 429 often means **no queries left** (50/50) or rate limit — "
            "check GET /budget before running."
        )
        raise requests.HTTPError(
            f"429 after {max_attempts} retries for {path}. {detail}{hint}",
            response=last,
        )

    def get_rounds(self) -> list[dict[str, Any]]:
        return self._get_json("/astar-island/rounds")

    def get_active_round(self) -> dict[str, Any] | None:
        rounds = self.get_rounds()
        return next((r for r in rounds if r.get("status") == "active"), None)

    def get_round_detail(self, round_id: str) -> RoundDetail:
        data = self._get_json(f"/astar-island/rounds/{round_id}")
        return RoundDetail.model_validate(data)

    def get_budget(self) -> BudgetInfo:
        data = self._get_json("/astar-island/budget")
        return BudgetInfo.model_validate(data)

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        x: int,
        y: int,
        w: int = config.DEFAULT_VIEWPORT_W,
        h: int = config.DEFAULT_VIEWPORT_H,
    ) -> SimResult:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": x,
            "viewport_y": y,
            "viewport_w": w,
            "viewport_h": h,
        }
        data = self._post_json_retry_429(
            "/astar-island/simulate", payload, read_timeout=config.SIMULATE_TIMEOUT
        )
        return SimResult.model_validate(data)

    def submit(self, round_id: str, seed_index: int, prediction: list[list[list[float]]]) -> dict[str, Any]:
        payload = {"round_id": round_id, "seed_index": seed_index, "prediction": prediction}
        return self._post_json_retry_429(
            "/astar-island/submit", payload, read_timeout=config.SIMULATE_TIMEOUT
        )

    def my_rounds(self) -> list[dict[str, Any]]:
        return self._get_json("/astar-island/my-rounds")

    def analysis(self, round_id: str, seed_index: int) -> dict[str, Any]:
        return self._get_json(f"/astar-island/analysis/{round_id}/{seed_index}", read_timeout=60)
