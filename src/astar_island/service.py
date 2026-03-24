"""Optional FastAPI wrapper for Cloud Run–style deployment."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from astar_island.runner import solve_active_round

app = FastAPI(title="Astar Island solver")


class SolveRequest(BaseModel):
    token: str = Field(..., description="AINM JWT (same as browser access_token)")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve")
def solve(req: SolveRequest) -> dict[str, object]:
    result = solve_active_round(req.token)
    return {"status": "completed", "result": result}
