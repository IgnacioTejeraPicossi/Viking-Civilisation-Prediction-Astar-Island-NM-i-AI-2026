"""Pydantic models for Astar Island JSON payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InitialSettlement(BaseModel):
    x: int
    y: int
    has_port: bool
    alive: bool


class InitialState(BaseModel):
    grid: list[list[int]]
    settlements: list[InitialSettlement]


class RoundSummary(BaseModel):
    """Subset of fields from GET /rounds list."""

    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int = 5
    prediction_window_minutes: int | None = None
    started_at: str | None = None
    closes_at: str | None = None
    round_weight: float | None = None
    event_date: str | None = None


class RoundDetail(BaseModel):
    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: list[InitialState]


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


class ViewportInfo(BaseModel):
    x: int
    y: int
    w: int
    h: int


class SimResult(BaseModel):
    grid: list[list[int]]
    settlements: list[SimSettlement]
    viewport: ViewportInfo
    width: int
    height: int
    queries_used: int
    queries_max: int

    def map_shape(self) -> tuple[int, int]:
        """Full map (height, width) as returned by API."""
        return self.height, self.width


class BudgetInfo(BaseModel):
    round_id: str
    queries_used: int
    queries_max: int
    active: bool


class SubmitResponse(BaseModel):
    status: str
    round_id: str
    seed_index: int
    detail: Any | None = None
