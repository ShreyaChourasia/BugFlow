from typing import Literal

from pydantic import BaseModel


class ForecastCurvePoint(BaseModel):
    day: float
    probability_unresolved: float


class ForecastRead(BaseModel):
    # US-31: "unavailable" (no trained model yet) is distinct from a real
    # forecast — never silently absent.
    status: Literal["available", "unavailable"]
    median_days: float | None = None
    p90_days: float | None = None
    curve: list[ForecastCurvePoint] | None = None
    confidence: float | None = None
    at_risk: bool = False
    estimate_text: str | None = None
