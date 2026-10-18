from typing import Literal

from pydantic import BaseModel


class ForecastCurvePoint(BaseModel):
    day: float
    probability_unresolved: float


class ForecastRead(BaseModel):
    # US-31: never silently absent. "unavailable" = no trained model yet;
    # "insufficient_data" = this report has no severity/priority to forecast
    # from (not triaged, and the triage model abstained).
    status: Literal["available", "unavailable", "insufficient_data"]
    median_days: float | None = None
    p90_days: float | None = None
    curve: list[ForecastCurvePoint] | None = None
    confidence: float | None = None
    at_risk: bool = False
    estimate_text: str | None = None
