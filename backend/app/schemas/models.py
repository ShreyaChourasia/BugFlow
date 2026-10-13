from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

RETRAIN_GROUPS = [
    "commit_risk",
    "line_risk",
    "triage",
    "resolver_suitability",
    "resolution_forecast",
]


class MLModelRead(BaseModel):
    id: int
    task: str
    version: str
    stage: str
    metrics: dict[str, Any] | None
    data_version: str
    seed: int
    trained_at: datetime


class RetrainRequest(BaseModel):
    group: Literal[
        "commit_risk", "line_risk", "triage", "resolver_suitability", "resolution_forecast"
    ]
    seed: int = 42


class RetrainResponse(BaseModel):
    status: Literal["queued"]
    group: str


class ShadowComparisonRead(BaseModel):
    status: Literal["available", "unavailable"]
    champion_version: str | None = None
    challenger_version: str | None = None
    n_compared: int = 0
    mean_absolute_difference: float | None = None
    risk_level_agreement_rate: float | None = None
    mean_champion_probability: float | None = None
    mean_challenger_probability: float | None = None


class DriftAlertRead(BaseModel):
    id: int
    model_id: int
    feature: str
    psi: float
    severity: str
    created_at: datetime
    acknowledged: bool


class DriftCheckResult(BaseModel):
    status: Literal["available", "unavailable"]
    alerts: list[DriftAlertRead] = []
    checked_features: int = 0
