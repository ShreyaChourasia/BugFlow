from datetime import datetime

from pydantic import BaseModel


class PredictCommitRequest(BaseModel):
    repository_id: int
    sha: str


class ExplanationRead(BaseModel):
    text: str
    factors: list[dict]


class RiskPredictionRead(BaseModel):
    id: int
    commit_id: int | None
    model_version: str
    probability: float
    calibrated_probability: float
    confidence: float
    risk_level: str
    latency_ms: float
    created_at: datetime
    explanation: ExplanationRead
