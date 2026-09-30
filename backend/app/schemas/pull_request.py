from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PullRequestRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    repository_id: int
    number: int
    title: str
    status: str
    head_sha: str
    base_sha: str
    created_at: datetime
    check_status: str
    check_conclusion: str | None
    check_summary: str | None


class PullRequestRiskRead(BaseModel):
    probability: float
    calibrated_probability: float
    confidence: float
    risk_level: str
    explanation_text: str
    factors: list[dict]


class PullRequestDetailRead(PullRequestRead):
    comment_body: str | None
    risk: PullRequestRiskRead | None
