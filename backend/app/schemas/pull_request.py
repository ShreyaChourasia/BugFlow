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


class LineRiskRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_path: str
    line_no: int
    code: str
    risk_score: float
    rank: int
    reason: str | None
    marked_false_alarm: bool


class PullRequestRiskRead(BaseModel):
    probability: float
    calibrated_probability: float
    confidence: float
    risk_level: str
    explanation_text: str
    factors: list[dict]
    # US-13: "not_applicable" (below threshold), "unavailable" (no model yet
    # or its diff couldn't be fetched), or "available" (see `lines`).
    line_risk_status: str
    lines: list[LineRiskRead]


class PullRequestDetailRead(PullRequestRead):
    comment_body: str | None
    risk: PullRequestRiskRead | None


class ReviewQueueItemRead(BaseModel):
    repository_id: int
    repository_name: str
    number: int
    title: str
    check_status: str
    calibrated_probability: float | None
    risk_level: str | None
    lines_added: int | None
    lines_deleted: int | None
    files_changed: int | None
