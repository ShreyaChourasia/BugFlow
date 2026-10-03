from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator


class ResolverCandidateRead(BaseModel):
    developer_id: int
    developer_name: str
    score: float
    confidence: float
    is_cold_start: bool
    reason: str
    current_open_defects: int
    capacity: int


class ResolverRecommendationsRead(BaseModel):
    # US-24 AC2: "no_confident_candidate" routes to the default triage owner
    # rather than showing a guess; "unavailable" means no model is trained yet.
    status: Literal["available", "no_confident_candidate", "unavailable"]
    candidates: list[ResolverCandidateRead]
    default_owner_id: int | None = None


class BatchAssignRequest(BaseModel):
    defect_ids: list[int]


class BatchAssignmentItem(BaseModel):
    defect_id: int
    developer_id: int
    suitability: float


class BatchAssignResult(BaseModel):
    assignments: list[BatchAssignmentItem]
    shortfall: list[int]
    total_suitability: float


class AssignmentOverrideRequest(BaseModel):
    developer_id: int
    reason: str

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        # US-29: "confirming without a reason is refused."
        if not value.strip():
            raise ValueError("A reason is required to override an assignment.")
        return value


class ObjectionRequest(BaseModel):
    reason: str


class WorkloadEntry(BaseModel):
    developer_id: int
    developer_name: str
    capacity: int
    current_queue_depth: int


class MyAssignmentRead(BaseModel):
    id: int
    defect_id: int
    defect_title: str
    source: str
    assigned_at: datetime
    reason: str
