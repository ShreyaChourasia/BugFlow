from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class DefectReportCreate(BaseModel):
    repository_id: int
    title: str
    description: str
    component: str | None = None


class DefectReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    repository_id: int
    reporter_id: int
    title: str
    description: str
    component: str | None
    status: str
    severity: str | None
    priority: str | None
    duplicate_of_id: int | None
    assignee_id: int | None
    reported_at: datetime
    resolved_at: datetime | None


class SuggestionItem(BaseModel):
    id: int
    title: str
    status: str


class SuggestionsRead(BaseModel):
    # C2: "rebuilding" is a distinct state from an empty `results` list — the
    # index genuinely has nothing indexed yet vs. isn't searchable right now.
    status: Literal["ready", "rebuilding"]
    results: list[SuggestionItem]


class DuplicateCandidate(BaseModel):
    id: int
    title: str
    status: str
    score: float
    shared_phrases: list[str]


class DuplicatesRead(BaseModel):
    status: Literal["ready", "rebuilding"]
    results: list[DuplicateCandidate]


class MergeRequest(BaseModel):
    original_id: int
