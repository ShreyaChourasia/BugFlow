from pydantic import BaseModel, ConfigDict


class RepositoryCreate(BaseModel):
    name: str
    url: str
    issue_tracker_url: str | None = None
    default_branch: str = "main"


class RepositoryUpdate(BaseModel):
    merge_blocking_enabled: bool | None = None
    risk_threshold: float | None = None
    issue_tracker_url: str | None = None
    default_branch: str | None = None


class RepositoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    url: str
    issue_tracker_url: str | None
    default_branch: str
    merge_blocking_enabled: bool
    risk_threshold: float
