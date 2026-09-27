from datetime import datetime

from pydantic import BaseModel, ConfigDict


class MiningRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    repository_id: int
    status: str
    checkpoint: dict | None
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
