from pydantic import BaseModel, ConfigDict


class SystemConfigUpsert(BaseModel):
    value: str


class SystemConfigRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    key: str
    value: str
