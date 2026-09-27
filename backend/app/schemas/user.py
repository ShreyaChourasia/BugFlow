from pydantic import BaseModel, ConfigDict, EmailStr

from app.models.enums import Role


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: Role


class UserUpdate(BaseModel):
    role: Role | None = None
    is_active: bool | None = None


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str
    role: str
    is_active: bool
