from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(50))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    developer_id: Mapped[int | None] = mapped_column(ForeignKey("developers.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
