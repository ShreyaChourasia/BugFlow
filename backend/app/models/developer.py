from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Developer(Base):
    __tablename__ = "developers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    git_emails: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    capacity: Mapped[int] = mapped_column(Integer, default=5)
    current_queue_depth: Mapped[int] = mapped_column(Integer, default=0)
    skills: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
