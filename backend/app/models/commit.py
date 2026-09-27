from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Commit(Base):
    __tablename__ = "commits"

    id: Mapped[int] = mapped_column(primary_key=True)
    sha: Mapped[str] = mapped_column(String(40), index=True)
    repository_id: Mapped[int] = mapped_column(ForeignKey("repositories.id"))
    author_id: Mapped[int | None] = mapped_column(ForeignKey("developers.id"), nullable=True)
    message: Mapped[str] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    lines_added: Mapped[int] = mapped_column(Integer, default=0)
    lines_deleted: Mapped[int] = mapped_column(Integer, default=0)
    files_changed: Mapped[int] = mapped_column(Integer, default=0)
    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_bug_inducing: Mapped[bool] = mapped_column(Boolean, default=False)
    is_fix: Mapped[bool] = mapped_column(Boolean, default=False)
    data_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
