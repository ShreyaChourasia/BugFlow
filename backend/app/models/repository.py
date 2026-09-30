from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Repository(Base):
    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    url: Mapped[str] = mapped_column(String(500))
    issue_tracker_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    github_installation_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    default_branch: Mapped[str] = mapped_column(String(100), default="main")
    merge_blocking_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_threshold: Mapped[float] = mapped_column(Float, default=0.7)
    # US-13: line-level analysis only runs when calibrated risk clears this
    # same threshold; N (how many lines to highlight) is configurable here.
    # Not in the original §7 table — see docs/decisions/003-line-risk-top-n.md.
    line_risk_top_n: Mapped[int] = mapped_column(Integer, default=5)


class MiningRun(Base):
    __tablename__ = "mining_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    repository_id: Mapped[int] = mapped_column(ForeignKey("repositories.id"))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    checkpoint: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
