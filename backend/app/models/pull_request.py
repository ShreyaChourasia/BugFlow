from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class PullRequest(Base):
    __tablename__ = "pull_requests"
    __table_args__ = (
        UniqueConstraint("repository_id", "number", name="uq_pull_requests_repository_number"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    repository_id: Mapped[int] = mapped_column(ForeignKey("repositories.id"))
    number: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(500))
    author_id: Mapped[int | None] = mapped_column(ForeignKey("developers.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="open")
    head_sha: Mapped[str] = mapped_column(String(40))
    base_sha: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    check_run_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    comment_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Phase 4: current write-back state, so the UI can show exactly what was
    # (or would have been) posted to GitHub without re-deriving it — the
    # "fake Checks sink" doubles as this in offline/replay mode. Not in the
    # original §7 table — see docs/decisions/002-pr-check-state-columns.md.
    check_status: Mapped[str] = mapped_column(String(20), default="pending")
    check_conclusion: Mapped[str | None] = mapped_column(String(20), nullable=True)
    check_summary: Mapped[str | None] = mapped_column(String, nullable=True)
    comment_body: Mapped[str | None] = mapped_column(String, nullable=True)


class RiskPrediction(Base):
    __tablename__ = "risk_predictions"
    __table_args__ = (
        CheckConstraint(
            "(commit_id IS NOT NULL) OR (pr_id IS NOT NULL)",
            name="risk_prediction_target_required",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    commit_id: Mapped[int | None] = mapped_column(ForeignKey("commits.id"), nullable=True)
    pr_id: Mapped[int | None] = mapped_column(ForeignKey("pull_requests.id"), nullable=True)
    model_version: Mapped[str] = mapped_column(String(100))
    probability: Mapped[float] = mapped_column(Float)
    calibrated_probability: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(20))
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LineRisk(Base):
    __tablename__ = "line_risks"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("risk_predictions.id"))
    file_path: Mapped[str] = mapped_column(String(1000))
    line_no: Mapped[int] = mapped_column(Integer)
    code: Mapped[str] = mapped_column(String)
    risk_score: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)
    marked_false_alarm: Mapped[bool] = mapped_column(Boolean, default=False)
