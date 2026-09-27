from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Explanation(Base):
    """Polymorphic: (decision_type, decision_id) points at the decision row
    being explained (a RiskPrediction, TriageAssessment, etc.), so no FK
    constraint — the target table varies by decision_type."""

    __tablename__ = "explanations"

    id: Mapped[int] = mapped_column(primary_key=True)
    decision_type: Mapped[str] = mapped_column(String(50))
    decision_id: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(String)
    factors: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Feedback(Base):
    """Polymorphic in the same way as Explanation."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    decision_type: Mapped[str] = mapped_column(String(50))
    decision_id: Mapped[int] = mapped_column(Integer)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    accepted: Mapped[bool] = mapped_column(Boolean)
    original_value: Mapped[str | None] = mapped_column(String, nullable=True)
    new_value: Mapped[str | None] = mapped_column(String, nullable=True)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
