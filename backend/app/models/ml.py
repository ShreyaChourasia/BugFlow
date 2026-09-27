from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    task: Mapped[str] = mapped_column(String(100))
    version: Mapped[str] = mapped_column(String(100))
    mlflow_run_id: Mapped[str] = mapped_column(String(100))
    stage: Mapped[str] = mapped_column(String(20))  # champion / challenger / archived
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    data_version: Mapped[str] = mapped_column(String(100))
    seed: Mapped[int] = mapped_column()
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DriftAlert(Base):
    __tablename__ = "drift_alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("ml_models.id"))
    feature: Mapped[str] = mapped_column(String(200))
    psi: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
