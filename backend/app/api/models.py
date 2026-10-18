from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.queue import get_queue
from app.core.security import require_role
from app.models.enums import Role
from app.schemas.models import (
    DriftCheckResult,
    MLModelRead,
    RetrainRequest,
    RetrainResponse,
    ShadowComparisonRead,
)
from app.services import drift_service, models_service
from app.services.training_service import export_experiment_record
from app.workers.jobs.retrain import retrain_job

router = APIRouter(
    prefix="/models", tags=["models"], dependencies=[Depends(require_role(Role.ML_ENGINEER))]
)


@router.get("", response_model=list[MLModelRead])
def list_models(task: str | None = None, db: Session = Depends(get_db)) -> list[MLModelRead]:
    """US-43/45: every trained model's version/stage/metrics."""
    return [
        MLModelRead.model_validate(m, from_attributes=True)
        for m in models_service.list_models(db, task)
    ]


@router.post("/retrain", response_model=RetrainResponse, status_code=status.HTTP_202_ACCEPTED)
def retrain(body: RetrainRequest) -> RetrainResponse:
    """ "Retrain from feedback plus new data" — dispatched to a worker job,
    off the request path, same as mining/scoring already are."""
    get_queue().enqueue(retrain_job, body.group, body.seed, job_timeout=1800)
    return RetrainResponse(status="queued", group=body.group)


@router.post("/{model_id}/rollback", response_model=MLModelRead)
def rollback(model_id: int, db: Session = Depends(get_db)) -> MLModelRead:
    """US-43: promote any previous version back to champion."""
    try:
        model = models_service.rollback(db, model_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return MLModelRead.model_validate(model, from_attributes=True)


@router.get("/{mlflow_run_id}/export")
def export_model(mlflow_run_id: str) -> dict[str, Any]:
    """US-44: a complete, self-contained experiment record."""
    return export_experiment_record(mlflow_run_id)


@router.get("/shadow-comparison", response_model=ShadowComparisonRead)
def shadow_comparison(db: Session = Depends(get_db)) -> ShadowComparisonRead:
    """US-45: champion vs. challenger on the same commits (commit-risk
    only — see docs/decisions/008)."""
    return models_service.get_shadow_comparison(db)


@router.post("/drift-check", response_model=DriftCheckResult)
def check_drift(db: Session = Depends(get_db)) -> DriftCheckResult:
    """US-41: runs the PSI drift check now (commit-risk only — see
    docs/decisions/008). The `drift-scheduler` compose service runs the
    same check on an interval, for the "scheduled" half of this story."""
    return drift_service.run_drift_check(db)


@router.get("/drift-alerts")
def list_drift_alerts(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    alerts = drift_service.list_drift_alerts(db)
    return [
        {
            "id": a.id,
            "model_id": a.model_id,
            "feature": a.feature,
            "psi": a.psi,
            "severity": a.severity,
            "created_at": a.created_at.isoformat(),
            "acknowledged": a.acknowledged,
        }
        for a in alerts
    ]
