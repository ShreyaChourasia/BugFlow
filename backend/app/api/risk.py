from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import get_current_user
from app.models.commit import Commit
from app.models.decision import Explanation
from app.models.pull_request import RiskPrediction
from app.models.user import User
from app.schemas.risk import ExplanationRead, PredictCommitRequest, RiskPredictionRead
from app.services.prediction_service import predict_commit

router = APIRouter(tags=["risk"])


def _to_read(
    risk_prediction: RiskPrediction, explanation: Explanation | None
) -> RiskPredictionRead:
    # Invariant: predict_commit always creates these together in one
    # transaction (C3 — no decision without an explanation), so this should
    # never be None; fail loudly if it somehow is rather than fake one up.
    assert explanation is not None, "RiskPrediction exists without its Explanation"
    return RiskPredictionRead(
        id=risk_prediction.id,
        commit_id=risk_prediction.commit_id,
        model_version=risk_prediction.model_version,
        probability=risk_prediction.probability,
        calibrated_probability=risk_prediction.calibrated_probability,
        confidence=risk_prediction.confidence,
        risk_level=risk_prediction.risk_level,
        latency_ms=risk_prediction.latency_ms,
        created_at=risk_prediction.created_at,
        explanation=ExplanationRead(
            text=explanation.text, factors=(explanation.factors or {}).get("items", [])
        ),
    )


def _get_commit_or_404(db: Session, repository_id: int, sha: str) -> Commit:
    commit = db.scalar(
        select(Commit).where(Commit.repository_id == repository_id, Commit.sha == sha)
    )
    if commit is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Commit not found")
    return commit


@router.post(
    "/predict/commit", response_model=RiskPredictionRead, status_code=status.HTTP_201_CREATED
)
def predict_commit_endpoint(
    body: PredictCommitRequest,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> RiskPredictionRead:
    commit = _get_commit_or_404(db, body.repository_id, body.sha)

    try:
        risk_prediction, explanation = predict_commit(db, commit)
    except LookupError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    return _to_read(risk_prediction, explanation)


@router.get("/commits/{sha}/risk", response_model=RiskPredictionRead)
def get_commit_risk(
    sha: str,
    repository_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> RiskPredictionRead:
    commit = _get_commit_or_404(db, repository_id, sha)

    risk_prediction = db.scalar(
        select(RiskPrediction)
        .where(RiskPrediction.commit_id == commit.id)
        .order_by(RiskPrediction.created_at.desc())
    )
    if risk_prediction is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "No risk assessment available for this commit yet"
        )

    explanation = db.scalar(
        select(Explanation).where(
            Explanation.decision_type == "RiskPrediction",
            Explanation.decision_id == risk_prediction.id,
        )
    )
    return _to_read(risk_prediction, explanation)
