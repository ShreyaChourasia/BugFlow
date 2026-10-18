import time
from typing import Any

import mlflow
import numpy as np
from bugflow_ml.explain.commit_risk_explainer import build_explainer, explain_prediction
from bugflow_ml.explain.counterfactual import CounterfactualResult, search_counterfactual
from bugflow_ml.models.commit_risk import FEATURE_COLUMNS, confidence_for, risk_level_for
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.queue import get_shadow_queue
from app.models.commit import Commit
from app.models.decision import Explanation
from app.models.ml import MLModel
from app.models.pull_request import RiskPrediction
from app.services.training_service import TASK

logger = get_logger(__name__)

# US-10: "load the champion once, score in memory" — reloaded only when the
# champion model actually changes, not on every request. "champion" and
# "shadow" are separate cache slots (US-45) so each only reloads when its
# own model_id changes.
_cache: dict[str, dict[str, Any]] = {}


def _get_champion(db: Session) -> MLModel | None:
    return db.scalar(select(MLModel).where(MLModel.task == TASK, MLModel.stage == "champion"))


def _get_challenger(db: Session) -> MLModel | None:
    return db.scalar(
        select(MLModel)
        .where(MLModel.task == TASK, MLModel.stage == "challenger")
        .order_by(MLModel.trained_at.desc())
    )


def _load_artifacts(
    db: Session, model: MLModel, cache_key: str, tracking_uri: str | None = None
) -> tuple[Any, Any]:
    """Shared by champion and shadow (challenger) loading — same cache
    shape, keyed separately so promoting a challenger to champion doesn't
    require re-downloading artifacts it already had cached as a shadow."""
    cached = _cache.get(cache_key)
    if cached is not None and cached["model_id"] == model.id:
        return cached["raw_model"], cached["calibrated_model"]

    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)
    raw_model = mlflow.sklearn.load_model(f"runs:/{model.mlflow_run_id}/raw_model")
    calibrated_model = mlflow.sklearn.load_model(f"runs:/{model.mlflow_run_id}/calibrated_model")
    _cache[cache_key] = {
        "model_id": model.id,
        "raw_model": raw_model,
        "calibrated_model": calibrated_model,
    }
    return raw_model, calibrated_model


def _load_champion_artifacts(
    db: Session, tracking_uri: str | None = None
) -> tuple[MLModel, Any, Any, Any]:
    champion = _get_champion(db)
    if champion is None:
        raise LookupError("No champion commit-risk model has been trained yet.")

    raw_model, calibrated_model = _load_artifacts(db, champion, "champion", tracking_uri)
    explainer_cache = _cache.get("champion")
    if explainer_cache is not None and "explainer" in explainer_cache:
        explainer = explainer_cache["explainer"]
    else:
        explainer = build_explainer(raw_model)
        _cache["champion"]["explainer"] = explainer
    return champion, raw_model, calibrated_model, explainer


def predict_commit(
    db: Session, commit: Commit, tracking_uri: str | None = None
) -> tuple[RiskPrediction, Explanation]:
    """US-08/US-10/US-12: scores a mined commit with the champion model and
    persists both the prediction and its explanation (C3: no decision is
    ever shown without one)."""
    champion, raw_model, calibrated_model, explainer = _load_champion_artifacts(db, tracking_uri)

    feature_vector = commit.features or {}
    start = time.perf_counter()

    X = np.array([[float(feature_vector.get(col, 0.0)) for col in FEATURE_COLUMNS]])
    raw_probability = float(raw_model.predict_proba(X)[0, 1])
    calibrated_probability = float(calibrated_model.predict_proba(X)[0, 1])
    explanation_payload = explain_prediction(explainer, feature_vector)

    latency_ms = (time.perf_counter() - start) * 1000

    risk_prediction = RiskPrediction(
        commit_id=commit.id,
        model_version=champion.version,
        probability=raw_probability,
        calibrated_probability=calibrated_probability,
        confidence=confidence_for(calibrated_probability),
        risk_level=risk_level_for(calibrated_probability),
        latency_ms=latency_ms,
    )
    db.add(risk_prediction)
    db.flush()

    explanation = Explanation(
        decision_type="RiskPrediction",
        decision_id=risk_prediction.id,
        text=explanation_payload["text"],
        factors={"items": explanation_payload["factors"]},
    )
    db.add(explanation)
    db.commit()
    db.refresh(risk_prediction)
    db.refresh(explanation)

    _enqueue_shadow(db, commit)
    return risk_prediction, explanation


def _enqueue_shadow(db: Session, commit: Commit) -> None:
    """US-45: the challenger's score is queued on the low-priority "shadow"
    queue instead of computed inline — inline shadow-scoring doubled the
    inference work per webhook and pushed k6's p95 to 10.3s (Phase 10).
    Never raises: a queue failure must not break the real prediction."""
    if _get_challenger(db) is None:
        return
    from app.workers.jobs.shadow import score_shadow_job

    try:
        get_shadow_queue().enqueue(score_shadow_job, commit.id)
    except Exception:
        logger.warning("shadow_enqueue_failed", commit_id=commit.id)


def score_shadow(db: Session, commit: Commit, tracking_uri: str | None = None) -> None:
    """US-45: if a challenger exists, scores it on the same commit too and
    persists the result as `is_shadow=True` — never surfaced as the real
    decision (no Explanation row), purely for the champion/challenger
    comparison page. Never raises: runs in a background job, where a
    failure should just log."""
    challenger = _get_challenger(db)
    if challenger is None:
        return
    feature_vector = commit.features or {}
    try:
        raw_model, calibrated_model = _load_artifacts(db, challenger, "shadow", tracking_uri)
        X = np.array([[float(feature_vector.get(col, 0.0)) for col in FEATURE_COLUMNS]])
        start = time.perf_counter()
        raw_probability = float(raw_model.predict_proba(X)[0, 1])
        calibrated_probability = float(calibrated_model.predict_proba(X)[0, 1])
        latency_ms = (time.perf_counter() - start) * 1000
        db.add(
            RiskPrediction(
                commit_id=commit.id,
                model_version=challenger.version,
                probability=raw_probability,
                calibrated_probability=calibrated_probability,
                confidence=confidence_for(calibrated_probability),
                risk_level=risk_level_for(calibrated_probability),
                latency_ms=latency_ms,
                is_shadow=True,
            )
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("shadow_scoring_failed", commit_id=commit.id, challenger_id=challenger.id)


def get_counterfactual(
    db: Session, commit: Commit, risk_threshold: float, tracking_uri: str | None = None
) -> CounterfactualResult:
    """US-35: "what would lower the risk" — searched only over actionable
    features, against the champion's own calibrated model and the
    repository's own risk threshold."""
    _champion, _raw_model, calibrated_model, _explainer = _load_champion_artifacts(db, tracking_uri)
    feature_vector = commit.features or {}
    return search_counterfactual(calibrated_model, feature_vector, FEATURE_COLUMNS, risk_threshold)
