import time
from typing import Any

import mlflow
import numpy as np
from bugflow_ml.explain.commit_risk_explainer import build_explainer, explain_prediction
from bugflow_ml.models.commit_risk import FEATURE_COLUMNS, confidence_for, risk_level_for
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.commit import Commit
from app.models.decision import Explanation
from app.models.ml import MLModel
from app.models.pull_request import RiskPrediction
from app.services.training_service import TASK

# US-10: "load the champion once, score in memory" — reloaded only when the
# champion model actually changes, not on every request.
_cache: dict[str, Any] = {"model_id": None}


def _get_champion(db: Session) -> MLModel | None:
    return db.scalar(select(MLModel).where(MLModel.task == TASK, MLModel.stage == "champion"))


def _load_champion_artifacts(
    db: Session, tracking_uri: str | None = None
) -> tuple[MLModel, Any, Any, Any]:
    champion = _get_champion(db)
    if champion is None:
        raise LookupError("No champion commit-risk model has been trained yet.")

    if _cache.get("model_id") == champion.id:
        return champion, _cache["raw_model"], _cache["calibrated_model"], _cache["explainer"]

    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)
    raw_model = mlflow.sklearn.load_model(f"runs:/{champion.mlflow_run_id}/raw_model")
    calibrated_model = mlflow.sklearn.load_model(f"runs:/{champion.mlflow_run_id}/calibrated_model")
    explainer = build_explainer(raw_model)

    _cache.update(
        model_id=champion.id,
        raw_model=raw_model,
        calibrated_model=calibrated_model,
        explainer=explainer,
    )
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

    return risk_prediction, explanation
