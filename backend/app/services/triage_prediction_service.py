from dataclasses import dataclass
from typing import Any

import mlflow
from bugflow_ml.models.triage_classifier import (
    explain_label,
    predict_label,
    should_abstain,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.ml import MLModel
from app.services.triage_training_service import TASK_PRIORITY, TASK_SEVERITY

# Same "load once, reload only when the champion changes" cache prediction_service
# and line_prediction_service already use.
_cache: dict[str, Any] = {}


@dataclass
class TargetPrediction:
    label: str
    confidence: float
    top_words: list[str]


def _get_champion(db: Session, task: str) -> MLModel | None:
    return db.scalar(select(MLModel).where(MLModel.task == task, MLModel.stage == "champion"))


def _load_pipeline(db: Session, task: str, tracking_uri: str | None) -> tuple[Any, Any] | None:
    champion = _get_champion(db, task)
    if champion is None:
        return None

    cache_key = f"{task}:{champion.id}"
    if cache_key in _cache:
        return _cache[cache_key]

    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)
    pipeline = mlflow.sklearn.load_model(f"runs:/{champion.mlflow_run_id}/pipeline")
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["clf"]
    _cache[cache_key] = (vectorizer, model)
    return vectorizer, model


def _predict_target(
    db: Session, task: str, text: str, tracking_uri: str | None
) -> TargetPrediction | None:
    loaded = _load_pipeline(db, task, tracking_uri)
    if loaded is None:
        return None
    vectorizer, model = loaded

    label, confidence = predict_label(vectorizer, model, text)
    top_words = [token for token, _ in explain_label(vectorizer, model, text, label)]
    return TargetPrediction(label=label, confidence=confidence, top_words=top_words)


def predict_triage(
    db: Session, text: str, tracking_uri: str | None = None
) -> tuple[TargetPrediction, TargetPrediction] | None:
    """US-21: returns (severity, priority) predictions, or None if either
    abstains (too little text, AC2) or no champion has been trained yet for
    either target — callers distinguish those two cases themselves."""
    if should_abstain(text):
        return None

    severity = _predict_target(db, TASK_SEVERITY, text, tracking_uri)
    priority = _predict_target(db, TASK_PRIORITY, text, tracking_uri)
    if severity is None or priority is None:
        return None
    return severity, priority
