from typing import Any

import mlflow
from bugflow_ml.models.resolution_forecast import (
    ForecastPrediction,
    ForecastTrainingResult,
    predict_forecast,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.ml import MLModel
from app.services.forecast_training_service import TASK

_cache: dict[int, Any] = {}


def _get_champion(db: Session) -> MLModel | None:
    return db.scalar(select(MLModel).where(MLModel.task == TASK, MLModel.stage == "champion"))


def get_champion_result(
    db: Session, tracking_uri: str | None = None
) -> ForecastTrainingResult | None:
    """Exposed so callers that need more than `get_forecast()`'s single
    prediction (e.g. the per-forecast explanation's driver attribution) can
    reuse the same cached, already-loaded champion model."""
    return _load_result(db, tracking_uri)


def _load_result(db: Session, tracking_uri: str | None) -> ForecastTrainingResult | None:
    champion = _get_champion(db)
    if champion is None:
        return None
    if champion.id in _cache:
        return _cache[champion.id]

    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)
    model = mlflow.sklearn.load_model(f"runs:/{champion.mlflow_run_id}/model")
    assert champion.metrics is not None  # always set by train_and_register_champion
    result = ForecastTrainingResult(
        seed=champion.seed,
        data_version=champion.data_version,
        model=model,
        feature_columns=champion.metrics["feature_columns"],
        component_categories=champion.metrics["component_categories"],
        metrics=champion.metrics,
        train_size=0,
        test_size=0,
    )
    _cache[champion.id] = result
    return result


def get_forecast(
    db: Session,
    severity: str,
    priority: str,
    component: str | None,
    tracking_uri: str | None = None,
) -> ForecastPrediction | None:
    """US-31: a probability curve plus median/P90, or None if no champion
    has been trained yet ("unavailable" — never a silent guess)."""
    result = _load_result(db, tracking_uri)
    if result is None:
        return None
    return predict_forecast(result, severity, priority, component)
