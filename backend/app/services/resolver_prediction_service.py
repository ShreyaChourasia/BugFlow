from datetime import datetime
from typing import Any

import mlflow
from bugflow_ml.features.resolver_features import ResolvedReport
from bugflow_ml.models.resolver_suitability import (
    DeveloperCandidate,
    ScoredCandidate,
    score_candidates,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.ml import MLModel
from app.services.resolver_training_service import TASK

# Same "load once, reload only when the champion changes" cache every other
# prediction service in this project uses.
_cache: dict[int, Any] = {}


def _get_champion(db: Session) -> MLModel | None:
    return db.scalar(select(MLModel).where(MLModel.task == TASK, MLModel.stage == "champion"))


def _load_model(db: Session, tracking_uri: str | None) -> Any | None:
    champion = _get_champion(db)
    if champion is None:
        return None

    if champion.id in _cache:
        return _cache[champion.id]

    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)
    model = mlflow.sklearn.load_model(f"runs:/{champion.mlflow_run_id}/model")
    _cache[champion.id] = model
    return model


def get_candidates(
    db: Session,
    report_embedding: list[float],
    report_component: str | None,
    candidates: list[DeveloperCandidate],
    past_resolutions_by_developer: dict[int, list[ResolvedReport]],
    as_of: datetime,
    tracking_uri: str | None = None,
) -> list[ScoredCandidate] | None:
    """US-24: ranked candidates for a report, or None if no champion has
    been trained yet ("unavailable" — never a silent empty list)."""
    model = _load_model(db, tracking_uri)
    if model is None:
        return None
    return score_candidates(
        model, report_embedding, report_component, candidates, past_resolutions_by_developer, as_of
    )
