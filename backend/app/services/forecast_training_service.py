import math
from datetime import UTC, datetime
from typing import Any

import mlflow
from bugflow_ml.models.resolution_forecast import (
    MIN_TRAINING_EXAMPLES,
    ForecastExample,
    generate_bootstrap_history,
    train_forecast_model,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.defect import DefectReport
from app.models.ml import MLModel

logger = get_logger(__name__)

TASK = "resolution_forecast"


def _set_tracking_uri(tracking_uri: str | None) -> None:
    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)


def _json_safe(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else value


def load_labelled_examples(db: Session, as_of: datetime | None = None) -> list[ForecastExample]:
    """US-31 AC: every defect report is an example — resolved ones are
    real events, still-open ones are **censored** observations (duration =
    time elapsed so far, not time to resolution), not dropped."""
    as_of = as_of or datetime.now(UTC)
    reports = db.scalars(select(DefectReport)).all()

    examples = []
    for report in reports:
        if report.severity is None or report.priority is None:
            continue
        if report.resolved_at is not None:
            duration = (report.resolved_at - report.reported_at).total_seconds() / 86400
            event_observed = True
        else:
            duration = (as_of - report.reported_at).total_seconds() / 86400
            event_observed = False
        if duration < 0:
            continue
        examples.append(
            ForecastExample(
                defect_id=report.id,
                severity=report.severity,
                priority=report.priority,
                component=report.component,
                duration_days=duration,
                event_observed=event_observed,
                reported_at=report.reported_at,
            )
        )
    return examples


def train_and_register_champion(
    db: Session, seed: int = 42, tracking_uri: str | None = None
) -> MLModel:
    """US-31/US-32/US-33: trains the resolution-forecast survival model.
    Real defect history (both resolved and still-open) is used whenever
    there's enough of it; otherwise falls back to the documented synthetic
    bootstrap set (see docs/decisions/007)."""
    _set_tracking_uri(tracking_uri)
    mlflow.set_experiment(TASK)

    examples = load_labelled_examples(db)
    used_bootstrap = len(examples) < MIN_TRAINING_EXAMPLES
    if used_bootstrap:
        logger.info(
            "forecast_training_using_bootstrap_data", real_examples=len(examples), seed=seed
        )
        examples = generate_bootstrap_history(seed=seed)

    result = train_forecast_model(examples, seed=seed)

    with mlflow.start_run() as run:
        mlflow.log_param("seed", seed)
        mlflow.log_param("data_version", result.data_version)
        mlflow.log_param("train_size", result.train_size)
        mlflow.log_param("test_size", result.test_size)
        for key, value in result.metrics.items():
            mlflow.log_metric(key, value)

        mlflow.sklearn.log_model(result.model, artifact_path="model", serialization_format="pickle")
        run_id = run.info.run_id

    existing_champion = db.scalar(
        select(MLModel).where(MLModel.task == TASK, MLModel.stage == "champion")
    )
    new_score = result.metrics["cox_concordance"]
    old_score = (
        (existing_champion.metrics or {}).get("cox_concordance") if existing_champion else None
    )
    should_promote = existing_champion is None or (
        not math.isnan(new_score) and (old_score is None or new_score > old_score)
    )

    stored_metrics: dict[str, Any] = {k: _json_safe(v) for k, v in result.metrics.items()}
    stored_metrics["used_bootstrap_data"] = used_bootstrap
    stored_metrics["feature_columns"] = result.feature_columns
    stored_metrics["component_categories"] = result.component_categories

    model_row = MLModel(
        task=TASK,
        version=run_id,
        mlflow_run_id=run_id,
        stage="champion" if should_promote else "challenger",
        metrics=stored_metrics,
        data_version=result.data_version,
        seed=seed,
        trained_at=datetime.now(UTC),
    )
    db.add(model_row)
    if should_promote and existing_champion is not None:
        existing_champion.stage = "archived"

    db.commit()
    db.refresh(model_row)
    logger.info(
        "forecast_training_complete",
        run_id=run_id,
        promoted=should_promote,
        used_bootstrap=used_bootstrap,
        cox_concordance=new_score,
    )
    return model_row
