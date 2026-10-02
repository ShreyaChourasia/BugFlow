import math
from datetime import UTC, datetime
from typing import Any

import mlflow
from bugflow_ml.models.triage_classifier import (
    MIN_TRAINING_EXAMPLES,
    TriageExample,
    generate_bootstrap_examples,
    train_priority_classifier,
    train_severity_classifier,
)
from sklearn.pipeline import Pipeline
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.defect import DefectReport

logger = get_logger(__name__)

TASK_SEVERITY = "triage_severity"
TASK_PRIORITY = "triage_priority"


def _set_tracking_uri(tracking_uri: str | None) -> None:
    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)


def _json_safe(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else value


def load_labelled_examples(db: Session) -> list[TriageExample]:
    """Only reports a human has actually decided on — set either directly or
    via accepting/overriding a prior suggestion (US-23) — count as training
    data. The loop closes itself over time as more decisions accumulate."""
    reports = db.scalars(
        select(DefectReport).where(
            DefectReport.severity.isnot(None), DefectReport.priority.isnot(None)
        )
    ).all()
    return [
        TriageExample(
            text=f"{r.title}\n{r.description}",
            severity=r.severity,
            priority=r.priority,
            timestamp=r.reported_at,
            defect_id=r.id,
        )
        for r in reports
    ]


def _train_and_log(
    task: str, examples: list[TriageExample], train_fn: Any, seed: int
) -> dict[str, Any]:
    mlflow.set_experiment(task)
    result = train_fn(examples, seed=seed)

    with mlflow.start_run() as run:
        mlflow.log_param("seed", seed)
        mlflow.log_param("data_version", result.data_version)
        mlflow.log_param("train_size", result.train_size)
        mlflow.log_param("test_size", result.test_size)
        mlflow.log_metric("macro_f1", result.metrics["macro_f1"])

        pipeline = Pipeline([("tfidf", result.vectorizer), ("clf", result.model)])
        mlflow.sklearn.log_model(pipeline, artifact_path="pipeline", serialization_format="pickle")
        run_id = run.info.run_id

    return {
        "run_id": run_id,
        "data_version": result.data_version,
        "metrics": {
            "macro_f1": _json_safe(result.metrics["macro_f1"]),
            "confusion_matrix": result.metrics["confusion_matrix"],
            "labels": result.metrics["labels"],
        },
    }


def _promote_if_better(db: Session, task: str, run_info: dict[str, Any], seed: int) -> Any:
    from app.models.ml import MLModel

    existing_champion = db.scalar(
        select(MLModel).where(MLModel.task == task, MLModel.stage == "champion")
    )
    new_score = run_info["metrics"]["macro_f1"]
    old_score = (existing_champion.metrics or {}).get("macro_f1") if existing_champion else None
    should_promote = existing_champion is None or (
        new_score is not None and (old_score is None or new_score > old_score)
    )

    model_row = MLModel(
        task=task,
        version=run_info["run_id"],
        mlflow_run_id=run_info["run_id"],
        stage="champion" if should_promote else "challenger",
        metrics=run_info["metrics"],
        data_version=run_info["data_version"],
        seed=seed,
        trained_at=datetime.now(UTC),
    )
    db.add(model_row)
    if should_promote and existing_champion is not None:
        existing_champion.stage = "archived"
    return model_row


def train_and_register_champions(
    db: Session, seed: int = 42, tracking_uri: str | None = None
) -> dict[str, Any]:
    """US-21/US-22/US-23: trains both the severity and priority classifiers.
    Real human-decided labels are used whenever there are enough of them;
    otherwise falls back to the documented synthetic bootstrap set (see
    docs/decisions/005) so the classifier has something to suggest from
    before any real triage decisions exist."""
    _set_tracking_uri(tracking_uri)

    examples = load_labelled_examples(db)
    used_bootstrap = len(examples) < MIN_TRAINING_EXAMPLES
    if used_bootstrap:
        logger.info("triage_training_using_bootstrap_data", real_examples=len(examples), seed=seed)
        examples = generate_bootstrap_examples(seed=seed)

    severity_run = _train_and_log(TASK_SEVERITY, examples, train_severity_classifier, seed)
    priority_run = _train_and_log(TASK_PRIORITY, examples, train_priority_classifier, seed)

    severity_model = _promote_if_better(db, TASK_SEVERITY, severity_run, seed)
    priority_model = _promote_if_better(db, TASK_PRIORITY, priority_run, seed)
    db.commit()
    db.refresh(severity_model)
    db.refresh(priority_model)

    logger.info(
        "triage_training_complete",
        used_bootstrap=used_bootstrap,
        severity_macro_f1=severity_run["metrics"]["macro_f1"],
        priority_macro_f1=priority_run["metrics"]["macro_f1"],
    )
    return {
        "used_bootstrap_data": used_bootstrap,
        "severity_model": severity_model,
        "priority_model": priority_model,
    }
