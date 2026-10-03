import math
from datetime import UTC, datetime
from typing import Any

import mlflow
from bugflow_ml.features.resolver_features import ResolvedReport, compute_resolver_features
from bugflow_ml.models.resolver_suitability import (
    MIN_TRAINING_EXAMPLES,
    SuitabilityExample,
    generate_bootstrap_history,
    train_suitability_model,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.defect import DefectReport
from app.models.developer import Developer
from app.models.ml import MLModel

logger = get_logger(__name__)

TASK = "resolver_suitability"


def _set_tracking_uri(tracking_uri: str | None) -> None:
    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)


def _json_safe(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else value


def load_labelled_examples(db: Session) -> list[SuitabilityExample]:
    """Only defects a developer has actually resolved count as training
    data — the loop closes itself as real resolutions accumulate, same
    pattern as triage's labelled examples.
    ponytail: uses each developer's *current* capacity/queue_depth for every
    historical example, not a snapshot of what it was back when that defect
    was resolved (this project doesn't keep that history) — a known
    simplification, not a correctness bug for the features that matter most
    (text similarity, component ownership, recency)."""
    developers = db.scalars(select(Developer)).all()
    resolved_reports = db.scalars(
        select(DefectReport)
        .where(
            DefectReport.assignee_id.isnot(None),
            DefectReport.resolved_at.isnot(None),
            DefectReport.embedding.isnot(None),
        )
        .order_by(DefectReport.resolved_at)
    ).all()

    history: dict[int, list[ResolvedReport]] = {d.id: [] for d in developers}
    examples: list[SuitabilityExample] = []
    for report in resolved_reports:
        embedding = list(report.embedding) if report.embedding is not None else None
        if embedding is None:
            continue
        for dev in developers:
            features = compute_resolver_features(
                report_embedding=embedding,
                report_component=report.component,
                past_resolutions=history[dev.id],
                developer_skills=dev.skills,
                developer_capacity=dev.capacity,
                developer_current_queue_depth=dev.current_queue_depth,
                as_of=report.resolved_at,
            )
            examples.append(
                SuitabilityExample(
                    report_id=report.id,
                    developer_id=dev.id,
                    features=features,
                    label=dev.id == report.assignee_id,
                    timestamp=report.resolved_at,
                )
            )
        assert report.assignee_id is not None  # query filters .isnot(None)
        history[report.assignee_id].append(
            ResolvedReport(
                embedding=embedding, component=report.component, resolved_at=report.resolved_at
            )
        )
    return examples


def train_and_register_champion(
    db: Session, seed: int = 42, tracking_uri: str | None = None
) -> MLModel:
    """US-24/US-30: trains the resolver suitability classifier. Real
    developer-resolution history is used whenever there's enough of it;
    otherwise falls back to the documented synthetic bootstrap set (see
    docs/decisions/006) purely to exercise and evaluate the model-based
    path — the cold-start skill-matching path (US-30) is what the real
    system leans on regardless, by design."""
    _set_tracking_uri(tracking_uri)
    mlflow.set_experiment(TASK)

    examples = load_labelled_examples(db)
    with_history = [e for e in examples if e.features.has_history]
    used_bootstrap = len(with_history) < MIN_TRAINING_EXAMPLES
    if used_bootstrap:
        logger.info(
            "resolver_training_using_bootstrap_data", real_examples=len(with_history), seed=seed
        )
        _, examples = generate_bootstrap_history(seed=seed)

    result = train_suitability_model(examples, seed=seed)

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
    new_score = result.metrics["mrr"]
    old_score = (existing_champion.metrics or {}).get("mrr") if existing_champion else None
    should_promote = existing_champion is None or (
        not math.isnan(new_score) and (old_score is None or new_score > old_score)
    )

    stored_metrics: dict[str, Any] = {k: _json_safe(v) for k, v in result.metrics.items()}
    stored_metrics["used_bootstrap_data"] = used_bootstrap

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
        "resolver_training_complete",
        run_id=run_id,
        promoted=should_promote,
        used_bootstrap=used_bootstrap,
        mrr=new_score,
    )
    return model_row
