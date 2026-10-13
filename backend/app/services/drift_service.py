import json
import tempfile
from pathlib import Path

import mlflow
from bugflow_ml.models.commit_risk import FEATURE_COLUMNS
from bugflow_ml.monitoring.drift import DEFAULT_PSI_THRESHOLD, compute_feature_drift
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.commit import Commit
from app.models.ml import DriftAlert, MLModel
from app.schemas.models import DriftAlertRead, DriftCheckResult
from app.services.training_service import TASK as COMMIT_RISK_TASK


def _set_tracking_uri(tracking_uri: str | None) -> None:
    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)


def run_drift_check(
    db: Session, tracking_uri: str | None = None, threshold: float = DEFAULT_PSI_THRESHOLD
) -> DriftCheckResult:
    """US-41: PSI per feature, reference vs. current. The reference
    population is exactly the commits the champion was trained on —
    reusing `reproduce_run()`'s own training-snapshot artifact (see
    docs/decisions/008) rather than storing a second copy of the
    distribution. "Current" is every commit mined since, that the champion
    has never seen."""
    champion = db.scalar(
        select(MLModel).where(MLModel.task == COMMIT_RISK_TASK, MLModel.stage == "champion")
    )
    if champion is None:
        return DriftCheckResult(status="unavailable")

    _set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as tmp:
        artifact_path = client.download_artifacts(
            champion.mlflow_run_id, "training_commits.json", tmp
        )
        snapshot = json.loads(Path(artifact_path).read_text())
    reference_ids = set(snapshot["train_ids"] + snapshot["calib_ids"] + snapshot["test_ids"])

    all_commits = db.scalars(select(Commit).where(Commit.features.isnot(None))).all()
    reference_commits = [c for c in all_commits if c.id in reference_ids]
    current_commits = [c for c in all_commits if c.id not in reference_ids]
    if not reference_commits or not current_commits:
        return DriftCheckResult(status="unavailable")

    reference_features = {
        col: [float((c.features or {}).get(col, 0.0)) for c in reference_commits]
        for col in FEATURE_COLUMNS
    }
    current_features = {
        col: [float((c.features or {}).get(col, 0.0)) for c in current_commits]
        for col in FEATURE_COLUMNS
    }

    drift_results = compute_feature_drift(reference_features, current_features, threshold)

    alerts: list[DriftAlertRead] = []
    for result in drift_results:
        if result.severity == "none":
            continue
        alert = DriftAlert(
            model_id=champion.id, feature=result.feature, psi=result.psi, severity=result.severity
        )
        db.add(alert)
        db.flush()
        alerts.append(
            DriftAlertRead(
                id=alert.id,
                model_id=alert.model_id,
                feature=alert.feature,
                psi=alert.psi,
                severity=alert.severity,
                created_at=alert.created_at,
                acknowledged=alert.acknowledged,
            )
        )
    db.commit()

    return DriftCheckResult(status="available", alerts=alerts, checked_features=len(drift_results))


def list_drift_alerts(db: Session, model_id: int | None = None) -> list[DriftAlert]:
    query = select(DriftAlert).order_by(DriftAlert.created_at.desc())
    if model_id is not None:
        query = query.where(DriftAlert.model_id == model_id)
    return list(db.scalars(query))
