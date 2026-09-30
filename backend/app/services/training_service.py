import json
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from bugflow_ml.models.commit_risk import TrainingCommit, compute_data_version, train_commit_risk
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.commit import Commit
from app.models.developer import Developer
from app.models.ml import MLModel

logger = get_logger(__name__)

TASK = "commit_risk"
DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_CALIB_FRAC = 0.15
REPRODUCIBILITY_TOLERANCE = 0.005  # C4: within 0.5%


def _set_tracking_uri(tracking_uri: str | None) -> None:
    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)


def _json_safe(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else value


def _load_training_commits(
    db: Session, commit_ids: list[int] | None = None
) -> list[TrainingCommit]:
    query = select(Commit, Developer.joined_at).join(
        Developer, Commit.author_id == Developer.id, isouter=True
    )
    if commit_ids is not None:
        query = query.where(Commit.id.in_(commit_ids))

    commits = []
    for commit, joined_at in db.execute(query).all():
        tenure_days = None
        if joined_at is not None:
            tenure_days = (commit.timestamp - joined_at).total_seconds() / 86400
        commits.append(
            TrainingCommit(
                id=commit.id,
                sha=commit.sha,
                timestamp=commit.timestamp,
                features=commit.features or {},
                is_bug_inducing=commit.is_bug_inducing,
                author_tenure_days=tenure_days,
            )
        )
    return commits


def train_and_register_champion(
    db: Session, seed: int = 42, tracking_uri: str | None = None
) -> MLModel:
    """US-08/US-42/US-44/NFR-US-07/NFR-US-11: trains on every mined commit,
    logs a complete experiment record to MLflow (including the exact commit
    ids used, so `reproduce_run` can refetch identical data), and promotes
    the new model to champion if it beats the current one on PR-AUC."""
    _set_tracking_uri(tracking_uri)
    mlflow.set_experiment(TASK)

    commits = _load_training_commits(db)
    result = train_commit_risk(
        commits, seed=seed, train_frac=DEFAULT_TRAIN_FRAC, calib_frac=DEFAULT_CALIB_FRAC
    )

    with mlflow.start_run() as run:
        mlflow.log_param("seed", seed)
        mlflow.log_param("data_version", result.data_version)
        mlflow.log_param("train_frac", DEFAULT_TRAIN_FRAC)
        mlflow.log_param("calib_frac", DEFAULT_CALIB_FRAC)
        mlflow.log_param("train_size", len(result.train_ids))
        mlflow.log_param("calib_size", len(result.calib_ids))
        mlflow.log_param("test_size", len(result.test_ids))
        mlflow.log_param("feature_columns", ",".join(result.feature_columns))

        for key, value in result.metrics.items():
            if not math.isnan(value):
                mlflow.log_metric(key, value)
        for key, value in result.baseline_metrics.items():
            if not math.isnan(value):
                mlflow.log_metric(f"baseline_{key}", value)
        for key, value in result.fairness.items():
            if isinstance(value, int | float):
                mlflow.log_metric(f"fairness_{key}", value)

        # skops (MLflow's default sklearn serialization) refuses to load
        # LightGBM's own types unless explicitly told they're trusted; these
        # models are only ever loaded by us, so plain pickle is simpler.
        #
        # artifact_path= (not MLflow 3.x's new name=) deliberately: name=
        # logs it as a separate "Logged Model" entity at models:/<model_id>,
        # not as a plain run artifact — prediction_service loads champions
        # via runs:/<run_id>/<artifact_path>, which needs the classic path.
        # artifact_path is deprecated but still fully functional.
        mlflow.sklearn.log_model(
            result.raw_model, artifact_path="raw_model", serialization_format="pickle"
        )
        mlflow.sklearn.log_model(
            result.calibrated_model, artifact_path="calibrated_model", serialization_format="pickle"
        )

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = Path(tmp) / "training_commits.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "train_ids": result.train_ids,
                        "calib_ids": result.calib_ids,
                        "test_ids": result.test_ids,
                    }
                )
            )
            mlflow.log_artifact(str(snapshot_path))

        run_id = run.info.run_id

    existing_champion = db.scalar(
        select(MLModel).where(MLModel.task == TASK, MLModel.stage == "champion")
    )
    new_pr_auc = result.metrics["pr_auc"]
    old_pr_auc = (existing_champion.metrics or {}).get("pr_auc") if existing_champion else None
    should_promote = existing_champion is None or (
        not math.isnan(new_pr_auc) and (old_pr_auc is None or new_pr_auc > old_pr_auc)
    )

    stored_metrics = {k: _json_safe(v) for k, v in result.metrics.items()}
    stored_metrics.update(
        {f"baseline_{k}": _json_safe(v) for k, v in result.baseline_metrics.items()}
    )
    stored_metrics["fairness"] = result.fairness

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
        "commit_risk_training_complete",
        run_id=run_id,
        promoted=should_promote,
        pr_auc=new_pr_auc,
    )
    return model_row


def reproduce_run(
    db: Session, mlflow_run_id: str, tracking_uri: str | None = None
) -> dict[str, Any]:
    """C4: re-runs a recorded experiment and checks its headline metrics
    reproduce within 0.5%. NFR-US-07: a run missing a seed or data_version
    is flagged as non-reproducible rather than silently "passing"."""
    _set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    original_run = client.get_run(mlflow_run_id)
    params = original_run.data.params

    if "seed" not in params or "data_version" not in params:
        return {
            "run_id": mlflow_run_id,
            "reproducible": False,
            "reason": "Run is missing a seed or data_version — flagged as non-reproducible.",
        }

    seed = int(params["seed"])
    train_frac = float(params.get("train_frac", DEFAULT_TRAIN_FRAC))
    calib_frac = float(params.get("calib_frac", DEFAULT_CALIB_FRAC))
    original_data_version = params["data_version"]

    with tempfile.TemporaryDirectory() as tmp:
        artifact_path = client.download_artifacts(mlflow_run_id, "training_commits.json", tmp)
        snapshot = json.loads(Path(artifact_path).read_text())

    commit_ids = snapshot["train_ids"] + snapshot["calib_ids"] + snapshot["test_ids"]
    commits = _load_training_commits(db, commit_ids)

    recomputed_data_version = compute_data_version([c.id for c in commits])
    if recomputed_data_version != original_data_version:
        return {
            "run_id": mlflow_run_id,
            "reproducible": False,
            "reason": (
                "The underlying commit data has changed since this run "
                "(data_version mismatch) — cannot verify reproducibility."
            ),
        }

    reproduced = train_commit_risk(commits, seed=seed, train_frac=train_frac, calib_frac=calib_frac)

    original_metrics = {
        k: v
        for k, v in original_run.data.metrics.items()
        if not k.startswith("baseline_") and not k.startswith("fairness_")
    }

    mismatches = []
    for key, original_value in original_metrics.items():
        reproduced_value = reproduced.metrics.get(key)
        if reproduced_value is None or math.isnan(reproduced_value) or math.isnan(original_value):
            continue
        relative_diff = (
            0.0
            if original_value == 0 and reproduced_value == 0
            else abs(reproduced_value - original_value) / max(abs(original_value), 1e-12)
        )
        if relative_diff > REPRODUCIBILITY_TOLERANCE:
            mismatches.append(
                {
                    "metric": key,
                    "original": original_value,
                    "reproduced": reproduced_value,
                    "relative_diff": relative_diff,
                }
            )

    return {
        "run_id": mlflow_run_id,
        "reproducible": not mismatches,
        "mismatches": mismatches,
        "original_metrics": original_metrics,
        "reproduced_metrics": reproduced.metrics,
    }


def export_experiment_record(mlflow_run_id: str, tracking_uri: str | None = None) -> dict[str, Any]:
    """US-44: a complete, self-contained record of one experiment."""
    _set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(mlflow_run_id)
    artifacts = client.list_artifacts(mlflow_run_id)

    return {
        "run_id": mlflow_run_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
        "artifacts": [a.path for a in artifacts],
    }
