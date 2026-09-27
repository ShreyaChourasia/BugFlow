from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.services.training_service import reproduce_run, train_and_register_champion

from .conftest import make_commits_for_training


@pytest.mark.story("US-42")
@pytest.mark.story("NFR-US-07")
def test_reproduce_run_matches_within_tolerance(db_session: Session, tmp_path: Path) -> None:
    make_commits_for_training(db_session, n=150)
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    model = train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)

    result = reproduce_run(db_session, model.mlflow_run_id, tracking_uri=tracking_uri)

    assert result["reproducible"] is True
    assert result["mismatches"] == []


@pytest.mark.story("NFR-US-07")
def test_reproduce_run_flags_missing_seed_as_non_reproducible(
    db_session: Session, tmp_path: Path
) -> None:
    import mlflow

    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("commit_risk")
    with mlflow.start_run() as run:
        # Deliberately omit seed/data_version params.
        mlflow.log_metric("roc_auc", 0.9)
        run_id = run.info.run_id

    result = reproduce_run(db_session, run_id, tracking_uri=tracking_uri)

    assert result["reproducible"] is False
    assert "seed" in result["reason"] or "data_version" in result["reason"]
