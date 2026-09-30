from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.models.ml import MLModel
from app.models.repository import Repository
from app.services.line_risk_training_service import TASK, train_and_register_champion


@pytest.mark.story("US-13")
def test_train_and_register_champion_promotes_first_model(
    db_session: Session, repository_with_commits: Repository, tmp_path: Path
) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    model = train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)

    assert model.task == TASK
    assert model.stage == "champion"
    assert model.mlflow_run_id
    assert model.data_version
    for key in ("roc_auc", "top_k_accuracy", "recall_at_20pct_lines"):
        assert key in model.metrics


@pytest.mark.story("US-13")
def test_training_again_does_not_demote_a_better_champion(
    db_session: Session, repository_with_commits: Repository, tmp_path: Path
) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    first = train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)
    second = train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)

    db_session.refresh(first)
    assert first.stage == "champion"
    assert second.stage == "challenger"

    champions = [
        m
        for m in db_session.query(MLModel).filter(MLModel.task == TASK).all()
        if m.stage == "champion"
    ]
    assert len(champions) == 1
    assert champions[0].id == first.id
