from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.ml import MLModel
from app.services.training_service import TASK, train_and_register_champion

from .conftest import make_commits_for_training


@pytest.mark.story("US-08")
@pytest.mark.story("US-44")
@pytest.mark.story("NFR-US-11")
def test_train_and_register_champion_promotes_first_model(
    db_session: Session, tmp_path: Path
) -> None:
    make_commits_for_training(db_session, n=150)
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    model = train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)

    assert model.task == TASK
    assert model.stage == "champion"
    assert model.seed == 42
    assert model.data_version
    assert model.mlflow_run_id
    for key in ("roc_auc", "pr_auc", "f1", "brier_score", "recall_at_20pct_effort"):
        assert key in model.metrics
    assert "disparity" in model.metrics["fairness"]  # C10: present in every evaluation


@pytest.mark.story("US-08")
def test_training_again_does_not_demote_a_better_champion(
    db_session: Session, tmp_path: Path
) -> None:
    commits = make_commits_for_training(db_session, n=150)
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    first = train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)
    assert first.stage == "champion"

    # Same data and seed -> the same pr_auc, which is not a strict
    # improvement, so the original champion should stay champion.
    second = train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)

    db_session.refresh(first)
    assert first.stage == "champion"
    assert second.stage == "challenger"

    all_models = list(db_session.scalars(select(MLModel).where(MLModel.task == TASK)))
    champions = [m for m in all_models if m.stage == "champion"]
    assert len(champions) == 1
    assert champions[0].id == first.id
    assert len(commits) == 150  # sanity: fixture actually created the data
