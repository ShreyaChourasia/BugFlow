from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from bugflow_ml.embeddings.duplicate_detection import embed_text
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.defect import DefectReport
from app.models.developer import Developer
from app.models.enums import Role
from app.models.ml import MLModel
from app.models.repository import Repository
from app.services.resolver_training_service import TASK, train_and_register_champion

from .conftest import make_user


@pytest.fixture(autouse=True)
def _clean_defect_reports(db_session: Session) -> None:
    # Same reasoning as triage's training tests: training reads every
    # resolved DefectReport globally, so a stray real row from manual live
    # verification against this shared dev Postgres would change
    # `used_bootstrap_data` out from under these tests.
    db_session.execute(text("TRUNCATE defect_reports, developers CASCADE"))


@pytest.fixture
def repo(db_session: Session) -> Repository:
    repository = Repository(name="demo", url="/tmp/x")
    db_session.add(repository)
    db_session.commit()
    return repository


@pytest.mark.story("US-24")
def test_train_falls_back_to_bootstrap_when_no_real_history(
    db_session: Session, tmp_path: Path
) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    model = train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)

    assert model.stage == "champion"
    assert model.task == TASK
    assert model.metrics["used_bootstrap_data"] is True
    assert "mrr" in model.metrics


@pytest.mark.story("US-24")
def test_train_prefers_real_resolution_history_once_enough_exists(
    db_session: Session, tmp_path: Path, repo: Repository
) -> None:
    reporter = make_user(db_session, "reporter-resolver-real@example.com", Role.REPORTER)
    developers = [
        Developer(name=f"Dev {i}", skills=["login"], capacity=5, current_queue_depth=0)
        for i in range(3)
    ]
    db_session.add_all(developers)
    db_session.commit()

    base = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(45):
        developer = developers[i % len(developers)]
        db_session.add(
            DefectReport(
                repository_id=repo.id,
                reporter_id=reporter.id,
                title=f"Login bug {i}",
                description="the login page throws an error when submitting credentials",
                status="resolved",
                component="login",
                assignee_id=developer.id,
                embedding=embed_text(f"login bug {i} credentials submit error"),
                reported_at=base + timedelta(hours=i),
                resolved_at=base + timedelta(hours=i, minutes=30),
            )
        )
    db_session.commit()
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    model = train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)

    assert model.metrics["used_bootstrap_data"] is False


@pytest.mark.story("US-24")
def test_training_again_does_not_demote_a_better_champion(
    db_session: Session, tmp_path: Path
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
