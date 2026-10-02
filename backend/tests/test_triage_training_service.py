from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.defect import DefectReport
from app.models.enums import Role
from app.models.ml import MLModel
from app.models.repository import Repository
from app.services.triage_training_service import (
    TASK_PRIORITY,
    TASK_SEVERITY,
    train_and_register_champions,
)

from .conftest import make_user


@pytest.fixture(autouse=True)
def _clean_defect_reports(db_session: Session) -> None:
    # Training reads every labelled DefectReport globally (by design — same
    # as commit-risk/line-risk) — manual live verification against this
    # shared dev Postgres leaves real labelled rows that would otherwise
    # change `used_bootstrap_data` out from under these tests.
    db_session.execute(text("TRUNCATE defect_reports CASCADE"))


@pytest.fixture
def repo(db_session: Session) -> Repository:
    repository = Repository(name="demo", url="/tmp/x")
    db_session.add(repository)
    db_session.commit()
    return repository


@pytest.mark.story("US-21")
@pytest.mark.story("US-22")
def test_train_falls_back_to_bootstrap_when_no_real_data(
    db_session: Session, tmp_path: Path
) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    result = train_and_register_champions(db_session, seed=42, tracking_uri=tracking_uri)

    assert result["used_bootstrap_data"] is True
    assert result["severity_model"].stage == "champion"
    assert result["priority_model"].stage == "champion"
    assert result["severity_model"].task == TASK_SEVERITY
    assert result["priority_model"].task == TASK_PRIORITY
    assert "macro_f1" in result["severity_model"].metrics


@pytest.mark.story("US-21")
def test_train_prefers_real_labelled_data_once_enough_exists(
    db_session: Session, tmp_path: Path, repo: Repository
) -> None:
    reporter = make_user(db_session, "reporter-real-data@example.com", Role.REPORTER)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    severities = ["blocker", "critical", "major", "minor", "trivial"]
    for i in range(30):
        db_session.add(
            DefectReport(
                repository_id=repo.id,
                reporter_id=reporter.id,
                title=f"Report {i}",
                description="a real human-labelled report about something breaking",
                status="open",
                severity=severities[i % len(severities)],
                priority="P1" if i % 2 == 0 else "P3",
                reported_at=base + timedelta(minutes=i),
            )
        )
    db_session.commit()
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    result = train_and_register_champions(db_session, seed=42, tracking_uri=tracking_uri)

    assert result["used_bootstrap_data"] is False


@pytest.mark.story("US-21")
def test_training_again_does_not_demote_a_better_champion(
    db_session: Session, tmp_path: Path
) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    first = train_and_register_champions(db_session, seed=42, tracking_uri=tracking_uri)
    second = train_and_register_champions(db_session, seed=42, tracking_uri=tracking_uri)

    db_session.refresh(first["severity_model"])
    assert first["severity_model"].stage == "champion"
    assert second["severity_model"].stage == "challenger"

    champions = [
        m
        for m in db_session.query(MLModel).filter(MLModel.task == TASK_SEVERITY).all()
        if m.stage == "champion"
    ]
    assert len(champions) == 1
