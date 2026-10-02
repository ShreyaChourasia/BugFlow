from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.triage_prediction_service import predict_triage
from app.services.triage_training_service import train_and_register_champions


@pytest.fixture
def champion_tracking_uri(db_session: Session, tmp_path: Path) -> str:
    # Training reads every labelled DefectReport globally — clear any real
    # rows this shared dev Postgres picked up from manual live verification,
    # so this always trains on the documented bootstrap set the assertions
    # below expect.
    db_session.execute(text("TRUNCATE defect_reports CASCADE"))
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    train_and_register_champions(db_session, seed=42, tracking_uri=tracking_uri)
    return tracking_uri


@pytest.mark.story("US-21")
def test_predict_triage_returns_severity_and_priority(
    db_session: Session, champion_tracking_uri: str
) -> None:
    text = "The login feature has an issue: data loss. fix immediately."

    result = predict_triage(db_session, text, tracking_uri=champion_tracking_uri)

    assert result is not None
    severity, priority = result
    assert severity.label == "blocker"
    assert 0.0 < severity.confidence <= 1.0
    assert severity.top_words
    assert priority.label == "P1"
    assert 0.0 < priority.confidence <= 1.0


@pytest.mark.story("US-21")
def test_predict_triage_abstains_on_short_text(
    db_session: Session, champion_tracking_uri: str
) -> None:
    result = predict_triage(db_session, "too short", tracking_uri=champion_tracking_uri)

    assert result is None


def test_predict_triage_returns_none_when_no_champion_exists(db_session: Session) -> None:
    result = predict_triage(
        db_session,
        "a sufficiently long description of a real problem",
        tracking_uri="sqlite:///:memory:",
    )

    assert result is None
