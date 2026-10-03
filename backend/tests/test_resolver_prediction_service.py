from datetime import UTC, datetime
from pathlib import Path

import pytest
from bugflow_ml.embeddings.duplicate_detection import embed_text
from bugflow_ml.models.resolver_suitability import DeveloperCandidate
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.resolver_prediction_service import get_candidates
from app.services.resolver_training_service import train_and_register_champion


@pytest.fixture(autouse=True)
def _clean(db_session: Session) -> None:
    db_session.execute(text("TRUNCATE defect_reports, developers CASCADE"))


@pytest.fixture
def champion_tracking_uri(db_session: Session, tmp_path: Path) -> str:
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)
    return tracking_uri


@pytest.mark.story("US-24")
def test_get_candidates_returns_none_when_no_champion_exists(db_session: Session) -> None:
    candidates = get_candidates(
        db_session,
        embed_text("a login bug"),
        "login",
        [DeveloperCandidate(id=1, name="Dev", skills=["login"], capacity=5, current_queue_depth=0)],
        {1: []},
        datetime.now(UTC),
        tracking_uri="sqlite:///:memory:",
    )

    assert candidates is None


@pytest.mark.story("US-24")
@pytest.mark.story("US-30")
def test_get_candidates_ranks_cold_start_developers_by_skill_match(
    db_session: Session, champion_tracking_uri: str
) -> None:
    candidates = get_candidates(
        db_session,
        embed_text("the login page throws an error on submit"),
        "login",
        [
            DeveloperCandidate(
                id=1, name="Matches", skills=["login"], capacity=5, current_queue_depth=0
            ),
            DeveloperCandidate(
                id=2, name="No match", skills=["billing"], capacity=5, current_queue_depth=0
            ),
        ],
        {1: [], 2: []},
        datetime.now(UTC),
        tracking_uri=champion_tracking_uri,
    )

    assert candidates is not None
    assert all(c.is_cold_start for c in candidates)
    assert candidates[0].developer_id == 1
    assert candidates[0].score > candidates[1].score
