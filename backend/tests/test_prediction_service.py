import time
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.models.commit import Commit
from app.services.prediction_service import predict_commit
from app.services.training_service import train_and_register_champion

from .conftest import make_commits_for_training


@pytest.fixture
def champion_and_commit(db_session: Session, tmp_path: Path) -> tuple[str, Commit]:
    commits = make_commits_for_training(db_session, n=150)
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)
    return tracking_uri, commits[-1]


@pytest.mark.story("US-08")
@pytest.mark.story("US-10")
@pytest.mark.story("US-12")
def test_predict_commit_persists_prediction_and_explanation(
    db_session: Session, champion_and_commit: tuple[str, Commit]
) -> None:
    tracking_uri, commit = champion_and_commit

    risk_prediction, explanation = predict_commit(db_session, commit, tracking_uri=tracking_uri)

    assert risk_prediction.commit_id == commit.id
    assert 0.0 <= risk_prediction.probability <= 1.0
    assert 0.0 <= risk_prediction.calibrated_probability <= 1.0
    assert 0.0 <= risk_prediction.confidence <= 1.0
    assert risk_prediction.risk_level in {"low", "medium", "high"}
    assert risk_prediction.latency_ms > 0

    # C3: no automated decision is ever shown without a plain-language
    # explanation — verified at the persistence layer, not just the API.
    assert explanation.text
    assert 1 <= len(explanation.factors["items"]) <= 3


@pytest.mark.story("US-10")
def test_scoring_latency_micro_benchmark(
    db_session: Session, champion_and_commit: tuple[str, Commit]
) -> None:
    """A micro-benchmark of the scoring function itself (model already
    loaded/cached) — not the full request-path p95 that Phase 4's k6/Locust
    test measures against a running server."""
    tracking_uri, commit = champion_and_commit

    predict_commit(db_session, commit, tracking_uri=tracking_uri)  # warm the cache

    durations = []
    for _ in range(20):
        start = time.perf_counter()
        predict_commit(db_session, commit, tracking_uri=tracking_uri)
        durations.append(time.perf_counter() - start)

    average_ms = (sum(durations) / len(durations)) * 1000
    assert average_ms < 500, f"scoring averaged {average_ms:.1f}ms per call, expected < 500ms"
