from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.commit import Commit
from app.models.ml import MLModel
from app.models.pull_request import LineRisk, RiskPrediction
from app.models.repository import Repository
from app.services.line_risk_training_service import (
    TASK,
    _load_line_examples,
    train_and_register_champion,
)


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


@pytest.mark.story("US-15")
def test_false_alarm_feedback_relabels_that_line_as_clean_for_training(
    db_session: Session, repository_with_commits: Repository
) -> None:
    examples, relabelled = _load_line_examples(db_session)
    assert relabelled == 0
    flagged = next(e for e in examples if e.label)

    commit = db_session.scalar(select(Commit).where(Commit.sha == flagged.commit_sha))
    assert commit is not None
    prediction = RiskPrediction(
        commit_id=commit.id,
        model_version="v1",
        probability=0.9,
        calibrated_probability=0.9,
        confidence=0.8,
        risk_level="high",
        latency_ms=1.0,
    )
    db_session.add(prediction)
    db_session.flush()
    db_session.add(
        LineRisk(
            prediction_id=prediction.id,
            file_path=flagged.file_path,
            line_no=flagged.line_no,
            code=flagged.text,
            risk_score=0.9,
            rank=1,
            marked_false_alarm=True,
        )
    )
    db_session.commit()

    examples, relabelled = _load_line_examples(db_session)

    assert relabelled == 1
    relabelled_example = next(
        e
        for e in examples
        if (e.commit_sha, e.file_path, e.line_no)
        == (flagged.commit_sha, flagged.file_path, flagged.line_no)
    )
    assert relabelled_example.label is False
