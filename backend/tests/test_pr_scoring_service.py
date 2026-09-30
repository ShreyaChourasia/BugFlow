from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session

from app.models.commit import Commit
from app.models.pull_request import PullRequest
from app.models.repository import Repository
from app.services import pr_scoring_service
from app.services.pr_scoring_service import _decide_conclusion, score_pull_request
from app.services.training_service import train_and_register_champion

from .conftest import make_commits_for_training


@pytest.fixture
def champion_and_pr(db_session: Session, tmp_path: Path) -> tuple[str, PullRequest, Repository]:
    commits = make_commits_for_training(db_session, n=150)
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)

    head_commit = commits[-1]
    repo = db_session.get(Repository, head_commit.repository_id)
    pr = PullRequest(
        repository_id=repo.id,
        number=1,
        title="Test PR",
        head_sha=head_commit.sha,
        base_sha=commits[0].sha,
        check_run_id="fake-check-existing",
    )
    db_session.add(pr)
    db_session.commit()
    db_session.refresh(pr)
    return tracking_uri, pr, repo


# --- conclusion logic (pure, no model needed) -------------------------------


def test_merge_blocking_off_is_always_neutral() -> None:
    repo = SimpleNamespace(merge_blocking_enabled=False, risk_threshold=0.5)
    prediction = SimpleNamespace(calibrated_probability=0.99, risk_level="high", confidence=0.9)

    conclusion, _ = _decide_conclusion(repo, prediction)

    assert conclusion == "neutral"


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.9, "failure"), (0.1, "success")],
)
def test_merge_blocking_on_reflects_risk_threshold(probability: float, expected: str) -> None:
    repo = SimpleNamespace(merge_blocking_enabled=True, risk_threshold=0.5)
    prediction = SimpleNamespace(calibrated_probability=probability, risk_level="x", confidence=0.5)

    conclusion, _ = _decide_conclusion(repo, prediction)

    assert conclusion == expected


# --- full scoring flow -------------------------------------------------------


@pytest.mark.story("US-08")
@pytest.mark.story("US-11")
@pytest.mark.story("US-47")
def test_score_pull_request_completes_and_edits_not_creates_a_comment(
    db_session: Session, champion_and_pr: tuple[str, PullRequest, Repository]
) -> None:
    tracking_uri, pr, _repo = champion_and_pr

    score_pull_request(db_session, pr.id, tracking_uri=tracking_uri)
    db_session.refresh(pr)

    assert pr.check_status == "completed"
    assert pr.check_conclusion == "neutral"  # merge_blocking_enabled defaults False
    assert pr.comment_body
    first_comment_id = pr.comment_id
    assert first_comment_id

    # Re-scoring (e.g. another push) must edit, not add, the comment (US-46).
    score_pull_request(db_session, pr.id, tracking_uri=tracking_uri)
    db_session.refresh(pr)

    assert pr.comment_id == first_comment_id


@pytest.mark.story("US-08")
def test_score_pull_request_when_head_commit_was_never_mined(db_session: Session) -> None:
    repo = Repository(name="demo", url="/tmp/does-not-need-to-exist")
    db_session.add(repo)
    db_session.commit()
    pr = PullRequest(
        repository_id=repo.id, number=1, title="t", head_sha="unmined-sha", base_sha="base"
    )
    db_session.add(pr)
    db_session.commit()

    score_pull_request(db_session, pr.id)
    db_session.refresh(pr)

    assert pr.check_status == "completed"
    assert pr.check_conclusion == "neutral"
    assert "hasn't been mined" in pr.check_summary


@pytest.mark.story("US-08")
def test_score_pull_request_when_no_champion_model_exists(db_session: Session) -> None:
    repo = Repository(name="demo", url="/tmp/does-not-need-to-exist")
    db_session.add(repo)
    db_session.commit()
    commit = Commit(
        sha="sha1",
        repository_id=repo.id,
        message="msg",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        features={
            "lines_added": 1, "lines_deleted": 1, "churn": 2, "files_changed": 1,
            "directories_touched": 1, "subsystems_touched": 1, "entropy": 0.0,
            "author_prior_commits": 0, "is_fix": False,
        },
    )
    db_session.add(commit)
    db_session.commit()
    pr = PullRequest(repository_id=repo.id, number=1, title="t", head_sha="sha1", base_sha="sha1")
    db_session.add(pr)
    db_session.commit()

    score_pull_request(db_session, pr.id, tracking_uri="sqlite:///:memory:")
    db_session.refresh(pr)

    assert pr.check_status == "completed"
    assert pr.check_conclusion == "neutral"
    assert "No trained risk model" in pr.check_summary


@pytest.mark.story("NFR-US-04")
@pytest.mark.story("NFR-US-05")
def test_score_pull_request_marks_unavailable_on_unexpected_failure(
    db_session: Session,
    champion_and_pr: tuple[str, PullRequest, Repository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C7: if BugFlow itself fails, the check must say so (neutral — never
    block a merge) rather than leave a stale or wrong result."""
    tracking_uri, pr, _repo = champion_and_pr

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated outage")

    monkeypatch.setattr(pr_scoring_service, "predict_commit", boom)

    with pytest.raises(RuntimeError):
        score_pull_request(db_session, pr.id, tracking_uri=tracking_uri)

    db_session.refresh(pr)
    assert pr.check_status == "completed"
    assert pr.check_conclusion == "neutral"
    assert "could not be computed" in pr.check_summary
