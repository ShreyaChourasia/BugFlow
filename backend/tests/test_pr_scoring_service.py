from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models.commit import Commit
from app.models.pull_request import LineRisk, PullRequest, RiskPrediction
from app.models.repository import MiningRun, Repository
from app.services import pr_scoring_service
from app.services.line_risk_training_service import train_and_register_champion as train_line_risk
from app.services.pr_scoring_service import _decide_conclusion, score_pull_request
from app.services.training_service import train_and_register_champion

from .conftest import make_commits_for_training
from .gitutil import commit_all, init_repo

RISKY_LINE = "except: pass\n"
SAFE_LINE = "return result\n"


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
            "lines_added": 1,
            "lines_deleted": 1,
            "churn": 2,
            "files_changed": 1,
            "directories_touched": 1,
            "subsystems_touched": 1,
            "entropy": 0.0,
            "author_prior_commits": 0,
            "is_fix": False,
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


# --- line-level highlights (US-13/US-14) -------------------------------------


@pytest.fixture
def champion_and_pr_with_line_risk(
    db_session: Session, tmp_path: Path
) -> tuple[str, PullRequest, Repository]:
    """A real local git repo (so `mine_specific_commits` can recover added-
    line text) with both a commit-risk and a line-risk champion trained, and
    `risk_threshold=0.0` so line-level analysis always triggers regardless of
    the commit-risk model's actual output."""
    # Line-risk training is global and re-clones every repo it finds — clear
    # out anything left in this shared dev Postgres by a manual demo (see the
    # same cleanup in conftest's repository_with_commits).
    db_session.execute(delete(LineRisk))
    db_session.execute(delete(RiskPrediction))
    db_session.execute(delete(PullRequest))
    db_session.execute(delete(Commit))
    db_session.execute(delete(MiningRun))
    db_session.execute(delete(Repository))

    repo_path = tmp_path / "repo"
    init_repo(repo_path)
    file_path = repo_path / "a.py"
    file_path.write_text("")

    # 150 commits, matching make_commits_for_training's count: commit-risk's
    # isotonic calibration does an internal 5-fold CV over the calib split
    # (15% of this), which needs more samples than line-risk's own 50-line
    # minimum would otherwise require here.
    shas = []
    for i in range(150):
        line = RISKY_LINE if i % 3 == 0 else SAFE_LINE
        with file_path.open("a") as f:
            f.write(line)
        shas.append(commit_all(repo_path, f"commit {i}"))

    repository = Repository(
        name="pr-line-risk-demo", url=str(repo_path), risk_threshold=0.0, line_risk_top_n=3
    )
    db_session.add(repository)
    db_session.commit()

    base = datetime(2024, 1, 1, tzinfo=UTC)
    features = {
        "lines_added": 4,
        "lines_deleted": 0,
        "churn": 4,
        "files_changed": 1,
        "directories_touched": 1,
        "subsystems_touched": 1,
        "entropy": 0.0,
        "author_prior_commits": 0,
        "is_fix": False,
    }
    for i, sha in enumerate(shas):
        db_session.add(
            Commit(
                sha=sha,
                repository_id=repository.id,
                message=f"commit {i}",
                timestamp=base + timedelta(minutes=i),
                is_bug_inducing=(i % 3 == 0),
                features=features,
            )
        )
    db_session.commit()

    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    train_and_register_champion(db_session, seed=42, tracking_uri=tracking_uri)
    train_line_risk(db_session, seed=42, tracking_uri=tracking_uri)

    pr = PullRequest(
        repository_id=repository.id, number=1, title="t", head_sha=shas[-1], base_sha=shas[0]
    )
    db_session.add(pr)
    db_session.commit()
    db_session.refresh(pr)
    return tracking_uri, pr, repository


@pytest.mark.story("US-13")
@pytest.mark.story("US-14")
def test_score_pull_request_persists_and_posts_line_highlights(
    db_session: Session, champion_and_pr_with_line_risk: tuple[str, PullRequest, Repository]
) -> None:
    tracking_uri, pr, _repo = champion_and_pr_with_line_risk

    score_pull_request(db_session, pr.id, tracking_uri=tracking_uri)
    db_session.refresh(pr)

    assert pr.check_status == "completed"
    prediction = db_session.query(RiskPrediction).filter(RiskPrediction.commit_id.isnot(None)).one()
    lines = (
        db_session.query(LineRisk)
        .filter(LineRisk.prediction_id == prediction.id)
        .order_by(LineRisk.rank)
        .all()
    )
    assert 1 <= len(lines) <= 3  # repository.line_risk_top_n
    assert "Riskiest lines" in pr.comment_body
    assert lines[0].reason


@pytest.mark.story("US-13")
def test_score_pull_request_reports_not_applicable_below_threshold(
    db_session: Session, champion_and_pr_with_line_risk: tuple[str, PullRequest, Repository]
) -> None:
    tracking_uri, pr, repository = champion_and_pr_with_line_risk
    repository.risk_threshold = 1.1  # nothing clears this
    db_session.commit()

    score_pull_request(db_session, pr.id, tracking_uri=tracking_uri)
    db_session.refresh(pr)

    assert "did not meet the threshold for line-level analysis" in pr.comment_body
