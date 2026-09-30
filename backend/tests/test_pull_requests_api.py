from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models.commit import Commit
from app.models.decision import Explanation, Feedback
from app.models.enums import Role
from app.models.pull_request import LineRisk, PullRequest, RiskPrediction
from app.models.repository import Repository

from .conftest import make_user


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    tokens = client.post("/auth/login", data={"username": email, "password": password}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.mark.story("US-11")
def test_list_pull_requests_requires_auth(client: TestClient, db_session: Session) -> None:
    repo = Repository(name="demo", url="/tmp/x")
    db_session.add(repo)
    db_session.commit()

    response = client.get(f"/repositories/{repo.id}/pull-requests")

    assert response.status_code == 401


@pytest.mark.story("US-11")
def test_list_and_detail_pull_requests(client: TestClient, db_session: Session) -> None:
    repo = Repository(name="demo", url="/tmp/x")
    db_session.add(repo)
    db_session.commit()
    pr = PullRequest(
        repository_id=repo.id,
        number=7,
        title="Add feature",
        head_sha="sha1",
        base_sha="sha0",
        check_status="completed",
        check_conclusion="neutral",
        check_summary="Calibrated risk: 20%.",
        comment_body="### BugFlow risk assessment\n\nLow risk.",
    )
    db_session.add(pr)
    db_session.commit()

    make_user(db_session, "reviewer-pr@example.com", Role.REVIEWER, password="s3cret")
    headers = _login(client, "reviewer-pr@example.com", "s3cret")

    list_response = client.get(f"/repositories/{repo.id}/pull-requests", headers=headers)
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1
    assert list_response.json()[0]["number"] == 7

    detail_response = client.get(f"/repositories/{repo.id}/pull-requests/7", headers=headers)
    assert detail_response.status_code == 200
    body = detail_response.json()
    assert body["title"] == "Add feature"
    assert body["check_conclusion"] == "neutral"
    assert body["comment_body"].startswith("### BugFlow")
    assert body["risk"] is None  # no RiskPrediction stored yet


@pytest.mark.story("US-11")
def test_pull_request_detail_includes_risk_when_available(
    client: TestClient, db_session: Session
) -> None:
    repo = Repository(name="demo", url="/tmp/x")
    db_session.add(repo)
    db_session.commit()
    commit = Commit(
        sha="sha1",
        repository_id=repo.id,
        message="msg",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        features={},
    )
    db_session.add(commit)
    db_session.commit()
    pr = PullRequest(repository_id=repo.id, number=1, title="t", head_sha="sha1", base_sha="sha0")
    db_session.add(pr)
    db_session.commit()

    prediction = RiskPrediction(
        commit_id=commit.id,
        model_version="v1",
        probability=0.6,
        calibrated_probability=0.55,
        confidence=0.1,
        risk_level="medium",
        latency_ms=10.0,
    )
    db_session.add(prediction)
    db_session.flush()
    factor = {"feature": "churn", "value": 10, "shap_value": 0.2, "impact": "increases"}
    db_session.add(
        Explanation(
            decision_type="RiskPrediction",
            decision_id=prediction.id,
            text="Explanation text.",
            factors={"items": [factor]},
        )
    )
    db_session.commit()

    make_user(db_session, "reviewer-pr2@example.com", Role.REVIEWER, password="s3cret")
    headers = _login(client, "reviewer-pr2@example.com", "s3cret")

    response = client.get(f"/repositories/{repo.id}/pull-requests/1", headers=headers)

    assert response.status_code == 200
    risk = response.json()["risk"]
    assert risk["risk_level"] == "medium"
    assert risk["explanation_text"] == "Explanation text."
    assert len(risk["factors"]) == 1


def test_pull_request_detail_404s_for_unknown_number(
    client: TestClient, db_session: Session
) -> None:
    repo = Repository(name="demo", url="/tmp/x")
    db_session.add(repo)
    db_session.commit()
    make_user(db_session, "reviewer-pr3@example.com", Role.REVIEWER, password="s3cret")
    headers = _login(client, "reviewer-pr3@example.com", "s3cret")

    response = client.get(f"/repositories/{repo.id}/pull-requests/999", headers=headers)

    assert response.status_code == 404


@pytest.mark.story("US-13")
def test_pull_request_detail_reports_not_applicable_below_threshold(
    client: TestClient, db_session: Session
) -> None:
    repo = Repository(name="demo", url="/tmp/x", risk_threshold=0.7)
    db_session.add(repo)
    db_session.commit()
    commit = Commit(
        sha="sha1",
        repository_id=repo.id,
        message="msg",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        features={},
    )
    db_session.add(commit)
    db_session.commit()
    pr = PullRequest(repository_id=repo.id, number=1, title="t", head_sha="sha1", base_sha="sha0")
    db_session.add(pr)
    db_session.commit()
    prediction = RiskPrediction(
        commit_id=commit.id,
        model_version="v1",
        probability=0.2,
        calibrated_probability=0.2,
        confidence=0.5,
        risk_level="low",
        latency_ms=10.0,
    )
    db_session.add(prediction)
    db_session.flush()
    db_session.add(
        Explanation(
            decision_type="RiskPrediction", decision_id=prediction.id, text="Low risk.", factors={}
        )
    )
    db_session.commit()

    make_user(db_session, "reviewer-lr1@example.com", Role.REVIEWER, password="s3cret")
    headers = _login(client, "reviewer-lr1@example.com", "s3cret")

    response = client.get(f"/repositories/{repo.id}/pull-requests/1", headers=headers)

    risk = response.json()["risk"]
    assert risk["line_risk_status"] == "not_applicable"
    assert risk["lines"] == []


@pytest.mark.story("US-13")
@pytest.mark.story("US-14")
def test_pull_request_detail_includes_highlighted_lines_when_available(
    client: TestClient, db_session: Session
) -> None:
    repo = Repository(name="demo", url="/tmp/x", risk_threshold=0.5)
    db_session.add(repo)
    db_session.commit()
    commit = Commit(
        sha="sha1",
        repository_id=repo.id,
        message="msg",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        features={},
    )
    db_session.add(commit)
    db_session.commit()
    pr = PullRequest(repository_id=repo.id, number=1, title="t", head_sha="sha1", base_sha="sha0")
    db_session.add(pr)
    db_session.commit()
    prediction = RiskPrediction(
        commit_id=commit.id,
        model_version="v1",
        probability=0.9,
        calibrated_probability=0.9,
        confidence=0.8,
        risk_level="high",
        latency_ms=10.0,
    )
    db_session.add(prediction)
    db_session.flush()
    db_session.add(
        Explanation(
            decision_type="RiskPrediction", decision_id=prediction.id, text="High risk.", factors={}
        )
    )
    line = LineRisk(
        prediction_id=prediction.id,
        file_path="a.py",
        line_no=3,
        code="except: pass",
        risk_score=0.95,
        rank=1,
        reason="except (+1.20)",
    )
    db_session.add(line)
    db_session.commit()
    db_session.refresh(line)

    make_user(db_session, "reviewer-lr2@example.com", Role.REVIEWER, password="s3cret")
    headers = _login(client, "reviewer-lr2@example.com", "s3cret")

    response = client.get(f"/repositories/{repo.id}/pull-requests/1", headers=headers)

    risk = response.json()["risk"]
    assert risk["line_risk_status"] == "available"
    assert len(risk["lines"]) == 1
    assert risk["lines"][0]["code"] == "except: pass"
    assert risk["lines"][0]["marked_false_alarm"] is False

    false_alarm_response = client.post(f"/line-risks/{line.id}/false-alarm", headers=headers)
    assert false_alarm_response.status_code == 204

    db_session.refresh(line)
    assert line.marked_false_alarm is True
    feedback = db_session.query(Feedback).filter(Feedback.decision_id == line.id).one()
    assert feedback.decision_type == "LineRisk"
    assert feedback.accepted is False


def test_false_alarm_requires_auth(client: TestClient) -> None:
    response = client.post("/line-risks/1/false-alarm")
    assert response.status_code == 401


def test_false_alarm_404s_for_unknown_line(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "reviewer-lr3@example.com", Role.REVIEWER, password="s3cret")
    headers = _login(client, "reviewer-lr3@example.com", "s3cret")

    response = client.post("/line-risks/999999/false-alarm", headers=headers)

    assert response.status_code == 404


@pytest.mark.story("US-09")
def test_review_queue_sorts_open_prs_by_calibrated_risk(
    client: TestClient, db_session: Session
) -> None:
    # /review-queue is intentionally global (no repository filter) — clear
    # out any open PRs left in this shared dev Postgres by a manual demo so
    # this test only sees the two PRs it creates below.
    db_session.execute(delete(LineRisk))
    db_session.execute(delete(RiskPrediction))
    db_session.execute(delete(PullRequest))
    db_session.execute(delete(Commit))
    db_session.commit()

    repo = Repository(name="demo", url="/tmp/x")
    db_session.add(repo)
    db_session.commit()

    low_commit = Commit(
        sha="low",
        repository_id=repo.id,
        message="m",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        lines_added=5,
        lines_deleted=1,
        files_changed=1,
        features={},
    )
    high_commit = Commit(
        sha="high",
        repository_id=repo.id,
        message="m",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        lines_added=200,
        lines_deleted=50,
        files_changed=10,
        features={},
    )
    db_session.add_all([low_commit, high_commit])
    db_session.commit()

    low_pr = PullRequest(
        repository_id=repo.id, number=1, title="low", head_sha="low", base_sha="base"
    )
    high_pr = PullRequest(
        repository_id=repo.id, number=2, title="high", head_sha="high", base_sha="base"
    )
    closed_pr = PullRequest(
        repository_id=repo.id,
        number=3,
        title="closed",
        head_sha="low",
        base_sha="base",
        status="closed",
    )
    db_session.add_all([low_pr, high_pr, closed_pr])
    db_session.commit()

    db_session.add_all(
        [
            RiskPrediction(
                commit_id=low_commit.id,
                model_version="v1",
                probability=0.1,
                calibrated_probability=0.1,
                confidence=0.5,
                risk_level="low",
                latency_ms=1.0,
            ),
            RiskPrediction(
                commit_id=high_commit.id,
                model_version="v1",
                probability=0.9,
                calibrated_probability=0.9,
                confidence=0.9,
                risk_level="high",
                latency_ms=1.0,
            ),
        ]
    )
    db_session.commit()

    make_user(db_session, "reviewer-rq@example.com", Role.REVIEWER, password="s3cret")
    headers = _login(client, "reviewer-rq@example.com", "s3cret")

    response = client.get("/review-queue", headers=headers)

    assert response.status_code == 200
    body = response.json()
    numbers = [item["number"] for item in body]
    assert numbers == [2, 1]  # high risk first, closed PR excluded
    assert body[0]["lines_added"] == 200
    assert body[0]["files_changed"] == 10
