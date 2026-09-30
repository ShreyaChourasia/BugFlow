from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.commit import Commit
from app.models.decision import Explanation
from app.models.enums import Role
from app.models.pull_request import PullRequest, RiskPrediction
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
