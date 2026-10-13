from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import app.api.risk as risk_module
from app.models.commit import Commit
from app.models.decision import Explanation
from app.models.enums import Role
from app.models.pull_request import RiskPrediction
from app.models.repository import Repository

from .conftest import make_user


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    tokens = client.post("/auth/login", data={"username": email, "password": password}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def _make_commit(db_session: Session) -> Commit:
    repo = Repository(name="demo", url="/tmp/does-not-need-to-exist")
    db_session.add(repo)
    db_session.commit()
    commit = Commit(
        sha="abc123",
        repository_id=repo.id,
        message="a commit",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        features={
            "churn": 10,
            "files_changed": 2,
            "lines_added": 5,
            "lines_deleted": 5,
            "directories_touched": 1,
            "subsystems_touched": 1,
            "entropy": 0.5,
            "author_prior_commits": 3,
            "is_fix": False,
        },
    )
    db_session.add(commit)
    db_session.commit()
    db_session.refresh(commit)
    return commit


@pytest.mark.story("US-12")
def test_predict_commit_requires_auth(client: TestClient, db_session: Session) -> None:
    commit = _make_commit(db_session)

    response = client.post(
        "/predict/commit", json={"repository_id": commit.repository_id, "sha": commit.sha}
    )

    assert response.status_code == 401


@pytest.mark.story("US-12")
def test_predict_commit_404s_for_an_unknown_commit(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "dev-risk@example.com", Role.DEVELOPER, password="s3cret")
    headers = _login(client, "dev-risk@example.com", "s3cret")

    response = client.post(
        "/predict/commit", json={"repository_id": 999999, "sha": "nope"}, headers=headers
    )

    assert response.status_code == 404


@pytest.mark.story("US-12")
@pytest.mark.story("US-08")
def test_predict_commit_response_always_includes_an_explanation(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C3: no automated decision is ever shown without a plain-language
    explanation. The scoring/explanation logic itself is exercised for real
    in test_prediction_service.py; here we're verifying the API always
    surfaces it, using a stand-in for the (expensive, model-dependent)
    prediction call."""
    commit = _make_commit(db_session)
    make_user(db_session, "dev-risk2@example.com", Role.DEVELOPER, password="s3cret")
    headers = _login(client, "dev-risk2@example.com", "s3cret")

    def fake_predict_commit(
        db: Session, commit_arg: Commit, tracking_uri: str | None = None
    ) -> tuple[RiskPrediction, Explanation]:
        rp = RiskPrediction(
            id=1,
            commit_id=commit_arg.id,
            model_version="test-v1",
            probability=0.7,
            calibrated_probability=0.65,
            confidence=0.3,
            risk_level="medium",
            latency_ms=12.3,
            created_at=datetime.now(UTC),
        )
        explanation = Explanation(
            id=1,
            decision_type="RiskPrediction",
            decision_id=1,
            text="This change touches 2 files; changes spread across files are riskier.",
            factors={
                "items": [
                    {
                        "feature": "files_changed",
                        "value": 2,
                        "shap_value": 0.1,
                        "impact": "increases",
                    }
                ]
            },
        )
        return rp, explanation

    monkeypatch.setattr(risk_module, "predict_commit", fake_predict_commit)

    response = client.post(
        "/predict/commit",
        json={"repository_id": commit.repository_id, "sha": commit.sha},
        headers=headers,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["risk_level"] == "medium"
    assert body["explanation"]["text"]
    assert len(body["explanation"]["factors"]) >= 1


@pytest.mark.story("US-12")
def test_get_commit_risk_404s_before_any_prediction_exists(
    client: TestClient, db_session: Session
) -> None:
    commit = _make_commit(db_session)
    make_user(db_session, "dev-risk3@example.com", Role.DEVELOPER, password="s3cret")
    headers = _login(client, "dev-risk3@example.com", "s3cret")

    response = client.get(
        f"/commits/{commit.sha}/risk",
        params={"repository_id": commit.repository_id},
        headers=headers,
    )

    assert response.status_code == 404


@pytest.mark.story("US-12")
def test_get_commit_risk_returns_the_latest_prediction(
    client: TestClient, db_session: Session
) -> None:
    commit = _make_commit(db_session)
    make_user(db_session, "dev-risk4@example.com", Role.DEVELOPER, password="s3cret")
    headers = _login(client, "dev-risk4@example.com", "s3cret")

    risk_prediction = RiskPrediction(
        commit_id=commit.id,
        model_version="v1",
        probability=0.4,
        calibrated_probability=0.35,
        confidence=0.3,
        risk_level="low",
        latency_ms=5.0,
    )
    db_session.add(risk_prediction)
    db_session.flush()
    db_session.add(
        Explanation(
            decision_type="RiskPrediction",
            decision_id=risk_prediction.id,
            text="Low risk change.",
            factors={"items": []},
        )
    )
    db_session.commit()

    response = client.get(
        f"/commits/{commit.sha}/risk",
        params={"repository_id": commit.repository_id},
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] == "low"
    assert body["explanation"]["text"] == "Low risk change."


@pytest.mark.story("US-45")
def test_get_commit_risk_ignores_shadow_rows(client: TestClient, db_session: Session) -> None:
    commit = _make_commit(db_session)
    make_user(db_session, "dev-risk5@example.com", Role.DEVELOPER, password="s3cret")
    headers = _login(client, "dev-risk5@example.com", "s3cret")

    real = RiskPrediction(
        commit_id=commit.id,
        model_version="champion-v1",
        probability=0.4,
        calibrated_probability=0.35,
        confidence=0.3,
        risk_level="low",
        latency_ms=5.0,
    )
    db_session.add(real)
    db_session.flush()
    db_session.add(
        Explanation(
            decision_type="RiskPrediction",
            decision_id=real.id,
            text="Low risk change.",
            factors={"items": []},
        )
    )
    # Inserted after the real row, so a naive "latest by created_at" query
    # would pick this shadow row instead — exactly the bug US-45 introduced
    # and this test guards against.
    db_session.add(
        RiskPrediction(
            commit_id=commit.id,
            model_version="challenger-v1",
            probability=0.9,
            calibrated_probability=0.9,
            confidence=0.8,
            risk_level="high",
            latency_ms=5.0,
            is_shadow=True,
        )
    )
    db_session.commit()

    response = client.get(
        f"/commits/{commit.sha}/risk",
        params={"repository_id": commit.repository_id},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["risk_level"] == "low"
    assert response.json()["model_version"] == "champion-v1"


@pytest.mark.story("US-35")
def test_get_commit_counterfactual_returns_the_search_result(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = _make_commit(db_session)
    make_user(db_session, "dev-risk6@example.com", Role.DEVELOPER, password="s3cret")
    headers = _login(client, "dev-risk6@example.com", "s3cret")

    from bugflow_ml.explain.counterfactual import CounterfactualResult

    def fake_get_counterfactual(db, commit_arg, risk_threshold, tracking_uri=None):
        return CounterfactualResult(
            found=True,
            original_probability=0.8,
            new_probability=0.4,
            reduction_fraction=0.3,
            text="Splitting this into smaller commits would lower risk from 80% to 40%.",
        )

    monkeypatch.setattr(risk_module, "get_counterfactual", fake_get_counterfactual)

    response = client.get(
        f"/commits/{commit.sha}/counterfactual",
        params={"repository_id": commit.repository_id},
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["new_probability"] == 0.4
    assert "Splitting" in body["text"]


@pytest.mark.story("US-35")
def test_get_commit_counterfactual_503s_without_a_trained_model(
    client: TestClient, db_session: Session
) -> None:
    commit = _make_commit(db_session)
    make_user(db_session, "dev-risk7@example.com", Role.DEVELOPER, password="s3cret")
    headers = _login(client, "dev-risk7@example.com", "s3cret")

    response = client.get(
        f"/commits/{commit.sha}/counterfactual",
        params={"repository_id": commit.repository_id},
        headers=headers,
    )

    assert response.status_code == 503
