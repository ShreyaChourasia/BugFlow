from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.commit import Commit
from app.models.decision import Explanation, Feedback
from app.models.defect import DefectReport
from app.models.enums import Role
from app.models.ml import MLModel
from app.models.pull_request import RiskPrediction
from app.models.repository import Repository

from .conftest import make_user


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    tokens = client.post("/auth/login", data={"username": email, "password": password}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.fixture
def repo(db_session: Session) -> Repository:
    db_session.execute(
        text(
            "TRUNCATE feedback, explanations, risk_predictions, defect_reports, "
            "commits, ml_models CASCADE"
        )
    )
    repository = Repository(name="demo", url="/tmp/x")
    db_session.add(repository)
    db_session.commit()
    return repository


@pytest.mark.story("US-37")
def test_analytics_requires_manager_or_qa_role(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "reporter-a1@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-a1@example.com", "s3cret")

    response = client.get("/analytics/risk-trends", headers=headers)

    assert response.status_code == 403


@pytest.mark.story("US-38")
def test_risk_trends_averages_per_week(client: TestClient, db_session: Session, repo) -> None:
    make_user(db_session, "manager-a1@example.com", Role.MANAGER, password="s3cret")
    headers = _login(client, "manager-a1@example.com", "s3cret")
    commit = Commit(
        sha="a1", repository_id=repo.id, message="x", timestamp=datetime.now(UTC), features={}
    )
    db_session.add(commit)
    db_session.commit()
    now = datetime.now(UTC)
    db_session.add_all(
        [
            RiskPrediction(
                commit_id=commit.id,
                model_version="v1",
                probability=0.2,
                calibrated_probability=0.2,
                confidence=0.6,
                risk_level="low",
                latency_ms=1.0,
                created_at=now,
            ),
            RiskPrediction(
                commit_id=commit.id,
                model_version="v1",
                probability=0.8,
                calibrated_probability=0.8,
                confidence=0.6,
                risk_level="high",
                latency_ms=1.0,
                created_at=now,
            ),
            # A shadow row with an extreme value — must not pull the mean.
            RiskPrediction(
                commit_id=commit.id,
                model_version="v2",
                probability=0.99,
                calibrated_probability=0.99,
                confidence=0.9,
                risk_level="high",
                latency_ms=1.0,
                created_at=now,
                is_shadow=True,
            ),
        ]
    )
    db_session.commit()

    response = client.get("/analytics/risk-trends", headers=headers)

    assert response.status_code == 200
    points = response.json()["points"]
    assert len(points) == 1
    assert points[0]["count"] == 2
    assert points[0]["mean_calibrated_probability"] == pytest.approx(0.5)


@pytest.mark.story("US-39")
def test_acceptance_rate_per_decision_type(client: TestClient, db_session: Session, repo) -> None:
    reporter = make_user(db_session, "reporter-a2@example.com", Role.REPORTER, password="s3cret")
    make_user(db_session, "manager-a2@example.com", Role.MANAGER, password="s3cret")
    headers = _login(client, "manager-a2@example.com", "s3cret")

    db_session.add_all(
        [
            Feedback(
                decision_type="TriageAssessment", decision_id=1, user_id=reporter.id, accepted=True
            ),
            Feedback(
                decision_type="TriageAssessment",
                decision_id=2,
                user_id=reporter.id,
                accepted=False,
            ),
            Feedback(
                decision_type="ResolverRecommendation",
                decision_id=1,
                user_id=reporter.id,
                accepted=True,
            ),
        ]
    )
    db_session.commit()

    response = client.get("/analytics/acceptance-rate", headers=headers)

    assert response.status_code == 200
    entries = {e["decision_type"]: e for e in response.json()["entries"]}
    assert entries["TriageAssessment"]["total"] == 2
    assert entries["TriageAssessment"]["acceptance_rate"] == pytest.approx(0.5)
    assert entries["ResolverRecommendation"]["acceptance_rate"] == pytest.approx(1.0)
    assert response.json()["overall_rate"] == pytest.approx(2 / 3)


@pytest.mark.story("US-40")
def test_hotspots_counts_by_component(client: TestClient, db_session: Session, repo) -> None:
    reporter = make_user(db_session, "reporter-a3@example.com", Role.REPORTER, password="s3cret")
    make_user(db_session, "manager-a3@example.com", Role.MANAGER, password="s3cret")
    headers = _login(client, "manager-a3@example.com", "s3cret")

    db_session.add_all(
        [
            DefectReport(
                repository_id=repo.id,
                reporter_id=reporter.id,
                title="a",
                description="a",
                component="login",
            ),
            DefectReport(
                repository_id=repo.id,
                reporter_id=reporter.id,
                title="b",
                description="b",
                component="login",
            ),
            DefectReport(
                repository_id=repo.id, reporter_id=reporter.id, title="c", description="c"
            ),
        ]
    )
    db_session.commit()

    response = client.get("/analytics/hotspots", headers=headers)

    assert response.status_code == 200
    entries = {e["component"]: e["defect_count"] for e in response.json()["entries"]}
    assert entries["login"] == 2
    assert entries["uncategorized"] == 1


@pytest.mark.story("US-37")
def test_top_risk_factors_counts_only_increasing_factors(
    client: TestClient, db_session: Session, repo
) -> None:
    make_user(db_session, "manager-a4@example.com", Role.MANAGER, password="s3cret")
    headers = _login(client, "manager-a4@example.com", "s3cret")

    db_session.add_all(
        [
            Explanation(
                decision_type="RiskPrediction",
                decision_id=1,
                text="x",
                factors={
                    "items": [
                        {"feature": "churn", "value": 5, "shap_value": 0.3, "impact": "increases"},
                        {
                            "feature": "is_fix",
                            "value": 1,
                            "shap_value": -0.1,
                            "impact": "decreases",
                        },
                    ]
                },
            ),
            Explanation(
                decision_type="RiskPrediction",
                decision_id=2,
                text="x",
                factors={
                    "items": [
                        {"feature": "churn", "value": 7, "shap_value": 0.5, "impact": "increases"}
                    ]
                },
            ),
        ]
    )
    db_session.commit()

    response = client.get("/analytics/top-risk-factors", headers=headers)

    assert response.status_code == 200
    entries = response.json()["entries"]
    assert entries[0]["feature"] == "churn"
    assert entries[0]["occurrence_count"] == 2
    assert entries[0]["mean_shap_value"] == pytest.approx(0.4)
    assert all(e["feature"] != "is_fix" for e in entries)


@pytest.mark.story("NFR-US-11")
def test_fairness_unavailable_without_a_champion(
    client: TestClient, db_session: Session, repo
) -> None:
    make_user(db_session, "manager-a5@example.com", Role.MANAGER, password="s3cret")
    headers = _login(client, "manager-a5@example.com", "s3cret")

    response = client.get("/analytics/fairness", headers=headers)

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"


@pytest.mark.story("NFR-US-11")
def test_fairness_surfaces_the_stored_training_time_report(
    client: TestClient, db_session: Session, repo
) -> None:
    make_user(db_session, "manager-a6@example.com", Role.MANAGER, password="s3cret")
    headers = _login(client, "manager-a6@example.com", "s3cret")
    db_session.add(
        MLModel(
            task="commit_risk",
            version="run-1",
            mlflow_run_id="run-1",
            stage="champion",
            metrics={
                "fairness": {
                    "junior_mean_risk": 0.6,
                    "senior_mean_risk": 0.3,
                    "junior_count": 10,
                    "senior_count": 20,
                    "disparity": 0.3,
                }
            },
            data_version="dv1",
            seed=42,
        )
    )
    db_session.commit()

    response = client.get("/analytics/fairness", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "available"
    assert body["disparity"] == pytest.approx(0.3)
