from datetime import UTC, datetime, timedelta

import pytest
from bugflow_ml.models.resolution_forecast import ForecastPrediction
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

import app.services.forecast_prediction_service as forecast_prediction_service_module
from app.models.defect import DefectReport
from app.models.enums import Role

from .conftest import make_user


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    tokens = client.post("/auth/login", data={"username": email, "password": password}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.fixture
def repo(db_session: Session):
    from app.models.repository import Repository

    db_session.execute(text("TRUNCATE explanations, resolution_forecasts, defect_reports CASCADE"))
    repository = Repository(name="demo", url="/tmp/x")
    db_session.add(repository)
    db_session.commit()
    return repository


def _create_report(client: TestClient, headers: dict[str, str], repo, description: str) -> dict:
    return client.post(
        "/defect-reports",
        headers=headers,
        json={"repository_id": repo.id, "title": "Bug", "description": description},
    ).json()


_FAKE_PREDICTION = ForecastPrediction(
    median_days=3.0,
    p90_days=10.0,
    curve=[(0.0, 1.0), (5.0, 0.6), (10.0, 0.2)],
    confidence=0.75,
)


@pytest.fixture
def fake_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        forecast_prediction_service_module,
        "get_forecast",
        lambda db, severity, priority, component, tracking_uri=None: _FAKE_PREDICTION,
    )


@pytest.mark.story("US-31")
def test_forecast_unavailable_without_a_trained_model_or_severity(
    client: TestClient, db_session: Session, repo
) -> None:
    make_user(db_session, "reporter-f1@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-f1@example.com", "s3cret")
    report = _create_report(client, headers, repo, "too short")

    response = client.get(f"/defect-reports/{report['id']}/forecast", headers=headers)

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"


@pytest.mark.story("US-31")
@pytest.mark.story("US-32")
def test_forecast_available_with_estimate_and_curve(
    client: TestClient, db_session: Session, repo, fake_forecast: None
) -> None:
    make_user(db_session, "reporter-f2@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-f2@example.com", "s3cret")
    make_user(db_session, "triager-f2@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager-f2@example.com", "s3cret")
    report = _create_report(
        client, headers, repo, "a sufficiently long description of a real problem"
    )
    client.post(
        f"/defect-reports/{report['id']}/triage",
        headers=triager_headers,
        json={"severity": "blocker", "priority": "P1"},
    )

    response = client.get(f"/defect-reports/{report['id']}/forecast", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "available"
    assert body["median_days"] == 3.0
    assert body["p90_days"] == 10.0
    assert body["estimate_text"] == "Likely within 3-10 days"
    assert len(body["curve"]) == 3
    assert body["at_risk"] is False


@pytest.mark.story("US-33")
def test_forecast_at_risk_for_an_overdue_open_report(
    client: TestClient, db_session: Session, repo, fake_forecast: None
) -> None:
    make_user(db_session, "reporter-f3@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-f3@example.com", "s3cret")
    make_user(db_session, "triager-f3@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager-f3@example.com", "s3cret")
    report = _create_report(
        client, headers, repo, "a sufficiently long description of a real problem"
    )
    client.post(
        f"/defect-reports/{report['id']}/triage",
        headers=triager_headers,
        json={"severity": "blocker", "priority": "P1"},
    )
    db_report = db_session.get(DefectReport, report["id"])
    db_report.reported_at = datetime.now(UTC) - timedelta(days=20)  # past the fake p90 of 10
    db_session.commit()

    response = client.get(f"/defect-reports/{report['id']}/forecast", headers=headers)

    assert response.status_code == 200
    assert response.json()["at_risk"] is True


@pytest.mark.story("US-33")
def test_forecast_never_at_risk_once_resolved(
    client: TestClient, db_session: Session, repo, fake_forecast: None
) -> None:
    make_user(db_session, "reporter-f4@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-f4@example.com", "s3cret")
    make_user(db_session, "triager-f4@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager-f4@example.com", "s3cret")
    report = _create_report(
        client, headers, repo, "a sufficiently long description of a real problem"
    )
    client.post(
        f"/defect-reports/{report['id']}/triage",
        headers=triager_headers,
        json={"severity": "blocker", "priority": "P1"},
    )
    db_report = db_session.get(DefectReport, report["id"])
    db_report.reported_at = datetime.now(UTC) - timedelta(days=20)
    db_report.resolved_at = datetime.now(UTC) - timedelta(days=19)
    db_session.commit()

    response = client.get(f"/defect-reports/{report['id']}/forecast", headers=headers)

    assert response.status_code == 200
    assert response.json()["at_risk"] is False
