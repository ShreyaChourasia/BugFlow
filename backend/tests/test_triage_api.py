import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

import app.services.triage_service as triage_service_module
from app.models.enums import Role
from app.models.repository import Repository
from app.services.triage_prediction_service import TargetPrediction

from .conftest import make_user


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    tokens = client.post("/auth/login", data={"username": email, "password": password}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.fixture
def repo(db_session: Session) -> Repository:
    # `feedback`/`explanations` are polymorphic with no real FK to
    # defect_reports/triage_assessments, so CASCADE from defect_reports alone
    # wouldn't clear stray rows this shared dev Postgres picks up from manual
    # live verification — truncate them directly too.
    db_session.execute(text("TRUNCATE feedback, explanations, defect_reports CASCADE"))
    repository = Repository(name="demo", url="/tmp/x")
    db_session.add(repository)
    db_session.commit()
    return repository


def _fake_predict_triage(
    db: Session, text: str, tracking_uri: str | None = None
) -> tuple[TargetPrediction, TargetPrediction]:
    return (
        TargetPrediction(label="blocker", confidence=0.9, top_words=["crash", "data"]),
        TargetPrediction(label="P1", confidence=0.8, top_words=["urgent"]),
    )


@pytest.fixture
def fake_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triage_service_module, "predict_triage", _fake_predict_triage)


def _create_report(
    client: TestClient, headers: dict[str, str], repo: Repository, description: str
) -> dict:
    return client.post(
        "/defect-reports",
        headers=headers,
        json={"repository_id": repo.id, "title": "Bug", "description": description},
    ).json()


@pytest.mark.story("US-21")
def test_triage_suggestion_abstains_on_short_description(
    client: TestClient, db_session: Session, repo: Repository
) -> None:
    make_user(db_session, "reporter-t1@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-t1@example.com", "s3cret")
    report = _create_report(client, headers, repo, "too short")

    response = client.get(f"/defect-reports/{report['id']}/triage-suggestion", headers=headers)

    assert response.status_code == 200
    assert response.json() == {
        "status": "abstained",
        "severity": None,
        "priority": None,
        "severity_confidence": None,
        "priority_confidence": None,
        "severity_top_words": None,
        "priority_top_words": None,
        "is_automated": True,
    }


@pytest.mark.story("US-21")
def test_triage_suggestion_unavailable_when_no_champion_trained(
    client: TestClient, db_session: Session, repo: Repository
) -> None:
    make_user(db_session, "reporter-t2@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-t2@example.com", "s3cret")
    report = _create_report(
        client, headers, repo, "a sufficiently long description of a real problem happening"
    )

    response = client.get(f"/defect-reports/{report['id']}/triage-suggestion", headers=headers)

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"


@pytest.mark.story("US-21")
@pytest.mark.story("US-22")
def test_triage_suggestion_available_and_cached(
    client: TestClient, db_session: Session, repo: Repository, fake_predict: None
) -> None:
    make_user(db_session, "reporter-t3@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-t3@example.com", "s3cret")
    report = _create_report(
        client, headers, repo, "a sufficiently long description of a real problem happening"
    )

    first = client.get(f"/defect-reports/{report['id']}/triage-suggestion", headers=headers)
    assert first.status_code == 200
    body = first.json()
    assert body["status"] == "available"
    assert body["severity"] == "blocker"
    assert body["priority"] == "P1"
    assert body["severity_top_words"] == ["crash", "data"]

    # Cached — a second view doesn't recompute (the fixture would still work
    # either way, but a fresh TriageAssessment row would mean it wasn't).
    second = client.get(f"/defect-reports/{report['id']}/triage-suggestion", headers=headers)
    assert second.json() == body


@pytest.mark.story("US-23")
def test_decide_triage_accepting_the_suggestion_marks_feedback_accepted(
    client: TestClient, db_session: Session, repo: Repository, fake_predict: None
) -> None:
    make_user(db_session, "reporter-t4@example.com", Role.REPORTER, password="s3cret")
    reporter_headers = _login(client, "reporter-t4@example.com", "s3cret")
    make_user(db_session, "triager-t1@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager-t1@example.com", "s3cret")

    report = _create_report(
        client, reporter_headers, repo, "a sufficiently long description of a real problem"
    )
    client.get(f"/defect-reports/{report['id']}/triage-suggestion", headers=reporter_headers)

    response = client.post(
        f"/defect-reports/{report['id']}/triage",
        headers=triager_headers,
        json={"severity": "blocker", "priority": "P1"},
    )

    assert response.status_code == 200
    assert response.json()["severity"] == "blocker"
    assert response.json()["priority"] == "P1"

    suggestion = client.get(
        f"/defect-reports/{report['id']}/triage-suggestion", headers=reporter_headers
    )
    assert suggestion.json()["status"] == "decided"


@pytest.mark.story("US-23")
def test_decide_triage_overriding_the_suggestion_marks_feedback_not_accepted(
    client: TestClient, db_session: Session, repo: Repository, fake_predict: None
) -> None:
    make_user(db_session, "reporter-t5@example.com", Role.REPORTER, password="s3cret")
    reporter_headers = _login(client, "reporter-t5@example.com", "s3cret")
    make_user(db_session, "triager-t2@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager-t2@example.com", "s3cret")

    report = _create_report(
        client, reporter_headers, repo, "a sufficiently long description of a real problem"
    )
    client.get(f"/defect-reports/{report['id']}/triage-suggestion", headers=reporter_headers)

    response = client.post(
        f"/defect-reports/{report['id']}/triage",
        headers=triager_headers,
        json={"severity": "trivial", "priority": "P5"},
    )

    assert response.status_code == 200
    assert response.json()["severity"] == "trivial"

    from app.models.decision import Feedback

    feedback = db_session.query(Feedback).filter(Feedback.decision_type == "TriageAssessment").one()
    assert feedback.accepted is False
    assert feedback.original_value == "blocker/P1"
    assert feedback.new_value == "trivial/P5"


@pytest.mark.story("US-23")
def test_decide_triage_requires_triager_role(
    client: TestClient, db_session: Session, repo: Repository, fake_predict: None
) -> None:
    make_user(db_session, "reporter-t6@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-t6@example.com", "s3cret")
    report = _create_report(client, headers, repo, "a sufficiently long real problem description")

    response = client.post(
        f"/defect-reports/{report['id']}/triage",
        headers=headers,
        json={"severity": "blocker", "priority": "P1"},
    )

    assert response.status_code == 403


@pytest.mark.story("US-23")
def test_misclassification_report_counts_corrections_by_suggested_value(
    client: TestClient, db_session: Session, repo: Repository, fake_predict: None
) -> None:
    make_user(db_session, "reporter-t7@example.com", Role.REPORTER, password="s3cret")
    reporter_headers = _login(client, "reporter-t7@example.com", "s3cret")
    make_user(db_session, "triager-t3@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager-t3@example.com", "s3cret")

    report = _create_report(
        client, reporter_headers, repo, "a sufficiently long description of a real problem"
    )
    client.get(f"/defect-reports/{report['id']}/triage-suggestion", headers=reporter_headers)
    client.post(
        f"/defect-reports/{report['id']}/triage",
        headers=triager_headers,
        json={"severity": "trivial", "priority": "P5"},
    )

    response = client.get("/defect-reports/misclassification-report", headers=triager_headers)

    assert response.status_code == 200
    body = response.json()
    severity_bucket = next(b for b in body["severity"] if b["suggested"] == "blocker")
    assert severity_bucket["corrected_count"] == 1
    assert severity_bucket["total_count"] == 1
    assert severity_bucket["correction_rate"] == 1.0


def test_misclassification_report_requires_appropriate_role(
    client: TestClient, db_session: Session
) -> None:
    make_user(db_session, "reporter-t8@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-t8@example.com", "s3cret")

    response = client.get("/defect-reports/misclassification-report", headers=headers)

    assert response.status_code == 403
