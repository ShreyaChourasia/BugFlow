from datetime import datetime

import pytest
from bugflow_ml.models.resolver_suitability import DeveloperCandidate, ScoredCandidate
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

import app.services.resolver_service as resolver_service_module
from app.models.decision import Feedback
from app.models.defect import Assignment
from app.models.developer import Developer
from app.models.enums import Role
from app.models.system import SystemConfig

from .conftest import make_user


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    tokens = client.post("/auth/login", data={"username": email, "password": password}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.fixture
def repo(db_session: Session):
    from app.models.repository import Repository

    db_session.execute(
        text(
            "TRUNCATE feedback, explanations, assignments, resolver_recommendations, "
            "defect_reports, developers, system_config CASCADE"
        )
    )
    repository = Repository(name="demo", url="/tmp/x")
    db_session.add(repository)
    db_session.commit()
    return repository


def _create_report(client: TestClient, headers: dict[str, str], repo, description: str) -> dict:
    return client.post(
        "/defect-reports",
        headers=headers,
        json={
            "repository_id": repo.id,
            "title": "Bug",
            "description": description,
            "component": "login",
        },
    ).json()


def _rank_by_skill(
    db: Session,
    report_embedding,
    report_component: str | None,
    candidates: list[DeveloperCandidate],
    past_resolutions_by_developer,
    as_of: datetime,
    tracking_uri: str | None = None,
) -> list[ScoredCandidate]:
    """Deterministic stand-in for the trained model, same role as triage
    API tests' `_fake_predict_triage`: skill-matched developers rank first,
    everyone is cold-start since no real history is wired up in these tests."""
    scored = [
        ScoredCandidate(
            developer_id=c.id,
            score=0.8 if report_component in c.skills else 0.1,
            confidence=0.8 if report_component in c.skills else 0.1,
            is_cold_start=True,
            reason=f"Matches skill: {report_component}"
            if report_component in c.skills
            else "No matching skill on file",
        )
        for c in candidates
    ]
    return sorted(scored, key=lambda s: s.score, reverse=True)


@pytest.fixture
def fake_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resolver_service_module.resolver_prediction_service, "get_candidates", _rank_by_skill
    )


@pytest.mark.story("US-24")
def test_resolver_recommendations_unavailable_with_no_developers(
    client: TestClient, db_session: Session, repo
) -> None:
    make_user(db_session, "reporter-r1@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-r1@example.com", "s3cret")
    report = _create_report(client, headers, repo, "the login page fails on submit")
    make_user(db_session, "triager-r1@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager-r1@example.com", "s3cret")

    response = client.get(
        f"/defect-reports/{report['id']}/resolver-recommendations", headers=triager_headers
    )

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"


@pytest.mark.story("US-24")
@pytest.mark.story("US-30")
def test_resolver_recommendations_available_and_ranked(
    client: TestClient, db_session: Session, repo, fake_candidates: None
) -> None:
    make_user(db_session, "reporter-r1b@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-r1b@example.com", "s3cret")
    db_session.add_all(
        [
            Developer(name="Matches", skills=["login"], capacity=5, current_queue_depth=0),
            Developer(name="No match 1", skills=["billing"], capacity=5, current_queue_depth=0),
            Developer(name="No match 2", skills=["search"], capacity=5, current_queue_depth=0),
        ]
    )
    db_session.commit()
    report = _create_report(client, headers, repo, "the login page fails on submit")
    make_user(db_session, "triager-r1b@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager-r1b@example.com", "s3cret")

    response = client.get(
        f"/defect-reports/{report['id']}/resolver-recommendations", headers=triager_headers
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "available"
    assert len(body["candidates"]) >= 3
    assert body["candidates"][0]["developer_name"] == "Matches"
    assert body["candidates"][0]["is_cold_start"] is True


@pytest.mark.story("US-24")
def test_resolver_recommendations_no_confident_candidate_routes_to_default_owner(
    client: TestClient, db_session: Session, repo, fake_candidates: None
) -> None:
    make_user(db_session, "reporter-r2@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-r2@example.com", "s3cret")
    db_session.add(Developer(name="Dev", skills=["billing"], capacity=5, current_queue_depth=0))
    owner = make_user(db_session, "owner-r2@example.com", Role.TRIAGER, password="s3cret")
    db_session.add(SystemConfig(key="default_triage_owner_id", value=str(owner.id)))
    db_session.commit()
    report = _create_report(client, headers, repo, "the login page fails on submit for everyone")
    triager_headers = _login(client, "owner-r2@example.com", "s3cret")

    response = client.get(
        f"/defect-reports/{report['id']}/resolver-recommendations", headers=triager_headers
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_confident_candidate"
    assert body["default_owner_id"] == owner.id


@pytest.mark.story("US-29")
def test_override_assignment_without_reason_is_rejected(
    client: TestClient, db_session: Session, repo
) -> None:
    make_user(db_session, "reporter-r3@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-r3@example.com", "s3cret")
    developer = Developer(name="Dev", skills=["login"], capacity=5, current_queue_depth=0)
    db_session.add(developer)
    db_session.commit()
    report = _create_report(client, headers, repo, "the login page fails on submit")
    make_user(db_session, "triager-r3@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager-r3@example.com", "s3cret")

    response = client.post(
        f"/defect-reports/{report['id']}/assignment",
        headers=triager_headers,
        json={"developer_id": developer.id, "reason": "   "},
    )

    assert response.status_code == 422


@pytest.mark.story("US-29")
def test_override_assignment_with_reason_assigns_and_records_feedback(
    client: TestClient, db_session: Session, repo
) -> None:
    make_user(db_session, "reporter-r4@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-r4@example.com", "s3cret")
    developer = Developer(name="Dev", skills=["login"], capacity=5, current_queue_depth=0)
    db_session.add(developer)
    db_session.commit()
    report = _create_report(client, headers, repo, "the login page fails on submit")
    make_user(db_session, "triager-r4@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager-r4@example.com", "s3cret")

    response = client.post(
        f"/defect-reports/{report['id']}/assignment",
        headers=triager_headers,
        json={"developer_id": developer.id, "reason": "best fit for the login component"},
    )

    assert response.status_code == 200
    assert response.json()["assignee_id"] == developer.id
    feedback = (
        db_session.query(Feedback).filter(Feedback.decision_type == "ResolverRecommendation").one()
    )
    assert feedback.reason == "best fit for the login component"
    db_session.refresh(developer)
    assert developer.current_queue_depth == 1


@pytest.mark.story("US-25")
@pytest.mark.story("US-26")
def test_batch_assign_respects_capacity_and_reports_shortfall(
    client: TestClient, db_session: Session, repo, fake_candidates: None
) -> None:
    make_user(db_session, "reporter-r5@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-r5@example.com", "s3cret")
    developer = Developer(name="Dev", skills=["login"], capacity=1, current_queue_depth=0)
    db_session.add(developer)
    db_session.commit()
    make_user(db_session, "manager-r5@example.com", Role.MANAGER, password="s3cret")
    manager_headers = _login(client, "manager-r5@example.com", "s3cret")

    report_a = _create_report(client, headers, repo, "the login page fails on submit")
    report_b = _create_report(client, headers, repo, "the login page also fails on logout")

    response = client.post(
        "/assignments/batch",
        headers=manager_headers,
        json={"defect_ids": [report_a["id"], report_b["id"]]},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["assignments"]) == 1
    assert len(body["shortfall"]) == 1


@pytest.mark.story("US-26")
def test_batch_assign_reports_shortfall_for_nonexistent_defect_ids(
    client: TestClient, db_session: Session, repo, fake_candidates: None
) -> None:
    make_user(db_session, "manager-r5b@example.com", Role.MANAGER, password="s3cret")
    manager_headers = _login(client, "manager-r5b@example.com", "s3cret")

    response = client.post(
        "/assignments/batch", headers=manager_headers, json={"defect_ids": [999999]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assignments"] == []
    assert body["shortfall"] == [999999]


@pytest.mark.story("US-27")
def test_objection_requires_own_assignment(client: TestClient, db_session: Session, repo) -> None:
    make_user(db_session, "reporter-r6@example.com", Role.REPORTER, password="s3cret")
    reporter_headers = _login(client, "reporter-r6@example.com", "s3cret")
    report = _create_report(client, reporter_headers, repo, "the login page fails on submit")
    developer = Developer(name="Dev", skills=["login"], capacity=5, current_queue_depth=0)
    other_developer = Developer(name="Other", skills=["login"], capacity=5, current_queue_depth=0)
    db_session.add_all([developer, other_developer])
    db_session.commit()
    dev_user = make_user(db_session, "dev-r6@example.com", Role.DEVELOPER, password="s3cret")
    dev_user.developer_id = other_developer.id
    db_session.commit()
    dev_headers = _login(client, "dev-r6@example.com", "s3cret")

    assignment = Assignment(defect_id=report["id"], developer_id=developer.id, source="auto")
    db_session.add(assignment)
    db_session.commit()

    response = client.post(
        f"/assignments/{assignment.id}/objection",
        headers=dev_headers,
        json={"reason": "I don't own this component"},
    )

    assert response.status_code == 403


@pytest.mark.story("US-27")
def test_objection_on_own_assignment_records_feedback(
    client: TestClient, db_session: Session, repo
) -> None:
    make_user(db_session, "reporter-r7@example.com", Role.REPORTER, password="s3cret")
    reporter_headers = _login(client, "reporter-r7@example.com", "s3cret")
    report = _create_report(client, reporter_headers, repo, "the login page fails on submit")
    developer = Developer(name="Dev", skills=["login"], capacity=5, current_queue_depth=0)
    db_session.add(developer)
    db_session.commit()
    dev_user = make_user(db_session, "dev-r7@example.com", Role.DEVELOPER, password="s3cret")
    dev_user.developer_id = developer.id
    db_session.commit()
    dev_headers = _login(client, "dev-r7@example.com", "s3cret")

    assignment = Assignment(defect_id=report["id"], developer_id=developer.id, source="auto")
    db_session.add(assignment)
    db_session.commit()

    response = client.post(
        f"/assignments/{assignment.id}/objection",
        headers=dev_headers,
        json={"reason": "I don't own this component"},
    )

    assert response.status_code == 204
    feedback = db_session.query(Feedback).filter(Feedback.decision_type == "Assignment").one()
    assert feedback.accepted is False
    assert feedback.reason == "I don't own this component"


@pytest.mark.story("US-28")
def test_workload_distribution_requires_manager_role(
    client: TestClient, db_session: Session
) -> None:
    make_user(db_session, "reporter-r8@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter-r8@example.com", "s3cret")

    response = client.get("/assignments/workload", headers=headers)

    assert response.status_code == 403


@pytest.mark.story("US-28")
def test_workload_distribution_returns_developers(
    client: TestClient, db_session: Session, repo
) -> None:
    db_session.add(Developer(name="Dev", skills=["login"], capacity=5, current_queue_depth=2))
    db_session.commit()
    make_user(db_session, "manager-r8@example.com", Role.MANAGER, password="s3cret")
    manager_headers = _login(client, "manager-r8@example.com", "s3cret")

    response = client.get("/assignments/workload", headers=manager_headers)

    assert response.status_code == 200
    body = response.json()
    assert body[0]["capacity"] == 5
    assert body[0]["current_queue_depth"] == 2


@pytest.mark.story("US-27")
def test_my_assignments_requires_a_linked_developer(
    client: TestClient, db_session: Session
) -> None:
    make_user(db_session, "dev-r9@example.com", Role.DEVELOPER, password="s3cret")
    dev_headers = _login(client, "dev-r9@example.com", "s3cret")

    response = client.get("/assignments/mine", headers=dev_headers)

    assert response.status_code == 403


@pytest.mark.story("US-27")
def test_my_assignments_surfaces_the_recommendation_reason(
    client: TestClient, db_session: Session, repo
) -> None:
    make_user(db_session, "reporter-r10@example.com", Role.REPORTER, password="s3cret")
    reporter_headers = _login(client, "reporter-r10@example.com", "s3cret")
    developer = Developer(name="Dev", skills=["login"], capacity=5, current_queue_depth=0)
    db_session.add(developer)
    db_session.commit()
    dev_user = make_user(db_session, "dev-r10@example.com", Role.DEVELOPER, password="s3cret")
    dev_user.developer_id = developer.id
    db_session.commit()
    dev_headers = _login(client, "dev-r10@example.com", "s3cret")

    report = _create_report(client, reporter_headers, repo, "the login page fails on submit")
    make_user(db_session, "triager-r10@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager-r10@example.com", "s3cret")
    client.post(
        f"/defect-reports/{report['id']}/assignment",
        headers=triager_headers,
        json={"developer_id": developer.id, "reason": "only option available"},
    )

    response = client.get("/assignments/mine", headers=dev_headers)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["defect_id"] == report["id"]
    assert body[0]["reason"] == "Manually assigned, no reason on file."
