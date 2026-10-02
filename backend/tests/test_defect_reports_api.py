import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

import app.api.defect_reports as defect_reports_module
import app.services.defect_service as defect_service_module
from app.models.enums import Role
from app.models.repository import Repository

from .conftest import make_user


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    tokens = client.post("/auth/login", data={"username": email, "password": password}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


class _FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[object, tuple, dict]] = []

    def enqueue(self, func: object, *args: object, **kwargs: object) -> None:
        self.enqueued.append((func, args, kwargs))


@pytest.fixture
def fake_queue(monkeypatch: pytest.MonkeyPatch) -> _FakeQueue:
    queue = _FakeQueue()
    monkeypatch.setattr(defect_reports_module, "get_queue", lambda: queue)
    monkeypatch.setattr(defect_service_module, "get_queue", lambda: queue)
    return queue


@pytest.fixture
def repo(db_session: Session) -> Repository:
    # This shared dev Postgres accumulates real rows from manual/live
    # verification — clear defect-domain state so assertions here reflect
    # only what this test creates.
    db_session.execute(text("TRUNCATE defect_reports CASCADE"))
    repository = Repository(name="demo", url="/tmp/x")
    db_session.add(repository)
    db_session.commit()
    return repository


@pytest.mark.story("US-16")
def test_create_defect_report_requires_reporter_role(
    client: TestClient, db_session: Session, repo: Repository
) -> None:
    make_user(db_session, "dev@example.com", Role.DEVELOPER, password="s3cret")
    headers = _login(client, "dev@example.com", "s3cret")

    response = client.post(
        "/defect-reports",
        headers=headers,
        json={"repository_id": repo.id, "title": "Bug", "description": "Something broke"},
    )

    assert response.status_code == 403


@pytest.mark.story("US-16")
@pytest.mark.story("US-17")
def test_create_and_fetch_defect_report(
    client: TestClient, db_session: Session, repo: Repository
) -> None:
    make_user(db_session, "reporter1@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter1@example.com", "s3cret")

    create_response = client.post(
        "/defect-reports",
        headers=headers,
        json={
            "repository_id": repo.id,
            "title": "Login crash",
            "description": "The app crashes with a null pointer on login",
        },
    )
    assert create_response.status_code == 201
    report_id = create_response.json()["id"]

    get_response = client.get(f"/defect-reports/{report_id}", headers=headers)
    assert get_response.status_code == 200
    assert get_response.json()["title"] == "Login crash"
    assert get_response.json()["status"] == "open"


@pytest.mark.story("NFR-US-02")
def test_list_defect_reports_filters_by_status_and_respects_limit(
    client: TestClient, db_session: Session, repo: Repository
) -> None:
    # The `repo` fixture above already truncates defect_reports (TRUNCATE,
    # not DELETE: a row-by-row DELETE against a table this size with an HNSW
    # index attached was measured taking 10+ minutes — each deleted row
    # updates the index). This test additionally needs hundreds of thousands
    # of rows gone if scripts/load_test_defects.py has run against this same
    # dev Postgres — the fixture's truncate already covers that case too.

    make_user(db_session, "reporter7@example.com", Role.REPORTER, password="s3cret")
    reporter_headers = _login(client, "reporter7@example.com", "s3cret")
    make_user(db_session, "triager2@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager2@example.com", "s3cret")

    original = client.post(
        "/defect-reports",
        headers=reporter_headers,
        json={"repository_id": repo.id, "title": "Original", "description": "Something broke"},
    ).json()
    duplicate = client.post(
        "/defect-reports",
        headers=reporter_headers,
        json={"repository_id": repo.id, "title": "Dup", "description": "Same thing broke"},
    ).json()
    client.post(
        f"/defect-reports/{duplicate['id']}/merge",
        headers=triager_headers,
        json={"original_id": original["id"]},
    )

    open_response = client.get(
        "/defect-reports", params={"status": "open"}, headers=reporter_headers
    )
    assert open_response.status_code == 200
    ids = [r["id"] for r in open_response.json()]
    assert original["id"] in ids
    assert duplicate["id"] not in ids

    limited_response = client.get("/defect-reports", params={"limit": 1}, headers=reporter_headers)
    assert len(limited_response.json()) == 1


@pytest.mark.story("US-16")
def test_suggestions_endpoint_respects_min_word_count(
    client: TestClient, db_session: Session
) -> None:
    make_user(db_session, "reporter2@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter2@example.com", "s3cret")

    response = client.get("/defect-reports/suggestions", params={"text": "short"}, headers=headers)

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "results": []}


@pytest.mark.story("US-17")
@pytest.mark.story("US-18")
def test_duplicates_endpoint_returns_candidates_with_shared_phrases(
    client: TestClient, db_session: Session, repo: Repository
) -> None:
    make_user(db_session, "reporter3@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter3@example.com", "s3cret")

    first = client.post(
        "/defect-reports",
        headers=headers,
        json={
            "repository_id": repo.id,
            "title": "Login NPE",
            "description": "Clicking login causes a null pointer exception crash",
        },
    ).json()
    second = client.post(
        "/defect-reports",
        headers=headers,
        json={
            "repository_id": repo.id,
            "title": "Crash on login",
            "description": "The app crashes with a null pointer exception when clicking login",
        },
    ).json()

    response = client.get(f"/defect-reports/{second['id']}/duplicates", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["results"][0]["id"] == first["id"]
    assert "null pointer exception" in " ".join(body["results"][0]["shared_phrases"])


@pytest.mark.story("US-20")
def test_merge_requires_triager_role(
    client: TestClient, db_session: Session, repo: Repository
) -> None:
    make_user(db_session, "reporter4@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter4@example.com", "s3cret")
    original = client.post(
        "/defect-reports",
        headers=headers,
        json={"repository_id": repo.id, "title": "Original", "description": "Something broke"},
    ).json()
    duplicate = client.post(
        "/defect-reports",
        headers=headers,
        json={"repository_id": repo.id, "title": "Dup", "description": "Same thing broke"},
    ).json()

    response = client.post(
        f"/defect-reports/{duplicate['id']}/merge",
        headers=headers,
        json={"original_id": original["id"]},
    )

    assert response.status_code == 403


@pytest.mark.story("US-20")
def test_merge_marks_duplicate_and_enqueues_notification(
    client: TestClient, db_session: Session, repo: Repository, fake_queue: _FakeQueue
) -> None:
    make_user(db_session, "reporter5@example.com", Role.REPORTER, password="s3cret")
    reporter_headers = _login(client, "reporter5@example.com", "s3cret")
    make_user(db_session, "triager1@example.com", Role.TRIAGER, password="s3cret")
    triager_headers = _login(client, "triager1@example.com", "s3cret")

    original = client.post(
        "/defect-reports",
        headers=reporter_headers,
        json={"repository_id": repo.id, "title": "Original", "description": "Something broke"},
    ).json()
    duplicate = client.post(
        "/defect-reports",
        headers=reporter_headers,
        json={"repository_id": repo.id, "title": "Dup", "description": "Same thing broke"},
    ).json()

    response = client.post(
        f"/defect-reports/{duplicate['id']}/merge",
        headers=triager_headers,
        json={"original_id": original["id"]},
    )

    assert response.status_code == 200
    assert response.json()["duplicate_of_id"] == original["id"]
    assert response.json()["status"] == "duplicate"
    assert len(fake_queue.enqueued) == 1


@pytest.mark.story("NFR-US-02")
def test_rebuild_index_requires_admin_or_ml_engineer(
    client: TestClient, db_session: Session
) -> None:
    make_user(db_session, "reporter6@example.com", Role.REPORTER, password="s3cret")
    headers = _login(client, "reporter6@example.com", "s3cret")

    response = client.post("/defect-reports/index/rebuild", headers=headers)

    assert response.status_code == 403
