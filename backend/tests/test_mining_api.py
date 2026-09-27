import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.enums import Role
from app.models.repository import Repository

from .conftest import make_user


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    tokens = client.post("/auth/login", data={"username": email, "password": password}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def _make_repo(db_session: Session) -> Repository:
    repo = Repository(name="demo", url="/tmp/does-not-need-to-exist-for-this-test")
    db_session.add(repo)
    db_session.commit()
    db_session.refresh(repo)
    return repo


@pytest.mark.story("US-02")
def test_developer_cannot_start_mining_run(client: TestClient, db_session: Session) -> None:
    repo = _make_repo(db_session)
    make_user(db_session, "dev-mining@example.com", Role.DEVELOPER, password="s3cret")
    headers = _login(client, "dev-mining@example.com", "s3cret")

    response = client.post(f"/repositories/{repo.id}/mining-runs", headers=headers)

    assert response.status_code == 403


@pytest.mark.story("US-02")
def test_ml_engineer_can_start_mining_run(client: TestClient, db_session: Session) -> None:
    repo = _make_repo(db_session)
    make_user(db_session, "mle@example.com", Role.ML_ENGINEER, password="s3cret")
    headers = _login(client, "mle@example.com", "s3cret")

    response = client.post(f"/repositories/{repo.id}/mining-runs", headers=headers)

    assert response.status_code == 201
    body = response.json()
    assert body["repository_id"] == repo.id
    assert body["status"] == "pending"


@pytest.mark.story("US-05")
def test_resume_rejects_a_run_that_is_not_failed(client: TestClient, db_session: Session) -> None:
    repo = _make_repo(db_session)
    make_user(db_session, "admin-mining@example.com", Role.ADMIN, password="s3cret")
    headers = _login(client, "admin-mining@example.com", "s3cret")

    started = client.post(f"/repositories/{repo.id}/mining-runs", headers=headers).json()

    response = client.post(
        f"/repositories/{repo.id}/mining-runs/{started['id']}/resume", headers=headers
    )

    assert response.status_code == 409
