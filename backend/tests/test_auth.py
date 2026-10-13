import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core import security as security_module
from app.models.enums import Role
from tests.conftest import make_user


@pytest.mark.story("US-01")
def test_login_returns_tokens(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "dev@example.com", Role.DEVELOPER, password="s3cret")

    response = client.post(
        "/auth/login", data={"username": "dev@example.com", "password": "s3cret"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] and body["refresh_token"]
    assert body["token_type"] == "bearer"


@pytest.mark.story("US-01")
def test_login_rejects_wrong_password(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "dev2@example.com", Role.DEVELOPER, password="s3cret")

    response = client.post(
        "/auth/login", data={"username": "dev2@example.com", "password": "wrong"}
    )

    assert response.status_code == 401


@pytest.mark.story("US-01")
def test_refresh_issues_new_access_token(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "dev3@example.com", Role.DEVELOPER, password="s3cret")
    tokens = client.post(
        "/auth/login", data={"username": "dev3@example.com", "password": "s3cret"}
    ).json()

    response = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})

    assert response.status_code == 200
    assert response.json()["access_token"]


@pytest.mark.story("US-01")
def test_me_returns_current_user(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "dev5@example.com", Role.DEVELOPER, password="s3cret")
    tokens = client.post(
        "/auth/login", data={"username": "dev5@example.com", "password": "s3cret"}
    ).json()

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})

    assert response.status_code == 200
    assert response.json()["email"] == "dev5@example.com"
    assert response.json()["role"] == "developer"


@pytest.mark.story("NFR-US-10")
def test_login_rate_limits_repeated_attempts(client: TestClient, db_session: Session) -> None:
    email = "rate-limited@example.com"
    make_user(db_session, email, Role.DEVELOPER, password="s3cret")
    security_module._redis.delete(f"login_attempts:{email}")

    try:
        for _ in range(5):
            response = client.post("/auth/login", data={"username": email, "password": "wrong"})
            assert response.status_code == 401

        blocked = client.post("/auth/login", data={"username": email, "password": "wrong"})
        assert blocked.status_code == 429
    finally:
        security_module._redis.delete(f"login_attempts:{email}")


@pytest.mark.story("US-01")
def test_logout_revokes_access_token(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "dev4@example.com", Role.DEVELOPER, password="s3cret")
    tokens = client.post(
        "/auth/login", data={"username": "dev4@example.com", "password": "s3cret"}
    ).json()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    logout_response = client.post("/auth/logout", headers=headers)
    assert logout_response.status_code == 204

    reused_response = client.get("/repositories", headers=headers)
    assert reused_response.status_code == 401
