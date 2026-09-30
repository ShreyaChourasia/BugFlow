import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.enums import Role
from tests.conftest import make_user


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    tokens = client.post("/auth/login", data={"username": email, "password": password}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.mark.story("US-01")
def test_non_admin_cannot_create_repository(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "notadmin@example.com", Role.DEVELOPER, password="s3cret")
    headers = _login(client, "notadmin@example.com", "s3cret")

    response = client.post(
        "/repositories",
        json={"name": "demo", "url": "https://github.com/example/demo"},
        headers=headers,
    )

    assert response.status_code == 403


@pytest.mark.story("US-01")
def test_admin_can_create_repository(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "admin@example.com", Role.ADMIN, password="s3cret")
    headers = _login(client, "admin@example.com", "s3cret")

    response = client.post(
        "/repositories",
        json={"name": "demo", "url": "https://github.com/example/demo"},
        headers=headers,
    )

    assert response.status_code == 201
    assert response.json()["name"] == "demo"


@pytest.mark.story("US-49")
def test_non_admin_cannot_edit_system_config(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "qa@example.com", Role.QA, password="s3cret")
    headers = _login(client, "qa@example.com", "s3cret")

    response = client.put("/admin/config/some_setting", json={"value": "true"}, headers=headers)

    assert response.status_code == 403


@pytest.mark.story("US-49")
def test_admin_can_edit_system_config(client: TestClient, db_session: Session) -> None:
    make_user(db_session, "admin2@example.com", Role.ADMIN, password="s3cret")
    headers = _login(client, "admin2@example.com", "s3cret")

    response = client.put("/admin/config/some_setting", json={"value": "true"}, headers=headers)

    assert response.status_code == 200
    assert response.json() == {"key": "some_setting", "value": "true"}
