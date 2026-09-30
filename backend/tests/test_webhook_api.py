import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

import app.api.webhooks as webhooks_module
from app.core.config import get_settings
from app.models.pull_request import PullRequest
from app.models.repository import Repository

WEBHOOK_SECRET = "test-webhook-secret"


class _FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[object, tuple, dict]] = []

    def enqueue(self, func: object, *args: object, **kwargs: object) -> None:
        self.enqueued.append((func, args, kwargs))


@pytest.fixture
def webhook_secret(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(get_settings(), "github_webhook_secret", WEBHOOK_SECRET)
    return WEBHOOK_SECRET


@pytest.fixture
def fake_queue(monkeypatch: pytest.MonkeyPatch) -> _FakeQueue:
    queue = _FakeQueue()
    monkeypatch.setattr(webhooks_module, "get_queue", lambda: queue)
    return queue


def _sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _pull_request_payload(
    action: str, number: int, head_sha: str, repo_html_url: str
) -> bytes:
    return json.dumps(
        {
            "action": action,
            "pull_request": {
                "number": number,
                "title": "A test PR",
                "head": {"sha": head_sha},
                "base": {"sha": "basesha"},
            },
            "repository": {"full_name": "owner/repo", "html_url": repo_html_url},
        }
    ).encode()


def _post(client: TestClient, body: bytes, secret: str, event: str = "pull_request") -> object:
    return client.post(
        "/webhooks/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": event,
            "X-Hub-Signature-256": _sign(body, secret),
        },
    )


@pytest.mark.story("NFR-US-01")
def test_webhook_rejects_an_invalid_signature(client: TestClient, webhook_secret: str) -> None:
    body = _pull_request_payload("opened", 1, "sha1", "https://github.com/owner/repo")

    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": "sha256=wrong",
        },
    )

    assert response.status_code == 401


def test_webhook_requires_a_configured_secret(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "github_webhook_secret", "")
    body = _pull_request_payload("opened", 1, "sha1", "https://github.com/owner/repo")

    response = client.post(
        "/webhooks/github",
        content=body,
        headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": "sha256=whatever"},
    )

    assert response.status_code == 503


def test_webhook_ignores_non_pull_request_events(client: TestClient, webhook_secret: str) -> None:
    body = _pull_request_payload("opened", 1, "sha1", "https://github.com/owner/repo")

    response = _post(client, body, webhook_secret, event="push")

    assert response.json()["status"] == "ignored"


def test_webhook_ignores_unhandled_actions(client: TestClient, webhook_secret: str) -> None:
    body = _pull_request_payload("closed", 1, "sha1", "https://github.com/owner/repo")

    response = _post(client, body, webhook_secret)

    assert response.json()["status"] == "ignored"


def test_webhook_ignores_unregistered_repositories(client: TestClient, webhook_secret: str) -> None:
    body = _pull_request_payload("opened", 1, "sha1", "https://github.com/owner/not-registered")

    response = _post(client, body, webhook_secret)

    assert response.json()["status"] == "ignored"


@pytest.mark.story("US-46")
@pytest.mark.story("NFR-US-01")
def test_webhook_opened_creates_pr_with_pending_check_and_enqueues_scoring(
    client: TestClient, db_session: Session, webhook_secret: str, fake_queue: _FakeQueue
) -> None:
    repo = Repository(name="demo", url="https://github.com/owner/repo")
    db_session.add(repo)
    db_session.commit()

    body = _pull_request_payload("opened", 42, "sha-abc", "https://github.com/owner/repo")
    response = _post(client, body, webhook_secret)

    assert response.status_code == 202
    pr = db_session.scalar(
        select(PullRequest).where(PullRequest.repository_id == repo.id, PullRequest.number == 42)
    )
    assert pr is not None
    assert pr.head_sha == "sha-abc"
    assert pr.check_status == "pending"
    assert pr.check_run_id is not None
    assert len(fake_queue.enqueued) == 1
    assert fake_queue.enqueued[0][1] == (pr.id,)


@pytest.mark.story("US-46")
def test_webhook_synchronize_creates_a_fresh_check_run_for_the_new_sha(
    client: TestClient, db_session: Session, webhook_secret: str, fake_queue: _FakeQueue
) -> None:
    repo = Repository(name="demo", url="https://github.com/owner/repo")
    db_session.add(repo)
    db_session.commit()

    first = _post(
        client,
        _pull_request_payload("opened", 42, "sha-1", "https://github.com/owner/repo"),
        webhook_secret,
    )
    pr_id = first.json()["pull_request_id"]
    first_pr = db_session.get(PullRequest, pr_id)
    first_check_run_id = first_pr.check_run_id

    _post(
        client,
        _pull_request_payload("synchronize", 42, "sha-2", "https://github.com/owner/repo"),
        webhook_secret,
    )

    db_session.refresh(first_pr)
    assert first_pr.head_sha == "sha-2"
    assert first_pr.check_run_id != first_check_run_id
    assert first_pr.check_status == "pending"
    assert len(fake_queue.enqueued) == 2
