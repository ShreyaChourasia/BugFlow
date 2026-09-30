from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

from app.integrations.github.auth import create_app_jwt, get_installation_token
from app.integrations.github.client import (
    FakeGitHubClient,
    RealGitHubClient,
    get_github_client,
    parse_owner_repo,
)
from app.integrations.github.signature import verify_signature
from app.models.repository import Repository

# --- signature verification --------------------------------------------


def test_verify_signature_accepts_a_correctly_signed_payload() -> None:
    import hashlib
    import hmac

    body = b'{"hello": "world"}'
    secret = "s3cret"
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    assert verify_signature(body, signature, secret) is True


def test_verify_signature_rejects_a_tampered_payload() -> None:
    import hashlib
    import hmac

    secret = "s3cret"
    signature = "sha256=" + hmac.new(secret.encode(), b"original", hashlib.sha256).hexdigest()

    assert verify_signature(b"tampered", signature, secret) is False


@pytest.mark.parametrize("header", [None, "", "not-sha256=abc", "sha256="])
def test_verify_signature_rejects_malformed_headers(header: str | None) -> None:
    assert verify_signature(b"body", header, "secret") is False


# --- owner/repo parsing ---------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/owner/repo", ("owner", "repo")),
        ("https://github.com/owner/repo.git", ("owner", "repo")),
        ("https://github.com/owner/repo/", ("owner", "repo")),
    ],
)
def test_parse_owner_repo(url: str, expected: tuple[str, str]) -> None:
    assert parse_owner_repo(url) == expected


# --- GitHub App auth -------------------------------------------------------


@pytest.fixture
def rsa_private_key_path(tmp_path: Path) -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path = tmp_path / "key.pem"
    path.write_bytes(pem)
    return str(path)


def test_create_app_jwt_has_the_expected_claims(rsa_private_key_path: str) -> None:
    token = create_app_jwt("12345", rsa_private_key_path)

    claims = jwt.get_unverified_claims(token)
    assert claims["iss"] == "12345"
    assert claims["exp"] > claims["iat"]


def test_get_installation_token_exchanges_the_app_jwt(rsa_private_key_path: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/app/installations/999/access_tokens"
        assert request.headers["Authorization"].startswith("Bearer ")
        return httpx.Response(201, json={"token": "installation-token-abc"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    token = get_installation_token(
        client, "https://api.github.com", "12345", rsa_private_key_path, "999"
    )

    assert token == "installation-token-abc"


# --- RealGitHubClient -------------------------------------------------------


def _mock_transport(handlers: dict[str, httpx.Response]) -> httpx.MockTransport:
    """Maps "METHOD path" -> canned response; also always answers the
    installation-token exchange so callers don't need to special-case it."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "fake-installation-token"})
        key = f"{request.method} {request.url.path}"
        if key not in handlers:
            raise AssertionError(f"Unexpected request: {key}")
        return handlers[key]

    return httpx.MockTransport(handler)


API_BASE = "https://api.github.com"


def _real_client(rsa_private_key_path: str, transport: httpx.MockTransport) -> RealGitHubClient:
    return RealGitHubClient(API_BASE, "1", rsa_private_key_path, transport=transport)


@pytest.fixture
def github_repo() -> Repository:
    return Repository(
        id=1, name="demo", url="https://github.com/owner/repo", github_installation_id="999"
    )


def test_real_client_creates_a_check_run(
    rsa_private_key_path: str, github_repo: Repository
) -> None:
    transport = _mock_transport(
        {"POST /repos/owner/repo/check-runs": httpx.Response(201, json={"id": 42})}
    )
    client = _real_client(rsa_private_key_path, transport)

    check_run_id = client.create_check_run(github_repo, "abc123")

    assert check_run_id == "42"


def test_real_client_updates_a_check_run(
    rsa_private_key_path: str, github_repo: Repository
) -> None:
    transport = _mock_transport(
        {"PATCH /repos/owner/repo/check-runs/42": httpx.Response(200, json={"id": 42})}
    )
    client = _real_client(rsa_private_key_path, transport)

    client.update_check_run(
        github_repo, "42", conclusion="success", title="Low risk", summary="..."
    )


def test_real_client_creates_then_edits_a_comment(
    rsa_private_key_path: str, github_repo: Repository
) -> None:
    transport = _mock_transport(
        {
            "POST /repos/owner/repo/issues/7/comments": httpx.Response(201, json={"id": 555}),
            "PATCH /repos/owner/repo/issues/comments/555": httpx.Response(200, json={"id": 555}),
        }
    )
    client = _real_client(rsa_private_key_path, transport)

    created_id = client.upsert_comment(github_repo, 7, None, "first version")
    edited_id = client.upsert_comment(github_repo, 7, created_id, "edited version")

    assert created_id == "555"
    assert edited_id == "555"


def test_real_client_fetches_commit_details(
    rsa_private_key_path: str, github_repo: Repository
) -> None:
    author = {"name": "A Dev", "email": "a@example.com", "date": "2024-01-01T00:00:00Z"}
    transport = _mock_transport(
        {
            "GET /repos/owner/repo/commits/abc123": httpx.Response(
                200,
                json={
                    "sha": "abc123",
                    "commit": {"author": author, "message": "fixes #1"},
                    "stats": {"additions": 10, "deletions": 2},
                    "files": [{"filename": "a.py", "additions": 10, "deletions": 2}],
                    "parents": [{"sha": "parent1"}],
                },
            )
        }
    )
    client = _real_client(rsa_private_key_path, transport)

    details = client.fetch_commit_details(github_repo, "abc123")

    assert details.author_email == "a@example.com"
    assert details.lines_added == 10
    assert details.is_merge is False
    assert len(details.files) == 1


# --- FakeGitHubClient / factory ---------------------------------------------


def test_fake_client_never_makes_a_network_call() -> None:
    client = FakeGitHubClient()
    repo = Repository(id=1, name="demo", url="https://github.com/owner/repo")

    check_run_id = client.create_check_run(repo, "sha")
    client.update_check_run(repo, check_run_id, conclusion="neutral", title="t", summary="s")
    comment_id = client.upsert_comment(repo, 1, None, "body")
    edited_id = client.upsert_comment(repo, 1, comment_id, "body 2")

    assert check_run_id.startswith("fake-check-")
    assert edited_id == comment_id  # reused, not a new one

    with pytest.raises(LookupError):
        client.fetch_commit_details(repo, "sha")


def test_get_github_client_falls_back_to_fake_without_full_configuration() -> None:
    # No github_installation_id on the repo -> fake, regardless of app config.
    repo = Repository(id=1, name="demo", url="https://github.com/owner/repo")
    assert isinstance(get_github_client(repo), FakeGitHubClient)
