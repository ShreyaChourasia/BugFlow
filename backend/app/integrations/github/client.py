import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from urllib.parse import urlparse

import httpx
from bugflow_ml.mining.git_miner import ModifiedFileInfo

from app.core.config import get_settings
from app.integrations.github.auth import get_installation_token
from app.models.repository import Repository


def parse_owner_repo(url: str) -> tuple[str, str]:
    """"https://github.com/owner/repo(.git)" -> ("owner", "repo")."""
    path = urlparse(url).path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    owner, repo = path.split("/", 1)
    return owner, repo


@dataclass
class CommitDetails:
    sha: str
    author_name: str
    author_email: str
    message: str
    timestamp: datetime
    lines_added: int
    lines_deleted: int
    files: list[ModifiedFileInfo]
    is_merge: bool


class GitHubClient(Protocol):
    def create_check_run(self, repo: Repository, head_sha: str) -> str: ...

    def update_check_run(
        self, repo: Repository, check_run_id: str, *, conclusion: str, title: str, summary: str
    ) -> None: ...

    def upsert_comment(
        self, repo: Repository, pr_number: int, existing_comment_id: str | None, body: str
    ) -> str: ...

    def fetch_commit_details(self, repo: Repository, sha: str) -> CommitDetails: ...


class RealGitHubClient:
    """Talks to the real GitHub REST API, authenticated as the App's
    installation on this specific repository (least-privilege — see
    docs/github-app.md for the exact permissions requested)."""

    def __init__(
        self,
        api_base_url: str,
        app_id: str,
        private_key_path: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_base_url = api_base_url
        self._app_id = app_id
        self._private_key_path = private_key_path
        self._transport = transport  # injectable for tests; None uses real network

    def _client_for(self, repo: Repository) -> httpx.Client:
        assert repo.github_installation_id is not None
        # A separate, throwaway client for the token exchange: the one
        # returned below is handed to the caller as `with self._client_for(...)
        # as client:`, and httpx.Client refuses to be __enter__'d after
        # already being used for a request.
        with httpx.Client(
            base_url=self._api_base_url, timeout=10.0, transport=self._transport
        ) as token_client:
            token = get_installation_token(
                token_client, self._api_base_url, self._app_id, self._private_key_path,
                repo.github_installation_id,
            )

        client = httpx.Client(base_url=self._api_base_url, timeout=10.0, transport=self._transport)
        client.headers["Authorization"] = f"Bearer {token}"
        client.headers["Accept"] = "application/vnd.github+json"
        return client

    def create_check_run(self, repo: Repository, head_sha: str) -> str:
        owner, name = parse_owner_repo(repo.url)
        with self._client_for(repo) as client:
            response = client.post(
                f"/repos/{owner}/{name}/check-runs",
                json={
                    "name": "BugFlow risk assessment",
                    "head_sha": head_sha,
                    "status": "in_progress",
                },
            )
            response.raise_for_status()
            return str(response.json()["id"])

    def update_check_run(
        self, repo: Repository, check_run_id: str, *, conclusion: str, title: str, summary: str
    ) -> None:
        owner, name = parse_owner_repo(repo.url)
        with self._client_for(repo) as client:
            response = client.patch(
                f"/repos/{owner}/{name}/check-runs/{check_run_id}",
                json={
                    "status": "completed",
                    "conclusion": conclusion,
                    "output": {"title": title, "summary": summary},
                },
            )
            response.raise_for_status()

    def upsert_comment(
        self, repo: Repository, pr_number: int, existing_comment_id: str | None, body: str
    ) -> str:
        owner, name = parse_owner_repo(repo.url)
        with self._client_for(repo) as client:
            if existing_comment_id:
                response = client.patch(
                    f"/repos/{owner}/{name}/issues/comments/{existing_comment_id}",
                    json={"body": body},
                )
            else:
                response = client.post(
                    f"/repos/{owner}/{name}/issues/{pr_number}/comments", json={"body": body}
                )
            response.raise_for_status()
            return str(response.json()["id"])

    def fetch_commit_details(self, repo: Repository, sha: str) -> CommitDetails:
        owner, name = parse_owner_repo(repo.url)
        with self._client_for(repo) as client:
            response = client.get(f"/repos/{owner}/{name}/commits/{sha}")
            response.raise_for_status()
            data = response.json()

        commit = data["commit"]
        files = [
            ModifiedFileInfo(
                path=f["filename"],
                added_lines=f.get("additions", 0),
                deleted_lines=f.get("deletions", 0),
            )
            for f in data.get("files", [])
        ]
        return CommitDetails(
            sha=data["sha"],
            author_name=commit["author"]["name"],
            author_email=commit["author"]["email"],
            message=commit["message"],
            timestamp=datetime.fromisoformat(commit["author"]["date"].replace("Z", "+00:00")),
            lines_added=data.get("stats", {}).get("additions", 0),
            lines_deleted=data.get("stats", {}).get("deletions", 0),
            files=files,
            is_merge=len(data.get("parents", [])) > 1,
        )


class FakeGitHubClient:
    """The "fake Checks sink" for offline demos (§5): never makes a network
    call. Whatever it "posts" is only ever visible through PullRequest's own
    check_status/check_conclusion/check_summary/comment_body columns, which
    the caller (pr_scoring_service) always updates regardless of which
    client is in use."""

    def create_check_run(self, repo: Repository, head_sha: str) -> str:
        return f"fake-check-{uuid.uuid4().hex[:10]}"

    def update_check_run(
        self, repo: Repository, check_run_id: str, *, conclusion: str, title: str, summary: str
    ) -> None:
        pass

    def upsert_comment(
        self, repo: Repository, pr_number: int, existing_comment_id: str | None, body: str
    ) -> str:
        return existing_comment_id or f"fake-comment-{uuid.uuid4().hex[:10]}"

    def fetch_commit_details(self, repo: Repository, sha: str) -> CommitDetails:
        raise LookupError(
            "No GitHub API access configured — offline/replay mode can only "
            "score commits that have already been mined."
        )


def get_github_client(repo: Repository) -> GitHubClient:
    """A repo only gets real GitHub calls once both the App is configured
    (settings) and it's been installed on that specific repo
    (github_installation_id, set from the installation webhook payload).
    Otherwise everything routes through the fake sink — this is what makes
    the whole system usable offline by default."""
    settings = get_settings()
    if (
        settings.github_app_id
        and settings.github_app_private_key_path
        and repo.github_installation_id
    ):
        return RealGitHubClient(
            settings.github_api_base_url,
            settings.github_app_id,
            settings.github_app_private_key_path,
        )
    return FakeGitHubClient()
