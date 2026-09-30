#!/usr/bin/env python3
"""Offline demo (§5): replays a PR event for an already-mined commit into our
own webhook endpoint, so the whole pre-merge flow — check posted, comment
posted, risk + explanation computed — can be demoed without internet or a
real GitHub App. It's a real webhook request (properly HMAC-signed) hitting
the real endpoint; only the "GitHub" on the other end of the write-back is
fake (app.integrations.github.client.FakeGitHubClient).

Usage (inside the api container, which has `app` installed):
    docker compose exec api python /app/scripts/replay_pr_events.py \
        --repository-id 1 --pr-number 101 --sha <a previously mined sha>
"""

import argparse
import hashlib
import hmac
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import httpx  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.db import SessionLocal  # noqa: E402
from app.models.commit import Commit  # noqa: E402
from app.models.repository import Repository  # noqa: E402


def build_payload(repo: Repository, commit: Commit, pr_number: int, action: str) -> dict:
    return {
        "action": action,
        "pull_request": {
            "number": pr_number,
            "title": commit.message.splitlines()[0][:200],
            "head": {"sha": commit.sha},
            "base": {"sha": commit.sha},  # base doesn't matter for scoring
        },
        "repository": {
            "full_name": repo.url.rsplit("/", 1)[-1],
            "html_url": repo.url,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-id", type=int, required=True)
    parser.add_argument("--sha", required=True, help="A sha already mined for this repository")
    parser.add_argument("--pr-number", type=int, default=1)
    parser.add_argument("--action", choices=["opened", "synchronize"], default="opened")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.github_webhook_secret:
        print("GITHUB_WEBHOOK_SECRET is not set — put any shared secret in .env first.")
        sys.exit(1)

    db = SessionLocal()
    try:
        repo = db.get(Repository, args.repository_id)
        if repo is None:
            print(f"No repository with id {args.repository_id}")
            sys.exit(1)
        commit = db.scalar(
            select(Commit).where(Commit.repository_id == repo.id, Commit.sha == args.sha)
        )
        if commit is None:
            print(f"Commit {args.sha} hasn't been mined for repository {repo.id} yet.")
            sys.exit(1)
        payload = build_payload(repo, commit, args.pr_number, args.action)
    finally:
        db.close()

    body = json.dumps(payload).encode()
    signature = hmac.new(settings.github_webhook_secret.encode(), body, hashlib.sha256).hexdigest()

    response = httpx.post(
        f"{args.base_url}/webhooks/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": f"sha256={signature}",
        },
    )
    print(f"{response.status_code}: {response.text}")
    print(
        f"Check the result at GET {args.base_url}/repositories/{args.repository_id}"
        f"/pull-requests/{args.pr_number} once the worker's picked it up."
    )


if __name__ == "__main__":
    main()
