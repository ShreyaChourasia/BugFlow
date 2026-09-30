from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import get_db
from app.core.queue import get_queue
from app.integrations.github.client import get_github_client
from app.integrations.github.signature import verify_signature
from app.models.pull_request import PullRequest
from app.models.repository import Repository
from app.schemas.webhook import PullRequestWebhookEvent
from app.workers.jobs.pr_scoring import score_pull_request_job

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

HANDLED_ACTIONS = {"opened", "synchronize"}
PR_SCORING_JOB_TIMEOUT_SECONDS = 60


def _normalize_url(url: str) -> str:
    return url.rstrip("/").removesuffix(".git").lower()


def _find_repository(db: Session, html_url: str) -> Repository | None:
    target = _normalize_url(html_url)
    for repo in db.scalars(select(Repository)):
        if _normalize_url(repo.url) == target:
            return repo
    return None


@router.post("/github", status_code=status.HTTP_202_ACCEPTED)
async def github_webhook(request: Request, db: Session = Depends(get_db)) -> dict:
    """§5's pre-merge flow, step 1: verify the signature, respond quickly,
    and post a pending check. Everything else (fetching the diff, scoring,
    explaining, editing the comment) happens in the worker (US-46)."""
    settings = get_settings()
    if not settings.github_webhook_secret:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "GitHub webhook secret is not configured"
        )

    raw_body = await request.body()
    if not verify_signature(
        raw_body, request.headers.get("X-Hub-Signature-256"), settings.github_webhook_secret
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid webhook signature")

    if request.headers.get("X-GitHub-Event") != "pull_request":
        return {"status": "ignored", "reason": "not a pull_request event"}

    try:
        payload = PullRequestWebhookEvent.model_validate_json(raw_body)
    except ValidationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Malformed payload: {exc}") from exc

    if payload.action not in HANDLED_ACTIONS:
        return {"status": "ignored", "reason": f"action '{payload.action}' not handled"}

    repository = _find_repository(db, payload.repository.html_url)
    if repository is None:
        return {"status": "ignored", "reason": "repository not registered with BugFlow"}

    if payload.installation is not None:
        installation_id = str(payload.installation.id)
        if repository.github_installation_id != installation_id:
            repository.github_installation_id = installation_id

    pr = db.scalar(
        select(PullRequest).where(
            PullRequest.repository_id == repository.id,
            PullRequest.number == payload.pull_request.number,
        )
    )
    if pr is None:
        pr = PullRequest(
            repository_id=repository.id,
            number=payload.pull_request.number,
            title=payload.pull_request.title,
            head_sha=payload.pull_request.head.sha,
            base_sha=payload.pull_request.base.sha,
        )
        db.add(pr)
    else:
        pr.title = payload.pull_request.title
        pr.head_sha = payload.pull_request.head.sha
        pr.base_sha = payload.pull_request.base.sha
    db.commit()
    db.refresh(pr)

    # A check run is tied permanently to one head_sha, so opened/synchronize
    # both create a fresh one for the new sha (§5) — the comment, tracked
    # separately below by the worker, is what gets edited in place (US-46).
    client = get_github_client(repository)
    pr.check_run_id = client.create_check_run(repository, pr.head_sha)
    pr.check_status = "pending"
    pr.check_conclusion = None
    pr.check_summary = None
    db.commit()

    get_queue().enqueue(
        score_pull_request_job, pr.id, job_timeout=PR_SCORING_JOB_TIMEOUT_SECONDS
    )

    return {"status": "accepted", "pull_request_id": pr.id}
