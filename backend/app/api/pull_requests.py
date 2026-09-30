from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, tuple_
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import get_current_user
from app.models.commit import Commit
from app.models.decision import Explanation, Feedback
from app.models.pull_request import LineRisk, PullRequest, RiskPrediction
from app.models.repository import Repository
from app.models.user import User
from app.schemas.pull_request import (
    LineRiskRead,
    PullRequestDetailRead,
    PullRequestRead,
    PullRequestRiskRead,
    ReviewQueueItemRead,
)

router = APIRouter(prefix="/repositories", tags=["pull-requests"])
line_risk_router = APIRouter(tags=["pull-requests"])


def _get_repository_or_404(db: Session, repository_id: int) -> Repository:
    repo = db.get(Repository, repository_id)
    if repo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repository not found")
    return repo


@router.get("/{repository_id}/pull-requests", response_model=list[PullRequestRead])
def list_pull_requests(
    repository_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[PullRequest]:
    _get_repository_or_404(db, repository_id)
    return list(
        db.scalars(
            select(PullRequest)
            .where(PullRequest.repository_id == repository_id)
            .order_by(PullRequest.number.desc())
        )
    )


@router.get("/{repository_id}/pull-requests/{number}", response_model=PullRequestDetailRead)
def get_pull_request(
    repository_id: int,
    number: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> PullRequestDetailRead:
    repository = _get_repository_or_404(db, repository_id)

    pr = db.scalar(
        select(PullRequest).where(
            PullRequest.repository_id == repository_id, PullRequest.number == number
        )
    )
    if pr is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Pull request not found")

    risk: PullRequestRiskRead | None = None
    commit = db.scalar(
        select(Commit).where(Commit.repository_id == repository_id, Commit.sha == pr.head_sha)
    )
    if commit is not None:
        prediction = db.scalar(
            select(RiskPrediction)
            .where(RiskPrediction.commit_id == commit.id)
            .order_by(RiskPrediction.created_at.desc())
        )
        if prediction is not None:
            explanation = db.scalar(
                select(Explanation).where(
                    Explanation.decision_type == "RiskPrediction",
                    Explanation.decision_id == prediction.id,
                )
            )
            line_risks = list(
                db.scalars(
                    select(LineRisk)
                    .where(LineRisk.prediction_id == prediction.id)
                    .order_by(LineRisk.rank)
                )
            )
            if line_risks:
                line_risk_status = "available"
            elif prediction.calibrated_probability < repository.risk_threshold:
                line_risk_status = "not_applicable"
            else:
                line_risk_status = "unavailable"
            risk = PullRequestRiskRead(
                probability=prediction.probability,
                calibrated_probability=prediction.calibrated_probability,
                confidence=prediction.confidence,
                risk_level=prediction.risk_level,
                explanation_text=explanation.text if explanation else "",
                factors=(explanation.factors or {}).get("items", []) if explanation else [],
                line_risk_status=line_risk_status,
                lines=[LineRiskRead.model_validate(line) for line in line_risks],
            )

    return PullRequestDetailRead(
        id=pr.id,
        repository_id=pr.repository_id,
        number=pr.number,
        title=pr.title,
        status=pr.status,
        head_sha=pr.head_sha,
        base_sha=pr.base_sha,
        created_at=pr.created_at,
        check_status=pr.check_status,
        check_conclusion=pr.check_conclusion,
        check_summary=pr.check_summary,
        comment_body=pr.comment_body,
        risk=risk,
    )


@line_risk_router.post("/line-risks/{line_id}/false-alarm", status_code=status.HTTP_204_NO_CONTENT)
def mark_line_risk_false_alarm(
    line_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    """US-15: a reviewer flags a highlighted line as wrong. Both the line
    itself and a `Feedback` row are updated, so training can later learn from
    which flags reviewers actually trusted."""
    line = db.get(LineRisk, line_id)
    if line is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Line risk not found")

    line.marked_false_alarm = True
    db.add(
        Feedback(
            decision_type="LineRisk",
            decision_id=line.id,
            user_id=user.id,
            accepted=False,
            reason="false_alarm",
        )
    )
    db.commit()


@line_risk_router.get("/review-queue", response_model=list[ReviewQueueItemRead])
def get_review_queue(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[ReviewQueueItemRead]:
    """US-09: every open PR across all repos, ranked by calibrated risk so
    reviewers see the riskiest changes first.
    ponytail: 3 batched queries regardless of PR count, not one pair of
    lookups per PR — found by a live check against ~900 real PRs left over
    from earlier k6 perf testing, which took the naive per-row version ~6.6s."""
    prs = list(
        db.execute(
            select(PullRequest, Repository.name)
            .join(Repository, PullRequest.repository_id == Repository.id)
            .where(PullRequest.status == "open")
        ).all()
    )
    if not prs:
        return []

    head_keys = {(pr.repository_id, pr.head_sha) for pr, _ in prs}
    commits = db.scalars(
        select(Commit).where(tuple_(Commit.repository_id, Commit.sha).in_(head_keys))
    ).all()
    commit_by_key = {(c.repository_id, c.sha): c for c in commits}

    predictions = db.scalars(
        select(RiskPrediction)
        .where(RiskPrediction.commit_id.in_([c.id for c in commits]))
        .order_by(RiskPrediction.created_at.desc())
    ).all()
    latest_prediction_by_commit_id: dict[int, RiskPrediction] = {}
    for prediction in predictions:
        if prediction.commit_id is not None:
            latest_prediction_by_commit_id.setdefault(prediction.commit_id, prediction)

    items = []
    for pr, repository_name in prs:
        commit = commit_by_key.get((pr.repository_id, pr.head_sha))
        latest_prediction = latest_prediction_by_commit_id.get(commit.id) if commit else None
        items.append(
            ReviewQueueItemRead(
                repository_id=pr.repository_id,
                repository_name=repository_name,
                number=pr.number,
                title=pr.title,
                check_status=pr.check_status,
                calibrated_probability=(
                    latest_prediction.calibrated_probability if latest_prediction else None
                ),
                risk_level=latest_prediction.risk_level if latest_prediction else None,
                lines_added=commit.lines_added if commit else None,
                lines_deleted=commit.lines_deleted if commit else None,
                files_changed=commit.files_changed if commit else None,
            )
        )

    items.sort(
        key=lambda item: (item.calibrated_probability is None, -(item.calibrated_probability or 0))
    )
    return items
