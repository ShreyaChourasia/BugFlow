from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import get_current_user
from app.models.commit import Commit
from app.models.decision import Explanation
from app.models.pull_request import PullRequest, RiskPrediction
from app.models.repository import Repository
from app.models.user import User
from app.schemas.pull_request import PullRequestDetailRead, PullRequestRead, PullRequestRiskRead

router = APIRouter(prefix="/repositories", tags=["pull-requests"])


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
    _get_repository_or_404(db, repository_id)

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
            risk = PullRequestRiskRead(
                probability=prediction.probability,
                calibrated_probability=prediction.calibrated_probability,
                confidence=prediction.confidence,
                risk_level=prediction.risk_level,
                explanation_text=explanation.text if explanation else "",
                factors=(explanation.factors or {}).get("items", []) if explanation else [],
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
