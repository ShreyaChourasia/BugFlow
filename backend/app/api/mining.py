from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.queue import get_queue
from app.core.security import require_role
from app.models.enums import Role
from app.models.repository import MiningRun, Repository
from app.models.user import User
from app.schemas.mining import MiningRunRead
from app.workers.jobs.mining import run_mining_job

router = APIRouter(prefix="/repositories", tags=["mining"])

# Matches the "Repositories: register, mining status, settings" roles (§9).
MINING_ROLES = (Role.ADMIN, Role.ML_ENGINEER)


def _get_repository_or_404(db: Session, repository_id: int) -> Repository:
    repo = db.get(Repository, repository_id)
    if repo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repository not found")
    return repo


@router.get("/{repository_id}/mining-runs", response_model=list[MiningRunRead])
def list_mining_runs(
    repository_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_role(*MINING_ROLES)),
) -> list[MiningRun]:
    _get_repository_or_404(db, repository_id)
    return list(
        db.scalars(
            select(MiningRun)
            .where(MiningRun.repository_id == repository_id)
            .order_by(MiningRun.id.desc())
        )
    )


@router.post(
    "/{repository_id}/mining-runs",
    response_model=MiningRunRead,
    status_code=status.HTTP_201_CREATED,
)
def start_mining_run(
    repository_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_role(*MINING_ROLES)),
) -> MiningRun:
    _get_repository_or_404(db, repository_id)

    # Carry the checkpoint forward from the latest run for this repo, so a
    # fresh "start mining" click also serves as incremental ingestion (US-07)
    # once the first run has completed.
    latest = db.scalar(
        select(MiningRun)
        .where(MiningRun.repository_id == repository_id)
        .order_by(MiningRun.id.desc())
    )
    checkpoint = latest.checkpoint if latest else None

    run = MiningRun(repository_id=repository_id, status="pending", checkpoint=checkpoint)
    db.add(run)
    db.commit()
    db.refresh(run)

    get_queue().enqueue(run_mining_job, run.id)
    return run


@router.post("/{repository_id}/mining-runs/{run_id}/resume", response_model=MiningRunRead)
def resume_mining_run(
    repository_id: int,
    run_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_role(*MINING_ROLES)),
) -> MiningRun:
    _get_repository_or_404(db, repository_id)

    run = db.get(MiningRun, run_id)
    if run is None or run.repository_id != repository_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Mining run not found")
    if run.status != "failed":
        raise HTTPException(status.HTTP_409_CONFLICT, "Only a failed run can be resumed")

    run.status = "pending"
    run.error = None
    db.commit()
    db.refresh(run)

    get_queue().enqueue(run_mining_job, run.id)
    return run
