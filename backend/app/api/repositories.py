from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import require_role
from app.models.enums import Role
from app.models.repository import Repository
from app.models.user import User
from app.schemas.repository import RepositoryCreate, RepositoryRead, RepositoryUpdate
from app.services.audit import write_audit_log

router = APIRouter(prefix="/repositories", tags=["repositories"])

# US-01: Admin registers/configures repositories; ML Engineer can view them too.
VIEW_ROLES = (Role.ADMIN, Role.ML_ENGINEER)
# Phase 6: filing a defect report means picking which repository it's
# against, so the list (name/id only, via RepositoryRead) is also readable by
# whoever can file or triage a report — not the mutating or single-repo
# endpoints below, which stay Admin/ML-Engineer-only.
LIST_ROLES = (*VIEW_ROLES, Role.REPORTER, Role.TRIAGER)


@router.get("", response_model=list[RepositoryRead])
def list_repositories(
    db: Session = Depends(get_db), _: User = Depends(require_role(*LIST_ROLES))
) -> list[Repository]:
    return list(db.scalars(select(Repository)))


@router.get("/{repository_id}", response_model=RepositoryRead)
def get_repository(
    repository_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_role(*VIEW_ROLES)),
) -> Repository:
    repo = db.get(Repository, repository_id)
    if repo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repository not found")
    return repo


@router.post("", response_model=RepositoryRead, status_code=status.HTTP_201_CREATED)
def create_repository(
    body: RepositoryCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role(Role.ADMIN)),
) -> Repository:
    repo = Repository(**body.model_dump())
    db.add(repo)
    db.commit()
    db.refresh(repo)
    write_audit_log(db, admin.id, "create", "Repository", repo.id)
    return repo


@router.patch("/{repository_id}", response_model=RepositoryRead)
def update_repository(
    repository_id: int,
    body: RepositoryUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role(Role.ADMIN)),
) -> Repository:
    repo = db.get(Repository, repository_id)
    if repo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repository not found")

    changes = body.model_dump(exclude_unset=True)
    for field, value in changes.items():
        setattr(repo, field, value)
    db.commit()
    db.refresh(repo)
    write_audit_log(db, admin.id, "update", "Repository", repo.id, changes)
    return repo


@router.delete("/{repository_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_repository(
    repository_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role(Role.ADMIN)),
) -> None:
    repo = db.get(Repository, repository_id)
    if repo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repository not found")
    db.delete(repo)
    db.commit()
    write_audit_log(db, admin.id, "delete", "Repository", repository_id)
