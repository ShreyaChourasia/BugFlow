from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import get_current_user, require_role
from app.models.defect import Assignment
from app.models.developer import Developer
from app.models.enums import Role
from app.models.user import User
from app.schemas.resolver import (
    BatchAssignRequest,
    BatchAssignResult,
    MyAssignmentRead,
    ObjectionRequest,
    WorkloadEntry,
)
from app.services import resolver_service

router = APIRouter(prefix="/assignments", tags=["resolver"])


def _get_own_developer_or_403(db: Session, user: User) -> Developer:
    if user.developer_id is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "User is not linked to a developer record")
    developer = db.get(Developer, user.developer_id)
    if developer is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "User is not linked to a developer record")
    return developer


@router.post("/batch", response_model=BatchAssignResult)
def batch_assign(
    body: BatchAssignRequest,
    db: Session = Depends(get_db),
    manager: User = Depends(require_role(Role.MANAGER, Role.TRIAGER)),
) -> BatchAssignResult:
    """US-25/US-26: capacity-constrained batch assignment. Defects the
    optimiser can't fit come back as `shortfall`, never a silent drop."""
    return resolver_service.batch_assign(db, body.defect_ids, manager)


@router.post("/{assignment_id}/objection", status_code=status.HTTP_204_NO_CONTENT)
def raise_objection(
    assignment_id: int,
    body: ObjectionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_role(Role.DEVELOPER)),
) -> None:
    """US-27: a developer's objection to being assigned a defect."""
    developer = _get_own_developer_or_403(db, user)
    assignment = db.get(Assignment, assignment_id)
    if assignment is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Assignment not found")
    if assignment.developer_id != developer.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your assignment")
    resolver_service.raise_objection(db, assignment, user, body.reason)


@router.get("/workload", response_model=list[WorkloadEntry])
def get_workload_distribution(
    db: Session = Depends(get_db),
    _: User = Depends(require_role(Role.MANAGER, Role.QA, Role.ADMIN)),
) -> list[WorkloadEntry]:
    """US-28: for the manager's workload chart."""
    return resolver_service.get_workload_distribution(db)


@router.get("/mine", response_model=list[MyAssignmentRead])
def get_my_assignments(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[MyAssignmentRead]:
    """US-27: "why me" — every assignment to the logged-in developer, with
    the reason behind it."""
    developer = _get_own_developer_or_403(db, user)
    return resolver_service.get_my_assignments(db, developer)
