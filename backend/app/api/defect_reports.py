from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.queue import get_queue
from app.core.security import get_current_user, require_role
from app.models.defect import DefectReport
from app.models.enums import Role
from app.models.user import User
from app.schemas.defect import (
    DefectReportCreate,
    DefectReportRead,
    DuplicatesRead,
    MergeRequest,
    MisclassificationReportRead,
    SuggestionsRead,
    TriageDecisionRequest,
    TriageSuggestionRead,
)
from app.schemas.forecast import ForecastRead
from app.schemas.resolver import (
    AssignmentOverrideRequest,
    ResolverRecommendationsRead,
)
from app.services import defect_service, forecast_service, resolver_service, triage_service
from app.workers.jobs.defect import rebuild_defect_index_job

router = APIRouter(prefix="/defect-reports", tags=["defect-reports"])


def _get_report_or_404(db: Session, report_id: int) -> DefectReport:
    report = db.get(DefectReport, report_id)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Defect report not found")
    return report


@router.post("/index/rebuild", status_code=status.HTTP_202_ACCEPTED)
def rebuild_index(
    db: Session = Depends(get_db),
    _: User = Depends(require_role(Role.ADMIN, Role.ML_ENGINEER)),
) -> dict:
    """C2: rebuilds the HNSW index (e.g. after a bulk load) — search
    endpoints report `status: "rebuilding"` while this runs."""
    get_queue().enqueue(rebuild_defect_index_job, job_timeout=3600)
    return {"status": "rebuilding"}


@router.get("/misclassification-report", response_model=MisclassificationReportRead)
def get_misclassification_report(
    db: Session = Depends(get_db),
    _: User = Depends(require_role(Role.TRIAGER, Role.ML_ENGINEER, Role.ADMIN)),
) -> MisclassificationReportRead:
    """US-23: which suggested severities/priorities get corrected most often."""
    return triage_service.get_misclassification_report(db)


@router.get("/suggestions", response_model=SuggestionsRead)
def get_suggestions(
    text: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> SuggestionsRead:
    """US-16: called by the new-report form as the reporter types."""
    return defect_service.get_suggestions(db, text)


@router.get("", response_model=list[DefectReportRead])
def list_defect_reports(
    report_status: str | None = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[DefectReport]:
    # ponytail: found live, not by a unit test — with no limit at all, this
    # returned every row (16.6MB, 11s at just 36k rows) once the defect-index
    # load test had real data in the table. Same class of bug as Phase 5's
    # /review-queue N+1: fine with a handful of demo rows, breaks at scale.
    query = select(DefectReport).order_by(DefectReport.reported_at.desc())
    if report_status is not None:
        query = query.where(DefectReport.status == report_status)
    return list(db.scalars(query.limit(limit).offset(offset)))


@router.get("/{report_id}", response_model=DefectReportRead)
def get_defect_report(
    report_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> DefectReport:
    return _get_report_or_404(db, report_id)


@router.post("", response_model=DefectReportRead, status_code=status.HTTP_201_CREATED)
def create_defect_report(
    body: DefectReportCreate,
    db: Session = Depends(get_db),
    reporter: User = Depends(require_role(Role.REPORTER)),
) -> DefectReport:
    return defect_service.create_defect_report(db, reporter, body)


@router.get("/{report_id}/duplicates", response_model=DuplicatesRead)
def get_duplicates(
    report_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> DuplicatesRead:
    """US-17/US-18: candidate duplicates for an already-saved report."""
    report = _get_report_or_404(db, report_id)
    return defect_service.get_duplicates(db, report)


@router.get("/{report_id}/triage-suggestion", response_model=TriageSuggestionRead)
def get_triage_suggestion(
    report_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> TriageSuggestionRead:
    """US-21: the current automated severity/priority suggestion, or why
    there isn't one yet (abstained / unavailable / already decided)."""
    report = _get_report_or_404(db, report_id)
    return triage_service.get_or_create_suggestion(db, report)


@router.post("/{report_id}/triage", response_model=DefectReportRead)
def decide_triage(
    report_id: int,
    body: TriageDecisionRequest,
    db: Session = Depends(get_db),
    triager: User = Depends(require_role(Role.TRIAGER)),
) -> DefectReport:
    """US-23: accept the automated suggestion as-is, or change it — either
    way this is the one-click action that records the final decision."""
    report = _get_report_or_404(db, report_id)
    return triage_service.decide_triage(
        db, report, triager, body.severity.value, body.priority.value
    )


@router.get("/{report_id}/resolver-recommendations", response_model=ResolverRecommendationsRead)
def get_resolver_recommendations(
    report_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_role(Role.TRIAGER, Role.MANAGER, Role.ADMIN)),
) -> ResolverRecommendationsRead:
    """US-24: at least 3 ranked candidates with confidence/reason/open-defect
    count, or "no_confident_candidate" routed to the default triage owner."""
    report = _get_report_or_404(db, report_id)
    return resolver_service.get_recommendations(db, report)


@router.post("/{report_id}/assignment", response_model=DefectReportRead)
def override_assignment(
    report_id: int,
    body: AssignmentOverrideRequest,
    db: Session = Depends(get_db),
    triager: User = Depends(require_role(Role.TRIAGER, Role.MANAGER)),
) -> DefectReport:
    """US-29: override with a reason — the schema layer already refused a
    blank one. Stores the original recommendation, the override and the
    reason as `Feedback`, the next training signal."""
    report = _get_report_or_404(db, report_id)
    return resolver_service.override_assignment(db, report, body.developer_id, body.reason, triager)


@router.get("/{report_id}/forecast", response_model=ForecastRead)
def get_forecast(
    report_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> ForecastRead:
    """US-31/US-32/US-33: a probability curve with median/P90 always
    stated, a plain-language estimate for the reporter, and an at-risk flag
    for a still-open defect already past its own P90 window."""
    report = _get_report_or_404(db, report_id)
    return forecast_service.get_or_create_forecast(db, report)


@router.post("/{report_id}/merge", response_model=DefectReportRead)
def merge_defect_report(
    report_id: int,
    body: MergeRequest,
    db: Session = Depends(get_db),
    _: User = Depends(require_role(Role.TRIAGER)),
) -> DefectReport:
    """US-20: merges `report_id` into `body.original_id` and notifies its
    reporter asynchronously."""
    duplicate = _get_report_or_404(db, report_id)
    original = _get_report_or_404(db, body.original_id)
    if original.id == duplicate.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "A report cannot be merged into itself")

    defect_service.merge_report(db, duplicate, original)
    db.refresh(duplicate)
    return duplicate
