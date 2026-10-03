from bugflow_ml.assignment.batch_assignment import (
    AssignmentCandidate,
    DeveloperCapacity,
    solve_batch_assignment,
)
from bugflow_ml.features.resolver_features import ResolvedReport
from bugflow_ml.models.resolver_suitability import DeveloperCandidate, has_confident_candidate
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.decision import Explanation, Feedback
from app.models.defect import Assignment, DefectReport, ResolverRecommendation
from app.models.developer import Developer
from app.models.system import SystemConfig
from app.models.user import User
from app.schemas.resolver import (
    BatchAssignmentItem,
    BatchAssignResult,
    MyAssignmentRead,
    ResolverCandidateRead,
    ResolverRecommendationsRead,
    WorkloadEntry,
)
from app.services import resolver_prediction_service

DECISION_TYPE_RECOMMENDATION = "ResolverRecommendation"
DECISION_TYPE_ASSIGNMENT = "Assignment"
DEFAULT_OWNER_CONFIG_KEY = "default_triage_owner_id"
TOP_N_CANDIDATES = 5


def _load_developer_candidates(
    db: Session, developers: list[Developer]
) -> list[DeveloperCandidate]:
    return [
        DeveloperCandidate(
            id=d.id,
            name=d.name,
            skills=d.skills,
            capacity=d.capacity,
            current_queue_depth=d.current_queue_depth,
        )
        for d in developers
    ]


def _load_past_resolutions_by_developer(
    db: Session, developers: list[Developer]
) -> dict[int, list[ResolvedReport]]:
    resolved = db.scalars(
        select(DefectReport).where(
            DefectReport.assignee_id.isnot(None),
            DefectReport.resolved_at.isnot(None),
            DefectReport.embedding.isnot(None),
        )
    ).all()
    by_developer: dict[int, list[ResolvedReport]] = {d.id: [] for d in developers}
    for report in resolved:
        if report.assignee_id not in by_developer or report.embedding is None:
            continue
        by_developer[report.assignee_id].append(
            ResolvedReport(
                embedding=list(report.embedding),
                component=report.component,
                resolved_at=report.resolved_at,
            )
        )
    return by_developer


def _get_default_owner_id(db: Session) -> int | None:
    row = db.get(SystemConfig, DEFAULT_OWNER_CONFIG_KEY)
    if row is None:
        return None
    try:
        return int(row.value)
    except ValueError:
        return None


def _persist_recommendations(
    db: Session, defect_id: int, scored: list, dev_by_id: dict[int, Developer]
) -> None:
    for rank, candidate in enumerate(scored, start=1):
        recommendation = ResolverRecommendation(
            defect_id=defect_id,
            developer_id=candidate.developer_id,
            score=candidate.score,
            rank=rank,
            workload_at_time=dev_by_id[candidate.developer_id].current_queue_depth,
            reason=candidate.reason,
        )
        db.add(recommendation)
        db.flush()
        # C3: no automated decision without a plain-language explanation.
        db.add(
            Explanation(
                decision_type=DECISION_TYPE_RECOMMENDATION,
                decision_id=recommendation.id,
                text=candidate.reason,
                factors={"score": candidate.score, "is_cold_start": candidate.is_cold_start},
            )
        )


def get_recommendations(
    db: Session, report: DefectReport, tracking_uri: str | None = None
) -> ResolverRecommendationsRead:
    """US-24: at least 3 ranked candidates with confidence, reason, and
    current open-defect count — or, below the confidence floor (US-24 AC2),
    say so and name the default triage owner instead of guessing."""
    developers = list(db.scalars(select(Developer)))
    dev_by_id = {d.id: d for d in developers}
    if not developers or report.embedding is None:
        return ResolverRecommendationsRead(status="unavailable", candidates=[])

    candidates = _load_developer_candidates(db, developers)
    past_resolutions = _load_past_resolutions_by_developer(db, developers)

    scored = resolver_prediction_service.get_candidates(
        db,
        list(report.embedding),
        report.component,
        candidates,
        past_resolutions,
        report.reported_at,
        tracking_uri,
    )
    if scored is None:
        return ResolverRecommendationsRead(status="unavailable", candidates=[])

    top = scored[:TOP_N_CANDIDATES]
    _persist_recommendations(db, report.id, top, dev_by_id)
    db.commit()

    candidate_reads = [
        ResolverCandidateRead(
            developer_id=c.developer_id,
            developer_name=dev_by_id[c.developer_id].name,
            score=c.score,
            confidence=c.confidence,
            is_cold_start=c.is_cold_start,
            reason=c.reason,
            current_open_defects=dev_by_id[c.developer_id].current_queue_depth,
            capacity=dev_by_id[c.developer_id].capacity,
        )
        for c in top
    ]

    if not has_confident_candidate(scored):
        return ResolverRecommendationsRead(
            status="no_confident_candidate",
            candidates=candidate_reads,
            default_owner_id=_get_default_owner_id(db),
        )
    return ResolverRecommendationsRead(status="available", candidates=candidate_reads)


def batch_assign(
    db: Session, defect_ids: list[int], actor: User, tracking_uri: str | None = None
) -> BatchAssignResult:
    """US-25/US-26: the novel contribution — assigns a whole batch at once to
    maximise total suitability, subject to capacity. Defects the optimiser
    can't fit come back as shortfall, not a silent drop."""
    reports = [db.get(DefectReport, defect_id) for defect_id in defect_ids]
    eligible_reports = [
        r for r in reports if r is not None and r.assignee_id is None and r.embedding is not None
    ]

    developers = list(db.scalars(select(Developer)))
    dev_by_id = {d.id: d for d in developers}
    candidates = _load_developer_candidates(db, developers)
    past_resolutions = _load_past_resolutions_by_developer(db, developers)

    assignment_candidates: list[AssignmentCandidate] = []
    score_by_pair: dict[tuple[int, int], tuple[float, str, bool]] = {}
    for report in eligible_reports:
        assert report.embedding is not None  # eligible_reports already filtered for this
        scored = resolver_prediction_service.get_candidates(
            db,
            list(report.embedding),
            report.component,
            candidates,
            past_resolutions,
            report.reported_at,
            tracking_uri,
        )
        if not scored:
            continue
        for c in scored:
            assignment_candidates.append(
                AssignmentCandidate(defect_id=report.id, developer_id=c.developer_id, score=c.score)
            )
            score_by_pair[(report.id, c.developer_id)] = (c.score, c.reason, c.is_cold_start)

    capacities = [
        DeveloperCapacity(
            developer_id=d.id, available_slots=max(d.capacity - d.current_queue_depth, 0)
        )
        for d in developers
    ]

    result = solve_batch_assignment(assignment_candidates, capacities)

    report_by_id = {r.id: r for r in eligible_reports}
    items = []
    for defect_id, developer_id in result.assignments:
        score, reason, is_cold_start = score_by_pair[(defect_id, developer_id)]
        db.add(
            Assignment(
                defect_id=defect_id, developer_id=developer_id, source="batch", assigned_by=actor.id
            )
        )
        recommendation = ResolverRecommendation(
            defect_id=defect_id,
            developer_id=developer_id,
            score=score,
            rank=1,
            workload_at_time=dev_by_id[developer_id].current_queue_depth,
            reason=reason,
        )
        db.add(recommendation)
        db.flush()
        db.add(
            Explanation(
                decision_type=DECISION_TYPE_RECOMMENDATION,
                decision_id=recommendation.id,
                text=reason,
                factors={"score": score, "is_cold_start": is_cold_start},
            )
        )
        report_by_id[defect_id].assignee_id = developer_id
        dev_by_id[developer_id].current_queue_depth += 1
        items.append(
            BatchAssignmentItem(defect_id=defect_id, developer_id=developer_id, suitability=score)
        )

    missing_ids = [
        defect_id for report, defect_id in zip(reports, defect_ids, strict=True) if report is None
    ]
    unassigned_originally = [
        r.id for r in reports if r is not None and r not in eligible_reports
    ] + missing_ids
    shortfall = sorted(
        set(result.shortfall)
        | {d for d in unassigned_originally if d not in {i.defect_id for i in items}}
    )

    db.commit()
    return BatchAssignResult(
        assignments=items, shortfall=shortfall, total_suitability=result.total_suitability
    )


def override_assignment(
    db: Session, report: DefectReport, developer_id: int, reason: str, actor: User
) -> DefectReport:
    """US-29: override with a reason, required — the schema layer already
    refused a blank one. Stores the original recommendation, the override,
    and the reason as a `Feedback` row, the training signal for next time."""
    latest_recommendation = db.scalar(
        select(ResolverRecommendation)
        .where(ResolverRecommendation.defect_id == report.id, ResolverRecommendation.rank == 1)
        .order_by(ResolverRecommendation.id.desc())
    )
    accepted = (
        latest_recommendation is not None and latest_recommendation.developer_id == developer_id
    )
    db.add(
        Feedback(
            decision_type=DECISION_TYPE_RECOMMENDATION,
            decision_id=latest_recommendation.id if latest_recommendation else report.id,
            user_id=actor.id,
            accepted=accepted,
            original_value=(
                str(latest_recommendation.developer_id) if latest_recommendation else None
            ),
            new_value=str(developer_id),
            reason=reason,
        )
    )

    if report.assignee_id is not None and report.assignee_id != developer_id:
        old_developer = db.get(Developer, report.assignee_id)
        if old_developer is not None:
            old_developer.current_queue_depth = max(old_developer.current_queue_depth - 1, 0)

    db.add(
        Assignment(
            defect_id=report.id, developer_id=developer_id, source="override", assigned_by=actor.id
        )
    )
    new_developer = db.get(Developer, developer_id)
    if new_developer is not None and report.assignee_id != developer_id:
        new_developer.current_queue_depth += 1
    report.assignee_id = developer_id

    db.commit()
    db.refresh(report)
    return report


def raise_objection(db: Session, assignment: Assignment, actor: User, reason: str) -> None:
    """US-27: a developer's objection to being assigned a defect."""
    db.add(
        Feedback(
            decision_type=DECISION_TYPE_ASSIGNMENT,
            decision_id=assignment.id,
            user_id=actor.id,
            accepted=False,
            reason=reason,
        )
    )
    db.commit()


def get_workload_distribution(db: Session) -> list[WorkloadEntry]:
    """US-28: for the manager's workload chart."""
    developers = db.scalars(select(Developer).order_by(Developer.name))
    return [
        WorkloadEntry(
            developer_id=d.id,
            developer_name=d.name,
            capacity=d.capacity,
            current_queue_depth=d.current_queue_depth,
        )
        for d in developers
    ]


def get_my_assignments(db: Session, developer: Developer) -> list[MyAssignmentRead]:
    """US-27: "why me" — every assignment to this developer, with the reason
    behind it (the matching recommendation's explanation, if one exists)."""
    assignments = db.scalars(
        select(Assignment)
        .where(Assignment.developer_id == developer.id)
        .order_by(Assignment.assigned_at.desc())
    )
    results = []
    for assignment in assignments:
        report = db.get(DefectReport, assignment.defect_id)
        if report is None:
            continue
        recommendation = db.scalar(
            select(ResolverRecommendation)
            .where(
                ResolverRecommendation.defect_id == assignment.defect_id,
                ResolverRecommendation.developer_id == developer.id,
            )
            .order_by(ResolverRecommendation.id.desc())
        )
        reason = (
            recommendation.reason
            if recommendation is not None and recommendation.reason is not None
            else "Manually assigned, no reason on file."
        )
        results.append(
            MyAssignmentRead(
                id=assignment.id,
                defect_id=assignment.defect_id,
                defect_title=report.title,
                source=assignment.source,
                assigned_at=assignment.assigned_at,
                reason=reason,
            )
        )
    return results
