from datetime import datetime

from bugflow_ml.models.triage_classifier import should_abstain
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.decision import Explanation, Feedback
from app.models.defect import DefectReport, TriageAssessment
from app.models.user import User
from app.schemas.defect import (
    MisclassificationBucket,
    MisclassificationReportRead,
    TriageSuggestionRead,
)
from app.services.triage_prediction_service import predict_triage

DECISION_TYPE = "TriageAssessment"


def _latest_assessment(db: Session, defect_id: int) -> TriageAssessment | None:
    return db.scalar(
        select(TriageAssessment)
        .where(TriageAssessment.defect_id == defect_id, TriageAssessment.is_automated)
        .order_by(TriageAssessment.assessed_at.desc())
    )


def get_or_create_suggestion(
    db: Session, report: DefectReport, tracking_uri: str | None = None
) -> TriageSuggestionRead:
    """US-21: the current triage suggestion for a report — "decided" once a
    human has set severity/priority, otherwise the latest automated
    suggestion (computed and persisted once, then reused on later views
    rather than recomputed every page load)."""
    if report.severity is not None and report.priority is not None:
        return TriageSuggestionRead(
            status="decided", severity=report.severity, priority=report.priority, is_automated=False
        )

    existing = _latest_assessment(db, report.id)
    if existing is not None:
        explanation = db.scalar(
            select(Explanation).where(
                Explanation.decision_type == DECISION_TYPE, Explanation.decision_id == existing.id
            )
        )
        factors = (explanation.factors if explanation else None) or {}
        return TriageSuggestionRead(
            status="available",
            severity=existing.severity,
            priority=existing.priority,
            severity_confidence=factors.get("severity_confidence"),
            priority_confidence=factors.get("priority_confidence"),
            severity_top_words=factors.get("severity_top_words"),
            priority_top_words=factors.get("priority_top_words"),
        )

    text = f"{report.title}\n{report.description}"
    if should_abstain(text):
        return TriageSuggestionRead(status="abstained")

    prediction = predict_triage(db, text, tracking_uri)
    if prediction is None:
        return TriageSuggestionRead(status="unavailable")
    severity_pred, priority_pred = prediction

    assessment = TriageAssessment(
        defect_id=report.id,
        severity=severity_pred.label,
        priority=priority_pred.label,
        confidence=(severity_pred.confidence + priority_pred.confidence) / 2,
        model_version="triage_severity+triage_priority",
        is_automated=True,
    )
    db.add(assessment)
    db.flush()
    # C3: no automated decision without a plain-language explanation.
    db.add(
        Explanation(
            decision_type=DECISION_TYPE,
            decision_id=assessment.id,
            text=(
                f"Suggested {severity_pred.label} severity "
                f"({severity_pred.confidence:.0%} confidence) and {priority_pred.label} priority "
                f"({priority_pred.confidence:.0%} confidence)."
            ),
            factors={
                "severity_confidence": severity_pred.confidence,
                "priority_confidence": priority_pred.confidence,
                "severity_top_words": severity_pred.top_words,
                "priority_top_words": priority_pred.top_words,
            },
        )
    )
    db.commit()

    return TriageSuggestionRead(
        status="available",
        severity=severity_pred.label,
        priority=priority_pred.label,
        severity_confidence=severity_pred.confidence,
        priority_confidence=priority_pred.confidence,
        severity_top_words=severity_pred.top_words,
        priority_top_words=priority_pred.top_words,
    )


def decide_triage(
    db: Session, report: DefectReport, user: User, severity: str, priority: str
) -> DefectReport:
    """US-23: stores the human-decided severity/priority. If an automated
    suggestion existed, records whether it was accepted as-is or changed —
    that `Feedback` row is what the misclassification report reads."""
    suggestion = _latest_assessment(db, report.id)
    if suggestion is not None:
        accepted = suggestion.severity == severity and suggestion.priority == priority
        db.add(
            Feedback(
                decision_type=DECISION_TYPE,
                decision_id=suggestion.id,
                user_id=user.id,
                accepted=accepted,
                original_value=f"{suggestion.severity}/{suggestion.priority}",
                new_value=f"{severity}/{priority}",
            )
        )

    report.severity = severity
    report.priority = priority
    db.commit()
    db.refresh(report)
    return report


def _buckets(totals: dict[str, int], corrections: dict[str, int]) -> list[MisclassificationBucket]:
    return sorted(
        (
            MisclassificationBucket(
                suggested=key,
                corrected_count=corrections.get(key, 0),
                total_count=total,
                correction_rate=corrections.get(key, 0) / total if total else 0.0,
            )
            for key, total in totals.items()
        ),
        key=lambda bucket: -bucket.correction_rate,
    )


def get_misclassification_report(
    db: Session, since: datetime | None = None, component: str | None = None
) -> MisclassificationReportRead:
    """US-23: which suggested severities/priorities get corrected most
    often — only counts feedback tied to a real automated suggestion, not
    reports a human classified with no suggestion to react to. Optionally
    narrowed to feedback given since a date, and/or one component."""
    query = (
        select(TriageAssessment.severity, TriageAssessment.priority, Feedback.accepted)
        .join(TriageAssessment, Feedback.decision_id == TriageAssessment.id)
        .where(Feedback.decision_type == DECISION_TYPE, TriageAssessment.is_automated)
    )
    if since is not None:
        query = query.where(Feedback.created_at >= since)
    if component is not None:
        query = query.join(DefectReport, TriageAssessment.defect_id == DefectReport.id).where(
            DefectReport.component == component
        )

    severity_totals: dict[str, int] = {}
    severity_corrections: dict[str, int] = {}
    priority_totals: dict[str, int] = {}
    priority_corrections: dict[str, int] = {}

    for severity, priority, accepted in db.execute(query):
        severity_totals[severity] = severity_totals.get(severity, 0) + 1
        priority_totals[priority] = priority_totals.get(priority, 0) + 1
        if not accepted:
            severity_corrections[severity] = severity_corrections.get(severity, 0) + 1
            priority_corrections[priority] = priority_corrections.get(priority, 0) + 1

    return MisclassificationReportRead(
        severity=_buckets(severity_totals, severity_corrections),
        priority=_buckets(priority_totals, priority_corrections),
    )
