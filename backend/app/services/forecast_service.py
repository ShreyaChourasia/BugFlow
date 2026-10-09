from datetime import UTC, datetime

from bugflow_ml.models.resolution_forecast import explain_forecast, is_at_risk
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.decision import Explanation
from app.models.defect import DefectReport, ResolutionForecast
from app.schemas.forecast import ForecastCurvePoint, ForecastRead
from app.services import forecast_prediction_service, triage_service

DECISION_TYPE = "ResolutionForecast"


def _resolve_severity_priority(db: Session, report: DefectReport) -> tuple[str, str] | None:
    """Forecasting needs a severity/priority to condition on — a human
    decision if there is one, otherwise the current automated triage
    suggestion (reusing US-21's own computation rather than guessing
    independently). None if neither is available yet."""
    if report.severity is not None and report.priority is not None:
        return report.severity, report.priority

    suggestion = triage_service.get_or_create_suggestion(db, report)
    if suggestion.status == "available":
        assert suggestion.severity is not None and suggestion.priority is not None
        return suggestion.severity, suggestion.priority
    return None


def _estimate_text(median_days: float, p90_days: float) -> str:
    """US-32: a simple estimate for the reporter, e.g. "likely within 3-10
    days" — the same median/P90 the chart shows, in plain language."""
    low = max(1, round(median_days))
    high = max(low, round(p90_days))
    if low == high:
        return f"Likely within about {low} day{'s' if low != 1 else ''}"
    return f"Likely within {low}-{high} days"


def get_or_create_forecast(
    db: Session, report: DefectReport, tracking_uri: str | None = None
) -> ForecastRead:
    """US-31/US-32/US-33: a probability curve with median/P90 always
    stated, a plain-language estimate, and an at-risk flag for a still-open
    defect already past its own P90 window."""
    resolved = _resolve_severity_priority(db, report)
    if resolved is None:
        return ForecastRead(status="unavailable")
    severity, priority = resolved

    prediction = forecast_prediction_service.get_forecast(
        db, severity, priority, report.component, tracking_uri
    )
    if prediction is None:
        return ForecastRead(status="unavailable")

    is_resolved = report.resolved_at is not None
    as_of = report.resolved_at or datetime.now(UTC)
    elapsed_days = (as_of - report.reported_at).total_seconds() / 86400
    at_risk = is_at_risk(elapsed_days, prediction.p90_days, is_resolved)

    existing = db.scalar(
        select(ResolutionForecast).where(ResolutionForecast.defect_id == report.id)
    )
    if existing is None:
        existing = ResolutionForecast(defect_id=report.id, model_version="resolution_forecast")
        db.add(existing)
    existing.median_days = prediction.median_days
    existing.p90_days = prediction.p90_days
    existing.curve = {
        "days": [p[0] for p in prediction.curve],
        "survival": [p[1] for p in prediction.curve],
    }
    existing.confidence = prediction.confidence
    existing.at_risk = at_risk
    db.flush()

    # C3: no automated decision without a plain-language explanation.
    result = forecast_prediction_service.get_champion_result(db, tracking_uri)
    drivers = explain_forecast(result, severity, priority, report.component) if result else []
    explanation_text = (
        f"Median {prediction.median_days:.0f} days, P90 {prediction.p90_days:.0f} days, "
        f"based on severity={severity}, priority={priority}."
    )
    explanation = db.scalar(
        select(Explanation).where(
            Explanation.decision_type == DECISION_TYPE, Explanation.decision_id == existing.id
        )
    )
    if explanation is None:
        explanation = Explanation(decision_type=DECISION_TYPE, decision_id=existing.id, text="")
        db.add(explanation)
    explanation.text = explanation_text
    explanation.factors = {"drivers": drivers, "confidence": prediction.confidence}

    db.commit()

    return ForecastRead(
        status="available",
        median_days=prediction.median_days,
        p90_days=prediction.p90_days,
        curve=[
            ForecastCurvePoint(day=day, probability_unresolved=prob)
            for day, prob in prediction.curve
        ],
        confidence=prediction.confidence,
        at_risk=at_risk,
        estimate_text=_estimate_text(prediction.median_days, prediction.p90_days),
    )
