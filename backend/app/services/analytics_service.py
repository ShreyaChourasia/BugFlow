from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.decision import Explanation, Feedback
from app.models.defect import DefectReport
from app.models.ml import MLModel
from app.models.pull_request import RiskPrediction
from app.schemas.analytics import (
    AcceptanceRateEntry,
    AcceptanceRateRead,
    FairnessRead,
    HotspotEntry,
    HotspotsRead,
    RiskTrendPoint,
    RiskTrendsRead,
    TopRiskFactorEntry,
    TopRiskFactorsRead,
)

COMMIT_RISK_TASK = "commit_risk"


def get_risk_trends(db: Session, days: int = 90) -> RiskTrendsRead:
    """US-38: mean calibrated risk per week, real decisions only — a
    shadow challenger's silent score never counts here (it never drove a
    real decision)."""
    since = datetime.now(UTC) - timedelta(days=days)
    week = func.date_trunc("week", RiskPrediction.created_at)
    rows = db.execute(
        select(
            week.label("week_start"),
            func.avg(RiskPrediction.calibrated_probability),
            func.count(),
        )
        .where(RiskPrediction.created_at >= since, ~RiskPrediction.is_shadow)
        .group_by(week)
        .order_by(week)
    ).all()
    return RiskTrendsRead(
        points=[
            RiskTrendPoint(
                week_start=week_start.date().isoformat(),
                mean_calibrated_probability=float(mean_prob),
                count=count,
            )
            for week_start, mean_prob, count in rows
        ]
    )


def get_acceptance_rate(db: Session) -> AcceptanceRateRead:
    """US-39: share of automated recommendations a human kept as-is,
    per decision type — covers every `Feedback` row regardless of which
    feature wrote it (triage, resolver override/objection, ...)."""
    rows = db.execute(
        select(Feedback.decision_type, Feedback.accepted, func.count()).group_by(
            Feedback.decision_type, Feedback.accepted
        )
    ).all()

    totals: dict[str, int] = defaultdict(int)
    accepted_counts: dict[str, int] = defaultdict(int)
    for decision_type, accepted, count in rows:
        totals[decision_type] += count
        if accepted:
            accepted_counts[decision_type] += count

    entries = [
        AcceptanceRateEntry(
            decision_type=decision_type,
            total=total,
            accepted=accepted_counts.get(decision_type, 0),
            acceptance_rate=accepted_counts.get(decision_type, 0) / total,
        )
        for decision_type, total in sorted(totals.items())
    ]
    overall_total = sum(totals.values())
    overall_accepted = sum(accepted_counts.values())
    overall_rate = overall_accepted / overall_total if overall_total else None
    return AcceptanceRateRead(entries=entries, overall_rate=overall_rate)


def get_hotspots(db: Session, limit: int = 15) -> HotspotsRead:
    """US-40: defect counts by module/component — reports with no
    component are bucketed as "uncategorized" rather than dropped.
    `component` is built once and reused in both SELECT and GROUP BY —
    two separate `func.coalesce(...)` calls each compile their own bind
    parameter, and Postgres then refuses the GROUP BY as two different
    expressions even though both render identically."""
    component = func.coalesce(DefectReport.component, "uncategorized")
    rows = db.execute(select(component, func.count()).group_by(component)).all()
    entries = sorted(
        (HotspotEntry(component=component, defect_count=count) for component, count in rows),
        key=lambda e: -e.defect_count,
    )[:limit]
    return HotspotsRead(entries=entries)


def get_top_risk_factors(db: Session, limit: int = 10) -> TopRiskFactorsRead:
    """US-37: which factors most often push risk up, project-wide — every
    `RiskPrediction`'s explanation already records its top-3 SHAP factors
    (US-34/36); this just aggregates them across every commit scored."""
    explanations = db.scalars(
        select(Explanation).where(Explanation.decision_type == "RiskPrediction")
    )
    occurrence_counts: Counter[str] = Counter()
    shap_sums: dict[str, float] = defaultdict(float)
    for explanation in explanations:
        for item in (explanation.factors or {}).get("items", []):
            if item.get("impact") != "increases":
                continue
            feature = item["feature"]
            occurrence_counts[feature] += 1
            shap_sums[feature] += item["shap_value"]

    entries = [
        TopRiskFactorEntry(
            feature=feature,
            occurrence_count=count,
            mean_shap_value=shap_sums[feature] / count,
        )
        for feature, count in occurrence_counts.most_common(limit)
    ]
    return TopRiskFactorsRead(entries=entries)


def get_fairness(db: Session) -> FairnessRead:
    """NFR-US-11: surfaces the fairness report already computed and stored
    at training time (C10) — no new computation, just reads the current
    commit-risk champion's `metrics["fairness"]`."""
    champion = db.scalar(
        select(MLModel).where(MLModel.task == COMMIT_RISK_TASK, MLModel.stage == "champion")
    )
    if champion is None or not champion.metrics or "fairness" not in champion.metrics:
        return FairnessRead(status="unavailable")

    fairness = champion.metrics["fairness"]
    return FairnessRead(
        status="available",
        junior_mean_risk=fairness.get("junior_mean_risk"),
        senior_mean_risk=fairness.get("senior_mean_risk"),
        junior_count=fairness.get("junior_count", 0),
        senior_count=fairness.get("senior_count", 0),
        disparity=fairness.get("disparity"),
        model_version=champion.version,
    )
