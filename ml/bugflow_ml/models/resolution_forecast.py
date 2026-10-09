"""US-31/US-32/US-33: resolution-time forecasting via survival analysis.

§8's stated design: Kaplan-Meier by severity as the baseline, then Cox
proportional hazards as the main model, scored by concordance index.
Unresolved defects are **censored observations**, not dropped (US-31 AC) —
"still open after 12 days" is real information a survival model can use,
unlike a plain regression on resolved reports alone, which would only ever
see completed cases and systematically underestimate how long things take.
"""

import hashlib
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

from bugflow_ml.taxonomy import Priority, Severity

MIN_TRAINING_EXAMPLES = 50

_SEVERITY_RANK: dict[str, int] = {
    Severity.TRIVIAL: 0,
    Severity.MINOR: 1,
    Severity.MAJOR: 2,
    Severity.CRITICAL: 3,
    Severity.BLOCKER: 4,
}
_PRIORITY_RANK: dict[str, int] = {
    Priority.P5: 0,
    Priority.P4: 1,
    Priority.P3: 2,
    Priority.P2: 3,
    Priority.P1: 4,
}
_COMPONENTS = ["login", "checkout", "search", "dashboard", "billing", "api"]
# Some components are genuinely slower to fix than others in the bootstrap
# world (e.g. "billing" touches more integrations) — gives Cox PH (which can
# see component) real signal the KM-by-severity-only baseline can't use,
# the same way a real codebase's slow modules would.
_COMPONENT_HAZARD = {
    "login": 0.0,
    "checkout": 0.3,
    "search": 0.1,
    "dashboard": 0.0,
    "billing": 0.4,
    "api": 0.2,
}


@dataclass
class ForecastExample:
    defect_id: int
    severity: str
    priority: str
    component: str | None
    duration_days: float
    event_observed: bool  # True = resolved (the event happened); False = still open (censored)
    reported_at: datetime


@dataclass
class ForecastTrainingResult:
    seed: int
    data_version: str
    model: CoxPHFitter
    feature_columns: list[str]
    component_categories: list[str]
    metrics: dict[str, Any]
    train_size: int
    test_size: int


@dataclass
class ForecastPrediction:
    median_days: float
    p90_days: float
    curve: list[tuple[float, float]]  # (day, probability still unresolved)
    confidence: float


def compute_data_version(examples: list[ForecastExample]) -> str:
    canonical = ",".join(
        sorted(f"{e.defect_id}:{e.duration_days}:{e.event_observed}" for e in examples)
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def chronological_split(
    examples: list[ForecastExample], train_frac: float = 0.8
) -> tuple[list[ForecastExample], list[ForecastExample]]:
    """C9: train on the past, test on the future."""
    ordered = sorted(examples, key=lambda e: (e.reported_at, e.defect_id))
    n_train = int(len(ordered) * train_frac)
    train, test = ordered[:n_train], ordered[n_train:]
    if train and test:
        assert max(e.reported_at for e in train) <= min(e.reported_at for e in test)
    return train, test


def _encode(examples: list[ForecastExample], component_categories: list[str]) -> pd.DataFrame:
    rows = []
    for e in examples:
        row: dict[str, float] = {
            "duration": e.duration_days,
            "event": float(e.event_observed),
            "severity_rank": float(_SEVERITY_RANK.get(e.severity, 2)),
            "priority_rank": float(_PRIORITY_RANK.get(e.priority, 2)),
        }
        for component in component_categories:
            row[f"component_{component}"] = 1.0 if e.component == component else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _km_baseline_concordance(train: list[ForecastExample], test: list[ForecastExample]) -> float:
    """§8's stated baseline: Kaplan-Meier fit separately per severity group.
    A KM-by-group model has no per-individual risk score of its own, so for
    an apples-to-apples concordance comparison against Cox PH, each test
    example is scored by its own severity group's median survival time.
    lifelines' `concordance_index()` expects a higher score to mean *longer*
    predicted survival (the same convention `-predict_partial_hazard` uses
    for Cox, since partial hazard runs the other way) — so the median
    itself is the score, not its negation."""
    km_by_severity = {}
    for severity in {e.severity for e in train}:
        group = [e for e in train if e.severity == severity]
        km = KaplanMeierFitter()
        km.fit([e.duration_days for e in group], event_observed=[e.event_observed for e in group])
        km_by_severity[severity] = km

    risk_scores = []
    for e in test:
        km = km_by_severity.get(e.severity)
        median = km.median_survival_time_ if km is not None else float("inf")
        risk_scores.append(median if median != float("inf") else 1e9)

    return float(
        concordance_index(
            [e.duration_days for e in test], risk_scores, [e.event_observed for e in test]
        )
    )


def train_forecast_model(
    examples: list[ForecastExample], seed: int = 42, train_frac: float = 0.8
) -> ForecastTrainingResult:
    if len(examples) < MIN_TRAINING_EXAMPLES:
        raise ValueError(
            f"Need at least {MIN_TRAINING_EXAMPLES} examples to train; got {len(examples)}."
        )
    train, test = chronological_split(examples, train_frac)
    if not train or not test:
        raise ValueError("Chronological split produced an empty train/test set.")

    component_categories = sorted({e.component for e in examples if e.component})
    train_df = _encode(train, component_categories)
    test_df = _encode(test, component_categories)
    feature_columns = [c for c in train_df.columns if c not in ("duration", "event")]

    # penalizer: a handful of one-hot component columns can be collinear
    # with few examples per category — a small L2 penalty keeps the fit
    # stable instead of failing to converge.
    model = CoxPHFitter(penalizer=0.1)
    model.fit(train_df, duration_col="duration", event_col="event")

    cox_concordance = float(
        concordance_index(
            test_df["duration"], -model.predict_partial_hazard(test_df), test_df["event"]
        )
    )
    km_concordance = _km_baseline_concordance(train, test)

    metrics = {
        "cox_concordance": cox_concordance,
        "km_baseline_concordance": km_concordance,
    }
    return ForecastTrainingResult(
        seed=seed,
        data_version=compute_data_version(examples),
        model=model,
        feature_columns=feature_columns,
        component_categories=component_categories,
        metrics=metrics,
        train_size=len(train),
        test_size=len(test),
    )


def _as_scalar(value: Any) -> float:
    """lifelines returns a bare scalar for some versions/single-row inputs
    and a length-1 Series for others — normalize either to a plain float."""
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def predict_forecast(
    result: ForecastTrainingResult, severity: str, priority: str, component: str | None
) -> ForecastPrediction:
    """US-31/US-32: a probability curve plus median and P90, always stated
    — lifelines returns +inf for a percentile the fitted curve never
    reaches within the observed horizon; clipped to the longest observed
    training duration so the output is always a concrete number, not inf."""
    row: dict[str, float] = {
        "severity_rank": float(_SEVERITY_RANK.get(severity, 2)),
        "priority_rank": float(_PRIORITY_RANK.get(priority, 2)),
    }
    for category in result.component_categories:
        row[f"component_{category}"] = 1.0 if component == category else 0.0
    X = pd.DataFrame([row])[result.feature_columns]

    max_observed_day = float(result.model.durations.max())
    median = _as_scalar(result.model.predict_median(X))
    p90 = _as_scalar(result.model.predict_percentile(X, p=0.1))
    if median == float("inf"):
        median = max_observed_day
    if p90 == float("inf"):
        p90 = max_observed_day
    p90 = max(p90, median)

    survival_function = result.model.predict_survival_function(X)
    curve = [
        (float(day), float(survival_function.iloc[i, 0]))
        for i, day in enumerate(survival_function.index)
    ]

    return ForecastPrediction(
        median_days=median,
        p90_days=p90,
        curve=curve,
        confidence=result.metrics["cox_concordance"],
    )


def explain_forecast(
    result: ForecastTrainingResult,
    severity: str,
    priority: str,
    component: str | None,
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Main drivers (§8) behind this specific forecast: coefficient × this
    report's own feature value — the same per-prediction attribution style
    every linear/log-linear model in this project uses (triage's
    coefficient × tfidf, line-risk's coefficient × feature value)."""
    row: dict[str, float] = {
        "severity_rank": float(_SEVERITY_RANK.get(severity, 2)),
        "priority_rank": float(_PRIORITY_RANK.get(priority, 2)),
    }
    for category in result.component_categories:
        row[f"component_{category}"] = 1.0 if component == category else 0.0

    coefficients = result.model.params_
    contributions = [
        (name, float(coefficients[name] * row.get(name, 0.0))) for name in result.feature_columns
    ]
    contributions.sort(key=lambda pair: -abs(pair[1]))
    return contributions[:top_n]


def is_at_risk(elapsed_days: float, p90_days: float, is_resolved: bool) -> bool:
    """US-33: a still-open defect whose elapsed time already exceeds the
    model's own P90 window for it — an honest "this is taking longer than
    90% of comparable cases," not a guess. A resolved defect is never
    at-risk; there's nothing left to wait for."""
    return (not is_resolved) and elapsed_days > p90_days


# --- bootstrap data -----------------------------------------------------
#
# §8 names no specific public resolution-time dataset the way it does for
# severity/duplicate detection — no verified, licensed one is available in
# this environment regardless (same constraint as ADRs 004-006). Real
# resolved/open `DefectReport` history is always preferred; this only fills
# the gap before enough of it exists.
#
# Durations are drawn from a genuine Weibull **proportional-hazards** model
# (not just "higher severity -> smaller random draw") — the same family of
# model CoxPHFitter is built to recover, so its concordance on this set
# measures how well it fits a well-specified problem, not an artifact of a
# mismatched data-generating process. `_SHAPE > 1` models hazard increasing
# with time (plausible: an aging defect gets more escalation attention), and
# ~25% of cases are deliberately left censored (still "open" at a random
# earlier cutoff), to exercise the censored-data path (US-31 AC) even here.
_SHAPE = 1.5
_SCALE_DAYS = 10.0


def generate_bootstrap_history(n: int = 250, seed: int = 42) -> list[ForecastExample]:
    rng = random.Random(seed)
    base = datetime(2024, 1, 1)
    severities = list(_SEVERITY_RANK)
    priorities = list(_PRIORITY_RANK)

    examples = []
    for i in range(n):
        severity = rng.choice(severities)
        priority = rng.choice(priorities)
        component = rng.choice(_COMPONENTS)
        linear_predictor = (
            0.7 * _SEVERITY_RANK[severity]
            + 0.4 * _PRIORITY_RANK[priority]
            + 1.0 * _COMPONENT_HAZARD[component]
        )
        u = rng.random()
        true_duration = (
            _SCALE_DAYS * (-math.log(u)) ** (1.0 / _SHAPE) * math.exp(-linear_predictor / _SHAPE)
        )
        reported_at = base + timedelta(hours=i)

        if rng.random() < 0.25:
            censored_duration = true_duration * rng.uniform(0.2, 0.9)
            examples.append(
                ForecastExample(
                    defect_id=-(i + 1),
                    severity=severity,
                    priority=priority,
                    component=component,
                    duration_days=censored_duration,
                    event_observed=False,
                    reported_at=reported_at,
                )
            )
        else:
            examples.append(
                ForecastExample(
                    defect_id=-(i + 1),
                    severity=severity,
                    priority=priority,
                    component=component,
                    duration_days=true_duration,
                    event_observed=True,
                    reported_at=reported_at,
                )
            )
    return examples
