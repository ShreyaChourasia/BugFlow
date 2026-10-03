"""US-24/US-30: per-(report, developer) features for resolver suitability —
§8's "text similarity to the developer's past fixes, component ownership,
recency." Framework-agnostic (plain dataclasses/numpy, no SQLAlchemy) so the
same code trains and predicts; the caller supplies each developer's past
resolved reports (already fetched from the DB) rather than this module
querying anything itself.
"""

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

RECENCY_HALF_LIFE_DAYS = 90.0


@dataclass
class ResolvedReport:
    """One defect report a developer has previously resolved — the raw
    material for "text similarity to past fixes" and "component ownership."
    """

    embedding: list[float]
    component: str | None
    resolved_at: datetime


@dataclass
class ResolverFeatures:
    has_history: bool
    text_similarity: float
    component_match_count: int
    recency_score: float
    skill_match: bool
    current_load_ratio: float


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def compute_resolver_features(
    report_embedding: list[float],
    report_component: str | None,
    past_resolutions: list[ResolvedReport],
    developer_skills: list[str],
    developer_capacity: int,
    developer_current_queue_depth: int,
    as_of: datetime,
) -> ResolverFeatures:
    """One feature vector for a single (report, developer) candidate pair."""
    skill_match = report_component is not None and report_component in developer_skills
    load_ratio = (
        developer_current_queue_depth / developer_capacity if developer_capacity > 0 else 1.0
    )

    if not past_resolutions:
        return ResolverFeatures(
            has_history=False,
            text_similarity=0.0,
            component_match_count=0,
            recency_score=0.0,
            skill_match=skill_match,
            current_load_ratio=load_ratio,
        )

    report_vec = np.array(report_embedding)
    similarities = [_cosine_similarity(report_vec, np.array(r.embedding)) for r in past_resolutions]
    component_match_count = sum(
        1
        for r in past_resolutions
        if report_component is not None and r.component == report_component
    )
    most_recent_days = min(
        (as_of - r.resolved_at).total_seconds() / 86400 for r in past_resolutions
    )
    recency_score = math.exp(-max(most_recent_days, 0.0) / RECENCY_HALF_LIFE_DAYS)

    return ResolverFeatures(
        has_history=True,
        text_similarity=max(similarities),
        component_match_count=component_match_count,
        recency_score=recency_score,
        skill_match=skill_match,
        current_load_ratio=load_ratio,
    )


def features_to_vector(features: ResolverFeatures) -> list[float]:
    """The numeric feature vector the trained model actually sees — only for
    candidates `features.has_history` is True; cold-start candidates (US-30)
    are scored by a fixed prior in `resolver_suitability.py`, not this model."""
    return [
        features.text_similarity,
        float(features.component_match_count),
        features.recency_score,
        float(features.skill_match),
        features.current_load_ratio,
    ]
