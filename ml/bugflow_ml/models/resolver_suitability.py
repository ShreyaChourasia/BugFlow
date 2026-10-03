"""US-24/US-30: resolver suitability — §8's "LightGBM ranker or logistic
regression, trained chronologically on actual fixers." Logistic regression
is used here (the simpler of the two baselines §8 names), predicting
P(this developer is the right fit) per (report, developer) candidate pair.

Cold-start candidates (US-30, a developer with no resolution history yet)
are never scored by this model — there's nothing numerically meaningful for
it to say about them. `score_candidates()` in this module gives them a fixed,
low-confidence prior based on declared skills instead, and marks them as such.
"""

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sklearn.linear_model import LogisticRegression

from bugflow_ml.embeddings.duplicate_detection import embed_texts
from bugflow_ml.features.resolver_features import (
    ResolvedReport,
    ResolverFeatures,
    compute_resolver_features,
    features_to_vector,
)

MIN_TRAINING_EXAMPLES = 40
COLD_START_SKILL_MATCH_SCORE = 0.3
COLD_START_NO_SKILL_SCORE = 0.05
NO_CONFIDENT_CANDIDATE_THRESHOLD = 0.2


@dataclass
class SuitabilityExample:
    report_id: int
    developer_id: int
    features: ResolverFeatures
    label: bool
    timestamp: datetime


@dataclass
class SuitabilityTrainingResult:
    seed: int
    data_version: str
    model: Any
    metrics: dict[str, Any]
    train_size: int
    test_size: int


@dataclass
class DeveloperCandidate:
    id: int
    name: str
    skills: list[str]
    capacity: int
    current_queue_depth: int


@dataclass
class ScoredCandidate:
    developer_id: int
    score: float
    confidence: float
    is_cold_start: bool
    reason: str


def compute_data_version(examples: list[SuitabilityExample]) -> str:
    canonical = ",".join(sorted(f"{e.report_id}:{e.developer_id}:{e.label}" for e in examples))
    return hashlib.sha256(canonical.encode()).hexdigest()


def chronological_split(
    examples: list[SuitabilityExample], train_frac: float = 0.8
) -> tuple[list[SuitabilityExample], list[SuitabilityExample]]:
    """C9. Splits by report, not by individual (report, developer) row, so
    every candidate for a given test report stays together for ranking
    metrics — splitting mid-report would leak some of a report's candidates
    into training and leave others for test."""
    report_times: dict[int, datetime] = {}
    for e in examples:
        report_times[e.report_id] = min(report_times.get(e.report_id, e.timestamp), e.timestamp)
    ordered_reports = sorted(report_times, key=lambda r: (report_times[r], r))
    n_train = int(len(ordered_reports) * train_frac)
    train_reports = set(ordered_reports[:n_train])

    train = [e for e in examples if e.report_id in train_reports]
    test = [e for e in examples if e.report_id not in train_reports]
    if train and test:
        assert max(report_times[e.report_id] for e in train) <= min(
            report_times[e.report_id] for e in test
        )
    return train, test


def _rank_metrics(model: Any, test: list[SuitabilityExample]) -> dict[str, float]:
    """Top-1/3/5 accuracy and MRR (§8): group by report, rank candidates by
    predicted probability, find where the real resolver lands."""
    by_report: dict[int, list[SuitabilityExample]] = {}
    for example in test:
        by_report.setdefault(example.report_id, []).append(example)

    reciprocal_ranks = []
    top_k_hits = {1: 0, 3: 0, 5: 0}
    evaluated = 0
    for candidates in by_report.values():
        positive = [c for c in candidates if c.label]
        if not positive:
            continue
        evaluated += 1
        probabilities = model.predict_proba([features_to_vector(c.features) for c in candidates])[
            :, 1
        ]
        ranked = sorted(zip(candidates, probabilities, strict=True), key=lambda pair: -pair[1])
        rank = next(
            i
            for i, (c, _) in enumerate(ranked, start=1)
            if c.developer_id == positive[0].developer_id
        )
        reciprocal_ranks.append(1.0 / rank)
        for k in top_k_hits:
            if rank <= k:
                top_k_hits[k] += 1

    if evaluated == 0:
        return {"top_1_accuracy": 0.0, "top_3_accuracy": 0.0, "top_5_accuracy": 0.0, "mrr": 0.0}
    return {
        "top_1_accuracy": top_k_hits[1] / evaluated,
        "top_3_accuracy": top_k_hits[3] / evaluated,
        "top_5_accuracy": top_k_hits[5] / evaluated,
        "mrr": sum(reciprocal_ranks) / evaluated,
    }


def train_suitability_model(
    examples: list[SuitabilityExample], seed: int = 42, train_frac: float = 0.8
) -> SuitabilityTrainingResult:
    """Trains only on candidates with resolution history — cold-start
    candidates (`has_history=False`) carry no information this model can
    learn from and are excluded, same reasoning as why they're scored by a
    fixed prior at prediction time."""
    with_history = [e for e in examples if e.features.has_history]
    if len(with_history) < MIN_TRAINING_EXAMPLES:
        raise ValueError(
            f"Need at least {MIN_TRAINING_EXAMPLES} examples with resolver history to train; "
            f"got {len(with_history)}."
        )

    train, test = chronological_split(with_history, train_frac)
    if not train or not test:
        raise ValueError("Chronological split produced an empty train/test set.")

    X_train = [features_to_vector(e.features) for e in train]
    y_train = [1 if e.label else 0 for e in train]

    model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    model.fit(X_train, y_train)

    metrics = _rank_metrics(model, test)

    return SuitabilityTrainingResult(
        seed=seed,
        data_version=compute_data_version(examples),
        model=model,
        metrics=metrics,
        train_size=len(train),
        test_size=len(test),
    )


def score_candidates(
    model: Any,
    report_embedding: list[float],
    report_component: str | None,
    candidates: list[DeveloperCandidate],
    past_resolutions_by_developer: dict[int, list[ResolvedReport]],
    as_of: datetime,
) -> list[ScoredCandidate]:
    """US-24/US-30: scores every candidate developer for a report — model-based
    where there's history, a fixed low-confidence prior where there isn't."""
    scored = []
    for candidate in candidates:
        features = compute_resolver_features(
            report_embedding=report_embedding,
            report_component=report_component,
            past_resolutions=past_resolutions_by_developer.get(candidate.id, []),
            developer_skills=candidate.skills,
            developer_capacity=candidate.capacity,
            developer_current_queue_depth=candidate.current_queue_depth,
            as_of=as_of,
        )
        if features.has_history:
            score = float(model.predict_proba([features_to_vector(features)])[0, 1])
            reason = (
                f"Fixed {features.component_match_count} similar report(s) in "
                f"{report_component or 'this area'}, current load "
                f"{candidate.current_queue_depth}/{candidate.capacity}."
            )
            scored.append(
                ScoredCandidate(
                    developer_id=candidate.id,
                    score=score,
                    confidence=score,
                    is_cold_start=False,
                    reason=reason,
                )
            )
        else:
            score = (
                COLD_START_SKILL_MATCH_SCORE if features.skill_match else COLD_START_NO_SKILL_SCORE
            )
            reason = (
                f"New developer; matched on declared skill {report_component}."
                if features.skill_match
                else "New developer; no matching declared skill or resolution history."
            )
            scored.append(
                ScoredCandidate(
                    developer_id=candidate.id,
                    score=score,
                    confidence=score,
                    is_cold_start=True,
                    reason=reason,
                )
            )
    scored.sort(key=lambda c: -c.score)
    return scored


def has_confident_candidate(scored: list[ScoredCandidate]) -> bool:
    """US-24 AC2: "no confident candidate" (e.g. an unowned component)."""
    return bool(scored) and scored[0].score >= NO_CONFIDENT_CANDIDATE_THRESHOLD


# --- bootstrap data -------------------------------------------------------
#
# Same gap as Phases 6/7 (docs/decisions/004, 005): no real resolution
# history exists until the system has actually been used to resolve defects.
# Unlike those phases, though, §8 itself names the fallback for exactly this
# situation — "cold start... prior from declared skills" is what the real
# system leans on until real history accumulates, so this bootstrap set only
# exists to exercise and evaluate the *model-based* path (training needs
# some resolved history to learn from), not to paper over a missing feature.

_COMPONENTS = ["login", "checkout", "search", "dashboard", "billing", "api"]


def generate_bootstrap_history(
    n_developers: int = 10, n_reports: int = 200, seed: int = 42
) -> tuple[list[dict], list[SuitabilityExample]]:
    """Returns (developers, examples). `developers` is a list of dicts
    (id, name, skills, capacity, current_queue_depth) the caller can persist;
    `examples` are the resulting (report, developer) training rows, with each
    developer's features computed from only the resolutions strictly before
    that report — no leakage, same discipline as every other model here."""
    rng = random.Random(seed)
    developers = [
        DeveloperCandidate(
            id=i + 1,
            name=f"Bootstrap Dev {i + 1}",
            skills=[rng.choice(_COMPONENTS)],
            capacity=5,
            current_queue_depth=rng.randint(0, 3),
        )
        for i in range(n_developers)
    ]
    by_component: dict[str, list[DeveloperCandidate]] = {}
    for dev in developers:
        for skill in dev.skills:
            by_component.setdefault(skill, []).append(dev)

    base = datetime(2024, 1, 1)
    texts = []
    components = []
    for i in range(n_reports):
        component = rng.choice(_COMPONENTS)
        texts.append(f"The {component} feature is broken and needs a fix for issue {i}.")
        components.append(component)
    embeddings = embed_texts(texts)

    history: dict[int, list[ResolvedReport]] = {dev.id: [] for dev in developers}
    examples: list[SuitabilityExample] = []
    for i in range(n_reports):
        timestamp = base + timedelta(hours=i)
        component = components[i]
        embedding = embeddings[i]
        owners = by_component.get(component, developers)
        resolver = rng.choice(owners)

        for dev in developers:
            features = compute_resolver_features(
                report_embedding=embedding,
                report_component=component,
                past_resolutions=history[dev.id],
                developer_skills=dev.skills,
                developer_capacity=dev.capacity,
                developer_current_queue_depth=dev.current_queue_depth,
                as_of=timestamp,
            )
            examples.append(
                SuitabilityExample(
                    report_id=i,
                    developer_id=dev.id,
                    features=features,
                    label=dev.id == resolver.id,
                    timestamp=timestamp,
                )
            )

        history[resolver.id].append(
            ResolvedReport(embedding=embedding, component=component, resolved_at=timestamp)
        )

    developer_dicts = [
        {
            "id": dev.id,
            "name": dev.name,
            "skills": dev.skills,
            "capacity": dev.capacity,
            "current_queue_depth": dev.current_queue_depth,
        }
        for dev in developers
    ]
    return developer_dicts, examples
