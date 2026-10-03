from datetime import datetime

import pytest

from bugflow_ml.features.resolver_features import ResolvedReport
from bugflow_ml.models.resolver_suitability import (
    MIN_TRAINING_EXAMPLES,
    DeveloperCandidate,
    generate_bootstrap_history,
    has_confident_candidate,
    score_candidates,
    train_suitability_model,
)


@pytest.mark.story("US-24")
def test_train_suitability_model_on_bootstrap_data() -> None:
    _, examples = generate_bootstrap_history(n_developers=10, n_reports=200, seed=42)

    result = train_suitability_model(examples, seed=42)

    assert result.metrics["top_3_accuracy"] > 0.5
    assert result.metrics["mrr"] > 0.0
    assert result.train_size > 0
    assert result.test_size > 0


def test_train_rejects_too_few_examples_with_history() -> None:
    _, examples = generate_bootstrap_history(n_developers=2, n_reports=3, seed=42)
    assert len([e for e in examples if e.features.has_history]) < MIN_TRAINING_EXAMPLES

    with pytest.raises(ValueError, match="at least"):
        train_suitability_model(examples, seed=42)


@pytest.mark.story("US-24")
def test_score_candidates_ranks_the_developer_with_matching_history_first() -> None:
    _, examples = generate_bootstrap_history(n_developers=10, n_reports=200, seed=42)
    result = train_suitability_model(examples, seed=42)

    developers = [
        DeveloperCandidate(id=1, name="A", skills=["login"], capacity=5, current_queue_depth=1),
        DeveloperCandidate(id=2, name="B", skills=["billing"], capacity=5, current_queue_depth=1),
    ]
    past_resolutions = {
        1: [
            ResolvedReport(
                embedding=[1.0] + [0.0] * 383, component="login", resolved_at=datetime(2024, 1, 1)
            )
        ]
    }

    scored = score_candidates(
        result.model,
        report_embedding=[1.0] + [0.0] * 383,
        report_component="login",
        candidates=developers,
        past_resolutions_by_developer=past_resolutions,
        as_of=datetime(2024, 6, 1),
    )

    assert scored[0].developer_id == 1
    assert scored[0].is_cold_start is False
    assert scored[1].is_cold_start is True


@pytest.mark.story("US-30")
def test_score_candidates_cold_start_uses_skill_match_prior() -> None:
    _, examples = generate_bootstrap_history(n_developers=10, n_reports=200, seed=42)
    result = train_suitability_model(examples, seed=42)

    developers = [
        DeveloperCandidate(id=1, name="New", skills=["login"], capacity=5, current_queue_depth=0),
        DeveloperCandidate(
            id=2, name="Unrelated", skills=["billing"], capacity=5, current_queue_depth=0
        ),
    ]

    scored = score_candidates(
        result.model,
        report_embedding=[1.0] + [0.0] * 383,
        report_component="login",
        candidates=developers,
        past_resolutions_by_developer={},
        as_of=datetime(2024, 6, 1),
    )

    skill_matched = next(s for s in scored if s.developer_id == 1)
    no_match = next(s for s in scored if s.developer_id == 2)
    assert skill_matched.is_cold_start is True
    assert skill_matched.score > no_match.score
    assert "declared skill" in skill_matched.reason


@pytest.mark.story("US-24")
def test_has_confident_candidate_detects_an_unowned_component() -> None:
    _, examples = generate_bootstrap_history(n_developers=10, n_reports=200, seed=42)
    result = train_suitability_model(examples, seed=42)

    developers = [
        DeveloperCandidate(
            id=1, name="Unrelated", skills=["billing"], capacity=5, current_queue_depth=0
        )
    ]
    scored = score_candidates(
        result.model,
        report_embedding=[1.0] + [0.0] * 383,
        report_component="an-unowned-component",
        candidates=developers,
        past_resolutions_by_developer={},
        as_of=datetime(2024, 6, 1),
    )

    assert has_confident_candidate(scored) is False


def test_has_confident_candidate_empty_list() -> None:
    assert has_confident_candidate([]) is False
