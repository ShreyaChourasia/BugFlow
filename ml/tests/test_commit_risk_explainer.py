import random
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from bugflow_ml.explain.commit_risk_explainer import build_explainer, explain_prediction
from bugflow_ml.models.commit_risk import TrainingCommit, train_commit_risk

BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _make_commits(n: int, seed: int = 0) -> list[TrainingCommit]:
    rng = random.Random(seed)
    commits = []
    for i in range(n):
        churn = rng.randint(1, 200)
        files_changed = rng.randint(1, 10)
        risk_signal = churn / 200 * 0.6 + files_changed / 10 * 0.4
        is_bug = rng.random() < (0.05 + 0.5 * risk_signal)
        features = {
            "lines_added": churn // 2,
            "lines_deleted": churn // 2,
            "churn": churn,
            "files_changed": files_changed,
            "directories_touched": rng.randint(1, files_changed),
            "subsystems_touched": rng.randint(1, min(3, files_changed)),
            "entropy": rng.random() * 2,
            "author_prior_commits": rng.randint(0, 50),
            "is_fix": rng.random() < 0.3,
        }
        commits.append(
            TrainingCommit(
                id=i,
                sha=f"sha{i}",
                timestamp=BASE_TIME + timedelta(hours=i),
                features=features,
                is_bug_inducing=is_bug,
            )
        )
    return commits


@pytest.fixture(scope="module")
def trained_explainer() -> Any:
    result = train_commit_risk(_make_commits(150), seed=42)
    return build_explainer(result.raw_model)


@pytest.mark.story("US-34")
def test_explanation_has_at_most_three_factors_with_nonempty_text(trained_explainer: Any) -> None:
    feature_vector = {
        "lines_added": 100,
        "lines_deleted": 20,
        "churn": 120,
        "files_changed": 8,
        "directories_touched": 5,
        "subsystems_touched": 3,
        "entropy": 1.5,
        "author_prior_commits": 1,
        "is_fix": 0,
    }

    explanation = explain_prediction(trained_explainer, feature_vector)

    assert explanation["text"]
    assert 1 <= len(explanation["factors"]) <= 3
    for factor in explanation["factors"]:
        assert factor["impact"] in {"increases", "decreases"}


@pytest.mark.story("US-36")
def test_explanation_is_deterministic_for_the_same_input(trained_explainer: Any) -> None:
    feature_vector = {
        "lines_added": 40,
        "lines_deleted": 5,
        "churn": 45,
        "files_changed": 3,
        "directories_touched": 2,
        "subsystems_touched": 1,
        "entropy": 0.8,
        "author_prior_commits": 20,
        "is_fix": 1,
    }

    first = explain_prediction(trained_explainer, feature_vector)
    second = explain_prediction(trained_explainer, feature_vector)

    assert first == second
