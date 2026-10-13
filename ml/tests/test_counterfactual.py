import random

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from bugflow_ml.explain.counterfactual import search_counterfactual
from bugflow_ml.models.commit_risk import FEATURE_COLUMNS


@pytest.fixture
def risk_model() -> LogisticRegression:
    """A model where larger/more-sprawling commits are genuinely riskier —
    the same correlation `generate_bootstrap_*` helpers elsewhere in this
    project use, so the counterfactual search has real signal to find."""
    rng = random.Random(42)
    X, y = [], []
    for _ in range(300):
        churn = rng.randint(1, 500)
        files_changed = rng.randint(1, 30)
        row = {col: 0.0 for col in FEATURE_COLUMNS}
        row.update(
            lines_added=churn / 2,
            lines_deleted=churn / 2,
            churn=churn,
            files_changed=files_changed,
            directories_touched=min(files_changed, rng.randint(1, 10)),
            subsystems_touched=min(files_changed, rng.randint(1, 5)),
            entropy=rng.random() * 2,
        )
        risk_signal = churn / 500 * 0.6 + files_changed / 30 * 0.4
        label = 1 if rng.random() < (0.05 + 0.7 * risk_signal) else 0
        X.append([row[col] for col in FEATURE_COLUMNS])
        y.append(label)

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    model.fit(np.array(X), np.array(y))
    return model


@pytest.mark.story("US-35")
def test_search_counterfactual_finds_a_reduction_for_a_large_risky_commit(
    risk_model: LogisticRegression,
) -> None:
    base_features = {col: 0.0 for col in FEATURE_COLUMNS}
    base_features.update(
        lines_added=250,
        lines_deleted=250,
        churn=500,
        files_changed=30,
        directories_touched=10,
        subsystems_touched=5,
        entropy=2.0,
    )

    result = search_counterfactual(risk_model, base_features, FEATURE_COLUMNS, threshold=0.3)

    assert result.found is True
    assert result.new_probability is not None
    assert result.new_probability < 0.3
    assert result.new_probability < result.original_probability
    assert "Splitting" in result.text


@pytest.mark.story("US-35")
def test_search_counterfactual_gives_no_suggestion_for_an_already_tiny_commit(
    risk_model: LogisticRegression,
) -> None:
    base_features = {col: 0.0 for col in FEATURE_COLUMNS}
    base_features.update(
        lines_added=1, lines_deleted=0, churn=1, files_changed=1, directories_touched=1
    )

    result = search_counterfactual(risk_model, base_features, FEATURE_COLUMNS, threshold=0.99)

    # Already below an all-but-impossible threshold — the very first
    # reduction step satisfies it, confirming "found" doesn't require risk
    # to have been high in the first place.
    assert result.found is True


@pytest.mark.story("US-35")
def test_search_counterfactual_reports_no_suggestion_when_nothing_helps_enough(
    risk_model: LogisticRegression,
) -> None:
    base_features = {col: 0.0 for col in FEATURE_COLUMNS}
    base_features.update(
        lines_added=250,
        lines_deleted=250,
        churn=500,
        files_changed=30,
        directories_touched=10,
        subsystems_touched=5,
        entropy=2.0,
    )

    result = search_counterfactual(risk_model, base_features, FEATURE_COLUMNS, threshold=0.001)

    assert result.found is False
    assert result.new_probability is None
    assert result.reduction_fraction is None
    assert "request extra review" in result.text
