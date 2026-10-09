import math
from datetime import datetime

import pytest

from bugflow_ml.models.resolution_forecast import (
    MIN_TRAINING_EXAMPLES,
    ForecastExample,
    compute_data_version,
    generate_bootstrap_history,
    is_at_risk,
    predict_forecast,
    train_forecast_model,
)


@pytest.mark.story("US-31")
def test_bootstrap_history_includes_both_events_and_censored_cases() -> None:
    examples = generate_bootstrap_history(seed=42)

    assert any(e.event_observed for e in examples)
    assert any(not e.event_observed for e in examples)


@pytest.mark.story("US-31")
def test_train_rejects_too_few_examples() -> None:
    examples = generate_bootstrap_history(n=MIN_TRAINING_EXAMPLES - 1, seed=42)

    with pytest.raises(ValueError, match="at least"):
        train_forecast_model(examples, seed=42)


@pytest.mark.story("US-31")
def test_train_forecast_model_beats_random_chance() -> None:
    examples = generate_bootstrap_history(seed=42)

    result = train_forecast_model(examples, seed=42)

    assert result.metrics["cox_concordance"] > 0.5
    assert result.metrics["km_baseline_concordance"] > 0.5


@pytest.mark.story("US-31")
@pytest.mark.story("US-32")
def test_predict_forecast_always_states_a_finite_median_and_p90() -> None:
    examples = generate_bootstrap_history(seed=42)
    result = train_forecast_model(examples, seed=42)

    prediction = predict_forecast(result, "blocker", "P1", "login")

    assert math.isfinite(prediction.median_days)
    assert math.isfinite(prediction.p90_days)
    assert prediction.p90_days >= prediction.median_days
    assert len(prediction.curve) > 0


@pytest.mark.story("US-31")
def test_higher_severity_and_priority_predicts_a_shorter_resolution_time() -> None:
    """The bootstrap generator makes higher severity/priority resolve
    sooner (worked on first) — the fitted model should recover that."""
    examples = generate_bootstrap_history(seed=42)
    result = train_forecast_model(examples, seed=42)

    urgent = predict_forecast(result, "blocker", "P1", "login")
    relaxed = predict_forecast(result, "trivial", "P5", "login")

    assert urgent.median_days < relaxed.median_days


@pytest.mark.story("US-33")
def test_is_at_risk_flags_a_still_open_defect_past_its_p90_window() -> None:
    assert is_at_risk(elapsed_days=15.0, p90_days=10.0, is_resolved=False) is True


@pytest.mark.story("US-33")
def test_is_at_risk_false_within_the_expected_window() -> None:
    assert is_at_risk(elapsed_days=5.0, p90_days=10.0, is_resolved=False) is False


@pytest.mark.story("US-33")
def test_is_at_risk_always_false_once_resolved() -> None:
    assert is_at_risk(elapsed_days=100.0, p90_days=10.0, is_resolved=True) is False


def test_duration_version_differs_for_different_data() -> None:
    a = [ForecastExample(1, "blocker", "P1", "login", 5.0, True, datetime(2024, 1, 1))]
    b = [ForecastExample(1, "blocker", "P1", "login", 6.0, True, datetime(2024, 1, 1))]

    assert compute_data_version(a) != compute_data_version(b)
