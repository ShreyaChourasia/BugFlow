import random

import pytest

from bugflow_ml.monitoring.drift import compute_feature_drift, population_stability_index


@pytest.mark.story("US-41")
def test_psi_is_near_zero_for_identical_distributions() -> None:
    rng = random.Random(42)
    values = [rng.gauss(0, 1) for _ in range(500)]

    psi = population_stability_index(values, values)

    assert psi < 0.01


@pytest.mark.story("US-41")
def test_psi_is_large_for_a_clearly_shifted_distribution() -> None:
    rng = random.Random(42)
    reference = [rng.gauss(0, 1) for _ in range(500)]
    shifted = [rng.gauss(5, 1) for _ in range(500)]

    psi = population_stability_index(reference, shifted)

    assert psi > 0.2


@pytest.mark.story("US-41")
def test_psi_handles_too_few_reference_points_without_raising() -> None:
    assert population_stability_index([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


@pytest.mark.story("US-41")
def test_psi_handles_a_constant_reference_without_raising() -> None:
    assert population_stability_index([5.0] * 20, [1.0, 2.0, 3.0]) == 0.0


@pytest.mark.story("US-41")
def test_compute_feature_drift_flags_only_the_shifted_feature() -> None:
    rng = random.Random(7)
    stable = [rng.gauss(0, 1) for _ in range(300)]
    reference = {"churn": stable, "files_changed": stable}
    current = {
        "churn": [rng.gauss(0, 1) for _ in range(300)],  # stable
        "files_changed": [rng.gauss(6, 1) for _ in range(300)],  # shifted
    }

    results = compute_feature_drift(reference, current, threshold=0.2)

    by_feature = {r.feature: r for r in results}
    assert by_feature["files_changed"].severity in {"moderate", "severe"}
    assert by_feature["churn"].severity == "none"
    assert results[0].feature == "files_changed"  # worst-first


def test_compute_feature_drift_skips_features_missing_from_current() -> None:
    results = compute_feature_drift({"churn": [1.0] * 20}, {}, threshold=0.2)
    assert results == []
