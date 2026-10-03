from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bugflow_ml.assignment.batch_assignment import (
    AssignmentCandidate,
    DeveloperCapacity,
    compare_optimizer_vs_greedy,
    gini_coefficient,
    greedy_top1_assignment,
    solve_batch_assignment,
)


def _instance(n_defects: int, n_developers: int, capacity: int, seed: int) -> tuple:
    import random

    rng = random.Random(seed)
    capacities = [
        DeveloperCapacity(developer_id=d, available_slots=capacity) for d in range(n_developers)
    ]
    candidates = [
        AssignmentCandidate(defect_id=defect, developer_id=dev, score=rng.random())
        for defect in range(n_defects)
        for dev in range(n_developers)
    ]
    return candidates, capacities


@given(
    n_defects=st.integers(min_value=0, max_value=15),
    n_developers=st.integers(min_value=1, max_value=6),
    capacity=st.integers(min_value=0, max_value=5),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.story("US-25")
def test_optimizer_never_exceeds_capacity(
    n_defects: int, n_developers: int, capacity: int, seed: int
) -> None:
    candidates, capacities = _instance(n_defects, n_developers, capacity, seed)

    result = solve_batch_assignment(candidates, capacities)

    loads = Counter(developer_id for _, developer_id in result.assignments)
    for cap in capacities:
        assert loads.get(cap.developer_id, 0) <= cap.available_slots


@given(
    n_defects=st.integers(min_value=0, max_value=15),
    n_developers=st.integers(min_value=1, max_value=6),
    capacity=st.integers(min_value=0, max_value=5),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.story("US-25")
def test_greedy_never_exceeds_capacity(
    n_defects: int, n_developers: int, capacity: int, seed: int
) -> None:
    candidates, capacities = _instance(n_defects, n_developers, capacity, seed)

    result = greedy_top1_assignment(candidates, capacities)

    loads = Counter(developer_id for _, developer_id in result.assignments)
    for cap in capacities:
        assert loads.get(cap.developer_id, 0) <= cap.available_slots


@given(
    n_defects=st.integers(min_value=0, max_value=15),
    n_developers=st.integers(min_value=1, max_value=6),
    capacity=st.integers(min_value=0, max_value=5),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.story("US-26")
def test_every_defect_is_assigned_or_in_shortfall_exactly_once(
    n_defects: int, n_developers: int, capacity: int, seed: int
) -> None:
    candidates, capacities = _instance(n_defects, n_developers, capacity, seed)

    result = solve_batch_assignment(candidates, capacities)

    assigned_defects = {defect_id for defect_id, _ in result.assignments}
    all_defects = {c.defect_id for c in candidates}
    assert assigned_defects.isdisjoint(result.shortfall)
    assert assigned_defects | set(result.shortfall) == all_defects
    assert len(assigned_defects) == len(result.assignments)  # no defect assigned twice


@pytest.mark.story("US-26")
def test_shortfall_reported_when_demand_exceeds_capacity() -> None:
    capacities = [DeveloperCapacity(developer_id=1, available_slots=1)]
    candidates = [
        AssignmentCandidate(defect_id=1, developer_id=1, score=0.9),
        AssignmentCandidate(defect_id=2, developer_id=1, score=0.5),
    ]

    result = solve_batch_assignment(candidates, capacities)

    assert len(result.assignments) == 1
    assert result.assignments[0] == (1, 1)  # the higher-scoring defect wins the one slot
    assert result.shortfall == [2]


def test_no_candidates_returns_empty_result() -> None:
    result = solve_batch_assignment([], [DeveloperCapacity(developer_id=1, available_slots=3)])
    assert result.assignments == []
    assert result.shortfall == []


@pytest.mark.story("US-25")
def test_optimizer_achieves_at_least_as_much_total_suitability_as_greedy() -> None:
    candidates, capacities = _instance(n_defects=20, n_developers=5, capacity=3, seed=1)

    comparison = compare_optimizer_vs_greedy(candidates, capacities)

    assert comparison.optimizer_total_suitability >= comparison.greedy_total_suitability


def test_gini_coefficient_zero_when_even() -> None:
    assert gini_coefficient([3, 3, 3, 3]) == 0.0


def test_gini_coefficient_positive_when_uneven() -> None:
    assert gini_coefficient([0, 0, 0, 10]) > 0.5


def test_ablation_without_capacity_can_overload_a_single_developer() -> None:
    """§8's ablation: dropping the capacity constraint should let the
    highest-scoring developer absorb more than their real capacity."""
    capacities = [
        DeveloperCapacity(developer_id=1, available_slots=1),
        DeveloperCapacity(developer_id=2, available_slots=1),
    ]
    candidates = [
        AssignmentCandidate(defect_id=d, developer_id=1, score=0.9) for d in range(1, 4)
    ] + [AssignmentCandidate(defect_id=d, developer_id=2, score=0.1) for d in range(1, 4)]

    with_capacity = solve_batch_assignment(candidates, capacities, enforce_capacity=True)
    without_capacity = solve_batch_assignment(candidates, capacities, enforce_capacity=False)

    with_loads = Counter(dev for _, dev in with_capacity.assignments)
    without_loads = Counter(dev for _, dev in without_capacity.assignments)
    assert with_loads[1] <= 1
    assert without_loads[1] == 3  # unconstrained, developer 1 takes everything
