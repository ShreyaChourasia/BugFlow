"""US-25/US-26: capacity-constrained batch assignment — the project's novel
contribution. Maximises total suitability across a whole batch of defects at
once, subject to each developer's remaining capacity, using OR-Tools CP-SAT
(§8 names "OR-Tools min-cost flow or CP-SAT" — CP-SAT needs no extra solver
backend beyond the `ortools` package itself).

Unassignable defects (demand > total available capacity, or no candidate for
that defect at all) are returned as a shortfall rather than silently
dropped or assigned somewhere unsuitable — §5 explicitly asks that the
shortfall be reported, not hidden.
"""

from dataclasses import dataclass

from ortools.sat.python import cp_model

# CP-SAT needs integer objective coefficients; suitability scores are floats
# in [0, 1], so scale them up before rounding rather than truncating to 0/1.
_SCORE_SCALE = 10_000


@dataclass
class AssignmentCandidate:
    defect_id: int
    developer_id: int
    score: float


@dataclass
class DeveloperCapacity:
    developer_id: int
    available_slots: int


@dataclass
class BatchAssignmentResult:
    assignments: list[tuple[int, int]]  # (defect_id, developer_id)
    shortfall: list[int]  # defect_ids with no assignment
    total_suitability: float


def _all_defect_ids(candidates: list[AssignmentCandidate]) -> list[int]:
    seen: dict[int, None] = {}
    for c in candidates:
        seen.setdefault(c.defect_id, None)
    return list(seen)


def solve_batch_assignment(
    candidates: list[AssignmentCandidate],
    capacities: list[DeveloperCapacity],
    enforce_capacity: bool = True,
) -> BatchAssignmentResult:
    """US-25: the optimal assignment. `enforce_capacity=False` is the
    ablation §8 asks for — every developer gets effectively unlimited
    capacity, isolating what the capacity constraint itself changes."""
    if not candidates:
        return BatchAssignmentResult(assignments=[], shortfall=[], total_suitability=0.0)

    model = cp_model.CpModel()
    available_slots = {c.developer_id: c.available_slots for c in capacities}

    by_defect: dict[int, list[AssignmentCandidate]] = {}
    by_developer: dict[int, list[AssignmentCandidate]] = {}
    variables: dict[tuple[int, int], cp_model.IntVar] = {}
    for c in candidates:
        by_defect.setdefault(c.defect_id, []).append(c)
        by_developer.setdefault(c.developer_id, []).append(c)
        variables[(c.defect_id, c.developer_id)] = model.new_bool_var(
            f"x_{c.defect_id}_{c.developer_id}"
        )

    for defect_id, defect_candidates in by_defect.items():
        model.add(sum(variables[(defect_id, c.developer_id)] for c in defect_candidates) <= 1)

    if enforce_capacity:
        for developer_id, developer_candidates in by_developer.items():
            slots = available_slots.get(developer_id, 0)
            model.add(
                sum(variables[(c.defect_id, developer_id)] for c in developer_candidates) <= slots
            )

    model.maximize(
        sum(
            round(c.score * _SCORE_SCALE) * variables[(c.defect_id, c.developer_id)]
            for c in candidates
        )
    )

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # No candidates fit within capacity at all — everything is shortfall,
        # not an error (C7-style "degrade, don't crash").
        return BatchAssignmentResult(
            assignments=[], shortfall=_all_defect_ids(candidates), total_suitability=0.0
        )

    assignments = []
    score_by_pair = {(c.defect_id, c.developer_id): c.score for c in candidates}
    for (defect_id, developer_id), var in variables.items():
        if solver.value(var):
            assignments.append((defect_id, developer_id))
    assigned_defects = {defect_id for defect_id, _ in assignments}
    shortfall = [d for d in by_defect if d not in assigned_defects]
    total_suitability = sum(score_by_pair[pair] for pair in assignments)

    return BatchAssignmentResult(
        assignments=assignments, shortfall=shortfall, total_suitability=total_suitability
    )


def greedy_top1_assignment(
    candidates: list[AssignmentCandidate], capacities: list[DeveloperCapacity]
) -> BatchAssignmentResult:
    """The baseline §8 asks the optimiser be compared against: process
    defects in score order, always take each one's single best remaining
    candidate. No look-ahead, so it can leave capacity stranded that the
    optimal batch solution would have used for a different defect."""
    remaining_slots = {c.developer_id: c.available_slots for c in capacities}
    by_defect: dict[int, list[AssignmentCandidate]] = {}
    for c in candidates:
        by_defect.setdefault(c.defect_id, []).append(c)

    assignments = []
    shortfall = []
    for defect_id, defect_candidates in by_defect.items():
        ranked = sorted(defect_candidates, key=lambda c: -c.score)
        chosen = next((c for c in ranked if remaining_slots.get(c.developer_id, 0) > 0), None)
        if chosen is None:
            shortfall.append(defect_id)
            continue
        assignments.append((chosen.defect_id, chosen.developer_id))
        remaining_slots[chosen.developer_id] -= 1

    total_suitability = sum(
        c.score for c in candidates if (c.defect_id, c.developer_id) in assignments
    )
    return BatchAssignmentResult(
        assignments=assignments, shortfall=shortfall, total_suitability=total_suitability
    )


def gini_coefficient(values: list[float]) -> float:
    """0 = perfectly even load distribution, 1 = maximally uneven."""
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0.0
    sorted_values = sorted(values)
    cumulative = sum((i + 1) * v for i, v in enumerate(sorted_values))
    return (2 * cumulative) / (n * sum(sorted_values)) - (n + 1) / n


@dataclass
class AssignmentComparison:
    optimizer_total_suitability: float
    greedy_total_suitability: float
    optimizer_shortfall_count: int
    greedy_shortfall_count: int
    optimizer_max_load: int
    greedy_max_load: int
    optimizer_load_gini: float
    greedy_load_gini: float


def compare_optimizer_vs_greedy(
    candidates: list[AssignmentCandidate], capacities: list[DeveloperCapacity]
) -> AssignmentComparison:
    """§8's "Research evaluation": the optimiser vs. the greedy baseline on
    total suitability, max load, and load balance (Gini) — the numbers this
    produces on the bootstrap dataset are recorded in docs/ml.md."""
    optimal = solve_batch_assignment(candidates, capacities)
    greedy = greedy_top1_assignment(candidates, capacities)

    def loads(result: BatchAssignmentResult) -> dict[int, int]:
        counts: dict[int, int] = {c.developer_id: 0 for c in capacities}
        for _, developer_id in result.assignments:
            counts[developer_id] = counts.get(developer_id, 0) + 1
        return counts

    optimal_loads = loads(optimal)
    greedy_loads = loads(greedy)

    return AssignmentComparison(
        optimizer_total_suitability=optimal.total_suitability,
        greedy_total_suitability=greedy.total_suitability,
        optimizer_shortfall_count=len(optimal.shortfall),
        greedy_shortfall_count=len(greedy.shortfall),
        optimizer_max_load=max(optimal_loads.values(), default=0),
        greedy_max_load=max(greedy_loads.values(), default=0),
        optimizer_load_gini=gini_coefficient(list(optimal_loads.values())),
        greedy_load_gini=gini_coefficient(list(greedy_loads.values())),
    )
