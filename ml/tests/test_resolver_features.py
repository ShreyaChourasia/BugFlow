from datetime import datetime

from bugflow_ml.features.resolver_features import ResolvedReport, compute_resolver_features


def test_no_history_is_a_cold_start() -> None:
    features = compute_resolver_features(
        report_embedding=[1.0, 0.0],
        report_component="login",
        past_resolutions=[],
        developer_skills=["login"],
        developer_capacity=5,
        developer_current_queue_depth=2,
        as_of=datetime(2024, 6, 1),
    )

    assert features.has_history is False
    assert features.text_similarity == 0.0
    assert features.skill_match is True
    assert features.current_load_ratio == 0.4


def test_history_computes_similarity_and_component_match() -> None:
    past = [
        ResolvedReport(embedding=[1.0, 0.0], component="login", resolved_at=datetime(2024, 5, 1)),
        ResolvedReport(embedding=[0.0, 1.0], component="billing", resolved_at=datetime(2024, 1, 1)),
    ]

    features = compute_resolver_features(
        report_embedding=[1.0, 0.0],
        report_component="login",
        past_resolutions=past,
        developer_skills=[],
        developer_capacity=5,
        developer_current_queue_depth=0,
        as_of=datetime(2024, 6, 1),
    )

    assert features.has_history is True
    assert features.text_similarity == 1.0
    assert features.component_match_count == 1


def test_recency_score_decays_with_time() -> None:
    recent = compute_resolver_features(
        report_embedding=[1.0, 0.0],
        report_component="login",
        past_resolutions=[
            ResolvedReport(
                embedding=[1.0, 0.0], component="login", resolved_at=datetime(2024, 5, 30)
            )
        ],
        developer_skills=[],
        developer_capacity=5,
        developer_current_queue_depth=0,
        as_of=datetime(2024, 6, 1),
    )
    old = compute_resolver_features(
        report_embedding=[1.0, 0.0],
        report_component="login",
        past_resolutions=[
            ResolvedReport(
                embedding=[1.0, 0.0], component="login", resolved_at=datetime(2023, 1, 1)
            )
        ],
        developer_skills=[],
        developer_capacity=5,
        developer_current_queue_depth=0,
        as_of=datetime(2024, 6, 1),
    )

    assert recent.recency_score > old.recency_score


def test_zero_capacity_treated_as_fully_loaded() -> None:
    features = compute_resolver_features(
        report_embedding=[1.0, 0.0],
        report_component=None,
        past_resolutions=[],
        developer_skills=[],
        developer_capacity=0,
        developer_current_queue_depth=0,
        as_of=datetime(2024, 6, 1),
    )

    assert features.current_load_ratio == 1.0
