import random
from datetime import UTC, datetime, timedelta

import pytest

from bugflow_ml.models.commit_risk import (
    FEATURE_COLUMNS,
    TrainingCommit,
    build_feature_matrix,
    chronological_split,
    compute_data_version,
    confidence_for,
    fairness_report,
    recall_at_k_percent_effort,
    risk_level_for,
    train_commit_risk,
)

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
                author_tenure_days=rng.choice([10, 300, None]),
            )
        )
    return commits


@pytest.mark.story("US-42")
def test_chronological_split_has_no_leakage() -> None:
    commits = _make_commits(100)
    # shuffle input order to prove the split itself re-sorts, not just luck
    random.Random(1).shuffle(commits)

    train, calib, test = chronological_split(commits, train_frac=0.7, calib_frac=0.15)

    assert len(train) + len(calib) + len(test) == 100
    assert max(c.timestamp for c in train) <= min(c.timestamp for c in calib)
    assert max(c.timestamp for c in calib) <= min(c.timestamp for c in test)

    train_ids = {c.id for c in train}
    calib_ids = {c.id for c in calib}
    test_ids = {c.id for c in test}
    assert not (train_ids & calib_ids)
    assert not (calib_ids & test_ids)
    assert not (train_ids & test_ids)


@pytest.mark.story("US-42")
def test_data_version_is_order_independent_but_content_sensitive() -> None:
    a = compute_data_version([3, 1, 2])
    b = compute_data_version([1, 2, 3])
    c = compute_data_version([1, 2, 4])

    assert a == b
    assert a != c


def test_build_feature_matrix_uses_declared_column_order() -> None:
    commits = [
        TrainingCommit(
            id=1,
            sha="a",
            timestamp=BASE_TIME,
            features={col: i for i, col in enumerate(FEATURE_COLUMNS)},
            is_bug_inducing=True,
        )
    ]

    X, y = build_feature_matrix(commits)

    assert X.shape == (1, len(FEATURE_COLUMNS))
    assert list(X[0]) == list(range(len(FEATURE_COLUMNS)))
    assert list(y) == [1]


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, "low"), (0.29, "low"), (0.3, "medium"), (0.69, "medium"), (0.7, "high"), (1.0, "high")],
)
def test_risk_level_boundaries(probability: float, expected: str) -> None:
    assert risk_level_for(probability) == expected


def test_confidence_is_zero_at_a_coin_flip_and_one_at_the_extremes() -> None:
    assert confidence_for(0.5) == 0.0
    assert confidence_for(0.0) == 1.0
    assert confidence_for(1.0) == 1.0


def test_recall_at_20pct_effort_prioritises_by_predicted_risk() -> None:
    import numpy as np

    # 5 commits; the 2 bug-inducing ones are ranked highest by risk and
    # together make up ~22% of total churn -> both should be captured.
    y_true = np.array([1, 0, 1, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.05])
    churn = np.array([10, 100, 10, 100, 100])

    recall = recall_at_k_percent_effort(y_true, y_prob, churn, k=0.2)

    assert recall == 1.0


def test_fairness_report_without_tenure_data_has_null_fields() -> None:
    import numpy as np

    report = fairness_report(np.array([0.2, 0.8]), [None, None])

    assert report["disparity"] is None
    assert report["junior_count"] == 0
    assert report["senior_count"] == 0


@pytest.mark.story("US-08")
def test_train_commit_risk_requires_a_minimum_amount_of_data() -> None:
    with pytest.raises(ValueError, match="at least"):
        train_commit_risk(_make_commits(5))


@pytest.mark.story("US-08")
@pytest.mark.story("NFR-US-11")
def test_train_commit_risk_end_to_end_produces_a_complete_result() -> None:
    commits = _make_commits(150)

    result = train_commit_risk(commits, seed=42)

    assert result.seed == 42
    assert len(result.train_ids) + len(result.calib_ids) + len(result.test_ids) == 150
    for key in ("roc_auc", "pr_auc", "f1", "brier_score", "recall_at_20pct_effort"):
        assert key in result.metrics
        assert key in result.baseline_metrics
    assert "disparity" in result.fairness  # C10: present in every evaluation


@pytest.mark.story("NFR-US-07")
def test_train_commit_risk_is_reproducible_given_the_same_seed_and_data() -> None:
    commits = _make_commits(150)

    first = train_commit_risk(commits, seed=42)
    second = train_commit_risk(commits, seed=42)

    assert first.data_version == second.data_version
    assert first.metrics == second.metrics
