import pytest

from bugflow_ml.features.commit_features import compute_commit_features
from bugflow_ml.mining.git_miner import ModifiedFileInfo


@pytest.mark.story("US-06")
def test_single_file_change_has_zero_entropy() -> None:
    files = [ModifiedFileInfo(path="a/b.py", added_lines=3, deleted_lines=1)]

    features = compute_commit_features(
        files, lines_added=3, lines_deleted=1, is_fix=False, author_prior_commits=2
    )

    assert features["files_changed"] == 1
    assert features["directories_touched"] == 1
    assert features["subsystems_touched"] == 1
    assert features["entropy"] == 0.0
    assert features["churn"] == 4
    assert features["author_prior_commits"] == 2
    assert features["is_fix"] is False


@pytest.mark.story("US-06")
def test_evenly_spread_changes_have_higher_entropy_than_skewed() -> None:
    even = [
        ModifiedFileInfo(path="a/one.py", added_lines=5, deleted_lines=0),
        ModifiedFileInfo(path="b/two.py", added_lines=5, deleted_lines=0),
    ]
    skewed = [
        ModifiedFileInfo(path="a/one.py", added_lines=9, deleted_lines=0),
        ModifiedFileInfo(path="b/two.py", added_lines=1, deleted_lines=0),
    ]

    even_features = compute_commit_features(
        even, lines_added=10, lines_deleted=0, is_fix=False, author_prior_commits=0
    )
    skewed_features = compute_commit_features(
        skewed, lines_added=10, lines_deleted=0, is_fix=False, author_prior_commits=0
    )

    assert even_features["entropy"] == 1.0  # two equally-sized buckets -> 1 bit
    assert skewed_features["entropy"] < even_features["entropy"]


@pytest.mark.story("US-06")
def test_diffusion_counts_distinct_directories_and_subsystems() -> None:
    files = [
        ModifiedFileInfo(path="api/routes/a.py", added_lines=1, deleted_lines=0),
        ModifiedFileInfo(path="api/routes/b.py", added_lines=1, deleted_lines=0),
        ModifiedFileInfo(path="worker/jobs/c.py", added_lines=1, deleted_lines=0),
    ]

    features = compute_commit_features(
        files, lines_added=3, lines_deleted=0, is_fix=True, author_prior_commits=5
    )

    assert features["files_changed"] == 3
    assert features["directories_touched"] == 2  # api/routes, worker/jobs
    assert features["subsystems_touched"] == 2  # api, worker
    assert features["is_fix"] is True
