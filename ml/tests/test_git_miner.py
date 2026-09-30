from pathlib import Path

import pytest

from bugflow_ml.mining.git_miner import mine_commits, mine_specific_commits

from .gitutil import commit_all, init_repo


@pytest.fixture
def three_commit_repo(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    init_repo(repo)

    (repo / "a.py").write_text("x = 1\n")
    c1 = commit_all(repo, "c1")

    (repo / "b").mkdir()
    (repo / "b" / "c.py").write_text("y = 2\n")
    c2 = commit_all(repo, "c2: add module")

    (repo / "a.py").write_text("x = 1\nz = 3\n")
    c3 = commit_all(repo, "c3: fixes #7")

    return {"path": str(repo), "shas": [c1, c2, c3]}


@pytest.mark.story("US-02")
def test_mine_commits_returns_chronological_order(three_commit_repo: dict[str, object]) -> None:
    mined = list(mine_commits(three_commit_repo["path"]))

    assert [m.sha for m in mined] == three_commit_repo["shas"]
    assert mined[1].files[0].path == "b/c.py"
    assert mined[2].lines_added == 1


@pytest.mark.story("US-05")
def test_mine_commits_resumes_after_checkpoint(three_commit_repo: dict[str, object]) -> None:
    shas = three_commit_repo["shas"]

    resumed = list(mine_commits(three_commit_repo["path"], after_sha=shas[0]))

    assert [m.sha for m in resumed] == shas[1:]


@pytest.mark.story("US-07")
def test_mine_commits_picks_up_newly_pushed_commits(three_commit_repo: dict[str, object]) -> None:
    shas = three_commit_repo["shas"]
    repo_path = Path(three_commit_repo["path"])

    # Simulate a prior run that stopped at the last commit seen so far...
    already_mined = list(mine_commits(three_commit_repo["path"]))
    assert [m.sha for m in already_mined] == shas

    # ...then new commits get pushed...
    (repo_path / "a.py").write_text("x = 1\nz = 3\nw = 4\n")
    new_sha = commit_all(repo_path, "c4: new work")

    # ...and a re-run from the old checkpoint only picks up what's new.
    incremental = list(mine_commits(three_commit_repo["path"], after_sha=shas[-1]))

    assert [m.sha for m in incremental] == [new_sha]


@pytest.mark.story("US-13")
def test_mine_specific_commits_filters_and_preserves_order(
    three_commit_repo: dict[str, object],
) -> None:
    shas = three_commit_repo["shas"]

    mined = list(mine_specific_commits(three_commit_repo["path"], [shas[2], shas[0]]))

    assert [m.sha for m in mined] == [shas[0], shas[2]]
    assert mined[1].files[0].diff_added


def test_mine_specific_commits_empty_shas_yields_nothing(
    three_commit_repo: dict[str, object],
) -> None:
    assert list(mine_specific_commits(three_commit_repo["path"], [])) == []
