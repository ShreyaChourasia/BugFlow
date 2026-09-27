from pathlib import Path

import pytest

from bugflow_ml.labeling.szz import find_bug_inducing_shas

from .gitutil import commit_all, init_repo


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> dict[str, str]:
    """A tiny repo with a known bug: c1 introduces `return a - b` (should be
    `+`); c2 makes an unrelated addition; c3 fixes the original line. SZZ
    should blame c1, not c2, for the bug c3 fixes."""
    repo = tmp_path / "repo"
    init_repo(repo)

    calc = repo / "calc.py"
    calc.write_text("def add(a, b):\n    return a - b\n")
    c1 = commit_all(repo, "c1: add calc")

    calc.write_text("def add(a, b):\n    return a - b\n\ndef sub(a, b):\n    return a + b\n")
    c2 = commit_all(repo, "c2: add sub")

    calc.write_text("def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a + b\n")
    c3 = commit_all(repo, "c3: fix add, fixes #1")

    return {"path": str(repo), "c1": c1, "c2": c2, "c3": c3}


@pytest.mark.story("US-04")
def test_szz_blames_the_commit_that_introduced_the_buggy_line(
    synthetic_repo: dict[str, str],
) -> None:
    inducing = find_bug_inducing_shas(synthetic_repo["path"], synthetic_repo["c3"])

    assert inducing == {synthetic_repo["c1"]}
    assert synthetic_repo["c2"] not in inducing
