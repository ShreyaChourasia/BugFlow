from pathlib import Path

import pytest
from sqlalchemy import any_, select
from sqlalchemy.orm import Session

from app.models.commit import Commit
from app.models.developer import Developer
from app.models.repository import MiningRun, Repository
from app.services import mining_service
from app.services.mining_service import run_mining

from .gitutil import commit_all, init_repo


@pytest.fixture
def demo_repo(tmp_path: Path) -> dict[str, object]:
    """c1 introduces a bug (`a - b` instead of `+`), c2 is unrelated, c3
    fixes c1's line — SZZ should mark c1 as bug-inducing."""
    repo = tmp_path / "repo"
    init_repo(repo)

    calc = repo / "calc.py"
    calc.write_text("def add(a, b):\n    return a - b\n")
    c1 = commit_all(repo, "c1: add calc")

    calc.write_text("def add(a, b):\n    return a - b\n\ndef sub(a, b):\n    return a + b\n")
    c2 = commit_all(repo, "c2: add sub")

    calc.write_text("def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a + b\n")
    c3 = commit_all(repo, "c3: fix add, fixes #1")

    return {"path": repo, "shas": [c1, c2, c3]}


@pytest.mark.story("US-04")
def test_run_mining_clones_a_remote_url_once_and_reuses_it_for_szz(
    db_session: Session, demo_repo: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: SZZ used to be called with `repository.url` directly,
    which crashes for a real remote URL — PyDriller's low-level `Git` class
    (unlike `Repository`) only accepts a local path, it doesn't auto-clone.
    Simulates a remote source by faking `git clone` to copy the local fixture
    repo, and checks git is only ever cloned once."""
    import shutil

    clone_count = 0

    def fake_clone(cmd: list[str], check: bool = True) -> None:
        nonlocal clone_count
        clone_count += 1
        shutil.copytree(demo_repo["path"], cmd[-1], dirs_exist_ok=True)

    monkeypatch.setattr(mining_service.subprocess, "run", fake_clone)

    repo_row = Repository(name="demo", url="https://example.invalid/org/demo")
    db_session.add(repo_row)
    db_session.commit()
    run = MiningRun(repository_id=repo_row.id, status="pending")
    db_session.add(run)
    db_session.commit()

    run_mining(db_session, run.id)

    db_session.refresh(run)
    assert run.status == "completed"
    assert clone_count == 1

    commits = list(db_session.scalars(select(Commit).where(Commit.repository_id == repo_row.id)))
    assert any(c.is_bug_inducing for c in commits)  # proves SZZ actually ran, not just mining


@pytest.mark.story("US-02")
@pytest.mark.story("US-04")
@pytest.mark.story("US-06")
def test_run_mining_stores_commits_developer_and_szz_label(
    db_session: Session, demo_repo: dict[str, object]
) -> None:
    repo_row = Repository(name="demo", url=str(demo_repo["path"]))
    db_session.add(repo_row)
    db_session.commit()

    run = MiningRun(repository_id=repo_row.id, status="pending")
    db_session.add(run)
    db_session.commit()

    run_mining(db_session, run.id)

    db_session.refresh(run)
    assert run.status == "completed"
    assert run.checkpoint == {"last_sha": demo_repo["shas"][2], "processed": 3}

    # Ordered by id (= mining/insertion order), not timestamp: git commit
    # timestamps only have 1-second resolution, so commits made in quick
    # succession in this test can tie and make ORDER BY timestamp flaky.
    commits = list(
        db_session.scalars(
            select(Commit).where(Commit.repository_id == repo_row.id).order_by(Commit.id)
        )
    )
    assert [c.sha for c in commits] == demo_repo["shas"]
    assert commits[0].is_bug_inducing is True  # c1
    assert commits[1].is_bug_inducing is False  # c2
    assert commits[2].is_fix is True  # c3
    assert commits[2].linked_issue_refs == [1]
    assert commits[2].features["author_prior_commits"] == 2

    developer = db_session.scalar(
        select(Developer).where(any_(Developer.git_emails) == "a@example.com")
    )
    assert developer is not None
    assert commits[0].author_id == developer.id


@pytest.mark.story("US-05")
@pytest.mark.story("US-07")
def test_run_mining_is_incremental_on_rerun(
    db_session: Session, demo_repo: dict[str, object]
) -> None:
    repo_row = Repository(name="demo", url=str(demo_repo["path"]))
    db_session.add(repo_row)
    db_session.commit()

    run = MiningRun(repository_id=repo_row.id, status="pending")
    db_session.add(run)
    db_session.commit()
    run_mining(db_session, run.id)

    # Re-running the SAME (completed) run is a no-op: nothing new to mine.
    run_mining(db_session, run.id)
    all_commits = list(
        db_session.scalars(select(Commit).where(Commit.repository_id == repo_row.id))
    )
    assert len(all_commits) == 3

    # New commits get pushed...
    calc = demo_repo["path"] / "calc.py"
    calc.write_text(calc.read_text() + "\ndef mul(a, b):\n    return a * b\n")
    new_sha = commit_all(demo_repo["path"], "c4: add mul")

    # ...a fresh mining run (carrying the old checkpoint forward, as the API
    # layer does) only picks up what's new.
    next_run = MiningRun(repository_id=repo_row.id, status="pending", checkpoint=run.checkpoint)
    db_session.add(next_run)
    db_session.commit()
    run_mining(db_session, next_run.id)

    all_commits = list(
        db_session.scalars(select(Commit).where(Commit.repository_id == repo_row.id))
    )
    assert len(all_commits) == 4
    assert {c.sha for c in all_commits} == set(demo_repo["shas"]) | {new_sha}
