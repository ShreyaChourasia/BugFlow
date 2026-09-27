"""US-02: turn a repository's git history into plain data structures, using
PyDriller. Deliberately has no knowledge of the database — the caller (the
backend's mining service) decides how to persist what's yielded here."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModifiedFileInfo:
    path: str
    added_lines: int
    deleted_lines: int
    diff_added: list[tuple[int, str]] = field(default_factory=list)
    diff_deleted: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class MinedCommit:
    sha: str
    author_name: str
    author_email: str
    message: str
    timestamp: datetime
    lines_added: int
    lines_deleted: int
    files: list[ModifiedFileInfo]
    is_merge: bool


def mine_commits(
    repo_path: str, after_sha: str | None = None, branch: str | None = None
) -> Iterator[MinedCommit]:
    """Yields commits in chronological (oldest-first) order. `repo_path` can
    be a local path or a remote URL (PyDriller clones it to a temp dir).

    If `after_sha` is given, resumes right after that commit (exclusive) —
    used both to resume an interrupted run (US-05) and to pick up only newly
    pushed commits on a re-run (US-07).
    """
    from pydriller import Repository

    kwargs: dict[str, str] = {}
    if after_sha:
        kwargs["from_commit"] = after_sha
    if branch:
        kwargs["only_in_branch"] = branch

    skipping = after_sha is not None
    for commit in Repository(repo_path, **kwargs).traverse_commits():
        if skipping:
            if commit.hash == after_sha:
                skipping = False
            continue

        yield MinedCommit(
            sha=commit.hash,
            author_name=commit.author.name or "",
            author_email=commit.author.email or "",
            message=commit.msg,
            timestamp=commit.committer_date,
            lines_added=commit.insertions,
            lines_deleted=commit.deletions,
            files=[
                ModifiedFileInfo(
                    path=mf.new_path or mf.old_path or "",
                    added_lines=len(mf.diff_parsed.get("added", [])),
                    deleted_lines=len(mf.diff_parsed.get("deleted", [])),
                    diff_added=mf.diff_parsed.get("added", []),
                    diff_deleted=mf.diff_parsed.get("deleted", []),
                )
                for mf in commit.modified_files
            ],
            is_merge=commit.merge,
        )
