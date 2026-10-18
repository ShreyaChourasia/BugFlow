"""US-04: SZZ labelling — for a bug-fixing commit, find the commit(s) that
last touched the lines it changes. See docs/ml.md for documented limitations.
"""

from pathlib import Path

# The git/GitHub convention for listing formatter runs and other non-semantic
# commits. `git blame -w` (which PyDriller always uses) already skips
# whitespace-only changes; this catches reformatting that isn't just
# whitespace (quote style, line wrapping) when the project lists it.
BLAME_IGNORE_FILE = ".git-blame-ignore-revs"


def find_bug_inducing_shas(repo_path: str, fix_sha: str) -> set[str]:
    """Blames the lines `fix_sha` deletes/changes back to whichever earlier
    commit last touched them, via PyDriller's built-in SZZ helper (which
    already ignores blank lines and Java/Python/JS-style comment lines when
    deciding which deleted lines are worth blaming). Commits the repo lists
    in `.git-blame-ignore-revs` are skipped by blame."""
    from pydriller.git import Git

    ignore_file = Path(repo_path) / BLAME_IGNORE_FILE
    git = Git(repo_path)
    try:
        commit = git.get_commit(fix_sha)
        per_file = git.get_commits_last_modified_lines(
            commit, hashes_to_ignore_path=str(ignore_file) if ignore_file.is_file() else None
        )
    finally:
        git.clear()

    inducing: set[str] = set()
    for shas in per_file.values():
        inducing.update(shas)
    inducing.discard(fix_sha)
    return inducing
