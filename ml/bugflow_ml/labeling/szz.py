"""US-04: SZZ labelling — for a bug-fixing commit, find the commit(s) that
last touched the lines it changes. See docs/ml.md for documented limitations.
"""


def find_bug_inducing_shas(repo_path: str, fix_sha: str) -> set[str]:
    """Blames the lines `fix_sha` deletes/changes back to whichever earlier
    commit last touched them, via PyDriller's built-in SZZ helper (which
    already ignores blank lines and Java/Python/JS-style comment lines when
    deciding which deleted lines are worth blaming)."""
    from pydriller.git import Git

    git = Git(repo_path)
    try:
        commit = git.get_commit(fix_sha)
        per_file = git.get_commits_last_modified_lines(commit)
    finally:
        git.clear()

    inducing: set[str] = set()
    for shas in per_file.values():
        inducing.update(shas)
    inducing.discard(fix_sha)
    return inducing
