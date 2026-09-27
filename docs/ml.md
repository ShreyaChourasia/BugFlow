# ML notes

Living document for the `bugflow_ml` package's model choices, data pipeline
behaviour, and — importantly — their known limitations. Update this file in
the same phase that changes the behaviour it describes.

## Repository mining (`bugflow_ml.mining`)

`git_miner.mine_commits()` wraps PyDriller's `Repository.traverse_commits()`.
Traversal is chronological (oldest first); `after_sha` resumes right after a
given commit, used both for resuming an interrupted run and for picking up
newly pushed commits on a re-mine. Merge commits are recorded in the
checkpoint (so resume position stays correct) but never stored as `Commit`
rows — their diff is against the wrong parent for JIT-style features, so
literature typically excludes them.

`issue_links.py` extracts issue references (`#123`) and a fix-commit flag
(message contains fix/close/resolve, in any tense) via regex on the commit
message alone — no GitHub API calls, so mining works fully offline.

## SZZ labelling (`bugflow_ml.labeling.szz`)

`find_bug_inducing_shas()` is a thin wrapper around PyDriller's own
`Git.get_commits_last_modified_lines()`, which implements the classic SZZ
algorithm: for a fix commit, blame the lines it deletes/changes back to
whichever earlier commit last touched them.

**Known limitations** (US-04 explicitly asks these be documented, not solved):

- **Comment/whitespace filtering is per-line, on a fixed keyword list.**
  PyDriller skips blank lines and lines starting with `//`, `#`, `/*`, `'''`,
  `"""`, or `*` when deciding which of the fix commit's deleted lines are
  worth blaming. That covers Python/Java/JS-style comments; it does not
  recognize e.g. HTML (`<!-- -->`) or SQL (`--`) comment syntax, so a
  same-line "fix" that's actually just a comment tweak in those languages can
  still be attributed as a real bug fix.
- **A large reformatting/rename commit can absorb blame it doesn't deserve.**
  If a bug-fixing commit's line was last touched by a big non-semantic
  refactor (rather than the commit that actually introduced the bug), SZZ
  will name the refactor as bug-inducing. PyDriller supports excluding known
  "cosmetic" commit hashes from blame (`hashes_to_ignore_path`) to redirect
  blame further back — not used yet in Phase 2 since it requires first
  classifying which commits are purely cosmetic, which no phase does yet.
- **One fix commit can implicate more than one earlier commit** (one per
  file it touches, potentially more than one per file). `find_bug_inducing_shas`
  returns the union across files as a set — the caller doesn't currently
  weight or rank them.
- **SZZ only runs for commits classified as "fix" by `is_fix_commit()`.** A
  bug fixed without ever using a fix/close/resolve keyword in its message
  (or its own PR title, once PRs are mined) will never trigger labelling,
  silently under-labelling `is_bug_inducing`.
- **Chronological tie-breaking.** Git commit timestamps only have
  one-second resolution. Two commits made within the same second (common in
  synthetic/test repos, rare but possible in real ones) are not
  distinguishable by `timestamp` alone. `Commit.id` (insertion order, which
  follows PyDriller's true git-log order) is the reliable sort key within a
  single mining run; anything that needs a global chronological ordering
  across repositories or re-mines (e.g. the Phase 3 train/test split, C9)
  should sort by `(timestamp, id)`, not `timestamp` alone.

Manual spot-checking a mined sample against the above is the verification
step called for in §8's "Training labels" row — no automated ground-truth
dataset exists to score SZZ's precision/recall against yet.

## Commit features (`bugflow_ml.features.commit_features`)

Implements the "size, diffusion, history, fix flag" feature families from
§8's commit-risk row:

| Feature | Meaning |
|---|---|
| `lines_added` / `lines_deleted` / `churn` | size |
| `files_changed` / `directories_touched` / `subsystems_touched` / `entropy` | diffusion — `entropy` is the Shannon entropy (bits) of how evenly changed lines are spread across the commit's files; 0 for a single-file commit, higher for changes spread evenly across many files |
| `author_prior_commits` | history / author experience, scoped to the repository being mined (not the author's activity elsewhere) |
| `is_fix` | fix flag |

No LightGBM/logistic-regression model consumes these yet — that's Phase 3.
