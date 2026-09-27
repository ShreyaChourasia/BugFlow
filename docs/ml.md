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

## Commit risk model (`bugflow_ml.models.commit_risk`)

`train_commit_risk(commits, seed)` trains on every mined `Commit` in the
database (not scoped to one repository — `MLModel` has no `repository_id` in
§7, so champion models are global by design):

1. **Chronological split** (C9): sorts by `(timestamp, id)`, then slices into
   contiguous train (70%) / calibration (15%) / test (15%) windows. A
   structural assertion (and a dedicated test) checks every train timestamp
   precedes every calibration timestamp, which precedes every test
   timestamp — the actual "leakage test" the phase asks for, not just a
   comment promising it.
2. **Models**: LightGBM (`scale_pos_weight` for class imbalance,
   `deterministic=True, force_row_wise=True, n_jobs=1` for exact
   reproducibility) as the real model, logistic regression
   (`class_weight="balanced"`) as the literature-standard baseline. Both are
   evaluated and logged; only LightGBM is ever served.
3. **Calibration**: isotonic regression (`CalibratedClassifierCV` +
   `FrozenEstimator`, sklearn's post-1.6 replacement for the removed
   `cv="prefit"`), fit on the calibration slice.
4. **Metrics**: ROC-AUC, PR-AUC, F1, Brier score, and
   `recall_at_20pct_effort` — the effort-aware recall JIT defect-prediction
   literature actually uses: rank commits by predicted risk, walk down that
   ranking until 20% of total *churn* (not commit count) has been "reviewed",
   and report what fraction of real bug-inducing commits were caught by then.
5. **Fairness** (C10): `fairness_report()` compares mean predicted risk
   between contributors with < 6 months and ≥ 6 months of tenure *in this
   repository* (first mined commit → `Developer.joined_at`, not real-world
   onboarding date — see Phase 2's PROGRESS.md note). Always returns the same
   shape (`None` fields, not an absent key, when no tenure data exists), and
   is logged as part of every training run's metrics, not just computed
   ad hoc.

**Known limitations:**
- `author_prior_commits` and `entropy` are the only two features whose
  explanation sentences don't assert a direction ("more experienced is
  safer") — real training runs show LightGBM does *not* treat these two
  monotonically (see the explainability section below), so the code states
  the observed SHAP direction rather than guessing a story that might be
  backwards for a given prediction.
- No hyperparameter search — a single fixed LightGBM config. On the real
  `six` mining run used to validate this phase, the calibrated LightGBM
  champion actually scored *worse* on PR-AUC (0.11) than the logistic
  regression baseline (0.24). That's an honest result from a tiny, heavily
  imbalanced real dataset (504 commits, 98 labelled bug-inducing) — logged
  and visible in MLflow rather than hidden, exactly so a real comparison is
  possible. Revisit once mining a larger corpus (§8's ApacheJIT-scale
  evaluation) makes the comparison meaningful.

## Explanation (`bugflow_ml.explain.commit_risk_explainer`)

SHAP's exact `TreeExplainer` on the *raw* (uncalibrated) LightGBM model —
`CalibratedClassifierCV` wraps it opaquely, so SHAP needs the underlying
model directly. Deterministic by construction (no sampling), which is what
US-36 actually requires ("the same input must always give the same
explanation"), not just a nice side effect.

Top-3 factors by `|SHAP value|` become sentences via a per-feature template.
Size/diffusion features (`churn`, `files_changed`, `directories_touched`,
`subsystems_touched`, `lines_added`, `lines_deleted`, `is_fix`) use confident
"higher = riskier" phrasing, matching both JIT literature and the master
prompt's own example sentence — that relationship is reliably monotonic.
`entropy` and `author_prior_commits` instead state the SHAP direction
plainly ("...which increases/decreases the predicted risk") rather than
assume one, precisely because real runs show tree models can relate them to
risk non-monotonically (see above).

## Experiment tracking (MLflow)

Every `train_and_register_champion()` call logs a **complete experiment
record**: params (`seed`, `data_version`, split fractions, sizes, feature
columns), every metric (including `baseline_*` and `fairness_*`), both
models as artifacts (`raw_model`, `calibrated_model` — logged with
`serialization_format="pickle"`, since MLflow 3.x's default `skops` format
refuses to load LightGBM's own types without an explicit trust list, and
these models are only ever loaded by BugFlow itself), and a
`training_commits.json` artifact recording the *exact* commit ids used for
train/calib/test. `reproduce_run()` uses that artifact to refetch identical
data (verified via `data_version`, not assumed), re-run the same split with
the same seed, and check every metric matches within 0.5% (C4) —
non-reproducible runs (missing seed/data_version, or a `data_version`
mismatch because the underlying commits changed) are flagged, not silently
passed (NFR-US-07).

**MLflow version notes** (hit running this phase for real, not just in
unit tests):
- MLflow 3.x deprecated the plain filesystem (`file://`) tracking store,
  requiring either a database backend or an explicit opt-out. Tests use a
  per-test `sqlite:///{tmp_path}/mlflow.db` URI — no server needed, and it's
  MLflow's own recommended lightweight backend.
- The MLflow *server* (the docker-compose `mlflow` service, not the client
  library) added DNS-rebinding protection that, by default, rejects the
  `Host: mlflow:5000` header other containers send it — only `localhost` and
  raw private IPs are allowed by default, not Docker Compose service names.
  Fixed with `--allowed-hosts '*'`, safe here since this server is never
  reachable from outside the compose network / localhost.
