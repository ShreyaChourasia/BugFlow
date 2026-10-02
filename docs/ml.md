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

## Line risk model (`bugflow_ml.models.line_risk`)

JITLine-style: tokenises each *added* line (`re.findall` on a
identifier/number/punctuation pattern — no external tokenizer dependency),
vectorises with TF-IDF, and trains a `LogisticRegression`
(`class_weight="balanced"`) to score individual lines rather than whole
commits (US-13).

**Weak labelling.** SZZ (above) only labels whole commits as bug-inducing —
there's no finer-grained ground truth about which exact line within a
bug-inducing commit actually caused the bug. Every added line in a commit
inherits that commit's `is_bug_inducing` flag. This is the standard JITLine
approach, not a shortcut unique to this project: it means a large, mostly-safe
commit that happens to be labelled bug-inducing will have all of its lines
(including the safe ones) trained as positive examples, adding label noise
that a bigger, more diverse training corpus would average out.

**Where the line text comes from.** Phase 2's `Commit` table deliberately
only stores aggregate features (churn, files_changed, ...), never raw diff
text — adding it there would bloat every mined commit with code text nothing
else needs. Phase 5 instead re-walks the exact commits it needs via
`git_miner.mine_specific_commits()` (PyDriller's `Repository(..., only_commits=shas)`)
at training and scoring time, re-cloning each repository once per call.
`ponytail:` this means training and PR-time scoring both pay a full
re-clone even for one commit; fine at demo scale (a handful of small
repos), worth caching the checkout (`mining_service._resolve_local_repo_path`
already does this for the mining phase itself) if line-risk gets used
against larger repos or scored on every push.

**Training is global**, same as commit-risk (§8/`MLModel` has no
`repository_id`) — one champion is trained across every mined repository's
added lines, not per-repo.

**Metrics**: ROC-AUC, PR-AUC, `recall_at_20pct_lines` (of all truly
bug-inducing lines in the test set, what fraction are captured by the
riskiest 20% of scored lines — the line-level analogue of commit-risk's
`recall_at_20pct_effort`), and `top_k_accuracy` (per test commit that has at
least one positive line, would its top-5 highlighted lines have surfaced
one — the metric that actually matches US-13's use case, not just a global
ranking metric). Champion promotion uses `top_k_accuracy` rather than
PR-AUC, since that's the number that reflects what a reviewer actually sees.

**Explanation (US-14)** needs no SHAP/LIME dependency, unlike the
commit-risk model's LightGBM: for a *linear* model over TF-IDF features,
`coefficient[token] * tfidf_value[token]` for each token present in a line
**is** its exact contribution to that line's score, not an approximation.
`explain_line()` returns the top-3 tokens by `|contribution|`.

**Trigger and configuration (docs/decisions/003).** Line-level analysis only
runs once a commit's calibrated commit-risk probability clears
`Repository.risk_threshold` — the same threshold Phase 4 already uses for
merge-blocking, not a second independent knob. `Repository.line_risk_top_n`
(default 5) controls how many ranked lines are shown. Below the threshold,
the PR comment and API both say line-level analysis didn't run (US-13 AC2)
rather than silently show nothing.

**Persisted result and feedback (US-15).** Each scored line becomes a
`LineRisk` row (file, line number, code, score, rank, reason) tied to the
`RiskPrediction` it came from. A reviewer marking one a false alarm sets
`LineRisk.marked_false_alarm` and writes a `Feedback` row
(`decision_type="LineRisk"`) — nothing currently *reads* that feedback back
into training; it's captured for a future phase to use as a training signal
or a per-repo precision metric.

## Severity & priority classification (`bugflow_ml.models.triage_classifier`)

§8's row for this task: "TF-IDF + linear model as baseline... Labels from a
standard taxonomy... Abstains when text is too short" → severity, priority,
confidence, with top words/features as the explanation, scored by macro-F1
and a confusion matrix.

- **Taxonomy (US-22)**: `bugflow_ml.taxonomy.Severity`/`Priority` — the one
  place these label sets are defined. Both the classifier and the backend's
  Pydantic schemas (`TriageDecisionRequest`) import from here rather than
  each keeping their own copy.
- **Two independent linear models, one shared tokenizer design**: severity
  and priority are trained and champion-tracked completely separately
  (`triage_severity`/`triage_priority` as two `MLModel` tasks) since one
  could regress while the other improves — but `_train_one_target()` is one
  generic TF-IDF-+-`LogisticRegression` trainer parameterized by which label
  to read off each example, called once per target, rather than duplicating
  the training loop twice.
- **Abstention (US-21 AC2)**: `should_abstain()` is a plain word-count check,
  applied *before* any model is even loaded — a report with too little text
  gets "please add more detail," never a low-confidence guess dressed up as
  a real suggestion.
- **Explanation (top words/features, §8)**: the same exact
  `coefficient × tfidf_value` attribution line_risk.py and duplicate
  detection's shared-phrase logic already established, just indexed into the
  *predicted class's own row* of a multi-class linear model's coefficients
  (`model.coef_[class_idx]`) instead of a binary model's single row.
- **Cold-start bootstrap data** (`generate_bootstrap_examples()`,
  `docs/decisions/005`): `DefectReport.severity`/`priority` are null until a
  human sets them — the same "no verified public dataset available" gap
  Phase 6 hit (§8 suggests Bugzilla/Eclipse/Mozilla data), compounded by this
  feature being *how* those labels get created in the first place. Training
  always prefers real human-decided labels; the keyword-correlated synthetic
  set only fills in below `MIN_TRAINING_EXAMPLES` (25), and stops being used
  automatically once enough real decisions accumulate.
- **Every suggestion gets a plain-language `Explanation` row (C3)** — `factors`
  stores both targets' confidences and top words, `text` is one sentence
  covering both, exactly like commit-risk's SHAP explanations. A `TriageAssessment`
  is only persisted (and only has one shot at being computed) the first time
  a given report's suggestion is viewed — a later request for the same report
  reads back that same row rather than re-predicting, matching how a
  cosine-distance search result wouldn't change between page loads either.

## Duplicate detection (`bugflow_ml.embeddings.duplicate_detection`)

§8's row for this task: "sentence-transformer embeddings in pgvector (HNSW
index), cosine similarity" → top-5 similar reports with scores, shared key
phrases highlighted. Implemented as:

- **Embedding**: `all-MiniLM-L6-v2` (384-dim, the dimension `DefectReport.embedding`
  was already sized for back in Phase 1) via `sentence-transformers`, loaded
  once per process and reused (loading it is the slow part — a few seconds —
  encoding itself is fast). `embed_texts()` batch-encodes, which is what
  makes the 300k-report load test (below) finish in minutes rather than
  hours: batching a single call over many texts is dramatically faster than
  the equivalent number of one-at-a-time calls.
- **Search**: a plain SQL `ORDER BY embedding <=> :query LIMIT 5` via
  pgvector's SQLAlchemy `cosine_distance()` comparator, against an HNSW
  index built with `vector_cosine_ops` (the default op class is L2 — using
  it would build an index the query planner can't use for a cosine-distance
  `ORDER BY`). `score = 1 - cosine_distance` is exactly the cosine
  similarity, since pgvector defines the distance that way regardless of
  whether the vectors are unit-normalized.
- **Shared phrases (US-18)**: `difflib.SequenceMatcher.get_matching_blocks()`
  over word-tokenized text — a stdlib-only way to find actual contiguous
  multi-word phrases common to two reports ("null pointer exception", not
  just the words "null", "pointer", "exception" separately), not a
  bag-of-words overlap. No NLP dependency needed for this.

**A real, environment-level gotcha found while wiring this in, not by unit
tests alone**: importing `sentence-transformers` (which pulls in PyTorch)
and `lightgbm` in the same process segfaults on this machine — both bundle
their own copy of the OpenMP runtime, and whichever loads second crashes.
Reproduced directly (`import sentence_transformers; ...; import lightgbm`
→ `SIGSEGV`; the reverse order doesn't crash). Since a single long-lived API
or RQ worker process (Phase 4's `SimpleWorker`) legitimately handles both
commit-risk and defect-duplicate requests, this isn't just a test-ordering
fluke — it's a real production crash risk. Fixed by `app/core/native_libs.py`,
imported first thing in every process entrypoint (`app.main`,
`app.workers.run`, `tests/conftest.py`), which imports LightGBM before
anything else gets a chance to import torch.

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
