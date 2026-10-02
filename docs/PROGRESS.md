# Progress log

## Phase 7 — Severity and priority classification

**Stories:** US-21, US-22, US-23.

### What was built

- **`bugflow_ml.taxonomy`** (US-22): `Severity` (blocker/critical/major/minor/trivial)
  and `Priority` (P1–P5) — the one place these label sets are defined. The
  backend's `TriageDecisionRequest` schema imports them directly, so an
  invalid value is a 422 at the API boundary, not a silently-stored typo.
- **`bugflow_ml.models.triage_classifier`** (US-21): TF-IDF + `LogisticRegression`
  per target (severity, priority), each champion-tracked independently as
  its own `MLModel` task. Abstains (US-21 AC2) on a plain word-count check
  *before* any model loads — "add more detail," never a low-confidence guess.
  Explanation is exact `coefficient × tfidf_value` attribution indexed into
  the predicted class's own row of a multi-class linear model.
- **Cold-start bootstrap data** (`docs/decisions/005`): training always
  prefers real human-decided labels; a small keyword-correlated synthetic
  set only fills in below 25 real examples, and stops being used
  automatically once enough real decisions exist.
- **`GET /defect-reports/{id}/triage-suggestion`**: `"decided"` once a human
  has set severity/priority, otherwise the latest automated suggestion
  (computed once and persisted, not recomputed on every page view),
  `"abstained"`, or `"unavailable"` — never silently empty.
- **`POST /defect-reports/{id}/triage`** (US-23, Triager-only): one-click
  accept-as-is or change-then-save. Either way writes a `Feedback` row
  (`accepted` = whether the final values matched the last suggestion) — that
  row is what the misclassification report reads.
- **`GET /defect-reports/misclassification-report`**: which suggested
  severities/priorities get corrected most often, as a correction rate per
  suggested value.
- **C3**: every automated suggestion gets an `Explanation` row (confidences
  and top words for both targets, one plain-language sentence) — the same
  "no decision without an explanation" discipline every other model in this
  project already follows.
- **Frontend**: a "Triage" panel on the report detail page (marked
  "Automated", Accept button, editable severity/priority + "Change & save"),
  and bulk-accept checkboxes in the triage queue (fetches each undecided
  report's suggestion in parallel — bounded by the same `limit=100` the list
  itself already applies).

### Decisions

- **`docs/decisions/005-triage-bootstrap-data.md`**: same "no verified public
  dataset available" gap Phase 6 hit (§8 suggests Bugzilla/Eclipse/Mozilla
  data), compounded here by this feature being *how* real labels get created
  in the first place — see the ADR for why training always prefers real data
  and only falls back to synthetic data below a low threshold.
- **Severity and priority are two independent `MLModel` tasks**
  (`triage_severity`, `triage_priority`), not one combined model — one
  could regress while the other improves, and the existing champion/challenger
  machinery already assumes one model per task.
- **A `TriageAssessment` is computed and persisted at most once per report**
  (reused on later views) rather than recomputed every time the detail page
  loads — matches how a duplicate-search result wouldn't change between page
  loads either, and avoids bloating the table with identical rows.

### Known gaps (expected — later phases)

- Nothing currently retrains automatically as real decisions accumulate —
  `scripts/train_triage.py` has to be re-run manually. Phase 10 covers model
  lifecycle more generally.
- The misclassification report has no time filtering or component
  breakdown yet — just suggested-value correction rates.

### How to demo it

1. Train: `docker compose exec api python /app/scripts/train_triage.py`
2. Log in as `reporter@bugflow.demo`, file a report with a short description
   — its triage panel shows "add more detail" (abstained). File one with a
   fuller description — the panel shows a severity/priority suggestion,
   marked "Automated", with confidence and the words that drove it.
3. Log in as `triager@bugflow.demo`, open `/triage-queue` — undecided reports
   show their suggested values; select some and click "Accept selected" to
   bulk-accept, or open one and use "Change & save" to override it.
4. `GET /defect-reports/misclassification-report` (Triager/ML
   Engineer/Admin) shows which suggested values get corrected most often.

### Real findings from testing against the live stack, not assumed

- **Repeated the exact "stale image" mistake Phase 6 had just documented.**
  After rebuilding and redeploying `api`/`worker` and verifying the backend
  endpoints worked perfectly via `curl`, the triage queue's bulk-accept
  checkboxes were completely missing in the browser — not disabled, not
  erroring, just absent, with zero `/triage-suggestion` network requests
  ever firing. Root cause: the `frontend` container had been running for 37
  hours and was never rebuilt — only `api`/`worker` were, since those are
  what Phase 7's *backend* changes touched, but the frontend code for the
  triage panel and bulk-accept was sitting unbuilt. `docker compose build
  frontend` + redeploy fixed it immediately. Lesson generalized in
  `CLAUDE.md`: rebuild *every* service whose source changed, not just the
  ones the current phase's backend work happened to focus on — verifying
  the API with `curl` alone doesn't catch a stale frontend bundle.
- Bootstrap-trained severity/priority classifiers scored `macro_f1 = 1.0` on
  the synthetic set (expected — the keyword correlation is clean by
  construction; see `docs/decisions/005` for why this number describes the
  bootstrap set, not real-world accuracy). Live predictions on hand-written
  examples matched expectations exactly (e.g. "security vulnerability...fix
  this sprint" → critical/P2; "minor glitch...low impact" → minor/P4).
- **Live curl/browser verification left real rows in the shared dev
  Postgres, which then broke the backend test suite** — 3 tests failed with
  wrong counts or `MultipleResultsFound` because they assumed a clean
  `defect_reports`/`feedback`/`explanations` table. Fixed by adding the same
  TRUNCATE-based isolation already used elsewhere (Phase 5/6) to every
  defect/triage test fixture that didn't have it yet.
- **That same live verification also broke two *unrelated* Phase 5 tests**:
  `DELETE FROM repositories` (a cleanup step in two line-risk fixtures
  predating Phase 6) started failing on `defect_reports_repository_id_fkey`
  the moment a real `DefectReport` referenced a repository those fixtures
  were trying to delete. Switched both to `TRUNCATE ... CASCADE`, which
  needs no table list to maintain as later phases add more FKs — see
  `CLAUDE.md`'s gotchas.

## Phase 6 — Defect reports and duplicate detection

**Stories:** US-16, US-17, US-18, US-19, US-20, NFR-US-02.

### What was built

- **`ml/bugflow_ml/embeddings/duplicate_detection.py`**: `all-MiniLM-L6-v2`
  sentence embeddings (384-dim, matching the dimension `DefectReport.embedding`
  was already sized for back in Phase 1), batch-encoded via `embed_texts()`.
  `shared_phrases()` finds actual contiguous multi-word phrases common to two
  reports via `difflib.SequenceMatcher` on word-tokenized text — stdlib only,
  no NLP dependency.
- **HNSW index** (`ix_defect_reports_embedding_hnsw`, `vector_cosine_ops`) via
  a hand-written migration — pgvector's default op class is L2, which the
  cosine-distance queries here can't use.
- **`app/services/defect_service.py`**: create-and-embed a report; search
  candidates via `ORDER BY embedding <=> :query LIMIT k`; live suggestions
  (US-16, gated on a minimum word count); duplicates-with-shared-phrases for
  an existing report (US-17/18); merge + async notify (US-20); index-rebuild
  status tracked in `SystemConfig` (C2 — `"rebuilding"` is a distinct,
  explicit state from an empty result list, never silently confused).
- **`POST /defect-reports/index/rebuild`**: `REINDEX INDEX CONCURRENTLY` in a
  background job, for after a bulk load.
- **US-20 "notify asynchronously"**: no email/SMTP integration exists
  anywhere in this project, so notification is an `AuditLog` entry
  (`duplicate_merge_notification`) written by an RQ job — same
  "offline-by-default, a real channel is a config choice" pattern Phase 4
  used for GitHub. What the story actually requires — this runs in the
  background, not inline in the merge request — is real either way.
- **Frontend**: `/defect-reports/new` (debounced live suggestions in a side
  panel that never touches the form's own text state, so opening a
  suggestion in a new tab can't lose what's typed), a report detail page
  with ranked candidates and `<mark>`-highlighted shared phrases, a merge
  button for triagers, and `/triage-queue`.
- **`scripts/load_test_defects.py`** (C2/US-19): bulk-loads synthetic defect
  reports and measures p50/p95/p99 search latency against the live index —
  see the real numbers below.

### Decisions

- **`docs/decisions/004-synthetic-load-test-data.md`**: the ≥300k-report load
  test uses generated synthetic text, not a real public dataset — there's no
  verified, licensed 300k+ real bug-report corpus available here, and what
  C2/US-19 actually tests (does pgvector/HNSW search latency hold up at
  scale) doesn't depend on the text being real.
- **`GET /repositories` list is now also readable by Reporter/Triager**, not
  just Admin/ML-Engineer — filing a report means picking which repository
  it's against. Only the list endpoint; single-repo GET and all mutations
  stay Admin/ML-Engineer-only.

### Known gaps (expected — later phases)

- Nothing reads `Feedback`/merge history back into anything yet — same gap
  as Phase 5's line-risk false-alarm feedback.
- Severity/priority classification is Phase 7.

### How to demo it

1. Log in as `reporter@bugflow.demo`, open `/defect-reports/new`, pick a
   repository, and start typing a description — similar existing reports
   appear in the side panel after a short pause.
2. Submit — the report's detail page shows ranked candidate duplicates with
   shared phrases highlighted.
3. Log in as `triager@bugflow.demo`, open `/triage-queue`, open a report,
   and merge it into a candidate.
4. `docker compose exec api python /app/scripts/load_test_defects.py --count 300000`
   to see the search stay fast at real scale (numbers below).

### Real findings from testing against the live stack, not assumed

- **LightGBM + PyTorch segfault if torch loads first in the same process.**
  Both bundle their own OpenMP runtime. Reproduced directly:
  `import sentence_transformers; ...; import lightgbm` → `SIGSEGV`; the
  reverse order doesn't crash. First surfaced as the backend test suite
  crashing partway through (alphabetical test collection loads the defect
  tests, which import sentence-transformers, before the PR-scoring tests,
  which train a LightGBM model) — not something any single test would ever
  catch in isolation. A real production risk too: a single long-lived API or
  RQ worker process legitimately handles both commit-risk and
  defect-duplicate requests. Fixed by `app/core/native_libs.py`, imported
  first in every process entrypoint (`app.main`, `app.workers.run`,
  `tests/conftest.py`), which imports LightGBM before anything else gets a
  chance to import torch. See `docs/ml.md`.
- **`pip install torch` on Linux defaults to the CUDA build** even in a
  CPU-only container, pulling several GB of `nvidia-*` packages the
  container never uses and dramatically slowing the Docker build.
  `backend/Dockerfile` now passes
  `--extra-index-url https://download.pytorch.org/whl/cpu`.
- **`GET /defect-reports` had no pagination at all.** Fine with a handful of
  demo rows; against the ~36k rows the load test had inserted so far, it
  returned a 16.6MB response in 11.3s. Fixed with `status`/`limit`/`offset`
  query params (default `limit=100`, capped at 500) — found by loading the
  actual `/triage-queue` page in a browser mid-load-test and watching it
  never resolve, not by a unit test.
- **A stale Docker image cost real debugging time.** After fixing the
  pagination bug above, the *already-running* API container kept crashing on
  the same request — the fix had been edited on disk but never rebuilt into
  the image. Confirmed by `docker compose exec api grep` showing the old
  function signature still running. Lesson written down here so it isn't
  repeated: after any backend/ml code change meant to fix a *running*
  container's behavior, rebuild and redeploy before re-testing, not just
  re-test.
- **A full-table `DELETE` against a large table with an HNSW index attached
  is very slow** — a test's cleanup `DELETE FROM defect_reports` against the
  300k-row post-load-test table took over 10 minutes (each deleted row
  updates the HNSW graph) and had to be cancelled. Switched that cleanup to
  `TRUNCATE ... CASCADE`, which resets the table instantly regardless of row
  count and is still transactional/rollback-safe in Postgres.

### Performance — measured against the real stack, not assumed (C2/US-19)

`scripts/load_test_defects.py --count 300000` against the live pgvector
HNSW index (single Docker Desktop VM, no dedicated benchmarking hardware):

| Metric | Value |
|---|---|
| Reports loaded | 300,000 (synthetic — see ADR 004) |
| Load time | 1389.8s (~23.2 min), ~216 rows/sec sustained |
| Search p50 | 1.18ms |
| Search p95 | **1.57ms** |
| Search p99 | 2.49ms |

500 search queries, each a real `ORDER BY embedding <=> :query LIMIT 5`
against the full 300k-row HNSW index. The index does exactly what it's for:
sub-2ms p95 lookup latency regardless of table size, which a sequential
brute-force cosine-distance scan over 300k rows could not deliver. Load
throughput (not the story's concern, but honestly reported) holds roughly
steady across the whole run rather than degrading, suggesting HNSW insert
cost on this table doesn't blow up as the graph grows, at least up to 300k
nodes on this hardware.

## Phase 5 — Line-level risk and the review queue

**Stories:** US-09, US-13, US-14, US-15.

### What was built

- **`ml/bugflow_ml/models/line_risk.py`**: JITLine-style line-level model —
  tokenises added lines, TF-IDF + `LogisticRegression`, chronological
  train/test split (C9) with the same leakage assertion as commit-risk.
  Weak labels: every added line inherits its commit's `is_bug_inducing` flag
  (documented limitation, not a shortcut — see `docs/ml.md`). Metrics:
  ROC-AUC, PR-AUC, `recall_at_20pct_lines`, and `top_k_accuracy` (the one
  that actually matches US-13's "would the top-N have surfaced it" use
  case). Explanation (US-14) is exact `coefficient × tfidf_value` per token
  — no SHAP/LIME dependency needed for a linear model.
- **`git_miner.mine_specific_commits()`**: re-walks just the given commits
  via PyDriller's `only_commits` filter to recover added-line text, which
  Phase 2 deliberately never persists on `Commit`.
- **`app/services/line_risk_training_service.py`**: trains a global
  champion (same "no `repository_id` on `MLModel`" design as commit-risk)
  across every mined repository's added lines, logs to MLflow as a single
  `Pipeline(tfidf, clf)` artifact, promotes on `top_k_accuracy`.
- **`app/services/line_prediction_service.py`**: loads the cached champion
  pipeline, scores every added line in a diff, returns the top-N
  (`Repository.line_risk_top_n`) ranked with a token-level reason each.
- **`pr_scoring_service.py`** now runs line-level analysis whenever a
  commit's calibrated risk clears `Repository.risk_threshold` (docs/decisions/003):
  persists `LineRisk` rows, adds a "Riskiest lines" section to the PR
  comment, and reports `not_applicable` (US-13 AC2's exact wording),
  `unavailable` (no line-risk model yet, or its diff couldn't be recovered),
  or `available` — without ever turning a missing line-risk model into a
  failed commit-risk assessment.
- **`POST /line-risks/{id}/false-alarm`** (US-15): marks the line and writes
  a `Feedback` row (`decision_type="LineRisk"`).
- **`GET /review-queue`** (US-09): every open PR across all repositories,
  sorted by calibrated risk, with change size (lines added/deleted, files
  changed) — global, matching the Code Reviewer role's "view all PRs"
  permission.
- **Frontend**: a diff viewer on the PR detail page (grouped by file,
  expandable lines showing the token-level reason and a false-alarm
  button), and a new `/review-queue` page.
- **Migration**: `Repository.line_risk_top_n` (default 5).

### Decisions

- **`docs/decisions/003-line-risk-top-n.md`**: reuses the existing
  `risk_threshold` as the "is this high risk" gate for line-level analysis
  rather than adding a second, redundant threshold; `line_risk_top_n` is the
  one genuinely new per-repo setting.
- **Line text is recovered on demand, not persisted on `Commit`.** Rather
  than retroactively changing Phase 2's schema to store raw diff lines
  (bloating every mined commit with text nothing else needs), both training
  and PR-time scoring call `mine_specific_commits()` to re-walk just the
  commits they need. `ponytail:` this means a fresh git clone per call —
  fine at this project's demo scale, worth caching the checkout
  (`mining_service` already has this pattern) if line-risk runs against
  larger repos or scores on every push.
- **Champion promotion uses `top_k_accuracy`, not PR-AUC** — unlike
  commit-risk. It's the metric that reflects what a reviewer actually sees
  (would the top-N lines shown have included a real one), not just a global
  ranking metric.

### Known gaps (expected — later phases)

- Nothing currently reads `Feedback` rows back into training — a false-alarm
  mark is captured but doesn't yet affect future line-risk training or a
  per-repo precision metric.
- The line-risk model has never been evaluated against a second, larger real
  repository — same single-small-real-dataset caveat Phase 3's commit-risk
  model already carries (see below).

### How to demo it

1. Mine a repo with real history (Phase 2) and train both champions:
   ```bash
   docker compose exec api python /app/scripts/train.py
   docker compose exec api python /app/scripts/train_line_risk.py
   ```
2. Lower the repo's `risk_threshold` if needed so a commit clears it, then
   replay a PR event (Phase 4):
   ```bash
   docker compose exec api python /app/scripts/replay_pr_events.py \
     --repository-id <id> --sha <a mined sha> --pr-number 101
   ```
3. Open `/repositories/<id>/pull-requests/101` — see "Riskiest lines" with
   expandable rows and a false-alarm button, or (below threshold) "did not
   meet the threshold for line-level analysis."
4. Open `/review-queue` — every open PR across every repo, riskiest first.

### Performance and honest results — measured against the real stack, not assumed

- **Line-risk training on the real `six` repo** (504 mined commits, run
  inside the live `api` container): took ~10 seconds end-to-end (one clone
  + walk + TF-IDF fit + LightGBM-free linear fit), `top_k_accuracy=1.0`,
  `pr_auc=0.31`, `roc_auc=0.64`, `recall_at_20pct_lines=0.46`. Reported
  honestly rather than cherry-picked: on a real end-to-end check, several of
  the top-ranked "risky" lines turned out to be `CHANGES`-file separator
  lines (`-----------------`) rather than meaningful code — the TF-IDF
  vectorizer treats punctuation-only tokens like any other, and with only
  504 commits' worth of added lines to learn from, a handful of
  coincidentally-labelled separator lines were enough to make `-` look
  predictive. Same class of honest limitation as Phase 3's commit-risk
  model scoring worse than its own baseline on this same small, imbalanced
  real dataset — a bigger, more diverse training corpus is what actually
  fixes this, not a different algorithm.
- **`/review-queue` N+1 query bug, found by live browser testing, not unit
  tests.** The first implementation looked up each PR's commit and
  prediction with two separate queries per row. Against ~900 real open PRs
  left over in the dev database from Phase 4's k6 perf testing, the page
  took **6.6 seconds** to load — invisible to unit tests (which use a
  handful of PRs) and only surfaced by loading the actual page in a browser
  and watching the request sit "pending." Rewritten to batch-fetch commits
  and predictions in two queries total regardless of PR count: **0.115
  seconds**, a ~57x improvement.

## Phase 4 — GitHub integration: results on the pull request (end of Release 1)

**Stories:** US-08, US-11, US-46, US-47, NFR-US-01, NFR-US-04, NFR-US-05.

### What was built

- `docs/github-app.md`: a full GitHub App setup guide with least-privilege
  permissions (Checks + Issues read/write, Contents/PRs/Metadata read-only)
  and the exact webhook/permission/event configuration.
- **Webhook endpoint** (`POST /webhooks/github`): verifies
  `X-Hub-Signature-256` (constant-time HMAC compare), handles
  `pull_request` `opened`/`synchronize`, creates/updates the `PullRequest`
  row, posts a `pending` check, and enqueues scoring — all before
  responding, matching §5's flow exactly.
- **`app/integrations/github/`**: a `GitHubClient` protocol with two
  implementations — `RealGitHubClient` (JWT App auth → installation token →
  Checks/Issues REST calls) and `FakeGitHubClient` (the offline "Checks
  sink" — never makes a network call; whatever it "posts" only ever lives in
  `PullRequest`'s own state columns, which is what the UI reads either way).
  `get_github_client()` picks automatically based on whether the App is
  configured *and* this specific repo has an installation id.
- **`pr_scoring_service.py`**: scores a PR's head commit by reusing Phase
  3's `predict_commit` completely unchanged — a PR is just "score this one
  commit." Fetches the commit live via GitHub's REST API if it hasn't been
  mined yet; creates a real `Commit` row for it either way, so re-scoring or
  later mining sees the same data.
- **Failure handling**, all verified with tests: a hard job timeout leaves
  the check `pending` (C1 — nothing runs after RQ kills a timed-out job, so
  there's nothing to leave in a stale state); a caught exception explicitly
  marks the check `completed`/`neutral`/"unavailable" (C7 — never blocks);
  no trained model marks `completed`/`neutral`/"no assessment available"
  (US-08 AC2).
- **Merge blocking** (US-47): `Repository.merge_blocking_enabled`
  (already existed, off by default since Phase 1) now actually gates the
  check's conclusion — `neutral` always when off; `success`/`failure`
  against `risk_threshold` when on. Verified both ways against the real
  running stack, not just unit tests.
- **Offline demo**: `scripts/replay_pr_events.py` builds and HMAC-signs a
  real webhook payload for an already-mined commit and posts it to the real
  endpoint — the only thing "fake" is the GitHub on the other end of the
  write-back.
- **Frontend**: PR list (`/repositories/[id]/pull-requests`) and detail
  pages showing check status/conclusion, score, confidence, factors, the
  explanation, and the exact comment body that was (or would have been)
  posted.
- **Performance**: `scripts/perf/webhook_p95.js` (k6), measuring wall-clock
  time from webhook receipt to the check resolving — see numbers below.

### Decisions

- **New `PullRequest` columns** (`check_status`, `check_conclusion`,
  `check_summary`, `comment_body`) — ADR
  `docs/decisions/002-pr-check-state-columns.md`. §7 only had opaque IDs
  (`check_run_id`, `comment_id`); these four are what both the offline fake
  sink and the frontend need to show *what* was posted, not just its ID.
- **A check run is created fresh on every `opened`/`synchronize`**, never
  reused — GitHub check runs are permanently tied to one `head_sha`, so a
  new push needs a new one. The **comment** is what persists and gets
  edited (US-46) via `PullRequest.comment_id`.
- **No SZZ/mining machinery for a live PR's head commit** — historical
  bug-inducing labelling doesn't apply to an unmerged commit. Fetching a
  fresh commit's diff uses GitHub's Commits REST API directly (fast, no
  clone needed), not the Phase 2 git-mining path.
- **`mlflow.sklearn.log_model(..., artifact_path=...)`, not MLflow 3.x's
  new `name=`.** `name=` logs a separate "Logged Model" entity at
  `models:/<model_id>`, not a plain run artifact — `prediction_service`
  loads champions via `runs:/<run_id>/<path>`, which needs the classic
  (deprecated-but-functional) `artifact_path` behavior. Retrained Phase 3's
  demo champion under the fix; the old one's artifacts are unrecoverable
  (archived, not deleted, for the record).
- **`mlflow_data` volume now also mounted into `api`/`worker`, not just
  `mlflow`.** MLflow's local artifact store writes the artifact_uri as a
  bare filesystem path, not a proxied `mlflow-artifacts:/` URI — any client
  loading that artifact reads the path directly off disk, so it needs the
  exact same mount the server writes to, not just network access to it.
- **RQ worker switched from `Worker` to `SimpleWorker`** — see the
  performance section below; this is the single most impactful fix this
  phase.

### Known gaps (expected — later phases)

- `PullRequest.author_id` is only ever backfilled from a resolved `Commit`'s
  author (git email), never from the webhook's GitHub username — there's no
  reliable username→git-email mapping available without additional GitHub
  API calls this phase doesn't make.
- No real GitHub App has actually been registered against this codebase —
  `RealGitHubClient` is tested against a mocked HTTP transport (request
  construction, auth flow, response parsing all verified), but the true
  end-to-end "open a PR on real GitHub" path has only been exercised via
  `docs/github-app.md`'s instructions, not run live. The offline replay path
  *has* been run live, repeatedly, against the real running stack.
- Line-level risk highlighting, review queue, and the false-alarm button are
  Phase 5.

### How to demo it

1. Register a repo with real bug-fix history and mine it (Phase 2), e.g.
   `https://github.com/benjaminp/six`
2. Train a champion (Phase 3): `docker compose exec api python /app/scripts/train.py`
3. Set a `GITHUB_WEBHOOK_SECRET` in `.env` (any random string — it only
   needs to match between whoever signs the webhook and this endpoint;
   see `docs/github-app.md` for the real-GitHub version)
4. Replay a PR event for an already-mined commit:
   ```bash
   docker compose exec api python /app/scripts/replay_pr_events.py \
     --repository-id <id> --sha <a mined sha> --pr-number 101
   ```
5. Open `/repositories/<id>/pull-requests` in the UI — see the check
   resolve (pending → neutral/success/failure) and the full risk breakdown
6. Toggle `merge_blocking_enabled` on the repo (`PATCH /repositories/{id}`)
   and replay again against a high-risk commit — watch the conclusion
   change from `neutral` to `failure`

### Performance — measured honestly, not assumed (C1)

Ran `scripts/perf/webhook_p95.js` against the real running stack (not a
synthetic estimate), replaying PR events for an already-mined `six` commit:

| Configuration | p95 (webhook receipt → check resolved) | Success rate |
|---|---|---|
| 1 VU, **before** fixing the worker's model caching | 2.55s | 100% |
| 5 VUs, before the fix | **10.07s** (hit the 10s poll timeout) | 65.5% |
| 1 VU, **after** the fix | **126ms** | 100% |
| 5 VUs, after the fix | **188ms** | 100% |

**What happened, and why it's reported this way, not hidden:** Phase 3's
`prediction_service` caches the loaded champion model at module level so
it's only loaded once (US-10). That caching silently never worked once
wired into the real RQ worker: RQ's default `Worker` **forks a new child
process for every job**, so the "module-level cache" started fresh every
single time — each PR-scoring job was paying the full ~2s MLflow
artifact-load cost from scratch, and under concurrent load those ~2s jobs
queued up behind a single worker fast enough to blow past a 10-second
timeout. This was invisible to every unit test (they call `predict_commit`
directly, in-process, never through RQ at all) and only showed up when
actually running the k6 script against the live stack. Switched the worker
to RQ's `SimpleWorker` (no forking) — a one-line fix that dropped p95 by
~20x and fixed the failures under concurrency, because the cache now
actually does what US-10 asked for.

The remaining ~110-125ms per request is almost entirely the HTTP round
trips themselves (webhook POST + polling GETs, each a few ms) plus one
real inference + SHAP explanation call — not network-fetch-dominated the
way the master prompt anticipated might happen, because this phase's
"fetch the diff" is a single GitHub REST call (or, in the offline/replay
case, already-mined data), not a full git clone.

## Phase 3 — Commit risk model, calibration and explanation

**Stories:** US-08 (model part), US-10, US-12, US-34, US-36, US-42, US-44, NFR-US-07, NFR-US-11.

### What was built

- `bugflow_ml.models.commit_risk`: the full training pipeline — chronological
  split with an explicit leakage test (C9), LightGBM + logistic-regression
  baseline with class weighting, isotonic calibration, effort-aware
  `recall_at_20pct_effort`, and a per-tenure fairness report (C10). Pure
  functions, no MLflow/DB access — independently unit-tested with synthetic
  data.
- `bugflow_ml.explain.commit_risk_explainer`: SHAP `TreeExplainer` on the raw
  LightGBM model, top-3 factors turned into plain sentences. Deterministic by
  construction (US-36) — no sampling-based SHAP variant used.
- `app/services/training_service.py`: loads every mined commit (global,
  cross-repository — `MLModel` has no `repository_id` in §7), trains, logs a
  complete experiment to MLflow (params, seed, data_version, all metrics,
  both models as artifacts, and the exact commit-id snapshot used), and
  promotes the new model to `champion` if it beats the current one on PR-AUC
  (US-08, US-44, NFR-US-11).
- `reproduce_run()`: refetches the *exact* commits a run used (via the logged
  artifact), verifies `data_version` still matches (flags drift instead of
  silently comparing against changed data), retrains with the same seed, and
  checks every metric is within 0.5% (C4). Flags a run missing a seed or
  data_version as non-reproducible rather than mis-scoring it (NFR-US-07).
- `app/services/prediction_service.py`: loads the champion once, re-loading
  only when the champion actually changes (US-10); `POST /predict/commit`
  and `GET /commits/{sha}/risk` (US-12) always persist/return a paired
  `Explanation` (C3) — enforced with an assertion, not just convention.
- `scripts/train.py`, `scripts/reproduce_run.py` (exit-code-aware, for
  `make reproduce`), `scripts/export_experiment.py` (US-44).
- **Tests:** the leakage test, a reproducibility test (genuine MLflow
  round-trip via a local SQLite tracking store, not mocked), an
  explanation-always-present test (C3), and a scoring-latency
  micro-benchmark (US-10) — distinct from Phase 4's real end-to-end p95 test.

### Decisions

- **`CalibratedClassifierCV(FrozenEstimator(model))`, not
  `cv="prefit"`.** The `prefit` string value was removed in scikit-learn
  1.6+; `FrozenEstimator` is the direct replacement. Pinned
  `scikit-learn>=1.6` accordingly.
- **MLflow model artifacts logged as `pickle`, not the new default
  `skops`.** `skops` refuses to load LightGBM's own types without an
  explicit trust list; since these models are only ever loaded by BugFlow
  itself (never a third party), plain pickle is simpler and equally safe
  here.
- **MLflow's filesystem tracking store is deprecated in 3.x** — tests use
  `sqlite:///{tmp_path}/mlflow.db` instead (no server needed, MLflow's own
  recommended lightweight backend).
- **The docker-compose `mlflow` service needs `--allowed-hosts '*'`.**
  MLflow 3.x's server rejects the `Host: mlflow:5000` header other
  containers send by default (DNS-rebinding protection scoped to localhost
  + raw private IPs, not Compose service names). Safe here since this
  server is never internet-facing.
- **`author_prior_commits` and `entropy` explanations state the SHAP
  direction rather than assert a story.** A real training run showed
  LightGBM relates these two to risk non-monotonically; the other,
  reliably-monotonic size/diffusion features keep the confident "higher =
  riskier" phrasing that matches the master prompt's own example.

### Known gaps (expected — later phases)

- No hyperparameter tuning; a real run on `six` (504 commits) shows the
  calibrated LightGBM champion underperforming the logistic-regression
  baseline on PR-AUC — an honest, visible result at this data scale, not
  swept under the rug. Revisit with more data / tuning if it still matters
  once real evaluation-scale datasets (§8) are in place.
- SZZ's `hashes_to_ignore_path` (excluding known-cosmetic commits from
  blame) still isn't used — same gap noted in Phase 2.
- No PR-level scoring yet (`RiskPrediction.pr_id` unused) — that's Phase 4.
- Each test that touches MLflow creates its own fresh SQLite tracking store,
  which is correctly isolated but re-runs MLflow's ~70-migration schema setup
  every time (a few seconds each). Fine at this suite's size; would need a
  shared fixture if the MLflow-touching test count grows much further.

### How to demo it

1. Mine a repository with real bug-fix history (Phase 2) — a tiny repo like
   `octocat/Hello-World` won't have the ≥20 commits training needs; something
   like `https://github.com/benjaminp/six` (branch `main`, ~500 commits)
   mines in well under a minute and has plenty of real fix commits
2. `docker compose exec api python /app/scripts/train.py` — prints the run
   id, whether it was promoted to champion, and every metric
3. `docker compose exec api python /app/scripts/reproduce_run.py --run-id <id>`
   — confirms it reproduces within 0.5%
4. `docker compose exec api python /app/scripts/export_experiment.py --run-id <id> --out record.json`
5. Log in, then:
   ```
   curl -X POST http://localhost:8000/predict/commit -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" -d '{"repository_id": <id>, "sha": "<a mined sha>"}'
   ```
   — returns probability, calibrated probability, confidence, risk level,
   and a plain-language explanation with its top-3 factors
6. `curl "http://localhost:8000/commits/<sha>/risk?repository_id=<id>"` —
   returns the same prediction read back

### Plain-English notes

- **Why train on every repository at once instead of one model per repo?**
  The data model's `MLModel` table has no `repository_id` — there's one
  global champion. This also means a repo with too little history of its own
  still benefits once other repos have been mined.
- **Why calibrate at all?** A raw model's "70% risk" often doesn't mean "7 in
  10 similar changes had bugs" — it's just a score that happens to rank
  things in a reasonable order. Calibration adjusts those raw scores so the
  probability number is actually meaningful on its own, which is what lets
  the UI honestly show "70% risk" as a real frequency rather than an
  arbitrary score.
- **Why does the explanation sometimes say a feature "increases" risk in a
  way that seems backwards (e.g. more prior commits raising risk)?** Tree
  models can learn genuinely non-monotonic relationships from real data —
  the explanation reports what the model actually did for this specific
  prediction, not a simplified assumption about what "should" be true.

## Phase 2 — Repository mining and training corpus

**Stories:** US-02 to US-07.

### What was built

- `bugflow_ml.mining.git_miner`: wraps PyDriller's `Repository.traverse_commits()`
  into a plain `MinedCommit` dataclass stream — chronological, resumable via
  `after_sha`, works against a local path or a remote URL (PyDriller clones
  it). Merge commits advance the checkpoint but are never stored as `Commit`
  rows (their diff is against the wrong parent for JIT-style features).
- `bugflow_ml.mining.issue_links`: regex-based issue reference extraction
  (`#123`) and a fix-commit flag (message contains fix/close/resolve, any
  tense) — no GitHub API calls, so mining works fully offline (US-03).
- `bugflow_ml.mining.backoff`: exponential-backoff retry wrapper for
  transient clone/network failures.
- `bugflow_ml.labeling.szz`: thin wrapper around PyDriller's own
  `Git.get_commits_last_modified_lines` (a ready-made SZZ implementation —
  see `docs/ml.md` for what it does and does not handle) (US-04).
- `bugflow_ml.features.commit_features`: size/diffusion/history/fix-flag
  features per §8 — churn, files/directories/subsystems touched, an entropy
  measure of how evenly a commit's changes are spread across files, and the
  author's prior commit count in the repository (US-06).
- `app/services/mining_service.py`: the orchestration that ties the above
  into the database — resolves/creates `Developer` rows from commit author
  emails, computes features, stores `Commit` rows, runs SZZ labelling for
  this run's fix commits, and updates `MiningRun.checkpoint` after every
  commit in the same transaction as that commit — so a crash mid-run loses
  at most one commit's work and always resumes from a consistent point
  (US-05). Re-running with the checkpoint carried forward is how incremental
  ingestion works (US-07) — "start mining" and "resume" are the same
  operation under the hood, just seeded from a different starting point.
- API: `POST/GET /repositories/{id}/mining-runs`,
  `POST /repositories/{id}/mining-runs/{run_id}/resume` (409 unless the run
  is `failed`), all Admin/ML-Engineer-only. Jobs run on the existing RQ
  worker (`app/workers/jobs/mining.py`).
- Schema: `Commit.linked_issue_refs` (new column) and a
  `(repository_id, sha)` uniqueness constraint — see
  `docs/decisions/001-issue-refs-on-commit.md`.
- **Frontend:** a repository detail page (`/repositories/[id]`) showing
  mining runs with status/progress/error, a "Start mining" button, and a
  "Resume" button on failed runs.
- **Tests:** SZZ and feature-computation tests on tiny synthetic git repos
  built inside the test with known answers (both in `ml/tests` as pure-logic
  unit tests, and in `backend/tests/test_mining_service.py` as an
  integration test through the real database); checkpoint/resume and
  incremental-ingestion tests; RBAC on the mining endpoints.

### Decisions

- **No `Issue` table** — issue references are stored as a plain integer
  array on `Commit` (ADR 001). Nothing yet needs issue titles/state/history;
  add a real table when something does.
- **Whole-history-in-memory retry.** `mining_service.run_mining` materializes
  the full commit list before processing, inside the backoff-retry wrapper,
  so a transient clone failure retries cleanly from scratch. Fine at the
  "one or two demo repos" scale this phase targets — flagged in the code as
  a `ponytail:` comment for whoever mines something bigger later.
- **SZZ only runs for fix commits found *by this run*.** A re-mine (US-07)
  doesn't re-scan already-labelled history, keeping incremental runs cheap.
  Documented as a limitation in `docs/ml.md`: a bug fixed without ever using
  a fix/close/resolve keyword is never labelled.
- **`git` had to be added to the backend Docker image.** PyDriller shells
  out to the `git` CLI; `python:3.11-slim` doesn't include it. Caught by
  actually running a mining job in the container, not just unit tests (which
  ran fine locally where `git` was already on `PATH`).
- **Chronological sort within a mining run uses `Commit.id`, not
  `timestamp`.** Git commit timestamps only have one-second resolution, so
  commits made in quick succession (routine in a synthetic test repo, rare
  but possible for real ones) can tie. Insertion order is the reliable
  ordering within one run. This will matter again for Phase 3's chronological
  train/test split (C9) — noted in `docs/ml.md`.

### Known gaps (expected — later phases)

- No model consumes `Commit.features` yet (Phase 3).
- SZZ doesn't yet use PyDriller's `hashes_to_ignore_path` to exclude known
  "cosmetic" (pure reformatting) commits from blame — would need a way to
  classify commits as cosmetic first, which no phase does yet.
- `PullRequest`/`RiskPrediction`/`LineRisk` tables still unused (Phase 3-5).
- Mining runs sequentially inside one RQ job; a very large repository's
  first (non-incremental) mine will take a while with no progress feedback
  finer than "N commits processed so far."

### How to demo it

1. Log in as `admin@bugflow.demo` (or `ml-engineer@bugflow.demo`)
2. Register a repository — a real small public repo works well for a demo,
   e.g. `https://github.com/octocat/Hello-World` (default branch `master`)
3. Open its detail page (`/repositories/{id}`), click "Start mining"
4. Refresh — the run shows `completed` with a commit count once the worker
   picks it up (`docker compose logs worker -f` to watch it happen live)
5. Click "Start mining" again — completes immediately, same commit count
   (nothing new to mine): that's incremental ingestion (US-07) working
6. `docker exec bugflow-postgres-1 psql -U bugflow -d bugflow -c "select sha, is_fix, is_bug_inducing, linked_issue_refs from commits;"`
   to see the mined/labelled data directly

### Plain-English notes

- **Why does "Start mining" sometimes finish instantly?** Because it's not
  starting from scratch — it picks up from wherever the last run for that
  repository left off (its "checkpoint"). If nothing's been pushed since the
  last run, there's nothing new to process.
- **What is SZZ, in one sentence?** For a commit that fixes a bug, look at
  exactly which earlier commit last touched the lines being fixed — that
  earlier commit is blamed for introducing the bug.
- **Why is a merge commit skipped?** A merge commit's "diff" is really a
  combination of two branches' history, not a single coherent change — the
  size/diffusion features this phase computes wouldn't mean the same thing
  for it as they do for a normal commit.

## Phase 1 — Core data, auth and roles

**Stories:** US-01, US-49.

### What was built

- SQLAlchemy 2.0 models (`backend/app/models/`) for all 19 entities in §7 —
  `User`, `Developer`, `Repository`, `MiningRun`, `Commit`, `PullRequest`,
  `RiskPrediction`, `LineRisk`, `DefectReport`, `TriageAssessment`,
  `ResolverRecommendation`, `Assignment`, `ResolutionForecast`, `Explanation`,
  `Feedback`, `MLModel`, `DriftAlert`, `AuditLog`, `SystemConfig` — with only
  foreign keys, no ORM `relationship()`s yet (added when a later phase actually
  traverses them). One Alembic migration (`alembic/versions/e8dbf3..._initial_schema.py`)
  creates all of them, including enabling the `pgvector` extension for
  `DefectReport.embedding`.
- JWT auth: `POST /auth/login` (OAuth2 password form, so the FastAPI docs'
  "Authorize" button works), `POST /auth/refresh`, `POST /auth/logout`
  (revokes the access token's `jti` in Redis until it would've expired anyway),
  `GET /auth/me`. Passwords hashed with `bcrypt` directly (not `passlib` — see
  Decisions).
- `require_role(*roles)` FastAPI dependency (`app/core/security.py`), applied
  to every admin-only route.
- `scripts/seed_demo.py`: one demo user per role (§2), password `bugflow-demo`,
  safe to re-run.
- Repository CRUD (`app/api/repositories.py`): list/get open to Admin + ML
  Engineer, create/update/delete Admin-only.
- `SystemConfig` key/value store and `AuditLog` (`app/api/admin.py`),
  Admin-only; every admin write (`create`/`update`/`delete` on Repository or
  User, `upsert` on SystemConfig) writes an audit log entry
  (`app/services/audit.py`).
- **Frontend:** `/login` page, a role-aware nav (`app/nav.tsx`) that shows
  Repositories / Admin·Users links only to the roles that have them,
  `/repositories` (list + create + merge-blocking toggle, gated to Admin in
  the UI) and `/admin/users` (list + create + activate/deactivate).
- **Tests:** auth flow (login/refresh/logout/me), RBAC 403s (non-admin blocked
  from repository create and system config), and a dedicated migration test
  that runs `alembic upgrade head` then `downgrade base` against a *throwaway*
  database (never the dev DB) to prove the migration is actually clean, not
  just that models import. CI now runs a Postgres + Redis service container
  for the backend job.

### Decisions

- **`bcrypt` directly, not `passlib[bcrypt]`.** `passlib` 1.7.4 (last release
  2020) is incompatible with `bcrypt>=4.1`'s changed wrap-bug self-test and
  throws `ValueError: password cannot be longer than 72 bytes` on every hash
  call, even for short passwords — this is `passlib`'s own internal probe
  string, not the caller's password. Since `passlib`'s abstraction wasn't
  buying us anything (only one scheme, bcrypt), dropped it for the `bcrypt`
  package directly.
- **`User.role` is a plain string, not a Postgres enum** — see the docstring
  on `app/models/enums.py::Role`. Adding a role later is a one-line change,
  not an `ALTER TYPE` migration. Same reasoning applies to other status-like
  columns whose values aren't finalized until later phases (e.g. `DefectReport.severity`,
  finalized in Phase 7's taxonomy).
- **No ORM `relationship()`s yet**, only FK columns. Nothing in Phase 1
  traverses e.g. `Repository.pull_requests`, so adding it now would be
  speculative; add per-relationship when a later phase's query actually needs
  the traversal.
- **Frontend auth uses `window.location.href`, not `router.push`, after
  login/logout.** Found via manual browser testing: the nav bar reads
  auth state once on mount, and Next's client-side `router.push` doesn't
  remount the shared root layout, so the nav kept showing the pre-login state
  after a successful login until a manual refresh. A hard navigation forces
  the whole app (nav included) to re-check `localStorage` and `/auth/me`.
- **Tokens live in `localStorage`, not an httpOnly cookie.** Simpler for a
  capstone demo login flow; the tradeoff is JS-readable tokens (XSS exposure).
  Worth revisiting in Phase 10's security pass if there's time.
- API container now runs `alembic upgrade head` on every startup (Dockerfile
  `CMD`), so `docker compose up` is still a genuine one-command bring-up with
  real tables, not just app code with an empty database.

### Known gaps (expected — later phases)

- No mining, ML scoring, triage, or forecasting endpoints yet — those tables
  exist but are unused until Phases 2–9 fill them in.
- No `relationship()`s on the models (see Decisions).
- `make seed` / `make reproduce` — seeding now works (`scripts/seed_demo.py`);
  `reproduce` still has nothing to call until Phase 3's training pipeline
  exists.
- Frontend has no automated test for the login/RBAC UI flows yet (only the
  Phase 0 health-check Vitest test) — the flow was verified manually in a
  real browser this phase; a Playwright pass is scheduled for Phase 10 per
  the master prompt's DoD (§11.11 vs. §10 Phase 10).

### How to demo it

1. `docker compose up -d` (from a clean state, Alembic runs automatically)
2. `docker compose exec api python /app/scripts/seed_demo.py`
3. Open http://localhost:3000/login, log in as `admin@bugflow.demo` /
   `bugflow-demo`
4. Nav shows Home / Repositories / Admin·Users / your name+role / Log out
5. `/repositories` — register a repo, toggle merge blocking
6. `/admin/users` — see all 8 seeded users, create a new one, deactivate one
7. Log out, then try `curl -X POST http://localhost:8000/repositories ...`
   as a non-admin token → `403`
8. `python scripts/list_story_coverage.py` now shows US-01, US-49, US-50 with
   tests

### Plain-English notes

- **Why does the login endpoint accept form data instead of JSON?** It follows
  the OAuth2 "password flow" convention FastAPI's own docs UI expects
  (`username`/`password` fields), so anyone poking at `/docs` can click
  "Authorize" and try protected routes immediately — no separate JSON login
  needed just to explore the API.
- **Why revoke tokens in Redis instead of just deleting them client-side?** A
  JWT is a self-contained, signed piece of data — the server has no
  built-in way to "cancel" one before it expires. Recording its ID in Redis
  with a matching expiry is the standard way to support real logout with
  short-lived JWTs.
- **Why a throwaway database for the migration test?** Running
  `alembic downgrade base` (which drops every table) against the same
  database other tests use would wipe their data mid-suite. Spinning up
  `bugflow_migration_test`, migrating it up and down, then dropping it proves
  the migration works without touching anything else.

## Phase 0 — Foundations and guardrails

**Stories:** US-50, NFR-US-09, NFR-US-10.

### What was built

- Monorepo skeleton matching §6 of the master prompt: `backend/` (FastAPI app,
  installable), `ml/` (installable `bugflow_ml` package with empty
  `mining/labeling/features/models/assignment/explain/monitoring/registry`
  sub-packages, filled in from Phase 2 onward), `frontend/` (Next.js), `scripts/`,
  `docs/`.
- `docker-compose.yml` with all six services (`postgres`, `redis`, `mlflow`,
  `api`, `worker`, `frontend`), each with a health check, wired so
  `api`/`worker`/`frontend` wait on their dependencies being healthy.
- `.env.example` and a `Makefile` (`make up`, `make down`, `make logs`,
  `make test`, `make lint`, `make seed`, `make reproduce`).
- **Backend:** FastAPI app (`app/main.py`) with `/health`, config via
  `pydantic-settings` (`app/core/config.py`), structured JSON logging via
  `structlog` (`app/core/logging.py`), a lifespan startup log line.
- **Worker:** a minimal RQ worker (`app/workers/run.py`) listening on the
  `default` queue — no real jobs yet, added in Phase 2+.
- **Frontend:** Next.js 14 (App Router) + TypeScript + Tailwind, `components.json`
  wired for shadcn/ui (`baseColor: slate`, CSS variables in `app/globals.css`),
  a layout shell (`app/layout.tsx`) and a home page (`app/page.tsx`) with a
  `<HealthStatus>` client component that fetches the API's `/health` endpoint
  and shows a green/red status dot.
- **Tooling:** ruff + mypy (backend/ml), ESLint + Prettier + Vitest (frontend),
  `pytest-cov` with a `story` marker registered in both Python `pyproject.toml`
  files, `.pre-commit-config.yaml` (gitleaks, ruff, mypy, eslint, prettier), a
  GitHub Actions workflow (`.github/workflows/ci.yml`) with separate
  `secret-scan` / `backend` / `ml` / `frontend` jobs.
- `scripts/list_story_coverage.py`: scans for `@pytest.mark.story("US-xx")` and
  `// US-xx` and prints which stories have at least one test.

### Decisions

- No ADR needed — Phase 0 followed the master prompt's stack and structure
  exactly, with no deviations.
- Bumped `next` from the prompt-adjacent default patch to `14.2.35` (latest
  patched release on the 14.x line) after `npm install` flagged a known
  security advisory on `14.2.15`.
- `worker`'s Docker health check reads `/proc/1/cmdline` (no HTTP endpoint;
  `pgrep` isn't installed in the slim base image and wasn't worth adding just
  for this). Revisit once Phase 2 mining jobs exist and there's real work to
  check liveness through.
- Redis's host-published port is `6380` (env `REDIS_HOST_PORT`), not the
  default `6379` — a pre-existing unrelated container on this machine already
  held `6379`. Container-to-container traffic (`redis://redis:6379`) is
  unaffected; this only changes the host-side mapping for local debugging.

### Known gaps (expected — later phases)

- No database models/migrations yet (Phase 1).
- No auth/RBAC yet (Phase 1).
- Worker has no real jobs (Phase 2+).
- `ml` package has empty sub-packages, no ML dependencies installed yet — they
  get added task-by-task from Phase 2 (mining) onward so nothing unused sits
  in the tree.
- No GitHub App / webhook integration yet (Phase 4).
- `make seed` / `make reproduce` targets exist but the scripts they call don't
  yet (Phase 1 / Phase 3).

### How to demo it

1. `cp .env.example .env`
2. `make up` (or `docker compose up --build`)
3. Open http://localhost:3000 — the home page shows a green "API is reachable"
   dot, confirming the frontend called the FastAPI backend's `/health` over
   HTTP.
4. `curl http://localhost:8000/health` → `{"status": "ok"}`
5. All six containers report healthy: `docker compose ps`.
6. `make test` runs backend (pytest), ml (pytest) and frontend (vitest) suites.
7. `make lint` runs ruff/mypy and ESLint/Prettier.
8. `python scripts/list_story_coverage.py` lists which stories currently have
   tests (US-50, from the health check tests on both backend and frontend).

### Plain-English notes

- **Why a monorepo with three installable Python-ish units (backend, ml,
  frontend)?** The `ml` package is meant to be reused by both the API (to score
  a request) and the worker (to train/retrain in the background), so it's kept
  as its own installable package rather than living inside `backend/`.
- **Why health checks on every service?** `docker-compose.yml` uses
  `depends_on: condition: service_healthy` so, e.g., the API doesn't start
  accepting traffic before Postgres/Redis/MLflow are actually ready — avoiding
  flaky "connection refused" errors on a fresh `docker compose up`.
- **Why structured (JSON) logging from day one?** Once real workers and
  webhooks exist, plain `print()` logs are hard to search; JSON logs can be
  piped into any log viewer without changing the code later.
