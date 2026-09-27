# Progress log

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
