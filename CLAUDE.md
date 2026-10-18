# BugFlow — working notes for future sessions

Full spec: `BUGFLOW_MASTER_PROMPT.md`. Read it fully before any phase.

## Rules (see master prompt §0 for full text)
- One phase at a time. Plan → implement → test/lint → update `docs/PROGRESS.md` →
  commit `phase-N: <summary>` → stop and report. Never start the next phase unsolicited.
- `docker compose up` must always start the whole system.
- Ask before changing the stack or an unclear requirement; record agreed decisions
  as ADRs in `docs/decisions/NNN-title.md`.
- No secrets in git. `.env` is gitignored; keep `.env.example` current.
- Every feature maps to a user story ID (`US-xx` / `NFR-US-xx`); tag tests with
  `@pytest.mark.story("US-xx")` (Python) or `// US-xx` (frontend).
- Never invent facts about datasets/libraries — verify availability and licence first.
- Explain new parts in plain English in `docs/PROGRESS.md` at the end of each phase.

## Stack
Python 3.11 + FastAPI + Pydantic v2 · PostgreSQL 16 + pgvector · SQLAlchemy 2.x + Alembic ·
Redis + RQ · scikit-learn/LightGBM/SHAP/sentence-transformers/lifelines/OR-Tools ·
PyDriller + GitHub REST (httpx) · MLflow · Next.js (App Router) + TS + Tailwind + shadcn/ui +
TanStack Query + Recharts · JWT + bcrypt + RBAC · GitHub App (webhooks, Checks API) ·
pytest/Vitest/Playwright · ruff/mypy/ESLint/Prettier/pre-commit · gitleaks · GitHub Actions.

## Repo layout
See master prompt §6. `backend/` = API + workers. `ml/` = installable `bugflow_ml` package
used by both. `frontend/` = Next.js app. `scripts/` = seeding, replay, perf tests.

## Status
Post-Phase-10 follow-ups done (shadow-scoring on its own low-priority queue, Next.js 15 + React 19, CI's Vitest/e2e clash fixed). Phase 10 complete (analytics, shadow mode, drift monitoring, model retrain/rollback lifecycle, counterfactual guidance, security/perf hardening, documentation) — end of Release 3, final phase. Phase 9 (resolution forecasting), Phase 8 (resolver recommendation, OR-Tools batch assignment, end of Release 2), Phase 7 (severity/priority classification), Phase 6 (defect reports, duplicate detection), Phase 5 (line-level risk, review queue), and Phase 4 (GitHub App integration, end of Release 1) precede it. See `docs/PROGRESS.md` for what exists and how to demo it.

## Gotchas worth knowing before touching process entrypoints
- **LightGBM + PyTorch (via sentence-transformers) segfault if torch loads first** in the same process (both bundle their own OpenMP runtime). `app/core/native_libs.py` imports LightGBM first and must stay imported at the top of `app.main`, `app.workers.run`, and `tests/conftest.py`. See `docs/ml.md`'s duplicate-detection section.
- **`pip install torch` on Linux defaults to the CUDA build** (multi-GB) even in a CPU-only container. `backend/Dockerfile` passes `--extra-index-url https://download.pytorch.org/whl/cpu` to avoid it — don't drop that flag when touching the Dockerfile.
- **Unbounded list endpoints will actually get hit at scale in this project** — `scripts/load_test_defects.py` exists specifically to bulk-load real data volume, and it found `GET /defect-reports` returning every row (no pagination) as a real bug, not a hypothetical one. Any new "list everything" endpoint should default to a `limit`.
- **After any change, rebuild and redeploy every service whose source
  changed — not just the one the phase's backend work happened to focus
  on.** Phase 7 touched backend + frontend; only `api`/`worker` got rebuilt,
  and the running `frontend` container silently kept serving the old bundle
  for the whole live-verification session — `curl`-testing the API looked
  perfect while the browser UI was just missing the new feature entirely,
  no error anywhere. `docker compose build <service>` + `up -d <service>`
  for every service with changed source, every time, before trusting a live
  check of any kind.
- **Test fixtures that isolate from this shared dev Postgres should
  `TRUNCATE ... CASCADE`, not `delete()` table-by-table.** A test fixture's
  `DELETE FROM repositories` (to clear ambient demo data) started failing on
  its own FK constraint the moment Phase 6 added
  `defect_reports.repository_id` — the delete chain had to list every
  dependent table by hand and nobody updated it when a later phase added a
  new one. `TRUNCATE a, b, c CASCADE` cascades to *any* table referencing
  those, including ones added in a later phase, with no list to maintain.
  See `tests/conftest.py`'s `repository_with_commits` and
  `tests/test_pr_scoring_service.py`'s `champion_and_pr_with_line_risk`.
- **`ortools` segfaults when imported via this project's host macOS Anaconda
  Python** — a few of its bundled `.so` files on that wheel have a hardcoded
  absolute `/opt/anaconda3/lib/...` load path baked in from its own build
  environment, which collides with a real, differently-built `libprotobuf`
  at that exact path if the host machine also has Anaconda installed there
  (confirmed with `otool -L`; two independently-built copies of libprotobuf
  loading into one process trips a duplicate-descriptor-registration abort).
  Irrelevant to the actual shipped system — the `api`/`worker` Docker images
  use Linux manylinux wheels and import `ortools` cleanly. Run
  `ortools`-dependent tests (`ml/tests/test_batch_assignment.py` and anything
  importing `bugflow_ml.assignment`) inside the Docker containers, not the
  host interpreter, if this recurs.
- **A new Alembic migration isn't applied just by rebuilding images.**
  `alembic upgrade head` only runs as part of the main `api` service's own
  startup command — `docker compose run --rm api ...` (what every ad hoc
  test/verification command in this project uses) shares the same Postgres
  volume but does **not** re-run that startup command, so a freshly-added
  column exists in the SQLAlchemy models but not yet in the actual shared
  dev database. Phase 10 hit this directly: adding
  `RiskPrediction.is_shadow` caused 18 unrelated-looking test failures
  (`column "is_shadow" ... does not exist`) until `docker compose run --rm
  api alembic upgrade head` was run by hand. After adding any migration,
  apply it explicitly before trusting any test run or live check — don't
  assume a rebuild did it.
- **Two separate `func.coalesce(...)` (or any repeated SQL function) calls
  passed to `.select()` and `.group_by()` aren't recognized as the same
  expression by Postgres**, even though they render identically — each
  compiles its own bind parameter, and Postgres then refuses the query
  ("column ... must appear in the GROUP BY clause"). Build the expression
  once, assign it to a variable, and reuse that same object in both places
  (see `analytics_service.py::get_hotspots`, found by a real test failure,
  not inspection).
