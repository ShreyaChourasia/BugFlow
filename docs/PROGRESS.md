# Progress log

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
