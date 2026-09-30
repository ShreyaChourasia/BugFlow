# BugFlow

BugFlow is a software quality assistant, built as a university capstone
project, that helps a development team at two points in the software
lifecycle:

- **Pre-merge:** when a pull request is opened, BugFlow predicts how likely
  the change is to introduce a bug, highlights the riskiest lines when risk
  is high, and posts the result — with a plain-language explanation — back
  onto the PR.
- **Triage:** while a bug report is being filed, BugFlow surfaces similar
  existing reports, detects duplicates, suggests severity and priority, and
  recommends a resolver while respecting each developer's current workload.

Its novel contribution is treating **resolver assignment as a
capacity-constrained batch optimisation problem** (via OR-Tools), rather than
greedily assigning bugs to whichever developer looks best one at a time.

The full design — user stories, non-negotiable constraints, data model, and
the phase-by-phase build plan this project follows — lives in
[`BUGFLOW_MASTER_PROMPT.md`](./BUGFLOW_MASTER_PROMPT.md).

## Status

Built phase by phase, each one fully working and demoable before the next
starts. **Release 1 is complete** (Phases 0–4 of 10) — see
[`docs/PROGRESS.md`](./docs/PROGRESS.md) for exactly what was built, what
decisions were made and why, and how to demo each phase.

| Phase | What it adds | Status |
|---|---|---|
| 0 | Monorepo skeleton, `docker compose up`, CI, tooling | ✅ Done |
| 1 | Auth, roles, repository CRUD, admin/system config | ✅ Done |
| 2 | Git mining, SZZ bug-inducing-commit labelling, commit features | ✅ Done |
| 3 | Commit risk model (LightGBM + calibration), SHAP explanations, MLflow | ✅ Done |
| 4 | GitHub App integration — risk posted on the actual PR (end of Release 1) | ✅ Done |
| 5–10 | Line-level risk, triage/duplicates, resolver assignment, forecasting, analytics, model lifecycle | ⏳ Not started |

## Tech stack

| Layer | Choice |
|---|---|
| Backend API | Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2.x + Alembic |
| Database | PostgreSQL 16 + pgvector |
| Background jobs | Redis + RQ |
| ML | scikit-learn, LightGBM, SHAP, MLflow, PyDriller (repo mining) |
| Frontend | Next.js (App Router) + TypeScript, Tailwind CSS |
| Auth | JWT (access + refresh), bcrypt, role-based access control |
| Testing | pytest, Vitest |
| Quality | ruff, mypy, ESLint, Prettier, gitleaks |
| CI | GitHub Actions |

## Quickstart

Requires Docker.

```bash
cp .env.example .env
make up          # docker compose up --build — starts all 6 services
make seed        # creates one demo user per role, password: bugflow-demo
```

Then open:
- **http://localhost:3000** — the web app (log in as e.g. `admin@bugflow.demo`)
- **http://localhost:8000/docs** — the API's interactive OpenAPI docs
- **http://localhost:5000** — the MLflow UI (experiment tracking)

### Demo accounts

Every account below shares the password `bugflow-demo` (created by `make seed`):

| Email | Role |
|---|---|
| `developer@bugflow.demo` | Developer |
| `reviewer@bugflow.demo` | Code Reviewer |
| `triager@bugflow.demo` | Bug Triager |
| `reporter@bugflow.demo` | Bug Reporter |
| `qa@bugflow.demo` | QA Engineer |
| `manager@bugflow.demo` | Engineering Manager |
| `ml-engineer@bugflow.demo` | ML Engineer |
| `admin@bugflow.demo` | System Administrator |

### Mining a repository and training the risk model

The demo starts with no mined data. As an Admin or ML Engineer:

1. Log in, go to **Repositories**, register one — a small real repo with
   genuine bug-fix history demos well, e.g. `https://github.com/benjaminp/six`
   (branch `main`, ~500 commits, mines in well under a minute)
2. Open its detail page and click **Start mining**
3. Once it shows `completed`, train the risk model:
   ```bash
   docker compose exec api python /app/scripts/train.py
   ```
4. Score a mined commit:
   ```bash
   curl -X POST http://localhost:8000/predict/commit \
     -H "Authorization: Bearer <token from POST /auth/login>" \
     -H "Content-Type: application/json" \
     -d '{"repository_id": <id>, "sha": "<a mined commit sha>"}'
   ```
   Returns a probability, calibrated probability, confidence, risk level, and
   a plain-language explanation with its top contributing factors.

### Seeing risk posted on a pull request

Works fully offline — no GitHub account needed (see
[`docs/github-app.md`](./docs/github-app.md) to connect a real repo instead):

```bash
docker compose exec api python /app/scripts/replay_pr_events.py \
  --repository-id <id> --sha <a mined commit sha> --pr-number 101
```

Then open `/repositories/<id>/pull-requests` in the UI — the check resolves
from `pending` to a conclusion within a couple of seconds, with the full
risk breakdown and the comment that was (or would have been) posted.

## Repository structure

```
backend/    FastAPI app + RQ workers (installable Python package)
ml/         bugflow_ml — mining, labelling, features, models, explanations
            (installable, used by both the API and the workers)
frontend/   Next.js app
scripts/    seed_demo.py, train.py, reproduce_run.py, export_experiment.py,
            replay_pr_events.py, perf/ (k6 scripts)
docs/       PROGRESS.md, architecture.md, ml.md, github-app.md, decisions/ (ADRs)
```

## Running tests and linters

The running containers only have runtime dependencies installed, not the
dev tooling (pytest, ruff, mypy) — install each package locally first
(requires a running Postgres + Redis, e.g. from `make up`):

```bash
pip install -e ./ml -e "./backend[dev]"
cd backend && DATABASE_URL=postgresql+psycopg://bugflow:bugflow@localhost:5432/bugflow \
  REDIS_URL=redis://localhost:6380/0 pytest
cd ml && pytest
cd frontend && npm ci && npm test
```

`make test` and `make lint` wrap the equivalent commands. CI
(`.github/workflows/ci.yml`) runs lint, type checks, and tests for all three
packages — with its own Postgres/Redis service containers — plus a gitleaks
secret scan, on every push and PR.

## API reference

Full interactive docs at `/docs` once the API is running. Implemented so far:

| Endpoint | Purpose |
|---|---|
| `POST /auth/login`, `/auth/refresh`, `/auth/logout`, `GET /auth/me` | JWT auth |
| `GET/POST/PATCH/DELETE /repositories` | Repository management (Admin) |
| `POST/GET /repositories/{id}/mining-runs`, `.../resume` | Trigger and track git mining |
| `POST /predict/commit`, `GET /commits/{sha}/risk` | Commit risk scoring + explanation |
| `POST /webhooks/github` | GitHub App webhook (HMAC-signed `pull_request` events) |
| `GET /repositories/{id}/pull-requests`, `.../{number}` | PR list/detail: check status, risk, comment |
| `GET/PUT /admin/config`, `GET /admin/audit-log` | System settings (Admin) |
| `GET/POST/PATCH /users` | User management (Admin) |

## Further reading

- [`docs/PROGRESS.md`](./docs/PROGRESS.md) — what was built each phase, why,
  known gaps, and how to demo it
- [`docs/ml.md`](./docs/ml.md) — model choices and their documented
  limitations (SZZ, commit features, calibration, explanations)
- [`docs/github-app.md`](./docs/github-app.md) — connect a real GitHub repo
  (optional — everything works offline without it)
- [`docs/architecture.md`](./docs/architecture.md) — service diagram
- [`docs/decisions/`](./docs/decisions/) — ADRs for anywhere the
  implementation deviated from the original data model
- [`CLAUDE.md`](./CLAUDE.md) — working notes for continuing this build
