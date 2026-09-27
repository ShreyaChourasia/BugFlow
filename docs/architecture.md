# Architecture

See `BUGFLOW_MASTER_PROMPT.md` §5 for the full diagram and pre-merge flow narrative.

## Services (docker-compose.yml)

| Service | Image/build | Purpose | Health check |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | Relational data + vector search for duplicates | `pg_isready` |
| `redis` | `redis:7-alpine` | Job queue backing RQ | `redis-cli ping` |
| `mlflow` | `./mlflow` (build) | Experiment tracking / model registry | HTTP GET `/` |
| `api` | `./backend` (build) | FastAPI application, serves REST API | HTTP GET `/health` |
| `worker` | `./backend` (build), `rq worker` entrypoint | Background jobs: mining, scoring, training | process check |
| `frontend` | `./frontend` (build) | Next.js UI | HTTP GET `/` |

## Phase 0 scope

Only the skeleton exists: every service boots, has a health check, and the frontend's
health page calls the API's `/health` endpoint end-to-end. No business logic yet —
that arrives in Phases 1+ per the master prompt's phase table (§10).

## Data flow (target, built up over phases)

GitHub webhook → API verifies HMAC, enqueues a job, posts a "pending" check →
RQ worker fetches diff, scores with the champion model, calibrates, explains →
API edits the existing PR comment and updates the check. See §5 for the diagram.
