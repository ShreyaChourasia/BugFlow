# Architecture

See `BUGFLOW_MASTER_PROMPT.md` §5 for the original target diagram. This
document reflects what's actually built as of Phase 10 (end of Release 3).

## Services (`docker-compose.yml`)

| Service | Image/build | Purpose | Health check |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | Relational data + vector search for duplicates (HNSW) | `pg_isready` |
| `redis` | `redis:7-alpine` | RQ job queue backing; also backs the login rate limiter | `redis-cli ping` |
| `mlflow` | `./mlflow` (build) | Experiment tracking / model registry for every trained model | HTTP GET `/` |
| `api` | `./backend` (build) | FastAPI application, serves the REST API | HTTP GET `/health` |
| `worker` | `./backend` (build), `rq worker` entrypoint | Background jobs: mining, scoring, training, retraining | process check |
| `drift-scheduler` | `./backend` (build), shell loop | Runs `scripts/check_drift.py` every `DRIFT_CHECK_INTERVAL_SECONDS` (US-41) | — |
| `frontend` | `./frontend` (build) | Next.js UI | HTTP GET `/` |

`backend/` is one installable Python package (`app`) used by both `api` and
`worker`; `ml/` is a separate installable package (`bugflow_ml`) with no
database or web-framework dependency, used by both of those plus the
standalone `scripts/`.

## Two main data flows

**Pre-merge (risk on a pull request):**
```
GitHub webhook ──► API verifies HMAC, enqueues a job, posts "pending" (<500ms, C1)
                        │
                        ▼
                  RQ worker: fetch diff → compute commit features →
                  score with the champion model → calibrate → SHAP explain →
                  edit the existing PR comment, resolve the check →
                  (if a challenger exists) queue a shadow-score job on the
                  low-priority "shadow" queue, run only when "default" is idle
```
Offline-friendly: `scripts/replay_pr_events.py` drives this exact path from
a local git repo, no GitHub account needed.

**Triage (a bug report's life after filing):**
```
Reporter types ──► live duplicate suggestions (pgvector HNSW, <300ms, C2)
Reporter submits ──► duplicate candidates + severity/priority suggestion
                      (abstains below a word-count floor)
Triager decides ──► resolver recommendations (ranked, cold-start aware) →
                     batch-assign under capacity (OR-Tools) or override+reason
                      │
                      ▼
              resolution forecast (Kaplan-Meier baseline + Cox PH,
              censored data included) → median/P90/at-risk flag
```

## Model lifecycle (Phase 10)

Every trainable task (`commit_risk`, `line_risk`, `triage_severity`,
`triage_priority`, `resolver_suitability`, `resolution_forecast`) follows
the same loop:

```
train_and_register_champion() ──► logs params/metrics/artifacts to MLflow
         │                         ──► writes one `MLModel` row (champion/
         │                             challenger/archived) to Postgres
         ▼
  ML Engineer, via /models:
    • retrain (dispatched to a worker job, not the request path)
    • rollback (any archived/challenger version ← champion)
    • (commit-risk only) shadow comparison: the current challenger scores
      every real commit silently alongside the champion; a comparison page
      shows agreement rate and mean difference on the exact same inputs
    • (commit-risk only) drift check: PSI per feature, reference = the
      commits the champion actually trained on, current = everything mined
      since — alerts when a feature has drifted past the threshold
```

Scoped to commit-risk only for shadow mode and drift — see
`docs/decisions/008-shadow-mode-scope.md` for why, and what a future phase
extending it to the other four models would reuse.

## Security

- JWT (short-lived access + refresh), bcrypt password hashing, 8 roles
  (`app/models/enums.py::Role`), enforced per-router via `require_role()`.
- `/auth/login` is rate-limited (5 attempts/60s per email) via a Redis
  INCR+EXPIRE counter — the same Redis connection JWT revocation already
  uses, no new service.
- GitHub webhooks authenticate via HMAC signature, not JWT (there's no user
  session to attach to a webhook delivery).
- `gitleaks` scans every push/PR in CI; `pip-audit` (backend, ml) and
  `npm audit` (frontend) run in CI as a dependency audit — see
  `docs/PROGRESS.md`'s Phase 10 entry for current findings.
