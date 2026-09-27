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
Phase 2 complete (repository mining, SZZ labelling, commit features) — see `docs/PROGRESS.md` for what exists and how to demo it.
