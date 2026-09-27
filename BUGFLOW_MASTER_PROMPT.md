# BugFlow — Master Build Prompt (for Claude Code)

> **How to use this file**
> 1. Create an empty folder or repo called `bugflow` and put this file in its root.
> 2. Open Claude Code in that folder and say:
>    **"Read BUGFLOW_MASTER_PROMPT.md fully. Then do Phase 0 only and stop."**
> 3. After each phase, check the demo, then say **"Start Phase N."**
>
> Never ask Claude Code to build everything in one go. Phases exist so each piece works before the next one starts.

---

## 0. Your role and working rules

You are the lead engineer building **BugFlow**, a full-stack web application with a machine-learning backend, as a university capstone project. The code will be read, demoed and examined, so **clarity beats cleverness**.

Follow these rules for the whole project:

1. **One phase at a time.** At the start of a phase:
   - read that phase's section here, plus `docs/PROGRESS.md`
   - write a short plan (files to create, key decisions, risks)
   - then implement it
2. **End every phase the same way:**
   - run all tests, linters and type checks
   - update `docs/PROGRESS.md`: what was built, decisions, known gaps, and how to demo it
   - commit with `phase-N: <summary>`
   - **stop and report** with a demo checklist

   Do not start the next phase until I say so.
3. **Always runnable.** `docker compose up` must start the whole system at the end of every phase.
4. **Ask before deviating.** If a requirement is unclear, or you want to change the stack or a design decision, ask first. Record agreed decisions as short ADRs in `docs/decisions/NNN-title.md`.
5. **No secrets in git, ever.** Use `.env` (gitignored) and keep `.env.example` up to date.
6. **Traceability.** Every feature maps to a user story ID (`US-xx` functional, `NFR-US-xx` non-functional). Tag tests with the story they verify: `@pytest.mark.story("US-08")` in Python, and a `// US-08` comment in frontend tests.
7. **Never invent facts about datasets or libraries.** Before relying on a dataset, check it is available and what licence it has. If a download or API fails, tell me instead of faking data silently.
8. **Explain as you go.** At the end of each phase, add a short plain-English section to `docs/PROGRESS.md` explaining how the new parts work. I am learning from this project.

---

## 1. What BugFlow is

BugFlow is a software quality assistant that helps a development team at **two points** in the software lifecycle.

**Stage 1: before code is merged (pre-merge)**
- When a pull request (PR) is opened, BugFlow predicts how likely the change is to introduce a bug.
- If the risk is high, it highlights the riskiest lines.
- It posts the result, with a plain-language explanation, back onto the PR.
- Reviewers get a review queue sorted by risk.

**Stage 2: after a bug is reported (triage)**
- It shows similar existing reports while the reporter is still typing.
- It detects duplicates.
- It suggests severity and priority.
- It recommends a ranked list of resolvers while **respecting each developer's workload**.
- It forecasts how long the fix is likely to take, as a probability over time.

**Learning loop**
- Humans can accept or override every automated decision.
- Overrides are stored as feedback.
- Models are monitored for drift, then retrained, shadow-tested and rolled back when needed.

**Novel contribution:** resolver assignment is treated as a **capacity-constrained assignment optimisation problem**. Instead of greedily picking the best developer for each bug one at a time, BugFlow assigns a whole batch of bugs to maximise overall suitability, while no developer exceeds their capacity (US-25).

---

## 2. User roles

| ID | Role | Main needs | Key permissions |
|---|---|---|---|
| U1 | Developer | Early risk warning, reasons, why a bug was assigned to them | View own PRs and risk; view own assignments; object to an assignment |
| U2 | Code Reviewer | Review queue ranked by risk; highlighted lines | View all PRs; mark lines as false alarms |
| U3 | Bug Triager | Duplicates filtered, severity pre-set, resolver suggested | Merge duplicates; change severity; assign, batch assign, override |
| U4 | Bug Reporter | Avoid duplicates; know when the bug will be fixed | Create reports; view own reports |
| U5 | QA Engineer | Consistent severity; defect-prone modules | View reports, analytics and historical risk |
| U6 | Engineering Manager | Trends, workload fairness, acceptance rates | View all analytics; configure merge blocking |
| U7 | ML Engineer | Drift alerts, retrain, rollback, reproducibility | Manage models and experiments |
| U8 | System Administrator | One-command setup, access control | Manage users, roles, repos, integrations, thresholds |

---

## 3. Non-negotiable system-wide constraints

Every phase must respect these. Each needs an **automated test** as soon as the relevant feature exists.

| ID | Constraint | Source |
|---|---|---|
| C1 | A PR risk assessment is posted within **500 ms at p95** for the classical model. If the timeout is hit, the check shows **"pending"**, never a stale or partial result. | NFR-US-01 |
| C2 | Duplicate search over **≥ 300,000 reports** returns in **≤ 300 ms at p95**. While the index is rebuilding, the user is told so, not given an empty result. | NFR-US-02 |
| C3 | **No automated decision is ever shown without a plain-language explanation.** This covers risk scores, severity labels, resolver recommendations and forecasts. | NFR-US-06 |
| C4 | Re-running a recorded experiment reproduces its headline metrics within **0.5%**. Results without a seed or data version are flagged as non-reproducible. | NFR-US-07 |
| C5 | Automated test coverage of the core pipeline stays **≥ 80% of lines** on the main branch. | NFR-US-08 |
| C6 | No credential, token or key is ever in the repo; CI fails if one appears. Platform tokens are **least-privilege**. | NFR-US-10 |
| C7 | If the scoring service fails, the affected workflow **says so explicitly** and **never blocks a merge**. | NFR-US-04, NFR-US-05 |
| C8 | The whole system starts with **one command** (`docker compose up`) and runs on **free or student-tier** infrastructure. | NFR-US-09, NFR-US-13 |
| C9 | Model evaluation is **chronological**: train on the past, test on the future. A leakage test must pass. | US-42 |
| C10 | Every evaluation reports **risk disparity by contributor tenure**. | NFR-US-11 |

---

## 4. Tech stack

Use this stack unless I approve a change.

| Layer | Choice | Why |
|---|---|---|
| Backend API | **Python 3.11 + FastAPI**, Pydantic v2 | The ML ecosystem is in Python; FastAPI is fast and self-documenting |
| Database | **PostgreSQL 16 + pgvector** | Relational data plus vector search for duplicates in one database |
| ORM and migrations | SQLAlchemy 2.x + Alembic | Standard, typed, versioned schema |
| Background jobs | **Redis + RQ** (or arq) | Mining, batch scoring and retraining run outside the request path |
| ML | scikit-learn, **LightGBM**, **SHAP**, **sentence-transformers** (`all-MiniLM-L6-v2`), **lifelines** or scikit-survival, **OR-Tools** | Classical models, explanations, embeddings, survival forecasts, optimisation |
| Repo mining | **PyDriller** and the GitHub REST API (httpx) | Commit history, diffs and blame |
| Experiment tracking | **MLflow** (local server in Docker) | Reproducibility, model registry, rollback |
| Frontend | **Next.js (App Router) + TypeScript**, Tailwind CSS, shadcn/ui, TanStack Query, Recharts | Modern, typed, clean UI components and charts |
| Auth | JWT (short-lived access + refresh), bcrypt, role-based access control | Simple, self-contained, no paid service |
| Platform integration | **GitHub App**: webhooks + Checks API + issue comments | Results appear where developers already work (US-46) |
| Testing | pytest, pytest-cov, Vitest + Testing Library, Playwright (a few end-to-end tests) | Unit, integration and end-to-end coverage |
| Quality | ruff, mypy, ESLint, Prettier, pre-commit | Linting and type checks in the Definition of Done |
| Security | **gitleaks** in pre-commit and CI; webhook HMAC verification | C6 |
| CI | GitHub Actions | Free for student projects |
| Performance tests | k6 or Locust scripts in `scripts/perf/` | Measure C1 and C2 |

---

## 5. Architecture

```
                 ┌───────────────┐
  GitHub ──────► │  Webhook      │  verify HMAC signature, enqueue job,
  (PR events)    │  endpoint     │  post "pending" check immediately
                 └──────┬────────┘
                        │
┌──────────────┐  REST  ▼          ┌──────────────┐     ┌───────────┐
│  Next.js     │ ◄────────────────►│  FastAPI     │◄───►│ Postgres  │
│  frontend    │                   │  API         │     │ + pgvector│
└──────────────┘                   └──────┬───────┘     └───────────┘
                                          │ enqueue
                                   ┌──────▼───────┐     ┌───────────┐
                                   │ Redis + RQ   │────►│  Workers  │
                                   └──────────────┘     │ (mining,  │
                                                        │ scoring,  │
                                   ┌──────────────┐     │ training) │
                                   │   MLflow     │◄───►└───────────┘
                                   │ (registry)   │
                                   └──────────────┘
```

**Pre-merge flow**
1. The webhook arrives. Verify the signature, respond `202` quickly, and post a **pending** check.
2. A worker fetches the diff, computes features and scores with the **champion** model.
3. It calibrates the score, ranks lines if risk is high, and generates the explanation.
4. It updates the check and **edits the existing comment** rather than adding a new one.

**Services in `docker-compose.yml`:** `frontend`, `api`, `worker`, `postgres`, `redis`, `mlflow`. All have health checks.

**Offline demo mode (important for viva/demos).** Include `scripts/replay_pr_events.py`. It replays PR events from a local git repository into the webhook endpoint, and a fake Checks sink records what would be posted. The full flow must be demoable **without internet or GitHub**.

---

## 6. Repository structure

```
bugflow/
├── BUGFLOW_MASTER_PROMPT.md
├── CLAUDE.md                  # short working notes for future sessions (Phase 0)
├── docker-compose.yml
├── .env.example
├── Makefile                   # make up, make test, make lint, make seed, make reproduce RUN_ID=...
├── backend/
│   ├── app/
│   │   ├── api/               # routers grouped by area
│   │   ├── core/              # config, security, RBAC, logging
│   │   ├── models/            # SQLAlchemy models
│   │   ├── schemas/           # Pydantic schemas (every decision schema has a required `explanation`)
│   │   ├── services/          # business logic
│   │   ├── integrations/github/
│   │   └── workers/           # RQ jobs
│   ├── alembic/
│   └── tests/
├── ml/                        # installable package `bugflow_ml`, used by api + worker
│   ├── bugflow_ml/
│   │   ├── mining/            # PyDriller, issue linking, checkpointing
│   │   ├── labeling/          # SZZ
│   │   ├── features/
│   │   ├── models/            # risk, line, duplicate, severity, resolver, forecast
│   │   ├── assignment/        # capacity-constrained optimiser
│   │   ├── explain/
│   │   ├── monitoring/        # drift, fairness
│   │   └── registry/          # MLflow helpers
│   └── tests/
├── frontend/
├── scripts/                   # seed_demo.py, replay_pr_events.py, perf/
└── docs/
    ├── PROGRESS.md
    ├── architecture.md
    ├── api.md
    ├── ml.md
    └── decisions/
```

---

## 7. Data model

Use the class names from the Experiment 5 class **tables**: `DefectReport`, `TriageAssessment`, and so on. This table is the starting point; extend it with an ADR if needed.

| Entity | Key fields |
|---|---|
| `User` | id, name, email, password_hash, role, is_active, developer_id (nullable) |
| `Repository` | id, name, url, issue_tracker_url, github_installation_id, default_branch, merge_blocking_enabled, risk_threshold |
| `MiningRun` | id, repository_id, status, checkpoint (json), started_at, finished_at, error |
| `Developer` | id, name, git_emails[], joined_at, capacity, current_queue_depth, skills/components[] |
| `Commit` | id, sha, repository_id, author_id, message, timestamp, lines_added, lines_deleted, files_changed, features (json), is_bug_inducing, is_fix, data_version |
| `PullRequest` | id, repository_id, number, title, author_id, status, head_sha, base_sha, created_at, check_run_id, comment_id |
| `RiskPrediction` | id, commit_id / pr_id, model_version, probability, calibrated_probability, confidence, risk_level, latency_ms, created_at |
| `LineRisk` | id, prediction_id, file_path, line_no, code, risk_score, rank, reason, marked_false_alarm |
| `DefectReport` | id, repository_id, reporter_id, title, description, component, status, severity, priority, embedding (vector 384), duplicate_of_id, assignee_id, reported_at, resolved_at |
| `TriageAssessment` | id, defect_id, severity, priority, confidence, model_version, is_automated, assessed_at |
| `ResolverRecommendation` | id, defect_id, developer_id, score, rank, workload_at_time, reason |
| `Assignment` | id, defect_id, developer_id, source (auto / batch / override), assigned_by, assigned_at |
| `ResolutionForecast` | id, defect_id, median_days, p90_days, curve (json), confidence, model_version, at_risk |
| `Explanation` | id, decision_type, decision_id, text, factors (json), created_at |
| `Feedback` | id, decision_type, decision_id, user_id, accepted, original_value, new_value, reason, created_at |
| `MLModel` | id, task, version, mlflow_run_id, stage (champion / challenger / archived), metrics (json), data_version, seed, trained_at |
| `DriftAlert` | id, model_id, feature, psi, severity, created_at, acknowledged |
| `AuditLog` | id, user_id, action, entity, entity_id, payload, created_at |
| `SystemConfig` | key, value, updated_by, updated_at |

---

## 8. ML components

| Task | Baseline approach | Output | Explanation | Main metrics |
|---|---|---|---|---|
| Training labels | **SZZ**: link fixing commits to issues, then `git blame` the fixed lines back to the commits that introduced them | `is_bug_inducing` per commit | — | Manual spot-check of a sample |
| Commit risk (JIT defect prediction) | Change metrics: size, diffusion (files, dirs, subsystems, entropy), history, author experience, fix flag. Model: **LightGBM**, with logistic regression as a baseline. Class weighting for imbalance. **Calibrated** with isotonic or Platt scaling. | Calibrated probability, confidence, risk level | **SHAP** top-3 factors, turned into sentences | ROC-AUC, PR-AUC, F1, recall@20% effort, Brier score |
| Line-level localisation | JITLine-style: tokenise added lines, TF-IDF combined with commit metrics, token attribution (LIME/SHAP) aggregated per line, top-N lines | Ranked risky lines | Tokens that drove each line | Top-k accuracy, recall@20% lines |
| Duplicate detection | Sentence-transformer embeddings in **pgvector (HNSW index)**, cosine similarity | Top-5 similar reports with scores | Shared key phrases highlighted | Recall@5, MAP, p95 latency |
| Severity & priority | TF-IDF + linear model as baseline; optional DistilBERT later. Labels from a **standard taxonomy** (for example blocker / critical / major / minor / trivial, and P1–P5). Abstains when text is too short. | Severity, priority, confidence | Top words/features | Macro-F1, confusion matrix |
| Resolver suitability | Features per (report, developer) pair: text similarity to the developer's past fixes, component ownership, recency. LightGBM ranker or logistic regression, trained chronologically on actual fixers. | Score per candidate | "Fixed 12 similar reports in *auth*, current load 2/5" | Top-1 / top-3 / top-5 accuracy, MRR |
| **Workload-aware assignment** (novel) | **OR-Tools** min-cost flow or CP-SAT. Maximise total suitability, subject to: each defect gets at most one developer, and each developer's new load ≤ capacity − current load. Unassignable defects are held and the shortfall reported. | Batch assignment | Per assignment: suitability and capacity used | Compare with greedy top-1: total suitability, max load, load std/Gini, % over capacity |
| Cold-start developers | Prior from declared skills/components, marked low confidence | Candidate score | "New developer; matched on declared skill *X*" | Share of assignments to new developers |
| Resolution forecast | **Survival analysis**, so unresolved (censored) defects are used too. Kaplan–Meier by severity as baseline, then Cox PH or a random survival forest. | Probability curve, median, P90, at-risk flag | Main drivers | Concordance index, calibration |
| Counterfactual guidance | Search over **actionable** features only (size, files touched, splitting a commit) for the smallest change that drops risk below the threshold | "Splitting this into 2 commits lowers risk from 78% to 41%" | The suggestion is the explanation | Validity rate |
| Drift | PSI per feature, comparing recent inputs with training data. Alert when PSI > 0.2 (configurable). | Drift alert | Which features drifted and by how much | — |
| Fairness | Compare predicted risk and false-positive rate across tenure buckets (for example < 6 months vs ≥ 6 months) | Fairness report per evaluation | — | Disparity ratio or difference |

**Datasets.** Verify availability and licence before use.
- **End-to-end demo:** mine one or two real open-source repositories that have linked issues (small to medium size, so mining finishes in reasonable time).
- **Commit-risk evaluation at scale:** a public JIT dataset such as **ApacheJIT**.
- **Duplicates, severity and triage at scale:** public Bugzilla datasets such as the Eclipse/Mozilla duplicate bug report data. These are also how C2 is tested at 300k reports.
- **Demo seeding:** `scripts/seed_demo.py` creates realistic demo users (one per role), developers, reports and PRs, so the UI never looks empty.

---

## 9. Frontend pages

The UI should be clean, calm and simple. Every automated decision is shown inside a shared `<DecisionCard>` component, whose `explanation` prop is **required by TypeScript** (C3). Risk is always shown with a **text label** as well as colour. Use sentence case and plain words.

| Page | Main roles | Stories |
|---|---|---|
| Login | all | US-49 |
| Home (role-based dashboard) | all | — |
| Repositories: register, mining status, settings | Admin, ML Eng | US-01, US-02, US-05, US-07, US-47 |
| Review queue: open PRs ranked by risk | Reviewer | US-09, US-10 |
| PR detail: score, confidence, factors, diff with highlighted lines, false-alarm button, "how to lower risk" | Developer, Reviewer | US-08, US-13–15, US-34–36 |
| Historical commit risk lookup | QA | US-12 |
| New defect report form with live duplicate suggestions | Reporter | US-16 |
| Report detail: triage suggestion, accept/override, candidates, forecast chart | Triager, Reporter | US-18, US-21, US-23, US-24, US-29, US-31, US-32 |
| Triage queue: batch assign, merge duplicates, at-risk flags | Triager | US-17, US-20, US-26, US-33 |
| My assignments: why me, raise objection | Developer | US-27 |
| Analytics: risk trends, acceptance rate, defect hotspots, workload distribution, fairness | Manager, QA | US-28, US-37–40, NFR-US-11 |
| Models: versions, metrics, drift alerts, retrain, shadow comparison, rollback, experiment export | ML Eng | US-41, US-43–45 |
| Admin: users and roles, thresholds, merge blocking, GitHub integration | Admin | US-47, US-49 |

---

## 10. The phases

Phases 0–4 = **Release 1** (a working risk check on PRs).
Phases 5–8 = **Release 2** (line-level risk and the full triage path).
Phases 9–10 = **Release 3** (forecasting, analytics, model lifecycle, hardening).

---

### Phase 0 — Foundations and guardrails

**Goal:** an empty but professional skeleton that runs with one command and has every quality gate in place.

- Create the monorepo structure from section 6, plus `CLAUDE.md` (a short summary of the rules and stack for future sessions), `docs/PROGRESS.md` and `docs/architecture.md`.
- `docker-compose.yml` with all six services and health checks. Add `.env.example` and a `Makefile`.
- **Backend:** FastAPI app with `/health`, config via pydantic-settings, and structured logging.
- **Frontend:** Next.js app with Tailwind + shadcn/ui, a layout shell and a health page that calls the API.
- **Tooling:**
  - ruff, mypy, ESLint, Prettier and pytest-cov
  - pre-commit hooks, including **gitleaks**
  - a GitHub Actions workflow running lint, type checks, tests and coverage, plus secret scanning that **fails the build on a secret** (C6)
- Add the pytest `story` marker and a small script that lists which stories have tests.

**Done when:** `docker compose up` shows the frontend talking to the API, and CI is green. **Stories:** US-50, NFR-US-09, NFR-US-10.

---

### Phase 1 — Core data, auth and roles

**Goal:** users can log in with a role, and admins can register repositories.

- SQLAlchemy models and Alembic migrations for **all** entities in section 7. The later phases fill them in.
- JWT auth (login, refresh, logout) and bcrypt password hashing.
- An **RBAC dependency**, `require_role(...)`, used on every protected route.
- `seed_demo.py` creates one demo user per role.
- Repository CRUD with a configuration screen (US-01).
- Settings stored in `SystemConfig`, editable only by Admin (US-49).
- Audit log for admin actions.
- **Frontend:** login page, role-aware navigation, Repositories page and Admin → Users page.
- **Tests:** auth flows, role checks (a forbidden role gets `403`), migrations run cleanly.

**Stories:** US-01, US-49.

---

### Phase 2 — Repository mining and training corpus

**Goal:** turn a repository's history into labelled training data.

- A mining job (RQ worker) using PyDriller that stores commits, authors and per-file changes (US-02).
- **Checkpointing:** an interrupted run resumes from its last checkpoint (US-05). Handle GitHub API rate limits with backoff.
- **Issue linking:** match fixing commits to issues through references like `#123` and `fixes #123`, and store the links (US-03).
- **SZZ labelling:** mark bug-inducing commits and ignore whitespace/comment-only changes (US-04). Document known SZZ limitations in `docs/ml.md`.
- **Feature computation** for every commit (US-06).
- **Incremental ingestion** of newly pushed commits (US-07).
- Map git identities to `Developer` records (several emails can belong to one developer).
- **Frontend:** mining status with progress, resume button and error display.
- **Tests:** SZZ on a tiny synthetic git repo built inside the test (with known answers), feature values, checkpoint/resume.

**Stories:** US-02 to US-07.

---

### Phase 3 — Commit risk model, calibration and explanation

**Goal:** a trained, calibrated and explainable risk model, tracked in MLflow.

- A training pipeline with:
  - a **chronological split** (C9) and a **leakage test** proving no future information is used
  - class weighting
  - LightGBM, plus logistic regression as a baseline
  - a calibration step
- Log everything to MLflow: config, **seed**, **data version** (a hash of the training snapshot), metrics and artefacts. Register the best model as **champion**.
- `make reproduce RUN_ID=...` re-runs an experiment and checks metrics are within 0.5% (C4). Results missing a seed or data version are flagged (NFR-US-07).
- Export a complete experiment record (US-44).
- **Prediction service:** load the champion once, score in memory, and return probability, confidence and risk level (US-08, US-10).
- **Explanation service:** SHAP top-3 factors turned into plain sentences, for example "This change touches 14 files; changes this spread out are riskier." The same input must always give the same explanation (US-34, US-36).
- **Fairness report** by tenure, included in every evaluation (C10).
- API: `POST /predict/commit`, `GET /commits/{sha}/risk` (US-12).
- **Tests:** leakage test, reproducibility test, explanation-always-present test (C3), and a latency micro-benchmark for the scoring function.

**Stories:** US-08 (model part), US-10, US-12, US-34, US-36, US-42, US-44, NFR-US-07, NFR-US-11.

---

### Phase 4 — GitHub integration: results on the pull request (end of Release 1)

**Goal:** opening a PR produces a risk status check and a comment automatically.

- GitHub App setup guide in `docs/github-app.md`, with **least-privilege permissions**:
  - read: contents, pull requests, metadata
  - write: checks, pull request comments
- **Webhook endpoint:**
  - verify `X-Hub-Signature-256`
  - handle `pull_request` opened/synchronize
  - respond quickly and post a **pending** check
  - enqueue scoring
- **Worker:** fetch the PR commits and diff, compute features, score, explain, then update the check and **edit the existing comment** instead of adding a new one (US-46).
- **Failure handling:**
  - timeout → check stays "pending" (C1)
  - service down → "unavailable" and the merge is not blocked (C7)
  - no trained model → "no assessment available" with the reason (US-08 AC2)
- The merge-blocking toggle is configurable per repo and **off by default** (US-47).
- **Offline demo:** `scripts/replay_pr_events.py`, plus a fake Checks sink that the UI can display.
- **Frontend:** PR list and PR detail page showing score, confidence, factors and explanation.
- **Performance:** a k6/Locust script measuring p95 from event receipt to check posted. Report the numbers honestly in `PROGRESS.md`. If network fetches dominate, measure and report scoring latency separately and explain why.

**Stories:** US-08, US-11, US-46, US-47, NFR-US-01, NFR-US-04, NFR-US-05. **Release 1 demo:** open a PR (real or replayed) and see the risk plus explanation appear.

---

### Phase 5 — Line-level risk and the review queue

**Goal:** reviewers see which lines matter and can give feedback.

- A line-level model (JITLine-style), run **only when risk is high**. Show the top-N lines, with N configurable (US-13).
- A token-level reason for each highlighted line (US-14).
- Low-risk changes show "did not meet the threshold for line-level analysis" (US-13 AC2).
- A false-alarm button stores `Feedback` (US-15).
- **Review queue:** open PRs sorted by calibrated risk, showing change size (US-09).
- **Frontend:** diff viewer with highlighted, expandable lines, and the review queue page.
- Add the highlighted lines to the PR comment.

**Stories:** US-09, US-13, US-14, US-15.

---

### Phase 6 — Defect reports and duplicate detection

**Goal:** reporters stop filing duplicates, and triagers merge them quickly.

- Defect report CRUD. Embed each report on creation and store it in pgvector with an **HNSW index**.
- **Live suggestions** while typing: a debounced endpoint returns up to 5 similar reports (id, title, status) once the text reaches a minimum word count (US-16).
- Candidate duplicates listed on every new report, with the shared phrases highlighted (US-17, US-18).
- **Merge** a report into an original and notify the reporter asynchronously (US-20).
- **Index-rebuilding state:** while the index rebuilds, the API returns a clear "index rebuilding" status, never an empty list (C2).
- **Load test:** bulk-load ≥ 300k public bug reports and measure p95 search latency (C2, US-19).
- **Frontend:** new report form with the suggestion panel (opening a suggestion must not lose the typed text), report detail and triage queue.

**Stories:** US-16 to US-20, NFR-US-02.

---

### Phase 7 — Severity and priority classification

**Goal:** reports arrive partly classified, consistently.

- Define the standard taxonomy in one place (US-22).
- A classifier that suggests severity and priority with confidence. It **abstains** and asks the reporter for more information when the text is too short (US-21 AC2).
- Suggestions are visibly marked **"Automated"**. Accept or change in one click. Changes are stored as `Feedback` (US-23).
- A misclassification report showing which kinds of defects get corrected most often.
- **Frontend:** triage panel on report detail and bulk accept in the triage queue.

**Stories:** US-21, US-22, US-23.

---

### Phase 8 — Resolver recommendation and workload-aware assignment (end of Release 2)

**Goal:** the novel contribution, which gets the most care.

- **Suitability model** (section 8), trained chronologically. Show at least 3 ranked candidates, each with confidence, reason and current open-defect count (US-24).
- **No confident candidate** (for example, a component with no ownership history) → say so and route the report to the default triage owner (US-24 AC2).
- **Capacity-constrained batch assignment** with OR-Tools (US-25, US-26):
  - no developer exceeds their capacity
  - total suitability is maximised
  - when demand exceeds capacity, the remaining defects stay queued and the manager is shown the shortfall
- **Override with reason:** confirming without a reason is refused. The original recommendation, the override and the reason are stored as a training signal (US-29).
- **"Why me?"** view for developers, with an objection button (US-27).
- **Cold start** for new developers, based on declared skills and marked low confidence (US-30).
- **Workload distribution** chart for managers (US-28).
- **Research evaluation**, documented in `docs/ml.md`: compare the optimiser against greedy top-1 on total suitability, max load, load standard deviation/Gini, and % of developers over capacity, using historical batches. Include a small ablation (with and without the capacity constraint).
- **Tests:** the optimiser never violates capacity (property-based tests with Hypothesis), the shortfall is reported, and an override without a reason is rejected.

**Stories:** US-24 to US-30. **Release 2 demo:** file a report → duplicate check → severity suggested → batch assign under capacity → override with reason.

---

### Phase 9 — Resolution forecasting

**Goal:** honest forecasts expressed as probabilities over time.

- A survival model that **uses unresolved defects as censored data** (US-31 AC).
- Output a probability curve, with the **median and P90** always stated.
- A simple estimate for the reporter, for example "likely within 3–10 days" (US-32).
- **At-risk flag** when a defect is likely to exceed its expected window (US-33).
- **Frontend:** forecast curve chart on report detail and at-risk badges in the triage queue.

**Stories:** US-31, US-32, US-33.

---

### Phase 10 — Analytics, model lifecycle, guidance and hardening (end of Release 3)

**Goal:** a complete, trustworthy, demo-ready system.

- **Analytics:**
  - risk trends over time (US-38)
  - recommendation acceptance rate (US-39)
  - defect hotspots by module (US-40)
  - project-wide top risk factors (US-37)
  - fairness view (NFR-US-11)
- **Drift monitoring:** a scheduled PSI check raises an alert naming the drifted features and by how much (US-41).
- **Model lifecycle:**
  - retrain from feedback plus new data
  - **shadow mode**: the challenger scores silently next to the champion, with a comparison page (US-45)
  - promote, and **roll back** to any previous version (US-43)
- **Counterfactual guidance:** "what would lower the risk", or "no actionable suggestion, request extra review" (US-35).
- **Hardening:**
  - coverage ≥ 80% on the core pipeline (C5)
  - Playwright end-to-end tests for the two main flows
  - rerun the performance tests and record the results
  - a security pass: RBAC review, rate limiting on auth, dependency audit
- **Documentation:**
  - README with a one-command quickstart
  - architecture diagram, API docs (FastAPI OpenAPI) and ML model cards
  - a **demo script** for the viva
- **Optional, only if time allows and I approve:** a VS Code extension showing risk before commit (US-48).

**Stories:** US-35, US-37 to US-41, US-43, US-45, NFR-US-08, remaining NFRs.

---

## 11. Global Definition of Done (applies to every story)

A story is done only when **all** of these hold:

1. All its acceptance criteria pass, verified by automated tests where possible.
2. It complies with every applicable constraint in section 3.
3. Every threshold it states has been **measured**, not just assumed.
4. Linting and type checks pass, and the code is readable enough for another student to review.
5. Unit tests pass, integration tests cover the main path, and core coverage stays ≥ 80%.
6. CI is green on main with no manual steps.
7. **Model work only:** results are logged with config, seed and data version; evaluation is chronological; the leakage test passes.
8. **Decision work only:** an explanation is produced, and no code path returns a decision without one.
9. **Docs are updated:** `PROGRESS.md`, API docs, and any new setting or threshold.
10. **Traceability:** tests are tagged with the story ID.
11. It works in the demo (real GitHub or replay mode).
12. No known defect of severity *major* or above remains open.

---

## 12. Mini glossary (for readers new to this)

- **Pull request (PR):** a request to merge a set of code changes into the main code.
- **Webhook:** a message GitHub sends to BugFlow automatically when something happens, like a doorbell.
- **Diff:** the exact lines added and removed by a change.
- **SZZ:** a method that finds which past commit introduced a bug, by tracing a bug fix backwards with `git blame`.
- **Calibration:** adjusting model scores so that "70% risk" really means about 7 in 10 similar changes had bugs.
- **SHAP:** a technique that shows which inputs pushed a prediction up or down.
- **Embedding:** a list of numbers representing the meaning of a text, so similar texts have similar numbers.
- **pgvector / HNSW:** a Postgres extension, and an index type, for fast "find the most similar" searches.
- **Survival analysis:** statistics for predicting *how long until* something happens, which can also use cases that haven't finished yet.
- **Champion / challenger:** the model currently in use, and a newer model being tested against it.
- **Drift / PSI:** incoming data slowly becoming different from the training data, and a number that measures how different it is.

---

**Start now with Phase 0 only.** Write your plan first, then build, then stop and report.
