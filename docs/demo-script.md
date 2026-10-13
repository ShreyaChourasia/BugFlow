# Demo script (viva walkthrough)

All demo accounts share the password `bugflow-demo` (created by `make seed`).
Run `make up && make seed` first. Full detail on each phase, including real
measured numbers, is in `docs/PROGRESS.md`; this is the condensed path
through all three releases.

## 0. Setup (once)

```bash
make up
make seed
```

Open http://localhost:3000, http://localhost:8000/docs (API), and
http://localhost:5000 (MLflow) in three tabs.

## Release 1 — pre-merge risk (Phases 0-4)

1. Log in as `admin@bugflow.demo`. **Repositories** → register
   `https://github.com/benjaminp/six` → **Start mining** (finishes in under
   a minute).
2. `docker compose exec api python /app/scripts/train.py` — trains the
   commit-risk model.
3. `docker compose exec api python /app/scripts/train_line_risk.py` —
   trains line-level localisation.
4. Replay a PR offline (no GitHub needed):
   ```bash
   docker compose exec api python /app/scripts/replay_pr_events.py \
     --repository-id <id> --sha <a mined sha> --pr-number 101
   ```
5. Open the repository's pull request list → open PR #101. Point out:
   risk level + confidence + the plain-language explanation (C3), the
   riskiest-lines panel if risk cleared the threshold, and the posted
   comment at the bottom.
6. **Review queue** — every open PR across every repository, riskiest
   first.

## Release 2 — triage (Phases 5-9)

1. Log in as `reporter@bugflow.demo` → **Report a bug**. Type a title and
   description — live duplicate suggestions appear in the sidebar before
   submitting (US-16). Submit.
2. On the new report's detail page: candidate duplicates with shared
   phrases highlighted, and a "Triage" panel (train first if it shows
   "unavailable": `docker compose exec api python /app/scripts/train_triage.py`).
3. Log in as `triager@bugflow.demo` → **Triage queue**. Accept a suggestion
   in one click, or open a report and use "Change & save." Point out the
   "At risk" badge next to any overdue report.
4. Still on a report's detail page: the "Resolution forecast" chart
   (train first if needed: `.../scripts/train_forecast.py`) — the
   probability curve, median/P90 lines, and the plain "likely within X-Y
   days" sentence.
5. The "Resolver recommendations" panel: ≥3 ranked candidates with
   confidence and reason (train first: `.../scripts/train_resolver.py`).
   Try **Assign** with a blank reason (refused), then with one filled in.
6. **Triage queue** → select several unassigned reports → **Batch
   assign** — the capacity-constrained optimizer assigns what it can and
   reports any shortfall.
7. Log in as `developer@bugflow.demo` → **My assignments** — "why me" and
   an objection form.
8. Log in as `manager@bugflow.demo` → **Workload** — capacity vs. current
   load per developer.

## Release 3 — analytics, model lifecycle, guidance (Phase 10)

1. Log in as `manager@bugflow.demo` or `qa@bugflow.demo` → **Analytics**:
   risk trend, acceptance rate, defect hotspots, top project-wide risk
   factors, and the fairness view (risk disparity by contributor tenure,
   surfaced from the commit-risk model's own training-time report).
2. On a high-risk PR's detail page (from Release 1's demo): the "How to
   lower risk" panel — a concrete suggestion ("splitting this into smaller
   commits would lower risk from X% to Y%"), or an honest "no actionable
   suggestion, request extra review" when nothing would help enough.
3. Log in as `ml-engineer@bugflow.demo` → **Models**:
   - Every trained model's version/stage/metrics, grouped by task.
   - Click **Retrain** on any group — queues a background job (check the
     `worker` container logs or MLflow for the new run).
   - Train the commit-risk model a second time with a different outcome to
     get a challenger, then show the **shadow comparison** — the
     challenger has already been silently scoring every new commit
     alongside the champion.
   - **Run drift check now** — compares recent mined commits' features
     against what the champion actually trained on; any feature past the
     PSI threshold shows up as an alert.
   - Pick an older (archived) model version and click **Make champion** —
     an instant rollback, no retraining needed.
4. Point out the login rate limiter (`docs/architecture.md`'s security
   section) and the CI pipeline's `pip-audit`/`npm audit`/`gitleaks` gates.

## If something shows "unavailable"

Every automated feature in BugFlow says so explicitly instead of a blank
screen or a crash — "unavailable" almost always means "train this model
first" (the script name is in the panel or in this script's matching
step above).
