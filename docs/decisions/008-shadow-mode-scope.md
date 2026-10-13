# 008 — Shadow mode and drift monitoring scoped to the commit-risk model

## Status
Accepted (Phase 10).

## Context
§8/Phase 10 ask for two cross-cutting model-lifecycle features: **shadow
mode** (the challenger scores silently next to the champion, with a
comparison page, US-45) and **drift monitoring** (a scheduled PSI check per
feature, US-41). BugFlow has five trained models by Phase 10 (commit risk,
line risk, triage severity/priority, resolver suitability, resolution
forecast), each with its own feature set, training service, and prediction
service.

Building both mechanisms generically across all five models — a shared
shadow-scoring harness and a shared reference-distribution snapshot per
model — would multiply the surface area five-fold for a capstone feature
whose value is in demonstrating the *mechanism*, not in running it
everywhere at once.

## Decision
Shadow mode and drift monitoring are implemented for the **commit-risk
model only** — the flagship, pre-merge model this whole project is named
after, and the one C1's 500ms latency budget and NFR-US-01 already center
on. `RiskPrediction.is_shadow` (migration `a1c3f6d9e2b4`) distinguishes a
challenger's silent score from the champion's real decision on the exact
same commit. Drift reuses `reproduce_run()`'s existing pattern of
downloading the champion's `training_commits.json` artifact: those commit
ids are the reference distribution, and any commit mined since training is
the "current" window — no new reference-snapshot storage needed.

## Why
- Commit risk is scored automatically on every PR already (the worker job
  in `app/workers/jobs/pr.py`), so shadow-scoring it costs one extra
  in-memory inference per PR — negligible, and it's the one place a
  "which model is actually better in practice" comparison matters most
  (it's the model gating `merge_blocking_enabled`).
- Reusing the training-snapshot artifact for drift's reference distribution
  means zero new storage: the same commit ids `reproduce_run()` already
  downloads to re-verify reproducibility are exactly the "what the model
  was trained on" population PSI needs to compare against.
- The other four models' training services already report their own
  held-out metrics (concordance, Top-k/MRR, macro-F1, optimizer-vs-greedy)
  every time they're retrained — the lifecycle actions that matter most for
  them (retrain, promote, rollback) are generic and still apply to every
  task via the `/models` API; only the *comparison* and *drift* mechanisms
  are commit-risk-specific in this phase.

## Consequences
- The `/models` page's shadow-comparison and drift-alert sections only ever
  show data for `task=commit_risk`; the other four tasks list normally
  (version, stage, metrics, retrain, rollback) but show "not monitored for
  drift" / "no shadow comparison available" rather than a fabricated empty
  chart.
- A future phase extending shadow mode or drift to another model would
  follow the same two patterns established here (an `is_shadow` flag
  alongside that model's own prediction row type; its own training-snapshot
  artifact as the drift reference) rather than inventing a new approach.
- `DriftAlert.model_id` already pointed at `ml_models.id` generically, so no
  schema change was needed to make this generalizable later — only the
  service-layer wiring is commit-risk-specific right now.
