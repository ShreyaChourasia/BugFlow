# 007 — Synthetic bootstrap data for the resolution-forecast survival model

## Status
Accepted (Phase 9).

## Context
The resolution-forecast model (§8) learns from every `DefectReport`'s
elapsed time: resolved reports contribute a real duration (`resolved_at -
reported_at`), still-open reports contribute a **censored** duration (now -
`reported_at`, with "hasn't happened yet" as the label) — US-31 AC
explicitly requires both. §8 names no specific public resolution-time
dataset the way it does for severity/duplicate detection (Bugzilla/Eclipse
data), and no verified, licensed one is available in this environment
regardless — the same constraint ADRs 004-006 already document for other
models. A fresh deployment also has zero resolved defects, so the usual
chicken-and-egg problem applies: nothing to forecast from until the system
has been used for a while.

## Decision
`forecast_training_service.train_and_register_champion()` always prefers
real data: every `DefectReport` with both severity and priority set becomes
an example, resolved or not. Only when fewer than `MIN_TRAINING_EXAMPLES`
(50) such examples exist does it fall back to `generate_bootstrap_history()`
— 250 synthetic reports with a resolution time that genuinely shortens as
severity/priority increase (so the concordance-index evaluation is a
meaningful, non-trivial number), with ~25% of cases deliberately left
censored (still "open" at a random earlier cutoff) so the bootstrap set
exercises the censored-data path too, not just the resolved-case path.

## Why
- Same reasoning as ADRs 005/006: real data unconditionally preferred,
  synthetic data only a gap-filler, so the model's forecasts improve as real
  resolution history accumulates instead of being anchored to synthetic
  durations forever.
- `MIN_TRAINING_EXAMPLES = 50` is higher than triage's 25: Cox PH fits a
  handful of numeric/categorical features (severity rank, priority rank,
  one-hot component) against *continuous* durations with censoring, which
  needs more observed events than a discrete multi-class classifier to
  converge to a stable, non-overfit coefficient estimate.
- Genuinely correlating severity/priority with simulated resolution time
  (rather than assigning random durations) keeps the reported concordance
  index honest about what the model is actually discriminating, instead of
  reporting a number that would be equally true of a random model.

## Consequences
- Once ≥50 real labelled reports exist, the bootstrap set stops being used
  automatically, with no migration step.
- `MLModel.metrics` records a `used_bootstrap_data` flag, same as ADRs
  005/006's models.
- Bootstrap-trained forecasts describe the synthetic set's severity/priority
  correlation, not real resolution-time dynamics — acceptable for demoing
  the mechanism (probability curve, median, P90, at-risk flag) end-to-end,
  not for judging real forecast accuracy until real resolution history
  takes over.
- The model only ever sees `severity`, `priority`, and `component` as
  features — not, say, assignee workload at report time or reporter
  history, since this project doesn't retain the former as a historical
  snapshot (noted already in `resolver_training_service.py`'s own
  `ponytail:` comment) and the latter isn't in the §7 data model. A known
  ceiling, not a bug.
