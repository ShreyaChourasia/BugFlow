# 006 — Synthetic bootstrap data for the resolver suitability model

## Status
Accepted (Phase 8).

## Context
The resolver suitability model (§8) learns from `(report, developer)` pairs
where the developer actually resolved that report — `DefectReport.assignee_id`
+ `resolved_at` + `embedding` all set. Like Phases 6 and 7, there's no
verified, licensed dataset of real assignment history available in this
environment, and a fresh deployment has zero resolved defects, so the same
chicken-and-egg problem applies: no training data until the feature has been
used, nothing to recommend until it's been trained.

This is a distinct problem from US-30's cold start, which is not a data
availability gap — the spec explicitly requires a cold-start path (a fixed,
low-confidence, skill-based prior) for any individual developer with no
personal resolution history, *regardless* of whether the model overall was
trained on bootstrap or real data. That mechanism (`score_candidates()`'s
`has_history=False` branch) is permanent production behavior, not a
stand-in for missing data, and is unaffected by this ADR.

## Decision
`resolver_training_service.train_and_register_champion()` always prefers
real data: every resolved `DefectReport` with a non-null assignee, embedding,
and `resolved_at` becomes a labelled example for the developer who resolved
it. Only when fewer than `MIN_TRAINING_EXAMPLES` (40) such examples exist
does it fall back to `generate_bootstrap_history()` — a synthetic set of 10
developers resolving 200 reports chronologically, with component/skill
correlation (so ranking is genuinely testable, not a trivial baseline) and
deliberate ambiguity (multiple developers can share a skill/component, since
real teams rarely have one unique owner per area).

## Why
- Same reasoning as ADR 005: real data unconditionally preferred, synthetic
  data only a gap-filler, so the model improves as real assignments
  accumulate instead of being anchored to synthetic text forever.
- `MIN_TRAINING_EXAMPLES = 40` is higher than triage's 25 because this model
  trains on 5 numeric features per candidate (not TF-IDF text), and with 10
  developers per report the class imbalance (1 true resolver out of ~10
  candidates) needs more examples before `LogisticRegression` has enough
  signal to separate "resolved it" from "didn't."
- Deliberately ambiguous synthetic data keeps the reported bootstrap ranking
  metrics (Top-1 ≈ 35%, Top-3 ≈ 82.5%, MRR ≈ 0.59) honest about what a
  10-way ranking problem with overlapping skills actually looks like, rather
  than engineering an artificially separable set that would overstate
  real-world performance.

## Consequences
- Once ≥40 real resolved-defect examples exist, the bootstrap set stops
  being used automatically, with no migration step.
- `MLModel.metrics` records a `used_bootstrap_data` flag so it's visible
  which case produced a given champion, same gap `MLModel.metrics` has for
  the triage classifier (noted in ADR 005's consequences, still unaddressed
  there) — worth a shared fix if this distinction becomes load-bearing for
  more than one model.
- Bootstrap-trained rankings describe the synthetic set's correlations, not
  real team dynamics — acceptable for demoing the mechanism end-to-end, not
  for judging real resolver-matching quality until real data takes over.
