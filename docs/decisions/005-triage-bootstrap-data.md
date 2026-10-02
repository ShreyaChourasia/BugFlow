# 005 — Synthetic bootstrap data for the severity/priority classifier

## Status
Accepted (Phase 7).

## Context
§8 suggests "public Bugzilla datasets such as the Eclipse/Mozilla duplicate
bug report data" for training severity/priority classification. As with
Phase 6's load test (ADR 004), there's no verified, licensed copy of one of
these datasets available in this environment.

Unlike the load test, though, this classifier is a real feature a demo needs
to show working — and `DefectReport.severity`/`priority` are null until a
human decides them, which this very feature exists to help with. Left alone,
that's a chicken-and-egg problem: no training data until the feature has
been used, and nothing to suggest until it's been trained.

## Decision
`triage_training_service.train_and_register_champions()` always prefers real
data: every `DefectReport` a human has actually set `severity`/`priority` on
(directly, or by accepting/overriding a suggestion — US-23) is training data.
Only when there are fewer than `MIN_TRAINING_EXAMPLES` (25) real labelled
reports does it fall back to `generate_bootstrap_examples()` — a small,
keyword-correlated synthetic set (§5.1 keywords like "data loss" → blocker,
"fix immediately" → P1) that exists purely to make the classifier
demoable before any real triage decisions exist.

## Why
- Preferring real data unconditionally, with the synthetic set only as a
  gap-filler, means the model actually improves as the system gets used,
  rather than being permanently anchored to synthetic text.
- The bootstrap set is keyword-correlated, not random, so the classifier it
  trains is genuinely testable (predictable inputs produce predictable
  outputs) rather than a trivial, meaningless baseline.
- `MIN_TRAINING_EXAMPLES = 25` is deliberately low relative to Phase 5's
  line-risk model's 50 — this is a 5-class problem per target already
  needing real per-class representation to train at all, and the whole point
  is to switch to real data as soon as there's barely enough of it.

## Consequences
- Every retrain re-checks real data first — once ≥25 real decisions exist,
  the bootstrap set stops being used automatically, with no separate
  migration step.
- Metrics (`macro_f1`, confusion matrix) logged while running on bootstrap
  data describe the synthetic set, not real-world accuracy — `MLModel.metrics`
  doesn't currently flag which case produced a given champion; worth adding
  if this distinction matters later (e.g. for the misclassification report).
