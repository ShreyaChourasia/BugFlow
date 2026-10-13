# Model card: severity & priority classification (`triage_severity`, `triage_priority`)

**Purpose.** Suggests a standard severity (blocker/critical/major/minor/
trivial) and priority (P1-P5) for a newly filed defect report.

**Inputs.** The report's title + description text only.

**Architecture.** TF-IDF + `LogisticRegression`, one independently
champion-tracked model per target (severity, priority) — they can regress
or improve independently. Abstains (returns no suggestion, asks for more
detail) below a plain word-count floor, before any model even loads.

**Outputs.** Label + confidence + top contributing words per target,
always marked "Automated" in the UI, one-click accept or change (US-23).

**Training data.** Every `DefectReport` a human has actually set
severity/priority on (directly, or by accepting/overriding a suggestion).
Falls back to a documented synthetic, keyword-correlated bootstrap set
below 25 real examples — see `docs/decisions/005-triage-bootstrap-data.md`.
Always switches back to real data automatically once enough exists.

**Metrics.** Macro-F1 and a confusion matrix per target. Full numbers:
`docs/ml.md`.

**Known limitations.** No access to the standard public Bugzilla
triage datasets (Eclipse/Mozilla) the master prompt names for at-scale
evaluation — not verified available/licensed in this environment (same
constraint `docs/decisions/004` documents for the duplicate-detection load
test).

**Intended use.** A starting suggestion for a human triager, never a final
decision — every accept/override is recorded as `Feedback`, which the
misclassification report reads to show which suggested values get
corrected most often.
