# Model card: resolution forecast (`resolution_forecast`)

**Purpose.** Forecasts how long a defect report is likely to take to
resolve, as a full probability curve — not just a point estimate.

**Inputs.** Severity, priority, and component only.

**Architecture.** Kaplan-Meier fit per severity group (baseline) and Cox
proportional hazards (main model). Every `DefectReport` with a severity and
priority set is an example — a resolved report contributes a real duration
and event; a **still-open** report contributes a censored duration (US-31
AC) — dropping unresolved reports would systematically bias the forecast
toward whatever happens to finish quickly.

**Outputs.** A full probability-vs-time curve, median and P90 (always
stated as concrete numbers — lifelines' `+inf` for an unreached percentile
is clipped, never surfaced as infinity), a plain "likely within X-Y days"
sentence for the reporter, and an at-risk flag once a still-open report's
elapsed time passes its own P90.

**Training data.** Real defect history (resolved + still-open). Falls back
to a documented synthetic bootstrap set below 50 real examples — a genuine
Weibull *proportional-hazards* process (not just "higher severity picks a
smaller random number"), so the reported concordance measures a well-
specified problem. See `docs/decisions/007-resolution-forecast-bootstrap-data.md`.

**Metrics.** Concordance index (Cox vs. the Kaplan-Meier baseline). Full
numbers, including a real modeling pitfall found and fixed during
development: `docs/ml.md`.

**Known limitations.** Only sees severity/priority/component — not
assignee workload at report time or reporter history (not retained/not in
the data model, respectively).

**Intended use.** An honest expectation-setting tool for reporters and
triagers, not a committed SLA — the at-risk flag is a prompt to check in,
not an automatic escalation.
