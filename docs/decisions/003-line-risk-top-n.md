# 003 — `line_risk_top_n` on `Repository`, and reusing `risk_threshold`

## Status
Accepted (Phase 5).

## Context
US-13 requires line-level analysis to run "only when risk is high" and to
show "the top-N lines, with N configurable." §7's `Repository` already has
`risk_threshold` (added Phase 1, used since Phase 4 for merge-blocking) but
nothing for N.

## Decision
- Add `Repository.line_risk_top_n` (default 5) as the configurable N.
- Reuse the *existing* `risk_threshold` as the bar for "is this commit high
  enough risk to bother with line-level analysis" — no second threshold.

## Why
- `risk_threshold` already means "the calibrated probability above which
  this repo considers a change high-risk." Line-level analysis needing its
  own, separate notion of "high risk" would be a second knob answering the
  same question, configurable independently for no stated reason — the kind
  of speculative flexibility YAGNI argues against. A repo can still leave
  `merge_blocking_enabled` off while `risk_threshold` alone decides when
  line highlights show up.
- `line_risk_top_n` genuinely is a new, independently-meaningful setting
  (how many lines to show, not whether to show any) — that's the one the
  master prompt explicitly calls out as needing to be configurable.

## Consequences
- Lowering `risk_threshold` on a repo now affects two things: the
  merge-blocking conclusion *and* whether line-level analysis runs. This is
  intentional — both are proxies for the same underlying question ("is this
  change risky enough to warrant more scrutiny").
