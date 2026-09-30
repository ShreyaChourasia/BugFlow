# 002 — Track current check/comment state directly on `PullRequest`

## Status
Accepted (Phase 4).

## Context
US-46/US-47 and C1/C7 require the system to track *what was last posted* to
a PR's status check and comment — pending vs. completed, its conclusion,
and the text shown — both to decide whether to create vs. edit next time,
and so the offline "fake Checks sink" has something for the UI to display
without a live GitHub connection. §7's `PullRequest` only has `check_run_id`
and `comment_id` (opaque IDs), not their content or state.

## Decision
Add four columns to `PullRequest`: `check_status` (`pending` | `completed`),
`check_conclusion` (`success` | `failure` | `neutral`, null while pending),
`check_summary`, `comment_body`.

## Why
- These are the *current write-back state*, analogous to the existing
  `check_run_id`/`comment_id` — not a new concept, just the two fields
  needed to act on that state (know whether to re-check it later, and what
  to show in the UI) that weren't in the original table.
- No separate history table: nothing in this phase needs *every* past
  check transition, only the current one. `RiskPrediction`/`Explanation`
  already retain the full scoring history per commit if that's ever needed.
- This doubles as the fake Checks sink for offline/replay demos: whether a
  real GitHub API call happened or not, these four columns are what the UI
  reads to show "what got posted."

## Consequences
- If a later phase needs full check-run history (e.g. an audit trail of
  every re-score), add a dedicated log table then — these columns aren't
  designed to hold more than "the latest."
