# Model card: resolver suitability (`resolver_suitability`)

**Purpose.** Ranks candidate developers for a defect report, feeding both
the single-report "resolver recommendations" panel and the project's novel
contribution — capacity-constrained batch assignment (OR-Tools; not a
trained model, see `docs/ml.md`'s dedicated section).

**Inputs.** Per (report, developer) pair: text similarity to the
developer's past resolutions, component match count, a recency-decayed
signal, declared skill match, current load ratio.

**Architecture.** `LogisticRegression` over those five numeric features,
trained **only** on candidates with real resolution history. A candidate
with no history never gets a model prediction at all — a fixed, low-
confidence, skill-based prior instead (US-30's cold start, a permanent
design feature, not a stand-in for missing data).

**Outputs.** A ranked candidate list, each with a score, confidence, a
plain-language reason ("fixed N similar reports in *component*, current
load X/Y"), and current open-defect count. Below a confidence floor, the
UI says so and names the configured default triage owner instead of
guessing (US-24 AC2).

**Training data.** Real resolved defects. Falls back to a documented
synthetic bootstrap set below 40 real examples — deliberately ambiguous
(multiple developers can share a skill/component), so the reported ranking
metrics describe a genuinely non-trivial problem. See
`docs/decisions/006-resolver-bootstrap-data.md`.

**Metrics.** Top-1/3/5 accuracy, MRR. Full numbers: `docs/ml.md`.

**Known limitations.** Uses each developer's *current* capacity/queue
depth for every historical training example, not a snapshot of what it
was at the time — this project doesn't retain that history.

**Intended use.** A ranked suggestion a triager/manager can accept or
override with a required reason (US-29) — the override itself becomes a
training signal for next time.
