# Model card: commit risk (`commit_risk`)

**Purpose.** Predicts how likely a commit is to introduce a bug (just-in-time
defect prediction), posted on pull requests before merge.

**Inputs.** Change metrics only: size/churn, files/directories/subsystems
touched, change entropy, author's prior commit count, whether it's tagged as
a fix. No source code content, no author identity beyond commit count.

**Architecture.** LightGBM (class-weighted for imbalance), isotonic-
calibrated, vs. a logistic-regression baseline. Chronological train/calib/
test split (C9) — never trained on the future to predict the past.

**Outputs.** Calibrated probability, confidence, a low/medium/high label,
and the top-3 SHAP factors as plain sentences (C3 — never shown without
one). Also feeds US-35's counterfactual search ("what would lower the
risk") over the same actionable features, and Phase 10's shadow mode
(the current challenger scores every commit silently alongside the
champion).

**Training data.** Real mined commit history for any registered,
mined repository — no synthetic bootstrap (unlike the triage/resolver/
forecast models), since Phase 2's mining pipeline is the project's own
real-data source from day one.

**Metrics.** ROC-AUC, PR-AUC, F1, Brier score, recall@20%-effort; the
champion/challenger gate promotes on PR-AUC. Full numbers: `docs/ml.md`.

**Fairness.** Every training run reports mean predicted risk and the
disparity between contributors with <6 months and ≥6 months of tenure
(C10) — surfaced project-wide on the `/analytics` page (NFR-US-11).

**Known limitations.**
- No code-content features (tokens, AST, diff semantics) — purely
  structural/historical. A large but mechanically simple change (e.g. a
  generated file) scores the same as a hand-written one of the same size.
- Fairness/drift are both scoped to the project's actual mined history —
  a repository with few or no tenure-diverse contributors yields an
  "unavailable" fairness report, not a fabricated one.
- Drift monitoring (US-41) and shadow mode (US-45) are implemented for this
  model only — see `docs/decisions/008-shadow-mode-scope.md`.

**Intended use.** Advisory signal for reviewers and a merge-blocking gate
only if a repository opts in (`merge_blocking_enabled`) — never blocks
silently, and a scoring failure always surfaces as "pending," never a false
pass (C7).
