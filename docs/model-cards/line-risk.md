# Model card: line-level risk (`line_risk`)

**Purpose.** When a commit's overall risk clears the repository's
threshold, ranks the individual added lines most likely to be the risky
part — JITLine-style localisation, so a reviewer knows where to look.

**Inputs.** TF-IDF over tokenised added-line text, combined with the parent
commit's own risk features.

**Architecture.** A linear model over the combined feature space; per-line
token attribution (exact `coefficient × tfidf_value`, the same attribution
style triage's classifier uses) picks the words that drove each line's
score.

**Outputs.** Top-N ranked lines (N configurable per repository,
`line_risk_top_n`) with a risk score and a short token-based reason each. A
"mark as false alarm" feedback button per line.

**Training data.** Weakly labelled from mined commits: an added line is a
positive example if it falls inside a range SZZ attributes to a later bug
fix. Real data, no bootstrap fallback.

**Metrics.** Top-k accuracy, recall@20%-lines. Full numbers: `docs/ml.md`.

**Known limitations.** Only runs when the parent commit's calibrated risk
clears the threshold — below it, line analysis is skipped entirely (by
design, not a failure) and the UI says so explicitly rather than showing
an empty list.

**Intended use.** A reviewer's starting point for where to focus attention
on a flagged PR, not a standalone severity judgement.
