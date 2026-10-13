# Model card: duplicate detection (embeddings, not a trained classifier)

**Purpose.** Surfaces similar existing reports while a reporter is still
typing (US-16), and candidate duplicates on a saved report's detail page
(US-17/18).

**Inputs.** Report title + description text.

**Architecture.** `all-MiniLM-L6-v2` sentence-transformer embeddings
(384-dim, fixed, not fine-tuned), stored in pgvector and searched via an
HNSW index with cosine distance. Not a trained/champion-tracked model —
there's no `MLModel` row for this task, since the embedding model itself
never changes; only the index and the reports in it grow.

**Outputs.** Top-5 similar reports with a similarity score and the actual
shared phrases highlighted (via stdlib `difflib`, not another model).

**Metrics.** Recall@5, MAP, and p95 search latency — measured at 300k
reports via `scripts/load_test_defects.py` against C2's 300ms budget. Full
numbers: `docs/ml.md`.

**Known limitations.** A fixed, general-purpose sentence embedding, not
fine-tuned on this project's own bug-report domain — two reports that are
semantically similar but phrased very differently may not surface as
duplicates.

**Intended use.** A candidate list for a human to confirm (merge) or
dismiss, never an automatic merge.
