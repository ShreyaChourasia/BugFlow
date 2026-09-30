# 004 — Synthetic data for the ≥300k-report load test

## Status
Accepted (Phase 6).

## Context
US-19/C2 asks for a load test that "bulk-loads ≥300k public bug reports and
measures p95 search latency." This project has no verified, licensed 300k+
real public bug-report corpus available to download and use here, and the
master prompt explicitly warns against inventing facts about datasets.

## Decision
`scripts/load_test_defects.py` generates 300k+ synthetic-but-realistic
title/description pairs (templated from a small vocabulary of components,
symptoms, and contexts) instead of a real public dataset, embeds them with
the same `bugflow_ml.embeddings.duplicate_detection.embed_texts` path
production uses, bulk-inserts them, and measures p50/p95/p99 search latency
against the live pgvector HNSW index.

## Why
- What C2/US-19 is actually testing is whether the search stays fast as the
  table grows to real scale — that's a property of the index and the data
  *volume*, not of the text being real bug reports specifically.
- Synthetic data is reproducible (fixed seed), needs no licensing review, and
  doesn't require this session to claim access to a dataset it can't verify.
- The vocabulary is varied enough (12 components × 12 symptoms × 5 contexts)
  to avoid every row being embedding-identical, which would make the ANN
  index's job artificially easy in a way real, more varied text wouldn't be.

## Consequences
- The load test's numbers say "pgvector/HNSW scales to 300k+ rows on this
  hardware," not "this specific duplicate-detection model is accurate on
  real-world bug reports at scale" — the latter would need real data and is
  out of scope here (same caveat Phase 3's commit-risk model already carries
  for evaluation on a single small real repo).
- If a real public bug-report dataset becomes available later, swapping the
  generator in `load_test_defects.py` for a bulk loader of that dataset
  wouldn't change anything else — the embedding, indexing, and measurement
  code is dataset-agnostic.
