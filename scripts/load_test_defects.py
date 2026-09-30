#!/usr/bin/env python3
"""C2/US-19: bulk-loads a large number of defect reports and measures p95
search latency against the real pgvector HNSW index.

The master prompt asks for "≥300k public bug reports"; this project has no
verified, licensed 300k+ real bug-report corpus available to download, so
this generates synthetic-but-realistic title/description text at that scale
instead (see docs/decisions/004-synthetic-load-test-data.md). What's being
proven is the same thing either way: does pgvector's HNSW index keep search
latency low as the table grows to hundreds of thousands of rows.

Usage (inside the api container, which has `app` installed):
    docker compose exec api python /app/scripts/load_test_defects.py --count 300000
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from bugflow_ml.embeddings.duplicate_detection import embed_texts  # noqa: E402
from sqlalchemy import insert, select  # noqa: E402

from app.core.db import SessionLocal  # noqa: E402
from app.models.defect import DefectReport  # noqa: E402
from app.models.enums import Role  # noqa: E402
from app.models.repository import Repository  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services.defect_service import _search_similar  # noqa: E402

COMPONENTS = [
    "login",
    "checkout",
    "search",
    "dashboard",
    "upload",
    "export",
    "notifications",
    "settings",
    "billing",
    "api",
    "sync",
    "onboarding",
]
SYMPTOMS = [
    "crashes",
    "hangs indefinitely",
    "throws an unhandled error",
    "returns the wrong result",
    "is extremely slow",
    "does not respond",
    "shows a blank page",
    "loses unsaved data",
    "times out",
    "fails silently",
    "shows a null pointer exception",
    "raises a permission error",
]
CONTEXTS = [
    "on Chrome",
    "on Firefox",
    "on mobile Safari",
    "after a recent deploy",
    "under heavy load",
]


def _generate_reports(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    reports = []
    for i in range(count):
        component = rng.choice(COMPONENTS)
        symptom = rng.choice(SYMPTOMS)
        context = rng.choice(CONTEXTS)
        title = f"{component} {symptom}"[:500]
        description = (
            f"When using the {component} feature, it {symptom} {context}. "
            f"Steps to reproduce: open {component}, perform the usual action, observe the issue. "
            f"Report #{i}."
        )
        reports.append({"title": title, "description": description})
    return reports


def _get_or_create_fixtures() -> tuple[int, int]:
    db = SessionLocal()
    try:
        repo = db.scalar(select(Repository).where(Repository.name == "load-test-repo"))
        if repo is None:
            repo = Repository(name="load-test-repo", url="/tmp/load-test-repo")
            db.add(repo)
            db.commit()
            db.refresh(repo)

        reporter = db.scalar(select(User).where(User.role == Role.REPORTER.value))
        if reporter is None:
            raise RuntimeError("No reporter user exists yet — run scripts/seed_demo.py first.")
        return repo.id, reporter.id
    finally:
        db.close()


def bulk_load(count: int, batch_size: int, seed: int) -> None:
    repository_id, reporter_id = _get_or_create_fixtures()
    reports = _generate_reports(count, seed)

    db = SessionLocal()
    try:
        start = time.perf_counter()
        for batch_start in range(0, len(reports), batch_size):
            batch = reports[batch_start : batch_start + batch_size]
            embeddings = embed_texts([f"{r['title']}\n{r['description']}" for r in batch])
            db.execute(
                insert(DefectReport),
                [
                    {
                        "repository_id": repository_id,
                        "reporter_id": reporter_id,
                        "title": r["title"],
                        "description": r["description"],
                        "embedding": embedding,
                    }
                    for r, embedding in zip(batch, embeddings, strict=True)
                ],
            )
            db.commit()
            done = batch_start + len(batch)
            elapsed = time.perf_counter() - start
            print(f"  {done}/{count} loaded ({done / elapsed:.0f} rows/sec)", flush=True)
        print(f"Loaded {count} reports in {time.perf_counter() - start:.1f}s")
    finally:
        db.close()


def measure_p95(trials: int) -> None:
    db = SessionLocal()
    try:
        # Embed the queries up front so the timed loop below measures only
        # the DB-level ANN search itself, not embedding time.
        query_texts = [f"{r['title']}\n{r['description']}" for r in _generate_reports(trials, 999)]
        query_embeddings = embed_texts(query_texts)

        latencies_ms = []
        for embedding in query_embeddings:
            start = time.perf_counter()
            _search_similar(db, embedding, exclude_id=None, limit=5)
            latencies_ms.append((time.perf_counter() - start) * 1000)

        latencies_ms.sort()

        def pct(p: float) -> float:
            return latencies_ms[int(len(latencies_ms) * p)]

        print(f"Search latency over {trials} queries against the live HNSW index:")
        print(f"  p50: {pct(0.50):.2f}ms")
        print(f"  p95: {pct(0.95):.2f}ms")
        print(f"  p99: {pct(0.99):.2f}ms")
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=200)
    parser.add_argument("--skip-load", action="store_true", help="only measure p95, don't reload")
    args = parser.parse_args()

    if not args.skip_load:
        bulk_load(args.count, args.batch_size, args.seed)
    measure_p95(args.search_trials)


if __name__ == "__main__":
    main()
