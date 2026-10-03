#!/usr/bin/env python3
"""Trains the resolver suitability model (US-24/US-30) on every defect a
developer has actually resolved, falling back to a documented synthetic
bootstrap set (see docs/decisions/006) when there isn't enough real
resolution history yet.

Usage (inside the api container, which has `app` installed):
    docker compose exec api python /app/scripts/train_resolver.py [--seed 42]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.db import SessionLocal  # noqa: E402
from app.services.resolver_training_service import train_and_register_champion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        model = train_and_register_champion(db, seed=args.seed)
        print(f"Used bootstrap data: {model.metrics.get('used_bootstrap_data')}")
        print(f"resolver_suitability: run {model.mlflow_run_id} — stage: {model.stage}")
        for key, value in model.metrics.items():
            print(f"  {key}: {value}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
