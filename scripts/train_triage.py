#!/usr/bin/env python3
"""Trains the severity and priority classifiers (US-21/US-22/US-23) on every
defect report a human has already decided on, falling back to a documented
synthetic bootstrap set (see docs/decisions/005) when there aren't enough
real decisions yet.

Usage (inside the api container, which has `app` installed):
    docker compose exec api python /app/scripts/train_triage.py [--seed 42]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.db import SessionLocal  # noqa: E402
from app.services.triage_training_service import train_and_register_champions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = train_and_register_champions(db, seed=args.seed)
        print(f"Used bootstrap data: {result['used_bootstrap_data']}")
        for name in ("severity_model", "priority_model"):
            model = result[name]
            print(f"{name}: run {model.mlflow_run_id} — stage: {model.stage}")
            for key, value in model.metrics.items():
                if key != "confusion_matrix":
                    print(f"  {key}: {value}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
