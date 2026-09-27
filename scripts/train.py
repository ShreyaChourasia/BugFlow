#!/usr/bin/env python3
"""Trains the commit-risk model on every mined commit and logs a complete
experiment to MLflow, promoting it to champion if it's the best so far.

Usage (inside the api container, which has `app` installed):
    docker compose exec api python /app/scripts/train.py [--seed 42]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.db import SessionLocal  # noqa: E402
from app.services.training_service import train_and_register_champion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        model = train_and_register_champion(db, seed=args.seed)
        print(f"Run {model.mlflow_run_id} — stage: {model.stage}")
        for key, value in model.metrics.items():
            if key != "fairness":
                print(f"  {key}: {value}")
        print(f"  fairness: {model.metrics.get('fairness')}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
