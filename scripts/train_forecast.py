#!/usr/bin/env python3
"""Trains the resolution-forecast survival model (US-31/US-32/US-33) on
every defect report with a severity and priority set — resolved reports as
real events, still-open reports as censored observations — falling back to
a documented synthetic bootstrap set (see docs/decisions/007) when there
isn't enough real history yet.

Usage (inside the api container, which has `app` installed):
    docker compose exec api python /app/scripts/train_forecast.py [--seed 42]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.db import SessionLocal  # noqa: E402
from app.services.forecast_training_service import train_and_register_champion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        model = train_and_register_champion(db, seed=args.seed)
        print(f"Used bootstrap data: {model.metrics.get('used_bootstrap_data')}")
        print(f"resolution_forecast: run {model.mlflow_run_id} — stage: {model.stage}")
        for key, value in model.metrics.items():
            if key not in {"feature_columns", "component_categories"}:
                print(f"  {key}: {value}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
