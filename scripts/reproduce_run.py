#!/usr/bin/env python3
"""C4: re-runs a recorded experiment and checks its headline metrics
reproduce within 0.5%. Exits non-zero if anything doesn't reproduce.

Usage:
    docker compose exec api python /app/scripts/reproduce_run.py --run-id <mlflow_run_id>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.db import SessionLocal  # noqa: E402
from app.services.training_service import reproduce_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = reproduce_run(db, args.run_id)
    finally:
        db.close()

    if result["reproducible"]:
        print(f"Run {args.run_id} reproduces within 0.5%.")
        return 0

    print(f"Run {args.run_id} did NOT reproduce.")
    if "reason" in result:
        print(f"  {result['reason']}")
    for mismatch in result.get("mismatches", []):
        print(
            f"  {mismatch['metric']}: original={mismatch['original']:.4f} "
            f"reproduced={mismatch['reproduced']:.4f} "
            f"(relative diff {mismatch['relative_diff']:.2%})"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
