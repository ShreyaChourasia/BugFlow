#!/usr/bin/env python3
"""US-41: runs the commit-risk PSI drift check and prints any alerts it
raised. Meant to be run on a schedule (cron, or a CI scheduled workflow) —
see docs/decisions/008-shadow-mode-scope.md for why this is a script rather
than an in-process scheduler, and why it's scoped to commit-risk only.

Usage (inside the api container, which has `app` installed):
    docker compose exec api python /app/scripts/check_drift.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.db import SessionLocal  # noqa: E402
from app.services.drift_service import run_drift_check  # noqa: E402


def main() -> None:
    db = SessionLocal()
    try:
        result = run_drift_check(db)
        if result.status == "unavailable":
            print("Drift check unavailable: no champion trained yet, or no new commits to check.")
            return
        print(f"Checked {result.checked_features} features.")
        if not result.alerts:
            print("No drift detected.")
        for alert in result.alerts:
            print(f"  [{alert.severity}] {alert.feature}: PSI={alert.psi:.3f}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
