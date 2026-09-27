#!/usr/bin/env python3
"""US-44: exports a complete experiment record to a JSON file.

Usage:
    docker compose exec api python /app/scripts/export_experiment.py --run-id <id> --out record.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.training_service import export_experiment_record  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", default="experiment_record.json")
    args = parser.parse_args()

    record = export_experiment_record(args.run_id)
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
