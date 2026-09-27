#!/usr/bin/env python3
"""Creates one demo user per role (see master prompt §2) so the UI never looks
empty. Safe to re-run: skips users that already exist by email.

Usage (inside the api container, which has `app` installed):
    docker compose exec api python /app/scripts/seed_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from sqlalchemy import select  # noqa: E402

from app.core.db import SessionLocal  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.models.enums import Role  # noqa: E402
from app.models.user import User  # noqa: E402

DEMO_PASSWORD = "bugflow-demo"

DEMO_USERS = [
    ("Dana Developer", "developer@bugflow.demo", Role.DEVELOPER),
    ("Rae Reviewer", "reviewer@bugflow.demo", Role.REVIEWER),
    ("Tara Triager", "triager@bugflow.demo", Role.TRIAGER),
    ("Remy Reporter", "reporter@bugflow.demo", Role.REPORTER),
    ("Quinn QA", "qa@bugflow.demo", Role.QA),
    ("Morgan Manager", "manager@bugflow.demo", Role.MANAGER),
    ("Mel MLEngineer", "ml-engineer@bugflow.demo", Role.ML_ENGINEER),
    ("Alex Admin", "admin@bugflow.demo", Role.ADMIN),
]


def main() -> None:
    db = SessionLocal()
    try:
        created = []
        for name, email, role in DEMO_USERS:
            if db.scalar(select(User).where(User.email == email)) is not None:
                continue
            db.add(
                User(
                    name=name,
                    email=email,
                    password_hash=hash_password(DEMO_PASSWORD),
                    role=role.value,
                )
            )
            created.append(email)
        db.commit()

        if created:
            print(f"Created {len(created)} demo user(s): {', '.join(created)}")
        else:
            print("Demo users already exist, nothing to do.")
        print(f"All demo users share the password: {DEMO_PASSWORD}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
