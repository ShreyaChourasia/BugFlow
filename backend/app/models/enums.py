from enum import StrEnum


class Role(StrEnum):
    """Matches the U1-U8 roles in the master prompt §2.

    Stored as plain text (see User.role), not a Postgres native enum: adding a
    role later is a one-line change here, not an ALTER TYPE migration.
    """

    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    TRIAGER = "triager"
    REPORTER = "reporter"
    QA = "qa"
    MANAGER = "manager"
    ML_ENGINEER = "ml_engineer"
    ADMIN = "admin"
