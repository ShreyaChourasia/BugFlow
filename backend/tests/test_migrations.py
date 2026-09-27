"""Validates that the Alembic migrations actually apply (and revert) cleanly,
against a throwaway database so the main dev/test database is untouched."""

from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import URL, Engine, create_engine, text
from sqlalchemy.engine import make_url

from alembic import command
from app.core.config import get_settings

BACKEND_DIR = Path(__file__).resolve().parent.parent
SCRATCH_DB = "bugflow_migration_test"


def _admin_engine(admin_url: URL) -> Engine:
    return create_engine(admin_url, isolation_level="AUTOCOMMIT")


@pytest.mark.story("US-50")
def test_alembic_migrations_apply_and_revert_cleanly() -> None:
    base_url = make_url(get_settings().database_url)
    admin_url = base_url.set(database="postgres")

    engine = _admin_engine(admin_url)
    with engine.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}"'))
        conn.execute(text(f'CREATE DATABASE "{SCRATCH_DB}"'))
    engine.dispose()

    scratch_url = base_url.set(database=SCRATCH_DB)
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", scratch_url.render_as_string(hide_password=False))

    try:
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "base")
    finally:
        engine = _admin_engine(admin_url)
        with engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}"'))
        engine.dispose()
