from app.core.db import SessionLocal
from app.services.mining_service import run_mining


def run_mining_job(mining_run_id: int) -> None:
    """RQ entry point: opens its own DB session (workers don't share the
    request-scoped one FastAPI hands out)."""
    db = SessionLocal()
    try:
        run_mining(db, mining_run_id)
    finally:
        db.close()
