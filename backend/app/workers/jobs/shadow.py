from app.core.db import SessionLocal
from app.models.commit import Commit
from app.services.prediction_service import score_shadow


def score_shadow_job(commit_id: int) -> None:
    """RQ entry point for US-45 shadow-scoring (see prediction_service)."""
    db = SessionLocal()
    try:
        commit = db.get(Commit, commit_id)
        if commit is not None:
            score_shadow(db, commit)
    finally:
        db.close()
