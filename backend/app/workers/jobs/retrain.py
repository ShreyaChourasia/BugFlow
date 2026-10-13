from app.core.db import SessionLocal
from app.core.logging import get_logger

logger = get_logger(__name__)


def retrain_job(group: str, seed: int = 42) -> None:
    """US-41/43/45's "retrain from feedback plus new data": runs the same
    per-task training entrypoint the standalone `scripts/train_*.py`
    scripts use, just triggered from the API and off the request path
    (background jobs run outside the request path, per the architecture) —
    each trainer already prefers real/feedback-influenced data over its
    bootstrap fallback on its own, this just lets a human kick it off now
    instead of waiting for the next scheduled run."""
    db = SessionLocal()
    try:
        if group == "commit_risk":
            from app.services.training_service import train_and_register_champion

            train_and_register_champion(db, seed=seed)
        elif group == "line_risk":
            from app.services.line_risk_training_service import train_and_register_champion

            train_and_register_champion(db, seed=seed)
        elif group == "triage":
            from app.services.triage_training_service import train_and_register_champions

            train_and_register_champions(db, seed=seed)
        elif group == "resolver_suitability":
            from app.services.resolver_training_service import train_and_register_champion

            train_and_register_champion(db, seed=seed)
        elif group == "resolution_forecast":
            from app.services.forecast_training_service import train_and_register_champion

            train_and_register_champion(db, seed=seed)
        else:
            raise ValueError(f"Unknown retrain group: {group}")

        logger.info("retrain_job_complete", group=group, seed=seed)
    except Exception:
        logger.exception("retrain_job_failed", group=group, seed=seed)
        raise
    finally:
        db.close()
