from app.core.db import SessionLocal
from app.services.pr_scoring_service import score_pull_request


def score_pull_request_job(pull_request_id: int) -> None:
    """RQ entry point. A hard job timeout (set at enqueue time) is what
    actually implements C1's "if the timeout is hit, stay pending" — a
    killed job never reaches the code that would mark it completed or
    unavailable, so the check simply stays at whatever it was set to right
    after the webhook fired."""
    db = SessionLocal()
    try:
        score_pull_request(db, pull_request_id)
    finally:
        db.close()
