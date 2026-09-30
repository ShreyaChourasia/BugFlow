from sqlalchemy import text

from app.core.db import SessionLocal, engine
from app.core.logging import get_logger
from app.services.audit import write_audit_log
from app.services.defect_service import set_index_status

logger = get_logger(__name__)

INDEX_NAME = "ix_defect_reports_embedding_hnsw"


def notify_merge_job(duplicate_id: int, original_id: int) -> None:
    """US-20: "notify the reporter" has no email/SMTP integration configured
    anywhere in this project (see docs/PROGRESS.md) — logged as an audit
    entry instead, the same "offline-by-default, real integration is a
    config choice" pattern Phase 4 used for GitHub. What matters for the
    story is that this runs asynchronously, in a background job, not inline
    in the merge request."""
    from app.models.defect import DefectReport

    db = SessionLocal()
    try:
        duplicate = db.get(DefectReport, duplicate_id)
        if duplicate is None:
            return
        write_audit_log(
            db,
            duplicate.reporter_id,
            "duplicate_merge_notification",
            "DefectReport",
            duplicate_id,
            {"merged_into": original_id},
        )
        logger.info("defect_merge_notified", duplicate_id=duplicate_id, original_id=original_id)
    finally:
        db.close()


def rebuild_defect_index_job() -> None:
    """C2: while this runs, `get_index_status` reports "rebuilding" so
    search endpoints return that explicitly rather than a misleadingly empty
    result list. `REINDEX ... CONCURRENTLY` needs to run outside a
    transaction block, hence the raw autocommit connection rather than the
    ORM session."""
    db = SessionLocal()
    try:
        set_index_status(db, "rebuilding")
    finally:
        db.close()

    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"REINDEX INDEX CONCURRENTLY {INDEX_NAME}"))
    finally:
        db = SessionLocal()
        try:
            set_index_status(db, "ready")
        finally:
            db.close()
