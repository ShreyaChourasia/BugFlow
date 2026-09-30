import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.enums import Role
from app.models.repository import Repository
from app.models.system import AuditLog
from app.schemas.defect import DefectReportCreate
from app.services import defect_service
from app.workers.jobs.defect import notify_merge_job

from .conftest import make_user


def test_notify_merge_job_writes_an_audit_log_entry(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # notify_merge_job opens its own SessionLocal() rather than taking the
    # test's transactional db_session — point it at the same session so its
    # writes land inside this test's own transaction, and stop its `finally:
    # db.close()` from tearing that session down before the test's own
    # assertions (and conftest's real cleanup) get to run.
    import app.workers.jobs.defect as defect_jobs_module

    monkeypatch.setattr(defect_jobs_module, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)

    repo = Repository(name="demo", url="/tmp/x")
    db_session.add(repo)
    db_session.commit()
    reporter = make_user(db_session, "reporter@example.com", Role.REPORTER)
    original = defect_service.create_defect_report(
        db_session, reporter, DefectReportCreate(repository_id=repo.id, title="A", description="a")
    )
    duplicate = defect_service.create_defect_report(
        db_session, reporter, DefectReportCreate(repository_id=repo.id, title="B", description="b")
    )

    notify_merge_job(duplicate.id, original.id)

    entry = db_session.scalar(
        select(AuditLog).where(
            AuditLog.action == "duplicate_merge_notification", AuditLog.entity_id == duplicate.id
        )
    )
    assert entry is not None
    assert entry.payload == {"merged_into": original.id}
