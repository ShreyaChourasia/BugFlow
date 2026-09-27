from sqlalchemy.orm import Session

from app.models.system import AuditLog


def write_audit_log(
    db: Session,
    user_id: int,
    action: str,
    entity: str,
    entity_id: int | None = None,
    payload: dict | None = None,
) -> None:
    db.add(
        AuditLog(
            user_id=user_id,
            action=action,
            entity=entity,
            entity_id=entity_id,
            payload=payload,
        )
    )
    db.commit()
