from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import require_role
from app.models.enums import Role
from app.models.system import AuditLog, SystemConfig
from app.models.user import User
from app.schemas.system_config import SystemConfigRead, SystemConfigUpsert
from app.services.audit import write_audit_log

router = APIRouter(
    prefix="/admin", tags=["admin"], dependencies=[Depends(require_role(Role.ADMIN))]
)


@router.get("/config", response_model=list[SystemConfigRead])
def list_config(db: Session = Depends(get_db)) -> list[SystemConfig]:
    return list(db.scalars(select(SystemConfig)))


@router.put("/config/{key}", response_model=SystemConfigRead)
def upsert_config(
    key: str,
    body: SystemConfigUpsert,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role(Role.ADMIN)),
) -> SystemConfig:
    entry = db.get(SystemConfig, key)
    if entry is None:
        entry = SystemConfig(key=key, value=body.value, updated_by=admin.id)
        db.add(entry)
    else:
        entry.value = body.value
        entry.updated_by = admin.id
    db.commit()
    db.refresh(entry)
    write_audit_log(db, admin.id, "upsert", "SystemConfig", None, {"key": key, "value": body.value})
    return entry


@router.get("/audit-log")
def list_audit_log(db: Session = Depends(get_db)) -> list[dict]:
    logs = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(200))
    return [
        {
            "id": log.id,
            "user_id": log.user_id,
            "action": log.action,
            "entity": log.entity,
            "entity_id": log.entity_id,
            "payload": log.payload,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]
