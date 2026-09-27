import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from redis import Redis
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import get_db
from app.models.enums import Role
from app.models.user import User

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
_redis = Redis.from_url(settings.redis_url)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def _create_token(subject: int, expires_delta: timedelta, token_type: str) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type,
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")


def create_access_token(user_id: int) -> str:
    return _create_token(
        user_id, timedelta(minutes=settings.jwt_access_token_expire_minutes), "access"
    )


def create_refresh_token(user_id: int) -> str:
    return _create_token(user_id, timedelta(days=settings.jwt_refresh_token_expire_days), "refresh")


def decode_token(token: str, expected_type: str) -> dict[str, Any]:
    try:
        payload: dict[str, Any] = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
    except JWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from exc

    if payload.get("type") != expected_type:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong token type")
    if _redis.exists(f"revoked_jti:{payload['jti']}"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token has been revoked")
    return payload


def revoke_token(token: str) -> None:
    """Blacklists a token's jti in Redis until its own expiry (logout)."""
    payload = decode_token(token, "access")
    ttl = int(payload["exp"] - datetime.now(UTC).timestamp())
    if ttl > 0:
        _redis.set(f"revoked_jti:{payload['jti']}", "1", ex=ttl)


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    payload = decode_token(token, "access")
    user = db.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive")
    return user


def require_role(*roles: Role) -> Callable[[User], User]:
    """FastAPI dependency: 403s unless the current user has one of `roles`."""

    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in {r.value for r in roles}:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Not permitted for this role")
        return user

    return dependency
