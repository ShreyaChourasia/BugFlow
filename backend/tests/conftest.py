from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session, sessionmaker

from app.core.db import Base, engine, get_db
from app.core.security import hash_password
from app.main import app
from app.models.enums import Role
from app.models.user import User


@pytest.fixture(scope="session", autouse=True)
def _tables() -> Generator[None, None, None]:
    """Tests run against the dev Postgres (see .env / DATABASE_URL); this only
    fills in any tables Alembic hasn't been run for yet, it never drops any."""
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Each test runs inside an outer transaction that's rolled back afterwards,
    so tests never leave data behind or see each other's writes. App code (and
    the routers under test) call `session.commit()` freely; a SAVEPOINT is
    restarted after each one so those commits never touch the outer
    transaction (see SQLAlchemy's "joining a session into an external
    transaction" recipe)."""
    connection = engine.connect()
    outer_transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    session.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def _restart_savepoint(session: Session, transaction: object) -> None:
        if not connection.in_nested_transaction():
            connection.begin_nested()

    try:
        yield session
    finally:
        session.close()
        outer_transaction.rollback()
        connection.close()


def make_user(db_session: Session, email: str, role: Role, password: str = "test-password") -> User:
    user = User(
        name=email.split("@")[0],
        email=email,
        password_hash=hash_password(password),
        role=role.value,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def _get_db_override() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = _get_db_override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
