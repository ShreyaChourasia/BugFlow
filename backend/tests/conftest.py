import random
from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session, sessionmaker

from app.core.db import Base, engine, get_db
from app.core.security import hash_password
from app.main import app
from app.models.commit import Commit
from app.models.developer import Developer
from app.models.enums import Role
from app.models.repository import Repository
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


def make_commits_for_training(
    db_session: Session, n: int = 150, seed: int = 0
) -> list[Commit]:
    """Enough synthetic, chronologically-ordered, feature-labelled commits to
    exercise the training pipeline's chronological split meaningfully — used
    instead of real mining so training tests are fast and deterministic."""
    repo = Repository(name="training-demo", url="/tmp/does-not-need-to-exist")
    db_session.add(repo)
    developer = Developer(
        name="Dev", git_emails=["dev@example.com"], joined_at=datetime(2023, 1, 1, tzinfo=UTC)
    )
    db_session.add(developer)
    db_session.commit()

    rng = random.Random(seed)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    commits = []
    for i in range(n):
        churn = rng.randint(1, 200)
        files_changed = rng.randint(1, 10)
        risk_signal = churn / 200 * 0.6 + files_changed / 10 * 0.4
        is_bug = rng.random() < (0.05 + 0.5 * risk_signal)
        features = {
            "lines_added": churn // 2,
            "lines_deleted": churn // 2,
            "churn": churn,
            "files_changed": files_changed,
            "directories_touched": rng.randint(1, files_changed),
            "subsystems_touched": rng.randint(1, min(3, files_changed)),
            "entropy": rng.random() * 2,
            "author_prior_commits": i,
            "is_fix": rng.random() < 0.3,
        }
        commits.append(
            Commit(
                sha=f"sha{i}",
                repository_id=repo.id,
                author_id=developer.id,
                message=f"commit {i}",
                timestamp=base + timedelta(hours=i),
                lines_added=features["lines_added"],
                lines_deleted=features["lines_deleted"],
                files_changed=files_changed,
                features=features,
                is_bug_inducing=is_bug,
            )
        )
    db_session.add_all(commits)
    db_session.commit()
    for commit in commits:
        db_session.refresh(commit)
    return commits


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def _get_db_override() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = _get_db_override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
