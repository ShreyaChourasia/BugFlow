import random
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.db import Base, engine, get_db
from app.core.security import hash_password
from app.main import app
from app.models.commit import Commit
from app.models.developer import Developer
from app.models.enums import Role
from app.models.ml import MLModel
from app.models.pull_request import LineRisk, PullRequest, RiskPrediction
from app.models.repository import MiningRun, Repository
from app.models.user import User

from .gitutil import commit_all, init_repo

RISKY_LINE = "except: pass\n"
SAFE_LINE = "return result\n"


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

    # The dev Postgres this runs against also gets used for manual demos
    # (make train, replay scripts, ...), which leave real committed rows a
    # test's own rollback can never see away. Model-registry state in
    # particular ("is there a champion?") needs to start clean every time,
    # regardless of what's been run against this database outside of tests —
    # safe to delete here since it's inside this test's own transaction.
    session.execute(delete(MLModel))

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


def make_commits_for_training(db_session: Session, n: int = 150, seed: int = 0) -> list[Commit]:
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
def line_risk_repo(tmp_path: Path) -> tuple[Path, list[str]]:
    """20 commits, each appending 4 lines — every 3rd commit is all risky
    lines, giving >= MIN_TRAINING_LINES (50) weakly-labelled examples."""
    repo = tmp_path / "repo"
    init_repo(repo)
    path = repo / "a.py"
    path.write_text("")

    shas = []
    for i in range(20):
        line = RISKY_LINE if i % 3 == 0 else SAFE_LINE
        with path.open("a") as f:
            f.writelines([line] * 4)
        shas.append(commit_all(repo, f"commit {i}"))
    return repo, shas


@pytest.fixture
def repository_with_commits(
    db_session: Session, line_risk_repo: tuple[Path, list[str]]
) -> Repository:
    # Line-risk training is global across every mined repo (same as
    # commit_risk), and re-clones each one to recover added-line text — any
    # repo left in this shared dev Postgres from a manual demo (`make seed`,
    # README's "six" walkthrough) would get network-cloned into this test.
    # Scope the test's world to just its own fixture repo (rolled back with
    # everything else at the end of this test's transaction).
    db_session.execute(delete(LineRisk))
    db_session.execute(delete(RiskPrediction))
    db_session.execute(delete(PullRequest))
    db_session.execute(delete(Commit))
    db_session.execute(delete(MiningRun))
    db_session.execute(delete(Repository))

    repo_path, shas = line_risk_repo
    repository = Repository(name="line-risk-demo", url=str(repo_path))
    db_session.add(repository)
    db_session.commit()

    base = datetime(2024, 1, 1, tzinfo=UTC)
    for i, sha in enumerate(shas):
        db_session.add(
            Commit(
                sha=sha,
                repository_id=repository.id,
                message=f"commit {i}",
                timestamp=base + timedelta(minutes=i),
                is_bug_inducing=(i % 3 == 0),
            )
        )
    db_session.commit()
    return repository


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def _get_db_override() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = _get_db_override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
