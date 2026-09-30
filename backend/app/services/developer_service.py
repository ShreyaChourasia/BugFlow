from datetime import datetime

from sqlalchemy import any_, func, select
from sqlalchemy.orm import Session

from app.models.commit import Commit
from app.models.developer import Developer


def get_or_create_developer(
    db: Session, cache: dict[str, Developer], name: str, email: str, seen_at: datetime
) -> Developer:
    if email in cache:
        return cache[email]

    developer = db.scalar(select(Developer).where(any_(Developer.git_emails) == email))
    if developer is None:
        developer = Developer(name=name, git_emails=[email], joined_at=seen_at)
        db.add(developer)
        db.flush()

    cache[email] = developer
    return developer


def seed_author_commit_counts(db: Session, repository_id: int) -> dict[int, int]:
    """Bulk-loads every author's commit count for a repo in one query —
    used when about to process many commits in sequence (mining)."""
    rows = db.execute(
        select(Commit.author_id, func.count())
        .where(Commit.repository_id == repository_id, Commit.author_id.is_not(None))
        .group_by(Commit.author_id)
    ).all()
    return {author_id: count for author_id, count in rows if author_id is not None}


def count_author_commits(db: Session, repository_id: int, developer_id: int) -> int:
    """A single author's commit count — used when only scoring one commit
    (PR scoring), where seeding the whole repo's counts would be wasteful."""
    return db.scalar(
        select(func.count()).where(
            Commit.repository_id == repository_id, Commit.author_id == developer_id
        )
    ) or 0
