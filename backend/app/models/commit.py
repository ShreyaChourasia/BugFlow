from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Commit(Base):
    __tablename__ = "commits"
    __table_args__ = (UniqueConstraint("repository_id", "sha", name="uq_commits_repository_sha"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    sha: Mapped[str] = mapped_column(String(40), index=True)
    repository_id: Mapped[int] = mapped_column(ForeignKey("repositories.id"))
    author_id: Mapped[int | None] = mapped_column(ForeignKey("developers.id"), nullable=True)
    message: Mapped[str] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    lines_added: Mapped[int] = mapped_column(Integer, default=0)
    lines_deleted: Mapped[int] = mapped_column(Integer, default=0)
    files_changed: Mapped[int] = mapped_column(Integer, default=0)
    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_bug_inducing: Mapped[bool] = mapped_column(Boolean, default=False)
    is_fix: Mapped[bool] = mapped_column(Boolean, default=False)
    # US-03: issue numbers referenced in the commit message (e.g. "fixes #123").
    # Not in the original §7 table — see docs/decisions/001-issue-refs-on-commit.md.
    linked_issue_refs: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)
    data_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
