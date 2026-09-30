"""add hnsw index on defect_reports embedding

Revision ID: 076429ecae21
Revises: fae879913708
Create Date: 2026-09-30 22:49:49.018023

"""
from collections.abc import Sequence

from alembic import op


revision: str = '076429ecae21'
down_revision: str | None = 'fae879913708'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # US-16/US-19: cosine-distance ANN search over defect_reports.embedding.
    # vector_cosine_ops matches the cosine_distance() comparator the service
    # layer queries with — the default op class is L2, which would silently
    # build an index the planner can't use for this query shape.
    op.execute(
        "CREATE INDEX ix_defect_reports_embedding_hnsw "
        "ON defect_reports USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_defect_reports_embedding_hnsw")
