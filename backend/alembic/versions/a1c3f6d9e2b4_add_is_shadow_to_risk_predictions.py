"""add is_shadow to risk_predictions

Revision ID: a1c3f6d9e2b4
Revises: 076429ecae21
Create Date: 2026-10-14 00:00:00.000000

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = 'a1c3f6d9e2b4'
down_revision: str | None = '076429ecae21'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Phase 10 (US-45): shadow mode scores every commit with the current
    # challenger alongside the champion, without it driving the real
    # decision — is_shadow distinguishes those rows so the comparison page
    # can compare champion vs. challenger on the exact same inputs. Not in
    # the original §7 table — see docs/decisions/008-shadow-mode-scope.md.
    op.add_column(
        'risk_predictions',
        sa.Column('is_shadow', sa.Boolean(), nullable=False, server_default='false'),
    )


def downgrade() -> None:
    op.drop_column('risk_predictions', 'is_shadow')
