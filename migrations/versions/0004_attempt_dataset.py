"""Persist attempt datasets required by interactive Stepik tasks.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("task_attempts", sa.Column("attempt_dataset", sa.JSON(), nullable=True))
    op.add_column("task_attempts", sa.Column("dataset_url", sa.String(length=2048), nullable=True))


def downgrade() -> None:
    op.drop_column("task_attempts", "dataset_url")
    op.drop_column("task_attempts", "attempt_dataset")
