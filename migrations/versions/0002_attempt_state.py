"""Add durable attempt state for crash-safe Stepik writes.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ATTEMPT_STATES = (
    "planned",
    "creating_attempt",
    "attempt_created",
    "creating_submission",
    "waiting_verdict",
    "correct",
    "wrong",
    "ambiguous",
    "failed",
)


def upgrade() -> None:
    with op.batch_alter_table("task_attempts") as batch_op:
        batch_op.add_column(
            sa.Column(
                "state",
                sa.Enum(*_ATTEMPT_STATES, name="attempt_state", native_enum=False),
                nullable=False,
                server_default="planned",
            )
        )
        batch_op.add_column(sa.Column("solution_language", sa.String(length=100), nullable=True))
        batch_op.create_index("ix_task_attempts_state", ["state"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("task_attempts") as batch_op:
        batch_op.drop_index("ix_task_attempts_state")
        batch_op.drop_column("solution_language")
        batch_op.drop_column("state")
