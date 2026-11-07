"""Initial persistent state schema.

Revision ID: 0001
Revises: None
Create Date: 2026-11-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "courses",
        sa.Column("course_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("source_url", sa.String(length=2048), nullable=False),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "status",
            sa.Enum("active", "archived", "unavailable", name="course_status", native_enum=False),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("course_id", name="pk_courses"),
    )
    op.create_table(
        "lessons",
        sa.Column("lesson_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("language", sa.String(length=20), nullable=True),
        sa.Column("last_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("lesson_id", name="pk_lessons"),
    )
    op.create_table(
        "course_units",
        sa.Column("unit_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("course_id", sa.BigInteger(), nullable=False),
        sa.Column("section_id", sa.BigInteger(), nullable=False),
        sa.Column("lesson_id", sa.BigInteger(), nullable=False),
        sa.Column("section_position", sa.Integer(), nullable=True),
        sa.Column("unit_position", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "section_position IS NULL OR section_position > 0",
            name="ck_course_units_section_position",
        ),
        sa.CheckConstraint("unit_position > 0", name="ck_course_units_unit_position"),
        sa.ForeignKeyConstraint(
            ["course_id"],
            ["courses.course_id"],
            name="fk_course_units_course_id_courses",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["lesson_id"],
            ["lessons.lesson_id"],
            name="fk_course_units_lesson_id_lessons",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("unit_id", name="pk_course_units"),
        sa.UniqueConstraint("course_id", "lesson_id", name="course_lesson"),
    )
    op.create_index("ix_course_units_section_id", "course_units", ["section_id"], unique=False)
    op.create_index(
        "ix_course_units_course_order",
        "course_units",
        ["course_id", "section_position", "unit_position"],
        unique=False,
    )
    op.create_table(
        "tasks",
        sa.Column("step_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("lesson_id", sa.BigInteger(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("task_type", sa.String(length=100), nullable=False),
        sa.Column("task_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "discovered",
                "ready",
                "analyzing",
                "solving",
                "local_testing",
                "submitting",
                "waiting_verdict",
                "correct",
                "retrying",
                "failed",
                "unsupported",
                "blocked",
                "paused",
                name="task_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("progress_id", sa.String(length=100), nullable=True),
        sa.Column("maximum_score", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column("achieved_score", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column("is_attempted", sa.Boolean(), nullable=False),
        sa.Column("is_passed", sa.Boolean(), nullable=False),
        sa.Column("last_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("achieved_score >= 0", name="ck_tasks_achieved_score"),
        sa.CheckConstraint("maximum_score >= 0", name="ck_tasks_maximum_score"),
        sa.CheckConstraint("position > 0", name="ck_tasks_position"),
        sa.ForeignKeyConstraint(
            ["lesson_id"],
            ["lessons.lesson_id"],
            name="fk_tasks_lesson_id_lessons",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("step_id", name="pk_tasks"),
        sa.UniqueConstraint("lesson_id", "position", name="lesson_position"),
        sa.UniqueConstraint("progress_id", name="uq_tasks_progress_id"),
    )
    op.create_index("ix_tasks_task_hash", "tasks", ["task_hash"], unique=False)
    op.create_index("ix_tasks_status", "tasks", ["status"], unique=False)
    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("course_id", sa.BigInteger(), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "execution_mode",
            sa.Enum(
                "dry_run", "auto", "single_task", "resume", name="execution_mode", native_enum=False
            ),
            nullable=False,
        ),
        sa.Column(
            "state",
            sa.Enum(
                "running",
                "completed",
                "partial",
                "stopped",
                "auth_required",
                "budget_exceeded",
                "error",
                name="run_state",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("total_tasks", sa.Integer(), nullable=False),
        sa.Column("completed_tasks", sa.Integer(), nullable=False),
        sa.Column("failed_tasks", sa.Integer(), nullable=False),
        sa.Column("estimated_cost", sa.Numeric(precision=14, scale=6), nullable=False),
        sa.CheckConstraint("completed_tasks >= 0", name="ck_runs_completed_tasks"),
        sa.CheckConstraint("estimated_cost >= 0", name="ck_runs_estimated_cost"),
        sa.CheckConstraint("failed_tasks >= 0", name="ck_runs_failed_tasks"),
        sa.CheckConstraint("total_tasks >= 0", name="ck_runs_total_tasks"),
        sa.ForeignKeyConstraint(
            ["course_id"],
            ["courses.course_id"],
            name="fk_runs_course_id_courses",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("run_id", name="pk_runs"),
    )
    op.create_index("ix_runs_course_id", "runs", ["course_id"], unique=False)
    op.create_index("ix_runs_state", "runs", ["state"], unique=False)
    op.create_table(
        "task_attempts",
        sa.Column("local_attempt_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("step_id", sa.BigInteger(), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("stepik_attempt_id", sa.BigInteger(), nullable=True),
        sa.Column("stepik_submission_id", sa.BigInteger(), nullable=True),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=True),
        sa.Column("solution_text", sa.Text(), nullable=True),
        sa.Column("solution_hash", sa.String(length=64), nullable=True),
        sa.Column("verdict", sa.String(length=100), nullable=True),
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("attempt_number > 0", name="ck_task_attempts_attempt_number"),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.run_id"],
            name="fk_task_attempts_run_id_runs",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["step_id"],
            ["tasks.step_id"],
            name="fk_task_attempts_step_id_tasks",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("local_attempt_id", name="pk_task_attempts"),
        sa.UniqueConstraint("step_id", "attempt_number", name="task_attempt_number"),
        sa.UniqueConstraint("stepik_attempt_id", name="stepik_attempt_id"),
        sa.UniqueConstraint("stepik_submission_id", name="stepik_submission_id"),
    )
    op.create_index("ix_task_attempts_run_id", "task_attempts", ["run_id"], unique=False)
    op.create_index(
        "ix_task_attempts_solution_hash", "task_attempts", ["solution_hash"], unique=False
    )
    op.create_index("ix_task_attempts_step_id", "task_attempts", ["step_id"], unique=False)
    op.create_table(
        "ai_requests",
        sa.Column("request_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("local_attempt_id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column(
            "role",
            sa.Enum(
                "solver",
                "reviewer",
                "repairer",
                "diagnostic",
                name="ai_role",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("provider_cost", sa.Numeric(precision=14, scale=6), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "succeeded",
                "failed",
                name="ai_request_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("input_tokens >= 0", name="ck_ai_requests_input_tokens"),
        sa.CheckConstraint("latency_ms >= 0", name="ck_ai_requests_latency_ms"),
        sa.CheckConstraint("output_tokens >= 0", name="ck_ai_requests_output_tokens"),
        sa.CheckConstraint("provider_cost >= 0", name="ck_ai_requests_provider_cost"),
        sa.ForeignKeyConstraint(
            ["local_attempt_id"],
            ["task_attempts.local_attempt_id"],
            name="fk_ai_requests_local_attempt_id_task_attempts",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("request_id", name="pk_ai_requests"),
    )
    op.create_index(
        "ix_ai_requests_local_attempt_id", "ai_requests", ["local_attempt_id"], unique=False
    )
    op.create_table(
        "event_logs",
        sa.Column("event_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("step_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "severity",
            sa.Enum(
                "debug",
                "info",
                "warning",
                "error",
                "critical",
                name="event_severity",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("sanitized_payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.run_id"],
            name="fk_event_logs_run_id_runs",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["step_id"],
            ["tasks.step_id"],
            name="fk_event_logs_step_id_tasks",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("event_id", name="pk_event_logs"),
    )
    op.create_index("ix_event_logs_run_created", "event_logs", ["run_id", "created_at"])
    op.create_index("ix_event_logs_step_id", "event_logs", ["step_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_event_logs_step_id", table_name="event_logs")
    op.drop_index("ix_event_logs_run_created", table_name="event_logs")
    op.drop_table("event_logs")
    op.drop_index("ix_ai_requests_local_attempt_id", table_name="ai_requests")
    op.drop_table("ai_requests")
    op.drop_index("ix_task_attempts_step_id", table_name="task_attempts")
    op.drop_index("ix_task_attempts_solution_hash", table_name="task_attempts")
    op.drop_index("ix_task_attempts_run_id", table_name="task_attempts")
    op.drop_table("task_attempts")
    op.drop_index("ix_runs_state", table_name="runs")
    op.drop_index("ix_runs_course_id", table_name="runs")
    op.drop_table("runs")
    op.drop_index("ix_tasks_status", table_name="tasks")
    op.drop_index("ix_tasks_task_hash", table_name="tasks")
    op.drop_table("tasks")
    op.drop_index("ix_course_units_course_order", table_name="course_units")
    op.drop_index("ix_course_units_section_id", table_name="course_units")
    op.drop_table("course_units")
    op.drop_table("lessons")
    op.drop_table("courses")
