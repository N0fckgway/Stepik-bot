from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from stepik_agent.core.states import (
    AIRequestStatus,
    AIRole,
    AttemptState,
    CourseStatus,
    EventSeverity,
    ExecutionMode,
    RunState,
    TaskStatus,
)

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def enum_values(enum_class: type[StrEnum]) -> list[str]:
    return [member.value for member in enum_class]


class Course(Base):
    __tablename__ = "courses"

    course_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(String(500))
    source_url: Mapped[str] = mapped_column(String(2048))
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[CourseStatus] = mapped_column(
        Enum(
            CourseStatus,
            name="course_status",
            native_enum=False,
            validate_strings=True,
            values_callable=enum_values,
        ),
        default=CourseStatus.ACTIVE,
    )

    units: Mapped[list["CourseUnit"]] = relationship(
        back_populates="course",
        cascade="all, delete-orphan",
    )
    runs: Mapped[list["Run"]] = relationship(back_populates="course")


class Lesson(Base):
    __tablename__ = "lessons"

    lesson_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(String(500))
    language: Mapped[str | None] = mapped_column(String(20))
    last_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    units: Mapped[list["CourseUnit"]] = relationship(back_populates="lesson")
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="lesson",
        cascade="all, delete-orphan",
    )


class CourseUnit(Base):
    __tablename__ = "course_units"
    __table_args__ = (
        UniqueConstraint("course_id", "lesson_id", name="course_lesson"),
        CheckConstraint(
            "section_position IS NULL OR section_position > 0", name="section_position"
        ),
        CheckConstraint("unit_position > 0", name="unit_position"),
        Index("ix_course_units_course_order", "course_id", "section_position", "unit_position"),
    )

    unit_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    course_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("courses.course_id", ondelete="CASCADE"),
        nullable=False,
    )
    section_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    lesson_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("lessons.lesson_id", ondelete="RESTRICT"),
        nullable=False,
    )
    section_position: Mapped[int | None] = mapped_column(Integer)
    unit_position: Mapped[int] = mapped_column(Integer)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    assignment_map: Mapped[dict[str, int]] = mapped_column(JSON, default=dict)

    course: Mapped[Course] = relationship(back_populates="units")
    lesson: Mapped[Lesson] = relationship(back_populates="units")


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        UniqueConstraint("lesson_id", "position", name="lesson_position"),
        CheckConstraint("position > 0", name="position"),
        CheckConstraint("maximum_score >= 0", name="maximum_score"),
        CheckConstraint("achieved_score >= 0", name="achieved_score"),
    )

    step_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    lesson_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("lessons.lesson_id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer)
    task_type: Mapped[str] = mapped_column(String(100))
    task_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    status: Mapped[TaskStatus] = mapped_column(
        Enum(
            TaskStatus,
            name="task_status",
            native_enum=False,
            validate_strings=True,
            values_callable=enum_values,
        ),
        default=TaskStatus.DISCOVERED,
        index=True,
    )
    progress_id: Mapped[str | None] = mapped_column(String(100), unique=True)
    maximum_score: Mapped[Decimal] = mapped_column(Numeric(12, 4), default=Decimal("0"))
    achieved_score: Mapped[Decimal] = mapped_column(Numeric(12, 4), default=Decimal("0"))
    is_attempted: Mapped[bool] = mapped_column(Boolean, default=False)
    is_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    last_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    lesson: Mapped[Lesson] = relationship(back_populates="tasks")
    attempts: Mapped[list["TaskAttempt"]] = relationship(back_populates="task")


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint("total_tasks >= 0", name="total_tasks"),
        CheckConstraint("completed_tasks >= 0", name="completed_tasks"),
        CheckConstraint("failed_tasks >= 0", name="failed_tasks"),
        CheckConstraint("estimated_cost >= 0", name="estimated_cost"),
    )

    run_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    course_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("courses.course_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    execution_mode: Mapped[ExecutionMode] = mapped_column(
        Enum(
            ExecutionMode,
            name="execution_mode",
            native_enum=False,
            validate_strings=True,
            values_callable=enum_values,
        )
    )
    state: Mapped[RunState] = mapped_column(
        Enum(
            RunState,
            name="run_state",
            native_enum=False,
            validate_strings=True,
            values_callable=enum_values,
        ),
        default=RunState.RUNNING,
        index=True,
    )
    total_tasks: Mapped[int] = mapped_column(Integer, default=0)
    completed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    failed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[Decimal] = mapped_column(Numeric(14, 6), default=Decimal("0"))

    course: Mapped[Course] = relationship(back_populates="runs")
    attempts: Mapped[list["TaskAttempt"]] = relationship(back_populates="run")
    events: Mapped[list["EventLog"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class TaskAttempt(Base):
    __tablename__ = "task_attempts"
    __table_args__ = (
        UniqueConstraint("step_id", "attempt_number", name="task_attempt_number"),
        UniqueConstraint("stepik_attempt_id", name="stepik_attempt_id"),
        UniqueConstraint("stepik_submission_id", name="stepik_submission_id"),
        CheckConstraint("attempt_number > 0", name="attempt_number"),
    )

    local_attempt_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    step_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("tasks.step_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("runs.run_id", ondelete="SET NULL"),
        index=True,
    )
    stepik_attempt_id: Mapped[int | None] = mapped_column(BigInteger)
    stepik_submission_id: Mapped[int | None] = mapped_column(BigInteger)
    attempt_number: Mapped[int] = mapped_column(Integer)
    state: Mapped[AttemptState] = mapped_column(
        Enum(
            AttemptState,
            name="attempt_state",
            native_enum=False,
            validate_strings=True,
            values_callable=enum_values,
        ),
        default=AttemptState.PLANNED,
        index=True,
    )
    model_id: Mapped[str | None] = mapped_column(String(255))
    solution_text: Mapped[str | None] = mapped_column(Text)
    solution_language: Mapped[str | None] = mapped_column(String(100))
    submission_reply: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    attempt_dataset: Mapped[dict[str, Any] | str | None] = mapped_column(JSON)
    dataset_url: Mapped[str | None] = mapped_column(String(2048))
    solution_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    verdict: Mapped[str | None] = mapped_column(String(100))
    feedback: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    task: Mapped[Task] = relationship(back_populates="attempts")
    run: Mapped[Run | None] = relationship(back_populates="attempts")
    ai_requests: Mapped[list["AIRequest"]] = relationship(
        back_populates="attempt",
        cascade="all, delete-orphan",
    )


class AIRequest(Base):
    __tablename__ = "ai_requests"
    __table_args__ = (
        CheckConstraint("input_tokens >= 0", name="input_tokens"),
        CheckConstraint("output_tokens >= 0", name="output_tokens"),
        CheckConstraint("provider_cost >= 0", name="provider_cost"),
        CheckConstraint("latency_ms >= 0", name="latency_ms"),
    )

    request_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    local_attempt_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("task_attempts.local_attempt_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_id: Mapped[str] = mapped_column(String(255))
    role: Mapped[AIRole] = mapped_column(
        Enum(
            AIRole,
            name="ai_role",
            native_enum=False,
            validate_strings=True,
            values_callable=enum_values,
        )
    )
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    provider_cost: Mapped[Decimal] = mapped_column(Numeric(14, 6), default=Decimal("0"))
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[AIRequestStatus] = mapped_column(
        Enum(
            AIRequestStatus,
            name="ai_request_status",
            native_enum=False,
            validate_strings=True,
            values_callable=enum_values,
        ),
        default=AIRequestStatus.PENDING,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    attempt: Mapped[TaskAttempt] = relationship(back_populates="ai_requests")


class EventLog(Base):
    __tablename__ = "event_logs"
    __table_args__ = (Index("ix_event_logs_run_created", "run_id", "created_at"),)

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    step_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("tasks.step_id", ondelete="SET NULL"),
        index=True,
    )
    severity: Mapped[EventSeverity] = mapped_column(
        Enum(
            EventSeverity,
            name="event_severity",
            native_enum=False,
            validate_strings=True,
            values_callable=enum_values,
        )
    )
    event_type: Mapped[str] = mapped_column(String(100))
    sanitized_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    run: Mapped[Run] = relationship(back_populates="events")
