"""Persistent course queue and run summaries."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select

from stepik_agent.core.states import ExecutionMode, RunState, TaskStatus
from stepik_agent.storage.database import Database
from stepik_agent.storage.models import Course, CourseUnit, Run, Task
from stepik_agent.storage.repositories import RunRepository


@dataclass(frozen=True, slots=True)
class CourseTaskRecord:
    step_id: int
    lesson_id: int
    position: int
    unit_id: int
    task_type: str
    status: TaskStatus
    is_passed: bool
    assignment_id: int | None = None


@dataclass(frozen=True, slots=True)
class CourseRunStatus:
    run_id: str
    course_id: int
    course_title: str
    course_url: str
    state: RunState
    start_time: datetime
    end_time: datetime | None
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    estimated_cost: Decimal
    task_counts: dict[str, int]


class CourseRunStore:
    def __init__(self, database: Database) -> None:
        self._database = database

    def list_tasks(self, course_id: int) -> list[CourseTaskRecord]:
        with self._database.session() as session:
            course = session.get(Course, course_id)
            if course is None:
                raise ValueError(f"Course {course_id} does not exist")
            if course.last_synced_at is None:
                raise ValueError(f"Course {course_id} has not been synchronized")
            rows = session.execute(
                select(Task, CourseUnit.unit_id, CourseUnit.assignment_map)
                .join(CourseUnit, CourseUnit.lesson_id == Task.lesson_id)
                .where(
                    CourseUnit.course_id == course_id,
                    CourseUnit.is_active.is_(True),
                    Task.last_updated_at >= course.last_synced_at,
                )
                .order_by(CourseUnit.section_position, CourseUnit.unit_position, Task.position)
            ).all()
            return [
                CourseTaskRecord(
                    step_id=task.step_id,
                    lesson_id=task.lesson_id,
                    position=task.position,
                    unit_id=unit_id,
                    task_type=task.task_type,
                    status=task.status,
                    is_passed=task.is_passed,
                    assignment_id=_assignment_id(assignment_map, task.step_id),
                )
                for task, unit_id, assignment_map in rows
            ]

    def latest_status(self) -> CourseRunStatus | None:
        with self._database.session() as session:
            run = session.scalar(select(Run).order_by(Run.start_time.desc()).limit(1))
            if run is None:
                return None
            course = session.get(Course, run.course_id)
            if course is None:
                raise ValueError(f"Run {run.run_id} references a missing course")

            rows = session.execute(
                select(Task.status, CourseUnit.unit_id)
                .join(CourseUnit, CourseUnit.lesson_id == Task.lesson_id)
                .where(
                    CourseUnit.course_id == run.course_id,
                    CourseUnit.is_active.is_(True),
                    Task.last_updated_at >= course.last_synced_at,
                )
            ).all()
            task_counts: dict[str, int] = {}
            for status, _unit_id in rows:
                task_counts[status.value] = task_counts.get(status.value, 0) + 1

            return CourseRunStatus(
                run_id=run.run_id,
                course_id=run.course_id,
                course_title=course.title,
                course_url=course.source_url,
                state=run.state,
                start_time=run.start_time,
                end_time=run.end_time,
                total_tasks=run.total_tasks,
                completed_tasks=run.completed_tasks,
                failed_tasks=run.failed_tasks,
                estimated_cost=run.estimated_cost,
                task_counts=task_counts,
            )

    def start(self, *, course_id: int, total_tasks: int) -> str:
        with self._database.session() as session:
            run = RunRepository(session).create(
                course_id=course_id,
                execution_mode=ExecutionMode.AUTO,
                total_tasks=total_tasks,
            )
            session.flush()
            return run.run_id

    def finish(
        self,
        run_id: str,
        *,
        state: RunState,
        completed_tasks: int,
        failed_tasks: int,
        estimated_cost: Decimal = Decimal("0"),
    ) -> None:
        with self._database.session() as session:
            run = session.get(Run, run_id)
            if run is None:
                raise ValueError(f"Run {run_id} does not exist")
            RunRepository(session).finish(
                run,
                state=state,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                estimated_cost=estimated_cost,
            )

    def mark_task(self, step_id: int, status: TaskStatus) -> None:
        with self._database.session() as session:
            task = session.get(Task, step_id)
            if task is None:
                raise ValueError(f"Task {step_id} does not exist")
            task.status = status


def _assignment_id(assignment_map: object, step_id: int) -> int | None:
    if not isinstance(assignment_map, dict):
        return None
    value = assignment_map.get(str(step_id))
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None
