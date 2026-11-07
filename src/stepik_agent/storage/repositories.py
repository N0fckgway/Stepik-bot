from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from stepik_agent.core.states import (
    CourseStatus,
    EventSeverity,
    ExecutionMode,
    RunState,
    TaskStatus,
)
from stepik_agent.storage.models import (
    Course,
    CourseUnit,
    EventLog,
    Lesson,
    Run,
    Task,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class CatalogRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_course(self, course_id: int) -> Course | None:
        return self._session.get(Course, course_id)

    def get_task_by_lesson_position(self, lesson_id: int, position: int) -> Task | None:
        return self._session.scalar(
            select(Task).where(Task.lesson_id == lesson_id, Task.position == position)
        )

    def upsert_course(
        self,
        *,
        course_id: int,
        title: str,
        source_url: str,
        status: CourseStatus = CourseStatus.ACTIVE,
        synced_at: datetime | None = None,
    ) -> Course:
        course = self._session.get(Course, course_id)
        if course is None:
            course = Course(course_id=course_id, title=title, source_url=source_url, status=status)
            self._session.add(course)
        else:
            course.title = title
            course.source_url = source_url
            course.status = status
        course.last_synced_at = synced_at or utc_now()
        return course

    def upsert_lesson(
        self,
        *,
        lesson_id: int,
        title: str,
        language: str | None,
        updated_at: datetime | None = None,
    ) -> Lesson:
        lesson = self._session.get(Lesson, lesson_id)
        if lesson is None:
            lesson = Lesson(lesson_id=lesson_id, title=title, language=language)
            self._session.add(lesson)
        else:
            lesson.title = title
            lesson.language = language
        lesson.last_updated_at = updated_at or utc_now()
        return lesson

    def upsert_unit(
        self,
        *,
        unit_id: int,
        course_id: int,
        section_id: int,
        lesson_id: int,
        section_position: int | None,
        unit_position: int,
        is_active: bool = True,
        assignment_map: dict[str, int] | None = None,
    ) -> CourseUnit:
        unit = self._session.get(CourseUnit, unit_id)
        if unit is None:
            unit = CourseUnit(
                unit_id=unit_id,
                course_id=course_id,
                section_id=section_id,
                lesson_id=lesson_id,
                section_position=section_position,
                unit_position=unit_position,
                is_active=is_active,
                assignment_map=assignment_map or {},
            )
            self._session.add(unit)
        else:
            unit.course_id = course_id
            unit.section_id = section_id
            unit.lesson_id = lesson_id
            unit.section_position = section_position
            unit.unit_position = unit_position
            unit.is_active = is_active
            unit.assignment_map = assignment_map or {}
        return unit

    def upsert_task(
        self,
        *,
        step_id: int,
        lesson_id: int,
        position: int,
        task_type: str,
        progress_id: str | None,
        task_hash: str | None = None,
        maximum_score: Decimal = Decimal("0"),
        achieved_score: Decimal = Decimal("0"),
        is_attempted: bool = False,
        is_passed: bool = False,
        status: TaskStatus = TaskStatus.DISCOVERED,
        updated_at: datetime | None = None,
    ) -> Task:
        task = self._session.get(Task, step_id)
        previous_hash: str | None = None
        if task is None:
            task = Task(
                step_id=step_id, lesson_id=lesson_id, position=position, task_type=task_type
            )
            self._session.add(task)
        else:
            previous_hash = task.task_hash
            task.lesson_id = lesson_id
            task.position = position
            task.task_type = task_type
        task.progress_id = progress_id
        task.task_hash = task_hash
        task.maximum_score = maximum_score
        task.achieved_score = achieved_score
        task.is_attempted = is_attempted
        task.is_passed = is_passed
        if (
            previous_hash is None
            or previous_hash != task_hash
            or status is TaskStatus.CORRECT
            or task.status is TaskStatus.CORRECT
        ):
            task.status = status
        task.last_updated_at = updated_at or utc_now()
        return task

    def list_course_tasks(self, course_id: int) -> list[Task]:
        statement = (
            select(Task)
            .join(Lesson, Lesson.lesson_id == Task.lesson_id)
            .join(CourseUnit, CourseUnit.lesson_id == Lesson.lesson_id)
            .where(CourseUnit.course_id == course_id)
            .order_by(CourseUnit.section_position, CourseUnit.unit_position, Task.position)
        )
        return list(self._session.scalars(statement).all())


class RunRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        course_id: int,
        execution_mode: ExecutionMode,
        total_tasks: int,
        started_at: datetime | None = None,
    ) -> Run:
        run = Run(
            course_id=course_id,
            execution_mode=execution_mode,
            state=RunState.RUNNING,
            total_tasks=total_tasks,
            start_time=started_at or utc_now(),
        )
        self._session.add(run)
        return run

    def finish(
        self,
        run: Run,
        *,
        state: RunState,
        completed_tasks: int,
        failed_tasks: int,
        estimated_cost: Decimal,
        finished_at: datetime | None = None,
    ) -> Run:
        if state is RunState.RUNNING:
            raise ValueError("a finished run cannot remain in RUNNING state")
        run.state = state
        run.completed_tasks = completed_tasks
        run.failed_tasks = failed_tasks
        run.estimated_cost = estimated_cost
        run.end_time = finished_at or utc_now()
        return run


class EventRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def append(
        self,
        *,
        run_id: str,
        event_type: str,
        severity: EventSeverity = EventSeverity.INFO,
        step_id: int | None = None,
        sanitized_payload: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> EventLog:
        event_log = EventLog(
            run_id=run_id,
            step_id=step_id,
            severity=severity,
            event_type=event_type,
            sanitized_payload=sanitized_payload or {},
            created_at=created_at or utc_now(),
        )
        self._session.add(event_log)
        return event_log
