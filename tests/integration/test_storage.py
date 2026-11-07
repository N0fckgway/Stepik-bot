import stat
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from stepik_agent.core.states import EventSeverity, ExecutionMode, RunState, TaskStatus
from stepik_agent.storage.course_runs import CourseRunStore
from stepik_agent.storage.database import Database
from stepik_agent.storage.models import Course, Run
from stepik_agent.storage.repositories import (
    CatalogRepository,
    EventRepository,
    RunRepository,
)


def test_catalog_upsert_preserves_course_lesson_membership(tmp_path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    database.create_schema()

    with database.session() as session:
        catalog = CatalogRepository(session)
        catalog.upsert_course(
            course_id=100707,
            title="Initial title",
            source_url="https://stepik.org/course/100707",
        )
        catalog.upsert_lesson(lesson_id=566987, title="Lesson", language="ru")
        catalog.upsert_unit(
            unit_id=561261,
            course_id=100707,
            section_id=198123,
            lesson_id=566987,
            section_position=1,
            unit_position=1,
        )
        catalog.upsert_task(
            step_id=5108558,
            lesson_id=566987,
            position=1,
            task_type="video",
            progress_id="77-5108558",
            is_passed=True,
            status=TaskStatus.CORRECT,
        )

    with database.session() as session:
        catalog = CatalogRepository(session)
        updated = catalog.upsert_course(
            course_id=100707,
            title="Updated title",
            source_url="https://stepik.org/course/100707",
        )
        tasks = catalog.list_course_tasks(100707)

        assert updated.title == "Updated title"
        assert [task.step_id for task in tasks] == [5108558]
        assert tasks[0].is_passed is True

    database.dispose()


def test_sqlite_database_is_owner_only(tmp_path) -> None:
    path = tmp_path / "agent.db"
    database = Database(f"sqlite:///{path}")

    database.create_schema()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    database.dispose()


def test_transaction_rolls_back_on_integrity_error(tmp_path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    database.create_schema()

    with pytest.raises(IntegrityError), database.session() as session:
        session.add(
            Course(
                course_id=1,
                title="Broken course",
                source_url="https://stepik.org/course/1",
            )
        )
        session.flush()
        session.add(
            Course(
                course_id=1,
                title="Duplicate",
                source_url="https://stepik.org/course/1",
            )
        )

    with database.session() as session:
        assert session.scalar(select(Course).where(Course.course_id == 1)) is None

    database.dispose()


def test_run_and_sanitized_event_are_persisted(tmp_path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    database.create_schema()

    with database.session() as session:
        catalog = CatalogRepository(session)
        catalog.upsert_course(
            course_id=100707,
            title="Course",
            source_url="https://stepik.org/course/100707",
        )
        run = RunRepository(session).create(
            course_id=100707,
            execution_mode=ExecutionMode.DRY_RUN,
            total_tasks=2,
        )
        session.flush()
        EventRepository(session).append(
            run_id=run.run_id,
            event_type="sync.completed",
            severity=EventSeverity.INFO,
            sanitized_payload={"tasks": 2},
        )
        RunRepository(session).finish(
            run,
            state=RunState.COMPLETED,
            completed_tasks=2,
            failed_tasks=0,
            estimated_cost=Decimal("0"),
        )

    with database.session() as session:
        stored_run = session.scalar(select(Run))
        assert stored_run is not None
        assert stored_run.state is RunState.COMPLETED
        assert stored_run.events[0].sanitized_payload == {"tasks": 2}

    database.dispose()


def test_sync_upsert_preserves_unchanged_failed_task_status(tmp_path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    database.create_schema()

    with database.session() as session:
        catalog = CatalogRepository(session)
        catalog.upsert_lesson(lesson_id=10, title="Lesson", language="en")
        catalog.upsert_task(
            step_id=20,
            lesson_id=10,
            position=1,
            task_type="code",
            progress_id="77-20",
            task_hash="same-hash",
            status=TaskStatus.FAILED,
        )

    with database.session() as session:
        task = CatalogRepository(session).upsert_task(
            step_id=20,
            lesson_id=10,
            position=1,
            task_type="code",
            progress_id="77-20",
            task_hash="same-hash",
            status=TaskStatus.DISCOVERED,
        )
        assert task.status is TaskStatus.FAILED

    database.dispose()


def test_course_queue_excludes_tasks_not_seen_in_latest_sync(tmp_path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    database.create_schema()
    synced_at = datetime.now(UTC)

    with database.session() as session:
        catalog = CatalogRepository(session)
        catalog.upsert_course(
            course_id=762,
            title="Linux",
            source_url="https://stepik.org/course/762",
            synced_at=synced_at,
        )
        catalog.upsert_lesson(lesson_id=10, title="Lesson", language="ru")
        catalog.upsert_unit(
            unit_id=20,
            course_id=762,
            section_id=30,
            lesson_id=10,
            section_position=1,
            unit_position=1,
        )
        catalog.upsert_task(
            step_id=100,
            lesson_id=10,
            position=1,
            task_type="choice",
            progress_id="1-100",
            updated_at=synced_at - timedelta(seconds=1),
        )
        catalog.upsert_task(
            step_id=101,
            lesson_id=10,
            position=2,
            task_type="choice",
            progress_id="1-101",
            updated_at=synced_at + timedelta(seconds=1),
        )

    assert [task.step_id for task in CourseRunStore(database).list_tasks(762)] == [101]
    database.dispose()
