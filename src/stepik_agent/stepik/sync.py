import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, TypeVar

from stepik_agent.core.errors import StepikApiError
from stepik_agent.core.states import CourseStatus, TaskStatus
from stepik_agent.stepik.urls import StepikTarget

T = TypeVar("T")


class CatalogClient(Protocol):
    async def get_by_id(self, resource: str, object_id: int) -> dict[str, Any]: ...

    async def get_many(
        self,
        resource: str,
        object_ids: Sequence[int | str],
        *,
        batch_size: int = 50,
    ) -> list[dict[str, Any]]: ...


class CatalogWriter(Protocol):
    def upsert_course(
        self,
        *,
        course_id: int,
        title: str,
        source_url: str,
        status: CourseStatus = CourseStatus.ACTIVE,
        synced_at: datetime | None = None,
    ) -> object: ...

    def upsert_lesson(
        self,
        *,
        lesson_id: int,
        title: str,
        language: str | None,
        updated_at: datetime | None = None,
    ) -> object: ...

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
    ) -> object: ...

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
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class SyncResult:
    course_id: int
    sections: int
    units: int
    lessons: int
    tasks: int
    passed_tasks: int
    missing_assignment_ids: tuple[int, ...] = ()


class CourseSyncService:
    """Synchronize a course catalog and progress using read-only Stepik calls."""

    def __init__(self, *, client: CatalogClient, writer: CatalogWriter) -> None:
        self._client = client
        self._writer = writer

    async def sync(self, target: StepikTarget) -> SyncResult:
        course_id = await self._resolve_course_id(target)
        course = await self._client.get_by_id("courses", course_id)
        section_ids = _integer_list(course.get("sections"), field="course.sections")
        sections = await self._client.get_many("sections", section_ids)
        sections_by_id = _index_by_integer_id(sections, resource="sections")

        ordered_sections: list[dict[str, Any]] = []
        unit_ids: list[int] = []
        for section_id in section_ids:
            section = _required(sections_by_id, section_id, resource="section")
            ordered_sections.append(section)
            unit_ids.extend(_integer_list(section.get("units"), field="section.units"))

        units = await self._client.get_many("units", unit_ids)
        units_by_id = _index_by_integer_id(units, resource="units")
        assignment_ids = _deduplicate(
            [
                assignment_id
                for unit in units
                for assignment_id in _integer_list(
                    unit.get("assignments", []), field="unit.assignments"
                )
            ]
        )
        assignments = await self._client.get_many("assignments", assignment_ids)
        assignments_by_id = _index_by_integer_id(assignments, resource="assignments")
        missing_assignment_ids = tuple(
            assignment_id
            for assignment_id in assignment_ids
            if assignment_id not in assignments_by_id
        )
        lesson_ids = _deduplicate(
            [_positive_int(unit.get("lesson"), field="unit.lesson") for unit in units]
        )
        lessons = await self._client.get_many("lessons", lesson_ids)
        lessons_by_id = _index_by_integer_id(lessons, resource="lessons")

        step_ids: list[int] = []
        for lesson_id in lesson_ids:
            lesson = _required(lessons_by_id, lesson_id, resource="lesson")
            step_ids.extend(_integer_list(lesson.get("steps"), field="lesson.steps"))
        step_ids = _deduplicate(step_ids)
        steps = await self._client.get_many("steps", step_ids)
        steps_by_id = _index_by_integer_id(steps, resource="steps")

        progress_ids = _deduplicate(
            [
                progress_id
                for step in steps
                if (progress_id := _optional_string(step.get("progress")))
            ]
        )
        progresses = await self._client.get_many("progresses", progress_ids)
        progresses_by_id = _index_by_string_id(progresses, resource="progresses")
        for progress_id in progress_ids:
            if progress_id not in progresses_by_id:
                raise StepikApiError(f"Stepik response is missing progress {progress_id}")

        self._writer.upsert_course(
            course_id=course_id,
            title=_required_string(course.get("title"), field="course.title"),
            source_url=f"https://stepik.org/course/{course_id}",
            status=CourseStatus.ACTIVE,
        )
        for lesson_id in lesson_ids:
            lesson = _required(lessons_by_id, lesson_id, resource="lesson")
            self._writer.upsert_lesson(
                lesson_id=lesson_id,
                title=_required_string(lesson.get("title"), field="lesson.title"),
                language=_optional_string(lesson.get("language")),
            )

        section_positions = {
            _positive_int(section.get("id"), field="section.id"): _optional_positive_int(
                section.get("position"), field="section.position"
            )
            for section in ordered_sections
        }
        for unit_id in unit_ids:
            unit = _required(units_by_id, unit_id, resource="unit")
            section_id = _positive_int(unit.get("section"), field="unit.section")
            assignment_map: dict[str, int] = {}
            for assignment_id in _integer_list(
                unit.get("assignments", []), field="unit.assignments"
            ):
                assignment = assignments_by_id.get(assignment_id)
                if assignment is None:
                    # Lessons and steps remain authoritative for course membership. An old or
                    # inaccessible assignment only removes the optional view-report association.
                    continue
                returned_unit_id = _positive_int(assignment.get("unit"), field="assignment.unit")
                if returned_unit_id != unit_id:
                    raise StepikApiError(
                        f"Assignment {assignment_id} belongs to unexpected unit {returned_unit_id}"
                    )
                assignment_step_id = _positive_int(assignment.get("step"), field="assignment.step")
                assignment_map[str(assignment_step_id)] = assignment_id
            self._writer.upsert_unit(
                unit_id=unit_id,
                course_id=course_id,
                section_id=section_id,
                lesson_id=_positive_int(unit.get("lesson"), field="unit.lesson"),
                section_position=section_positions.get(section_id),
                unit_position=_positive_int(unit.get("position"), field="unit.position"),
                is_active=_boolean(unit.get("is_active", True), field="unit.is_active"),
                assignment_map=assignment_map,
            )

        passed_tasks = 0
        for step_id in step_ids:
            step = _required(steps_by_id, step_id, resource="step")
            block = step.get("block")
            if not isinstance(block, dict):
                raise StepikApiError("step.block must be an object")
            task_type = _required_string(block.get("name"), field="step.block.name")
            progress_id = _optional_string(step.get("progress"))
            progress = progresses_by_id.get(progress_id, {}) if progress_id else {}
            is_passed = _boolean(progress.get("is_passed", False), field="progress.is_passed")
            passed_tasks += int(is_passed)
            self._writer.upsert_task(
                step_id=step_id,
                lesson_id=_positive_int(step.get("lesson"), field="step.lesson"),
                position=_positive_int(step.get("position"), field="step.position"),
                task_type=task_type,
                progress_id=progress_id,
                task_hash=_block_hash(block),
                maximum_score=_nonnegative_decimal(step.get("worth") or 0, field="step.worth"),
                achieved_score=_nonnegative_decimal(
                    progress.get("score") or 0, field="progress.score"
                ),
                is_attempted=_boolean(
                    progress.get("is_attempted", False), field="progress.is_attempted"
                ),
                is_passed=is_passed,
                status=TaskStatus.CORRECT if is_passed else TaskStatus.DISCOVERED,
            )

        return SyncResult(
            course_id=course_id,
            sections=len(section_ids),
            units=len(unit_ids),
            lessons=len(lesson_ids),
            tasks=len(step_ids),
            passed_tasks=passed_tasks,
            missing_assignment_ids=missing_assignment_ids,
        )

    async def _resolve_course_id(self, target: StepikTarget) -> int:
        if target.course_id is not None:
            return target.course_id
        if target.unit_id is None:
            raise StepikApiError("A lesson URL without unit cannot be resolved to one course")
        unit = await self._client.get_by_id("units", target.unit_id)
        section_id = _positive_int(unit.get("section"), field="unit.section")
        section = await self._client.get_by_id("sections", section_id)
        return _positive_int(section.get("course"), field="section.course")


def _required(
    mapping: dict[int, dict[str, Any]], object_id: int, *, resource: str
) -> dict[str, Any]:
    try:
        return mapping[object_id]
    except KeyError as error:
        raise StepikApiError(f"Stepik response is missing {resource} {object_id}") from error


def _index_by_integer_id(
    objects: list[dict[str, Any]], *, resource: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for item in objects:
        object_id = _positive_int(item.get("id"), field=f"{resource}.id")
        if object_id in indexed:
            raise StepikApiError(f"Stepik returned duplicate {resource} id {object_id}")
        indexed[object_id] = item
    return indexed


def _index_by_string_id(
    objects: list[dict[str, Any]], *, resource: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in objects:
        object_id = _required_string(item.get("id"), field=f"{resource}.id")
        if object_id in indexed:
            raise StepikApiError(f"Stepik returned duplicate {resource} id {object_id}")
        indexed[object_id] = item
    return indexed


def _integer_list(value: object, *, field: str) -> list[int]:
    if not isinstance(value, list):
        raise StepikApiError(f"{field} must be a list")
    return [_positive_int(item, field=field) for item in value]


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        parsed = int(value)
    else:
        raise StepikApiError(f"{field} must be a positive integer")
    if parsed <= 0:
        raise StepikApiError(f"{field} must be a positive integer")
    return parsed


def _optional_positive_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field=field)


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StepikApiError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise StepikApiError(f"{field} must be a boolean")
    return value


def _nonnegative_decimal(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise StepikApiError(f"{field} must be a non-negative number")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise StepikApiError(f"{field} must be a non-negative number") from error
    if not parsed.is_finite() or parsed < 0:
        raise StepikApiError(f"{field} must be a non-negative number")
    return parsed


def _block_hash(block: dict[str, Any]) -> str:
    canonical = json.dumps(block, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _deduplicate(values: list[T]) -> list[T]:
    return list(dict.fromkeys(values))
