import asyncio
from typing import Any

import pytest

from stepik_agent.core.errors import StepikApiError
from stepik_agent.core.states import TaskStatus
from stepik_agent.stepik.sync import CourseSyncService
from stepik_agent.stepik.urls import parse_stepik_url


class FakeClient:
    def __init__(self) -> None:
        self.objects: dict[str, dict[int | str, dict[str, Any]]] = {
            "courses": {
                100707: {"id": 100707, "title": "Python", "sections": [198123]},
            },
            "sections": {
                198123: {"id": 198123, "course": 100707, "position": 1, "units": [561261]},
            },
            "units": {
                561261: {
                    "id": 561261,
                    "section": 198123,
                    "lesson": 566987,
                    "position": 1,
                    "is_active": True,
                    "assignments": [9001, 9002],
                }
            },
            "assignments": {
                9001: {"id": 9001, "unit": 561261, "step": 5108558},
                9002: {"id": 9002, "unit": 561261, "step": 5446281},
            },
            "lessons": {
                566987: {
                    "id": 566987,
                    "title": "How to learn",
                    "language": "ru",
                    "steps": [5108558, 5446281],
                }
            },
            "steps": {
                5108558: {
                    "id": 5108558,
                    "lesson": 566987,
                    "position": 1,
                    "block": {"name": "video", "text": ""},
                    "progress": "77-5108558",
                    "worth": 0,
                },
                5446281: {
                    "id": 5446281,
                    "lesson": 566987,
                    "position": 2,
                    "block": {"name": "text", "text": "Intro"},
                    "progress": "77-5446281",
                    "worth": 0,
                },
            },
            "progresses": {
                "77-5108558": {
                    "id": "77-5108558",
                    "score": 0,
                    "is_attempted": True,
                    "is_passed": True,
                },
                "77-5446281": {
                    "id": "77-5446281",
                    "score": 0,
                    "is_attempted": False,
                    "is_passed": False,
                },
            },
        }

    async def get_by_id(self, resource: str, object_id: int) -> dict[str, Any]:
        return self.objects[resource][object_id]

    async def get_many(
        self,
        resource: str,
        object_ids: list[int | str],
        *,
        batch_size: int = 50,
    ) -> list[dict[str, Any]]:
        return [self.objects[resource][object_id] for object_id in object_ids]


class RecordingWriter:
    def __init__(self) -> None:
        self.courses: list[dict[str, Any]] = []
        self.lessons: list[dict[str, Any]] = []
        self.units: list[dict[str, Any]] = []
        self.tasks: list[dict[str, Any]] = []

    def upsert_course(self, **values: Any) -> object:
        self.courses.append(values)
        return values

    def upsert_lesson(self, **values: Any) -> object:
        self.lessons.append(values)
        return values

    def upsert_unit(self, **values: Any) -> object:
        self.units.append(values)
        return values

    def upsert_task(self, **values: Any) -> object:
        self.tasks.append(values)
        return values


def test_sync_resolves_lesson_url_and_persists_progress() -> None:
    writer = RecordingWriter()
    service = CourseSyncService(client=FakeClient(), writer=writer)
    target = parse_stepik_url("https://stepik.org/lesson/566987/step/1?unit=561261")

    result = asyncio.run(service.sync(target))

    assert result.course_id == 100707
    assert result.sections == 1
    assert result.units == 1
    assert result.lessons == 1
    assert result.tasks == 2
    assert result.passed_tasks == 1
    assert writer.courses[0]["course_id"] == 100707
    assert writer.units[0]["section_id"] == 198123
    assert writer.units[0]["assignment_map"] == {"5108558": 9001, "5446281": 9002}
    assert writer.tasks[0]["status"] is TaskStatus.CORRECT
    assert writer.tasks[1]["status"] is TaskStatus.DISCOVERED
    assert len(writer.tasks[0]["task_hash"]) == 64


def test_sync_rejects_missing_external_contract_fields() -> None:
    client = FakeClient()
    client.objects["courses"][100707].pop("sections")
    service = CourseSyncService(client=client, writer=RecordingWriter())
    target = parse_stepik_url("https://stepik.org/course/100707")

    with pytest.raises(StepikApiError, match=r"course\.sections"):
        asyncio.run(service.sync(target))


def test_sync_rejects_string_boolean_from_external_api() -> None:
    client = FakeClient()
    client.objects["progresses"]["77-5446281"]["is_passed"] = "false"
    service = CourseSyncService(client=client, writer=RecordingWriter())
    target = parse_stepik_url("https://stepik.org/course/100707")

    with pytest.raises(StepikApiError, match=r"progress\.is_passed"):
        asyncio.run(service.sync(target))


class MissingProgressClient(FakeClient):
    async def get_many(
        self,
        resource: str,
        object_ids: list[int | str],
        *,
        batch_size: int = 50,
    ) -> list[dict[str, Any]]:
        objects = await super().get_many(resource, object_ids, batch_size=batch_size)
        return objects[:-1] if resource == "progresses" else objects


def test_sync_rejects_missing_progress_object() -> None:
    service = CourseSyncService(client=MissingProgressClient(), writer=RecordingWriter())
    target = parse_stepik_url("https://stepik.org/course/100707")

    with pytest.raises(StepikApiError, match="missing progress"):
        asyncio.run(service.sync(target))


class MissingAssignmentClient(FakeClient):
    async def get_many(
        self,
        resource: str,
        object_ids: list[int | str],
        *,
        batch_size: int = 50,
    ) -> list[dict[str, Any]]:
        objects = await super().get_many(resource, object_ids, batch_size=batch_size)
        if resource == "assignments":
            return [item for item in objects if item["id"] != 9002]
        return objects


def test_sync_reports_but_does_not_abort_on_missing_optional_assignment() -> None:
    writer = RecordingWriter()
    service = CourseSyncService(client=MissingAssignmentClient(), writer=writer)
    target = parse_stepik_url("https://stepik.org/course/100707")

    result = asyncio.run(service.sync(target))

    assert result.tasks == 2
    assert result.missing_assignment_ids == (9002,)
    assert writer.units[0]["assignment_map"] == {"5108558": 9001}
