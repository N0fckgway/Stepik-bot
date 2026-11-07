import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from stepik_agent.core.errors import StepikApiError
from stepik_agent.stepik.task_loader import StepikTaskLoader
from stepik_agent.stepik.urls import parse_stepik_url


class FakeClient:
    def __init__(self) -> None:
        self.objects: dict[str, dict[int, dict[str, Any]]] = {
            "lessons": {1: {"id": 1, "steps": [10, 20, 30]}},
            "steps": {
                10: {
                    "id": 10,
                    "lesson": 1,
                    "block": {"name": "text", "text": "<p>Theory</p>"},
                },
                20: {
                    "id": 20,
                    "lesson": 1,
                    "block": {"name": "video", "text": ""},
                },
                30: {
                    "id": 30,
                    "lesson": 1,
                    "block": {
                        "name": "number",
                        "text": "<p>2 + 2?</p>",
                        "options": {"precision": 0},
                    },
                },
            },
        }

    async def get_by_id(self, resource: str, object_id: int) -> dict[str, Any]:
        return self.objects[resource][object_id]

    async def get_many(
        self,
        resource: str,
        object_ids: Sequence[int | str],
        *,
        batch_size: int = 50,
    ) -> list[dict[str, Any]]:
        return [self.objects[resource][int(object_id)] for object_id in object_ids]


def test_loader_resolves_position_and_collects_only_preceding_theory() -> None:
    target = parse_stepik_url("https://stepik.org/lesson/1/step/3")

    task = asyncio.run(StepikTaskLoader(FakeClient()).load(target))

    assert task.step_id == 30
    assert task.block_name == "number"
    assert task.options == {"precision": 0}
    assert task.preceding_materials_html == ("<p>Theory</p>",)


def test_loader_rejects_position_outside_lesson() -> None:
    target = parse_stepik_url("https://stepik.org/lesson/1/step/4")

    with pytest.raises(StepikApiError, match="outside the lesson"):
        asyncio.run(StepikTaskLoader(FakeClient()).load(target))


def test_loader_requires_concrete_step_url() -> None:
    target = parse_stepik_url("https://stepik.org/lesson/1")

    with pytest.raises(StepikApiError, match="one concrete"):
        asyncio.run(StepikTaskLoader(FakeClient()).load(target))


def test_loader_explains_incomplete_restricted_lesson() -> None:
    client = FakeClient()
    client.objects["lessons"][1] = {"id": 1, "steps": [10], "is_public": False}
    target = parse_stepik_url("https://stepik.org/lesson/1/step/4")

    with pytest.raises(StepikApiError, match="may require a Stepik OAuth token"):
        asyncio.run(StepikTaskLoader(client).load(target))
