"""Read a concrete Stepik step and nearby theory without creating an attempt."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from stepik_agent.core.errors import StepikApiError
from stepik_agent.stepik.urls import StepikTarget, StepikTargetType
from stepik_agent.tasks.models import StepTaskData


class TaskReadClient(Protocol):
    async def get_by_id(self, resource: str, object_id: int) -> dict[str, Any]: ...

    async def get_many(
        self,
        resource: str,
        object_ids: Sequence[int | str],
        *,
        batch_size: int = 50,
    ) -> list[dict[str, Any]]: ...


class StepikTaskLoader:
    def __init__(self, client: TaskReadClient, *, material_window: int = 10) -> None:
        if material_window < 0:
            raise ValueError("material_window must be nonnegative")
        self._client = client
        self._material_window = material_window

    async def load(self, target: StepikTarget) -> StepTaskData:
        if (
            target.target_type is not StepikTargetType.STEP
            or target.lesson_id is None
            or target.step_position is None
        ):
            raise StepikApiError("Dry run requires a URL for one concrete lesson step")

        lesson = await self._client.get_by_id("lessons", target.lesson_id)
        step_ids = _positive_integer_list(lesson.get("steps"), field="lesson.steps")
        if target.step_position > len(step_ids):
            if lesson.get("is_public") is False:
                raise StepikApiError(
                    "Step is unavailable in the current API response; this restricted lesson "
                    "may require a Stepik OAuth token"
                )
            raise StepikApiError("Step position is outside the lesson")
        step_id = step_ids[target.step_position - 1]
        step = await self._client.get_by_id("steps", step_id)
        if _positive_int(step.get("id"), field="step.id") != step_id:
            raise StepikApiError("Stepik returned a different step than requested")
        if _positive_int(step.get("lesson"), field="step.lesson") != target.lesson_id:
            raise StepikApiError("Step does not belong to the requested lesson")

        block = _mapping(step.get("block"), field="step.block")
        block_name = _required_string(block.get("name"), field="step.block.name")
        statement = block.get("text", "")
        if not isinstance(statement, str):
            raise StepikApiError("step.block.text must be a string")
        options = block.get("options", {})
        if not isinstance(options, Mapping):
            raise StepikApiError("step.block.options must be an object")

        preceding_start = max(0, target.step_position - 1 - self._material_window)
        preceding_ids = step_ids[preceding_start : target.step_position - 1]
        materials: list[str] = []
        if preceding_ids:
            preceding_steps = await self._client.get_many("steps", preceding_ids)
            by_id = {
                _positive_int(item.get("id"), field="step.id"): item for item in preceding_steps
            }
            for preceding_id in preceding_ids:
                item = by_id.get(preceding_id)
                if item is None:
                    raise StepikApiError(
                        f"Stepik response is missing preceding step {preceding_id}"
                    )
                preceding_block = _mapping(item.get("block"), field="step.block")
                if preceding_block.get("name") == "text":
                    text = preceding_block.get("text")
                    if isinstance(text, str) and text.strip():
                        materials.append(text)

        return StepTaskData(
            step_id=step_id,
            block_name=block_name,
            statement_html=statement,
            options=dict(options),
            preceding_materials_html=tuple(materials),
        )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise StepikApiError(f"{field} must be an object with string keys")
    return value


def _positive_integer_list(value: object, *, field: str) -> list[int]:
    if not isinstance(value, list):
        raise StepikApiError(f"{field} must be a list")
    return [_positive_int(item, field=field) for item in value]


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise StepikApiError(f"{field} must be a positive integer")


def _required_string(value: object, *, field: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise StepikApiError(f"{field} must be a non-empty string")
