import re
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import parse_qs, urlparse

from stepik_agent.core.errors import InvalidStepikUrlError

_COURSE_ID_RE = re.compile(r"(?:^|-)(?P<id>[1-9]\d*)$")
_POSITIVE_INT_RE = re.compile(r"^[1-9]\d*$")
_ALLOWED_HOSTS = frozenset({"stepik.org", "www.stepik.org"})


class StepikTargetType(StrEnum):
    COURSE = "course"
    LESSON = "lesson"
    STEP = "step"


@dataclass(frozen=True, slots=True)
class StepikTarget:
    source_url: str
    target_type: StepikTargetType
    course_id: int | None = None
    unit_id: int | None = None
    lesson_id: int | None = None
    step_position: int | None = None


def _positive_int(value: str, *, field: str) -> int:
    if not _POSITIVE_INT_RE.fullmatch(value):
        raise InvalidStepikUrlError(f"{field} must be a positive integer")
    return int(value)


def _course_id(segment: str) -> int:
    match = _COURSE_ID_RE.search(segment)
    if match is None:
        raise InvalidStepikUrlError("course path must end with a numeric course id")
    return int(match.group("id"))


def parse_stepik_url(url: str) -> StepikTarget:
    """Parse a supported HTTPS Stepik course, lesson, or step URL.

    The number after ``/step/`` is a position inside a lesson, not a global step id.
    When present, ``unit`` is retained because a lesson may belong to multiple courses.
    """

    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in _ALLOWED_HOSTS:
        raise InvalidStepikUrlError("expected an HTTPS URL on stepik.org")

    parts = [part for part in parsed.path.split("/") if part]
    query = parse_qs(parsed.query)
    unit_id: int | None = None
    if unit_values := query.get("unit"):
        unit_id = _positive_int(unit_values[0], field="unit")

    if len(parts) >= 2 and parts[0] == "course":
        course_id = _course_id(parts[1])
        if "lesson" not in parts:
            return StepikTarget(url, StepikTargetType.COURSE, course_id=course_id)
        lesson_index = parts.index("lesson")
        return _parse_lesson_path(
            url,
            parts,
            lesson_index=lesson_index,
            course_id=course_id,
            unit_id=unit_id,
        )

    if len(parts) >= 2 and parts[0] == "lesson":
        return _parse_lesson_path(url, parts, lesson_index=0, unit_id=unit_id)

    raise InvalidStepikUrlError("URL does not contain a supported course or lesson target")


def _parse_lesson_path(
    source_url: str,
    parts: list[str],
    *,
    lesson_index: int,
    course_id: int | None = None,
    unit_id: int | None = None,
) -> StepikTarget:
    try:
        lesson_id = _positive_int(parts[lesson_index + 1], field="lesson id")
    except IndexError as error:
        raise InvalidStepikUrlError("lesson id is missing") from error

    trailing = parts[lesson_index + 2 :]
    if not trailing:
        return StepikTarget(
            source_url,
            StepikTargetType.LESSON,
            course_id=course_id,
            unit_id=unit_id,
            lesson_id=lesson_id,
        )

    if len(trailing) >= 2 and trailing[0] == "step":
        step_position = _positive_int(trailing[1], field="step position")
        return StepikTarget(
            source_url,
            StepikTargetType.STEP,
            course_id=course_id,
            unit_id=unit_id,
            lesson_id=lesson_id,
            step_position=step_position,
        )

    raise InvalidStepikUrlError("unsupported content after lesson id")
