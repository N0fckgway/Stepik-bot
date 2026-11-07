import pytest

from stepik_agent.core.errors import InvalidStepikUrlError
from stepik_agent.stepik.urls import StepikTargetType, parse_stepik_url


def test_parse_lesson_step_url_from_specification() -> None:
    target = parse_stepik_url("https://stepik.org/lesson/41871/step/1?unit=20186")

    assert target.target_type is StepikTargetType.STEP
    assert target.lesson_id == 41871
    assert target.step_position == 1
    assert target.unit_id == 20186
    assert target.course_id is None


@pytest.mark.parametrize(
    ("url", "expected_course_id"),
    [
        ("https://stepik.org/course/67/syllabus", 67),
        ("https://stepik.org/course/operating-systems-12345/syllabus", 12345),
    ],
)
def test_parse_course_url(url: str, expected_course_id: int) -> None:
    target = parse_stepik_url(url)

    assert target.target_type is StepikTargetType.COURSE
    assert target.course_id == expected_course_id


def test_parse_course_lesson_step_url() -> None:
    target = parse_stepik_url("https://stepik.org/course/67/lesson/41871/step/3?unit=20186")

    assert target.target_type is StepikTargetType.STEP
    assert target.course_id == 67
    assert target.lesson_id == 41871
    assert target.step_position == 3
    assert target.unit_id == 20186


@pytest.mark.parametrize(
    "url",
    [
        "http://stepik.org/lesson/1/step/1",
        "https://example.com/lesson/1/step/1",
        "https://stepik.org/lesson/nope/step/1",
        "https://stepik.org/lesson/1/step/0",
        "https://stepik.org/catalog",
    ],
)
def test_reject_invalid_or_unsupported_url(url: str) -> None:
    with pytest.raises(InvalidStepikUrlError):
        parse_stepik_url(url)
