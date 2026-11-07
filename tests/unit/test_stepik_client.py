import asyncio

import pytest

httpx = pytest.importorskip("httpx")

from stepik_agent.core.errors import StepikApiError  # noqa: E402
from stepik_agent.stepik.client import StepikReadClient  # noqa: E402


def test_get_by_id_parses_flat_collection() -> None:
    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/courses/67"
            assert request.headers["Authorization"] == "Bearer test-token"
            return httpx.Response(
                200,
                json={
                    "meta": {"page": 1, "has_next": False, "has_previous": False},
                    "courses": [{"id": 67, "title": "Test course"}],
                },
            )

        async with StepikReadClient(
            access_token="test-token",
            transport=httpx.MockTransport(handler),
        ) as client:
            course = await client.get_by_id("courses", 67)

        assert course["id"] == 67

    asyncio.run(scenario())


def test_iter_all_follows_meta_has_next() -> None:
    async def scenario() -> None:
        requested_pages: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            page = int(request.url.params["page"])
            requested_pages.append(page)
            return httpx.Response(
                200,
                json={
                    "meta": {"page": page, "has_next": page == 1},
                    "sections": [{"id": page}],
                },
            )

        async with StepikReadClient(transport=httpx.MockTransport(handler)) as client:
            sections = [section async for section in client.iter_all("sections")]

        assert sections == [{"id": 1}, {"id": 2}]
        assert requested_pages == [1, 2]

    asyncio.run(scenario())


def test_get_many_uses_repeated_ids_parameters() -> None:
    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params.get_list("ids[]") == ["10", "20"]
            return httpx.Response(
                200,
                json={"meta": {"page": 1, "has_next": False}, "steps": [{"id": 10}, {"id": 20}]},
            )

        async with StepikReadClient(transport=httpx.MockTransport(handler)) as client:
            steps = await client.get_many("steps", [10, 20])

        assert [step["id"] for step in steps] == [10, 20]

    asyncio.run(scenario())


def test_get_many_follows_pagination_for_each_id_batch() -> None:
    async def scenario() -> None:
        requested_pages: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params.get_list("ids[]") == ["10", "20"]
            page = int(request.url.params["page"])
            requested_pages.append(page)
            return httpx.Response(
                200,
                json={
                    "meta": {"page": page, "has_next": page == 1},
                    "assignments": [{"id": 10 if page == 1 else 20}],
                },
            )

        async with StepikReadClient(transport=httpx.MockTransport(handler)) as client:
            assignments = await client.get_many("assignments", [10, 20])

        assert [item["id"] for item in assignments] == [10, 20]
        assert requested_pages == [1, 2]

    asyncio.run(scenario())


def test_invalid_collection_shape_is_rejected() -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"courses": {}}))
        async with StepikReadClient(transport=transport) as client:
            with pytest.raises(StepikApiError, match="valid 'courses' collection"):
                await client.get_page("courses")

    asyncio.run(scenario())


def test_resource_name_cannot_inject_a_path() -> None:
    async def scenario() -> None:
        async with StepikReadClient(transport=httpx.MockTransport(lambda _: None)) as client:
            with pytest.raises(ValueError, match="resource name"):
                await client.get_page("../submissions")

    asyncio.run(scenario())
