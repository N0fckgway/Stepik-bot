import asyncio

import pytest

httpx = pytest.importorskip("httpx")

from stepik_agent.core.errors import (  # noqa: E402
    AmbiguousStepikWriteError,
    StepikApiError,
    StepikAuthenticationError,
)
from stepik_agent.stepik.write_client import StepikWriteClient, _validate_reply  # noqa: E402


def test_code_submission_contract_and_verdict_polling() -> None:
    requests: list[httpx.Request] = []
    verdict_reads = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal verdict_reads
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-token"
        if request.method == "POST" and request.url.path == "/api/attempts":
            assert request.read() == b'{"attempt":{"step":170493}}'
            return httpx.Response(
                201,
                json={
                    "attempts": [
                        {
                            "id": 501,
                            "step": 170493,
                            "dataset": {"options": ["A", "B"]},
                            "dataset_url": "/media/dataset.txt",
                        }
                    ]
                },
            )
        if request.method == "POST" and request.url.path == "/api/submissions":
            payload = httpx.Response(200, content=request.read()).json()
            assert payload["submission"]["attempt"] == 501
            assert payload["submission"]["reply"] == {
                "code": "swap:\n    ret",
                "language": "asm64",
            }
            assert payload["submission"]["time"].endswith("Z")
            return httpx.Response(
                201,
                json={
                    "submissions": [
                        {"id": 601, "attempt": 501, "status": "evaluation", "hint": None}
                    ]
                },
            )
        if request.method == "GET" and request.url.path == "/api/submissions/601":
            verdict_reads += 1
            status = "evaluation" if verdict_reads == 1 else "correct"
            return httpx.Response(
                200,
                json={"submissions": [{"id": 601, "attempt": 501, "status": status, "hint": "ok"}]},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")

    async def scenario() -> None:
        async with StepikWriteClient(
            access_token="test-token",
            transport=httpx.MockTransport(handler),
        ) as client:
            attempt = await client.create_attempt(170493)
            submission = await client.create_code_submission(
                attempt_id=attempt.attempt_id,
                code="swap:\n    ret",
                language="asm64",
            )
            verdict = await client.wait_for_verdict(
                submission.submission_id,
                timeout_seconds=1,
                initial_delay_seconds=0.001,
                max_delay_seconds=0.001,
            )

        assert attempt.attempt_id == 501
        assert attempt.dataset == {"options": ["A", "B"]}
        assert attempt.dataset_url == "/media/dataset.txt"
        assert verdict.is_correct is True

    asyncio.run(scenario())
    assert [request.method for request in requests].count("POST") == 2


def test_network_failure_during_post_is_never_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection lost", request=request)

    async def scenario() -> None:
        async with StepikWriteClient(
            access_token="test-token",
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(AmbiguousStepikWriteError, match="automatic retry is blocked"):
                await client.create_attempt(170493)

    asyncio.run(scenario())
    assert calls == 1


def test_rejected_oauth_token_has_distinct_error() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401)

    async def scenario() -> None:
        async with StepikWriteClient(
            access_token="expired-token",
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(StepikAuthenticationError, match="stepik-agent auth"):
                await client.create_attempt(170493)

    asyncio.run(scenario())
    assert calls == 1


def test_view_report_uses_official_stepik_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/api/views"
        assert request.read() == b'{"view":{"assignment":9001,"step":5108558}}'
        return httpx.Response(204)

    async def scenario() -> None:
        async with StepikWriteClient(
            access_token="test-token",
            transport=httpx.MockTransport(handler),
        ) as client:
            await client.record_view(step_id=5108558, assignment_id=9001)

    asyncio.run(scenario())
    assert len(requests) == 1


def test_dataset_download_requires_same_origin_path_and_utf8_size_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        if request.url.path == "/media/input.txt":
            return httpx.Response(200, content="привет".encode())
        if request.url.path == "/media/large.txt":
            return httpx.Response(200, content=b"1234")
        return httpx.Response(200, content=b"\xff")

    async def scenario() -> None:
        async with StepikWriteClient(
            access_token="test-token",
            transport=httpx.MockTransport(handler),
        ) as client:
            assert await client.download_dataset("/media/input.txt") == "привет"
            with pytest.raises(StepikApiError, match="same-origin"):
                await client.download_dataset("https://attacker.invalid/input.txt")
            with pytest.raises(StepikApiError, match="size limit"):
                await client.download_dataset("/media/large.txt", max_bytes=3)
            with pytest.raises(StepikApiError, match="UTF-8"):
                await client.download_dataset("/media/binary.txt")

    asyncio.run(scenario())


def test_invalid_success_response_is_ambiguous_and_not_retried() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(201, content=b"not-json")

    async def scenario() -> None:
        async with StepikWriteClient(
            access_token="test-token",
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(AmbiguousStepikWriteError, match="invalid JSON"):
                await client.create_attempt(170493)

    asyncio.run(scenario())
    assert calls == 1


def test_generic_submission_rejects_mixed_answer_fields_before_post() -> None:
    async def scenario() -> None:
        async with StepikWriteClient(access_token="test-token") as client:
            with pytest.raises(ValueError, match="exactly one answer field"):
                await client.create_submission(
                    attempt_id=501,
                    reply={"number": "4", "text": "four"},
                )

    asyncio.run(scenario())


def test_fill_blanks_and_table_replies_pass_strict_write_validation() -> None:
    _validate_reply({"blanks": ["wc", "l"]})
    _validate_reply(
        {
            "choices": [
                {
                    "name_row": "owner",
                    "columns": [
                        {"name": "write", "answer": True},
                        {"name": "read", "answer": False},
                    ],
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="valid Stepik table"):
        _validate_reply(
            {
                "choices": [
                    {
                        "name_row": "owner",
                        "columns": [{"name": "write", "answer": "yes"}],
                    }
                ]
            }
        )


def test_submission_reconciliation_is_scoped_to_attempt_and_read_only() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        assert request.url.path == "/api/submissions"
        assert request.url.params["attempt"] == "501"
        assert request.url.params["order"] == "desc"
        return httpx.Response(
            200,
            json={"submissions": [{"id": 601, "attempt": 501, "status": "correct", "hint": "ok"}]},
        )

    async def scenario() -> None:
        async with StepikWriteClient(
            access_token="test-token",
            transport=httpx.MockTransport(handler),
        ) as client:
            submissions = await client.get_submissions_for_attempt(501)

        assert len(submissions) == 1
        assert submissions[0].submission_id == 601
        assert submissions[0].is_correct is True

    asyncio.run(scenario())
    assert len(requests) == 1
