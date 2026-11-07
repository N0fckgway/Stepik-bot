import asyncio
import json
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from stepik_agent.ai.gateway import AIRequestEvent, AnyModelGateway
from stepik_agent.ai.models import (
    DiagnosticResult,
    ErrorContext,
    ReviewResult,
    SolutionCandidate,
    TaskContext,
)
from stepik_agent.ai.transport import AITransport, CompletionRequest, CompletionResult
from stepik_agent.core.errors import AIGatewayError


class FakeTransport(AITransport):
    def __init__(self, responses: Sequence[CompletionResult | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[CompletionRequest] = []
        self.closed = False

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self) -> None:
        self.closed = True


def task_context() -> TaskContext:
    return TaskContext(
        step_id=42,
        task_type="string",
        statement="Reverse the supplied string.",
        expected_answer_format="Plain text",
    )


def candidate_payload(answer: str = "cba") -> dict[str, object]:
    return {
        "task_type": "string",
        "answer": answer,
        "language": None,
        "explanation": "The characters are reversed.",
        "assumptions": [],
        "confidence": 0.98,
        "test_cases": [{"description": "Basic case", "input": "abc", "expected_output": "cba"}],
        "warnings": [],
    }


def completion(payload: dict[str, object], *, model: str = "model-a") -> CompletionResult:
    return CompletionResult(
        content=json.dumps(payload),
        model=model,
        input_tokens=100,
        output_tokens=50,
    )


def test_generate_solution_returns_validated_candidate() -> None:
    transport = FakeTransport([completion(candidate_payload())])
    gateway = AnyModelGateway(
        transport,
        primary_model="model-a",
        review_model="review-model",
    )

    candidate = asyncio.run(gateway.generate_solution(task_context()))

    assert candidate.answer == "cba"
    assert transport.requests[0].json_mode is True
    assert transport.requests[0].max_output_tokens == 4096
    assert "Untrusted input data" in transport.requests[0].messages[1]["content"]


def test_invalid_response_is_retried_and_observed() -> None:
    events: list[AIRequestEvent] = []
    transport = FakeTransport(
        [
            CompletionResult("not-json", "model-a", 10, 5),
            completion(candidate_payload()),
        ]
    )
    gateway = AnyModelGateway(
        transport,
        primary_model="model-a",
        validation_retries=1,
        request_observer=events.append,
    )

    candidate = asyncio.run(gateway.generate_solution(task_context()))

    assert candidate.answer == "cba"
    assert len(transport.requests) == 2
    assert events[0].error_code == "invalid_structured_response"
    assert events[0].succeeded is False
    assert events[1].succeeded is True


def test_provider_failure_uses_configured_fallback_without_leaking_error() -> None:
    transport = FakeTransport(
        [RuntimeError("private-provider-message"), completion(candidate_payload(), model="backup")]
    )
    gateway = AnyModelGateway(
        transport,
        primary_model="primary",
        fallback_model="backup",
    )

    candidate = asyncio.run(gateway.generate_solution(task_context()))

    assert candidate.answer == "cba"
    assert [request.model for request in transport.requests] == ["primary", "backup"]


def test_bounded_failures_raise_sanitized_error() -> None:
    transport = FakeTransport([RuntimeError("secret-token-value")])
    gateway = AnyModelGateway(transport, primary_model="primary", validation_retries=3)

    with pytest.raises(AIGatewayError) as raised:
        asyncio.run(gateway.generate_solution(task_context()))

    assert len(transport.requests) == 1
    assert "secret-token-value" not in str(raised.value)


def test_all_gateway_roles_have_strict_structured_results() -> None:
    review = {
        "is_acceptable": True,
        "explanation": "The answer satisfies the task.",
        "issues": [],
        "confidence": 0.9,
        "warnings": [],
    }
    repaired = candidate_payload("updated")
    diagnostic = {
        "category": "format",
        "summary": "The answer used the wrong format.",
        "likely_causes": ["Extra markup"],
        "recommended_actions": ["Return plain text"],
        "retryable": True,
        "confidence": 0.8,
        "warnings": [],
    }
    transport = FakeTransport([completion(review), completion(repaired), completion(diagnostic)])
    gateway = AnyModelGateway(
        transport,
        primary_model="model-a",
        review_model="review-model",
    )
    original = SolutionCandidate.model_validate(candidate_payload())

    async def scenario() -> tuple[ReviewResult, SolutionCandidate, DiagnosticResult]:
        review_result = await gateway.review_solution(task_context(), original)
        repaired_result = await gateway.repair_solution(task_context(), original, "Wrong format")
        diagnostic_result = await gateway.analyze_error(
            ErrorContext(
                task=task_context(),
                error_type="format",
                error_message="Expected plain text",
                candidate=original,
            )
        )
        return review_result, repaired_result, diagnostic_result

    review_result, repaired_result, diagnostic_result = asyncio.run(scenario())

    assert isinstance(review_result, ReviewResult)
    assert repaired_result.answer == "updated"
    assert isinstance(diagnostic_result, DiagnosticResult)
    assert [request.model for request in transport.requests] == [
        "review-model",
        "model-a",
        "model-a",
    ]


def test_solution_candidate_rejects_unknown_fields_and_invalid_confidence() -> None:
    payload = candidate_payload()
    payload["unexpected"] = "value"
    payload["confidence"] = 1.5

    with pytest.raises(ValidationError):
        SolutionCandidate.model_validate(payload)


def test_gateway_closes_transport() -> None:
    transport = FakeTransport([])
    gateway = AnyModelGateway(transport, primary_model="model-a")

    asyncio.run(gateway.aclose())

    assert transport.closed is True
