import asyncio
from decimal import Decimal

import pytest

from stepik_agent.ai.budget import BudgetedAITransport, SharedAIBudget
from stepik_agent.ai.transport import AITransport, CompletionRequest, CompletionResult
from stepik_agent.core.errors import (
    AIRunBudgetExceededError,
    AITaskBudgetExceededError,
)


class CostTransport(AITransport):
    def __init__(self, costs: list[Decimal | None]) -> None:
        self.costs = costs
        self.closed = False

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        return CompletionResult("{}", request.model, 1, 1, self.costs.pop(0))

    async def aclose(self) -> None:
        self.closed = True


def request() -> CompletionRequest:
    return CompletionRequest("model", [], 10, True)


def test_call_limit_is_checked_before_next_provider_request() -> None:
    delegate = CostTransport([None])
    transport = BudgetedAITransport(delegate, max_calls=1, max_cost_usd=Decimal("1"))

    asyncio.run(transport.complete(request()))
    with pytest.raises(AITaskBudgetExceededError, match="maximum AI calls"):
        asyncio.run(transport.complete(request()))

    assert transport.calls == 1
    assert transport.has_unknown_cost is True


def test_reported_cost_stops_followup_calls() -> None:
    delegate = CostTransport([Decimal("0.50")])
    transport = BudgetedAITransport(delegate, max_calls=3, max_cost_usd=Decimal("0.50"))

    asyncio.run(transport.complete(request()))
    with pytest.raises(AITaskBudgetExceededError, match="reached the task budget"):
        asyncio.run(transport.complete(request()))

    assert transport.calls == 1
    assert transport.known_cost_usd == Decimal("0.50")


def test_shared_budget_limits_calls_across_task_transports() -> None:
    shared_budget = SharedAIBudget(Decimal("0.50"))
    first = BudgetedAITransport(
        CostTransport([Decimal("0.25")]),
        max_calls=1,
        max_cost_usd=Decimal("1"),
        shared_budget=shared_budget,
    )
    second = BudgetedAITransport(
        CostTransport([Decimal("0.25")]),
        max_calls=1,
        max_cost_usd=Decimal("1"),
        shared_budget=shared_budget,
    )

    asyncio.run(first.complete(request()))
    asyncio.run(second.complete(request()))
    third = BudgetedAITransport(
        CostTransport([Decimal("0.01")]),
        max_calls=1,
        max_cost_usd=Decimal("1"),
        shared_budget=shared_budget,
    )
    with pytest.raises(AIRunBudgetExceededError, match="run budget"):
        asyncio.run(third.complete(request()))

    assert shared_budget.calls == 2
    assert shared_budget.known_cost_usd == Decimal("0.50")


def test_shared_budget_limits_unknown_cost_calls_across_run() -> None:
    shared_budget = SharedAIBudget(Decimal("5"), max_calls=1)
    transport = BudgetedAITransport(
        CostTransport([None, None]),
        max_calls=2,
        max_cost_usd=Decimal("1"),
        shared_budget=shared_budget,
    )

    asyncio.run(transport.complete(request()))
    with pytest.raises(AIRunBudgetExceededError, match="maximum AI calls per run"):
        asyncio.run(transport.complete(request()))

    assert shared_budget.calls == 1
    assert shared_budget.has_unknown_cost is True
