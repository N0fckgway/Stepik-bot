"""Per-task and run-wide limits for potentially billable AI calls."""

from dataclasses import dataclass
from decimal import Decimal

from stepik_agent.ai.transport import AITransport, CompletionRequest, CompletionResult
from stepik_agent.core.errors import AIRunBudgetExceededError, AITaskBudgetExceededError


@dataclass(slots=True)
class SharedAIBudget:
    """Accumulate known AI spend across several task-scoped transports."""

    max_cost_usd: Decimal
    max_calls: int = 500
    known_cost_usd: Decimal = Decimal("0")
    calls: int = 0
    has_unknown_cost: bool = False

    def __post_init__(self) -> None:
        if not self.max_cost_usd.is_finite() or self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be finite and nonnegative")
        if self.max_calls <= 0:
            raise ValueError("max_calls must be positive")

    def before_call(self) -> None:
        if self.calls >= self.max_calls:
            raise AIRunBudgetExceededError("maximum AI calls per run reached")
        if self.known_cost_usd >= self.max_cost_usd:
            raise AIRunBudgetExceededError("known AI cost reached the run budget")
        self.calls += 1

    def record(self, cost_usd: Decimal | None) -> None:
        if cost_usd is None:
            self.has_unknown_cost = True
            return
        self.known_cost_usd += cost_usd
        if self.known_cost_usd > self.max_cost_usd:
            raise AIRunBudgetExceededError("provider-reported AI cost exceeded the run budget")


class BudgetedAITransport:
    def __init__(
        self,
        delegate: AITransport,
        *,
        max_calls: int,
        max_cost_usd: Decimal,
        shared_budget: SharedAIBudget | None = None,
    ) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if not max_cost_usd.is_finite() or max_cost_usd < 0:
            raise ValueError("max_cost_usd must be finite and nonnegative")
        self._delegate = delegate
        self._max_calls = max_calls
        self._max_cost_usd = max_cost_usd
        self.calls = 0
        self.known_cost_usd = Decimal("0")
        self.has_unknown_cost = False
        self._shared_budget = shared_budget

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        if self.calls >= self._max_calls:
            raise AITaskBudgetExceededError("maximum AI calls per task reached")
        if self.known_cost_usd >= self._max_cost_usd:
            raise AITaskBudgetExceededError("known AI cost reached the task budget")

        # Count before the request because a failed or timed-out call may still be billable.
        if self._shared_budget is not None:
            self._shared_budget.before_call()
        self.calls += 1
        result = await self._delegate.complete(request)
        if result.provider_cost_usd is None:
            self.has_unknown_cost = True
            if self._shared_budget is not None:
                self._shared_budget.record(None)
            return result

        self.known_cost_usd += result.provider_cost_usd
        if self.known_cost_usd > self._max_cost_usd:
            raise AITaskBudgetExceededError("provider-reported AI cost exceeded the task budget")
        if self._shared_budget is not None:
            self._shared_budget.record(result.provider_cost_usd)
        return result

    def reset_task_budget(self) -> None:
        """Start a new task without resetting the shared run-wide budget."""

        self.calls = 0
        self.known_cost_usd = Decimal("0")
        self.has_unknown_cost = False

    async def aclose(self) -> None:
        await self._delegate.aclose()
