import asyncio

from stepik_agent.ai.models import SolutionCandidate, TaskContext
from stepik_agent.application.ai_smoke import run_ai_smoke


class FakeGateway:
    def __init__(self) -> None:
        self.context: TaskContext | None = None

    async def generate_solution(self, task_context: TaskContext) -> SolutionCandidate:
        self.context = task_context
        return SolutionCandidate(
            task_type="number",
            answer="4",
            explanation="Two plus two equals four.",
            confidence=1,
        )


def test_ai_smoke_is_synthetic_and_stepik_independent() -> None:
    gateway = FakeGateway()

    result = asyncio.run(run_ai_smoke(gateway))

    assert result.candidate.answer == "4"
    assert gateway.context is not None
    assert gateway.context.step_id == 1
    assert "Synthetic" in result.warnings[0]
