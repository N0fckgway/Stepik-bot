"""A synthetic paid smoke check isolated from Stepik and user coursework."""

from stepik_agent.application.dry_run import DryRunResult, SolutionGenerator
from stepik_agent.tasks.analyzer import TaskAnalyzer
from stepik_agent.tasks.models import StepTaskData


async def run_ai_smoke(gateway: SolutionGenerator) -> DryRunResult:
    analyzer = TaskAnalyzer()
    analysis = analyzer.analyze(
        StepTaskData(
            step_id=1,
            block_name="number",
            statement_html="Calculate 2 + 2. Return only the numeric answer.",
            options={},
        )
    )
    candidate = await gateway.generate_solution(analysis.context)
    analyzer.validate(analysis, candidate)
    return DryRunResult(
        step_id=1,
        task_type="number",
        candidate=candidate,
        warnings=("Synthetic AI smoke check; no Stepik request was made",),
    )
