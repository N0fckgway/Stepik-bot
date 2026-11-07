import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from stepik_agent.ai.models import ReviewResult, SolutionCandidate, TaskContext
from stepik_agent.core.errors import CandidateValidationError
from stepik_agent.stepik.urls import StepikTarget
from stepik_agent.tasks.analyzer import TaskAnalyzer
from stepik_agent.tasks.models import StepTaskData, TaskAnalysis
from stepik_agent.tasks.replies import ReplyAdapterRegistry


class TaskLoader(Protocol):
    async def load(self, target: StepikTarget) -> StepTaskData: ...


class SolutionGenerator(Protocol):
    async def generate_solution(self, task_context: TaskContext) -> SolutionCandidate: ...


class SolutionGateway(SolutionGenerator, Protocol):
    async def review_solution(
        self,
        task_context: TaskContext,
        candidate: SolutionCandidate,
    ) -> ReviewResult: ...

    async def repair_solution(
        self,
        task_context: TaskContext,
        candidate: SolutionCandidate,
        feedback: str,
    ) -> SolutionCandidate: ...


@dataclass(frozen=True, slots=True)
class DryRunResult:
    step_id: int
    task_type: str
    candidate: SolutionCandidate
    warnings: tuple[str, ...]
    repair_attempts: int = 0
    reviewer_accepted: bool = False
    reply: dict[str, object] | None = None


class DryRunService:
    """Generate, repair, and review one candidate without a submission dependency."""

    def __init__(
        self,
        *,
        loader: TaskLoader,
        analyzer: TaskAnalyzer,
        gateway: SolutionGateway,
        max_repair_attempts: int = 2,
        reply_adapters: ReplyAdapterRegistry | None = None,
    ) -> None:
        if not 0 <= max_repair_attempts <= 3:
            raise ValueError("max_repair_attempts must be between 0 and 3")
        self._loader = loader
        self._analyzer = analyzer
        self._gateway = gateway
        self._max_repair_attempts = max_repair_attempts
        self._reply_adapters = reply_adapters or ReplyAdapterRegistry()

    def reset_task_budget(self) -> None:
        reset = getattr(self._gateway, "reset_task_budget", None)
        if reset is not None:
            reset()

    async def load_step_id(self, target: StepikTarget) -> int:
        return (await self._loader.load(target)).step_id

    async def execute(self, target: StepikTarget) -> DryRunResult:
        task = await self._loader.load(target)
        return await self.execute_task(task)

    async def execute_with_dataset(
        self,
        target: StepikTarget,
        *,
        dataset: Mapping[str, object] | str | None,
        dataset_text: str | None = None,
    ) -> DryRunResult:
        task = _with_attempt_dataset(await self._loader.load(target), dataset, dataset_text)
        return await self.execute_task(task)

    async def execute_task(self, task: StepTaskData) -> DryRunResult:
        analysis: TaskAnalysis = self._analyzer.analyze(task)
        candidate = await self._gateway.generate_solution(analysis.context)
        return await self._validate_and_review(task, analysis, candidate, repair_attempts=0)

    async def repair(
        self,
        target: StepikTarget,
        *,
        candidate: SolutionCandidate,
        feedback: str,
    ) -> DryRunResult:
        if self._max_repair_attempts == 0:
            raise CandidateValidationError("AI repairs are disabled by configuration")
        task = await self._loader.load(target)
        return await self.repair_task(task, candidate=candidate, feedback=feedback)

    async def repair_with_dataset(
        self,
        target: StepikTarget,
        *,
        candidate: SolutionCandidate,
        feedback: str,
        dataset: Mapping[str, object] | str | None,
        dataset_text: str | None = None,
    ) -> DryRunResult:
        task = _with_attempt_dataset(await self._loader.load(target), dataset, dataset_text)
        return await self.repair_task(task, candidate=candidate, feedback=feedback)

    async def repair_task(
        self,
        task: StepTaskData,
        *,
        candidate: SolutionCandidate,
        feedback: str,
    ) -> DryRunResult:
        analysis: TaskAnalysis = self._analyzer.analyze(task)
        feedback_material = (
            "Stepik platform failure history that the replacement must fully resolve:\n"
            f"{feedback[:90_000]}"
        )
        analysis = replace(
            analysis,
            context=analysis.context.model_copy(
                update={
                    "materials": [
                        *analysis.context.materials,
                        feedback_material,
                    ]
                }
            ),
        )
        repaired = await self._gateway.repair_solution(
            analysis.context,
            candidate,
            feedback[:100_000],
        )
        return await self._validate_and_review(
            task,
            analysis,
            repaired,
            repair_attempts=1,
            platform_feedback=feedback,
        )

    async def _validate_and_review(
        self,
        task: StepTaskData,
        analysis: TaskAnalysis,
        candidate: SolutionCandidate,
        *,
        repair_attempts: int,
        platform_feedback: str | None = None,
    ) -> DryRunResult:

        while True:
            try:
                self._analyzer.validate(analysis, candidate)
                if platform_feedback:
                    _validate_platform_feedback(candidate, platform_feedback)
            except CandidateValidationError as error:
                if repair_attempts >= self._max_repair_attempts:
                    raise CandidateValidationError(
                        f"AI candidate failed local validation after bounded repairs: {error}"
                    ) from error
                candidate = await self._gateway.repair_solution(
                    analysis.context,
                    candidate,
                    _local_validation_feedback(error),
                )
                repair_attempts += 1
                continue

            review = await self._gateway.review_solution(analysis.context, candidate)
            if review.is_acceptable:
                reply = self._reply_adapters.build_reply(task, analysis, candidate)
                return DryRunResult(
                    step_id=task.step_id,
                    task_type=task.block_name,
                    candidate=candidate,
                    warnings=analysis.warnings + tuple(review.warnings),
                    repair_attempts=repair_attempts,
                    reviewer_accepted=True,
                    reply=reply,
                )

            if repair_attempts >= self._max_repair_attempts:
                if task.block_name == "choice" and platform_feedback is not None:
                    # A real Stepik verdict is the authoritative evaluator. For a repaired
                    # choice candidate, a repeatedly dissenting AI reviewer must not create a
                    # permanent local deadlock. Duplicate rejected subsets are still stopped by
                    # CodeTaskRunner before any POST, and the normal attempt limits still apply.
                    reply = self._reply_adapters.build_reply(task, analysis, candidate)
                    return DryRunResult(
                        step_id=task.step_id,
                        task_type=task.block_name,
                        candidate=candidate,
                        warnings=(
                            *analysis.warnings,
                            *review.warnings,
                            "Reviewer rejection was advisory after authoritative Stepik feedback",
                        ),
                        repair_attempts=repair_attempts,
                        reviewer_accepted=False,
                        reply=reply,
                    )
                raise CandidateValidationError(
                    "AI reviewer rejected the candidate after bounded repairs"
                )
            candidate = await self._gateway.repair_solution(
                analysis.context,
                candidate,
                _review_feedback(review),
            )
            repair_attempts += 1


def _local_validation_feedback(error: CandidateValidationError) -> str:
    message = str(error)[:2_000]
    if "missing required function:" in message:
        message = (
            "Candidate still does not define platform-required function(s) from the complete "
            f"starter template: {message.removeprefix('Candidate is missing required function: ')}"
        )
    return (
        "Local validation rejected the candidate. Re-read the complete task context and return "
        f"a complete corrected solution. Validation error: {message}"
    )


_COMPILER_FUNCTION = re.compile("In function [\u2018'`](?P<function>[^\u2019'`]+)[\u2019'`]:")
_NOT_DECLARED = re.compile(
    "error:\\s*[\u2018'`](?P<symbol>[A-Za-z_]\\w*)[\u2019'`] was not declared in this scope"
)
_IMPLICIT_FUNCTION = re.compile(
    "implicit declaration of function [\u2018'`](?P<symbol>[A-Za-z_]\\w*)[\u2019'`]"
)
_REDEFINED_STRUCT = re.compile(
    "redefinition of [\u2018'`]struct (?P<symbol>[A-Za-z_]\\w*)[\u2019'`]"
)
_INCOMPLETE_STRUCT = re.compile(
    "(?:incomplete (?:element )?type|undefined type) "
    "[\u2018'`]struct (?P<symbol>[A-Za-z_]\\w*)[\u2019'`]"
)
_STRUCT_VARIABLE = re.compile(
    r"\bstruct\s+(?P<symbol>[A-Za-z_]\w*)\s+(?P<variable>[A-Za-z_]\w*)"
    r"(?:\[[^\]]+\])?\s*;"
)
_UNKNOWN_STORAGE = re.compile(
    "storage size of [\u2018'`](?P<variable>[A-Za-z_]\\w*)[\u2019'`] (?:isn[\u2019']t|is not) known"
)


def _validate_platform_feedback(candidate: SolutionCandidate, feedback: str) -> None:
    """Catch public functions missing from compiler harness feedback without compiling code."""

    required: set[str] = set()
    current_function = ""
    for line in feedback.splitlines():
        function_match = _COMPILER_FUNCTION.search(line)
        if function_match:
            current_function = function_match.group("function")
            continue
        symbol_match = _NOT_DECLARED.search(line)
        if symbol_match and _is_harness_function(current_function):
            required.add(symbol_match.group("symbol"))

    required.update(match.group("symbol") for match in _IMPLICIT_FUNCTION.finditer(feedback))

    missing = sorted(
        symbol
        for symbol in required
        if not re.search(rf"\b{re.escape(symbol)}\s*\([^;{{}}]*\)\s*\{{", candidate.answer)
    )
    if missing:
        raise CandidateValidationError(
            "Candidate still does not define platform-required function(s) reported by the "
            f"compiler harness: {', '.join(missing)}"
        )

    forbidden_structs = {match.group("symbol") for match in _REDEFINED_STRUCT.finditer(feedback)}
    still_redefined = sorted(
        symbol
        for symbol in forbidden_structs
        if re.search(rf"\bstruct\s+{re.escape(symbol)}\s*\{{", candidate.answer)
    )
    if still_redefined:
        raise CandidateValidationError(
            f"Candidate still redefines platform-provided struct(s): {', '.join(still_redefined)}"
        )

    required_structs = {match.group("symbol") for match in _INCOMPLETE_STRUCT.finditer(feedback)}
    unknown_variables = {match.group("variable") for match in _UNKNOWN_STORAGE.finditer(feedback)}
    required_structs.update(
        match.group("symbol")
        for match in _STRUCT_VARIABLE.finditer(feedback)
        if match.group("variable") in unknown_variables
    )
    missing_structs = sorted(
        symbol
        for symbol in required_structs
        if not re.search(rf"\bstruct\s+{re.escape(symbol)}\s*\{{", candidate.answer)
    )
    if missing_structs:
        raise CandidateValidationError(
            "Candidate still does not define compiler-required struct(s): "
            f"{', '.join(missing_structs)}"
        )


def _is_harness_function(signature: str) -> bool:
    name = signature.split("(", 1)[0].split()[-1] if signature else ""
    return name.startswith("call_") or name.startswith("___check_")


def _review_feedback(review: ReviewResult) -> str:
    issues = "; ".join(review.issues[:20]) or "No structured issue list was provided"
    return (
        "The reviewer rejected the candidate. Re-read the complete task context, resolve every "
        f"issue, and return a complete corrected solution. Review: {review.explanation[:4_000]}. "
        f"Issues: {issues[:4_000]}"
    )


def _with_attempt_dataset(
    task: StepTaskData,
    dataset: Mapping[str, object] | str | None,
    dataset_text: str | None,
) -> StepTaskData:
    options = dict(task.options)
    if isinstance(dataset, Mapping):
        options.update(dataset)
    elif isinstance(dataset, str) and dataset_text is None:
        dataset_text = dataset
    return StepTaskData(
        step_id=task.step_id,
        block_name=task.block_name,
        statement_html=task.statement_html,
        options=options,
        preceding_materials_html=task.preceding_materials_html,
        dataset_text=dataset_text,
    )
