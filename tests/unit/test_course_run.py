import asyncio
from datetime import UTC, datetime

from stepik_agent.ai.models import SolutionCandidate
from stepik_agent.application.code_task import CodeTaskResult
from stepik_agent.application.course_run import CourseRunner
from stepik_agent.application.dry_run import DryRunResult
from stepik_agent.core.errors import (
    AIRunBudgetExceededError,
    AITaskBudgetExceededError,
    AmbiguousStepikWriteError,
    StepikAuthenticationError,
)
from stepik_agent.core.states import AttemptState, RunState, TaskStatus
from stepik_agent.stepik.urls import StepikTarget
from stepik_agent.storage.attempts import AttemptRecord
from stepik_agent.storage.course_runs import CourseTaskRecord


def task(
    step_id: int,
    task_type: str,
    *,
    passed: bool = False,
    status: TaskStatus | None = None,
) -> CourseTaskRecord:
    return CourseTaskRecord(
        step_id=step_id,
        lesson_id=100 + step_id,
        position=1,
        unit_id=200 + step_id,
        task_type=task_type,
        status=status or (TaskStatus.CORRECT if passed else TaskStatus.DISCOVERED),
        is_passed=passed,
    )


class FakeQueue:
    def __init__(self, tasks: list[CourseTaskRecord]) -> None:
        self.tasks = tasks
        self.marked: list[tuple[int, TaskStatus]] = []
        self.finished: tuple[RunState, int, int] | None = None

    def list_tasks(self, course_id: int) -> list[CourseTaskRecord]:
        return self.tasks

    def start(self, *, course_id: int, total_tasks: int) -> str:
        return "run-id"

    def finish(
        self,
        run_id: str,
        *,
        state: RunState,
        completed_tasks: int,
        failed_tasks: int,
        estimated_cost,
    ) -> None:
        self.finished = (state, completed_tasks, failed_tasks, estimated_cost)

    def mark_task(self, step_id: int, status: TaskStatus) -> None:
        self.marked.append((step_id, status))


class FakeCodeExecutor:
    def __init__(self) -> None:
        self.targets: list[StepikTarget] = []

    async def execute(
        self,
        target: StepikTarget,
        *,
        run_id: str | None = None,
        task_type: str | None = None,
    ) -> CodeTaskResult:
        del task_type
        self.targets.append(target)
        candidate = SolutionCandidate(
            task_type="code",
            answer="ret",
            language="asm64",
            explanation="ok",
            confidence=1,
        )
        solution = DryRunResult(1, "code", candidate, (), reviewer_accepted=True)
        now = datetime.now(UTC)
        attempt = AttemptRecord(
            local_attempt_id=1,
            step_id=1,
            run_id=run_id,
            attempt_number=1,
            state=AttemptState.CORRECT,
            stepik_attempt_id=10,
            stepik_submission_id=20,
            model_id="model",
            solution_text="ret",
            solution_language="asm64",
            solution_hash="a" * 64,
            verdict="correct",
            feedback=None,
            started_at=now,
            finished_at=now,
        )
        return CodeTaskResult(solution, attempt, 1)


class FakeViewRecorder:
    def __init__(self, error: Exception | None = None) -> None:
        self.views: list[tuple[int, int | None]] = []
        self.error = error

    async def record_view(self, *, step_id: int, assignment_id: int | None) -> None:
        self.views.append((step_id, assignment_id))
        if self.error is not None:
            raise self.error


class FakeBlockedReconciler:
    def __init__(self, verdict: str | None) -> None:
        self.verdict = verdict
        self.calls: list[int] = []

    async def reconcile_ambiguous(self, step_id: int) -> AttemptRecord | None:
        self.calls.append(step_id)
        if self.verdict is None:
            return None
        now = datetime.now(UTC)
        return AttemptRecord(
            local_attempt_id=1,
            step_id=step_id,
            run_id=None,
            attempt_number=1,
            state=(AttemptState.CORRECT if self.verdict == "correct" else AttemptState.WRONG),
            stepik_attempt_id=10,
            stepik_submission_id=20,
            model_id="model",
            solution_text="answer",
            solution_language="choice",
            solution_hash="a" * 64,
            verdict=self.verdict,
            feedback=None,
            started_at=now,
            finished_at=now,
        )


class BudgetExecutor(FakeCodeExecutor):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error
        self.calls = 0

    async def execute(
        self,
        target: StepikTarget,
        *,
        run_id: str | None = None,
        task_type: str | None = None,
    ) -> CodeTaskResult:
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return await super().execute(target, run_id=run_id, task_type=task_type)


def test_course_runner_orders_supported_work_and_reports_unsupported() -> None:
    queue = FakeQueue(
        [
            task(1, "text"),
            task(2, "code"),
            task(3, "admin"),
            task(4, "code", passed=True),
        ]
    )
    executor = FakeCodeExecutor()
    view_recorder = FakeViewRecorder()

    result = asyncio.run(
        CourseRunner(
            queue=queue,
            code_executor=executor,
            view_recorder=view_recorder,
        ).execute(1780)
    )

    assert result.state is RunState.PARTIAL
    assert result.completed == 1
    assert result.informational == 1
    assert result.unsupported == 1
    assert result.already_passed == 1
    assert queue.marked == [(3, TaskStatus.UNSUPPORTED)]
    assert len(executor.targets) == 1
    assert view_recorder.views == [(1, None)]
    assert executor.targets[0].source_url.endswith("?unit=202")
    assert queue.finished == (RunState.PARTIAL, 1, 0, 0)


def test_task_ai_budget_marks_one_task_failed_and_continues_course() -> None:
    queue = FakeQueue([task(1, "code"), task(2, "code")])
    executor = BudgetExecutor(AITaskBudgetExceededError("task calls exhausted"))

    result = asyncio.run(CourseRunner(queue=queue, code_executor=executor).execute(1780))

    assert executor.calls == 2
    assert result.state is RunState.PARTIAL
    assert result.completed == 1
    assert result.failed == 1
    assert queue.marked == [(1, TaskStatus.FAILED)]


def test_run_ai_budget_stops_before_later_tasks() -> None:
    queue = FakeQueue([task(1, "code"), task(2, "code")])
    executor = BudgetExecutor(AIRunBudgetExceededError("run calls exhausted"))

    result = asyncio.run(CourseRunner(queue=queue, code_executor=executor).execute(1780))

    assert executor.calls == 1
    assert result.state is RunState.BUDGET_EXCEEDED
    assert result.blocked == 1


def test_rejected_stepik_token_stops_course_before_later_tasks() -> None:
    queue = FakeQueue([task(1, "choice"), task(2, "choice")])
    executor = BudgetExecutor(StepikAuthenticationError("token rejected"))

    result = asyncio.run(CourseRunner(queue=queue, code_executor=executor).execute(762))

    assert executor.calls == 1
    assert result.state is RunState.AUTH_REQUIRED
    assert result.blocked == 1
    assert queue.marked == []


def test_resume_skips_ambiguous_task_and_continues_with_later_work() -> None:
    queue = FakeQueue(
        [
            task(1, "choice", status=TaskStatus.BLOCKED),
            task(2, "code"),
        ]
    )
    executor = FakeCodeExecutor()

    result = asyncio.run(CourseRunner(queue=queue, code_executor=executor).execute(762))

    assert result.state is RunState.PARTIAL
    assert result.blocked == 1
    assert result.completed == 1
    assert len(executor.targets) == 1
    assert executor.targets[0].lesson_id == 102


def test_resume_reconciles_correct_ambiguous_submission_without_solving_again() -> None:
    queue = FakeQueue([task(1, "choice", status=TaskStatus.BLOCKED)])
    executor = FakeCodeExecutor()
    reconciler = FakeBlockedReconciler("correct")

    result = asyncio.run(
        CourseRunner(
            queue=queue,
            code_executor=executor,
            blocked_reconciler=reconciler,
        ).execute(762)
    )

    assert result.completed == 1
    assert result.blocked == 0
    assert reconciler.calls == [1]
    assert executor.targets == []


def test_local_correct_does_not_replace_platform_confirmation_for_information() -> None:
    queue = FakeQueue([task(1, "text", status=TaskStatus.CORRECT)])
    recorder = FakeViewRecorder()

    result = asyncio.run(
        CourseRunner(
            queue=queue,
            code_executor=FakeCodeExecutor(),
            view_recorder=recorder,
        ).execute(762)
    )

    assert result.informational == 1
    assert result.already_passed == 0
    assert recorder.views == [(1, None)]
    assert queue.marked == []


def test_ambiguous_view_is_persisted_as_blocked_and_stops_current_run() -> None:
    queue = FakeQueue([task(1, "text"), task(2, "code")])
    recorder = FakeViewRecorder(AmbiguousStepikWriteError("unknown outcome"))
    executor = FakeCodeExecutor()

    result = asyncio.run(
        CourseRunner(
            queue=queue,
            code_executor=executor,
            view_recorder=recorder,
        ).execute(762)
    )

    assert result.state is RunState.PARTIAL
    assert result.blocked == 1
    assert queue.marked == [(1, TaskStatus.BLOCKED)]
    assert executor.targets == []


def test_partial_run_still_refreshes_authoritative_platform_progress() -> None:
    queue = FakeQueue([task(1, "table")])
    checks: list[int] = []

    async def confirm_completion(course_id: int) -> bool:
        checks.append(course_id)
        return False

    result = asyncio.run(
        CourseRunner(
            queue=queue,
            code_executor=FakeCodeExecutor(),
            completion_checker=confirm_completion,
        ).execute(762)
    )

    assert result.state is RunState.PARTIAL
    assert result.completion_confirmed is False
    assert checks == [762]
