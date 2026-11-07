"""Autonomous ordered traversal for supported tasks in one synchronized course."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from stepik_agent.application.code_task import CodeTaskResult
from stepik_agent.core.errors import (
    AIRunBudgetExceededError,
    AITaskBudgetExceededError,
    AmbiguousStepikWriteError,
    StepikAgentError,
    StepikAuthenticationError,
    UnsupportedTaskError,
)
from stepik_agent.core.states import RunState, TaskStatus
from stepik_agent.stepik.urls import StepikTarget, StepikTargetType
from stepik_agent.storage.attempts import AttemptRecord
from stepik_agent.storage.course_runs import CourseTaskRecord

_INFORMATION_TYPES = frozenset({"text", "video", "animation"})
_SUPPORTED_TYPES = frozenset(
    {"code", "number", "string", "math", "choice", "matching", "fill-blanks", "table", "dataset"}
)


class CourseQueue(Protocol):
    def list_tasks(self, course_id: int) -> list[CourseTaskRecord]: ...

    def start(self, *, course_id: int, total_tasks: int) -> str: ...

    def finish(
        self,
        run_id: str,
        *,
        state: RunState,
        completed_tasks: int,
        failed_tasks: int,
        estimated_cost: Decimal,
    ) -> None: ...

    def mark_task(self, step_id: int, status: TaskStatus) -> None: ...


class CodeTaskExecutor(Protocol):
    async def execute(
        self,
        target: StepikTarget,
        *,
        run_id: str | None = None,
        task_type: str | None = None,
    ) -> CodeTaskResult: ...


class ViewRecorder(Protocol):
    async def record_view(self, *, step_id: int, assignment_id: int | None) -> None: ...


class BlockedTaskReconciler(Protocol):
    async def reconcile_ambiguous(self, step_id: int) -> AttemptRecord | None: ...


@dataclass(frozen=True, slots=True)
class CourseRunResult:
    run_id: str
    course_id: int
    total_tasks: int
    already_passed: int
    completed: int
    informational: int
    unsupported: int
    failed: int
    blocked: int
    state: RunState
    completion_confirmed: bool


class CourseRunner:
    def __init__(
        self,
        *,
        queue: CourseQueue,
        code_executor: CodeTaskExecutor,
        cost_provider: Callable[[], Decimal] | None = None,
        completion_checker: Callable[[int], Awaitable[bool]] | None = None,
        view_recorder: ViewRecorder | None = None,
        blocked_reconciler: BlockedTaskReconciler | None = None,
    ) -> None:
        self._queue = queue
        self._code_executor = code_executor
        self._cost_provider = cost_provider or (lambda: Decimal("0"))
        self._completion_checker = completion_checker
        self._view_recorder = view_recorder
        self._blocked_reconciler = blocked_reconciler

    async def execute(self, course_id: int) -> CourseRunResult:
        tasks = self._queue.list_tasks(course_id)
        run_id = self._queue.start(course_id=course_id, total_tasks=len(tasks))
        already_passed = completed = informational = unsupported = failed = blocked = 0
        state = RunState.COMPLETED
        completion_confirmed = False

        try:
            for task in tasks:
                if task.is_passed or (
                    task.status is TaskStatus.CORRECT and task.task_type not in _INFORMATION_TYPES
                ):
                    already_passed += 1
                    continue
                if task.status in {TaskStatus.BLOCKED, TaskStatus.WAITING_VERDICT}:
                    # Reconciliation uses a read scoped to the known Stepik attempt. A POST is
                    # repeated only after Stepik explicitly reports that no submission exists.
                    reconciled = None
                    if self._blocked_reconciler is not None:
                        try:
                            reconciled = await self._blocked_reconciler.reconcile_ambiguous(
                                task.step_id
                            )
                        except StepikAuthenticationError:
                            blocked += 1
                            state = RunState.AUTH_REQUIRED
                            break
                        except StepikAgentError:
                            reconciled = None
                    if reconciled is None or reconciled.verdict is None:
                        blocked += 1
                        state = RunState.PARTIAL
                        continue
                    if reconciled.verdict == "correct":
                        completed += 1
                        continue
                    # A reconciled wrong verdict is now safe repair context. Fall through to
                    # the normal solver loop instead of leaving the task permanently blocked.
                if task.task_type in _INFORMATION_TYPES:
                    informational += 1
                    if self._view_recorder is None:
                        failed += 1
                        state = RunState.PARTIAL
                        continue
                    try:
                        await self._view_recorder.record_view(
                            step_id=task.step_id,
                            assignment_id=task.assignment_id,
                        )
                    except StepikAuthenticationError:
                        blocked += 1
                        state = RunState.AUTH_REQUIRED
                        break
                    except AmbiguousStepikWriteError:
                        blocked += 1
                        self._queue.mark_task(task.step_id, TaskStatus.BLOCKED)
                        state = RunState.PARTIAL
                        break
                    except StepikAgentError:
                        failed += 1
                        self._queue.mark_task(task.step_id, TaskStatus.FAILED)
                        state = RunState.PARTIAL
                    continue
                if task.task_type not in _SUPPORTED_TYPES:
                    unsupported += 1
                    self._queue.mark_task(task.step_id, TaskStatus.UNSUPPORTED)
                    continue

                target = _target(task)
                try:
                    reset = getattr(self._code_executor, "reset_task_budget", None)
                    if reset is not None:
                        reset()
                    result = await self._code_executor.execute(
                        target,
                        run_id=run_id,
                        task_type=task.task_type,
                    )
                except StepikAuthenticationError:
                    blocked += 1
                    state = RunState.AUTH_REQUIRED
                    break
                except AmbiguousStepikWriteError:
                    blocked += 1
                    state = RunState.PARTIAL
                    break
                except AIRunBudgetExceededError:
                    blocked += 1
                    state = RunState.BUDGET_EXCEEDED
                    break
                except AITaskBudgetExceededError:
                    failed += 1
                    self._queue.mark_task(task.step_id, TaskStatus.FAILED)
                    state = RunState.PARTIAL
                    continue
                except UnsupportedTaskError:
                    unsupported += 1
                    self._queue.mark_task(task.step_id, TaskStatus.UNSUPPORTED)
                    state = RunState.PARTIAL
                    continue
                except StepikAgentError:
                    failed += 1
                    self._queue.mark_task(task.step_id, TaskStatus.FAILED)
                    state = RunState.PARTIAL
                    continue

                if result.attempt.verdict == "correct":
                    completed += 1
                else:
                    failed += 1
                    self._queue.mark_task(task.step_id, TaskStatus.FAILED)
        except asyncio.CancelledError:
            self._queue.finish(
                run_id,
                state=RunState.STOPPED,
                completed_tasks=completed,
                failed_tasks=failed + blocked,
                estimated_cost=self._cost_provider(),
            )
            raise
        except Exception:
            self._queue.finish(
                run_id,
                state=RunState.ERROR,
                completed_tasks=completed,
                failed_tasks=failed + blocked,
                estimated_cost=self._cost_provider(),
            )
            raise

        if state is RunState.COMPLETED and (unsupported or failed or blocked):
            state = RunState.PARTIAL
        if self._completion_checker is None:
            completion_confirmed = state is RunState.COMPLETED
        else:
            try:
                # Refresh platform progress after every run, including partial ones. Besides
                # providing the only valid completion proof, this can reconcile a write whose
                # response was lost after Stepik had already accepted it.
                completion_confirmed = await self._completion_checker(course_id)
            except StepikAgentError:
                completion_confirmed = False
        if completion_confirmed:
            state = RunState.COMPLETED
        elif state is RunState.COMPLETED:
            state = RunState.PARTIAL
        self._queue.finish(
            run_id,
            state=state,
            completed_tasks=completed,
            failed_tasks=failed + blocked,
            estimated_cost=self._cost_provider(),
        )
        return CourseRunResult(
            run_id=run_id,
            course_id=course_id,
            total_tasks=len(tasks),
            already_passed=already_passed,
            completed=completed,
            informational=informational,
            unsupported=unsupported,
            failed=failed,
            blocked=blocked,
            state=state,
            completion_confirmed=completion_confirmed,
        )


def _target(task: CourseTaskRecord) -> StepikTarget:
    source_url = (
        f"https://stepik.org/lesson/{task.lesson_id}/step/{task.position}?unit={task.unit_id}"
    )
    return StepikTarget(
        source_url=source_url,
        target_type=StepikTargetType.STEP,
        unit_id=task.unit_id,
        lesson_id=task.lesson_id,
        step_position=task.position,
    )
