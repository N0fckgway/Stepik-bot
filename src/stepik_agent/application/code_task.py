"""Bounded solve, submit, feedback, and repair loop for one code task."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from stepik_agent.ai.models import SolutionCandidate, TableSelection
from stepik_agent.application.dry_run import DryRunResult
from stepik_agent.core.errors import (
    AttemptLimitExceededError,
    CandidateValidationError,
    StepikAgentError,
)
from stepik_agent.stepik.urls import StepikTarget
from stepik_agent.storage.attempts import AttemptRecord
from stepik_agent.tasks.analyzer import html_to_text


class CodeSolver(Protocol):
    async def load_step_id(self, target: StepikTarget) -> int: ...

    async def execute(self, target: StepikTarget) -> DryRunResult: ...

    async def repair(
        self,
        target: StepikTarget,
        *,
        candidate: SolutionCandidate,
        feedback: str,
    ) -> DryRunResult: ...

    async def execute_with_dataset(
        self,
        target: StepikTarget,
        *,
        dataset: Mapping[str, object] | str | None,
        dataset_text: str | None = None,
    ) -> DryRunResult: ...

    async def repair_with_dataset(
        self,
        target: StepikTarget,
        *,
        candidate: SolutionCandidate,
        feedback: str,
        dataset: Mapping[str, object] | str | None,
        dataset_text: str | None = None,
    ) -> DryRunResult: ...


class CodeSubmitter(Protocol):
    async def submit_code(
        self,
        *,
        step_id: int,
        candidate: SolutionCandidate,
        model_id: str | None,
        run_id: str | None = None,
    ) -> AttemptRecord: ...

    def attempts_used(self, step_id: int) -> int: ...

    def get_prepared(self, step_id: int) -> AttemptRecord | None: ...

    def latest_wrong(self, step_id: int) -> AttemptRecord | None: ...

    def wrong_history(self, step_id: int, *, limit: int = 8) -> list[AttemptRecord]: ...

    async def prepare_attempt(
        self,
        *,
        step_id: int,
        model_id: str | None,
        run_id: str | None = None,
    ) -> AttemptRecord: ...

    async def submit_prepared(
        self,
        *,
        local_attempt_id: int,
        candidate: SolutionCandidate,
        reply: dict[str, object],
    ) -> AttemptRecord: ...

    async def download_dataset(self, dataset_url: str) -> str: ...

    async def submit_candidate(
        self,
        *,
        step_id: int,
        candidate: SolutionCandidate,
        reply: dict[str, object],
        model_id: str | None,
        run_id: str | None = None,
    ) -> AttemptRecord: ...


@dataclass(frozen=True, slots=True)
class CodeTaskResult:
    solution: DryRunResult
    attempt: AttemptRecord
    submitted_attempts: int


class CodeTaskRunner:
    def __init__(
        self,
        *,
        solver: CodeSolver,
        submitter: CodeSubmitter,
        max_attempts: int,
        max_total_attempts: int | None = None,
        model_id: str | None,
        candidate_observer: Callable[[DryRunResult], None] | None = None,
        event_observer: Callable[[str, int, str, str], None] | None = None,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        total_limit = max_total_attempts or max_attempts
        if total_limit < max_attempts:
            raise ValueError("max_total_attempts must be at least max_attempts")
        self._solver = solver
        self._submitter = submitter
        self._max_attempts = max_attempts
        self._max_total_attempts = total_limit
        self._model_id = model_id
        self._candidate_observer = candidate_observer
        self._event_observer = event_observer

    def reset_task_budget(self) -> None:
        reset = getattr(self._solver, "reset_task_budget", None)
        if reset is not None:
            reset()

    async def execute(
        self,
        target: StepikTarget,
        *,
        run_id: str | None = None,
        task_type: str | None = None,
    ) -> CodeTaskResult:
        if task_type in {"choice", "matching", "fill-blanks", "table", "dataset"}:
            return await self._execute_dataset_task(target, task_type=task_type, run_id=run_id)
        attempts_used = getattr(self._submitter, "attempts_used", lambda _step_id: 0)
        step_id = await self._solver.load_step_id(target)
        if attempts_used(step_id) >= self._max_total_attempts:
            raise AttemptLimitExceededError(
                f"Step {step_id} already reached total attempt limit {self._max_total_attempts}"
            )
        failure_history = self._submitter.wrong_history(step_id)
        previous = failure_history[-1] if failure_history else None
        if previous is not None and previous.solution_text:
            previous_candidate = _candidate_from_attempt(previous, task_type or "code")
            solution = await self._solver.repair(
                target,
                candidate=previous_candidate,
                feedback=_feedback_history(failure_history),
            )
        else:
            solution = await self._solver.execute(target)
        if solution.step_id != step_id:
            raise ValueError("Loaded step changed while preparing the solution")
        submitted_attempts = 0
        while True:
            solution = await self._avoid_duplicate_candidate(
                target,
                solution=solution,
                failure_history=failure_history,
            )
            if self._candidate_observer is not None:
                self._candidate_observer(solution)
            submit_candidate = getattr(self._submitter, "submit_candidate", None)
            if submit_candidate is None:
                attempt = await self._submitter.submit_code(
                    step_id=solution.step_id,
                    candidate=solution.candidate,
                    model_id=self._model_id,
                    run_id=run_id,
                )
            else:
                if solution.reply is None:
                    raise ValueError("Reviewed candidate has no validated Stepik reply")
                attempt = await submit_candidate(
                    step_id=solution.step_id,
                    candidate=solution.candidate,
                    reply=solution.reply,
                    model_id=self._model_id,
                    run_id=run_id,
                )
            submitted_attempts += 1
            if (
                attempt.verdict == "correct"
                or submitted_attempts >= self._max_attempts
                or attempts_used(step_id) >= self._max_total_attempts
            ):
                return CodeTaskResult(solution, attempt, submitted_attempts)
            failure_history.append(attempt)
            solution = await self._solver.repair(
                target,
                candidate=solution.candidate,
                feedback=_feedback_history(failure_history),
            )

    async def _execute_dataset_task(
        self,
        target: StepikTarget,
        *,
        task_type: str,
        run_id: str | None,
    ) -> CodeTaskResult:
        step_id = await self._resolve_step_id(target)
        failure_history = self._submitter.wrong_history(step_id)
        previous = failure_history[-1] if failure_history else None
        previous_candidate = (
            _candidate_from_attempt(previous, task_type)
            if previous is not None and previous.solution_text
            else None
        )
        submitted_attempts = 0
        while submitted_attempts < self._max_attempts:
            # Loading the public step first would not expose choice/matching data. The Stepik
            # attempt is intentionally created before the paid AI call and its dataset is saved.
            prepared = self._submitter.get_prepared(step_id)
            if prepared is None:
                if self._submitter.attempts_used(step_id) >= self._max_total_attempts:
                    raise AttemptLimitExceededError(
                        f"Step {step_id} already reached total attempt limit "
                        f"{self._max_total_attempts}"
                    )
                prepared = await self._submitter.prepare_attempt(
                    step_id=step_id,
                    model_id=self._model_id,
                    run_id=run_id,
                )
            dataset_text = None
            try:
                if task_type == "dataset":
                    if prepared.dataset_url:
                        dataset_text = await self._submitter.download_dataset(prepared.dataset_url)
                    elif isinstance(prepared.attempt_dataset, str):
                        dataset_text = prepared.attempt_dataset
                    else:
                        raise CandidateValidationError(
                            "Dataset attempt did not provide data or dataset_url"
                        )
                if previous_candidate is None:
                    solution = await self._solver.execute_with_dataset(
                        target,
                        dataset=prepared.attempt_dataset,
                        dataset_text=dataset_text,
                    )
                else:
                    solution = await self._solver.repair_with_dataset(
                        target,
                        candidate=previous_candidate,
                        feedback=_feedback_history(failure_history),
                        dataset=prepared.attempt_dataset,
                        dataset_text=dataset_text,
                    )
                solution = await self._avoid_duplicate_dataset_candidate(
                    target,
                    solution=solution,
                    failure_history=failure_history,
                    dataset=prepared.attempt_dataset,
                    dataset_text=dataset_text,
                )
                if solution.reply is None:
                    raise CandidateValidationError(
                        "Reviewed candidate has no validated Stepik reply"
                    )
                if self._candidate_observer is not None:
                    self._candidate_observer(solution)
                attempt = await self._submitter.submit_prepared(
                    local_attempt_id=prepared.local_attempt_id,
                    candidate=solution.candidate,
                    reply=solution.reply,
                )
            except StepikAgentError as error:
                self._observe_interactive_error(
                    run_id=run_id,
                    step_id=step_id,
                    task_type=task_type,
                    error=error,
                )
                raise
            submitted_attempts += 1
            if attempt.verdict == "correct":
                return CodeTaskResult(solution, attempt, submitted_attempts)
            failure_history.append(attempt)
            previous_candidate = solution.candidate
        return CodeTaskResult(solution, attempt, submitted_attempts)

    async def _resolve_step_id(self, target: StepikTarget) -> int:
        return await self._solver.load_step_id(target)

    async def _avoid_duplicate_candidate(
        self,
        target: StepikTarget,
        *,
        solution: DryRunResult,
        failure_history: list[AttemptRecord],
    ) -> DryRunResult:
        for _ in range(2):
            if not _duplicates_wrong_answer(solution, failure_history):
                return solution
            solution = await self._solver.repair(
                target,
                candidate=solution.candidate,
                feedback=_duplicate_feedback(failure_history),
            )
        if _duplicates_wrong_answer(solution, failure_history):
            raise CandidateValidationError(
                "AI repeated an answer already rejected by Stepik after bounded repairs"
            )
        return solution

    async def _avoid_duplicate_dataset_candidate(
        self,
        target: StepikTarget,
        *,
        solution: DryRunResult,
        failure_history: list[AttemptRecord],
        dataset: Mapping[str, object] | str | None,
        dataset_text: str | None,
    ) -> DryRunResult:
        for _ in range(2):
            if not _duplicates_wrong_answer(
                solution,
                failure_history,
                dataset=dataset,
            ):
                return solution
            solution = await self._solver.repair_with_dataset(
                target,
                candidate=solution.candidate,
                feedback=_duplicate_feedback(failure_history),
                dataset=dataset,
                dataset_text=dataset_text,
            )
        if _duplicates_wrong_answer(solution, failure_history, dataset=dataset):
            raise CandidateValidationError(
                "AI repeated an answer already rejected by Stepik after bounded repairs"
            )
        return solution

    def _observe_interactive_error(
        self,
        *,
        run_id: str | None,
        step_id: int,
        task_type: str,
        error: StepikAgentError,
    ) -> None:
        if run_id is None or self._event_observer is None:
            return
        try:
            self._event_observer(run_id, step_id, task_type, str(error)[:1_000])
        except Exception:
            return


def _candidate_from_attempt(record: AttemptRecord, task_type: str) -> SolutionCandidate:
    if record.solution_text is None:
        raise ValueError("Wrong attempt has no stored solution")
    ordering = _ordering_from_reply(record.submission_reply) if task_type == "matching" else None
    return SolutionCandidate(
        task_type=task_type,
        answer=record.solution_text,
        selected_choices=(
            sorted(selected_choices)
            if task_type == "choice"
            and (
                selected_choices := _selected_choice_texts(
                    record.submission_reply,
                    record.attempt_dataset,
                )
            )
            is not None
            else None
        ),
        ordering=ordering,
        blanks=_blanks_from_reply(record.submission_reply) if task_type == "fill-blanks" else None,
        table_selections=(
            _table_selections_from_reply(record.submission_reply, record.attempt_dataset)
            if task_type == "table"
            else None
        ),
        language=record.solution_language if task_type == "code" else None,
        explanation="Candidate restored from the previous wrong Stepik attempt",
        confidence=0.5,
    )


def _feedback_history(records: list[AttemptRecord]) -> str:
    if not records:
        return "Stepik returned a wrong answer"
    sections = []
    for record in records[-8:]:
        feedback = record.feedback or f"Stepik returned verdict: {record.verdict}"
        sections.append(
            f"Attempt {record.attempt_number} failed. Do not reintroduce this failure:\n"
            f"{feedback[:12_000]}"
        )
    header = (
        "Resolve the complete Stepik failure history below, not only the latest error. "
        "Keep previously corrected public symbols, signatures, allocator names, namespaces, "
        "and task requirements unchanged unless the history proves they are wrong."
    )
    combined = f"{header}\n\n" + "\n\n".join(sections)
    if len(combined) <= 90_000:
        return combined
    return f"{header}\n\n{combined[-88_000:]}"


def _duplicates_wrong_answer(
    solution: DryRunResult,
    records: list[AttemptRecord],
    *,
    dataset: Mapping[str, object] | str | None = None,
) -> bool:
    answer = solution.candidate.answer.strip()
    for record in records:
        if solution.task_type == "choice":
            current_choices = _selected_choice_texts(solution.reply, dataset)
            previous_choices = _selected_choice_texts(
                record.submission_reply,
                record.attempt_dataset,
            )
            if (
                current_choices is not None
                and previous_choices is not None
                and current_choices == previous_choices
            ):
                return True
        elif solution.task_type == "matching":
            current_matching = _matching_semantics(solution.reply, dataset)
            previous_matching = _matching_semantics(
                record.submission_reply,
                record.attempt_dataset,
            )
            if (
                current_matching is not None
                and previous_matching is not None
                and current_matching == previous_matching
            ):
                return True
        elif record.submission_reply is not None and record.submission_reply == solution.reply:
            return True
        if record.solution_text is not None and record.solution_text.strip() == answer:
            return True
    return False


def _selected_choice_texts(
    reply: Mapping[str, object] | None,
    dataset: Mapping[str, object] | str | None,
) -> frozenset[str] | None:
    if not isinstance(reply, Mapping) or not isinstance(dataset, Mapping):
        return None
    raw_choices = reply.get("choices")
    raw_options = dataset.get("options", dataset.get("choices"))
    if not isinstance(raw_choices, list) or not isinstance(raw_options, list):
        return None
    if len(raw_choices) != len(raw_options) or not all(
        isinstance(item, bool) for item in raw_choices
    ):
        return None
    options = [_choice_text(item) for item in raw_options]
    if any(option is None for option in options):
        return None
    return frozenset(
        option
        for selected, option in zip(raw_choices, options, strict=False)
        if selected and option is not None
    )


def _choice_text(value: object) -> str | None:
    if isinstance(value, str):
        text = html_to_text(value)
    elif isinstance(value, Mapping) and isinstance(value.get("text"), str):
        text = html_to_text(value["text"])
    else:
        return None
    return text or None


def _ordering_from_reply(reply: Mapping[str, object] | None) -> list[int] | None:
    if not isinstance(reply, Mapping):
        return None
    ordering = reply.get("ordering")
    if not isinstance(ordering, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) for index in ordering
    ):
        return None
    return ordering


def _blanks_from_reply(reply: Mapping[str, object] | None) -> list[str] | None:
    if not isinstance(reply, Mapping):
        return None
    blanks = reply.get("blanks")
    if not isinstance(blanks, list) or not all(isinstance(item, str) for item in blanks):
        return None
    return blanks


def _table_selections_from_reply(
    reply: Mapping[str, object] | None,
    dataset: Mapping[str, object] | str | None,
) -> list[TableSelection] | None:
    if not isinstance(reply, Mapping) or not isinstance(dataset, Mapping):
        return None
    raw_choices = reply.get("choices")
    rows = dataset.get("rows")
    columns = dataset.get("columns")
    if (
        not isinstance(raw_choices, list)
        or not isinstance(rows, list)
        or not isinstance(columns, list)
    ):
        return None
    if len(raw_choices) != len(rows):
        return None
    result: list[TableSelection] = []
    for row_index, row_reply in enumerate(raw_choices):
        if not isinstance(row_reply, Mapping):
            return None
        raw_cells = row_reply.get("columns")
        if not isinstance(raw_cells, list) or len(raw_cells) != len(columns):
            return None
        selected: list[int] = []
        for column_index, cell in enumerate(raw_cells):
            if not isinstance(cell, Mapping) or not isinstance(cell.get("answer"), bool):
                return None
            if cell["answer"]:
                selected.append(column_index)
        result.append(
            TableSelection(
                row_index=row_index,
                selected_column_indices=selected,
            )
        )
    return result


def _matching_semantics(
    reply: Mapping[str, object] | None,
    dataset: Mapping[str, object] | str | None,
) -> frozenset[tuple[str, str]] | None:
    if not isinstance(reply, Mapping) or not isinstance(dataset, Mapping):
        return None
    ordering = reply.get("ordering")
    raw_pairs = dataset.get("pairs")
    if not isinstance(ordering, list) or not isinstance(raw_pairs, list):
        return None
    pairs: list[tuple[str, str]] = []
    for item in raw_pairs:
        if not isinstance(item, Mapping):
            return None
        first = item.get("first")
        second = item.get("second")
        if not isinstance(first, str) or not isinstance(second, str):
            return None
        pairs.append((html_to_text(first), html_to_text(second)))
    if len(ordering) != len(pairs) or not all(
        isinstance(index, int) and not isinstance(index, bool) for index in ordering
    ):
        return None
    if any(index < 0 or index >= len(pairs) for index in ordering):
        return None
    seconds = [second for _first, second in pairs]
    return frozenset(
        (pairs[index][0], seconds[selected_index]) for index, selected_index in enumerate(ordering)
    )


def _duplicate_feedback(records: list[AttemptRecord]) -> str:
    return (
        f"{_feedback_history(records)}\n\n"
        "The proposed replacement is semantically identical to an answer Stepik already marked "
        "wrong. Re-solve the task and return a materially different answer."
    )
