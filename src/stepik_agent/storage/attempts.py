"""Transactional persistence for crash-safe Stepik submission state."""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import NoReturn

from sqlalchemy import func, select

from stepik_agent.core.errors import AttemptLimitExceededError
from stepik_agent.core.states import AttemptState, TaskStatus
from stepik_agent.storage.database import Database
from stepik_agent.storage.models import Task, TaskAttempt
from stepik_agent.storage.repositories import utc_now


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    local_attempt_id: int
    step_id: int
    run_id: str | None
    attempt_number: int
    state: AttemptState
    stepik_attempt_id: int | None
    stepik_submission_id: int | None
    model_id: str | None
    solution_text: str | None
    solution_language: str | None
    solution_hash: str | None
    verdict: str | None
    feedback: str | None
    started_at: datetime
    finished_at: datetime | None
    submission_reply: dict[str, object] | None = None
    attempt_dataset: dict[str, object] | str | None = None
    dataset_url: str | None = None


class AttemptStore:
    """Commit every external-write transition in its own SQLite transaction."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def create_intent(
        self,
        *,
        step_id: int,
        run_id: str | None,
        model_id: str | None,
        solution_text: str,
        solution_language: str,
        max_attempts: int,
        submission_reply: dict[str, object] | None = None,
    ) -> AttemptRecord:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if not solution_text.strip():
            raise ValueError("solution_text must not be empty")
        if not solution_language.strip():
            raise ValueError("solution_language must not be empty")
        solution_hash = hashlib.sha256(solution_text.encode("utf-8")).hexdigest()
        with self._database.session() as session:
            task = session.get(Task, step_id)
            if task is None:
                raise ValueError(f"Task {step_id} must be synchronized before submission")
            last_number = session.scalar(
                select(func.max(TaskAttempt.attempt_number)).where(TaskAttempt.step_id == step_id)
            )
            next_number = (last_number or 0) + 1
            if next_number > max_attempts:
                raise AttemptLimitExceededError(
                    f"Step {step_id} already reached the configured total attempt limit "
                    f"{max_attempts}"
                )
            attempt = TaskAttempt(
                step_id=step_id,
                run_id=run_id,
                attempt_number=next_number,
                state=AttemptState.PLANNED,
                model_id=model_id,
                solution_text=solution_text,
                solution_language=solution_language,
                submission_reply=submission_reply,
                solution_hash=solution_hash,
                started_at=utc_now(),
            )
            task.status = TaskStatus.SUBMITTING
            session.add(attempt)
            session.flush()
            return _record(attempt)

    def create_attempt_intent(
        self,
        *,
        step_id: int,
        run_id: str | None,
        model_id: str | None,
        max_attempts: int,
    ) -> AttemptRecord:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        with self._database.session() as session:
            task = session.get(Task, step_id)
            if task is None:
                raise ValueError(f"Task {step_id} must be synchronized before submission")
            last_number = session.scalar(
                select(func.max(TaskAttempt.attempt_number)).where(TaskAttempt.step_id == step_id)
            )
            next_number = (last_number or 0) + 1
            if next_number > max_attempts:
                raise AttemptLimitExceededError(
                    f"Step {step_id} already reached the configured attempt limit {max_attempts}"
                )
            attempt = TaskAttempt(
                step_id=step_id,
                run_id=run_id,
                attempt_number=next_number,
                state=AttemptState.PLANNED,
                model_id=model_id,
                started_at=utc_now(),
            )
            task.status = TaskStatus.SUBMITTING
            session.add(attempt)
            session.flush()
            return _record(attempt)

    def attach_solution(
        self,
        local_attempt_id: int,
        *,
        solution_text: str,
        solution_language: str,
        submission_reply: dict[str, object],
    ) -> AttemptRecord:
        if not solution_text.strip() or not solution_language.strip():
            raise ValueError("solution text and language must not be empty")
        with self._database.session() as session:
            attempt = _required_attempt(session.get(TaskAttempt, local_attempt_id))
            _require_state(attempt, {AttemptState.ATTEMPT_CREATED})
            if attempt.solution_hash is not None:
                raise ValueError("Prepared attempt already has a solution")
            attempt.solution_text = solution_text
            attempt.solution_language = solution_language
            attempt.submission_reply = submission_reply
            attempt.solution_hash = hashlib.sha256(solution_text.encode("utf-8")).hexdigest()
            session.flush()
            return _record(attempt)

    def attempts_used(self, step_id: int) -> int:
        with self._database.session() as session:
            return int(
                session.scalar(
                    select(func.count(TaskAttempt.local_attempt_id)).where(
                        TaskAttempt.step_id == step_id
                    )
                )
                or 0
            )

    def get_prepared(self, step_id: int) -> AttemptRecord | None:
        with self._database.session() as session:
            attempt = session.scalar(
                select(TaskAttempt)
                .where(
                    TaskAttempt.step_id == step_id,
                    TaskAttempt.state == AttemptState.ATTEMPT_CREATED,
                    TaskAttempt.solution_hash.is_(None),
                )
                .order_by(TaskAttempt.attempt_number.desc())
                .limit(1)
            )
            return _record(attempt) if attempt is not None else None

    def latest_wrong(self, step_id: int) -> AttemptRecord | None:
        with self._database.session() as session:
            attempt = session.scalar(
                select(TaskAttempt)
                .where(
                    TaskAttempt.step_id == step_id,
                    TaskAttempt.state == AttemptState.WRONG,
                )
                .order_by(TaskAttempt.attempt_number.desc())
                .limit(1)
            )
            return _record(attempt) if attempt is not None else None

    def latest_reconcilable(self, step_id: int) -> AttemptRecord | None:
        with self._database.session() as session:
            attempt = session.scalar(
                select(TaskAttempt)
                .where(
                    TaskAttempt.step_id == step_id,
                    TaskAttempt.state.in_({AttemptState.AMBIGUOUS, AttemptState.WAITING_VERDICT}),
                )
                .order_by(TaskAttempt.attempt_number.desc())
                .limit(1)
            )
            return _record(attempt) if attempt is not None else None

    def wrong_history(self, step_id: int, *, limit: int = 8) -> list[AttemptRecord]:
        if limit <= 0 or limit > 20:
            raise ValueError("wrong attempt history limit must be between 1 and 20")
        with self._database.session() as session:
            attempts = list(
                session.scalars(
                    select(TaskAttempt)
                    .where(
                        TaskAttempt.step_id == step_id,
                        TaskAttempt.state == AttemptState.WRONG,
                    )
                    .order_by(TaskAttempt.attempt_number.desc())
                    .limit(limit)
                ).all()
            )
            return [_record(attempt) for attempt in reversed(attempts)]

    def get(self, local_attempt_id: int) -> AttemptRecord:
        with self._database.session() as session:
            return _record(_required_attempt(session.get(TaskAttempt, local_attempt_id)))

    def mark_creating_attempt(self, local_attempt_id: int) -> AttemptRecord:
        return self._transition(
            local_attempt_id,
            expected={AttemptState.PLANNED},
            target=AttemptState.CREATING_ATTEMPT,
        )

    def record_stepik_attempt(
        self,
        local_attempt_id: int,
        *,
        stepik_attempt_id: int,
        attempt_dataset: dict[str, object] | str | None = None,
        dataset_url: str | None = None,
    ) -> AttemptRecord:
        if stepik_attempt_id <= 0:
            raise ValueError("stepik_attempt_id must be positive")
        with self._database.session() as session:
            attempt = _required_attempt(session.get(TaskAttempt, local_attempt_id))
            _require_state(attempt, {AttemptState.CREATING_ATTEMPT})
            attempt.stepik_attempt_id = stepik_attempt_id
            attempt.attempt_dataset = attempt_dataset
            attempt.dataset_url = dataset_url
            attempt.state = AttemptState.ATTEMPT_CREATED
            session.flush()
            return _record(attempt)

    def mark_creating_submission(self, local_attempt_id: int) -> AttemptRecord:
        return self._transition(
            local_attempt_id,
            expected={AttemptState.ATTEMPT_CREATED},
            target=AttemptState.CREATING_SUBMISSION,
        )

    def record_submission(
        self, local_attempt_id: int, *, stepik_submission_id: int
    ) -> AttemptRecord:
        if stepik_submission_id <= 0:
            raise ValueError("stepik_submission_id must be positive")
        with self._database.session() as session:
            attempt = _required_attempt(session.get(TaskAttempt, local_attempt_id))
            _require_state(attempt, {AttemptState.CREATING_SUBMISSION})
            attempt.stepik_submission_id = stepik_submission_id
            attempt.state = AttemptState.WAITING_VERDICT
            task = _required_task(session.get(Task, attempt.step_id))
            task.status = TaskStatus.WAITING_VERDICT
            session.flush()
            return _record(attempt)

    def restore_ambiguous_submission(self, local_attempt_id: int) -> AttemptRecord:
        """Permit one retry only after Stepik authoritatively reports no submission."""

        with self._database.session() as session:
            attempt = _required_attempt(session.get(TaskAttempt, local_attempt_id))
            _require_state(attempt, {AttemptState.AMBIGUOUS})
            if attempt.stepik_attempt_id is None:
                raise ValueError("Ambiguous attempt has no Stepik attempt ID")
            attempt.state = AttemptState.ATTEMPT_CREATED
            attempt.feedback = "Stepik reconciliation found no remote submission; retry permitted"
            attempt.finished_at = None
            task = _required_task(session.get(Task, attempt.step_id))
            task.status = TaskStatus.SUBMITTING
            session.flush()
            return _record(attempt)

    def record_reconciled_submission(
        self,
        local_attempt_id: int,
        *,
        stepik_submission_id: int,
    ) -> AttemptRecord:
        if stepik_submission_id <= 0:
            raise ValueError("stepik_submission_id must be positive")
        with self._database.session() as session:
            attempt = _required_attempt(session.get(TaskAttempt, local_attempt_id))
            _require_state(attempt, {AttemptState.AMBIGUOUS})
            attempt.stepik_submission_id = stepik_submission_id
            attempt.state = AttemptState.WAITING_VERDICT
            attempt.finished_at = None
            task = _required_task(session.get(Task, attempt.step_id))
            task.status = TaskStatus.WAITING_VERDICT
            session.flush()
            return _record(attempt)

    def finish(
        self,
        local_attempt_id: int,
        *,
        verdict: str,
        feedback: str | None,
    ) -> AttemptRecord:
        if not verdict.strip():
            raise ValueError("verdict must not be empty")
        with self._database.session() as session:
            attempt = _required_attempt(session.get(TaskAttempt, local_attempt_id))
            _require_state(attempt, {AttemptState.WAITING_VERDICT})
            is_correct = verdict == "correct"
            attempt.state = AttemptState.CORRECT if is_correct else AttemptState.WRONG
            attempt.verdict = verdict[:100]
            attempt.feedback = feedback[:100_000] if feedback is not None else None
            attempt.finished_at = utc_now()
            task = _required_task(session.get(Task, attempt.step_id))
            task.status = TaskStatus.CORRECT if is_correct else TaskStatus.RETRYING
            if is_correct:
                task.is_attempted = True
                task.is_passed = True
            session.flush()
            return _record(attempt)

    def mark_ambiguous(self, local_attempt_id: int, *, reason: str) -> AttemptRecord:
        with self._database.session() as session:
            attempt = _required_attempt(session.get(TaskAttempt, local_attempt_id))
            _require_state(
                attempt,
                {AttemptState.CREATING_ATTEMPT, AttemptState.CREATING_SUBMISSION},
            )
            attempt.state = AttemptState.AMBIGUOUS
            attempt.feedback = reason[:100_000]
            attempt.finished_at = utc_now()
            task = _required_task(session.get(Task, attempt.step_id))
            task.status = TaskStatus.BLOCKED
            session.flush()
            return _record(attempt)

    def mark_failed(self, local_attempt_id: int, *, reason: str) -> AttemptRecord:
        with self._database.session() as session:
            attempt = _required_attempt(session.get(TaskAttempt, local_attempt_id))
            if attempt.state in {
                AttemptState.CORRECT,
                AttemptState.WRONG,
                AttemptState.AMBIGUOUS,
                AttemptState.FAILED,
            }:
                _invalid_transition(attempt)
            attempt.state = AttemptState.FAILED
            attempt.feedback = reason[:100_000]
            attempt.finished_at = utc_now()
            task = _required_task(session.get(Task, attempt.step_id))
            task.status = TaskStatus.FAILED
            session.flush()
            return _record(attempt)

    def _transition(
        self,
        local_attempt_id: int,
        *,
        expected: set[AttemptState],
        target: AttemptState,
    ) -> AttemptRecord:
        with self._database.session() as session:
            attempt = _required_attempt(session.get(TaskAttempt, local_attempt_id))
            _require_state(attempt, expected)
            attempt.state = target
            session.flush()
            return _record(attempt)


def _record(attempt: TaskAttempt) -> AttemptRecord:
    if attempt.local_attempt_id is None:
        raise RuntimeError("Persisted attempt is missing its local identifier")
    return AttemptRecord(
        local_attempt_id=attempt.local_attempt_id,
        step_id=attempt.step_id,
        run_id=attempt.run_id,
        attempt_number=attempt.attempt_number,
        state=attempt.state,
        stepik_attempt_id=attempt.stepik_attempt_id,
        stepik_submission_id=attempt.stepik_submission_id,
        model_id=attempt.model_id,
        solution_text=attempt.solution_text,
        solution_language=attempt.solution_language,
        submission_reply=attempt.submission_reply,
        attempt_dataset=attempt.attempt_dataset,
        dataset_url=attempt.dataset_url,
        solution_hash=attempt.solution_hash,
        verdict=attempt.verdict,
        feedback=attempt.feedback,
        started_at=attempt.started_at,
        finished_at=attempt.finished_at,
    )


def _required_attempt(attempt: TaskAttempt | None) -> TaskAttempt:
    if attempt is None:
        raise ValueError("Local attempt does not exist")
    return attempt


def _required_task(task: Task | None) -> Task:
    if task is None:
        raise RuntimeError("Attempt references a missing task")
    return task


def _require_state(attempt: TaskAttempt, expected: set[AttemptState]) -> None:
    if attempt.state not in expected:
        _invalid_transition(attempt)


def _invalid_transition(attempt: TaskAttempt) -> NoReturn:
    raise ValueError(
        f"Invalid transition from {attempt.state.value} for local attempt "
        f"{attempt.local_attempt_id}"
    )
