"""Crash-safe orchestration for one already reviewed code candidate."""

from typing import Protocol

from stepik_agent.ai.models import SolutionCandidate
from stepik_agent.core.errors import AmbiguousStepikWriteError, StepikApiError
from stepik_agent.core.states import AttemptState
from stepik_agent.stepik.write_client import StepikAttempt, StepikSubmission
from stepik_agent.storage.attempts import AttemptRecord


class SubmissionClient(Protocol):
    async def create_attempt(self, step_id: int) -> StepikAttempt: ...

    async def create_code_submission(
        self,
        *,
        attempt_id: int,
        code: str,
        language: str,
    ) -> StepikSubmission: ...

    async def create_submission(
        self,
        *,
        attempt_id: int,
        reply: dict[str, object],
    ) -> StepikSubmission: ...

    async def wait_for_verdict(
        self,
        submission_id: int,
        *,
        timeout_seconds: float,
        initial_delay_seconds: float = 0.5,
        max_delay_seconds: float = 5.0,
    ) -> StepikSubmission: ...

    async def get_submissions_for_attempt(self, attempt_id: int) -> list[StepikSubmission]: ...

    async def download_dataset(self, dataset_url: str, *, max_bytes: int = 2_000_000) -> str: ...


class SubmissionStore(Protocol):
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
    ) -> AttemptRecord: ...

    def get(self, local_attempt_id: int) -> AttemptRecord: ...

    def create_attempt_intent(
        self,
        *,
        step_id: int,
        run_id: str | None,
        model_id: str | None,
        max_attempts: int,
    ) -> AttemptRecord: ...

    def attach_solution(
        self,
        local_attempt_id: int,
        *,
        solution_text: str,
        solution_language: str,
        submission_reply: dict[str, object],
    ) -> AttemptRecord: ...

    def attempts_used(self, step_id: int) -> int: ...

    def get_prepared(self, step_id: int) -> AttemptRecord | None: ...

    def latest_wrong(self, step_id: int) -> AttemptRecord | None: ...

    def latest_reconcilable(self, step_id: int) -> AttemptRecord | None: ...

    def wrong_history(self, step_id: int, *, limit: int = 8) -> list[AttemptRecord]: ...

    def mark_creating_attempt(self, local_attempt_id: int) -> AttemptRecord: ...

    def record_stepik_attempt(
        self,
        local_attempt_id: int,
        *,
        stepik_attempt_id: int,
        attempt_dataset: dict[str, object] | str | None = None,
        dataset_url: str | None = None,
    ) -> AttemptRecord: ...

    def mark_creating_submission(self, local_attempt_id: int) -> AttemptRecord: ...

    def record_submission(
        self, local_attempt_id: int, *, stepik_submission_id: int
    ) -> AttemptRecord: ...

    def restore_ambiguous_submission(self, local_attempt_id: int) -> AttemptRecord: ...

    def record_reconciled_submission(
        self, local_attempt_id: int, *, stepik_submission_id: int
    ) -> AttemptRecord: ...

    def finish(
        self,
        local_attempt_id: int,
        *,
        verdict: str,
        feedback: str | None,
    ) -> AttemptRecord: ...

    def mark_ambiguous(self, local_attempt_id: int, *, reason: str) -> AttemptRecord: ...

    def mark_failed(self, local_attempt_id: int, *, reason: str) -> AttemptRecord: ...


class SingleTaskSubmitService:
    """Persist each transition around non-idempotent Stepik POST operations."""

    def __init__(
        self,
        *,
        client: SubmissionClient,
        store: SubmissionStore,
        verdict_timeout_seconds: float,
        max_attempts: int = 5,
    ) -> None:
        if verdict_timeout_seconds <= 0:
            raise ValueError("verdict_timeout_seconds must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self._client = client
        self._store = store
        self._verdict_timeout_seconds = verdict_timeout_seconds
        self._max_attempts = max_attempts

    def attempts_used(self, step_id: int) -> int:
        return self._store.attempts_used(step_id)

    def get_prepared(self, step_id: int) -> AttemptRecord | None:
        return self._store.get_prepared(step_id)

    def latest_wrong(self, step_id: int) -> AttemptRecord | None:
        return self._store.latest_wrong(step_id)

    def wrong_history(self, step_id: int, *, limit: int = 8) -> list[AttemptRecord]:
        return self._store.wrong_history(step_id, limit=limit)

    async def reconcile_ambiguous(self, step_id: int) -> AttemptRecord | None:
        """Resolve a persisted uncertain submission without blindly replaying its POST."""

        record = self._store.latest_reconcilable(step_id)
        if record is None:
            return None
        if record.state is AttemptState.WAITING_VERDICT:
            return await self._wait_for_verdict(record)
        if record.stepik_attempt_id is None:
            return record

        submissions = await self._client.get_submissions_for_attempt(record.stepik_attempt_id)
        if not submissions:
            restored = self._store.restore_ambiguous_submission(record.local_attempt_id)
            return await self._continue(restored)

        remote = submissions[0]
        record = self._store.record_reconciled_submission(
            record.local_attempt_id,
            stepik_submission_id=remote.submission_id,
        )
        if remote.is_pending:
            return await self._wait_for_verdict(record)
        return self._store.finish(
            record.local_attempt_id,
            verdict=remote.status,
            feedback=remote.hint,
        )

    async def prepare_attempt(
        self,
        *,
        step_id: int,
        model_id: str | None,
        run_id: str | None = None,
    ) -> AttemptRecord:
        record = self._store.create_attempt_intent(
            step_id=step_id,
            run_id=run_id,
            model_id=model_id,
            max_attempts=self._max_attempts,
        )
        return await self._create_attempt(record)

    async def submit_prepared(
        self,
        *,
        local_attempt_id: int,
        candidate: SolutionCandidate,
        reply: dict[str, object],
    ) -> AttemptRecord:
        record = self._store.attach_solution(
            local_attempt_id,
            solution_text=candidate.answer,
            solution_language=candidate.language or candidate.task_type,
            submission_reply=reply,
        )
        return await self._continue(record)

    async def download_dataset(self, dataset_url: str) -> str:
        return await self._client.download_dataset(dataset_url)

    async def submit_code(
        self,
        *,
        step_id: int,
        candidate: SolutionCandidate,
        model_id: str | None,
        run_id: str | None = None,
    ) -> AttemptRecord:
        if candidate.task_type != "code" or candidate.language is None:
            raise ValueError("SingleTaskSubmitService requires a code candidate with language")
        record = self._store.create_intent(
            step_id=step_id,
            run_id=run_id,
            model_id=model_id,
            solution_text=candidate.answer,
            solution_language=candidate.language,
            max_attempts=self._max_attempts,
        )
        return await self._continue(record)

    async def submit_candidate(
        self,
        *,
        step_id: int,
        candidate: SolutionCandidate,
        reply: dict[str, object],
        model_id: str | None,
        run_id: str | None = None,
    ) -> AttemptRecord:
        record = self._store.create_intent(
            step_id=step_id,
            run_id=run_id,
            model_id=model_id,
            solution_text=candidate.answer,
            solution_language=candidate.language or candidate.task_type,
            max_attempts=self._max_attempts,
            submission_reply=reply,
        )
        return await self._continue(record)

    async def resume(self, local_attempt_id: int) -> AttemptRecord:
        record = self._store.get(local_attempt_id)
        if record.state in {AttemptState.CORRECT, AttemptState.WRONG}:
            return record
        if record.state in {AttemptState.CREATING_ATTEMPT, AttemptState.CREATING_SUBMISSION}:
            reason = (
                "Process stopped during a Stepik POST; remote outcome is unknown and automatic "
                "retry is blocked"
            )
            self._store.mark_ambiguous(local_attempt_id, reason=reason)
            raise AmbiguousStepikWriteError(reason)
        if record.state in {AttemptState.AMBIGUOUS, AttemptState.FAILED}:
            raise StepikApiError(
                f"Local attempt {local_attempt_id} cannot be resumed from {record.state.value}"
            )
        return await self._continue(record)

    async def _continue(self, record: AttemptRecord) -> AttemptRecord:
        if record.state is AttemptState.PLANNED:
            record = await self._create_attempt(record)
        if record.state is AttemptState.ATTEMPT_CREATED:
            record = await self._create_submission(record)
        if record.state is AttemptState.WAITING_VERDICT:
            return await self._wait_for_verdict(record)
        if record.state in {AttemptState.CORRECT, AttemptState.WRONG}:
            return record
        raise StepikApiError(f"Unsupported persisted attempt state: {record.state.value}")

    async def _create_attempt(self, record: AttemptRecord) -> AttemptRecord:
        record = self._store.mark_creating_attempt(record.local_attempt_id)
        try:
            remote_attempt = await self._client.create_attempt(record.step_id)
        except AmbiguousStepikWriteError as error:
            self._store.mark_ambiguous(record.local_attempt_id, reason=str(error))
            raise
        except StepikApiError as error:
            self._store.mark_failed(record.local_attempt_id, reason=str(error))
            raise
        return self._store.record_stepik_attempt(
            record.local_attempt_id,
            stepik_attempt_id=remote_attempt.attempt_id,
            attempt_dataset=remote_attempt.dataset,
            dataset_url=remote_attempt.dataset_url,
        )

    async def _create_submission(self, record: AttemptRecord) -> AttemptRecord:
        stepik_attempt_id = record.stepik_attempt_id
        if stepik_attempt_id is None:
            raise RuntimeError("Persisted attempt has no Stepik attempt ID")
        if record.solution_text is None or record.solution_language is None:
            raise RuntimeError("Prepared attempt has no validated solution")
        solution_text = record.solution_text
        solution_language = record.solution_language
        record = self._store.mark_creating_submission(record.local_attempt_id)
        try:
            if record.submission_reply is None:
                submission = await self._client.create_code_submission(
                    attempt_id=stepik_attempt_id,
                    code=solution_text,
                    language=solution_language,
                )
            else:
                submission = await self._client.create_submission(
                    attempt_id=stepik_attempt_id,
                    reply=record.submission_reply,
                )
        except AmbiguousStepikWriteError as error:
            self._store.mark_ambiguous(record.local_attempt_id, reason=str(error))
            raise
        except StepikApiError as error:
            self._store.mark_failed(record.local_attempt_id, reason=str(error))
            raise
        return self._store.record_submission(
            record.local_attempt_id,
            stepik_submission_id=submission.submission_id,
        )

    async def _wait_for_verdict(self, record: AttemptRecord) -> AttemptRecord:
        if record.stepik_submission_id is None:
            raise RuntimeError("Persisted attempt has no Stepik submission ID")
        verdict = await self._client.wait_for_verdict(
            record.stepik_submission_id,
            timeout_seconds=self._verdict_timeout_seconds,
        )
        return self._store.finish(
            record.local_attempt_id,
            verdict=verdict.status,
            feedback=verdict.hint,
        )
