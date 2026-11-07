import asyncio

import pytest

pytest.importorskip("sqlalchemy")

from stepik_agent.ai.models import SolutionCandidate
from stepik_agent.application.single_submit import SingleTaskSubmitService
from stepik_agent.core.errors import AmbiguousStepikWriteError
from stepik_agent.core.states import AttemptState, TaskStatus
from stepik_agent.stepik.write_client import StepikAttempt, StepikSubmission
from stepik_agent.storage.attempts import AttemptStore
from stepik_agent.storage.database import Database
from stepik_agent.storage.models import Task
from stepik_agent.storage.repositories import CatalogRepository


class FakeSubmissionClient:
    def __init__(
        self,
        *,
        verdict: str = "correct",
        attempt_error: Exception | None = None,
        reconciliation_submissions: list[StepikSubmission] | None = None,
    ) -> None:
        self.verdict = verdict
        self.attempt_error = attempt_error
        self.reconciliation_submissions = reconciliation_submissions or []
        self.create_attempt_calls = 0
        self.create_submission_calls = 0
        self.wait_calls = 0
        self.reconciliation_calls = 0

    async def create_attempt(self, step_id: int) -> StepikAttempt:
        self.create_attempt_calls += 1
        if self.attempt_error is not None:
            raise self.attempt_error
        return StepikAttempt(attempt_id=501, step_id=step_id)

    async def download_dataset(
        self,
        dataset_url: str,
        *,
        max_bytes: int = 2_000_000,
    ) -> str:
        return "dataset"

    async def create_code_submission(
        self,
        *,
        attempt_id: int,
        code: str,
        language: str,
    ) -> StepikSubmission:
        self.create_submission_calls += 1
        assert attempt_id == 501
        assert code == "swap:\n    ret"
        assert language == "asm64"
        return StepikSubmission(601, attempt_id, "evaluation")

    async def wait_for_verdict(
        self,
        submission_id: int,
        *,
        timeout_seconds: float,
        initial_delay_seconds: float = 0.5,
        max_delay_seconds: float = 5.0,
    ) -> StepikSubmission:
        self.wait_calls += 1
        assert submission_id == 601
        return StepikSubmission(601, 501, self.verdict, "platform feedback")

    async def get_submissions_for_attempt(self, attempt_id: int) -> list[StepikSubmission]:
        self.reconciliation_calls += 1
        assert attempt_id == 501
        return self.reconciliation_submissions


def candidate() -> SolutionCandidate:
    return SolutionCandidate(
        task_type="code",
        answer="swap:\n    ret",
        language="asm64",
        explanation="Test candidate",
        confidence=1,
    )


def database_with_task(tmp_path) -> Database:
    database = Database(f"sqlite:///{tmp_path / 'agent.db'}")
    database.create_schema()
    with database.session() as session:
        catalog = CatalogRepository(session)
        catalog.upsert_lesson(lesson_id=10, title="Lesson", language="ru")
        catalog.upsert_task(
            step_id=20,
            lesson_id=10,
            position=1,
            task_type="code",
            progress_id="1-20",
        )
    return database


def test_single_submit_persists_ids_and_terminal_verdict(tmp_path) -> None:
    database = database_with_task(tmp_path)
    client = FakeSubmissionClient()
    service = SingleTaskSubmitService(
        client=client,
        store=AttemptStore(database),
        verdict_timeout_seconds=10,
    )

    result = asyncio.run(
        service.submit_code(step_id=20, candidate=candidate(), model_id="solver-model")
    )

    assert result.state is AttemptState.CORRECT
    assert result.stepik_attempt_id == 501
    assert result.stepik_submission_id == 601
    assert result.verdict == "correct"
    assert result.feedback == "platform feedback"
    assert client.create_attempt_calls == 1
    assert client.create_submission_calls == 1
    with database.session() as session:
        stored_task = session.get(Task, 20)
        assert stored_task is not None
        assert stored_task.status is TaskStatus.CORRECT
        assert stored_task.is_passed is True
    database.dispose()


def test_ambiguous_post_blocks_task_without_retry(tmp_path) -> None:
    database = database_with_task(tmp_path)
    client = FakeSubmissionClient(attempt_error=AmbiguousStepikWriteError("unknown remote outcome"))
    store = AttemptStore(database)
    service = SingleTaskSubmitService(
        client=client,
        store=store,
        verdict_timeout_seconds=10,
    )

    with pytest.raises(AmbiguousStepikWriteError):
        asyncio.run(service.submit_code(step_id=20, candidate=candidate(), model_id=None))

    record = store.get(1)
    assert record.state is AttemptState.AMBIGUOUS
    assert client.create_attempt_calls == 1
    assert client.create_submission_calls == 0
    database.dispose()


def test_resume_waiting_verdict_does_not_repeat_post(tmp_path) -> None:
    database = database_with_task(tmp_path)
    store = AttemptStore(database)
    record = store.create_intent(
        step_id=20,
        run_id=None,
        model_id="solver-model",
        solution_text="swap:\n    ret",
        solution_language="asm64",
        max_attempts=5,
    )
    store.mark_creating_attempt(record.local_attempt_id)
    store.record_stepik_attempt(record.local_attempt_id, stepik_attempt_id=501)
    store.mark_creating_submission(record.local_attempt_id)
    store.record_submission(record.local_attempt_id, stepik_submission_id=601)
    client = FakeSubmissionClient(verdict="wrong")
    service = SingleTaskSubmitService(
        client=client,
        store=store,
        verdict_timeout_seconds=10,
    )

    result = asyncio.run(service.resume(record.local_attempt_id))

    assert result.state is AttemptState.WRONG
    assert client.create_attempt_calls == 0
    assert client.create_submission_calls == 0
    assert client.wait_calls == 1
    database.dispose()


def test_prepared_attempt_persists_dataset_for_crash_safe_ai_resume(tmp_path) -> None:
    database = database_with_task(tmp_path)
    store = AttemptStore(database)
    record = store.create_attempt_intent(
        step_id=20,
        run_id=None,
        model_id="solver-model",
        max_attempts=3,
    )
    store.mark_creating_attempt(record.local_attempt_id)
    store.record_stepik_attempt(
        record.local_attempt_id,
        stepik_attempt_id=501,
        attempt_dataset={"options": ["A", "B"]},
        dataset_url="/dataset.txt",
    )

    prepared = store.get_prepared(20)

    assert prepared is not None
    assert prepared.attempt_dataset == {"options": ["A", "B"]}
    assert prepared.dataset_url == "/dataset.txt"
    assert prepared.solution_hash is None
    database.dispose()


def test_ambiguous_submission_is_retried_only_after_empty_remote_read(tmp_path) -> None:
    database = database_with_task(tmp_path)
    store = AttemptStore(database)
    record = store.create_intent(
        step_id=20,
        run_id=None,
        model_id="solver-model",
        solution_text="swap:\n    ret",
        solution_language="asm64",
        max_attempts=5,
    )
    store.mark_creating_attempt(record.local_attempt_id)
    store.record_stepik_attempt(record.local_attempt_id, stepik_attempt_id=501)
    store.mark_creating_submission(record.local_attempt_id)
    store.mark_ambiguous(record.local_attempt_id, reason="connection lost")
    client = FakeSubmissionClient(reconciliation_submissions=[])
    service = SingleTaskSubmitService(
        client=client,
        store=store,
        verdict_timeout_seconds=10,
    )

    result = asyncio.run(service.reconcile_ambiguous(20))

    assert result is not None
    assert result.state is AttemptState.CORRECT
    assert result.stepik_submission_id == 601
    assert client.reconciliation_calls == 1
    assert client.create_submission_calls == 1
    database.dispose()


def test_ambiguous_submission_uses_existing_remote_verdict_without_post(tmp_path) -> None:
    database = database_with_task(tmp_path)
    store = AttemptStore(database)
    record = store.create_intent(
        step_id=20,
        run_id=None,
        model_id="solver-model",
        solution_text="swap:\n    ret",
        solution_language="asm64",
        max_attempts=5,
    )
    store.mark_creating_attempt(record.local_attempt_id)
    store.record_stepik_attempt(record.local_attempt_id, stepik_attempt_id=501)
    store.mark_creating_submission(record.local_attempt_id)
    store.mark_ambiguous(record.local_attempt_id, reason="connection lost")
    client = FakeSubmissionClient(
        reconciliation_submissions=[StepikSubmission(777, 501, "correct", "accepted")]
    )
    service = SingleTaskSubmitService(
        client=client,
        store=store,
        verdict_timeout_seconds=10,
    )

    result = asyncio.run(service.reconcile_ambiguous(20))

    assert result is not None
    assert result.state is AttemptState.CORRECT
    assert result.stepik_submission_id == 777
    assert client.reconciliation_calls == 1
    assert client.create_submission_calls == 0
    assert client.wait_calls == 0
    database.dispose()
