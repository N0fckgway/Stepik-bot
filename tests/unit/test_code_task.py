import asyncio
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from stepik_agent.ai.models import SolutionCandidate
from stepik_agent.application.code_task import CodeTaskRunner, _duplicates_wrong_answer
from stepik_agent.application.dry_run import DryRunResult
from stepik_agent.core.errors import AttemptLimitExceededError
from stepik_agent.core.states import AttemptState
from stepik_agent.stepik.urls import StepikTarget, StepikTargetType
from stepik_agent.storage.attempts import AttemptRecord


def solution(answer: str) -> DryRunResult:
    return DryRunResult(
        step_id=20,
        task_type="code",
        candidate=SolutionCandidate(
            task_type="code",
            answer=answer,
            language="asm64",
            explanation="Candidate",
            confidence=0.9,
        ),
        warnings=(),
        reviewer_accepted=True,
    )


def attempt(number: int, verdict: str) -> AttemptRecord:
    return AttemptRecord(
        local_attempt_id=number,
        step_id=20,
        run_id=None,
        attempt_number=number,
        state=AttemptState.CORRECT if verdict == "correct" else AttemptState.WRONG,
        stepik_attempt_id=100 + number,
        stepik_submission_id=200 + number,
        model_id="model",
        solution_text="code",
        solution_language="asm64",
        solution_hash="a" * 64,
        verdict=verdict,
        feedback="Compiler feedback" if verdict != "correct" else None,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )


class FakeSolver:
    def __init__(self) -> None:
        self.repairs: list[str] = []
        self.execute_calls = 0

    async def load_step_id(self, target: StepikTarget) -> int:
        return 20

    async def execute(self, target: StepikTarget) -> DryRunResult:
        self.execute_calls += 1
        return solution("wrong code")

    async def repair(
        self,
        target: StepikTarget,
        *,
        candidate: SolutionCandidate,
        feedback: str,
    ) -> DryRunResult:
        self.repairs.append(feedback)
        return solution("correct code")


class SequencedRepairSolver(FakeSolver):
    def __init__(self, answers: list[str]) -> None:
        super().__init__()
        self.answers = answers

    async def repair(
        self,
        target: StepikTarget,
        *,
        candidate: SolutionCandidate,
        feedback: str,
    ) -> DryRunResult:
        self.repairs.append(feedback)
        return solution(self.answers.pop(0))


class FakeSubmitter:
    def __init__(
        self,
        attempts: list[AttemptRecord],
        *,
        previous_wrong: AttemptRecord | None = None,
        previous_wrongs: list[AttemptRecord] | None = None,
    ) -> None:
        self.attempts = attempts
        self.used = attempts[0].attempt_number - 1 if attempts else 0
        self.previous_wrong = previous_wrong
        self.previous_wrongs = previous_wrongs or (
            [previous_wrong] if previous_wrong is not None else []
        )

    def attempts_used(self, step_id: int) -> int:
        return self.used

    def latest_wrong(self, step_id: int) -> AttemptRecord | None:
        return self.previous_wrong

    def wrong_history(self, step_id: int, *, limit: int = 8) -> list[AttemptRecord]:
        return self.previous_wrongs[-limit:]

    async def submit_code(
        self,
        *,
        step_id: int,
        candidate: SolutionCandidate,
        model_id: str | None,
        run_id: str | None = None,
    ) -> AttemptRecord:
        self.used += 1
        return self.attempts.pop(0)


def target() -> StepikTarget:
    return StepikTarget(
        "https://stepik.org/lesson/10/step/1",
        StepikTargetType.STEP,
        lesson_id=10,
        step_position=1,
    )


def test_wrong_verdict_is_repaired_and_resubmitted() -> None:
    solver = FakeSolver()
    observed: list[str] = []
    runner = CodeTaskRunner(
        solver=solver,
        submitter=FakeSubmitter([attempt(1, "wrong"), attempt(2, "correct")]),
        max_attempts=3,
        model_id="model",
        candidate_observer=lambda result: observed.append(result.candidate.answer),
    )

    result = asyncio.run(runner.execute(target()))

    assert result.attempt.verdict == "correct"
    assert result.submitted_attempts == 2
    assert len(solver.repairs) == 1
    assert "Compiler feedback" in solver.repairs[0]
    assert observed == ["wrong code", "correct code"]


def test_attempt_limit_stops_without_another_repair() -> None:
    solver = FakeSolver()
    runner = CodeTaskRunner(
        solver=solver,
        submitter=FakeSubmitter([attempt(3, "wrong")]),
        max_attempts=3,
        model_id="model",
    )

    result = asyncio.run(runner.execute(target()))

    assert result.attempt.verdict == "wrong"
    assert solver.repairs == []


def test_existing_attempt_limit_stops_before_paid_solver_call() -> None:
    solver = FakeSolver()
    submitter = FakeSubmitter([attempt(4, "wrong")])
    runner = CodeTaskRunner(
        solver=solver,
        submitter=submitter,
        max_attempts=3,
        model_id="model",
    )

    with pytest.raises(AttemptLimitExceededError, match="attempt limit 3"):
        asyncio.run(runner.execute(target()))

    assert solver.execute_calls == 0


def test_previous_wrong_answer_and_feedback_start_with_repair() -> None:
    solver = FakeSolver()
    previous = attempt(13, "wrong")
    submitter = FakeSubmitter(
        [attempt(14, "correct")],
        previous_wrong=previous,
    )
    runner = CodeTaskRunner(
        solver=solver,
        submitter=submitter,
        max_attempts=3,
        max_total_attempts=25,
        model_id="model",
    )

    result = asyncio.run(runner.execute(target(), task_type="code"))

    assert result.attempt.verdict == "correct"
    assert solver.execute_calls == 0
    assert len(solver.repairs) == 1
    assert "Compiler feedback" in solver.repairs[0]


def test_repair_receives_multiple_failures_without_forgetting_earlier_one() -> None:
    solver = FakeSolver()
    missing_setup = replace(
        attempt(12, "wrong"),
        feedback="cache_setup was not declared",
    )
    bad_namespace = replace(
        attempt(13, "wrong"),
        feedback="object_alignment was not declared; use detail::object_alignment",
    )
    submitter = FakeSubmitter(
        [attempt(14, "correct")],
        previous_wrongs=[missing_setup, bad_namespace],
    )
    runner = CodeTaskRunner(
        solver=solver,
        submitter=submitter,
        max_attempts=3,
        max_total_attempts=25,
        model_id="model",
    )

    asyncio.run(runner.execute(target(), task_type="code"))

    assert "Attempt 12" in solver.repairs[0]
    assert "cache_setup was not declared" in solver.repairs[0]
    assert "Attempt 13" in solver.repairs[0]
    assert "detail::object_alignment" in solver.repairs[0]


def test_answer_already_rejected_by_stepik_is_repaired_before_submission() -> None:
    previous = attempt(1, "wrong")
    solver = SequencedRepairSolver(["code", "materially different code"])
    submitter = FakeSubmitter(
        [attempt(2, "correct")],
        previous_wrong=previous,
    )
    runner = CodeTaskRunner(
        solver=solver,
        submitter=submitter,
        max_attempts=3,
        max_total_attempts=25,
        model_id="model",
    )

    result = asyncio.run(runner.execute(target(), task_type="code"))

    assert result.attempt.verdict == "correct"
    assert len(solver.repairs) == 2
    assert "semantically identical" in solver.repairs[1]
    assert submitter.used == 2


class FakeInteractiveSolver(FakeSolver):
    def __init__(self) -> None:
        super().__init__()
        self.datasets: list[object] = []

    async def execute_with_dataset(
        self,
        target: StepikTarget,
        *,
        dataset,
        dataset_text: str | None = None,
    ) -> DryRunResult:
        self.datasets.append(dataset)
        choice_candidate = SolutionCandidate(
            task_type="choice",
            answer="B",
            explanation="B is correct",
            confidence=1,
        )
        return DryRunResult(
            step_id=20,
            task_type="choice",
            candidate=choice_candidate,
            warnings=(),
            reviewer_accepted=True,
            reply={"choices": [False, True]},
        )


class FakeInteractiveSubmitter(FakeSubmitter):
    def __init__(self) -> None:
        super().__init__([])
        self.prepared = AttemptRecord(
            local_attempt_id=7,
            step_id=20,
            run_id="run",
            attempt_number=1,
            state=AttemptState.ATTEMPT_CREATED,
            stepik_attempt_id=501,
            stepik_submission_id=None,
            model_id="model",
            solution_text=None,
            solution_language=None,
            solution_hash=None,
            verdict=None,
            feedback=None,
            started_at=datetime.now(UTC),
            finished_at=None,
            attempt_dataset={"options": ["A", "B"], "is_multiple_choice": False},
        )
        self.prepared_calls = 0

    def get_prepared(self, step_id: int) -> AttemptRecord | None:
        return self.prepared

    async def prepare_attempt(self, **kwargs) -> AttemptRecord:
        self.prepared_calls += 1
        return self.prepared

    async def submit_prepared(
        self,
        *,
        local_attempt_id: int,
        candidate: SolutionCandidate,
        reply: dict[str, object],
    ) -> AttemptRecord:
        assert local_attempt_id == 7
        assert reply == {"choices": [False, True]}
        return attempt(1, "correct")

    async def download_dataset(self, dataset_url: str) -> str:
        raise AssertionError("choice must not download a file dataset")


def test_choice_uses_persisted_attempt_dataset_before_ai() -> None:
    solver = FakeInteractiveSolver()
    submitter = FakeInteractiveSubmitter()
    runner = CodeTaskRunner(
        solver=solver,
        submitter=submitter,
        max_attempts=3,
        model_id="model",
    )

    result = asyncio.run(runner.execute(target(), run_id="run", task_type="choice"))

    assert result.attempt.verdict == "correct"
    assert solver.datasets == [{"options": ["A", "B"], "is_multiple_choice": False}]
    assert submitter.prepared_calls == 0


class StructuredInteractiveSolver(FakeSolver):
    def __init__(self, task_type: str, reply: dict[str, object]) -> None:
        super().__init__()
        self.task_type = task_type
        self.reply = reply
        self.datasets: list[object] = []

    async def execute_with_dataset(
        self,
        target: StepikTarget,
        *,
        dataset,
        dataset_text: str | None = None,
    ) -> DryRunResult:
        del target, dataset_text
        self.datasets.append(dataset)
        return DryRunResult(
            step_id=20,
            task_type=self.task_type,
            candidate=SolutionCandidate(
                task_type=self.task_type,
                answer="structured answer",
                explanation="ok",
                confidence=1,
            ),
            warnings=(),
            reviewer_accepted=True,
            reply=self.reply,
        )


class StructuredInteractiveSubmitter(FakeInteractiveSubmitter):
    def __init__(self, dataset: dict[str, object], expected_reply: dict[str, object]) -> None:
        super().__init__()
        self.prepared = replace(self.prepared, attempt_dataset=dataset)
        self.expected_reply = expected_reply

    async def submit_prepared(
        self,
        *,
        local_attempt_id: int,
        candidate: SolutionCandidate,
        reply: dict[str, object],
    ) -> AttemptRecord:
        del candidate
        assert local_attempt_id == 7
        assert reply == self.expected_reply
        return attempt(1, "correct")


@pytest.mark.parametrize(
    ("task_type", "dataset", "reply"),
    [
        (
            "fill-blanks",
            {"components": [{"type": "input", "text": "", "options": []}]},
            {"blanks": ["-k2"]},
        ),
        (
            "table",
            {"rows": ["owner"], "columns": ["read"], "is_checkbox": True},
            {
                "choices": [
                    {
                        "name_row": "owner",
                        "columns": [{"name": "read", "answer": True}],
                    }
                ]
            },
        ),
    ],
)
def test_structured_interactive_types_use_attempt_dataset(
    task_type: str,
    dataset: dict[str, object],
    reply: dict[str, object],
) -> None:
    solver = StructuredInteractiveSolver(task_type, reply)
    submitter = StructuredInteractiveSubmitter(dataset, reply)
    runner = CodeTaskRunner(
        solver=solver,
        submitter=submitter,
        max_attempts=3,
        model_id="model",
    )

    result = asyncio.run(runner.execute(target(), run_id="run", task_type=task_type))

    assert result.attempt.verdict == "correct"
    assert solver.datasets == [dataset]


def test_choice_duplicate_comparison_uses_option_text_not_boolean_positions() -> None:
    previous = replace(
        attempt(1, "wrong"),
        solution_text="Нет",
        submission_reply={"choices": [True, False]},
        attempt_dataset={"options": ["Нет", "Да"]},
    )
    current = solution("Да").__class__(
        step_id=20,
        task_type="choice",
        candidate=SolutionCandidate(
            task_type="choice",
            answer="Да",
            selected_choices=["Да"],
            explanation="candidate",
            confidence=1,
        ),
        warnings=(),
        reviewer_accepted=True,
        reply={"choices": [True, False]},
    )

    assert not _duplicates_wrong_answer(
        current,
        [previous],
        dataset={"options": ["Да", "Нет"]},
    )


def test_choice_duplicate_comparison_is_order_independent_for_multiple_choice() -> None:
    previous = replace(
        attempt(1, "wrong"),
        solution_text="A, C",
        submission_reply={"choices": [True, False, True]},
        attempt_dataset={"options": ["A", "B", "C"], "is_multiple_choice": True},
    )
    current = replace(
        solution("C, A"),
        task_type="choice",
        candidate=SolutionCandidate(
            task_type="choice",
            answer="C, A",
            selected_choices=["C", "A"],
            explanation="candidate",
            confidence=1,
        ),
        reply={"choices": [True, False, True]},
    )

    assert _duplicates_wrong_answer(
        current,
        [previous],
        dataset={"options": ["A", "B", "C"], "is_multiple_choice": True},
    )


def test_choice_without_snapshot_does_not_compare_boolean_positions() -> None:
    previous = replace(
        attempt(1, "wrong"),
        solution_text="Нет",
        submission_reply={"choices": [True, False]},
        attempt_dataset=None,
    )
    current = replace(
        solution("Да"),
        task_type="choice",
        candidate=SolutionCandidate(
            task_type="choice",
            answer="Да",
            selected_choices=["Да"],
            explanation="candidate",
            confidence=1,
        ),
        reply={"choices": [True, False]},
    )

    assert not _duplicates_wrong_answer(current, [previous], dataset=None)


class MissingInteractiveDatasetSubmitter(FakeInteractiveSubmitter):
    def __init__(self) -> None:
        super().__init__()
        self.prepared = replace(self.prepared, attempt_dataset=None)


class MissingChoiceReplySolver(FakeInteractiveSolver):
    async def execute_with_dataset(
        self,
        target: StepikTarget,
        *,
        dataset,
        dataset_text: str | None = None,
    ) -> DryRunResult:
        del target, dataset, dataset_text
        return DryRunResult(
            step_id=20,
            task_type="choice",
            candidate=SolutionCandidate(
                task_type="choice",
                answer="B",
                explanation="candidate",
                confidence=1,
            ),
            warnings=(),
            reviewer_accepted=True,
            reply=None,
        )


def test_missing_dataset_records_interactive_error() -> None:
    events: list[tuple[str, int, str, str]] = []
    runner = CodeTaskRunner(
        solver=FakeInteractiveSolver(),
        submitter=MissingInteractiveDatasetSubmitter(),
        max_attempts=3,
        model_id="model",
        event_observer=lambda *event: events.append(event),
    )

    with pytest.raises(Exception, match="Dataset attempt did not provide"):
        asyncio.run(runner.execute(target(), run_id="run", task_type="dataset"))

    assert events == [("run", 20, "dataset", "Dataset attempt did not provide data or dataset_url")]


def test_choice_without_submission_records_interactive_error() -> None:
    events: list[tuple[str, int, str, str]] = []
    runner = CodeTaskRunner(
        solver=MissingChoiceReplySolver(),
        submitter=FakeInteractiveSubmitter(),
        max_attempts=3,
        model_id="model",
        event_observer=lambda *event: events.append(event),
    )

    with pytest.raises(Exception, match="validated Stepik reply"):
        asyncio.run(runner.execute(target(), run_id="run", task_type="choice"))

    assert events == [("run", 20, "choice", "Reviewed candidate has no validated Stepik reply")]
