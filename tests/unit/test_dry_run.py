import asyncio

import pytest

from stepik_agent.ai.models import ReviewResult, SolutionCandidate, TaskContext
from stepik_agent.application.dry_run import DryRunService
from stepik_agent.core.errors import CandidateValidationError, UnsupportedTaskError
from stepik_agent.stepik.urls import StepikTarget, parse_stepik_url
from stepik_agent.tasks.analyzer import TaskAnalyzer
from stepik_agent.tasks.models import StepTaskData


class FakeLoader:
    def __init__(self, task: StepTaskData) -> None:
        self.task = task

    async def load(self, target: StepikTarget) -> StepTaskData:
        return self.task


class FakeGateway:
    def __init__(
        self,
        candidate: SolutionCandidate,
        *,
        repairs: tuple[SolutionCandidate, ...] = (),
        reviews: tuple[ReviewResult, ...] = (),
    ) -> None:
        self.candidate = candidate
        self.repairs = list(repairs)
        self.reviews = list(reviews)
        self.calls = 0
        self.review_calls = 0
        self.repair_feedback: list[str] = []
        self.repair_contexts: list[TaskContext] = []
        self.review_contexts: list[TaskContext] = []

    async def generate_solution(self, task_context: TaskContext) -> SolutionCandidate:
        self.calls += 1
        return self.candidate

    async def review_solution(
        self,
        task_context: TaskContext,
        candidate: SolutionCandidate,
    ) -> ReviewResult:
        self.review_contexts.append(task_context)
        self.review_calls += 1
        if self.reviews:
            return self.reviews.pop(0)
        return review(is_acceptable=True)

    async def repair_solution(
        self,
        task_context: TaskContext,
        candidate: SolutionCandidate,
        feedback: str,
    ) -> SolutionCandidate:
        self.repair_contexts.append(task_context)
        self.repair_feedback.append(feedback)
        if not self.repairs:
            raise AssertionError("Unexpected repair call")
        return self.repairs.pop(0)


def solution(
    task_type: str,
    answer: str,
    *,
    language: str | None = None,
) -> SolutionCandidate:
    return SolutionCandidate(
        task_type=task_type,
        answer=answer,
        language=language,
        explanation="Because",
        confidence=0.9,
    )


def review(*, is_acceptable: bool, issue: str | None = None) -> ReviewResult:
    return ReviewResult(
        is_acceptable=is_acceptable,
        explanation="Candidate review",
        issues=[issue] if issue else [],
        confidence=0.9,
    )


def target() -> StepikTarget:
    return parse_stepik_url("https://stepik.org/lesson/1/step/1")


def test_dry_run_generates_and_validates_candidate() -> None:
    gateway = FakeGateway(solution("number", "4"))
    service = DryRunService(
        loader=FakeLoader(StepTaskData(30, "number", "2 + 2?", {})),
        analyzer=TaskAnalyzer(),
        gateway=gateway,
    )

    result = asyncio.run(service.execute(target()))

    assert result.step_id == 30
    assert result.candidate.answer == "4"
    assert gateway.calls == 1
    assert gateway.review_calls == 1
    assert result.reviewer_accepted is True


def test_unsupported_step_is_rejected_before_ai_call() -> None:
    gateway = FakeGateway(solution("text", "unused"))
    service = DryRunService(
        loader=FakeLoader(StepTaskData(10, "text", "Theory", {})),
        analyzer=TaskAnalyzer(),
        gateway=gateway,
    )

    with pytest.raises(UnsupportedTaskError):
        asyncio.run(service.execute(target()))

    assert gateway.calls == 0


def test_invalid_candidate_never_becomes_dry_run_result() -> None:
    gateway = FakeGateway(solution("number", "not-number"))
    service = DryRunService(
        loader=FakeLoader(StepTaskData(30, "number", "2 + 2?", {})),
        analyzer=TaskAnalyzer(),
        gateway=gateway,
        max_repair_attempts=0,
    )

    with pytest.raises(CandidateValidationError):
        asyncio.run(service.execute(target()))


def test_local_validation_error_is_repaired_with_complete_task_context() -> None:
    invalid = solution(
        "code",
        "swap:\n    movq (%rdi), %rax\n    xchgq %rax, (%rsi)",
        language="asm64",
    )
    corrected = solution(
        "code",
        "swap:\n    movq (%rdi), %rax\n    xchgq %rax, (%rsi)\n    movq %rax, (%rdi)\n    retq",
        language="asm64",
    )
    gateway = FakeGateway(invalid, repairs=(corrected,))
    service = DryRunService(
        loader=FakeLoader(
            StepTaskData(
                170493,
                "code",
                "Swap the values referenced by RDI and RSI.",
                {"code_templates": {"asm64": "swap:\n    // Put your code here"}},
            )
        ),
        analyzer=TaskAnalyzer(),
        gateway=gateway,
    )

    result = asyncio.run(service.execute(target()))

    assert result.candidate == corrected
    assert result.repair_attempts == 1
    assert result.reviewer_accepted is True
    assert gateway.review_calls == 1
    assert "must contain ret or retq" in gateway.repair_feedback[0]


def test_reviewer_rejection_triggers_repair_and_second_review() -> None:
    gateway = FakeGateway(
        solution("number", "5"),
        repairs=(solution("number", "4"),),
        reviews=(
            review(is_acceptable=False, issue="2 + 2 is 4, not 5"),
            review(is_acceptable=True),
        ),
    )
    service = DryRunService(
        loader=FakeLoader(StepTaskData(30, "number", "2 + 2?", {})),
        analyzer=TaskAnalyzer(),
        gateway=gateway,
    )

    result = asyncio.run(service.execute(target()))

    assert result.candidate.answer == "4"
    assert result.repair_attempts == 1
    assert gateway.review_calls == 2
    assert "2 + 2 is 4, not 5" in gateway.repair_feedback[0]


def test_stepik_feedback_starts_repair_with_full_validation_cycle() -> None:
    original = solution("number", "5")
    corrected = solution("number", "4")
    gateway = FakeGateway(original, repairs=(corrected,))
    service = DryRunService(
        loader=FakeLoader(StepTaskData(30, "number", "2 + 2?", {})),
        analyzer=TaskAnalyzer(),
        gateway=gateway,
    )

    result = asyncio.run(
        service.repair(target(), candidate=original, feedback="Stepik says the answer is wrong")
    )

    assert result.candidate.answer == "4"
    assert result.repair_attempts == 1
    assert result.reviewer_accepted is True
    assert "Stepik says the answer is wrong" in gateway.repair_feedback[0]
    assert any(
        "Stepik says the answer is wrong" in material
        for material in gateway.repair_contexts[0].materials
    )
    assert any(
        "Stepik says the answer is wrong" in material
        for material in gateway.review_contexts[0].materials
    )


def test_stepik_choice_repair_can_proceed_when_reviewer_keeps_dissenting() -> None:
    original = solution("choice", "openssl, crypt, passwd").model_copy(
        update={"selected_choice_indices": [2, 3, 4]}
    )
    corrected = solution("choice", "crypt, openssl").model_copy(
        update={"selected_choice_indices": [3, 4]}
    )
    gateway = FakeGateway(
        original,
        repairs=(corrected,),
        reviews=(review(is_acceptable=False, issue="Reviewer remains uncertain"),),
    )
    service = DryRunService(
        loader=FakeLoader(StepTaskData(123974, "choice", "Select utilities", {})),
        analyzer=TaskAnalyzer(),
        gateway=gateway,
        max_repair_attempts=1,
    )

    result = asyncio.run(
        service.repair_with_dataset(
            target(),
            candidate=original,
            feedback="Stepik returned verdict: wrong",
            dataset={
                "is_multiple_choice": True,
                "options": ["shadow", "passwd", "chage", "crypt", "openssl"],
            },
        )
    )

    assert result.candidate == corrected
    assert result.reply == {"choices": [False, False, False, True, True]}
    assert result.reviewer_accepted is False
    assert "advisory" in result.warnings[-1]


def test_stepik_choice_repair_uses_default_repair_budget_after_two_rejections() -> None:
    original = solution("choice", "openssl, crypt, passwd").model_copy(
        update={"selected_choice_indices": [2, 3, 4]}
    )
    first_repair = solution("choice", "crypt, passwd").model_copy(
        update={"selected_choice_indices": [1, 3]}
    )
    final_repair = solution("choice", "crypt, openssl").model_copy(
        update={"selected_choice_indices": [3, 4]}
    )
    gateway = FakeGateway(
        original,
        repairs=(first_repair, final_repair),
        reviews=(
            review(is_acceptable=False, issue="Reviewer remains uncertain"),
            review(is_acceptable=False, issue="Reviewer still disagrees"),
        ),
    )
    service = DryRunService(
        loader=FakeLoader(StepTaskData(123974, "choice", "Select utilities", {})),
        analyzer=TaskAnalyzer(),
        gateway=gateway,
    )

    result = asyncio.run(
        service.repair_with_dataset(
            target(),
            candidate=original,
            feedback="Stepik returned verdict: wrong",
            dataset={
                "is_multiple_choice": True,
                "options": ["shadow", "passwd", "chage", "crypt", "openssl"],
            },
        )
    )

    assert result.candidate == final_repair
    assert result.reply == {"choices": [False, False, False, True, True]}
    assert result.reviewer_accepted is False
    assert gateway.review_calls == 2
    assert len(gateway.repair_feedback) == 2


def test_fresh_choice_still_requires_reviewer_acceptance() -> None:
    candidate = solution("choice", "crypt, openssl").model_copy(
        update={"selected_choice_indices": [3, 4]}
    )
    gateway = FakeGateway(
        candidate,
        reviews=(review(is_acceptable=False, issue="Candidate is unverified"),),
    )
    service = DryRunService(
        loader=FakeLoader(
            StepTaskData(
                123974,
                "choice",
                "Select utilities",
                {
                    "is_multiple_choice": True,
                    "options": ["shadow", "passwd", "chage", "crypt", "openssl"],
                },
            )
        ),
        analyzer=TaskAnalyzer(),
        gateway=gateway,
        max_repair_attempts=0,
    )

    with pytest.raises(CandidateValidationError, match="reviewer rejected"):
        asyncio.run(service.execute(target()))


def test_compiler_harness_missing_function_forces_another_repair() -> None:
    original = solution("code", "void cache_init() {}", language="c++")
    still_missing = solution("code", "void cache_init() {}", language="c++")
    corrected = solution(
        "code",
        "void cache_init() {}\nvoid cache_setup(cache *item, std::size_t size) {}",
        language="c++",
    )
    gateway = FakeGateway(original, repairs=(still_missing, corrected))
    service = DryRunService(
        loader=FakeLoader(
            StepTaskData(
                183072,
                "code",
                "Implement the cache allocator.",
                {"code_templates": {"c++": "void cache_setup(cache*, std::size_t);"}},
            )
        ),
        analyzer=TaskAnalyzer(),
        gateway=gateway,
    )
    feedback = (
        "main.cpp: In function \u2018void call_setup(cache*, std::size_t)\u2019:\n"
        "main.cpp: error: \u2018cache_setup\u2019 was not declared in this scope"
    )

    result = asyncio.run(service.repair(target(), candidate=original, feedback=feedback))

    assert result.candidate == corrected
    assert result.repair_attempts == 2
    assert gateway.review_calls == 1
    assert "platform-required function(s)" in gateway.repair_feedback[1]


def test_compiler_feedback_rejects_redefined_and_incomplete_structs() -> None:
    original = solution("code", "struct lock { int value; };", language="c")
    still_wrong = solution("code", "struct lock { int value; };", language="c")
    corrected = solution(
        "code",
        (
            "struct wdlock { int value; };\n"
            "struct wdlock_ctx { int value; };\n"
            "void wdlock_init(struct wdlock *item) {}\n"
            "void wdlock_ctx_init(struct wdlock_ctx *ctx) {}"
        ),
        language="c",
    )
    gateway = FakeGateway(original, repairs=(still_wrong, corrected))
    service = DryRunService(
        loader=FakeLoader(
            StepTaskData(
                187325,
                "code",
                "Implement wound-wait locks without redefining the base lock.",
                {"code_templates": {"c": "struct wdlock; struct wdlock_ctx;"}},
            )
        ),
        analyzer=TaskAnalyzer(),
        gateway=gateway,
    )
    feedback = (
        "main.c: error: redefinition of \u2018struct lock\u2019\n"
        "main.c: error: array type has incomplete element type \u2018struct wdlock\u2019\n"
        "main.c: warning: implicit declaration of function \u2018wdlock_init\u2019\n"
        "main.c: warning: implicit declaration of function \u2018wdlock_ctx_init\u2019\n"
        "main.c: error: storage size of \u2018ctx\u2019 isn\u2019t known\n"
        "struct wdlock_ctx ctx;"
    )

    result = asyncio.run(service.repair(target(), candidate=original, feedback=feedback))

    assert result.candidate == corrected
    assert "platform-required function(s)" in gateway.repair_feedback[1]
