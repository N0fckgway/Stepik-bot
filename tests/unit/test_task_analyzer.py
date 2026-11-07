import pytest

from stepik_agent.ai.models import SolutionCandidate, TableSelection
from stepik_agent.core.errors import CandidateValidationError, UnsupportedTaskError
from stepik_agent.tasks.analyzer import TaskAnalyzer, html_to_text
from stepik_agent.tasks.models import StepTaskData


def candidate(*, task_type: str, answer: str, language: str | None = None) -> SolutionCandidate:
    return SolutionCandidate(
        task_type=task_type,
        answer=answer,
        language=language,
        explanation="Validated candidate",
        confidence=0.9,
    )


def test_code_adapter_builds_minimal_context_and_checks_language() -> None:
    task = StepTaskData(
        step_id=10,
        block_name="code",
        statement_html="<p>Print <code>Hello</code>.</p><script>ignore me</script>",
        options={"code_templates": {"python3": "", "asm64": ""}},
        preceding_materials_html=("<h2>Theory</h2><p>Use stdout.</p>",),
    )
    analyzer = TaskAnalyzer()

    analysis = analyzer.analyze(task)

    assert analysis.context.statement == "Print\nHello\n."
    assert analysis.context.materials == ["Theory\nUse stdout."]
    assert analysis.allowed_languages == ("asm64", "python3")
    analyzer.validate(analysis, candidate(task_type="code", answer="mov rax, 1", language="asm64"))
    with pytest.raises(CandidateValidationError, match="not allowed"):
        analyzer.validate(
            analysis,
            candidate(task_type="code", answer="puts 'x'", language="ruby"),
        )


def test_asm64_adapter_rejects_intel_syntax_and_preserves_starter_label() -> None:
    task = StepTaskData(
        step_id=170493,
        block_name="code",
        statement_html="<p>Implement swap for two 64-bit pointers.</p>",
        options={"code_templates": {"asm64": "swap:\n    // Put your code here"}},
    )
    analyzer = TaskAnalyzer()

    analysis = analyzer.analyze(task)

    assert analysis.assembler_syntax == "att"
    assert analysis.starter_code == "swap:\n    // Put your code here"
    assert any("GNU as AT&T" in constraint for constraint in analysis.context.constraints)
    assert any("swap:" in material for material in analysis.context.materials)

    with pytest.raises(CandidateValidationError, match="AT&T"):
        analyzer.validate(
            analysis,
            candidate(
                task_type="code",
                answer="mov rax, [rdi]\nxchg rax, [rsi]\nmov [rdi], rax\nret",
                language="asm64",
            ),
        )


def test_complete_long_starter_template_is_available_to_ai_context() -> None:
    marker = "void cache_setup(cache *item, std::size_t size);"
    starter = f"// {'x' * 2_000}\n{marker}"
    task = StepTaskData(
        step_id=183072,
        block_name="code",
        statement_html="Implement cache allocator",
        options={"code_templates": {"c++": starter}},
    )

    analysis = TaskAnalyzer().analyze(task)

    assert marker in analysis.context.materials[-1]


def test_code_candidate_preserves_all_required_starter_functions() -> None:
    task = StepTaskData(
        step_id=186190,
        block_name="code",
        statement_html="Implement atomic helpers",
        options={
            "code_templates": {
                "c++11": (
                    "int atomic_fetch_add(atomic_int *x, int increment);\n"
                    "bool atomic_compare_exchange(atomic_int *x, int *expected, int new_value);"
                )
            }
        },
    )
    analyzer = TaskAnalyzer()
    analysis = analyzer.analyze(task)

    with pytest.raises(CandidateValidationError, match="atomic_compare_exchange"):
        analyzer.validate(
            analysis,
            candidate(
                task_type="code",
                answer="int atomic_fetch_add(atomic_int *x, int increment) { return 0; }",
                language="c++11",
            ),
        )


def test_asm64_adapter_requires_starter_label_and_accepts_att_syntax() -> None:
    task = StepTaskData(
        step_id=170493,
        block_name="code",
        statement_html="<p>Implement swap for two 64-bit pointers.</p>",
        options={"code_templates": {"asm64": "swap:\n    // Put your code here"}},
    )
    analyzer = TaskAnalyzer()
    analysis = analyzer.analyze(task)
    body = "movq (%rdi), %rax\nxchgq %rax, (%rsi)\nmovq %rax, (%rdi)\nretq"

    with pytest.raises(CandidateValidationError, match="missing required label: swap"):
        analyzer.validate(
            analysis,
            candidate(task_type="code", answer=body, language="asm64"),
        )

    analyzer.validate(
        analysis,
        candidate(task_type="code", answer=f"swap:\n    {body}", language="asm64"),
    )


def test_number_adapter_rejects_non_numeric_candidate() -> None:
    analyzer = TaskAnalyzer()
    analysis = analyzer.analyze(StepTaskData(11, "number", "<p>What is 2 + 2?</p>", {}))

    with pytest.raises(CandidateValidationError, match="not a decimal"):
        analyzer.validate(analysis, candidate(task_type="number", answer="four"))


def test_choice_without_attempt_dataset_is_rejected_before_ai() -> None:
    analyzer = TaskAnalyzer()

    with pytest.raises(UnsupportedTaskError, match="attempt dataset"):
        analyzer.analyze(StepTaskData(12, "choice", "Choose one", {}))


def test_choice_uses_official_attempt_options_field() -> None:
    analyzer = TaskAnalyzer()
    task = StepTaskData(
        12,
        "choice",
        "Choose one",
        {"options": ["A", "B"], "is_multiple_choice": False},
    )

    analysis = analyzer.analyze(task)

    analyzer.validate(analysis, candidate(task_type="choice", answer="B"))


def test_choice_accepts_stable_zero_based_indices() -> None:
    analyzer = TaskAnalyzer()
    task = StepTaskData(
        12,
        "choice",
        "Choose all valid states",
        {"options": ["A", "B", "C"], "is_multiple_choice": True},
    )
    analysis = analyzer.analyze(task)
    indexed = candidate(task_type="choice", answer="B and C").model_copy(
        update={"selected_choice_indices": [1, 2]}
    )

    analyzer.validate(analysis, indexed)
    with pytest.raises(CandidateValidationError, match="out of range"):
        analyzer.validate(
            analysis,
            indexed.model_copy(update={"selected_choice_indices": [3]}),
        )


def test_matching_attempt_dataset_builds_and_validates_ordering() -> None:
    analyzer = TaskAnalyzer()
    task = StepTaskData(
        14,
        "matching",
        "Match terms",
        {"pairs": [{"first": "A", "second": "2"}, {"first": "B", "second": "1"}]},
    )
    analysis = analyzer.analyze(task)
    valid = candidate(task_type="matching", answer="B->1, A->2").model_copy(
        update={"ordering": [1, 0]}
    )
    analyzer.validate(analysis, valid)

    with pytest.raises(CandidateValidationError, match="permutation"):
        analyzer.validate(
            analysis,
            valid.model_copy(update={"ordering": [0, 0]}),
        )


def test_matching_allows_an_explicit_empty_right_option() -> None:
    task = StepTaskData(
        14,
        "matching",
        "Match binaries to directories",
        {
            "pairs": [
                {"first": "alias", "second": "/sbin"},
                {"first": "rm", "second": ""},
            ]
        },
    )
    analysis = TaskAnalyzer().analyze(task)
    valid = candidate(task_type="matching", answer="alias is empty").model_copy(
        update={"ordering": [1, 0]}
    )

    TaskAnalyzer().validate(analysis, valid)


def test_fill_blanks_uses_attempt_components_and_validates_selects() -> None:
    task = StepTaskData(
        15,
        "fill-blanks",
        "Count env output lines",
        {
            "components": [
                {"type": "text", "text": "env |"},
                {"type": "select", "text": "", "options": ["wc", "grep"]},
                {"type": "text", "text": "-"},
                {"type": "input", "text": "", "options": []},
            ]
        },
    )
    analyzer = TaskAnalyzer()
    analysis = analyzer.analyze(task)
    valid = candidate(task_type="fill-blanks", answer="wc and l").model_copy(
        update={"blanks": ["wc", "l"]}
    )

    analyzer.validate(analysis, valid)
    with pytest.raises(CandidateValidationError, match="not an allowed select option"):
        analyzer.validate(analysis, valid.model_copy(update={"blanks": ["cat", "l"]}))


def test_table_uses_rows_columns_and_validates_each_row() -> None:
    task = StepTaskData(
        16,
        "table",
        "Select permissions",
        {
            "rows": ["other", "owner", "group"],
            "columns": ["write", "read", "execute"],
            "is_checkbox": True,
        },
    )
    analyzer = TaskAnalyzer()
    analysis = analyzer.analyze(task)
    valid = candidate(task_type="table", answer="permission matrix").model_copy(
        update={
            "table_selections": [
                TableSelection(row_index=0, selected_column_indices=[]),
                TableSelection(row_index=1, selected_column_indices=[0, 1, 2]),
                TableSelection(row_index=2, selected_column_indices=[1, 2]),
            ]
        }
    )

    analyzer.validate(analysis, valid)
    with pytest.raises(CandidateValidationError, match="every row"):
        analyzer.validate(
            analysis,
            valid.model_copy(update={"table_selections": valid.table_selections[:2]}),
        )


def test_dataset_attempt_text_is_added_to_context() -> None:
    task = StepTaskData(15, "dataset", "Sum the values", {}, dataset_text="1 2 3")
    analysis = TaskAnalyzer().analyze(task)
    assert analysis.context.materials[-1] == "Attempt dataset part 1:\n1 2 3"


@pytest.mark.parametrize("block_name", ["text", "video", "animation"])
def test_informational_steps_do_not_use_ai(block_name: str) -> None:
    with pytest.raises(UnsupportedTaskError, match="informational"):
        TaskAnalyzer().analyze(StepTaskData(13, block_name, "Material", {}))


def test_html_parser_does_not_include_active_content() -> None:
    assert html_to_text("<p>Safe</p><style>.secret{}</style><script>steal()</script>") == "Safe"
