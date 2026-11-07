from stepik_agent.ai.models import SolutionCandidate, TableSelection
from stepik_agent.tasks.analyzer import TaskAnalyzer
from stepik_agent.tasks.models import StepTaskData
from stepik_agent.tasks.replies import ReplyAdapterRegistry


def candidate(
    task_type: str,
    answer: str,
    *,
    selected: list[str] | None = None,
    language: str | None = None,
) -> SolutionCandidate:
    return SolutionCandidate(
        task_type=task_type,
        answer=answer,
        selected_choices=selected,
        language=language,
        explanation="ok",
        confidence=1,
    )


def test_scalar_reply_contracts():
    registry = ReplyAdapterRegistry()
    for block_name, answer, expected in (
        ("number", "4", {"number": "4"}),
        ("string", "hello", {"text": "hello"}),
        ("math", "x^2", {"formula": "x^2"}),
    ):
        task = StepTaskData(1, block_name, "statement", {})
        analysis = TaskAnalyzer().analyze(task)
        assert registry.build_reply(task, analysis, candidate(block_name, answer)) == expected


def test_code_reply_contract():
    task = StepTaskData(1, "code", "statement", {})
    analysis = TaskAnalyzer().analyze(task)
    assert ReplyAdapterRegistry().build_reply(
        task, analysis, candidate("code", "ret", language="asm64")
    ) == {"code": "ret", "language": "asm64"}


def test_single_and_multiple_choice_replies():
    registry = ReplyAdapterRegistry()
    single = StepTaskData(1, "choice", "statement", {"options": ["A", "B"]})
    single_analysis = TaskAnalyzer().analyze(single)
    assert registry.build_reply(single, single_analysis, candidate("choice", "B")) == {
        "choices": [False, True]
    }

    multiple = StepTaskData(
        2,
        "choice",
        "statement",
        {"options": ["A", "B", "C"], "is_multiple_choice": True},
    )
    multiple_analysis = TaskAnalyzer().analyze(multiple)
    assert registry.build_reply(
        multiple,
        multiple_analysis,
        candidate("choice", "A,C", selected=["A", "C"]),
    ) == {"choices": [True, False, True]}

    indexed = candidate("choice", "B,C").model_copy(update={"selected_choice_indices": [1, 2]})
    assert registry.build_reply(multiple, multiple_analysis, indexed) == {
        "choices": [False, True, True]
    }


def test_matching_and_dataset_reply_contracts():
    registry = ReplyAdapterRegistry()
    matching = StepTaskData(
        3,
        "matching",
        "statement",
        {"pairs": [{"first": "A", "second": "2"}, {"first": "B", "second": "1"}]},
    )
    matching_analysis = TaskAnalyzer().analyze(matching)
    matching_candidate = candidate("matching", "mapping").model_copy(update={"ordering": [1, 0]})
    assert registry.build_reply(matching, matching_analysis, matching_candidate) == {
        "ordering": [1, 0]
    }

    dataset = StepTaskData(4, "dataset", "statement", {}, dataset_text="input")
    dataset_analysis = TaskAnalyzer().analyze(dataset)
    assert registry.build_reply(dataset, dataset_analysis, candidate("dataset", "42\n")) == {
        "file": "42"
    }


def test_fill_blanks_reply_contract() -> None:
    task = StepTaskData(
        5,
        "fill-blanks",
        "statement",
        {
            "components": [
                {"type": "text", "text": "sort "},
                {"type": "input", "text": "", "options": []},
                {"type": "text", "text": " tennis.txt"},
            ]
        },
    )
    analysis = TaskAnalyzer().analyze(task)
    solution = candidate("fill-blanks", "sort by second field").model_copy(
        update={"blanks": ["-k2"]}
    )

    assert ReplyAdapterRegistry().build_reply(task, analysis, solution) == {"blanks": ["-k2"]}


def test_table_reply_contract() -> None:
    task = StepTaskData(
        6,
        "table",
        "statement",
        {
            "rows": ["owner", "group"],
            "columns": ["write", "read"],
            "is_checkbox": True,
        },
    )
    analysis = TaskAnalyzer().analyze(task)
    solution = candidate("table", "matrix").model_copy(
        update={
            "table_selections": [
                TableSelection(row_index=0, selected_column_indices=[0, 1]),
                TableSelection(row_index=1, selected_column_indices=[1]),
            ]
        }
    )

    assert ReplyAdapterRegistry().build_reply(task, analysis, solution) == {
        "choices": [
            {
                "name_row": "owner",
                "columns": [
                    {"name": "write", "answer": True},
                    {"name": "read", "answer": True},
                ],
            },
            {
                "name_row": "group",
                "columns": [
                    {"name": "write", "answer": False},
                    {"name": "read", "answer": True},
                ],
            },
        ]
    }
