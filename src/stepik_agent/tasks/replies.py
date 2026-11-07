"""Build validated Stepik submission replies from reviewed candidates."""

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Protocol

from stepik_agent.ai.models import SolutionCandidate
from stepik_agent.core.errors import CandidateValidationError, UnsupportedTaskError
from stepik_agent.tasks.analyzer import html_to_text
from stepik_agent.tasks.models import StepTaskData, TaskAnalysis


class TaskReplyAdapter(Protocol):
    def supports(self, block_name: str) -> bool: ...

    def build_reply(
        self,
        task: StepTaskData,
        analysis: TaskAnalysis,
        candidate: SolutionCandidate,
    ) -> dict[str, object]: ...


class ReplyAdapterRegistry:
    def __init__(self, adapters: tuple[TaskReplyAdapter, ...] | None = None) -> None:
        self._adapters = adapters or (
            CodeReplyAdapter(),
            NumberReplyAdapter(),
            StringReplyAdapter(),
            MathReplyAdapter(),
            ChoiceReplyAdapter(),
            MatchingReplyAdapter(),
            FillBlanksReplyAdapter(),
            TableReplyAdapter(),
            DatasetReplyAdapter(),
        )

    def build_reply(
        self,
        task: StepTaskData,
        analysis: TaskAnalysis,
        candidate: SolutionCandidate,
    ) -> dict[str, object]:
        for adapter in self._adapters:
            if adapter.supports(task.block_name):
                return adapter.build_reply(task, analysis, candidate)
        raise UnsupportedTaskError(
            f"No submission adapter for Stepik block type: {task.block_name}"
        )


class _BaseReplyAdapter:
    block_names: frozenset[str] = frozenset()

    def supports(self, block_name: str) -> bool:
        return block_name in self.block_names


class CodeReplyAdapter(_BaseReplyAdapter):
    block_names = frozenset({"code"})

    def build_reply(
        self,
        task: StepTaskData,
        analysis: TaskAnalysis,
        candidate: SolutionCandidate,
    ) -> dict[str, object]:
        del task, analysis
        if candidate.language is None or not candidate.language.strip():
            raise CandidateValidationError("Code candidate must declare a language")
        return {"code": candidate.answer, "language": candidate.language.strip()}


class NumberReplyAdapter(_BaseReplyAdapter):
    block_names = frozenset({"number"})

    def build_reply(
        self,
        task: StepTaskData,
        analysis: TaskAnalysis,
        candidate: SolutionCandidate,
    ) -> dict[str, object]:
        del task, analysis
        value = candidate.answer.strip()
        try:
            parsed = Decimal(value)
        except InvalidOperation as error:
            raise CandidateValidationError("Number candidate must be a finite decimal") from error
        if not parsed.is_finite():
            raise CandidateValidationError("Number candidate must be finite")
        return {"number": value}


class StringReplyAdapter(_BaseReplyAdapter):
    block_names = frozenset({"string"})

    def build_reply(
        self,
        task: StepTaskData,
        analysis: TaskAnalysis,
        candidate: SolutionCandidate,
    ) -> dict[str, object]:
        del task, analysis
        return {"text": candidate.answer.strip()}


class MathReplyAdapter(_BaseReplyAdapter):
    block_names = frozenset({"math"})

    def build_reply(
        self,
        task: StepTaskData,
        analysis: TaskAnalysis,
        candidate: SolutionCandidate,
    ) -> dict[str, object]:
        del task, analysis
        return {"formula": candidate.answer.strip()}


class ChoiceReplyAdapter(_BaseReplyAdapter):
    block_names = frozenset({"choice"})

    def build_reply(
        self,
        task: StepTaskData,
        analysis: TaskAnalysis,
        candidate: SolutionCandidate,
    ) -> dict[str, object]:
        del analysis
        choices = _visible_choices(task.options)
        if not choices:
            raise UnsupportedTaskError("Choice submission requires visible options")
        selected_indices = candidate.selected_choice_indices
        if selected_indices is not None:
            if not selected_indices or len(selected_indices) != len(set(selected_indices)):
                raise CandidateValidationError("Choice indices must be non-empty and unique")
            if any(index < 0 or index >= len(choices) for index in selected_indices):
                raise CandidateValidationError("Choice index is out of range")
            if not _multiple_choice_option(task.options) and len(selected_indices) != 1:
                raise CandidateValidationError("Single choice task requires exactly one index")
            selected_index_set = set(selected_indices)
            return {"choices": [index in selected_index_set for index in range(len(choices))]}
        selected_choices = candidate.selected_choices
        if selected_choices is None:
            selected = candidate.answer.strip()
            if selected not in choices:
                raise CandidateValidationError("AI answer does not match a visible choice")
            selected_choices = [selected]
        if not _multiple_choice_option(task.options) and len(selected_choices) != 1:
            raise CandidateValidationError("Single choice task requires exactly one answer")
        if not set(selected_choices) <= set(choices):
            raise CandidateValidationError("AI selected choices contain an invisible choice")
        return {"choices": [choice in selected_choices for choice in choices]}


class MatchingReplyAdapter(_BaseReplyAdapter):
    block_names = frozenset({"matching"})

    def build_reply(
        self,
        task: StepTaskData,
        analysis: TaskAnalysis,
        candidate: SolutionCandidate,
    ) -> dict[str, object]:
        del analysis
        pairs = task.options.get("pairs")
        size = len(pairs) if isinstance(pairs, list) else 0
        if candidate.ordering is None or sorted(candidate.ordering) != list(range(size)):
            raise CandidateValidationError("Matching reply requires a complete ordering")
        return {"ordering": candidate.ordering}


class FillBlanksReplyAdapter(_BaseReplyAdapter):
    block_names = frozenset({"fill-blanks"})

    def build_reply(
        self,
        task: StepTaskData,
        analysis: TaskAnalysis,
        candidate: SolutionCandidate,
    ) -> dict[str, object]:
        del task
        if candidate.blanks is None or len(candidate.blanks) != len(analysis.blank_options):
            raise CandidateValidationError("Fill-blanks reply requires every blank value")
        return {"blanks": candidate.blanks}


class TableReplyAdapter(_BaseReplyAdapter):
    block_names = frozenset({"table"})

    def build_reply(
        self,
        task: StepTaskData,
        analysis: TaskAnalysis,
        candidate: SolutionCandidate,
    ) -> dict[str, object]:
        del task
        selections = candidate.table_selections
        if selections is None or len(selections) != len(analysis.table_rows):
            raise CandidateValidationError("Table reply requires every row")
        by_row = {selection.row_index: selection for selection in selections}
        if set(by_row) != set(range(len(analysis.table_rows))):
            raise CandidateValidationError("Table reply row indices are incomplete")
        return {
            "choices": [
                {
                    "name_row": row,
                    "columns": [
                        {"name": column, "answer": column_index in selected}
                        for column_index, column in enumerate(analysis.table_columns)
                    ],
                }
                for row_index, row in enumerate(analysis.table_rows)
                for selected in [set(by_row[row_index].selected_column_indices)]
            ]
        }


class DatasetReplyAdapter(_BaseReplyAdapter):
    block_names = frozenset({"dataset"})

    def build_reply(
        self,
        task: StepTaskData,
        analysis: TaskAnalysis,
        candidate: SolutionCandidate,
    ) -> dict[str, object]:
        del task, analysis
        answer = candidate.answer.strip()
        if not answer:
            raise CandidateValidationError("Dataset answer file must not be empty")
        return {"file": answer}


def _visible_choices(options: Mapping[str, object]) -> tuple[str, ...]:
    raw_choices = options.get("options", options.get("choices"))
    if not isinstance(raw_choices, list):
        return ()
    result: list[str] = []
    for item in raw_choices:
        if isinstance(item, str) and (text := html_to_text(item)):
            result.append(text)
        elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
            text = html_to_text(item["text"])
            if text:
                result.append(text)
    return tuple(result)


def _multiple_choice_option(options: Mapping[str, object]) -> bool:
    for key in ("is_multiple_choice", "multiple", "is_multiple"):
        value = options.get(key)
        if isinstance(value, bool):
            return value
    return False
