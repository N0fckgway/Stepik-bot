"""Extensible Stepik block adapters used before and after an AI call."""

import re
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import Protocol

from stepik_agent.ai.models import SolutionCandidate, TaskContext
from stepik_agent.core.errors import CandidateValidationError, UnsupportedTaskError
from stepik_agent.tasks.models import StepTaskData, TaskAnalysis

_INFORMATION_BLOCKS = frozenset({"text", "video", "animation"})
_ASM64_BARE_REGISTER = re.compile(
    r"(?<![%\w])(?:r(?:ax|bx|cx|dx|si|di|bp|sp|8|9|1[0-5]))\b",
    re.IGNORECASE,
)
_ASM64_LABEL = re.compile(r"(?m)^\s*([A-Za-z_.$][\w.$]*):")
_PUBLIC_FUNCTION = re.compile(
    r"(?m)^\s*[A-Za-z_][\w:\s<>,*&]*\b([A-Za-z_]\w*)\s*\([^;{}\n]*\)\s*(?:;|\{)"
)


class TaskAdapter(Protocol):
    def supports(self, block_name: str) -> bool: ...

    def analyze(self, task: StepTaskData) -> TaskAnalysis: ...

    def validate(self, analysis: TaskAnalysis, candidate: SolutionCandidate) -> None: ...


class TaskAnalyzer:
    def __init__(self, adapters: Iterable[TaskAdapter] | None = None) -> None:
        self._adapters = tuple(
            adapters
            or (
                TextAnswerAdapter(),
                NumberAdapter(),
                CodeAdapter(),
                ChoiceAdapter(),
                MatchingAdapter(),
                FillBlanksAdapter(),
                TableAdapter(),
                DatasetAdapter(),
            )
        )

    def analyze(self, task: StepTaskData) -> TaskAnalysis:
        if task.block_name in _INFORMATION_BLOCKS:
            raise UnsupportedTaskError(
                f"Step {task.step_id} is informational ({task.block_name}); no AI call is needed"
            )
        for adapter in self._adapters:
            if adapter.supports(task.block_name):
                return adapter.analyze(task)
        raise UnsupportedTaskError(f"Unsupported Stepik block type: {task.block_name}")

    def validate(self, analysis: TaskAnalysis, candidate: SolutionCandidate) -> None:
        for adapter in self._adapters:
            if adapter.supports(analysis.context.task_type):
                adapter.validate(analysis, candidate)
                return
        raise UnsupportedTaskError(
            f"No validator for Stepik block type: {analysis.context.task_type}"
        )


class BaseAdapter:
    block_names: frozenset[str] = frozenset()
    expected_answer_format = "Plain text"

    def supports(self, block_name: str) -> bool:
        return block_name in self.block_names

    def analyze(self, task: StepTaskData) -> TaskAnalysis:
        statement = html_to_text(task.statement_html)
        if not statement:
            raise UnsupportedTaskError(f"Step {task.step_id} has no readable task statement")
        if len(statement) > 100_000:
            raise UnsupportedTaskError(f"Step {task.step_id} statement exceeds the safe size limit")
        materials = tuple(
            text
            for item in task.preceding_materials_html[-5:]
            if (text := html_to_text(item)) and len(text) <= 100_000
        )
        return TaskAnalysis(
            context=TaskContext(
                step_id=task.step_id,
                task_type=task.block_name,
                statement=statement,
                materials=list(materials),
                constraints=[],
                expected_answer_format=self.expected_answer_format,
            )
        )

    def validate(self, analysis: TaskAnalysis, candidate: SolutionCandidate) -> None:
        if candidate.task_type != analysis.context.task_type:
            raise CandidateValidationError(
                "AI candidate task_type does not match the analyzed Stepik block"
            )


class TextAnswerAdapter(BaseAdapter):
    block_names = frozenset({"string", "math"})
    expected_answer_format = "One plain-text answer without commentary or Markdown fences"


class NumberAdapter(BaseAdapter):
    block_names = frozenset({"number"})
    expected_answer_format = "One finite decimal number using a dot as decimal separator"

    def validate(self, analysis: TaskAnalysis, candidate: SolutionCandidate) -> None:
        super().validate(analysis, candidate)
        try:
            number = Decimal(candidate.answer.strip())
        except InvalidOperation as error:
            raise CandidateValidationError("AI candidate is not a decimal number") from error
        if not number.is_finite():
            raise CandidateValidationError("AI candidate number must be finite")


class CodeAdapter(BaseAdapter):
    block_names = frozenset({"code"})
    expected_answer_format = "Source code only; language must identify a Stepik language"

    def analyze(self, task: StepTaskData) -> TaskAnalysis:
        base = super().analyze(task)
        allowed_languages = _code_languages(task.options)
        starter_code = _starter_code(task.options, allowed_languages)
        assembler_syntax = "att" if allowed_languages == ("asm64",) else None
        warnings: tuple[str, ...] = ()
        constraints = list(base.context.constraints)
        materials = list(base.context.materials)
        if allowed_languages:
            languages_text = ", ".join(allowed_languages)
            constraints.append(f"Allowed languages: {languages_text}"[:500])
        else:
            warnings = (
                "Step API did not expose code_templates; language cannot be confirmed in dry run",
            )
        if assembler_syntax == "att":
            constraints.append(
                "Assembler syntax: GNU as AT&T. Prefix registers with %, use "
                "displacement(base) memory operands, and do not emit Intel syntax."
            )
        if starter_code:
            constraints.append(
                "Preserve every required public symbol, signature, label, and declaration from "
                "the complete starter template included in the materials."
            )
            materials.append(f"Complete required starter template:\n{starter_code}")
        return TaskAnalysis(
            context=base.context.model_copy(
                update={"constraints": constraints, "materials": materials}
            ),
            allowed_languages=allowed_languages,
            starter_code=starter_code,
            assembler_syntax=assembler_syntax,
            warnings=warnings,
        )

    def validate(self, analysis: TaskAnalysis, candidate: SolutionCandidate) -> None:
        super().validate(analysis, candidate)
        if not candidate.language:
            raise CandidateValidationError("Code candidate must declare a language")
        if analysis.allowed_languages and candidate.language not in analysis.allowed_languages:
            raise CandidateValidationError(
                f"Language {candidate.language!r} is not allowed by the Stepik step"
            )
        required_functions = set(_PUBLIC_FUNCTION.findall(analysis.starter_code or ""))
        missing_functions = sorted(
            name
            for name in required_functions
            if not re.search(rf"\b{re.escape(name)}\s*\(", candidate.answer)
        )
        if missing_functions:
            raise CandidateValidationError(
                f"Candidate is missing required function: {missing_functions[0]}"
            )
        if candidate.language == "asm64" and analysis.assembler_syntax == "att":
            _validate_asm64_candidate(analysis, candidate.answer)


class ChoiceAdapter(BaseAdapter):
    block_names = frozenset({"choice"})
    expected_answer_format = "selected_choice_indices with exactly one zero-based option index"

    def analyze(self, task: StepTaskData) -> TaskAnalysis:
        base = super().analyze(task)
        choices = _visible_choices(task.options)
        if not choices:
            raise UnsupportedTaskError(
                "Choice options require a Stepik attempt dataset; dry run does not create attempts"
            )
        constraints = [f"Visible choice: {choice[:400]}" for choice in choices[:98]]
        constraints.append("Choice indices are zero-based and follow the visible option order")
        if _multiple_choice_option(task.options):
            constraints.append("Choice mode: multiple")
            expected_format = "One or more unique zero-based indices in selected_choice_indices"
        else:
            expected_format = self.expected_answer_format
        return TaskAnalysis(
            context=base.context.model_copy(
                update={"constraints": constraints, "expected_answer_format": expected_format}
            ),
        )

    def validate(self, analysis: TaskAnalysis, candidate: SolutionCandidate) -> None:
        super().validate(analysis, candidate)
        choices = {
            constraint.removeprefix("Visible choice: ")
            for constraint in analysis.context.constraints
            if constraint.startswith("Visible choice: ")
        }
        if candidate.selected_choice_indices is not None:
            indices = candidate.selected_choice_indices
            if not indices:
                raise CandidateValidationError("AI selected_choice_indices must not be empty")
            if len(indices) != len(set(indices)):
                raise CandidateValidationError("AI selected_choice_indices must be unique")
            if any(index < 0 or index >= len(choices) for index in indices):
                raise CandidateValidationError("AI selected_choice_indices are out of range")
            if not _is_multiple_choice(analysis.context.constraints) and len(indices) != 1:
                raise CandidateValidationError("Single choice task requires exactly one index")
            return
        if candidate.selected_choices is not None:
            if not candidate.selected_choices:
                raise CandidateValidationError("AI selected_choices must not be empty")
            if not set(candidate.selected_choices) <= choices:
                raise CandidateValidationError("AI selected_choices contain an invisible choice")
            if (
                not _is_multiple_choice(analysis.context.constraints)
                and len(candidate.selected_choices) != 1
            ):
                raise CandidateValidationError("Single choice task requires exactly one answer")
            return
        if _is_multiple_choice(analysis.context.constraints):
            raise CandidateValidationError(
                "Multiple choice candidate must provide selected_choices"
            )
        if candidate.answer not in choices:
            raise CandidateValidationError("AI answer does not match a visible choice")


class MatchingAdapter(BaseAdapter):
    block_names = frozenset({"matching"})
    expected_answer_format = (
        "ordering must be a permutation of zero-based right-column indices, one per left item"
    )

    def analyze(self, task: StepTaskData) -> TaskAnalysis:
        base = super().analyze(task)
        pairs = _matching_pairs(task.options)
        if not pairs:
            raise UnsupportedTaskError("Matching task requires pairs from a Stepik attempt dataset")
        constraints = list(base.context.constraints)
        constraints.extend(
            f"Pair index {index}: left={left}; right option={right}"[:500]
            for index, (left, right) in enumerate(pairs)
        )
        constraints.append(
            "Return ordering[i] = index of the right option matching left item i. "
            "ordering must contain every index exactly once."
        )
        return TaskAnalysis(context=base.context.model_copy(update={"constraints": constraints}))

    def validate(self, analysis: TaskAnalysis, candidate: SolutionCandidate) -> None:
        super().validate(analysis, candidate)
        size = sum(
            constraint.startswith("Pair index ") for constraint in analysis.context.constraints
        )
        if candidate.ordering is None or sorted(candidate.ordering) != list(range(size)):
            raise CandidateValidationError(
                "Matching candidate ordering must be a complete zero-based permutation"
            )


class FillBlanksAdapter(BaseAdapter):
    block_names = frozenset({"fill-blanks"})
    expected_answer_format = "blanks must contain one exact string for every input/select blank"

    def analyze(self, task: StepTaskData) -> TaskAnalysis:
        base = super().analyze(task)
        components = _fill_blank_components(task.options)
        if not components:
            raise UnsupportedTaskError(
                "Fill-blanks task requires components from a Stepik attempt dataset"
            )
        constraints = list(base.context.constraints)
        blank_options: list[tuple[str, ...] | None] = []
        for component_type, text, options in components:
            if component_type == "text":
                if text:
                    constraints.append(f"Literal text: {text[:450]}")
                continue
            blank_index = len(blank_options)
            blank_options.append(options if component_type == "select" else None)
            if component_type == "select":
                rendered = ", ".join(
                    f"{index}={option}" for index, option in enumerate(options or ())
                )
                constraints.append(
                    f"Blank {blank_index} is a select; allowed values: {rendered[:430]}"
                )
            else:
                constraints.append(f"Blank {blank_index} is a free-text input")
        if not blank_options:
            raise UnsupportedTaskError("Fill-blanks attempt dataset contains no blanks")
        return TaskAnalysis(
            context=base.context.model_copy(update={"constraints": constraints}),
            blank_options=tuple(blank_options),
        )

    def validate(self, analysis: TaskAnalysis, candidate: SolutionCandidate) -> None:
        super().validate(analysis, candidate)
        if candidate.blanks is None or len(candidate.blanks) != len(analysis.blank_options):
            raise CandidateValidationError(
                "Fill-blanks candidate must provide one value for every blank"
            )
        for index, (value, options) in enumerate(
            zip(candidate.blanks, analysis.blank_options, strict=True)
        ):
            if options is not None and value not in options:
                raise CandidateValidationError(
                    f"Fill-blanks value for blank {index} is not an allowed select option"
                )


class TableAdapter(BaseAdapter):
    block_names = frozenset({"table"})
    expected_answer_format = (
        "table_selections must contain every zero-based row index and selected column indices"
    )

    def analyze(self, task: StepTaskData) -> TaskAnalysis:
        base = super().analyze(task)
        rows = _string_list(task.options.get("rows"))
        columns = _string_list(task.options.get("columns"))
        if not rows or not columns:
            raise UnsupportedTaskError(
                "Table task requires rows and columns from a Stepik attempt dataset"
            )
        is_checkbox = task.options.get("is_checkbox") is True
        constraints = list(base.context.constraints)
        constraints.extend(f"Table row {index}: {row[:440]}" for index, row in enumerate(rows))
        constraints.extend(
            f"Table column {index}: {column[:430]}" for index, column in enumerate(columns)
        )
        constraints.append(
            "Table mode: multiple cells per row"
            if is_checkbox
            else "Table mode: exactly one cell per row"
        )
        return TaskAnalysis(
            context=base.context.model_copy(update={"constraints": constraints}),
            table_rows=rows,
            table_columns=columns,
            table_is_checkbox=is_checkbox,
        )

    def validate(self, analysis: TaskAnalysis, candidate: SolutionCandidate) -> None:
        super().validate(analysis, candidate)
        selections = candidate.table_selections
        if selections is None or len(selections) != len(analysis.table_rows):
            raise CandidateValidationError("Table candidate must provide every row exactly once")
        row_indices = [selection.row_index for selection in selections]
        if sorted(row_indices) != list(range(len(analysis.table_rows))):
            raise CandidateValidationError("Table candidate row indices must be a permutation")
        for selection in selections:
            indices = selection.selected_column_indices
            if len(indices) != len(set(indices)):
                raise CandidateValidationError("Table selected column indices must be unique")
            if any(index < 0 or index >= len(analysis.table_columns) for index in indices):
                raise CandidateValidationError("Table selected column index is out of range")
            if not analysis.table_is_checkbox and len(indices) != 1:
                raise CandidateValidationError("Single-choice table requires one cell per row")


class DatasetAdapter(BaseAdapter):
    block_names = frozenset({"dataset"})
    expected_answer_format = "Exact plain-text contents of the answer file"

    def analyze(self, task: StepTaskData) -> TaskAnalysis:
        base = super().analyze(task)
        if not task.dataset_text:
            raise UnsupportedTaskError("Dataset task requires downloaded attempt data")
        if len(task.dataset_text) > 200_000:
            raise UnsupportedTaskError("Dataset contents exceed the AI context safety limit")
        chunks = [
            task.dataset_text[index : index + 90_000]
            for index in range(0, len(task.dataset_text), 90_000)
        ]
        materials = [
            *base.context.materials,
            *(f"Attempt dataset part {index + 1}:\n{chunk}" for index, chunk in enumerate(chunks)),
        ]
        return TaskAnalysis(context=base.context.model_copy(update={"materials": materials}))


def _matching_pairs(options: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    raw_pairs = options.get("pairs")
    if not isinstance(raw_pairs, list):
        return ()
    result: list[tuple[str, str]] = []
    for item in raw_pairs:
        if not isinstance(item, Mapping):
            return ()
        first = item.get("first")
        second = item.get("second")
        if not isinstance(first, str) or not isinstance(second, str):
            return ()
        left = html_to_text(first)
        right = html_to_text(second)
        if not left:
            return ()
        result.append((left, right))
    return tuple(result)


def _fill_blank_components(
    options: Mapping[str, object],
) -> tuple[tuple[str, str, tuple[str, ...] | None], ...]:
    raw_components = options.get("components")
    if not isinstance(raw_components, list):
        return ()
    result: list[tuple[str, str, tuple[str, ...] | None]] = []
    for item in raw_components:
        if not isinstance(item, Mapping):
            return ()
        component_type = item.get("type")
        if component_type not in {"text", "input", "select"}:
            return ()
        raw_text = item.get("text", "")
        if not isinstance(raw_text, str):
            return ()
        text = html_to_text(raw_text)
        raw_options = item.get("options", [])
        parsed_options = _string_list(raw_options)
        if component_type == "select" and not parsed_options:
            return ()
        result.append(
            (
                component_type,
                text,
                parsed_options if component_type == "select" else None,
            )
        )
    return tuple(result)


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not (text := html_to_text(item)):
            return ()
        result.append(text)
    return tuple(result)


def _is_multiple_choice(constraints: list[str]) -> bool:
    return "Choice mode: multiple" in constraints


def _multiple_choice_option(options: Mapping[str, object]) -> bool:
    for key in ("is_multiple_choice", "multiple", "is_multiple"):
        value = options.get(key)
        if isinstance(value, bool):
            return value
    return False


def _code_languages(options: Mapping[str, object]) -> tuple[str, ...]:
    templates = options.get("code_templates")
    if not isinstance(templates, Mapping):
        return ()
    return tuple(
        sorted(key.strip()[:100] for key in templates if isinstance(key, str) and key.strip())
    )[:50]


def _starter_code(options: Mapping[str, object], allowed_languages: tuple[str, ...]) -> str | None:
    if len(allowed_languages) != 1:
        return None
    templates = options.get("code_templates")
    if not isinstance(templates, Mapping):
        return None
    template = templates.get(allowed_languages[0])
    if not isinstance(template, str):
        return None
    template = template.strip()
    return template[:20_000] or None


def _validate_asm64_candidate(analysis: TaskAnalysis, answer: str) -> None:
    if ".intel_syntax" in answer.lower() or _ASM64_BARE_REGISTER.search(answer):
        raise CandidateValidationError(
            "ASM64 candidate must use GNU as AT&T syntax, not Intel syntax"
        )

    if not re.search(r"(?m)^\s*retq?\s*(?://.*|#.*)?$", answer):
        raise CandidateValidationError("ASM64 function candidate must contain ret or retq")

    required_labels = set(_ASM64_LABEL.findall(analysis.starter_code or ""))
    candidate_labels = set(_ASM64_LABEL.findall(answer))
    missing_labels = sorted(required_labels - candidate_labels)
    if missing_labels:
        raise CandidateValidationError(
            f"ASM64 candidate is missing required label: {missing_labels[0]}"
        )


def _visible_choices(options: Mapping[str, object]) -> tuple[str, ...]:
    raw_choices = options.get("options", options.get("choices"))
    if not isinstance(raw_choices, list):
        return ()
    choices: list[str] = []
    for item in raw_choices:
        if isinstance(item, str):
            text = html_to_text(item)
        elif isinstance(item, Mapping):
            raw_text = item.get("text")
            text = html_to_text(raw_text) if isinstance(raw_text, str) else ""
        else:
            text = ""
        if text:
            choices.append(text)
    return tuple(choices)


class _TextExtractor(HTMLParser):
    _BLOCK_TAGS = frozenset(
        {"br", "div", "p", "pre", "code", "li", "ul", "ol", "h1", "h2", "h3", "h4"}
    )
    _IGNORED_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
        elif self._ignored_depth == 0 and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        elif self._ignored_depth == 0 and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)


def html_to_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line).strip()
