"""Validated domain models exchanged with an AI provider."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
LongText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100_000),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SuggestedTestCase(StrictModel):
    description: ShortText
    input: str | None = Field(default=None, max_length=20_000)
    expected_output: str | None = Field(default=None, max_length=20_000)


class TaskContext(StrictModel):
    step_id: int = Field(gt=0)
    task_type: ShortText
    statement: LongText
    materials: list[LongText] = Field(default_factory=list, max_length=50)
    constraints: list[ShortText] = Field(default_factory=list, max_length=100)
    expected_answer_format: ShortText


class TableSelection(StrictModel):
    row_index: int = Field(ge=0)
    selected_column_indices: list[int] = Field(default_factory=list, max_length=100)


class SolutionCandidate(StrictModel):
    task_type: ShortText
    answer: LongText
    selected_choices: list[ShortText] | None = Field(default=None, max_length=100)
    selected_choice_indices: list[int] | None = Field(default=None, max_length=100)
    ordering: list[int] | None = Field(default=None, max_length=100)
    blanks: list[ShortText] | None = Field(default=None, max_length=100)
    table_selections: list[TableSelection] | None = Field(default=None, max_length=100)
    reply: dict[str, object] | None = None
    language: str | None = Field(default=None, max_length=100)
    explanation: LongText
    assumptions: list[ShortText] = Field(default_factory=list, max_length=100)
    confidence: float = Field(ge=0, le=1)
    test_cases: list[SuggestedTestCase] = Field(default_factory=list, max_length=100)
    warnings: list[ShortText] = Field(default_factory=list, max_length=100)


class ReviewResult(StrictModel):
    is_acceptable: bool
    explanation: LongText
    issues: list[ShortText] = Field(default_factory=list, max_length=100)
    confidence: float = Field(ge=0, le=1)
    warnings: list[ShortText] = Field(default_factory=list, max_length=100)


class ErrorContext(StrictModel):
    task: TaskContext
    error_type: ShortText
    error_message: LongText
    candidate: SolutionCandidate | None = None
    feedback: str | None = Field(default=None, max_length=100_000)


class DiagnosticResult(StrictModel):
    category: ShortText
    summary: LongText
    likely_causes: list[ShortText] = Field(default_factory=list, max_length=100)
    recommended_actions: list[ShortText] = Field(default_factory=list, max_length=100)
    retryable: bool
    confidence: float = Field(ge=0, le=1)
    warnings: list[ShortText] = Field(default_factory=list, max_length=100)
