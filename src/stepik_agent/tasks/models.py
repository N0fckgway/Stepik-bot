from collections.abc import Mapping
from dataclasses import dataclass

from stepik_agent.ai.models import TaskContext


@dataclass(frozen=True, slots=True)
class StepTaskData:
    step_id: int
    block_name: str
    statement_html: str
    options: Mapping[str, object]
    preceding_materials_html: tuple[str, ...] = ()
    dataset_text: str | None = None


@dataclass(frozen=True, slots=True)
class TaskAnalysis:
    context: TaskContext
    allowed_languages: tuple[str, ...] = ()
    starter_code: str | None = None
    assembler_syntax: str | None = None
    blank_options: tuple[tuple[str, ...] | None, ...] = ()
    table_rows: tuple[str, ...] = ()
    table_columns: tuple[str, ...] = ()
    table_is_checkbox: bool = False
    warnings: tuple[str, ...] = ()
