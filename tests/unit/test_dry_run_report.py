import json
import stat

from stepik_agent.ai.models import SolutionCandidate
from stepik_agent.application.dry_run import DryRunResult
from stepik_agent.reporting.dry_run import DryRunReportStore


def test_report_is_private_and_contains_no_unrequested_context(tmp_path) -> None:
    result = DryRunResult(
        step_id=42,
        task_type="string",
        candidate=SolutionCandidate(
            task_type="string",
            answer="answer",
            explanation="explanation",
            confidence=0.8,
        ),
        warnings=(),
    )

    path = DryRunReportStore(tmp_path).save(result, [])
    content = path.read_text(encoding="utf-8")
    payload = json.loads(content)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert payload["schema_version"] == 2
    assert payload["step_id"] == 42
    assert payload["repair_attempts"] == 0
    assert payload["reviewer_accepted"] is False
    assert "task_context" not in payload
    assert "api_key" not in content


def test_ai_smoke_report_has_distinct_mode(tmp_path) -> None:
    result = DryRunResult(
        step_id=1,
        task_type="number",
        candidate=SolutionCandidate(
            task_type="number",
            answer="4",
            explanation="Synthetic result",
            confidence=1,
        ),
        warnings=(),
    )

    path = DryRunReportStore(tmp_path).save(result, [], mode="ai_smoke")

    assert json.loads(path.read_text(encoding="utf-8"))["mode"] == "ai_smoke"
