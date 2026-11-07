import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from stepik_agent.ai.gateway import AIRequestEvent
from stepik_agent.application.dry_run import DryRunResult


class DryRunReportStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def save(
        self,
        result: DryRunResult,
        events: list[AIRequestEvent],
        *,
        mode: str = "dry_run",
    ) -> Path:
        if mode not in {"dry_run", "ai_smoke"}:
            raise ValueError("unsupported report mode")
        self._directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        destination = self._directory / f"step-{result.step_id}-{timestamp}.json"
        payload = {
            "schema_version": 2,
            "created_at": datetime.now(UTC).isoformat(),
            "mode": mode,
            "step_id": result.step_id,
            "task_type": result.task_type,
            "candidate": result.candidate.model_dump(mode="json"),
            "warnings": list(result.warnings),
            "repair_attempts": result.repair_attempts,
            "reviewer_accepted": result.reviewer_accepted,
            "ai_requests": [
                {
                    "role": event.role.value,
                    "requested_model": event.requested_model,
                    "response_model": event.response_model,
                    "input_tokens": event.input_tokens,
                    "output_tokens": event.output_tokens,
                    "provider_cost_usd": (
                        str(event.provider_cost_usd)
                        if event.provider_cost_usd is not None
                        else None
                    ),
                    "latency_ms": event.latency_ms,
                    "succeeded": event.succeeded,
                    "error_code": event.error_code,
                }
                for event in events
            ],
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=self._directory,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.chmod(0o600)
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return destination
