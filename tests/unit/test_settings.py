import pytest

pytest.importorskip("pydantic_settings")

from pydantic import ValidationError

from stepik_agent.config import Settings
from stepik_agent.core.errors import ConfigurationError


def test_defaults_disable_submissions() -> None:
    settings = Settings(_env_file=None)

    assert settings.dry_run is True
    assert settings.auto_submit is False
    assert settings.max_solution_repair_attempts == 2
    assert settings.max_total_attempts_per_task == 25


def test_conflicting_submission_flags_are_rejected() -> None:
    with pytest.raises((ConfigurationError, ValidationError)):
        Settings(_env_file=None, dry_run=True, auto_submit=True)


def test_empty_ai_credentials_are_not_configured() -> None:
    settings = Settings(
        _env_file=None,
        anymodel_api_key="",
        anymodel_model="",
        anymodel_review_model="",
    )

    assert settings.ai_configured is False
    assert settings.anymodel_model is None
    assert settings.anymodel_review_model is None


def test_task_budget_cannot_exceed_run_budget() -> None:
    with pytest.raises((ConfigurationError, ValidationError)):
        Settings(_env_file=None, max_task_cost_usd="2", max_run_cost_usd="1")


def test_per_run_attempt_limit_cannot_exceed_total_limit() -> None:
    with pytest.raises((ConfigurationError, ValidationError)):
        Settings(
            _env_file=None,
            max_attempts_per_task=6,
            max_total_attempts_per_task=5,
        )


def test_remote_service_urls_require_https() -> None:
    with pytest.raises(ValidationError, match="HTTPS"):
        Settings(_env_file=None, stepik_base_url="http://example.com")

    settings = Settings(_env_file=None, anymodel_base_url="http://127.0.0.1:9000/v1")

    assert str(settings.anymodel_base_url).startswith("http://127.0.0.1:9000")
