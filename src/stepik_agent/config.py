from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, Field, PositiveInt, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from stepik_agent.core.errors import ConfigurationError


class Settings(BaseSettings):
    """Environment-backed settings with submission-safe defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anymodel_base_url: AnyHttpUrl = AnyHttpUrl("https://anymodel.org/v1")
    anymodel_api_key: SecretStr | None = None
    anymodel_model: str | None = None
    anymodel_review_model: str | None = None
    anymodel_fallback_model: str | None = None
    anymodel_max_output_tokens: PositiveInt = 4096
    anymodel_validation_retries: int = Field(default=1, ge=0, le=3)
    anymodel_json_mode: bool = True

    stepik_base_url: AnyHttpUrl = AnyHttpUrl("https://stepik.org")
    stepik_client_id: SecretStr | None = None
    stepik_client_secret: SecretStr | None = None
    stepik_redirect_uri: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8080/callback")
    stepik_token_path: Path = Path("data/stepik_token.json")
    oauth_callback_timeout_seconds: PositiveInt = 180

    database_url: str = "sqlite:///./data/agent.db"
    max_attempts_per_task: PositiveInt = 5
    max_total_attempts_per_task: PositiveInt = 25
    max_llm_calls_per_task: PositiveInt = 8
    max_llm_calls_per_run: PositiveInt = 500
    max_solution_repair_attempts: int = Field(default=2, ge=0, le=3)
    max_task_cost_usd: Decimal = Field(default=Decimal("1.00"), ge=0)
    max_run_cost_usd: Decimal = Field(default=Decimal("5.00"), ge=0)
    ai_request_timeout_seconds: PositiveInt = 60
    verdict_timeout_seconds: PositiveInt = 300

    dry_run: bool = True
    auto_submit: bool = False

    @field_validator(
        "anymodel_model",
        "anymodel_review_model",
        "anymodel_fallback_model",
        mode="before",
    )
    @classmethod
    def empty_model_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("anymodel_base_url", "stepik_base_url")
    @classmethod
    def service_urls_require_tls(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        parsed = urlsplit(str(value))
        is_loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and is_loopback):
            raise ValueError("service URL must use HTTPS unless it is a loopback address")
        return value

    @model_validator(mode="after")
    def validate_submission_mode(self) -> "Settings":
        if self.auto_submit and self.dry_run:
            raise ConfigurationError(
                "AUTO_SUBMIT=true conflicts with DRY_RUN=true; disable dry-run explicitly"
            )
        if self.max_task_cost_usd > self.max_run_cost_usd:
            raise ConfigurationError("MAX_TASK_COST_USD cannot exceed MAX_RUN_COST_USD")
        if self.max_attempts_per_task > self.max_total_attempts_per_task:
            raise ConfigurationError(
                "MAX_ATTEMPTS_PER_TASK cannot exceed MAX_TOTAL_ATTEMPTS_PER_TASK"
            )
        return self

    @property
    def stepik_oauth_configured(self) -> bool:
        return _secret_is_set(self.stepik_client_id) and _secret_is_set(self.stepik_client_secret)

    @property
    def ai_configured(self) -> bool:
        return _secret_is_set(self.anymodel_api_key) and bool(self.anymodel_model)


def _secret_is_set(value: SecretStr | None) -> bool:
    return value is not None and bool(value.get_secret_value().strip())
