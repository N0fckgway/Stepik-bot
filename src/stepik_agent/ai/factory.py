"""Construct the AnyModel adapter from validated application settings."""

from openai import AsyncOpenAI

from stepik_agent.ai.budget import BudgetedAITransport, SharedAIBudget
from stepik_agent.ai.gateway import AnyModelGateway, RequestObserver
from stepik_agent.ai.transport import OpenAIChatTransport
from stepik_agent.config import Settings
from stepik_agent.core.errors import ConfigurationError


def create_anymodel_gateway(
    settings: Settings,
    *,
    request_observer: RequestObserver | None = None,
    shared_budget: SharedAIBudget | None = None,
) -> AnyModelGateway:
    if not settings.ai_configured:
        raise ConfigurationError("AnyModel requires ANYMODEL_API_KEY and ANYMODEL_MODEL")
    if settings.anymodel_api_key is None or settings.anymodel_model is None:
        raise ConfigurationError("AnyModel configuration is incomplete")

    client = AsyncOpenAI(
        api_key=settings.anymodel_api_key.get_secret_value(),
        base_url=str(settings.anymodel_base_url).rstrip("/"),
        timeout=float(settings.ai_request_timeout_seconds),
        # SDK retries are disabled so every potentially billable call is made by the
        # gateway's visible, bounded policy.
        max_retries=0,
    )
    transport = BudgetedAITransport(
        OpenAIChatTransport(client),
        max_calls=settings.max_llm_calls_per_task,
        max_cost_usd=settings.max_task_cost_usd,
        shared_budget=shared_budget,
    )
    return AnyModelGateway(
        transport,
        primary_model=settings.anymodel_model,
        review_model=settings.anymodel_review_model,
        fallback_model=settings.anymodel_fallback_model,
        max_output_tokens=settings.anymodel_max_output_tokens,
        validation_retries=settings.anymodel_validation_retries,
        json_mode=settings.anymodel_json_mode,
        request_observer=request_observer,
    )
