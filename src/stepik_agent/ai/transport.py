"""OpenAI SDK adapter kept outside domain-level gateway logic."""

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONObject


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    model: str
    messages: list[dict[str, str]]
    max_output_tokens: int
    json_mode: bool


@dataclass(frozen=True, slots=True)
class CompletionResult:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    provider_cost_usd: Decimal | None = None


class AITransport(Protocol):
    async def complete(self, request: CompletionRequest) -> CompletionResult: ...

    async def aclose(self) -> None: ...


class OpenAIChatTransport:
    """Map the provider-neutral request onto OpenAI-compatible chat completions."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def aclose(self) -> None:
        await self._client.close()

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        messages = cast(Iterable[ChatCompletionMessageParam], request.messages)
        if request.json_mode:
            response = await self._client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_output_tokens,
                response_format=ResponseFormatJSONObject(type="json_object"),
                temperature=0,
            )
        else:
            response = await self._client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_output_tokens,
                temperature=0,
            )
        if not response.choices:
            raise ValueError("AI provider returned no choices")
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("AI provider returned empty content")

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage is not None else 0
        output_tokens = usage.completion_tokens if usage is not None else 0
        return CompletionResult(
            content=content,
            model=response.model or request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_cost_usd=_read_provider_cost(usage),
        )


def _read_provider_cost(usage: object | None) -> Decimal | None:
    """Use a provider extension when present; never invent an exact cost."""
    if usage is None:
        return None
    raw_cost = getattr(usage, "cost", None)
    if raw_cost is None:
        extra = getattr(usage, "model_extra", None)
        if isinstance(extra, dict):
            raw_cost = extra.get("cost")
    if isinstance(raw_cost, bool) or raw_cost is None:
        return None
    try:
        cost = Decimal(str(raw_cost))
    except (InvalidOperation, ValueError):
        return None
    return cost if cost.is_finite() and cost >= 0 else None
