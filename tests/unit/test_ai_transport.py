import asyncio
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from stepik_agent.ai.transport import CompletionRequest, OpenAIChatTransport


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs = kwargs
        usage = SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=7,
            cost="0.0025",
            model_extra=None,
        )
        message = SimpleNamespace(content='{"answer":"ok"}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=usage,
            model="provider-model",
        )


def test_openai_transport_maps_json_request_and_usage() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    transport = OpenAIChatTransport(client)  # type: ignore[arg-type]

    result = asyncio.run(
        transport.complete(
            CompletionRequest(
                model="configured-model",
                messages=[{"role": "user", "content": "Return JSON"}],
                max_output_tokens=321,
                json_mode=True,
            )
        )
    )

    assert completions.kwargs["model"] == "configured-model"
    assert completions.kwargs["max_tokens"] == 321
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert result.model == "provider-model"
    assert result.input_tokens == 12
    assert result.output_tokens == 7
    assert result.provider_cost_usd == Decimal("0.0025")
