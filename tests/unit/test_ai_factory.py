import asyncio

import pytest

from stepik_agent.ai.factory import create_anymodel_gateway
from stepik_agent.ai.gateway import AnyModelGateway
from stepik_agent.config import Settings
from stepik_agent.core.errors import ConfigurationError


def test_factory_rejects_missing_credentials() -> None:
    with pytest.raises(ConfigurationError, match="ANYMODEL_API_KEY"):
        create_anymodel_gateway(Settings(_env_file=None))


def test_factory_builds_gateway_without_network_call() -> None:
    settings = Settings(
        _env_file=None,
        anymodel_api_key="test-key",
        anymodel_model="test-model",
        anymodel_fallback_model="backup-model",
    )

    gateway = create_anymodel_gateway(settings)

    assert isinstance(gateway, AnyModelGateway)
    asyncio.run(gateway.aclose())
