"""Provider-independent AI gateway and structured response models."""

from stepik_agent.ai.factory import create_anymodel_gateway
from stepik_agent.ai.gateway import AnyModelGateway
from stepik_agent.ai.models import (
    DiagnosticResult,
    ErrorContext,
    ReviewResult,
    SolutionCandidate,
    SuggestedTestCase,
    TaskContext,
)

__all__ = [
    "AnyModelGateway",
    "DiagnosticResult",
    "ErrorContext",
    "ReviewResult",
    "SolutionCandidate",
    "SuggestedTestCase",
    "TaskContext",
    "create_anymodel_gateway",
]
