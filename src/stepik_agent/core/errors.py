class StepikAgentError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(StepikAgentError):
    """Raised when configuration is unsafe or incomplete."""


class InvalidStepikUrlError(StepikAgentError, ValueError):
    """Raised when a URL does not identify a supported Stepik target."""


class StepikApiError(StepikAgentError):
    """Raised when the Stepik API returns an invalid or failed response."""


class StepikAuthenticationError(StepikApiError):
    """Raised when Stepik rejects the OAuth token for an authenticated operation."""


class AmbiguousStepikWriteError(StepikApiError):
    """Raised when a write may have succeeded and therefore must not be retried automatically."""


class OAuthError(StepikAgentError):
    """Raised when OAuth authorization cannot be completed safely."""


class AIGatewayError(StepikAgentError):
    """Raised when a bounded AI operation cannot produce a valid result."""


class AIBudgetExceededError(AIGatewayError):
    """Base class for a bounded AI budget exhaustion."""


class AITaskBudgetExceededError(AIBudgetExceededError):
    """Raised when one task exhausted its AI budget but the course may continue."""


class AIRunBudgetExceededError(AIBudgetExceededError):
    """Raised when the shared run budget requires stopping the whole course."""


class UnsupportedTaskError(StepikAgentError):
    """Raised when no safe adapter can process a Stepik task."""


class CandidateValidationError(StepikAgentError):
    """Raised when a structured AI candidate violates local task rules."""


class AttemptLimitExceededError(StepikAgentError):
    """Raised before another Stepik attempt can exceed the configured task limit."""
