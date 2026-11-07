from enum import StrEnum


class CourseStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    UNAVAILABLE = "unavailable"


class TaskStatus(StrEnum):
    DISCOVERED = "discovered"
    READY = "ready"
    ANALYZING = "analyzing"
    SOLVING = "solving"
    LOCAL_TESTING = "local_testing"
    SUBMITTING = "submitting"
    WAITING_VERDICT = "waiting_verdict"
    CORRECT = "correct"
    RETRYING = "retrying"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    BLOCKED = "blocked"
    PAUSED = "paused"


class ExecutionMode(StrEnum):
    DRY_RUN = "dry_run"
    AUTO = "auto"
    SINGLE_TASK = "single_task"
    RESUME = "resume"


class RunState(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    STOPPED = "stopped"
    AUTH_REQUIRED = "auth_required"
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR = "error"


class AIRole(StrEnum):
    SOLVER = "solver"
    REVIEWER = "reviewer"
    REPAIRER = "repairer"
    DIAGNOSTIC = "diagnostic"


class AIRequestStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AttemptState(StrEnum):
    PLANNED = "planned"
    CREATING_ATTEMPT = "creating_attempt"
    ATTEMPT_CREATED = "attempt_created"
    CREATING_SUBMISSION = "creating_submission"
    WAITING_VERDICT = "waiting_verdict"
    CORRECT = "correct"
    WRONG = "wrong"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"


class EventSeverity(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
