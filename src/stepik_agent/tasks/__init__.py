"""Task-type adapters and local candidate validation."""

from stepik_agent.tasks.analyzer import TaskAnalyzer
from stepik_agent.tasks.models import StepTaskData, TaskAnalysis
from stepik_agent.tasks.replies import ReplyAdapterRegistry

__all__ = ["ReplyAdapterRegistry", "StepTaskData", "TaskAnalysis", "TaskAnalyzer"]
