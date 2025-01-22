from ..models import Agent
from .errors import WorkflowValidationError


def validate_agent_config(agent: Agent) -> None:
    """Validate agent configuration"""
    if not agent.name:
        raise WorkflowValidationError("Agent name is required")
    if not agent.role:
        raise WorkflowValidationError("Agent role is required")
