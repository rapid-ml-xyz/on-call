class OrchestratorError(Exception):
    """Base class for orchestrator errors"""
    pass


class AgentNotFoundError(OrchestratorError):
    """Raised when agent is not found"""
    pass


class DuplicateAgentError(OrchestratorError):
    """Raised when adding duplicate agent"""
    pass


class WorkflowValidationError(OrchestratorError):
    """Raised when workflow validation fails"""
    pass


class GraphBuildError(OrchestratorError):
    """Raised when graph construction fails"""
    pass
