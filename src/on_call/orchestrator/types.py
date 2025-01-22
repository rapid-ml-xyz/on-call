from typing import Any, Dict, Generic, List, Protocol, TypeVar, runtime_checkable


S = TypeVar('S')
C = TypeVar('C')
T = TypeVar('T')

NodeFunction = callable([[Dict[str, Any]], Dict[str, Any]])


@runtime_checkable
class Tool(Protocol):
    """Protocol for tool interface"""
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


class WorkflowState(Generic[S]):
    """Generic container for workflow state"""
    def __init__(self, initial_state: S):
        self.state: S = initial_state
        self.messages: List[Dict[str, Any]] = []
        self.artifacts: Dict[str, Any] = {}
