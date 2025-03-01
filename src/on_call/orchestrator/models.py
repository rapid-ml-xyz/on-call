from typing import Any, Dict, Generic, List, Protocol, TypeVar, runtime_checkable, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

S = TypeVar('S')
T = TypeVar('T')


class NodeType(Enum):
    AGENT = "agent"
    FUNCTION = "function"


class RouteType(Enum):
    DYNAMIC = "dynamic"
    AGENTIC = "agentic"


@runtime_checkable
class Tool(Protocol):
    """Protocol for tools that can be used by agents."""
    name: str
    description: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool functionality."""
        ...


@runtime_checkable
class RoutingAgent(Protocol):
    """Protocol for agents that make routing decisions."""
    def decide_route(self, state: Any) -> str:
        """Decide which route to take based on the current state."""
        ...


RouteCondition = Union[Callable[[Any], Any], RoutingAgent]


@dataclass
class EdgeConfig:
    """Configuration for conditional edges between nodes."""
    route_type: RouteType
    condition: RouteCondition
    routes: Dict[Any, str]

    def __post_init__(self):
        if self.route_type == RouteType.AGENTIC and not isinstance(self.condition, RoutingAgent):
            raise ValueError("Agentic routing requires a RoutingAgent condition")

        if self.route_type == RouteType.DYNAMIC and not callable(self.condition):
            raise ValueError("Dynamic routing requires a callable condition")


class WorkflowState(Generic[S]):
    """Generic container for workflow state."""
    def __init__(self, state: S):
        self._state = state

    @property
    def state(self) -> S:
        return self._state


class Node(ABC, Generic[S]):
    """Base class for all nodes in the workflow."""
    @abstractmethod
    def invoke(self, state: WorkflowState[S]) -> WorkflowState[S]:
        pass


class AgentNode(Node[S]):
    """Node that uses an agent to process state."""
    def __init__(self, agent: 'Agent[S]'):
        self.agent = agent

    def invoke(self, state: WorkflowState[S]) -> WorkflowState[S]:
        return self.agent.invoke(state)


class FunctionNode(Node[S]):
    """Node that uses a pure Python function to process state."""
    def __init__(self, func: Callable[[WorkflowState[S]], WorkflowState[S]]):
        self.func = func

    def invoke(self, state: WorkflowState[S]) -> WorkflowState[S]:
        return self.func(state)


@runtime_checkable
class Agent(Protocol[S]):
    """Protocol for agents that can be used in workflow nodes."""
    def invoke(self, state: WorkflowState[S]) -> WorkflowState[S]:
        ...


class NodeConfig(Generic[S, T]):
    """Configuration and runtime representation of a workflow node."""
    def __init__(
        self,
        name: str,
        node_type: NodeType = NodeType.AGENT,
        allowed_tools: List[str] | None = None,
        agent_config: Dict[str, Any] | None = None,
        agent: Agent[S] | None = None,
        function: Callable[[WorkflowState[S]], WorkflowState[S]] | None = None
    ):
        self.name = name
        self.node_type = node_type
        self.agent_config = agent_config
        self.agent = agent
        self.function = function
        self.allowed_tools = set(allowed_tools) if allowed_tools else set()

    @property
    def is_configured(self) -> bool:
        return (self.node_type == NodeType.AGENT and self.agent is not None) or \
               (self.node_type == NodeType.FUNCTION and self.function is not None)

    def create_node(self) -> Node[S]:
        match self.node_type:
            case NodeType.AGENT:
                if not self.agent:
                    raise ValueError("Agent not configured")
                return AgentNode(self.agent)
            case NodeType.FUNCTION:
                if not self.function:
                    raise ValueError("Function not configured")
                return FunctionNode(self.function)
