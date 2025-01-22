from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any, Generic
from .types import T, Tool


class AgentRole(Enum):
    COORDINATOR = "coordinator"
    WORKER = "worker"
    REVIEWER = "reviewer"
    CUSTOM = "custom"


@dataclass
class Agent(Generic[T]):
    name: str
    role: AgentRole
    description: str
    tools: Optional[List[T]] = None
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.tools:
            for tool in self.tools:
                if not isinstance(tool, Tool):
                    raise ValueError(f"Tool {tool} must implement the Tool protocol")


@dataclass
class Edge:
    """Represents a connection between agents"""
    from_agent: str
    to_agent: str
    condition: Optional[callable] = None
    config: Dict[str, Any] = field(default_factory=dict)
