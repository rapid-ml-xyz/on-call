from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Generic

from .types import S, C, T, WorkflowState
from .models import Agent, Edge


class BaseOrchestrator(ABC, Generic[S, C, T]):
    def __init__(self, config: Optional[C] = None):
        self.agents: Dict[str, Agent[T]] = {}
        self.workflow_graph: Dict[str, List[Edge]] = {}
        self.config: Optional[C] = config
        self.tool_executor: Optional[Any] = None
        self._entry_point: Optional[str] = None
        self._is_initialized: bool = False

    @abstractmethod
    def add_agent(self, agent: Agent[T]) -> None:
        """Add an agent to the orchestrator."""
        pass

    @abstractmethod
    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the orchestrator."""
        pass

    @abstractmethod
    def connect(self,
                from_agent: str,
                to_agent: str,
                condition: Optional[callable] = None,
                edge_config: Optional[Dict[str, Any]] = None) -> None:
        """Define workflow connections between agents."""
        pass

    @abstractmethod
    def set_entry_point(self, agent_name: str) -> None:
        """Set the starting agent for the workflow."""
        pass

    @abstractmethod
    async def run(self, initial_state: S) -> WorkflowState[S]:
        """Execute the workflow with given input state."""
        pass

    @abstractmethod
    def get_trace(self) -> List[Dict[str, Any]]:
        """Return the execution trace/history."""
        pass

    def validate_workflow(self) -> bool:
        """Validate the workflow configuration."""
        if not self.agents:
            raise ValueError("No agents defined in the workflow")

        if not self._entry_point:
            raise ValueError("No entry point defined")

        for from_agent, edges in self.workflow_graph.items():
            if from_agent not in self.agents:
                raise ValueError(f"Agent {from_agent} not found")
            for edge in edges:
                if edge.to_agent not in self.agents:
                    raise ValueError(f"Agent {edge.to_agent} not found")

        return True
