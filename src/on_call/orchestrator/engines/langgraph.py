from typing import Dict, List, Optional, Any, cast
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor

from ..base import BaseOrchestrator
from ..types import S, C, T, WorkflowState, NodeFunction
from ..models import Agent, Edge
from ..utils.validation import validate_agent_config
from ..utils.errors import (
    AgentNotFoundError,
    DuplicateAgentError,
    WorkflowValidationError,
    GraphBuildError
)


class LangGraphOrchestrator(BaseOrchestrator[S, C, T]):
    def __init__(self, config: Optional[C] = None):
        """Initialize LangGraph orchestrator with optional configuration."""
        super().__init__(config)
        self.state_graph: Optional[StateGraph] = None
        self.graph: Optional[Graph] = None
        self._node_functions: Dict[str, NodeFunction] = {}

    def add_agent(self, agent: Agent[T]) -> None:
        """Add an agent to the workflow."""
        validate_agent_config(agent)

        if agent.name in self.agents:
            raise DuplicateAgentError(f"Agent {agent.name} already exists")

        self.agents[agent.name] = agent
        if agent.tools:
            if not self.tool_executor:
                self.tool_executor = ToolExecutor(agent.tools)
            else:
                for tool in agent.tools:
                    self.tool_executor.add_tool(tool)

        self._is_initialized = False

    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the workflow."""
        if agent_name not in self.agents:
            raise AgentNotFoundError(f"Agent {agent_name} not found")

        agent = self.agents[agent_name]

        if agent.tools and self.tool_executor:
            for tool in agent.tools:
                self.tool_executor.remove_tool(tool)

        if agent_name in self.workflow_graph:
            del self.workflow_graph[agent_name]

        for _, edges in self.workflow_graph.items():
            edges[:] = [e for e in edges if e.to_agent != agent_name]

        del self.agents[agent_name]

        self._is_initialized = False

    def connect(self,
                from_agent: str,
                to_agent: str,
                condition: Optional[callable] = None,
                edge_config: Optional[Dict[str, Any]] = None) -> None:
        """Connect two agents in the workflow."""
        if from_agent not in self.agents:
            raise AgentNotFoundError(f"Source agent {from_agent} not found")
        if to_agent not in self.agents:
            raise AgentNotFoundError(f"Target agent {to_agent} not found")

        if from_agent not in self.workflow_graph:
            self.workflow_graph[from_agent] = []

        edge = Edge(
            from_agent=from_agent,
            to_agent=to_agent,
            condition=condition,
            config=edge_config or {}
        )
        self.workflow_graph[from_agent].append(edge)

        self._is_initialized = False

    def set_entry_point(self, agent_name: str) -> None:
        """Set the workflow entry point."""
        if agent_name not in self.agents:
            raise AgentNotFoundError(f"Agent {agent_name} not found")
        self._entry_point = agent_name

        self._is_initialized = False

    async def run(self, initial_state: S) -> WorkflowState[S]:
        """Execute the workflow with given initial state."""
        if not self._is_initialized:
            try:
                self.validate_workflow()
                self._build_graph()
                self._is_initialized = True
            except Exception as e:
                raise WorkflowValidationError(f"Workflow validation failed: {str(e)}")

        workflow_state = WorkflowState(initial_state)

        if not self.graph:
            raise GraphBuildError("Graph not properly initialized")

        try:
            result = await self.graph.ainvoke({
                "state": workflow_state.state,
                "tools": self.tool_executor if self.tool_executor else None
            })
            workflow_state.state = cast(S, result["state"])
            if "messages" in result:
                workflow_state.messages.extend(result["messages"])
            if "artifacts" in result:
                workflow_state.artifacts.update(result["artifacts"])

        except Exception as e:
            workflow_state.messages.append({
                "type": "error",
                "content": str(e)
            })
            raise

        return workflow_state

    def get_trace(self) -> List[Dict[str, Any]]:
        """Get execution trace history."""
        if self.graph:
            return self.graph.get_trace()
        return []

    def _build_graph(self) -> None:
        """Build the LangGraph graph from workflow configuration."""
        try:
            self.state_graph = StateGraph()

            for agent_name, agent in self.agents.items():
                node_func = self._create_agent_node(agent)
                self._node_functions[agent_name] = node_func
                self.state_graph.add_node(agent_name, node_func)

            for from_agent, edges in self.workflow_graph.items():
                for edge in edges:
                    self.state_graph.add_edge(
                        from_agent,
                        edge.to_agent,
                    )

            if self._entry_point:
                self.state_graph.set_entry_point(self._entry_point)

            self.graph = self.state_graph.compile()

        except Exception as e:
            raise GraphBuildError(f"Failed to build graph: {str(e)}")

    def _create_agent_node(self, agent: Agent[T]) -> NodeFunction:
        """Create a node function for the agent."""
        async def node_func(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                current_state = state.get("state", {})
                tools = state.get("tools")

                context = {
                    "agent": agent,
                    "tools": tools,
                    "state": current_state
                }

                if agent.tools and tools:
                    for tool in agent.tools:
                        try:
                            result = await tools.arun(tool, current_state)
                            current_state.update(result)
                        except Exception as e:
                            current_state["errors"] = current_state.get("errors", [])
                            current_state["errors"].append({
                                "tool": tool.__class__.__name__,
                                "error": str(e)
                            })

                if hasattr(agent, "process"):
                    current_state = await agent.process(context)

                return {
                    "state": current_state,
                    "messages": state.get("messages", []),
                    "artifacts": state.get("artifacts", {})
                }

            except Exception as e:
                return {
                    "state": state.get("state", {}),
                    "messages": state.get("messages", []) + [{
                        "type": "error",
                        "content": f"Error in agent {agent.name}: {str(e)}"
                    }],
                    "artifacts": state.get("artifacts", {})
                }

        return node_func
