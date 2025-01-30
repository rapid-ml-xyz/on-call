from typing import Any, Dict, List
from langchain.agents import ZeroShotAgent
from langchain.agents import AgentExecutor
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent
from src.on_call.logger import logging
from ..base import BaseOrchestrator
from ..models import Agent, EdgeConfig, NodeConfig, NodeType, RouteType, Tool, WorkflowState

LangGraphMessageState = Dict[str, Any]


class LangGraphToolWrapper:
    def __init__(self, tool: BaseTool):
        self.tool = tool
        self.name = tool.name
        self.description = tool.description

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        logging.info(f"Invoking tool: {self.name}")
        result = self.tool.invoke(*args, **kwargs)
        return result


class LangGraphAgentFactory:
    @staticmethod
    def create_react_agent(tools: List[Tool], agent_config: Dict[str, Any]) -> Agent[LangGraphMessageState]:
        logging.info("Creating react agent")
        llm = agent_config["llm"]
        langchain_tools = [
            tool.tool if isinstance(tool, LangGraphToolWrapper) else tool
            for tool in tools
        ]
        system_message = agent_config["system_message"]
        agent = create_react_agent(llm, langchain_tools, state_modifier=system_message)

        class WrappedReActAgent:
            def invoke(self, state: WorkflowState[LangGraphMessageState]) -> WorkflowState[LangGraphMessageState]:
                result = agent.invoke(state.state)
                return WorkflowState(result)

        return WrappedReActAgent()

    @staticmethod
    def create_simple_agent(tools: List[Tool], agent_config: Dict[str, Any]) -> Agent[LangGraphMessageState]:
        logging.info("Creating simple agent")
        llm = agent_config["llm"]
        langchain_tools = [
            tool.tool if isinstance(tool, LangGraphToolWrapper) else tool
            for tool in tools
        ]
        system_message = agent_config["system_message"]
        agent = ZeroShotAgent.from_llm_and_tools(
            llm=llm,
            tools=langchain_tools,
            prefix=system_message
        )
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=langchain_tools,
            verbose=True
        )

        class WrappedSimpleAgent:
            def invoke(self, state: WorkflowState[LangGraphMessageState]) -> WorkflowState[LangGraphMessageState]:
                input_text = state.state.get("input", "")
                result = agent_executor.invoke({"input": input_text})
                state.state["output"] = str(result["output"])
                return WorkflowState(state.state)

        return WrappedSimpleAgent()


class LangGraphOrchestrator(BaseOrchestrator[LangGraphMessageState, LangGraphToolWrapper]):
    def __init__(self):
        super().__init__()
        self.agent_factory = LangGraphAgentFactory()
        self._graph = StateGraph(LangGraphMessageState)

    def create_agent(self, tools: List[LangGraphToolWrapper], agent_config: Dict[str, Any]) \
            -> Agent[LangGraphMessageState]:
        if agent_config.get("type", "react") == "simple":
            return self.agent_factory.create_simple_agent(tools, agent_config)
        return self.agent_factory.create_react_agent(tools, agent_config)

    def _add_node_to_graph(self, node: NodeConfig[LangGraphMessageState, LangGraphToolWrapper]) -> None:
        if not node.is_configured:
            raise ValueError(f"Node {node.name} must be configured before adding to graph")

        workflow_node = node.create_node()

        def node_fn(state: LangGraphMessageState) -> LangGraphMessageState:
            if node.node_type == NodeType.AGENT and "messages" not in state:
                raise ValueError(f"Agent node {node.name} requires 'messages' in state")

            workflow_state = WorkflowState(state)
            logging.info(f"Executing node: {node.name}")
            result = workflow_node.invoke(workflow_state)
            return result.state

        self._graph.add_node(node.name, node_fn)

    def _add_edge_to_graph(self, source: str, edge: EdgeConfig) -> None:
        if edge.route_type == RouteType.AGENTIC:
            def route_fn(state):
                return edge.condition.decide_route(state)
        else:
            route_fn = edge.condition

        self._graph.add_conditional_edges(source, route_fn, edge.routes)

    def set_entry_point(self, node_name: str) -> None:
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found")
        self.entry_point = node_name
        self._graph.add_edge(START, node_name)

    def run(self, initial_state: LangGraphMessageState) -> LangGraphMessageState:
        if not self.entry_point:
            raise ValueError("Entry point not set")

        compiled_graph = self._graph.compile()
        result = compiled_graph.invoke(initial_state)
        return result

    def visualize_graph(self) -> None:
        self._graph.compile().get_graph().print_ascii()
