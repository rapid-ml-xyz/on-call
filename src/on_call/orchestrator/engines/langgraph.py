from typing import Dict, List, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from src.on_call.logger import logging
from ..base import BaseOrchestrator
from ..models import Agent, NodeConfig, NodeType, Tool, WorkflowState

LangGraphMessageState = Dict[str, List[BaseMessage]]


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
        from langgraph.prebuilt import create_react_agent

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
        """Create a simple agent that executes one tool call."""
        from langchain.agents import ZeroShotAgent
        from langchain.agents import AgentExecutor

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
                messages = state.state["messages"]
                last_message = messages[-1].content
                result = agent_executor.invoke({"input": last_message})
                messages.append(HumanMessage(content=str(result["output"])))
                return WorkflowState({"messages": messages})

        return WrappedSimpleAgent()


class LangGraphOrchestrator(BaseOrchestrator[LangGraphMessageState, LangGraphToolWrapper]):
    def __init__(self):
        super().__init__()
        from langgraph.graph import StateGraph, MessagesState
        self.agent_factory = LangGraphAgentFactory()
        self._graph = StateGraph(MessagesState)

    def configure_nodes(self, nodes: List[NodeConfig]) -> None:
        from langgraph.graph import END

        for node in nodes:
            if node.node_type == NodeType.AGENT:
                node_tools = [self.tools[tool_name] for tool_name in node.allowed_tools]
                node.agent = self.create_agent(node_tools, node.agent_config)

            self.nodes[node.name] = node
            self._add_node_to_graph(node)

            if node.next_node is None:
                self._graph.add_edge(node.name, END)
            else:
                self._graph.add_edge(node.name, node.next_node)

    def create_agent(self, tools: List[LangGraphToolWrapper], agent_config: Dict[str, Any]) \
            -> Agent[LangGraphMessageState]:
        if agent_config.get("type", "react") == "simple":
            return self.agent_factory.create_simple_agent(tools, agent_config)
        return self.agent_factory.create_react_agent(tools, agent_config)

    def _add_node_to_graph(self, node: NodeConfig[LangGraphMessageState, LangGraphToolWrapper]) -> None:
        if not node.is_configured:
            raise ValueError(f"Node {node.name} must be configured before adding to graph")

        workflow_node = node.create_node()

        def node_fn(state: Dict[str, List[BaseMessage]]) -> Dict[str, Any]:
            workflow_state = WorkflowState(state)

            logging.info(f"Executing node: {node.name}")
            result = workflow_node.invoke(workflow_state)
            goto = node.next_node

            if node.node_type == NodeType.AGENT and result.state["messages"]:
                last_message = result.state["messages"][-1].content
                result.state["messages"][-1] = HumanMessage(
                    content=last_message,
                    name=node.name
                )

            return {
                "messages": result.state["messages"],
                "__goto__": goto
            }

        self._graph.add_node(node.name, node_fn)

    def set_entry_point(self, node_name: str) -> None:
        from langgraph.graph import START
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found")
        self.entry_point = node_name
        self._graph.add_edge(START, node_name)

    def run(self, messages: List[BaseMessage]) -> LangGraphMessageState:
        if not self.entry_point:
            raise ValueError("Entry point not set")

        compiled_graph = self._graph.compile()
        initial_state = {"messages": messages}
        result = compiled_graph.invoke(initial_state)
        return result

    def visualize_graph(self) -> None:
        self._graph.compile().get_graph().print_ascii()
