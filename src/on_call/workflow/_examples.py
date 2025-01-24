# Building a tree with one agentic & one deterministic node

from dotenv import load_dotenv
from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage
from ..orchestrator.engines import LangGraphOrchestrator, LangGraphToolWrapper, LangGraphMessageState
from ..orchestrator import EdgeConfig, NodeConfig, NodeType, RouteType
from .enums import Step
from .models import LLMFactory

load_dotenv()


def setup_analysis_workflow() -> LangGraphOrchestrator:
    orchestrator = LangGraphOrchestrator()
    open_ai_llm = LLMFactory.open_ai_llm()

    researcher_node = NodeConfig[LangGraphMessageState, LangGraphToolWrapper](
        name="researcher",
        node_type=NodeType.AGENT,
        allowed_tools=[],
        agent_config={
            "type": "react",
            "llm": open_ai_llm,
            "system_message": "You are a software engineer. Solve the problem you are provided."
        }
    )

    impact_window_node = NodeConfig[LangGraphMessageState, LangGraphToolWrapper](
        name=Step.IMPACT_WINDOW.name,
        node_type=NodeType.FUNCTION,
        function=lambda x: x
    )

    orchestrator.configure_nodes([researcher_node, impact_window_node])
    orchestrator.add_edge("researcher", EdgeConfig(
        route_type=RouteType.DYNAMIC,
        condition=lambda state: Step.IMPACT_WINDOW.name,
        routes={Step.IMPACT_WINDOW.name: Step.IMPACT_WINDOW.name}
    ))
    orchestrator.set_entry_point("researcher")

    return orchestrator


def run_analysis(initial_data: str) -> Dict[str, List[BaseMessage]]:
    """Run the performance analysis workflow"""
    messages = [HumanMessage(content=initial_data)]
    workflow = setup_analysis_workflow()
    workflow.visualize_graph()
    return workflow.run(messages)


if __name__ == "__main__":
    data = "Performance metrics show a sudden drop in model accuracy last week"
    response = run_analysis(data)
    print(f"Analysis Results: {response}")
