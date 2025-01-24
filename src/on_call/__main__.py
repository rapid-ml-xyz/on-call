from dotenv import load_dotenv
from typing import Dict, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .logger import logging
from .notebook_manager import NotebookManager
from .orchestrator.engines import LangGraphOrchestrator, LangGraphToolWrapper, LangGraphMessageState
from .orchestrator import EdgeConfig, NodeConfig, NodeType, RouteType, WorkflowState

load_dotenv()


def process_data(state: WorkflowState[LangGraphMessageState]) -> WorkflowState[LangGraphMessageState]:
    messages = state.state["messages"]
    last_content = messages[-1].content

    try:
        notebook_manager = NotebookManager()
        code = last_content
        cell_id = notebook_manager.add_cell(code, "code")
        notebook_manager.execute_cell(cell_id)
        cell = notebook_manager.notebook.cells[0]

        if hasattr(cell.outputs[0], 'traceback'):
            raise Exception(cell.outputs[0].traceback)

        output = cell.outputs[0].text
        messages.append(HumanMessage(content=output))

    except Exception as e:
        messages.append(HumanMessage(
            content=f"Error executing code: {str(e)}",
            metadata={"error": True}
        ))

    return WorkflowState({"messages": messages})


def handle_error(state: WorkflowState[LangGraphMessageState]) -> WorkflowState[LangGraphMessageState]:
    messages = state.state["messages"]
    error_msg = messages[-1].content

    # Add error handling message
    messages.append(HumanMessage(
        content=f"Handling error: {error_msg}. Requesting code revision."
    ))
    return WorkflowState({"messages": messages})


def route_output(state: Dict[str, List[BaseMessage]]) -> str:
    """Route based on the content of the last message."""
    last_message = state["messages"][-1]

    if hasattr(last_message, 'metadata') and last_message.metadata.get('error'):
        return "error_handler"

    if "Error executing code" in last_message.content:
        return "error_handler"

    return "processor"


def setup_workflow() -> LangGraphOrchestrator:
    orchestrator = LangGraphOrchestrator()

    llm = ChatOpenAI(model="gpt-4o-mini")
    researcher_node = NodeConfig[LangGraphMessageState, LangGraphToolWrapper](
        name="researcher",
        node_type=NodeType.AGENT,
        allowed_tools=[],
        agent_config={
            "type": "react",
            "llm": llm,
            "system_message": "You are a software engineer. Write a working hello-world program in Python. "
                              "Only give me the code, nothing else. Not even ```python. "
                              "If you receive an error message, fix the code and try again."
        }
    )

    processor_node = NodeConfig[LangGraphMessageState, LangGraphToolWrapper](
        name="processor",
        node_type=NodeType.FUNCTION,
        function=process_data
    )

    error_handler_node = NodeConfig[LangGraphMessageState, LangGraphToolWrapper](
        name="error_handler",
        node_type=NodeType.FUNCTION,
        function=handle_error
    )

    edge = EdgeConfig(
        route_type=RouteType.DYNAMIC,
        condition=route_output,
        routes={
            "processor": "processor",
            "error_handler": "error_handler"
        }
    )

    orchestrator.configure_nodes([researcher_node, processor_node, error_handler_node])
    orchestrator.add_edge("researcher", edge)
    orchestrator.set_entry_point("researcher")

    return orchestrator


def run_workflow(question: str) -> Dict[str, List[BaseMessage]]:
    logging.info(f"\nStarting workflow with question: {question}")
    messages = [HumanMessage(content=question)]

    workflow = setup_workflow()
    workflow.visualize_graph()
    return workflow.run(messages)


if __name__ == "__main__":
    _question = "Write a hello-world python program."
    response = run_workflow(_question)
    logging.info(f"Final Result: {response}")
