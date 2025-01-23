from dotenv import load_dotenv
from typing import Dict, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .logger import logging
from .notebook_manager import NotebookManager
from .orchestrator.engines import LangGraphOrchestrator, LangGraphToolWrapper, LangGraphMessageState
from .orchestrator import NodeConfig, NodeType, WorkflowState

load_dotenv()


def process_data(state: WorkflowState[LangGraphMessageState]) -> WorkflowState[LangGraphMessageState]:
    messages = state.state["messages"]
    last_content = messages[-1].content

    # TODO: Create clean integration
    notebook_manager = NotebookManager()
    code = last_content
    cell_id = notebook_manager.add_cell(code, "code")
    notebook_manager.execute_cell(cell_id)
    cell = notebook_manager.notebook.cells[0]
    output = cell.outputs[0].text

    messages.append(HumanMessage(content=output))
    return WorkflowState({"messages": messages})


def setup_workflow() -> LangGraphOrchestrator:
    orchestrator = LangGraphOrchestrator()

    llm = ChatOpenAI(model="gpt-4o-mini")
    researcher_node = NodeConfig[LangGraphMessageState, LangGraphToolWrapper](
        name="researcher",
        next_node="processor",
        node_type=NodeType.AGENT,
        allowed_tools=[],
        agent_config={
            "type": "react",
            "llm": llm,
            "system_message": "You are a software engineer. Write a working hello-world program in Python. "
                              "Only give me the code, nothing else. Not even ```python"
        }
    )

    processor_node = NodeConfig[LangGraphMessageState, LangGraphToolWrapper](
        name="processor",
        next_node=None,
        node_type=NodeType.FUNCTION,
        function=process_data
    )

    nodes = [researcher_node, processor_node]
    logging.info(f"\nConfiguring nodes: {[n.name for n in nodes]}")
    orchestrator.configure_nodes(nodes)
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

