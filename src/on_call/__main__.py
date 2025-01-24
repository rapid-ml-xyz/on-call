from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage
from workflow.setup import setup_analysis_workflow


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
