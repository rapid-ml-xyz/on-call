from typing import TypedDict, List
from ..types import Tool
from ..models import Agent, AgentRole
from ..engines.langgraph import LangGraphOrchestrator


class ResearchState(TypedDict):
    query: str
    results: List[str]
    status: str


class ResearchConfig(TypedDict):
    max_iterations: int
    timeout: float


def create_research_workflow() -> LangGraphOrchestrator[ResearchState, ResearchConfig, Tool]:
    orchestrator = LangGraphOrchestrator[ResearchState, ResearchConfig](
        config={"max_iterations": 3, "timeout": 300.0}
    )

    researcher = Agent[Tool](
        name="researcher",
        role=AgentRole.WORKER,
        description="Conducts initial research and analysis",
        tools=[]
    )

    reviewer = Agent[Tool](
        name="reviewer",
        role=AgentRole.REVIEWER,
        description="Reviews and validates research findings",
        tools=[]
    )

    orchestrator.add_agent(researcher)
    orchestrator.add_agent(reviewer)

    orchestrator.set_entry_point("researcher")
    orchestrator.connect("researcher", "reviewer")

    return orchestrator


async def main():
    workflow = create_research_workflow()
    initial_state: ResearchState = {
        "query": "Research quantum computing",
        "results": [],
        "status": "started"
    }
    result = await workflow.run(initial_state)
    print(result.state)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
