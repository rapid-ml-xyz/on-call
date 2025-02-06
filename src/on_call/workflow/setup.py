from typing import Callable, Dict
from ..modules import do_nothing, BaselineAnalyzer, ImpactWindowAnalyzer
from ..orchestrator.engines import LangGraphOrchestrator, LangGraphToolWrapper, LangGraphMessageState
from ..orchestrator import EdgeConfig, NodeConfig, NodeType, RouteType
from .enums import Step


def setup_analysis_workflow() -> LangGraphOrchestrator:
    orchestrator = LangGraphOrchestrator()

    node_functions: Dict[str, Callable[[LangGraphMessageState], LangGraphMessageState]] = {
        Step.IMPACT_WINDOW.name: lambda state: ImpactWindowAnalyzer(state).run(),
        Step.BASELINE.name: lambda state: BaselineAnalyzer(state).run(),
        Step.PATTERN.name: do_nothing
    }

    nodes = [
        NodeConfig[LangGraphMessageState, LangGraphToolWrapper](
            name=step.name,
            node_type=NodeType.FUNCTION,
            function=node_functions[step.name]
        ) for step in Step
    ]
    orchestrator.configure_nodes(nodes)

    sequential_flows = {
        Step.IMPACT_WINDOW.name: Step.BASELINE.name,
        Step.BASELINE.name: Step.PATTERN.name
    }

    for source, dest in sequential_flows.items():
        orchestrator.add_edge(source, EdgeConfig(
            route_type=RouteType.DYNAMIC,
            condition=lambda state, next_node=dest: next_node,
            routes={dest: dest}
        ))

    orchestrator.set_entry_point(Step.IMPACT_WINDOW.name)
    return orchestrator
