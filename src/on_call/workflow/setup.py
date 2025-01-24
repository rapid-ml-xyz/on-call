from ..orchestrator.engines import LangGraphOrchestrator, LangGraphToolWrapper, LangGraphMessageState
from ..orchestrator import EdgeConfig, NodeConfig, NodeType, RouteType
from .enums import Step
from .conditions import (route_pattern, route_cohort, route_distribution, route_metrics, route_root_cause)


def setup_analysis_workflow() -> LangGraphOrchestrator:
    orchestrator = LangGraphOrchestrator()

    nodes = [
        NodeConfig[LangGraphMessageState, LangGraphToolWrapper](
            name=stage.name,
            node_type=NodeType.FUNCTION,
            function=lambda x: x
        ) for stage in Step
    ]
    orchestrator.configure_nodes(nodes)

    edges = [
        (Step.PATTERN.name, EdgeConfig(
            route_type=RouteType.DYNAMIC,
            condition=route_pattern,
            routes={
                Step.TIME.name: Step.TIME.name,
                Step.DRIFT.name: Step.DRIFT.name
            }
        )),
        (Step.COHORT.name, EdgeConfig(
            route_type=RouteType.DYNAMIC,
            condition=route_cohort,
            routes={
                Step.FEATURE_DIST.name: Step.FEATURE_DIST.name,
                Step.GLOBAL_PERF.name: Step.GLOBAL_PERF.name
            }
        )),
        (Step.DISTRIBUTION_SHIFT.name, EdgeConfig(
            route_type=RouteType.DYNAMIC,
            condition=route_distribution,
            routes={
                Step.FEATURE_IMPORTANCE.name: Step.FEATURE_IMPORTANCE.name,
                Step.ERROR_ANALYSIS.name: Step.ERROR_ANALYSIS.name
            }
        )),
        (Step.GLOBAL_METRICS.name, EdgeConfig(
            route_type=RouteType.DYNAMIC,
            condition=route_metrics,
            routes={
                Step.METRIC_FOCUSED.name: Step.METRIC_FOCUSED.name,
                Step.MODEL_BEHAVIOR.name: Step.MODEL_BEHAVIOR.name
            }
        )),
        (Step.ROOT_CAUSE.name, EdgeConfig(
            route_type=RouteType.DYNAMIC,
            condition=route_root_cause,
            routes={
                Step.RECOMMENDATIONS.name: Step.RECOMMENDATIONS.name,
                Step.ITERATE_COHORT.name: Step.ITERATE_COHORT.name
            }
        ))
    ]

    sequential_flows = {
        Step.IMPACT_WINDOW.name: Step.BASELINE.name,
        Step.BASELINE.name: Step.PATTERN.name,
        Step.TIME.name: Step.COHORT.name,
        Step.DRIFT.name: Step.COHORT.name,
        Step.FEATURE_DIST.name: Step.DISTRIBUTION_SHIFT.name,
        Step.GLOBAL_PERF.name: Step.GLOBAL_METRICS.name,
        Step.FEATURE_IMPORTANCE.name: Step.LOCAL_EXPLANATION.name,
        Step.ERROR_ANALYSIS.name: Step.LOCAL_EXPLANATION.name,
        Step.METRIC_FOCUSED.name: Step.LOCAL_EXPLANATION.name,
        Step.MODEL_BEHAVIOR.name: Step.LOCAL_EXPLANATION.name,
        Step.LOCAL_EXPLANATION.name: Step.ROOT_CAUSE.name,
        Step.ITERATE_COHORT.name: Step.COHORT.name
    }

    for source, edge in edges:
        orchestrator.add_edge(source, edge)

    for source, dest in sequential_flows.items():
        orchestrator.add_edge(source, EdgeConfig(
            route_type=RouteType.DYNAMIC,
            condition=lambda state, next_node=dest: next_node,
            routes={dest: dest}
        ))

    orchestrator.set_entry_point(Step.IMPACT_WINDOW.name)
    return orchestrator
