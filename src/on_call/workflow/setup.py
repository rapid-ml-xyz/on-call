from typing import Callable, Dict
from ..modules import do_nothing, BaselineAnalyzer, ImpactWindowAnalyzer, TrendAnalyzer
from ..orchestrator.engines import LangGraphOrchestrator, LangGraphToolWrapper, LangGraphMessageState
from ..orchestrator import EdgeConfig, NodeConfig, NodeType, RouteType
from .enums import Step
from .conditions import route_pattern, route_cohort, route_distribution, route_metrics, route_root_cause


def setup_analysis_workflow() -> LangGraphOrchestrator:
    orchestrator = LangGraphOrchestrator()

    node_functions: Dict[str, Callable[[LangGraphMessageState], LangGraphMessageState]] = {
        Step.TREND_ANALYZER.name: lambda state: TrendAnalyzer(state).run(),
        Step.IMPACT_WINDOW.name: lambda state: ImpactWindowAnalyzer(state).run(),
        Step.BASELINE.name: lambda state: BaselineAnalyzer(state).run(),
        Step.PATTERN.name: do_nothing,
        Step.TIME.name: do_nothing,
        Step.DRIFT.name: do_nothing,
        Step.COHORT.name: do_nothing,
        Step.FEATURE_DIST.name: do_nothing,
        Step.GLOBAL_PERF.name: do_nothing,
        Step.DISTRIBUTION_SHIFT.name: do_nothing,
        Step.FEATURE_IMPORTANCE.name: do_nothing,
        Step.ERROR_ANALYSIS.name: do_nothing,
        Step.GLOBAL_METRICS.name: do_nothing,
        Step.METRIC_FOCUSED.name: do_nothing,
        Step.MODEL_BEHAVIOR.name: do_nothing,
        Step.LOCAL_EXPLANATION.name: do_nothing,
        Step.ROOT_CAUSE.name: do_nothing,
        Step.RECOMMENDATIONS.name: do_nothing,
        Step.ITERATE_COHORT.name: do_nothing
    }

    # Configure nodes with actual functions
    nodes = [
        NodeConfig[LangGraphMessageState, LangGraphToolWrapper](
            name=step.name,
            node_type=NodeType.FUNCTION,
            function=node_functions[step.name]
        ) for step in Step
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
        Step.TREND_ANALYZER.name: Step.IMPACT_WINDOW.name,
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

    orchestrator.set_entry_point(Step.TREND_ANALYZER.name)
    return orchestrator
