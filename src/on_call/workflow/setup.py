from on_call.modules import FeatureAnalyses, FeatureVisualizations, FlagCriticalWindows, ImportsModule, \
    TimeSeriesReports, TimeSeriesSummary, WindowVisualizations
from on_call.orchestrator.engines import LangGraphOrchestrator, LangGraphToolWrapper, LangGraphMessageState
from on_call.orchestrator import EdgeConfig, NodeConfig, NodeType, RouteType
from on_call.workflow.enums import Step


def setup_analysis_workflow() -> LangGraphOrchestrator:
    orchestrator = LangGraphOrchestrator()

    steps = [
        (step.name, lambda state, module=module: module(state).run())
        for step, module in [
            (Step.IMPORTS, ImportsModule),
            (Step.TIME_SERIES_REPORTS, TimeSeriesReports),
            (Step.WINDOW_VISUALIZATIONS, WindowVisualizations),
            (Step.FLAG_CRITICAL_WINDOWS, FlagCriticalWindows),
            (Step.TIME_SERIES_SUMMARY, TimeSeriesSummary),
            (Step.FEATURE_ANALYSES, FeatureAnalyses),
            (Step.FEATURE_VISUALIZATIONS, FeatureVisualizations)
        ]
    ]

    orchestrator.configure_nodes([
        NodeConfig[LangGraphMessageState, LangGraphToolWrapper](
            name=name,
            node_type=NodeType.FUNCTION,
            function=func
        ) for name, func in steps
    ])

    for (current_name, _), (next_step_name, _) in zip(steps[:-1], steps[1:]):
        orchestrator.add_edge(
            current_name,
            EdgeConfig(
                route_type=RouteType.DYNAMIC,
                condition=lambda state, target=next_step_name: target,
                routes={next_step_name: next_step_name}
            )
        )

    orchestrator.set_entry_point(steps[0][0])
    return orchestrator
