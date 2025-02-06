from .base_module import BaseModule
from ..orchestrator import WorkflowState


class ImpactWindowAnalyzer(BaseModule):
    def run(self) -> WorkflowState:
        self.state['impact_results'] = {
            "impact_score": 0.85,
            "window_size": "7d",
            "anomalies_detected": 3
        }
        return WorkflowState(self.state)
