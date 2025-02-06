from .base_analyzer import BaseAnalyzer
from ..orchestrator import WorkflowState


class BaselineAnalyzer(BaseAnalyzer):
    def run(self) -> WorkflowState:
        model = self.state['model']
        self.state['baseline_results'] = {"ref": model.prediction_col}
        return WorkflowState(self.state)
