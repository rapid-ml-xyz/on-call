from .base_module import BaseModule
from ..orchestrator import WorkflowState


class BaselineAnalyzer(BaseModule):
    def run(self) -> WorkflowState:
        model = self.state['model']
        self.state['baseline_results'] = {"ref": model.prediction_col}
        return WorkflowState(self.state)
