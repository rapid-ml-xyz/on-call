from .base_analyzer import BaseAnalyzer
from ..model_pipeline import ModelPipeline
from ..orchestrator import WorkflowState


class BaselineAnalyzer(BaseAnalyzer):
    def run(self) -> WorkflowState:
        pipeline: ModelPipeline = self.state['pipeline']
        self.state['baseline_results'] = pipeline.metadata.baseline_metrics
        return WorkflowState(self.state)
