from abc import ABC, abstractmethod
from typing import Any, Dict
from ..orchestrator import WorkflowState


class BaseAnalyzer(ABC):

    def __init__(self, workflow_state: WorkflowState):
        self.state: Dict[str, Any] = workflow_state.state

    @abstractmethod
    def run(self) -> WorkflowState:
        pass
