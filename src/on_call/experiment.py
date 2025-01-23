import pandas as pd
import sklearn

from constants import ClassificationMetrics, RegressionMetrics, ModelTask
from utils.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)


class Experiment:
    """
    The Experiment class holds all the information for a single experiment including,
    training data, test data, predictions, model, metrics, and model task.
    """

    def __init__(self) -> None:
        self.training_data: pd.DataFrame = None
        self.test_data: pd.DataFrame = None
        self.pred_y: pd.Series = None
        self.true_y: pd.Series = None
        self.model: sklearn.base.BaseEstimator = None
        self.metrics: dict[ClassificationMetrics | RegressionMetrics, float] = None
        self.model_task: ModelTask = None

    def train(self):
        pass

    def evaluate(self):
        if self.model_task == ModelTask.CLASSIFICATION:
            self.metrics = calculate_classification_metrics(self.pred_y, self.true_y)
        elif self.model_task == ModelTask.REGRESSION:
            self.metrics = calculate_regression_metrics(self.pred_y, self.true_y)

    def save(self):
        pass

    def load(self):
        pass
