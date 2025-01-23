from enum import Enum


class ModelTask(str, Enum):
    """
    Provide model task constants.
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class ClassificationMetrics(str, Enum):
    """
    Provide classification metrics constants.
    """

    ACCURACY_SCORE = "accuracy_score"
    AUC_SCORE = "auc_score"
    F1_SCORE = "f1_score"
    RECALL_SCORE = "recall_score"
    PRECISION_SCORE = "precision_score"
    FALSE_NEGATIVE_RATE = "false_negative_rate"
    SELECTION_RATE = "selection_rate"
    ERROR_RATE = "error_rate"
    FALSE_POSITIVE_RATE = "false_positive_rate"


class RegressionMetrics(str, Enum):
    """
    Provide regression metrics constants.
    """

    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    MEAN_PREDICTION = "mean_prediction"
    MEDIAN_ABSOLUTE_ERROR = "median_absolute_error"
    R2_SCORE = "r2_score"
