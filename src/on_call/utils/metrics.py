"""
This module contains the functions to calculate different eval metrics 
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from constants import ClassificationMetrics, RegressionMetrics


def _confusion_matrix_helper(y_true, y_pred, classes=None):
    """
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param classes: List of classes or None.

    :return: Tuple of true positive, false positive, false negative
             and true negative.
    """
    return confusion_matrix(y_true, y_pred, labels=classes).ravel()


def false_negative_rate(y_true, y_pred, classes=None):
    """
    Computes false negative rate.
    """
    tn, fp, fn, tp = _confusion_matrix_helper(y_true, y_pred, classes)
    return fn / (fn + tp)


def false_positive_rate(y_true, y_pred, classes=None):
    """
    Computes false positive rate.
    """
    tn, fp, fn, tp = _confusion_matrix_helper(y_true, y_pred, classes)
    return fp / (fp + tn)


def selection_rate(y_true, y_pred, classes=None):
    """
    Computes selection rate.
    """
    tn, fp, fn, tp = _confusion_matrix_helper(y_true, y_pred, classes)
    return (fn + tp) / (tp + fp + fn + tn)


def error_rate(y_true, y_pred, diff=None):
    """
    Computes error rate.
    """
    if diff is None:
        diff = abs(y_pred - y_true)
    total = len(diff)
    error = int(diff.sum())
    if total == 0:
        metric_value = 0
    else:
        metric_value = error / total
    return metric_value


def mean_prediction(y_true, y_pred):
    """
    Computes mean prediction.
    """
    return np.mean(y_pred)


metrics_to_func = {
    ClassificationMetrics.ACCURACY_SCORE: accuracy_score,
    ClassificationMetrics.AUC_SCORE: roc_auc_score,
    ClassificationMetrics.F1_SCORE: f1_score,
    ClassificationMetrics.RECALL_SCORE: recall_score,
    ClassificationMetrics.PRECISION_SCORE: precision_score,
    ClassificationMetrics.FALSE_NEGATIVE_RATE: false_negative_rate,
    ClassificationMetrics.SELECTION_RATE: selection_rate,
    ClassificationMetrics.ERROR_RATE: error_rate,
    ClassificationMetrics.FALSE_POSITIVE_RATE: false_positive_rate,
    RegressionMetrics.MEAN_ABSOLUTE_ERROR: mean_absolute_error,
    RegressionMetrics.MEAN_SQUARED_ERROR: mean_squared_error,
    RegressionMetrics.MEAN_PREDICTION: mean_prediction,
    RegressionMetrics.MEDIAN_ABSOLUTE_ERROR: median_absolute_error,
    RegressionMetrics.R2_SCORE: r2_score,
}


def calculate_classification_metrics(
    pred_y: pd.Series, true_y: pd.Series, **kwargs
) -> dict[ClassificationMetrics, float]:
    return {m: metrics_to_func[m](pred_y, true_y, **kwargs) for m in ClassificationMetrics}


def calculate_regression_metrics(
    pred_y: pd.Series, true_y: pd.Series, **kwargs
) -> dict[RegressionMetrics, float]:
    return {m: metrics_to_func[m](pred_y, true_y, **kwargs) for m in RegressionMetrics}
