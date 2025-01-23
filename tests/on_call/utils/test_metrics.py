import pytest
import pandas as pd
from src.on_call.utils.metrics import (
    _confusion_matrix_helper,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    error_rate,
    mean_prediction,
    calculate_classification_metrics,
    calculate_regression_metrics,
)
from src.on_call.constants import ClassificationMetrics, RegressionMetrics


@pytest.fixture
def binary_classification_data():
    """Fixture for binary classification test data"""
    y_true = pd.Series([0, 0, 1, 1, 1, 0])
    y_pred = pd.Series([0, 1, 1, 1, 0, 0])
    return y_true, y_pred


@pytest.fixture
def regression_data():
    """Fixture for regression test data"""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = pd.Series([1.1, 2.2, 2.8, 4.2, 4.8])
    return y_true, y_pred


def test_confusion_matrix_helper(binary_classification_data):
    y_true, y_pred = binary_classification_data
    tn, fp, fn, tp = _confusion_matrix_helper(y_true, y_pred)
    assert tn == 2  # True negatives
    assert fp == 1  # False positives
    assert fn == 1  # False negatives
    assert tp == 2  # True positives


def test_false_negative_rate(binary_classification_data):
    y_true, y_pred = binary_classification_data
    fnr = false_negative_rate(y_true, y_pred)
    assert fnr == pytest.approx(1 / 3)  # 1 false negative out of 3 positive cases


def test_false_positive_rate(binary_classification_data):
    y_true, y_pred = binary_classification_data
    fpr = false_positive_rate(y_true, y_pred)
    assert fpr == pytest.approx(1 / 3)  # 1 false positive out of 3 negative cases


def test_selection_rate(binary_classification_data):
    y_true, y_pred = binary_classification_data
    sr = selection_rate(y_true, y_pred)
    assert sr == 0.5  # (1 + 2) / 6 total cases


def test_error_rate():
    y_true = pd.Series([1, 2, 3, 4])
    y_pred = pd.Series([1, 2, 4, 4])
    er = error_rate(y_true, y_pred)
    assert er == 0.25  # 1 error out of 4 cases


def test_error_rate_with_diff():
    diff = pd.Series([0, 0, 1, 0])  # Explicitly provided differences
    er = error_rate(None, None, diff=diff)
    assert er == 0.25


def test_error_rate_empty():
    diff = pd.Series([])
    er = error_rate(None, None, diff=diff)
    assert er == 0


def test_mean_prediction(regression_data):
    y_true, y_pred = regression_data
    mp = mean_prediction(y_true, y_pred)
    assert mp == pytest.approx(3.02)


def test_calculate_classification_metrics(binary_classification_data):
    y_true, y_pred = binary_classification_data
    metrics = calculate_classification_metrics(y_true, y_pred)

    assert isinstance(metrics, dict)
    assert len(metrics) == len(ClassificationMetrics)
    assert metrics[ClassificationMetrics.ACCURACY_SCORE.value] == pytest.approx(
        0.6666666
    )
    assert metrics[ClassificationMetrics.FALSE_NEGATIVE_RATE.value] == pytest.approx(
        1 / 3
    )
    assert metrics[ClassificationMetrics.FALSE_POSITIVE_RATE.value] == pytest.approx(
        1 / 3
    )


def test_calculate_regression_metrics(regression_data):
    y_true, y_pred = regression_data
    metrics = calculate_regression_metrics(y_true, y_pred)

    assert isinstance(metrics, dict)
    assert len(metrics) == len(RegressionMetrics)
    assert metrics[RegressionMetrics.MEAN_ABSOLUTE_ERROR] == pytest.approx(0.18, abs=1e-2)
    assert metrics[RegressionMetrics.R2_SCORE] == pytest.approx(0.98, abs=1e-2)


@pytest.mark.parametrize(
    "y_true,y_pred,expected",
    [
        (pd.Series([0, 1]), pd.Series([0, 1]), 1.0),  # Perfect prediction
        (pd.Series([0, 1]), pd.Series([1, 0]), 0.0),  # Worst prediction
        (pd.Series([0, 0, 1, 1]), pd.Series([0, 1, 1, 0]), 0.5),  # Mixed case
    ],
)
def test_accuracy_score_parametrized(y_true, y_pred, expected):
    metrics = calculate_classification_metrics(y_true, y_pred)
    assert metrics[ClassificationMetrics.ACCURACY_SCORE.value] == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "y_true,y_pred,expected_mae",
    [
        (pd.Series([1, 2]), pd.Series([1, 2]), 0.0),  # Perfect prediction
        (pd.Series([1, 2]), pd.Series([2, 3]), 1.0),  # Constant error
        (pd.Series([1, 2, 3]), pd.Series([1.5, 2.5, 3.5]), 0.5),  # Consistent offset
    ],
)
def test_regression_metrics_parametrized(y_true, y_pred, expected_mae):
    metrics = calculate_regression_metrics(y_true, y_pred)
    assert metrics[RegressionMetrics.MEAN_ABSOLUTE_ERROR] == pytest.approx(expected_mae)
