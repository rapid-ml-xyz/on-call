import pytest
import pandas as pd
import numpy as np
from src.on_call.modules.trend_analyzer import TrendAnalyzer
from src.on_call.modules.trend_analyzer.trend_patterns import SeasonalityType, TrendType
from datetime import datetime


@pytest.fixture
def sample_data():
    """Create a sample dataset with timestamps and error counts."""
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq="H")
    error_counts = np.random.poisson(
        lam=5, size=100
    )  # Random errors with Poisson distribution
    df = pd.DataFrame({"timestamp": timestamps, "error_count": error_counts})
    return df


@pytest.fixture
def seasonal_data():
    """Create a dataset with a clear seasonal pattern."""
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq="H")
    error_counts = np.tile([5, 10, 15, 20, 25], 20)  # Repeating seasonal pattern
    df = pd.DataFrame({"timestamp": timestamps, "error_count": error_counts})
    return df


@pytest.fixture
def trend_data():
    """Create a dataset with a clear increasing trend."""
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq="H")
    error_counts = np.arange(100)  # Increasing trend
    df = pd.DataFrame({"timestamp": timestamps, "error_count": error_counts})
    return df


@pytest.fixture
def outlier_data():
    """Create a dataset with random outliers."""
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq="H")
    error_counts = np.random.poisson(lam=5, size=100)
    error_counts[20] = 100  # Large outlier
    error_counts[50] = 150  # Another outlier
    df = pd.DataFrame({"timestamp": timestamps, "error_count": error_counts})
    return df


@pytest.fixture
def random_data():
    """Create a dataset with purely random error values."""
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq="H")
    error_counts = np.random.randint(0, 50, size=100)  # High variance, no pattern
    df = pd.DataFrame({"timestamp": timestamps, "error_count": error_counts})
    return df


@pytest.fixture
def heteroscedastic_data():
    """Create a dataset with periods of high and low variance."""
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq="H")
    error_counts = np.concatenate([np.random.poisson(5, 50), np.random.poisson(20, 50)])
    df = pd.DataFrame({"timestamp": timestamps, "error_count": error_counts})
    return df


# Test Seasonality Detection
def test_seasonality(seasonal_data):
    analyzer = TrendAnalyzer(seasonal_data)
    result = analyzer.detect_seasonality()
    assert result.seasonal is True
    assert result.pattern in [
        SeasonalityType.HOURLY,
        SeasonalityType.DAILY,
        SeasonalityType.WEEKLY,
    ]
    assert result.periodicity is not None
    assert len(result.time_windows) > 0


# Test Trend Detection
def test_trend(trend_data):
    analyzer = TrendAnalyzer(trend_data)
    result = analyzer.detect_trend()
    assert result.trend == TrendType.INCREASING
    assert result.slope > 0


# Test Outlier Detection
def test_outliers(outlier_data):
    analyzer = TrendAnalyzer(outlier_data)
    result = analyzer.detect_outliers()
    assert result.count == 2  # We introduced two outliers
    assert len(result.time_windows) == 2


# Test Randomness
def test_randomness(random_data):
    analyzer = TrendAnalyzer(random_data)
    result = analyzer.detect_randomness()
    assert result.random is True
    assert result.entropy > 2.0


# Test Autocorrelation
def test_autocorrelation(seasonal_data):
    analyzer = TrendAnalyzer(seasonal_data)
    result = analyzer.detect_autocorrelation()
    assert result.autocorrelated is True


# Test Heteroscedasticity
def test_heteroscedasticity(heteroscedastic_data):
    analyzer = TrendAnalyzer(heteroscedastic_data)
    result = analyzer.detect_heteroscedasticity()
    assert result.heteroscedastic is True
    assert result.variance > 1


# Run Full Analysis
def test_analyze(sample_data):
    analyzer = TrendAnalyzer(sample_data)
    result = analyzer.analyze()
    assert isinstance(result.seasonality.seasonal, bool)
    assert isinstance(result.trend.trend, TrendType)
    assert isinstance(result.outliers.count, int)
    assert isinstance(result.randomness.random, bool)
    assert isinstance(result.autocorrelation.autocorrelated, bool)
    assert isinstance(result.heteroscedasticity.heteroscedastic, bool)


@pytest.mark.parametrize("detection_method", [
    "detect_seasonality",
    "detect_trend",
    "detect_outliers",
    "detect_randomness",
    "detect_autocorrelation",
    "detect_heteroscedasticity"
])
def test_time_windows_format(sample_data, detection_method):
    """Test that all detection methods return properly formatted time windows."""
    analyzer = TrendAnalyzer(sample_data)
    
    result = getattr(analyzer, detection_method)()
    for window in result.time_windows:
        assert isinstance(window, tuple)
        assert len(window) == 2
        assert all(isinstance(t, datetime) for t in window)
        assert window[0] < window[1]
