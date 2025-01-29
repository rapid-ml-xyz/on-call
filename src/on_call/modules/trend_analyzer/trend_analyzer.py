import pandas as pd
import numpy as np
from scipy.stats import zscore, entropy, pearsonr
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
from typing import List, Tuple
from datetime import datetime

from on_call.model_pipeline import ModelPipeline
from .trend_patterns import (
    TrendType,
    SeasonalityType,
    TrendResult,
    SeasonalityResult,
    OutlierResult,
    RandomnessResult,
    AutocorrelationResult,
    HeteroscedasticityResult,
    TrendAnalysisResult,
)


class TrendAnalyzer:
    def __init__(self, pipeline: ModelPipeline, df: pd.DataFrame):
        """Initialize TrendAnalyzer with a DataFrame containing 'timestamp' and 'error_count'."""
        self.pipeline = pipeline
        self.error_column = pipeline._metadata.error_column
        self.df = df
        self._prepare_df()

    def _prepare_df(self):
        """Prepare error dataframe by extracting relevant columns and calculating error counts."""
        # Extract relevant columns and set multi-index
        self.df = self._extract_and_index_df(
            self.pipeline._metadata.target_column,
            self.pipeline._metadata.id_column,
            self.pipeline._metadata.timestamp_column,
            self.pipeline._metadata.prediction_column,
            self.pipeline._metadata.error_column,
        )

        self.df[self.pipeline._metadata.error_column] = self.df[
            self.pipeline._metadata.error_column
        ].astype(int)

        # Reset index to get timestamp as a column
        self.df = self.df.reset_index()

        # Convert timestamp to datetime
        self.df[self.pipeline._metadata.timestamp_column] = pd.to_datetime(
            self.df[self.pipeline._metadata.timestamp_column]
        )
        self.df = self.df.set_index(self.pipeline._metadata.timestamp_column)

        # Resample to hourly frequency and fill missing values
        self.df = (
            self.df.loc[:, ["error"]]
            .groupby(level=self.pipeline._metadata.timestamp_column)
            .sum()
            .resample("H")
            .sum()
            .fillna(0)
        )

    def _extract_and_index_df(
        self,
        target_column,
        id_column,
        timestamp_column,
        prediction_column,
        error_column,
    ):
        """Helper method to extract columns and set index."""
        return (
            self.df.loc[
                :,
                [
                    target_column,
                    id_column,
                    timestamp_column,
                    prediction_column,
                    error_column,
                ],
            ]
            .copy()
            .set_index([timestamp_column, id_column])
        )

    def detect_seasonality(self) -> SeasonalityResult:
        """Detect seasonality using autocorrelation and identify the pattern and time windows."""
        autocorr_values = acf(self.df[self.error_column], nlags=24)
        periodicity = (
            np.argmax(autocorr_values[1:]) + 1
            if max(autocorr_values[1:]) > 0.5
            else None
        )

        pattern = SeasonalityType.NO_PATTERN
        if periodicity:
            if periodicity <= 24:
                pattern = SeasonalityType.HOURLY
            elif periodicity <= 168:  # 24 * 7
                pattern = SeasonalityType.DAILY
            else:
                pattern = SeasonalityType.WEEKLY

        time_windows = (
            self.df.index[
                self.df[self.error_column].rolling(periodicity).mean().notna()
            ].tolist()
            if periodicity
            else []
        )

        return SeasonalityResult(
            seasonal=periodicity is not None,
            pattern=pattern,
            periodicity=periodicity,
            time_windows=self._create_time_windows(time_windows),
        )

    def detect_trend(self) -> TrendResult:
        """Detect increasing or decreasing trend using linear regression slope."""
        x = np.arange(len(self.df))
        y = self.df[self.error_column].values
        slope = np.polyfit(x, y, 1)[0]

        # Detect change points using rolling statistics
        window_size = max(int(len(self.df) * 0.1), 2)  # 10% of data points or minimum 2
        rolling_mean = self.df[self.error_column].rolling(window=window_size).mean()
        rolling_std = self.df[self.error_column].rolling(window=window_size).std()

        # Find points where the trend changes significantly
        change_points = self.df.index[
            (abs(rolling_mean.diff()) > 2 * rolling_std)
            & (rolling_std > rolling_std.mean())
        ].tolist()

        trend = (
            TrendType.INCREASING
            if slope > 0
            else TrendType.DECREASING if slope < 0 else TrendType.NO_TREND
        )

        return TrendResult(
            trend=trend,
            slope=slope,
            time_windows=self._create_time_windows(change_points),
        )

    def detect_outliers(self) -> OutlierResult:
        """Detect outliers using Z-score method and return detailed time windows."""
        z_scores = np.abs(zscore(self.df[self.error_column]))
        outlier_indices = np.where(z_scores > 3)[0]
        time_windows = self.df.index[outlier_indices].tolist()
        return OutlierResult(
            count=len(outlier_indices),
            indices=outlier_indices.tolist(),
            time_windows=self._create_time_windows(time_windows),
        )

    def detect_randomness(self) -> RandomnessResult:
        """Measure entropy and other randomness indicators."""
        # Calculate normalized entropy
        hist, _ = np.histogram(self.df[self.error_column], bins="auto", density=True)
        entropy_val = entropy(hist)

        # Check for autocorrelation
        acf_val = pearsonr(
            self.df[self.error_column].values[:-1], self.df[self.error_column].values[1:]
        )[0]

        # Data is considered random if entropy is high and autocorrelation is low
        is_random = bool(
            entropy_val > 1.8 and abs(acf_val) < 0.3
        )  # Explicitly cast to bool

        # Find windows of high entropy
        window_size = max(int(len(self.df) * 0.1), 2)
        values = []
        valid_indices = []

        for i in range(len(self.df) - window_size + 1):
            window_data = self.df[self.error_column].iloc[i : i + window_size]
            hist, _ = np.histogram(window_data, bins="auto", density=True)
            values.append(entropy(hist))
            valid_indices.append(self.df.index[i + window_size - 1])

        rolling_entropy = pd.Series(values, index=valid_indices)

        high_entropy_windows = rolling_entropy[
            rolling_entropy > rolling_entropy.mean() + rolling_entropy.std()
        ].index.tolist()

        return RandomnessResult(
            random=is_random,  # This is now guaranteed to be bool
            entropy=float(entropy_val),  # Ensure float type
            time_windows=self._create_time_windows(high_entropy_windows),
        )

    def detect_autocorrelation(self) -> AutocorrelationResult:
        """Detect autocorrelation patterns."""
        # Calculate autocorrelation at different lags
        max_lag = min(24, len(self.df) // 4)  # Up to 24 hours or 1/4 of data length
        acf_values = [
            pearsonr(
                self.df[self.error_column].values[:-i], self.df[self.error_column].values[i:]
            )[0]
            for i in range(1, max_lag)
        ]

        # Find significant autocorrelation peaks
        peaks, _ = find_peaks(np.abs(acf_values), height=0.3)

        # Find windows with strong autocorrelation
        window_size = max(int(len(self.df) * 0.1), 2)
        values = []
        valid_indices = []

        for i in range(len(self.df) - window_size + 1):
            window_data = self.df[self.error_column].iloc[i : i + window_size]
            corr = pearsonr(window_data.values[:-1], window_data.values[1:])[0]
            values.append(corr)
            valid_indices.append(self.df.index[i + window_size - 1])

        rolling_acf = pd.Series(values, index=valid_indices)

        autocorr_windows = rolling_acf[
            rolling_acf > rolling_acf.mean() + rolling_acf.std()
        ].index.tolist()

        return AutocorrelationResult(
            autocorrelated=len(peaks) > 0,
            p_value=np.max(np.abs(acf_values)),
            time_windows=self._create_time_windows(autocorr_windows),
        )

    def detect_heteroscedasticity(self) -> HeteroscedasticityResult:
        """Check if error variance changes significantly over time and identify affected windows."""
        rolling_std = self.df[self.error_column].rolling(window=24).std()
        time_windows = self.df.index[
            rolling_std > rolling_std.mean() + rolling_std.std()
        ].tolist()

        return HeteroscedasticityResult(
            heteroscedastic=rolling_std.var() > 1,
            variance=rolling_std.var(),
            time_windows=self._create_time_windows(time_windows),
        )

    def analyze(self) -> TrendAnalysisResult:
        """Run all trend detection functions and return a comprehensive analysis."""
        randomness_result = self.detect_randomness()

        # Create the result with explicit type checking
        result = TrendAnalysisResult(
            seasonality=self.detect_seasonality(),
            trend=self.detect_trend(),
            outliers=self.detect_outliers(),
            randomness=RandomnessResult(
                random=bool(randomness_result.random),  # Ensure bool type
                entropy=float(randomness_result.entropy),  # Ensure float type
                time_windows=randomness_result.time_windows,
            ),
            autocorrelation=self.detect_autocorrelation(),
            heteroscedasticity=self.detect_heteroscedasticity(),
        )

        return result

    def _create_time_windows(self, indices) -> List[Tuple[datetime, datetime]]:
        """Create time windows from consecutive indices."""
        if not indices:
            return []

        # Convert timestamps to integer positions
        index_positions = [self.df.index.get_loc(idx) for idx in indices]
        index_positions.sort()

        windows = []
        window_start_pos = None
        prev_pos = None

        for pos in index_positions:
            if window_start_pos is None:
                window_start_pos = pos
                prev_pos = pos
            elif pos - prev_pos > 1:  # Gap found
                # Convert positions back to timestamps for the window
                # Add 1 hour to end timestamp to ensure it's greater than start
                windows.append(
                    (
                        self.df.index[window_start_pos],
                        self.df.index[prev_pos] + pd.Timedelta(hours=1),
                    )
                )
                window_start_pos = pos
            prev_pos = pos

        # Add the last window
        if window_start_pos is not None:
            windows.append(
                (
                    self.df.index[window_start_pos],
                    self.df.index[prev_pos] + pd.Timedelta(hours=1),
                )
            )

        return windows


# Example Usage:
# df = pd.read_csv("errors.csv")
# analyzer = TrendAnalyzer(df)
# result_summary = analyzer.analyze()
# print(result_summary)
