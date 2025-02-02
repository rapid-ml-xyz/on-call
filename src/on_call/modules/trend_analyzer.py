import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import acf
from torch_frame import TaskType

from .base_analyzer import BaseAnalyzer
from on_call.model.lgbm_model import LGBMModel
from on_call.model_pipeline import ModelPipeline
from on_call.orchestrator import WorkflowState


class TrendAnalyzer(BaseAnalyzer):
    def __init__(self, workflow_state: WorkflowState):
        super().__init__(workflow_state)
        self.patterns = {}
        self.window_size = None

    def _determine_window_size(self, df: pd.DataFrame, timestamp_col: str) -> int:
        if len(df) < 2:
            return 1

        timestamps = pd.to_datetime(df[timestamp_col])
        time_diffs = timestamps.diff().dropna()
        median_diff = time_diffs.median()
        print("median_diff")
        print(median_diff)

        if median_diff <= pd.Timedelta(minutes=5):
            window_size = 60
        elif median_diff <= pd.Timedelta(hours=1):
            window_size = 24
        elif median_diff <= pd.Timedelta(days=1):
            window_size = 15
        elif median_diff <= pd.Timedelta(weeks=1):
            window_size = 8
        else:
            window_size = 6
        return min(window_size, len(df) // 3)

    def run(self) -> WorkflowState:
        pipeline: ModelPipeline = self.state['pipeline']
        df = pipeline.ref_data.test_df
        model: LGBMModel = pipeline.model
        task_type = model.metadata.task_type
        error_col = pipeline.metadata.error_column
        timestamp_col = pipeline.metadata.timestamp_column

        # Data preparation
        df[error_col] = pd.to_numeric(df[error_col], errors='coerce')
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col)

        # Dynamically set the window size
        self.window_size = self._determine_window_size(df, timestamp_col)
        self.patterns['trend'] = self._analyze_trend(df, error_col, timestamp_col, task_type)
        self.patterns['seasonality'] = self._analyze_seasonality(df, error_col, timestamp_col, task_type)
        self.patterns['spikes'] = self._analyze_spikes(df, error_col, timestamp_col, task_type)
        self.patterns['autocorrelation'] = self._analyze_autocorrelation(df, error_col, timestamp_col, task_type)
        self.patterns['heteroscedasticity'] = self._analyze_heteroscedasticity(df, error_col, timestamp_col, task_type)
        self.patterns['lagged_errors'] = self._analyze_lagged_errors(df, error_col, timestamp_col, task_type)

        self.state['trend_analysis'] = self.patterns
        return WorkflowState(self.state)

    def create_time_windows(self, indices: List[int], df: pd.DataFrame, timestamp_col: str) \
            -> List[Tuple[datetime, datetime]]:
        if not indices:
            return []

        valid_indices = [idx for idx in indices if idx in df.index]
        if not valid_indices:
            return []

        dates = pd.to_datetime(df.loc[valid_indices, timestamp_col]).dt.normalize().unique()
        dates = sorted(dates)

        windows = []
        window_start = None
        prev_date = None

        for date in dates:
            if window_start is None:
                window_start = date
                prev_date = date
                continue

            if date != prev_date:
                if (date - prev_date).days > 1:
                    windows.append((window_start, prev_date))
                    window_start = date
                prev_date = date

        # Add the last window
        if window_start is not None:
            windows.append((window_start, prev_date))

        return windows

    def _analyze_trend(self, df: pd.DataFrame, error_col: str, timestamp_col: str, task_type: TaskType) -> Dict:
        if task_type == TaskType.BINARY_CLASSIFICATION:
            values = df[error_col].astype(int)
            rolling_prop = values.rolling(window=self.window_size, min_periods=1).mean()
            trend_direction = np.polyfit(np.arange(len(rolling_prop)), rolling_prop.bfill(), 1)[0]
            trend_threshold = 0.001
        elif task_type == TaskType.REGRESSION:
            rolling_mean = df[error_col].rolling(window=self.window_size, min_periods=1).mean()
            trend_direction = np.polyfit(np.arange(len(rolling_mean)), rolling_mean.bfill(), 1)[0]
            trend_threshold = 0.01
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        trend_exists = abs(trend_direction) > trend_threshold
        result = {
            'exists': trend_exists,
            'direction': 'up' if trend_direction > 0 else 'down',
            'strength': float(abs(trend_direction))
        }

        if trend_exists:
            indices = list(range(len(df)))
            result['time_windows'] = self.create_time_windows(indices, df, timestamp_col)

        return result

    def _analyze_seasonality(self, df: pd.DataFrame, error_col: str, timestamp_col: str, task_type: TaskType) -> Dict:
        if len(df) < 2:
            return {'exists': False}

        if task_type == TaskType.BINARY_CLASSIFICATION:
            values = df[error_col].astype(int)
            error_values = values.rolling(window=self.window_size, min_periods=1).mean().fillna(0).values
            significant_threshold = 0.1
        elif task_type == TaskType.REGRESSION:
            error_values = df[error_col].fillna(0).astype(float).values
            significant_threshold = 0.2
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        max_lags = min(len(df) // 2, 52)
        acf_values = acf(error_values, nlags=max_lags, fft=True)

        peaks = []
        indices = []
        for i in range(1, len(acf_values)-1):
            if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                peaks.append((i, float(acf_values[i])))
                if float(acf_values[i]) > significant_threshold:
                    indices.extend(range(i, len(df), i))

        exists = len(peaks) > 0
        result = {
            'exists': exists,
            'periods': [p[0] for p in peaks if p[1] > significant_threshold],
            'strength': float(max([p[1] for p in peaks], default=0))
        }

        if exists:
            result['time_windows'] = self.create_time_windows(sorted(list(set(indices))), df, timestamp_col)

        return result

    def _analyze_spikes(self, df: pd.DataFrame, error_col: str, timestamp_col: str, task_type: TaskType) -> Dict:
        result = {
            'exists': False,
            'count_zscore': 0,
            'count_iqr': 0
        }

        if task_type == TaskType.BINARY_CLASSIFICATION:
            clean_values = pd.Series(df[error_col].astype(int))
            if len(clean_values) > 0:
                outliers = clean_values == 1
                outlier_count = int(outliers.sum())
                indices = list(outliers[outliers].index)
                exists = outlier_count > 0
                result.update({
                    'exists': exists,
                    'count_zscore': outlier_count,
                    'count_iqr': outlier_count,
                    'indices': indices
                })
                if exists:
                    result['time_windows'] = self.create_time_windows(indices, df, timestamp_col)

        elif task_type == TaskType.REGRESSION:
            error_values = pd.to_numeric(df[error_col], errors='coerce')
            clean_values = error_values.dropna()

            if len(clean_values) > 0:
                z_scores = np.abs(stats.zscore(clean_values))
                z_score_outliers = z_scores > 3
                z_score_count = int(z_score_outliers.sum())

                quartile1 = float(clean_values.quantile(0.25))
                quartile3 = float(clean_values.quantile(0.75))
                iqr_value = quartile3 - quartile1
                lower_bound = quartile1 - 1.5 * iqr_value
                upper_bound = quartile3 + 1.5 * iqr_value
                iqr_outliers = (clean_values < lower_bound) | (clean_values > upper_bound)
                iqr_count = int(iqr_outliers.sum())
                indices = list(clean_values[iqr_outliers].index)
                exists = bool(z_score_count > 0 or iqr_count > 0)

                result.update({
                    'exists': exists,
                    'count_zscore': z_score_count,
                    'count_iqr': iqr_count,
                    'indices': indices
                })
                if exists:
                    result['time_windows'] = self.create_time_windows(indices, df, timestamp_col)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        return result

    def _analyze_autocorrelation(self, df: pd.DataFrame, error_col: str, timestamp_col: str, task_type: TaskType) -> Dict:
        if len(df) < 2:
            return {'exists': False}

        if task_type == TaskType.BINARY_CLASSIFICATION:
            values = df[error_col].astype(int)
            lag_1_corr = np.corrcoef(values[:-1], values[1:])[0, 1]
            correlation_threshold = 0.1
        elif task_type == TaskType.REGRESSION:
            error_values = pd.to_numeric(df[error_col], errors='coerce')
            lag_1_corr = error_values.autocorr(lag=1)
            correlation_threshold = 0.2
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        if pd.isna(lag_1_corr):
            return {'exists': False, 'reason': 'insufficient_data'}

        exists = abs(lag_1_corr) > correlation_threshold
        result = {
            'exists': exists,
            'strength': float(abs(lag_1_corr)),
            'direction': 'positive' if lag_1_corr > 0 else 'negative'
        }

        if exists:
            indices = list(range(len(df)))
            result['time_windows'] = self.create_time_windows(indices, df, timestamp_col)

        return result

    def _analyze_heteroscedasticity(self, df: pd.DataFrame, error_col: str, timestamp_col: str, task_type: TaskType) -> Dict:
        if task_type == TaskType.BINARY_CLASSIFICATION:
            values = df[error_col].astype(int)
            rolling_prop = values.rolling(window=self.window_size, min_periods=1).mean()
            rolling_std = np.sqrt(rolling_prop * (1 - rolling_prop))
            trend_threshold = 0.001
            residuals = values - values.rolling(window=2, min_periods=1).mean().shift(1)
        elif task_type == TaskType.REGRESSION:
            error_values = pd.to_numeric(df[error_col], errors='coerce')
            rolling_std = error_values.rolling(window=self.window_size, min_periods=1).std()
            trend_threshold = 0.01
            residuals = error_values
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        std_trend = np.polyfit(np.arange(len(rolling_std)), rolling_std.bfill(), 1)[0]

        try:
            _, p_value, _, _ = het_breuschpagan(residuals.fillna(0), residuals.fillna(0).shift(1))
        except:
            p_value = 1.0

        significance_level = 0.05
        exists = abs(std_trend) > trend_threshold or p_value < significance_level
        result = {
            'exists': exists,
            'trend_strength': float(abs(std_trend)),
            'p_value': float(p_value)
        }

        if exists:
            indices = list(range(len(df)))
            result['time_windows'] = self.create_time_windows(indices, df, timestamp_col)

        return result

    def _analyze_lagged_errors(self, df: pd.DataFrame, error_col: str, timestamp_col: str, task_type: TaskType) -> Dict:
        if len(df) < 2:
            return {'exists': False}

        if task_type == TaskType.BINARY_CLASSIFICATION:
            values = df[error_col].astype(int)
            rolling_mean = values.rolling(window=self.window_size, min_periods=1).mean()
        elif task_type == TaskType.REGRESSION:
            error_values = pd.to_numeric(df[error_col], errors='coerce')
            rolling_mean = error_values.rolling(window=self.window_size, min_periods=1).mean()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        peaks = []
        peak_indices = []
        for i in range(1, len(rolling_mean)-1):
            if rolling_mean.iloc[i] > rolling_mean.iloc[i-1] and rolling_mean.iloc[i] > rolling_mean.iloc[i+1]:
                peaks.append((df[timestamp_col].iloc[i], float(rolling_mean.iloc[i])))
                peak_indices.append(i)

        intervals = []
        if len(peaks) >= 2:
            intervals = [(peaks[i+1][0] - peaks[i][0]).total_seconds() for i in range(len(peaks)-1)]
            std_interval = float(np.std(intervals))
        else:
            std_interval = float('inf')

        day_in_seconds = 86400
        exists = len(peaks) > 1 and std_interval < day_in_seconds
        result = {
            'exists': exists,
            'peak_count': len(peaks),
            'avg_interval': float(np.mean(intervals)) if intervals else None
        }

        if exists:
            result['time_windows'] = self.create_time_windows(peak_indices, df, timestamp_col)

        return result
