import pandas as pd
from evidently.report import Report
from evidently.metric_preset import (
    TargetDriftPreset,
    RegressionPreset,
)


def calculate_dynamic_window_info(df: pd.DataFrame) -> tuple[int, pd.Timedelta]:
    """Calculate appropriate window size based on dataset timespan."""
    if len(df) < 2:
        return 1, pd.Timedelta(hours=1)

    timestamps = pd.to_datetime(df.index)
    total_span = timestamps.max() - timestamps.min()

    if total_span >= pd.Timedelta(weeks=4):
        window_unit = pd.Timedelta(weeks=1)
    elif total_span >= pd.Timedelta(weeks=1):
        window_unit = pd.Timedelta(days=1)
    elif total_span >= pd.Timedelta(days=2):
        window_unit = pd.Timedelta(hours=6)
    else:
        window_unit = pd.Timedelta(hours=1)

    window_size = max(1, int(total_span / window_unit))
    n_windows = min(window_size, len(df) // 4)
    actual_window_timeframe = total_span / n_windows if n_windows > 0 else window_unit

    return n_windows, actual_window_timeframe


def generate_time_windows_datetime(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate time windows for analysis based on dataset characteristics."""
    n_windows, timeframe = calculate_dynamic_window_info(df)
    timestamps = pd.to_datetime(df.index)
    start_time = timestamps.min()
    windows = []

    for i in range(n_windows):
        window_start = start_time + (i * timeframe)
        window_end = window_start + timeframe
        windows.append((window_start, window_end))

    return windows


# TODO: Move this to a more appropriate place
def run_performance_reports(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    column_mapping: dict
) -> list[tuple[Report, Report]]:
    """Generate performance and drift reports for each time window."""
    windows = generate_time_windows_datetime(current)
    reports = []

    for i, (start_time, end_time) in enumerate(windows, 1):
        regression_performance = Report(
            metrics=[RegressionPreset()],
            options={"render": {"raw_data": True}}
        )
        regression_performance.run(
            current_data=current.loc[start_time:end_time],
            reference_data=reference,
            column_mapping=column_mapping
        )

        target_drift = Report(
            metrics=[TargetDriftPreset()],
            options={"render": {"raw_data": True}}
        )
        target_drift.run(
            current_data=current.loc[start_time:end_time],
            reference_data=reference,
            column_mapping=column_mapping
        )

        reports.append((regression_performance, target_drift))
        print(f"\nCompleted analysis for window {i}: {start_time} to {end_time}\n")
        print("-" * 80 + "\n")

    return reports
