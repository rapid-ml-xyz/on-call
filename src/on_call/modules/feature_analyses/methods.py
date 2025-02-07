import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset


@dataclass
class WindowAnalysis:
    window_id: int
    start_time: datetime
    end_time: datetime
    drift_report: Report
    quality_report: Report


def run_critical_window_feature_reports(current: pd.DataFrame,
                                        reference: pd.DataFrame,
                                        numerical_features,
                                        categorical_features,
                                        column_mapping: ColumnMapping,
                                        critical_windows: list) -> list[WindowAnalysis]:
    window_analyses = []

    for window in critical_windows:
        start_time = pd.Timestamp(window['time_range']['start'])
        end_time = pd.Timestamp(window['time_range']['end'])

        feature_drift = Report(
            metrics=[DataDriftPreset(columns=numerical_features+categorical_features)],
            options={"render": {"raw_data": True}}
        )
        feature_drift.run(
            current_data=current.loc[start_time:end_time],
            reference_data=reference,
            column_mapping=column_mapping
        )

        feature_quality = Report(
            metrics=[DataQualityPreset(columns=numerical_features+categorical_features)],
            options={"render": {"raw_data": True}}
        )
        feature_quality.run(
            current_data=current.loc[start_time:end_time],
            reference_data=reference,
            column_mapping=column_mapping
        )

        window_analyses.append(WindowAnalysis(
            window_id=window['window_id'],
            start_time=start_time,
            end_time=end_time,
            drift_report=feature_drift,
            quality_report=feature_quality
        ))

        print(f"\nCompleted feature analysis for critical window {window['window_id']}: "
              f"{start_time} to {end_time}\n")
        print("-" * 80 + "\n")

    return window_analyses
