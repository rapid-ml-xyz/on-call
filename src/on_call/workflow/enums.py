from enum import Enum


class Step(Enum):
    IMPORTS = "import"
    TIME_SERIES_REPORTS = "time_series_reports"
    WINDOW_VISUALIZATIONS = "window_visualization"
    FLAG_CRITICAL_WINDOWS = "flag_critical_windows"
    TIME_SERIES_SUMMARY = "time_series_summary"
    FEATURE_ANALYSES = "feature_analyses"


class Pattern(Enum):
    SUDDEN = "sudden_drop"
    GRADUAL = "gradual_decline"


class CohortImpact(Enum):
    HIGH = "high_impact_segments"
    LOW = "no_clear_segments"


class DistributionShift(Enum):
    YES = "yes"
    NO = "no"


class MetricDegradation(Enum):
    SPECIFIC = "specific_metric_drop"
    OVERALL = "overall_degradation"


class RootCause(Enum):
    IDENTIFIED = "yes"
    NOT_IDENTIFIED = "no"
