from enum import Enum


class Step(Enum):
    IMPORT = "import"
    IMPACT_WINDOW = "define_impact_window"
    WINDOW_VISUALIZATION = "window_visualization"
    BASELINE = "compare_baseline"
    PATTERN = "identify_pattern"


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
