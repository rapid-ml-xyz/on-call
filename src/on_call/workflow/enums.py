from enum import Enum


class Step(Enum):
    TREND_ANALYZER = "_trend_analyzer"
    IMPACT_WINDOW = "define_impact_window"
    BASELINE = "compare_baseline"
    PATTERN = "identify_pattern"
    TIME = "time_analysis"
    DRIFT = "drift_analysis"
    COHORT = "cohort_analysis"
    FEATURE_DIST = "feature_distribution"
    GLOBAL_PERF = "global_performance"
    DISTRIBUTION_SHIFT = "distribution_shift"
    FEATURE_IMPORTANCE = "feature_importance"
    ERROR_ANALYSIS = "error_analysis"
    GLOBAL_METRICS = "global_metrics"
    METRIC_FOCUSED = "metric_focused"
    MODEL_BEHAVIOR = "model_behavior"
    LOCAL_EXPLANATION = "local_explanation"
    ROOT_CAUSE = "root_cause"
    RECOMMENDATIONS = "recommendations"
    ITERATE_COHORT = "iterate_cohort"


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
