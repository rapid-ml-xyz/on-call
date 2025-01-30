from ..orchestrator.engines import LangGraphMessageState
from .enums import Step
import random


def route_pattern(state: LangGraphMessageState) -> str:
    return Step.TIME.name if random.random() < 0.5 else Step.DRIFT.name


def route_cohort(state: LangGraphMessageState) -> str:
    return Step.FEATURE_DIST.name if random.random() < 0.5 else Step.GLOBAL_PERF.name


def route_distribution(state: LangGraphMessageState) -> str:
    return Step.FEATURE_IMPORTANCE.name if random.random() < 0.5 else Step.ERROR_ANALYSIS.name


def route_metrics(state: LangGraphMessageState) -> str:
    return Step.METRIC_FOCUSED.name if random.random() < 0.5 else Step.MODEL_BEHAVIOR.name


def route_root_cause(state: LangGraphMessageState) -> str:
    # Don't recurse for now
    return Step.RECOMMENDATIONS.name
