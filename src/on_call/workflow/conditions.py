from typing import Dict, List
from langchain_core.messages import BaseMessage
from .enums import Step


def route_pattern(state: Dict[str, List[BaseMessage]]) -> str:
    last_message = state["messages"][-1]
    if "sudden" in last_message.content.lower():
        return Step.TIME.name
    return Step.DRIFT.name


def route_cohort(state: Dict[str, List[BaseMessage]]) -> str:
    last_message = state["messages"][-1]
    if "high impact" in last_message.content.lower():
        return Step.FEATURE_DIST.name
    return Step.GLOBAL_PERF.name


def route_distribution(state: Dict[str, List[BaseMessage]]) -> str:
    last_message = state["messages"][-1]
    if "shift detected" in last_message.content.lower():
        return Step.FEATURE_IMPORTANCE.name
    return Step.ERROR_ANALYSIS.name


def route_metrics(state: Dict[str, List[BaseMessage]]) -> str:
    last_message = state["messages"][-1]
    if "specific metric" in last_message.content.lower():
        return Step.METRIC_FOCUSED.name
    return Step.MODEL_BEHAVIOR.name


def route_root_cause(state: Dict[str, List[BaseMessage]]) -> str:
    # Don't recurse for now
    return Step.RECOMMENDATIONS.name
