from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime


class TrendType(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    NO_TREND = "no trend"


class SeasonalityType(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    NO_PATTERN = "no pattern"


@dataclass
class TrendResult:
    trend: TrendType
    slope: float
    time_windows: List[Tuple[datetime, datetime]]


@dataclass
class SeasonalityResult:
    seasonal: bool
    pattern: SeasonalityType
    periodicity: Optional[int]
    time_windows: List[Tuple[datetime, datetime]]


@dataclass
class OutlierResult:
    count: int
    indices: List[int]
    time_windows: List[Tuple[datetime, datetime]]


@dataclass
class RandomnessResult:
    random: bool
    entropy: float
    time_windows: List[Tuple[datetime, datetime]]


@dataclass
class AutocorrelationResult:
    autocorrelated: bool
    p_value: float
    time_windows: List[Tuple[datetime, datetime]]


@dataclass
class HeteroscedasticityResult:
    heteroscedastic: bool
    variance: float
    time_windows: List[Tuple[datetime, datetime]]


@dataclass
class TrendAnalysisResult:
    seasonality: SeasonalityResult
    trend: TrendResult
    outliers: OutlierResult
    randomness: RandomnessResult
    autocorrelation: AutocorrelationResult
    heteroscedasticity: HeteroscedasticityResult
