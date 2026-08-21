from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import cos, exp, log, pi, sin, sqrt
from statistics import fmean
from typing import Sequence
from zoneinfo import ZoneInfo

from .models import MarketBar


FEATURE_NAMES = (
    "ret_1_bps",
    "ret_3_bps",
    "ret_6_bps",
    "ret_12_bps",
    "ret_24_bps",
    "vol_6_bps",
    "vol_12_bps",
    "vol_24_bps",
    "vol_48_bps",
    "sma_dist_6_bps",
    "sma_dist_12_bps",
    "sma_dist_24_bps",
    "sma_dist_48_bps",
    "ema_dist_12_bps",
    "ema_dist_26_bps",
    "rsi_14",
    "atr_14_bps",
    "macd_bps",
    "macd_signal_bps",
    "macd_hist_bps",
    "range_bps",
    "body_bps",
    "upper_wick_bps",
    "lower_wick_bps",
    "range_position_12",
    "range_position_24",
    "range_position_48",
    "utc_time_sin",
    "utc_time_cos",
    "dow_sin",
    "dow_cos",
    "london_session",
    "new_york_session",
    "london_ny_overlap",
    "us_regular_session",
)

WARMUP_BARS = 48

_LONDON = ZoneInfo("Europe/London")
_NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class FeaturePoint:
    bar_index: int
    timestamp: datetime
    values: tuple[float, ...]


def _utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _ema(values: Sequence[float], span: int) -> list[float]:
    if span <= 0:
        raise ValueError("span must be positive")
    alpha = 2.0 / (span + 1.0)
    result: list[float] = []
    current: float | None = None
    for value in values:
        current = value if current is None else alpha * value + (1.0 - alpha) * current
        result.append(current)
    return result


def _rolling_vol_bps(closes: Sequence[float], index: int, window: int) -> float:
    returns = [
        log(closes[j] / closes[j - 1]) * 10_000.0
        for j in range(index - window + 1, index + 1)
    ]
    mean = fmean(returns)
    variance = fmean((value - mean) ** 2 for value in returns)
    return sqrt(variance)


def _rsi(closes: Sequence[float], index: int, window: int = 14) -> float:
    deltas = [closes[j] - closes[j - 1] for j in range(index - window + 1, index + 1)]
    gains = fmean(max(delta, 0.0) for delta in deltas)
    losses = fmean(max(-delta, 0.0) for delta in deltas)
    if losses == 0.0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return 100.0 - 100.0 / (1.0 + rs)


def _atr_bps(bars: Sequence[MarketBar], index: int, window: int = 14) -> float:
    true_ranges: list[float] = []
    for j in range(index - window + 1, index + 1):
        previous_close = bars[j - 1].close
        bar = bars[j]
        true_ranges.append(
            max(
                bar.high - bar.low,
                abs(bar.high - previous_close),
                abs(bar.low - previous_close),
            )
        )
    return fmean(true_ranges) / bars[index].close * 10_000.0


def _range_position(bars: Sequence[MarketBar], index: int, window: int) -> float:
    window_bars = bars[index - window + 1 : index + 1]
    low = min(bar.low for bar in window_bars)
    high = max(bar.high for bar in window_bars)
    if high == low:
        return 0.5
    return (bars[index].close - low) / (high - low)


def _session_flags(timestamp: datetime) -> tuple[float, float, float, float]:
    timestamp = _utc(timestamp)
    london = timestamp.astimezone(_LONDON)
    new_york = timestamp.astimezone(_NEW_YORK)

    london_open = london.weekday() < 5 and 8 <= london.hour < 17
    ny_open = new_york.weekday() < 5 and 8 <= new_york.hour < 17
    overlap = london_open and ny_open

    ny_minutes = new_york.hour * 60 + new_york.minute
    us_regular = new_york.weekday() < 5 and (9 * 60 + 30) <= ny_minutes < 16 * 60
    return tuple(float(value) for value in (london_open, ny_open, overlap, us_regular))


def build_feature_points(bars: Sequence[MarketBar]) -> list[FeaturePoint]:
    """Create causal features: every row uses only bars at or before its timestamp."""
    if len(bars) <= WARMUP_BARS:
        return []

    closes = [bar.close for bar in bars]
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd = [a - b for a, b in zip(ema12, ema26)]
    macd_signal = _ema(macd, 9)

    points: list[FeaturePoint] = []
    for index in range(WARMUP_BARS, len(bars)):
        close = closes[index]
        returns = [
            (close / closes[index - lag] - 1.0) * 10_000.0
            for lag in (1, 3, 6, 12, 24)
        ]
        vols = [_rolling_vol_bps(closes, index, window) for window in (6, 12, 24, 48)]
        sma_dist = [
            (close / fmean(closes[index - window + 1 : index + 1]) - 1.0) * 10_000.0
            for window in (6, 12, 24, 48)
        ]
        ema_dist = [
            (close / ema12[index] - 1.0) * 10_000.0,
            (close / ema26[index] - 1.0) * 10_000.0,
        ]

        bar = bars[index]
        denominator = close if close else 1.0
        range_bps = (bar.high - bar.low) / denominator * 10_000.0
        body_bps = (bar.close - bar.open) / denominator * 10_000.0
        upper_wick_bps = (bar.high - max(bar.open, bar.close)) / denominator * 10_000.0
        lower_wick_bps = (min(bar.open, bar.close) - bar.low) / denominator * 10_000.0

        timestamp = _utc(bar.timestamp)
        minute_of_day = timestamp.hour * 60 + timestamp.minute
        utc_angle = 2.0 * pi * minute_of_day / (24.0 * 60.0)
        dow_angle = 2.0 * pi * timestamp.weekday() / 7.0
        sessions = _session_flags(timestamp)

        macd_scale = 10_000.0 / close
        values = tuple(
            returns
            + vols
            + sma_dist
            + ema_dist
            + [
                _rsi(closes, index),
                _atr_bps(bars, index),
                macd[index] * macd_scale,
                macd_signal[index] * macd_scale,
                (macd[index] - macd_signal[index]) * macd_scale,
                range_bps,
                body_bps,
                upper_wick_bps,
                lower_wick_bps,
                _range_position(bars, index, 12),
                _range_position(bars, index, 24),
                _range_position(bars, index, 48),
                sin(utc_angle),
                cos(utc_angle),
                sin(dow_angle),
                cos(dow_angle),
            ]
            + list(sessions)
        )
        if len(values) != len(FEATURE_NAMES):
            raise AssertionError("feature vector length does not match FEATURE_NAMES")
        if any(value != value or value in (float("inf"), float("-inf")) for value in values):
            raise ValueError(f"non-finite feature at {timestamp.isoformat()}")
        points.append(FeaturePoint(index, timestamp, values))
    return points
