from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True, slots=True)
class MarketBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("OHLC prices must be positive")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("Invalid OHLC bar")


@dataclass(frozen=True, slots=True)
class Prediction:
    timestamp: datetime
    direction: Direction
    confidence: float
    reference_price: float
    horizon_bars: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.reference_price <= 0:
            raise ValueError("reference_price must be positive")
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")


@dataclass(frozen=True, slots=True)
class TradeResult:
    prediction: Prediction
    exit_timestamp: datetime
    exit_price: float
    gross_return_bps: float
    net_return_bps: float
    won: bool
    exit_reason: str
