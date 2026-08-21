from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Protocol, Sequence

from .models import Direction, MarketBar, Prediction


class Predictor(Protocol):
    def predict(self, history: Sequence[MarketBar], horizon_bars: int) -> Prediction:
        """Predict using history that ends exactly at the decision timestamp."""


@dataclass(slots=True)
class MomentumBaselinePredictor:
    """A deliberately simple, auditable baseline.

    This is not presented as a profitable strategy. Its purpose is to prove that
    the replay engine, scoring and execution simulation work before replacing it
    with learned models.
    """

    lookback_bars: int = 20
    neutral_momentum_bps: float = 2.0

    def predict(self, history: Sequence[MarketBar], horizon_bars: int) -> Prediction:
        if len(history) < self.lookback_bars:
            raise ValueError("insufficient history for configured lookback")

        window = history[-self.lookback_bars :]
        start = window[0].close
        end = window[-1].close
        momentum_bps = (end / start - 1.0) * 10_000.0
        magnitude = abs(momentum_bps)

        confidence = min(0.95, 0.5 + 0.45 * (1.0 - exp(-magnitude / 20.0)))

        if magnitude < self.neutral_momentum_bps:
            direction = Direction.NO_TRADE
        elif momentum_bps > 0:
            direction = Direction.LONG
        else:
            direction = Direction.SHORT

        return Prediction(
            timestamp=history[-1].timestamp,
            direction=direction,
            confidence=confidence,
            reference_price=end,
            horizon_bars=horizon_bars,
        )
