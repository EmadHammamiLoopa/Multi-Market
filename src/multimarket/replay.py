from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .execution import simulate_barrier_trade
from .metrics import Metrics, calculate_metrics
from .models import Direction, MarketBar, Prediction, TradeResult
from .strategy import Predictor


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    horizon_bars: int = 4
    confidence_threshold: float = 0.60
    take_profit_bps: float = 20.0
    stop_loss_bps: float = 10.0
    round_trip_cost_bps: float = 2.0

    def __post_init__(self) -> None:
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")


@dataclass(slots=True)
class ReplayReport:
    predictions: list[Prediction]
    directional_hits: list[bool]
    trades: list[TradeResult]
    metrics: Metrics

    def as_dict(self) -> dict[str, object]:
        data = asdict(self.metrics)
        data["profit_factor"] = self.metrics.profit_factor
        return data


class ReplayEngine:
    def __init__(self, predictor: Predictor, config: ReplayConfig) -> None:
        self.predictor = predictor
        self.config = config

    def run(self, bars: Sequence[MarketBar], *, min_history_bars: int) -> ReplayReport:
        if min_history_bars <= 0:
            raise ValueError("min_history_bars must be positive")
        if len(bars) < min_history_bars + self.config.horizon_bars:
            raise ValueError("not enough bars for replay")

        predictions: list[Prediction] = []
        directional_hits: list[bool] = []
        trades: list[TradeResult] = []

        last_decision_index = len(bars) - self.config.horizon_bars - 1
        for index in range(min_history_bars - 1, last_decision_index + 1):
            history = bars[: index + 1]
            prediction = self.predictor.predict(history, self.config.horizon_bars)
            if prediction.timestamp != bars[index].timestamp:
                raise ValueError("predictor timestamp must equal history endpoint")
            if prediction.reference_price != bars[index].close:
                raise ValueError("predictor reference_price must equal current close")

            if prediction.confidence < self.config.confidence_threshold:
                prediction = Prediction(
                    timestamp=prediction.timestamp,
                    direction=Direction.NO_TRADE,
                    confidence=prediction.confidence,
                    reference_price=prediction.reference_price,
                    horizon_bars=prediction.horizon_bars,
                )

            future = bars[index + 1 : index + 1 + self.config.horizon_bars]
            actual_return = future[-1].close / bars[index].close - 1.0
            actual_direction = Direction.LONG if actual_return > 0 else Direction.SHORT
            hit = prediction.direction is not Direction.NO_TRADE and prediction.direction is actual_direction

            predictions.append(prediction)
            directional_hits.append(hit)

            if prediction.direction is not Direction.NO_TRADE:
                trades.append(
                    simulate_barrier_trade(
                        prediction,
                        future,
                        take_profit_bps=self.config.take_profit_bps,
                        stop_loss_bps=self.config.stop_loss_bps,
                        round_trip_cost_bps=self.config.round_trip_cost_bps,
                    )
                )

        metrics = calculate_metrics(predictions, directional_hits, trades)
        return ReplayReport(predictions, directional_hits, trades, metrics)
