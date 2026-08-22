from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Collection, Sequence

from .execution import simulate_barrier_trade
from .models import Direction, MarketBar, Prediction


@dataclass(frozen=True, slots=True)
class V2BarrierOutcome:
    direction: Direction
    won: bool
    exit_reason: str
    gross_return_bps: float
    net_return_bps: float


@dataclass(frozen=True, slots=True)
class V2EconomicEvent:
    decision_index: int
    future_end_index: int
    long: V2BarrierOutcome
    short: V2BarrierOutcome

    @property
    def best_realized_direction(self) -> Direction:
        """Diagnostic only: best realized net side, not a live decision rule."""
        long_net = self.long.net_return_bps
        short_net = self.short.net_return_bps
        if max(long_net, short_net) <= 0.0:
            return Direction.NO_TRADE
        if long_net > short_net:
            return Direction.LONG
        if short_net > long_net:
            return Direction.SHORT
        return Direction.NO_TRADE


def horizon_is_contiguous(
    bars: Sequence[MarketBar],
    decision_index: int,
    horizon_bars: int,
    *,
    expected_interval: timedelta = timedelta(minutes=5),
) -> bool:
    """Require a true bar-by-bar horizon; never bridge market/data gaps."""
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    future_end = decision_index + horizon_bars
    if decision_index < 0 or future_end >= len(bars):
        return False
    if expected_interval <= timedelta(0):
        raise ValueError("expected_interval must be positive")

    for index in range(decision_index + 1, future_end + 1):
        if bars[index].timestamp - bars[index - 1].timestamp != expected_interval:
            return False
    return True


def event_is_eligible(
    bars: Sequence[MarketBar],
    decision_index: int,
    horizon_bars: int,
    *,
    eligible_indices: Collection[int] | None = None,
    expected_interval: timedelta = timedelta(minutes=5),
) -> bool:
    future_end = decision_index + horizon_bars
    if not horizon_is_contiguous(
        bars,
        decision_index,
        horizon_bars,
        expected_interval=expected_interval,
    ):
        return False
    if eligible_indices is None:
        return True
    eligible = eligible_indices if isinstance(eligible_indices, set) else set(eligible_indices)
    return all(index in eligible for index in range(decision_index, future_end + 1))


def build_economic_event(
    bars: Sequence[MarketBar],
    decision_index: int,
    *,
    horizon_bars: int,
    take_profit_bps: float,
    stop_loss_bps: float,
    round_trip_cost_bps: float,
    eligible_indices: Collection[int] | None = None,
    expected_interval: timedelta = timedelta(minutes=5),
) -> V2EconomicEvent | None:
    """Build leakage-free realized long/short barrier outcomes for one event.

    The function is a label builder only. It does not choose a live side and does
    not expose any future information to a predictor. Same-bar TP/SL ambiguity is
    inherited from the existing conservative execution simulator (STOP first).
    """
    if not event_is_eligible(
        bars,
        decision_index,
        horizon_bars,
        eligible_indices=eligible_indices,
        expected_interval=expected_interval,
    ):
        return None

    decision_bar = bars[decision_index]
    future_end = decision_index + horizon_bars
    future_bars = bars[decision_index + 1 : future_end + 1]

    outcomes: dict[Direction, V2BarrierOutcome] = {}
    for direction in (Direction.LONG, Direction.SHORT):
        prediction = Prediction(
            timestamp=decision_bar.timestamp,
            direction=direction,
            confidence=1.0,
            reference_price=decision_bar.close,
            horizon_bars=horizon_bars,
        )
        trade = simulate_barrier_trade(
            prediction,
            future_bars,
            take_profit_bps=take_profit_bps,
            stop_loss_bps=stop_loss_bps,
            round_trip_cost_bps=round_trip_cost_bps,
        )
        outcomes[direction] = V2BarrierOutcome(
            direction=direction,
            won=trade.won,
            exit_reason=trade.exit_reason,
            gross_return_bps=trade.gross_return_bps,
            net_return_bps=trade.net_return_bps,
        )

    return V2EconomicEvent(
        decision_index=decision_index,
        future_end_index=future_end,
        long=outcomes[Direction.LONG],
        short=outcomes[Direction.SHORT],
    )
