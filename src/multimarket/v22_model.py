from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Collection, Mapping, Sequence

from .models import Direction, MarketBar, Prediction
from .v2_labels import build_economic_event
from .v2_model import (
    V2Config,
    V2Decision,
    V2TrainingPoint,
    _calibrate_probability,
    _fit_classifier,
    _fit_regressor,
)
from .v2_volatility import build_causal_ewma_volatility_bps, volatility_scaled_barriers_bps
from .v21_features import PeerMarket
from .v22_features import build_v22_feature_points, v22_feature_names
from .v22_macro import MacroLedgerIndex


@dataclass(frozen=True, slots=True)
class V22Config:
    base: V2Config = V2Config()
    max_peer_staleness_minutes: int = 15

    def __post_init__(self) -> None:
        if self.max_peer_staleness_minutes <= 0:
            raise ValueError("max_peer_staleness_minutes must be positive")


class V22MacroContextPredictor:
    """Frozen V2.1 policy/features plus preregistered causal macro context."""

    def __init__(
        self,
        bars: Sequence[MarketBar],
        config: V22Config,
        *,
        eligible_indices: Collection[int],
        peers: Mapping[str, PeerMarket],
        macro_ledger: MacroLedgerIndex,
    ) -> None:
        self.bars = bars
        self.config = config
        self.base_config = config.base
        self.eligible_indices = set(eligible_indices)
        self.peers = {symbol.upper(): peer for symbol, peer in peers.items()}
        self.macro_ledger = macro_ledger
        self.feature_points, self.clean_feature_indices = build_v22_feature_points(
            bars,
            eligible_indices=self.eligible_indices,
            peers=self.peers,
            macro_ledger=macro_ledger,
            max_peer_staleness=timedelta(minutes=config.max_peer_staleness_minutes),
        )
        self.feature_by_index = {point.bar_index: point for point in self.feature_points}
        self.volatility = build_causal_ewma_volatility_bps(
            bars,
            eligible_indices=self.eligible_indices,
            span=self.base_config.volatility_span,
            min_observations=self.base_config.volatility_min_observations,
        )
        self.training_points = self._build_training_points()
        self._models = None
        self._model_trained_through = -1
        self._long_validation: list[tuple[float, int]] = []
        self._short_validation: list[tuple[float, int]] = []
        self._fit_count = 0

    @property
    def feature_names(self) -> tuple[str, ...]:
        return v22_feature_names(self.peers)

    def _build_training_points(self) -> list[V2TrainingPoint]:
        result: list[V2TrainingPoint] = []
        for point in self.feature_points:
            index = point.bar_index
            vol = self.volatility[index]
            if vol is None or vol <= 0:
                continue
            tp_bps, sl_bps = volatility_scaled_barriers_bps(
                vol,
                horizon_bars=self.base_config.horizon_bars,
                take_profit_vol_multiplier=self.base_config.take_profit_vol_multiplier,
                stop_loss_vol_multiplier=self.base_config.stop_loss_vol_multiplier,
            )
            event = build_economic_event(
                self.bars,
                index,
                horizon_bars=self.base_config.horizon_bars,
                take_profit_bps=tp_bps,
                stop_loss_bps=sl_bps,
                round_trip_cost_bps=self.base_config.round_trip_cost_bps,
                eligible_indices=self.eligible_indices,
            )
            if event is None:
                continue
            result.append(
                V2TrainingPoint(
                    bar_index=index,
                    future_end_index=event.future_end_index,
                    features=point.values,
                    long_positive=int(event.long.net_return_bps > 0.0),
                    short_positive=int(event.short.net_return_bps > 0.0),
                    long_net_bps=event.long.net_return_bps,
                    short_net_bps=event.short.net_return_bps,
                )
            )
        return result

    def _train_for_index(self, decision_index: int) -> None:
        cfg = self.base_config
        if (
            self._models is not None
            and decision_index >= self._model_trained_through
            and decision_index - self._model_trained_through < cfg.retrain_every
        ):
            return
        cutoff = decision_index - cfg.embargo_bars
        eligible = [point for point in self.training_points if point.future_end_index <= cutoff]
        if len(eligible) < cfg.min_train_rows:
            raise ValueError(
                f"V2.2 needs at least {cfg.min_train_rows} labeled historical rows; only {len(eligible)} are available at this timestamp"
            )
        split = max(cfg.min_train_rows, int(len(eligible) * (1.0 - cfg.validation_fraction)))
        split = min(split, len(eligible) - 1)
        train = eligible[:split]
        validation = eligible[split:]
        X_train = [list(point.features) for point in train]
        long_y = [point.long_positive for point in train]
        short_y = [point.short_positive for point in train]
        long_ev = [point.long_net_bps for point in train]
        short_ev = [point.short_net_bps for point in train]
        self._models = (
            _fit_classifier(X_train, long_y, cfg.random_state),
            _fit_classifier(X_train, short_y, cfg.random_state + 1),
            _fit_regressor(X_train, long_ev, cfg.random_state + 2),
            _fit_regressor(X_train, short_ev, cfg.random_state + 3),
        )
        self._fit_count += 1
        self._long_validation = []
        self._short_validation = []
        if validation:
            X_val = [list(point.features) for point in validation]
            long_probs = self._models[0].predict_proba(X_val)[:, 1]
            short_probs = self._models[1].predict_proba(X_val)[:, 1]
            self._long_validation = [(float(p), point.long_positive) for p, point in zip(long_probs, validation)]
            self._short_validation = [(float(p), point.short_positive) for p, point in zip(short_probs, validation)]
        self._model_trained_through = decision_index

    def predict_at_index(self, decision_index: int) -> V2Decision:
        cfg = self.base_config
        point = self.feature_by_index.get(decision_index)
        if point is None:
            raise ValueError("decision index does not have a clean contiguous V2.2 feature history")
        vol = self.volatility[decision_index]
        if vol is None or vol <= 0:
            raise ValueError("decision index does not have a valid causal volatility estimate")
        self._train_for_index(decision_index)
        assert self._models is not None
        X = [list(point.values)]
        raw_long = float(self._models[0].predict_proba(X)[0][1])
        raw_short = float(self._models[1].predict_proba(X)[0][1])
        p_long = _calibrate_probability(raw_long, self._long_validation)
        p_short = _calibrate_probability(raw_short, self._short_validation)
        ev_long = float(self._models[2].predict(X)[0])
        ev_short = float(self._models[3].predict(X)[0])
        long_ok = p_long >= cfg.min_positive_probability and ev_long > cfg.min_predicted_ev_bps
        short_ok = p_short >= cfg.min_positive_probability and ev_short > cfg.min_predicted_ev_bps
        if long_ok and short_ok:
            direction = Direction.LONG if ev_long > ev_short else Direction.SHORT
        elif long_ok:
            direction = Direction.LONG
        elif short_ok:
            direction = Direction.SHORT
        else:
            direction = Direction.NO_TRADE
        bar = self.bars[decision_index]
        prediction = Prediction(
            timestamp=bar.timestamp,
            direction=direction,
            confidence=max(p_long, p_short),
            reference_price=bar.close,
            horizon_bars=cfg.horizon_bars,
        )
        return V2Decision(
            prediction=prediction,
            probability_long_positive=p_long,
            probability_short_positive=p_short,
            predicted_long_ev_bps=ev_long,
            predicted_short_ev_bps=ev_short,
        )
