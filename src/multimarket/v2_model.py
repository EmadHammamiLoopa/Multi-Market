from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Sequence

from .features import FEATURE_NAMES, build_feature_points
from .models import Direction, MarketBar, Prediction
from .v2_features import clean_feature_indices
from .v2_labels import build_economic_event
from .v2_volatility import build_causal_ewma_volatility_bps, volatility_scaled_barriers_bps


@dataclass(frozen=True, slots=True)
class V2Config:
    horizon_bars: int = 6
    volatility_span: int = 48
    volatility_min_observations: int = 24
    take_profit_vol_multiplier: float = 1.0
    stop_loss_vol_multiplier: float = 1.0
    round_trip_cost_bps: float = 2.0
    min_train_rows: int = 5000
    retrain_every: int = 500
    validation_fraction: float = 0.20
    min_positive_probability: float = 0.55
    min_predicted_ev_bps: float = 0.0
    embargo_bars: int = 6
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        if self.volatility_span <= 1:
            raise ValueError("volatility_span must be greater than 1")
        if self.volatility_min_observations <= 0:
            raise ValueError("volatility_min_observations must be positive")
        if self.take_profit_vol_multiplier <= 0 or self.stop_loss_vol_multiplier <= 0:
            raise ValueError("volatility multipliers must be positive")
        if self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps cannot be negative")
        if self.min_train_rows < 100:
            raise ValueError("min_train_rows must be at least 100")
        if self.retrain_every <= 0:
            raise ValueError("retrain_every must be positive")
        if not 0.05 <= self.validation_fraction <= 0.40:
            raise ValueError("validation_fraction must be in [0.05, 0.40]")
        if not 0.50 <= self.min_positive_probability <= 1.0:
            raise ValueError("min_positive_probability must be in [0.50, 1.0]")
        if self.embargo_bars < self.horizon_bars:
            raise ValueError("embargo_bars must be at least horizon_bars")


@dataclass(frozen=True, slots=True)
class V2TrainingPoint:
    bar_index: int
    future_end_index: int
    features: tuple[float, ...]
    long_positive: int
    short_positive: int
    long_net_bps: float
    short_net_bps: float


@dataclass(frozen=True, slots=True)
class V2Decision:
    prediction: Prediction
    probability_long_positive: float
    probability_short_positive: float
    predicted_long_ev_bps: float
    predicted_short_ev_bps: float


def _fit_classifier(X, y, random_state: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("V2 requires xgboost. Install with: python -m pip install -e '.[ml]'") from exc
    model = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.15,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def _fit_regressor(X, y, random_state: int):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("V2 requires xgboost. Install with: python -m pip install -e '.[ml]'") from exc
    model = XGBRegressor(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.15,
        reg_lambda=1.5,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def _calibrate_probability(raw_probability: float, validation_pairs: Sequence[tuple[float, int]]) -> float:
    if not validation_pairs:
        return raw_probability
    lower = max(0.0, raw_probability - 0.05)
    upper = min(1.0, raw_probability + 0.05)
    matches = [label for probability, label in validation_pairs if lower <= probability <= upper]
    if len(matches) < 30:
        return raw_probability
    return min(0.99, max(0.01, sum(matches) / len(matches)))


class V2PriceOnlyPredictor:
    """Causal expanding-window V2 model with side-specific probability and EV heads."""

    def __init__(
        self,
        bars: Sequence[MarketBar],
        config: V2Config,
        *,
        eligible_indices: Collection[int],
    ) -> None:
        self.bars = bars
        self.config = config
        self.eligible_indices = set(eligible_indices)
        self.feature_points = build_feature_points(bars)
        self.feature_by_index = {point.bar_index: point for point in self.feature_points}
        self.clean_feature_indices = clean_feature_indices(
            bars,
            eligible_indices=self.eligible_indices,
        )
        self.volatility = build_causal_ewma_volatility_bps(
            bars,
            eligible_indices=self.eligible_indices,
            span=config.volatility_span,
            min_observations=config.volatility_min_observations,
        )
        self.training_points = self._build_training_points()
        self._models = None
        self._model_trained_through = -1
        self._long_validation: list[tuple[float, int]] = []
        self._short_validation: list[tuple[float, int]] = []
        self._fit_count = 0

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def _build_training_points(self) -> list[V2TrainingPoint]:
        result: list[V2TrainingPoint] = []
        for point in self.feature_points:
            index = point.bar_index
            if index not in self.clean_feature_indices:
                continue
            vol = self.volatility[index]
            if vol is None or vol <= 0:
                continue
            tp_bps, sl_bps = volatility_scaled_barriers_bps(
                vol,
                horizon_bars=self.config.horizon_bars,
                take_profit_vol_multiplier=self.config.take_profit_vol_multiplier,
                stop_loss_vol_multiplier=self.config.stop_loss_vol_multiplier,
            )
            event = build_economic_event(
                self.bars,
                index,
                horizon_bars=self.config.horizon_bars,
                take_profit_bps=tp_bps,
                stop_loss_bps=sl_bps,
                round_trip_cost_bps=self.config.round_trip_cost_bps,
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
        if (
            self._models is not None
            and decision_index >= self._model_trained_through
            and decision_index - self._model_trained_through < self.config.retrain_every
        ):
            return

        cutoff = decision_index - self.config.embargo_bars
        eligible = [point for point in self.training_points if point.future_end_index <= cutoff]
        if len(eligible) < self.config.min_train_rows:
            raise ValueError(
                f"V2 needs at least {self.config.min_train_rows} labeled historical rows; only {len(eligible)} are available at this timestamp"
            )

        split = max(self.config.min_train_rows, int(len(eligible) * (1.0 - self.config.validation_fraction)))
        split = min(split, len(eligible) - 1)
        train = eligible[:split]
        validation = eligible[split:]

        X_train = [list(point.features) for point in train]
        long_y = [point.long_positive for point in train]
        short_y = [point.short_positive for point in train]
        long_ev = [point.long_net_bps for point in train]
        short_ev = [point.short_net_bps for point in train]

        self._models = (
            _fit_classifier(X_train, long_y, self.config.random_state),
            _fit_classifier(X_train, short_y, self.config.random_state + 1),
            _fit_regressor(X_train, long_ev, self.config.random_state + 2),
            _fit_regressor(X_train, short_ev, self.config.random_state + 3),
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
        if decision_index not in self.clean_feature_indices:
            raise ValueError("decision index does not have a clean contiguous feature history")
        point = self.feature_by_index.get(decision_index)
        if point is None:
            raise ValueError("decision index does not have enough feature history")
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

        long_ok = p_long >= self.config.min_positive_probability and ev_long > self.config.min_predicted_ev_bps
        short_ok = p_short >= self.config.min_positive_probability and ev_short > self.config.min_predicted_ev_bps
        if long_ok and short_ok:
            direction = Direction.LONG if ev_long > ev_short else Direction.SHORT
        elif long_ok:
            direction = Direction.LONG
        elif short_ok:
            direction = Direction.SHORT
        else:
            direction = Direction.NO_TRADE

        confidence = max(p_long, p_short)
        bar = self.bars[decision_index]
        prediction = Prediction(
            timestamp=bar.timestamp,
            direction=direction,
            confidence=confidence,
            reference_price=bar.close,
            horizon_bars=self.config.horizon_bars,
        )
        return V2Decision(
            prediction=prediction,
            probability_long_positive=p_long,
            probability_short_positive=p_short,
            predicted_long_ev_bps=ev_long,
            predicted_short_ev_bps=ev_short,
        )
