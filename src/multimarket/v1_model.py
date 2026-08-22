from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .features import FEATURE_NAMES, FeaturePoint, build_feature_points
from .models import Direction, MarketBar, Prediction


@dataclass(frozen=True, slots=True)
class V1Config:
    horizon_bars: int = 6
    confidence_threshold: float = 0.60
    min_train_rows: int = 1000
    retrain_every: int = 250
    validation_fraction: float = 0.20
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        if not 0.5 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0.5, 1]")
        if self.min_train_rows < 100:
            raise ValueError("min_train_rows must be at least 100")
        if self.retrain_every <= 0:
            raise ValueError("retrain_every must be positive")
        if not 0.05 <= self.validation_fraction <= 0.40:
            raise ValueError("validation_fraction must be in [0.05, 0.40]")


@dataclass(frozen=True, slots=True)
class LabeledPoint:
    bar_index: int
    features: tuple[float, ...]
    label: int


def build_labeled_points(
    bars: Sequence[MarketBar], feature_points: Sequence[FeaturePoint], horizon_bars: int
) -> list[LabeledPoint]:
    result: list[LabeledPoint] = []
    for point in feature_points:
        future_index = point.bar_index + horizon_bars
        if future_index >= len(bars):
            continue
        current = bars[point.bar_index].close
        future = bars[future_index].close
        label = 1 if future > current else 0
        result.append(LabeledPoint(point.bar_index, point.values, label))
    return result


def _fit_xgb(X, y, random_state: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "V1 requires xgboost. Install with: python -m pip install -e '.[ml]'"
        ) from exc

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


def _calibrate_probability(raw_probability: float, validation_pairs: Sequence[tuple[float, int]]) -> float:
    """Simple leakage-safe calibration using only a historical validation tail."""
    confidence = max(raw_probability, 1.0 - raw_probability)
    if not validation_pairs:
        return confidence

    lower = max(0.50, confidence - 0.05)
    upper = min(1.00, confidence + 0.05)
    matching = []
    for probability, label in validation_pairs:
        predicted = 1 if probability >= 0.5 else 0
        item_conf = max(probability, 1.0 - probability)
        if lower <= item_conf <= upper:
            matching.append(1.0 if predicted == label else 0.0)
    if len(matching) < 15:
        return confidence
    empirical = sum(matching) / len(matching)
    return min(0.99, max(0.50, empirical))


class WalkForwardXGBoostPredictor:
    """Expanding-window V1 predictor with historical-only calibration."""

    def __init__(self, bars: Sequence[MarketBar], config: V1Config) -> None:
        self.bars = bars
        self.config = config
        self.feature_points = build_feature_points(bars)
        self.feature_by_index = {point.bar_index: point for point in self.feature_points}
        self.labeled = build_labeled_points(bars, self.feature_points, config.horizon_bars)
        self._model = None
        self._model_trained_through = -1
        self._validation_pairs: list[tuple[float, int]] = []
        self._fit_count = 0

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def _train_for_index(self, decision_index: int) -> None:
        # Reuse the cached model before scanning the full labeled history. This is
        # behavior-equivalent to the old ordering for monotonic walk-forward use,
        # but avoids an O(history) label scan on every bar between retrains.
        if (
            self._model is not None
            and decision_index >= self._model_trained_through
            and decision_index - self._model_trained_through < self.config.retrain_every
        ):
            return

        # A label is eligible only after its complete future horizon is already known.
        eligible = [
            point
            for point in self.labeled
            if point.bar_index + self.config.horizon_bars <= decision_index
        ]
        if len(eligible) < self.config.min_train_rows:
            raise ValueError(
                f"V1 needs at least {self.config.min_train_rows} labeled historical rows; "
                f"only {len(eligible)} are available at this timestamp"
            )

        split = max(
            self.config.min_train_rows,
            int(len(eligible) * (1.0 - self.config.validation_fraction)),
        )
        split = min(split, len(eligible) - 1)
        train = eligible[:split]
        validation = eligible[split:]

        X_train = [list(point.features) for point in train]
        y_train = [point.label for point in train]
        self._model = _fit_xgb(X_train, y_train, self.config.random_state)
        self._fit_count += 1

        self._validation_pairs = []
        if validation:
            probabilities = self._model.predict_proba(
                [list(point.features) for point in validation]
            )[:, 1]
            self._validation_pairs = [
                (float(probability), point.label)
                for probability, point in zip(probabilities, validation)
            ]
        self._model_trained_through = decision_index

    def predict_at_index(self, decision_index: int) -> Prediction:
        point = self.feature_by_index.get(decision_index)
        if point is None:
            raise ValueError("decision index does not have enough feature history")
        self._train_for_index(decision_index)
        assert self._model is not None
        probability_long = float(
            self._model.predict_proba([list(point.values)])[0][1]
        )
        direction = Direction.LONG if probability_long >= 0.5 else Direction.SHORT
        calibrated_confidence = _calibrate_probability(
            probability_long, self._validation_pairs
        )
        if calibrated_confidence < self.config.confidence_threshold:
            direction = Direction.NO_TRADE
        bar = self.bars[decision_index]
        return Prediction(
            timestamp=bar.timestamp,
            direction=direction,
            confidence=calibrated_confidence,
            reference_price=bar.close,
            horizon_bars=self.config.horizon_bars,
        )
