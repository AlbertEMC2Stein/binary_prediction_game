"""scikit-learn based online predictors."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from binary_prediction_game.models.base import Bit, BinaryPredictor, History, Prediction
from binary_prediction_game.models.frequency import GlobalFrequencyPredictor
from binary_prediction_game.models.utils import (
    recent_bit_features,
    validate_bit,
    validate_positive_int,
)

CLASSES = [0, 1]


class SklearnOnlinePredictor(BinaryPredictor):
    """Base class for fixed-window online scikit-learn predictors.

    The supplied context is already horizon-adjusted by the game state. This
    class therefore only extracts the last ``l_past`` entries of that context
    and learns the revealed target bit.
    """

    def __init__(self, l_past: int, *, name: str, random_state: int | None = 0) -> None:
        validate_positive_int(l_past, "l_past")
        self.l_past = l_past
        self.name = name
        self.random_state = random_state
        self._estimator = self._make_estimator()
        self._is_fitted = False
        self._fallback = GlobalFrequencyPredictor()

    def reset(self) -> None:
        self._estimator = self._make_estimator()
        self._is_fitted = False
        self._fallback.reset()

    def predict(self, context_at_prediction_time: History) -> Prediction:
        if not self._is_fitted:
            return self._fallback.predict(context_at_prediction_time)

        features = recent_bit_features(context_at_prediction_time, self.l_past)
        probabilities = self._estimator.predict_proba([features])[0]
        probability_one = float(probabilities[1])
        predicted_bit = int(probability_one >= 0.5)
        confidence = max(probability_one, 1.0 - probability_one)
        return Prediction(bit=predicted_bit, confidence=confidence)

    def learn(
        self,
        history_at_prediction_time: History,
        target_bit: Bit,
    ) -> None:
        validate_bit(target_bit)
        self._fallback.learn(history_at_prediction_time, target_bit)

        features = recent_bit_features(history_at_prediction_time, self.l_past)
        if self._is_fitted:
            self._estimator.partial_fit([features], [target_bit])
        else:
            self._estimator.partial_fit([features], [target_bit], classes=CLASSES)
            self._is_fitted = True

    @abstractmethod
    def _make_estimator(self) -> Any:
        """Create a fresh scikit-learn estimator."""


class OnlineLogisticRegressionPredictor(SklearnOnlinePredictor):
    """Online logistic regression using ``SGDClassifier`` with log loss."""

    def __init__(self, l_past: int, *, random_state: int | None = 0) -> None:
        super().__init__(
            l_past,
            name=f"{l_past}-past logistic regression",
            random_state=random_state,
        )

    def _make_estimator(self) -> SGDClassifier:
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1.0e-4,
            learning_rate="constant",
            eta0=0.05,
            random_state=self.random_state,
        )


class OnlineNeuralNetworkPredictor(SklearnOnlinePredictor):
    """Small online neural network using scikit-learn's ``MLPClassifier``."""

    def __init__(self, l_past: int, *, random_state: int | None = 0) -> None:
        super().__init__(
            l_past,
            name=f"{l_past}-past neural network",
            random_state=random_state,
        )

    def _make_estimator(self) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=(6, 6, 6),
            activation="relu",
            solver="adam",
            alpha=1.0e-4,
            learning_rate_init=0.01,
            random_state=self.random_state,
            max_iter=1,
            # warm_start=True,
        )
