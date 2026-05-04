"""Global-frequency binary predictor."""

from __future__ import annotations

from binary_prediction_game.models.base import Bit, BinaryPredictor, History, Prediction
from binary_prediction_game.models.utils import majority_prediction, validate_bit


class GlobalFrequencyPredictor(BinaryPredictor):
    """Predict the historically more frequent target bit.

    This predictor does not use the supplied context directly. The context is
    still accepted to satisfy the common interface; the model only learns from
    the revealed target bits passed to :meth:`learn`.
    """

    def __init__(self, *, name: str = "Global frequency", tie_breaker: Bit = 0) -> None:
        validate_bit(tie_breaker)
        self.name = name
        self._tie_breaker = tie_breaker
        self._zero_count = 0
        self._one_count = 0

    def reset(self) -> None:
        self._zero_count = 0
        self._one_count = 0

    def predict(self, context_at_prediction_time: History) -> Prediction:
        return majority_prediction(
            self._zero_count,
            self._one_count,
            tie_breaker=self._tie_breaker,
        )

    def learn(
        self,
        history_at_prediction_time: History,
        target_bit: Bit,
    ) -> None:
        del history_at_prediction_time

        validate_bit(target_bit)
        if target_bit == 0:
            self._zero_count += 1
        else:
            self._one_count += 1
