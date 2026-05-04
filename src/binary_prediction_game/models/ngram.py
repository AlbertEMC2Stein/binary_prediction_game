"""N-gram / finite-context binary predictors."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

from binary_prediction_game.models.base import Bit, BinaryPredictor, History, Prediction
from binary_prediction_game.models.frequency import GlobalFrequencyPredictor
from binary_prediction_game.models.utils import (
    majority_prediction,
    validate_bit,
    validate_positive_int,
)

ContextKey = tuple[Bit, ...]


class NGramPredictor(BinaryPredictor):
    """Predict from empirical continuation frequencies of a k-bit context.

    The game state is responsible for applying the horizon convention. Hence,
    this model receives the already horizon-adjusted context and simply learns
    mappings of the form

        last k bits of the supplied context -> revealed target bit.

    If fewer than k context bits are available, the model falls back to the
    global-frequency predictor.
    """

    def __init__(
        self,
        k: int,
        *,
        name: str | None = None,
        tie_breaker: Bit = 0,
    ) -> None:
        validate_positive_int(k, "k")
        validate_bit(tie_breaker)
        self.k = k
        self.name = name or f"{k}-gram"
        self._tie_breaker = tie_breaker
        self._counts: DefaultDict[ContextKey, list[int]] = defaultdict(lambda: [0, 0])
        self._fallback = GlobalFrequencyPredictor(tie_breaker=tie_breaker)

    def reset(self) -> None:
        self._counts.clear()
        self._fallback.reset()

    def predict(self, context_at_prediction_time: History) -> Prediction:
        if len(context_at_prediction_time) < self.k:
            return self._fallback.predict(context_at_prediction_time)

        context = tuple(context_at_prediction_time[-self.k :])
        counts = self._counts.get(context)
        if counts is None:
            return self._fallback.predict(context_at_prediction_time)

        zero_count, one_count = counts
        return majority_prediction(
            zero_count,
            one_count,
            tie_breaker=self._tie_breaker,
        )

    def learn(
        self,
        history_at_prediction_time: History,
        target_bit: Bit,
    ) -> None:
        validate_bit(target_bit)
        self._fallback.learn(history_at_prediction_time, target_bit)

        if len(history_at_prediction_time) < self.k:
            return

        context = tuple(history_at_prediction_time[-self.k :])
        counts = self._counts[context]
        counts[target_bit] += 1
