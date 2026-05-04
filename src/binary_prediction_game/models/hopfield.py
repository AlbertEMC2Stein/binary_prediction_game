"""Hopfield-style predictor for fixed-window binary sequence prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from binary_prediction_game.models.base import Bit, BinaryPredictor, History, Prediction
from binary_prediction_game.models.frequency import GlobalFrequencyPredictor
from binary_prediction_game.models.utils import (
    recent_bit_features,
    validate_bit,
    validate_positive_int,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HopfieldSettings:
    """Configuration for the Hopfield predictor.

    ``confidence_temperature`` controls how quickly energy differences are
    translated into confident predictions. Smaller values make the model more
    confident; larger values keep predictions closer to 0.5.
    """

    confidence_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.confidence_temperature <= 0.0:
            raise ValueError(
                "confidence_temperature must be strictly positive, "
                f"got {self.confidence_temperature!r}."
            )


class HopfieldNetworkPredictor(BinaryPredictor):
    """Associative-memory predictor using patterns of length ``L + 1``.

    Each labelled example is stored as the bipolar pattern

    ``(x_{t-L+1}, ..., x_t, x_{t+1})``.

    The first ``L`` entries are the horizon-adjusted context supplied by the
    game state, and the final entry is the revealed target bit. Early contexts
    are left-padded with neutral nodes. The resulting Hopfield network therefore
    has ``L + 1`` states/nodes.

    Prediction compares the Hopfield energies of the two candidate completions

    ``(context, 0)`` and ``(context, 1)``

    and chooses the lower-energy candidate.
    """

    def __init__(
        self,
        l_past: int,
        *,
        settings: HopfieldSettings | None = None,
    ) -> None:
        validate_positive_int(l_past, "l_past")
        self.l_past = l_past
        self.name = f"{l_past + 1}-state Hopfield network"
        self.settings = settings or HopfieldSettings()
        self._fallback = GlobalFrequencyPredictor()
        self._weights = self._empty_weight_matrix()
        self._stored_patterns = 0

    def reset(self) -> None:
        """Reset all stored associations."""

        self._fallback.reset()
        self._weights = self._empty_weight_matrix()
        self._stored_patterns = 0

    def predict(self, context_at_prediction_time: History) -> Prediction:
        """Predict the next bit by comparing candidate Hopfield energies."""

        if self._stored_patterns == 0:
            return self._fallback.predict(context_at_prediction_time)

        context_nodes = self._context_nodes(context_at_prediction_time)
        pattern_zero = self._candidate_pattern(context_nodes, target_bit=0)
        pattern_one = self._candidate_pattern(context_nodes, target_bit=1)

        energy_zero = self._energy(pattern_zero)
        energy_one = self._energy(pattern_one)

        if math.isclose(energy_zero, energy_one, rel_tol=1.0e-12, abs_tol=1.0e-12):
            return self._fallback.predict(context_at_prediction_time)

        predicted_bit = int(energy_one < energy_zero)
        confidence = self._confidence_from_energy_gap(abs(energy_zero - energy_one))
        return Prediction(bit=predicted_bit, confidence=confidence)

    def learn(
        self,
        history_at_prediction_time: History,
        target_bit: Bit,
    ) -> None:
        """Store the newly revealed ``L + 1``-state pattern."""

        validate_bit(target_bit)
        self._fallback.learn(history_at_prediction_time, target_bit)

        context_nodes = self._context_nodes(history_at_prediction_time)
        pattern = self._candidate_pattern(context_nodes, target_bit=target_bit)
        self._store_pattern(pattern)

    def _empty_weight_matrix(self) -> FloatArray:
        return np.zeros((self.l_past + 1, self.l_past + 1), dtype=np.float64)

    def _context_nodes(self, history_at_prediction_time: History) -> FloatArray:
        """Return the horizon-adjusted context as bipolar/neutral nodes.

        Observed zeros and ones are encoded as ``-1`` and ``+1``. Missing early
        context entries are encoded as ``0`` so that they do not contribute to
        the Hopfield energy.
        """

        features = recent_bit_features(
            history_at_prediction_time,
            self.l_past,
            padding_value=0.5,
        )
        return np.asarray(
            [self._feature_to_node(value) for value in features], dtype=np.float64
        )

    def _candidate_pattern(
        self, context_nodes: FloatArray, *, target_bit: Bit
    ) -> FloatArray:
        validate_bit(target_bit)
        target_node = 1.0 if target_bit == 1 else -1.0
        return np.concatenate(
            (context_nodes, np.asarray([target_node], dtype=np.float64))
        )

    def _store_pattern(self, pattern: FloatArray) -> None:
        """Update the normalized Hebbian weight matrix with one pattern."""

        outer = np.outer(pattern, pattern)
        np.fill_diagonal(outer, 0.0)

        old_weight = self._stored_patterns / (self._stored_patterns + 1)
        new_weight = 1.0 / (self._stored_patterns + 1)
        self._weights = old_weight * self._weights + new_weight * outer
        self._stored_patterns += 1

    def _energy(self, pattern: FloatArray) -> float:
        return float(-0.5 * pattern @ self._weights @ pattern)

    def _confidence_from_energy_gap(self, energy_gap: float) -> float:
        scaled_gap = min(energy_gap / self.settings.confidence_temperature, 60.0)
        return 1.0 / (1.0 + math.exp(-scaled_gap))

    @staticmethod
    def _feature_to_node(value: float) -> float:
        if value <= 0.0:
            return -1.0
        if value >= 1.0:
            return 1.0
        return 0.0
