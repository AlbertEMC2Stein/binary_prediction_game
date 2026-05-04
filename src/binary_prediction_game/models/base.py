"""Common predictor interface for binary sequence prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

Bit = int
History = Sequence[Bit]


@dataclass(frozen=True)
class Prediction:
    """A binary prediction together with a confidence value.

    The confidence is the probability assigned to the predicted bit. A value of
    0.5 means that the predictor has no preference between zero and one.
    """

    bit: Bit
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.bit not in (0, 1):
            raise ValueError(f"Prediction bit must be 0 or 1, got {self.bit!r}.")
        if not 0.5 <= self.confidence <= 1.0:
            raise ValueError(
                "Prediction confidence must lie in [0.5, 1.0], "
                f"got {self.confidence!r}."
            )


class BinaryPredictor(ABC):
    """Abstract interface implemented by all binary predictors.

    The interface is horizon-aware. For horizon h, ``predict`` is called after
    observing x_1, ..., x_{t-h+1} and predicts x_{t+1}. Once the hidden target x_{t+1}
    has arrived, ``learn`` receives exactly the history that was available at
    prediction time and the now-revealed target bit.
    """

    name: str

    @abstractmethod
    def reset(self) -> None:
        """Reset the predictor to its initial state."""

    @abstractmethod
    def predict(self, context_at_prediction_time: History) -> Prediction:
        """Predict x_{t+1} from the context available h steps ago."""

    @abstractmethod
    def learn(
        self,
        history_at_prediction_time: History,
        target_bit: Bit,
    ) -> None:
        """Update the model based on the revealed target bit and the history h steps ago."""
