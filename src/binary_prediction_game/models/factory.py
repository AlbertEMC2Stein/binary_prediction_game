"""Factory for constructing the active predictor set."""

from __future__ import annotations

from binary_prediction_game.models.base import BinaryPredictor
from binary_prediction_game.models.eSPA import eSPAPredictor
from binary_prediction_game.models.frequency import GlobalFrequencyPredictor
from binary_prediction_game.models.hopfield import HopfieldNetworkPredictor
from binary_prediction_game.models.ngram import NGramPredictor
from binary_prediction_game.models.sklearn_predictors import (
    OnlineLogisticRegressionPredictor,
    OnlineNeuralNetworkPredictor,
)
from binary_prediction_game.models.utils import validate_positive_int

PRESET_NGRAMS = (1, 2, 5)


def build_predictors(
    l_past: int,
    *,
    random_state: int | None = 0,
    include_eSPA: bool = True,
) -> list[BinaryPredictor]:
    """Construct the predictor list for a freshly reset run.

    The variable L-gram is omitted when L is already one of the fixed preset
    contexts 1, 2, or 5.
    """

    validate_positive_int(l_past, "l_past")

    predictors: list[BinaryPredictor] = [
        GlobalFrequencyPredictor(),
        NGramPredictor(1),
        NGramPredictor(2),
        NGramPredictor(5),
    ]

    if l_past not in PRESET_NGRAMS:
        predictors.append(NGramPredictor(l_past, name=f"{l_past}-gram"))

    predictors.extend(
        [
            OnlineLogisticRegressionPredictor(l_past, random_state=random_state),
            OnlineNeuralNetworkPredictor(l_past, random_state=random_state),
            HopfieldNetworkPredictor(l_past),
        ]
    )

    if include_eSPA:
        predictors.append(eSPAPredictor(l_past))

    return predictors
