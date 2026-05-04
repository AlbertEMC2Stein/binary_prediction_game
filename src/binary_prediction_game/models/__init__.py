"""Prediction models for the binary prediction game.

The package root intentionally imports only lightweight interface objects. The
scikit-learn based predictors are loaded lazily when requested.
"""

from __future__ import annotations

from binary_prediction_game.models.base import BinaryPredictor, Prediction

__all__ = [
    "BinaryPredictor",
    "eSPAPredictor",
    "GlobalFrequencyPredictor",
    "HopfieldNetworkPredictor",
    "NGramPredictor",
    "OnlineLogisticRegressionPredictor",
    "OnlineNeuralNetworkPredictor",
    "Prediction",
    "build_predictors",
]


def __getattr__(name: str) -> object:
    if name == "eSPAPredictor":
        from binary_prediction_game.models.eSPA import eSPAPredictor

        return eSPAPredictor

    if name == "GlobalFrequencyPredictor":
        from binary_prediction_game.models.frequency import GlobalFrequencyPredictor

        return GlobalFrequencyPredictor

    if name == "HopfieldNetworkPredictor":
        from binary_prediction_game.models.hopfield import HopfieldNetworkPredictor

        return HopfieldNetworkPredictor

    if name == "NGramPredictor":
        from binary_prediction_game.models.ngram import NGramPredictor

        return NGramPredictor

    if name == "OnlineLogisticRegressionPredictor":
        from binary_prediction_game.models.sklearn_predictors import (
            OnlineLogisticRegressionPredictor,
        )

        return OnlineLogisticRegressionPredictor

    if name == "OnlineNeuralNetworkPredictor":
        from binary_prediction_game.models.sklearn_predictors import (
            OnlineNeuralNetworkPredictor,
        )

        return OnlineNeuralNetworkPredictor

    if name == "build_predictors":
        from binary_prediction_game.models.factory import build_predictors

        return build_predictors

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
