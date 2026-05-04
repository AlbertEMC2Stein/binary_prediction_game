"""eSPA predictor.

This module contains a lightweight NumPy port of the MATLAB eSPA model used in
our department. The model is not online in the strict sense: every call to
``learn`` stores the newly revealed training example and refits the prototype
model on all examples seen so far.

The surrounding game state is responsible for applying the horizon convention.
Consequently, this predictor only sees the already horizon-adjusted context and
uses its last ``l_past`` entries as features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from binary_prediction_game.models.base import Bit, BinaryPredictor, History, Prediction
from binary_prediction_game.models.frequency import GlobalFrequencyPredictor
from binary_prediction_game.models.utils import (
    recent_bit_features,
    validate_bit,
    validate_positive_int,
)

_EPS = 1.0e-12
_NUM_CLASSES = 2


def _safe_log(values: np.ndarray) -> np.ndarray:
    """Return a numerically safe elementwise logarithm."""

    return np.log(np.maximum(values, _EPS))


def _normalise_columns(values: np.ndarray, fallback_value: float) -> np.ndarray:
    """Normalise each column, replacing empty columns by a fallback value."""

    column_sums = values.sum(axis=0, keepdims=True)
    result = np.divide(
        values,
        column_sums,
        out=np.full_like(values, fallback_value, dtype=float),
        where=column_sums > _EPS,
    )
    return result


def _one_hot_bits(targets: list[Bit]) -> np.ndarray:
    """Return a ``2 x T`` one-hot label matrix for binary targets."""

    labels = np.zeros((_NUM_CLASSES, len(targets)), dtype=float)
    for column, target in enumerate(targets):
        labels[target, column] = 1.0
    return labels


@dataclass(frozen=True)
class ESPASettings:
    """Hyperparameters for the small real-time eSPA fit.

    The MATLAB code is usually used with a grid search over ``K``, ``epsE`` and
    ``epsC``. That is too expensive for the interactive game, so this class
    fixes one conservative configuration.
    """

    max_clusters: int = 15
    eps_entropy_reg: float = 1.0e-2
    eps_bayes_reg: float = 1.0e-3
    max_iter: int = 25
    tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        validate_positive_int(self.max_clusters, "max_clusters")
        validate_positive_int(self.max_iter, "max_iter")
        if self.eps_entropy_reg <= 0.0:
            raise ValueError("eps_entropy_reg must be strictly positive.")
        if self.eps_bayes_reg < 0.0:
            raise ValueError("eps_bayes_reg must be non-negative.")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative.")


class _ESPAPrototypeClassifier:
    """Small NumPy implementation of the eSPA prototype classifier.

    The implementation follows the supplied MATLAB code closely:

    * ``W`` is a feature-weight vector on the simplex.
    * ``S`` contains ``K`` prototypes/centroids.
    * ``Gamma`` is a hard cluster assignment matrix.
    * ``Lambda`` contains class probabilities per cluster.
    """

    def __init__(self, settings: ESPASettings) -> None:
        self.settings = settings
        self.w: np.ndarray | None = None
        self.s: np.ndarray | None = None
        self.gamma: np.ndarray | None = None
        self.lambda_: np.ndarray | None = None

    def fit(self, x: np.ndarray, targets: list[Bit]) -> None:
        """Fit the model from scratch on all currently available examples.

        Parameters
        ----------
        x:
            Feature matrix with shape ``D x T``.
        targets:
            Binary target labels of length ``T``.
        """

        if x.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {x.shape!r}.")
        feature_count, sample_count = x.shape
        if sample_count == 0:
            self.w = None
            self.s = None
            self.gamma = None
            self.lambda_ = None
            return
        if len(targets) != sample_count:
            raise ValueError(
                "Number of targets must match the number of feature columns."
            )

        pi = _one_hot_bits(targets)
        cluster_count = min(self.settings.max_clusters, sample_count)

        self._initialise(x, feature_count, sample_count, cluster_count)

        previous_loss = np.inf
        x_squared = x**2
        for _ in range(self.settings.max_iter):
            self._w_step(x)
            self._gamma_step(x, x_squared, pi)
            self._s_step(x)
            self._lambda_step(pi)

            loss = self._loss(x, pi)
            improvement = previous_loss - loss
            if improvement <= self.settings.tolerance:
                break
            previous_loss = loss

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities for columns of ``x`` as a ``2 x T`` matrix."""

        if self.lambda_ is None or self.s is None or self.w is None:
            raise RuntimeError("Cannot predict before the eSPA model has been fitted.")
        gamma = self._compute_gamma(x)
        probabilities = self.lambda_ @ gamma
        return np.clip(probabilities, 0.0, 1.0)

    def _initialise(
        self,
        x: np.ndarray,
        feature_count: int,
        sample_count: int,
        cluster_count: int,
    ) -> None:
        """Initialise parameters deterministically from the current data set."""

        self.w = np.full(feature_count, 1.0 / feature_count, dtype=float)

        if cluster_count == 1:
            centroid_indices = np.array([sample_count - 1], dtype=int)
        else:
            centroid_indices = np.linspace(
                0,
                sample_count - 1,
                num=cluster_count,
                dtype=int,
            )
        self.s = x[:, centroid_indices].copy()

        self.lambda_ = np.full(
            (_NUM_CLASSES, cluster_count),
            1.0 / _NUM_CLASSES,
            dtype=float,
        )
        self.gamma = self._compute_gamma(x)

    def _loss(self, x: np.ndarray, pi: np.ndarray) -> float:
        assert self.w is not None
        assert self.s is not None
        assert self.gamma is not None
        assert self.lambda_ is not None

        sample_count = x.shape[1]
        reconstruction = x - self.s @ self.gamma
        weighted_reconstruction = np.sqrt(self.w)[:, np.newaxis] * reconstruction
        reconstruction_loss = float(np.sum(weighted_reconstruction**2) / sample_count)
        feature_entropy = float(
            self.settings.eps_entropy_reg * np.dot(self.w, _safe_log(self.w))
        )
        label_probabilities = self.lambda_ @ self.gamma
        label_loss = float(
            -self.settings.eps_bayes_reg
            / sample_count
            * np.sum(pi * _safe_log(label_probabilities))
        )
        return reconstruction_loss + feature_entropy + label_loss

    def _w_step(self, x: np.ndarray) -> None:
        assert self.s is not None
        assert self.gamma is not None

        sample_count = x.shape[1]
        residual = x - self.s @ self.gamma
        raw_weights = np.sum(residual**2, axis=1)
        raw_weights = raw_weights / (sample_count * self.settings.eps_entropy_reg)

        shifted = -raw_weights + np.min(raw_weights)
        weights = np.exp(shifted)
        weight_sum = float(np.sum(weights))
        if weight_sum <= _EPS:
            self.w = np.full_like(weights, 1.0 / len(weights), dtype=float)
        else:
            self.w = weights / weight_sum

    def _gamma_step(self, x: np.ndarray, x_squared: np.ndarray, pi: np.ndarray) -> None:
        assert self.w is not None
        assert self.s is not None
        assert self.lambda_ is not None

        distance = self._weighted_squared_distances(x, x_squared)
        label_term = -self.settings.eps_bayes_reg * _safe_log(self.lambda_.T) @ pi
        self.gamma = self._assign_clusters(distance + label_term)

    def _s_step(self, x: np.ndarray) -> None:
        assert self.s is not None
        assert self.gamma is not None

        assignments = np.argmax(self.gamma, axis=0)
        for cluster in range(self.s.shape[1]):
            mask = assignments == cluster
            if np.any(mask):
                self.s[:, cluster] = np.mean(x[:, mask], axis=1)

    def _lambda_step(self, pi: np.ndarray) -> None:
        assert self.gamma is not None

        label_counts = pi @ self.gamma.T
        self.lambda_ = _normalise_columns(
            label_counts,
            fallback_value=1.0 / _NUM_CLASSES,
        )

    def _compute_gamma(self, x: np.ndarray) -> np.ndarray:
        assert self.w is not None
        assert self.s is not None

        distance = self._weighted_squared_distances(x, x**2)
        return self._assign_clusters(distance)

    def _weighted_squared_distances(
        self,
        x: np.ndarray,
        x_squared: np.ndarray,
    ) -> np.ndarray:
        assert self.w is not None
        assert self.s is not None

        centroid_norms = (self.s**2).T @ self.w
        sample_norms = self.w @ x_squared
        cross_terms = 2.0 * (self.s * self.w[:, np.newaxis]).T @ x
        return centroid_norms[:, np.newaxis] + sample_norms[np.newaxis, :] - cross_terms

    @staticmethod
    def _assign_clusters(scores: np.ndarray) -> np.ndarray:
        assignments = np.argmin(scores, axis=0)
        gamma = np.zeros((scores.shape[0], scores.shape[1]), dtype=float)
        gamma[assignments, np.arange(scores.shape[1])] = 1.0
        return gamma


class eSPAPredictor(BinaryPredictor):
    """L-past eSPA.

    The model stores all revealed training examples. Whenever ``learn`` is
    called, it extracts the last ``l_past`` bits of the supplied
    horizon-adjusted history, appends the new labelled example, and retrains the
    eSPA classifier on the full stored data set.
    """

    def __init__(
        self,
        l_past: int,
        *,
        name: str | None = None,
        settings: ESPASettings | None = None,
    ) -> None:
        validate_positive_int(l_past, "l_past")
        self.l_past = l_past
        self.name = name or f"{l_past}-past eSPA"
        self.settings = settings or ESPASettings()
        self._fallback = GlobalFrequencyPredictor()
        self._features: list[list[float]] = []
        self._targets: list[Bit] = []
        self._model: _ESPAPrototypeClassifier | None = None

    def reset(self) -> None:
        self._fallback.reset()
        self._features.clear()
        self._targets.clear()
        self._model = None

    def predict(self, context_at_prediction_time: History) -> Prediction:
        if self._model is None:
            return self._fallback.predict(context_at_prediction_time)

        features = np.asarray(
            recent_bit_features(context_at_prediction_time, self.l_past),
            dtype=float,
        ).reshape(self.l_past, 1)
        probabilities = self._model.predict_proba(features)[:, 0]
        probability_one = float(probabilities[1])
        predicted_bit = int(probability_one >= 0.5)
        confidence = max(probability_one, 1.0 - probability_one)
        confidence = max(0.5, min(1.0, confidence))
        return Prediction(bit=predicted_bit, confidence=confidence)

    def learn(
        self,
        history_at_prediction_time: History,
        target_bit: Bit,
    ) -> None:
        validate_bit(target_bit)
        self._fallback.learn(history_at_prediction_time, target_bit)

        features = recent_bit_features(history_at_prediction_time, self.l_past)
        self._features.append(features)
        self._targets.append(target_bit)

        x = np.asarray(self._features, dtype=float).T
        model = _ESPAPrototypeClassifier(self.settings)
        model.fit(x, self._targets)
        self._model = model
