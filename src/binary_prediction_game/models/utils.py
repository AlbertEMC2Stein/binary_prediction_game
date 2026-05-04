"""Utility functions shared by predictor implementations."""

from __future__ import annotations

from collections.abc import Sequence

from binary_prediction_game.models.base import Bit, History, Prediction


def validate_bit(bit: Bit) -> None:
    """Raise ``ValueError`` if ``bit`` is not binary."""

    if bit not in (0, 1):
        raise ValueError(f"Expected a binary bit 0 or 1, got {bit!r}.")


def validate_positive_int(value: int, name: str) -> None:
    """Raise ``ValueError`` if ``value`` is not strictly positive."""

    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value!r}.")


def majority_prediction(
    zero_count: int,
    one_count: int,
    *,
    tie_breaker: Bit = 0,
) -> Prediction:
    """Return the majority bit and empirical confidence.

    In a tie or with no observations, the configured tie breaker is returned
    with confidence 0.5.
    """

    validate_bit(tie_breaker)
    total = zero_count + one_count
    if total <= 0 or zero_count == one_count:
        return Prediction(bit=tie_breaker, confidence=0.5)

    if one_count > zero_count:
        return Prediction(bit=1, confidence=one_count / total)

    return Prediction(bit=0, confidence=zero_count / total)


def recent_bit_features(
    history_at_prediction_time: History,
    l_past: int,
    *,
    padding_value: float = 0.5,
) -> list[float]:
    """Return the last ``l_past`` bits as fixed-length numeric features.

    Early in a run, when fewer than ``l_past`` bits exist, the feature vector is
    left-padded with ``padding_value``. This keeps the feature dimension fixed
    for scikit-learn estimators.
    """

    validate_positive_int(l_past, "l_past")
    if len(history_at_prediction_time) >= l_past:
        window: Sequence[Bit | float] = history_at_prediction_time[-l_past:]
    else:
        missing = l_past - len(history_at_prediction_time)
        window = [padding_value] * missing + list(history_at_prediction_time)

    return [float(value) for value in window]
