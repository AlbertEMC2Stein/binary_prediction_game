"""Game-state and model-evaluation logic for the binary prediction game."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from binary_prediction_game import config
from binary_prediction_game.models.base import Bit, BinaryPredictor, History, Prediction
from binary_prediction_game.models.factory import build_predictors
from binary_prediction_game.sequence_io import (
    LoadedBitSequence,
    bits_needed_for_store,
    load_bit_sequence,
    save_user_sequence,
)


@dataclass(frozen=True)
class PendingPrediction:
    """A hidden prediction waiting for the corresponding target bit."""

    prediction: Prediction
    context_at_prediction_time: tuple[Bit, ...]


@dataclass(frozen=True)
class RevealedPrediction:
    """A prediction after the target bit has been entered."""

    prediction: Prediction
    target_bit: Bit
    is_correct: bool


@dataclass
class PredictorScore:
    """Running score for one predictor."""

    correct: int = 0
    evaluated: int = 0
    accuracy_history: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float | None:
        """Return the cumulative accuracy, or ``None`` if unevaluated."""

        if self.evaluated == 0:
            return None
        return self.correct / self.evaluated

    def add_result(self, is_correct: bool) -> None:
        """Add one evaluated prediction."""

        self.evaluated += 1
        if is_correct:
            self.correct += 1
        self.accuracy_history.append(self.correct / self.evaluated)


class GameState:
    """Mutable game state connecting GUI inputs to the predictors.

    The models always predict the next visible input cell. The selected horizon
    changes only how much of the past they are allowed to see. After observing
    ``x_1, ..., x_t`` and with horizon ``h``, the context passed to the models is
    ``x_1, ..., x_{t-h+1}``, truncated to the empty history if this index is not
    positive yet.
    """

    def __init__(self) -> None:
        self.bits: list[Bit] = []
        self.horizon: int = config.HORIZON_DEFAULT
        self.l_past: int = config.L_PAST_DEFAULT
        self.status_message: str = "Press 0 or 1 to add bits."
        self.input_origin: str = "manual"
        self.loaded_sequence_description: str | None = None

        self.predictors: list[BinaryPredictor] = []
        self.scores: list[PredictorScore] = []
        self.revealed_predictions: list[list[RevealedPrediction | None]] = []
        self._pending_next_predictions: list[PendingPrediction] = []

        self._simulation_queue: deque[Bit] = deque()
        self._rng = random.Random()

        self._rebuild_predictors()
        self._prepare_next_predictions()

    @property
    def controls_locked(self) -> bool:
        """Whether horizon and L-past controls are locked."""

        return len(self.bits) > 0 or self.simulation_running

    @property
    def latest_bit(self) -> Bit | None:
        """Return the most recently entered bit, if any."""

        return self.bits[-1] if self.bits else None

    @property
    def simulation_running(self) -> bool:
        """Whether a queued sequence is currently being consumed."""

        return bool(self._simulation_queue)

    @property
    def save_eligible(self) -> bool:
        """Whether the current sequence may be saved as a user sequence."""

        return (
            self.input_origin == "manual"
            and not self.simulation_running
            and len(self.bits) >= config.MIN_BITS_REQUIRED_FOR_STORE
        )

    @property
    def remaining_bits_until_save(self) -> int:
        """Return how many manual bits are still required before saving."""

        if self.input_origin != "manual":
            return config.MIN_BITS_REQUIRED_FOR_STORE
        return bits_needed_for_store(len(self.bits))

    def reset(self) -> None:
        """Reset the sequence, scores, simulation queue, and all predictors."""

        self.bits.clear()
        self._simulation_queue.clear()
        self.input_origin = "manual"
        self.loaded_sequence_description = None
        self._rebuild_predictors()
        self._prepare_next_predictions()
        self.status_message = "Reset complete. Horizon and L are editable again."

    def set_horizon(self, value: int) -> None:
        """Set the horizon while the controls are unlocked."""

        if self.controls_locked:
            return
        self.horizon = max(config.HORIZON_MIN, min(config.HORIZON_MAX, value))
        self.status_message = f"Horizon set to h = {self.horizon}."
        self._prepare_next_predictions()

    def set_l_past(self, value: int) -> None:
        """Set the L-past value while the controls are unlocked."""

        if self.controls_locked:
            return
        self.l_past = max(config.L_PAST_MIN, min(config.L_PAST_MAX, value))
        self.status_message = f"L-past set to L = {self.l_past}."
        self._rebuild_predictors()
        self._prepare_next_predictions()

    def active_model_names(self) -> list[str]:
        """Return the currently active model names."""

        return [predictor.name for predictor in self.predictors]

    def latest_revealed_predictions(self) -> list[RevealedPrediction | None]:
        """Return the latest revealed prediction for each model."""

        if not self.revealed_predictions:
            return [None for _ in self.predictors]
        return self.revealed_predictions[-1]

    def append_bit(self, bit: Bit) -> None:
        """Append one bit and evaluate the pending predictions."""

        if bit not in (0, 1):
            return

        if len(self._pending_next_predictions) != len(self.predictors):
            self._prepare_next_predictions()

        target_bit = bit
        row: list[RevealedPrediction | None] = []

        for predictor, score, pending in zip(
            self.predictors,
            self.scores,
            self._pending_next_predictions,
            strict=True,
        ):
            is_correct = pending.prediction.bit == target_bit
            revealed = RevealedPrediction(
                prediction=pending.prediction,
                target_bit=target_bit,
                is_correct=is_correct,
            )
            row.append(revealed)
            score.add_result(is_correct)
            predictor.learn(pending.context_at_prediction_time, target_bit)

        self.bits.append(target_bit)
        self.revealed_predictions.append(row)
        self._prepare_next_predictions()
        self.status_message = f"Accepted input: {target_bit}"

    def rng_simulation(self, steps: int = config.DEFAULT_SIMULATION_STEPS) -> None:
        """Reset and queue a pseudo-random binary simulation."""

        bits = [self._rng.randint(0, 1) for _ in range(steps)]
        self.start_benchmark_sequence(
            bits,
            origin="rng",
            description=f"Python pseudo-random sequence ({steps} bits)",
        )

    def advance_simulation(
        self, max_steps: int = config.SIMULATION_STEPS_PER_FRAME
    ) -> None:
        """Consume a small batch of queued benchmark bits."""

        steps = min(max_steps, len(self._simulation_queue))
        for _ in range(steps):
            self.append_bit(self._simulation_queue.popleft())

        if not self._simulation_queue and self.bits:
            if self.input_origin == "manual":
                return
            self.status_message = (
                f"Read {len(self.bits)} bits from "
                f"{self.loaded_sequence_description or self.input_origin}."
            )

    def start_benchmark_sequence(
        self,
        bits: list[Bit] | tuple[Bit, ...],
        *,
        origin: str,
        description: str,
        horizon: int | None = None,
        l_past: int | None = None,
    ) -> None:
        """Reset and queue an externally supplied benchmark sequence.

        If a loaded YAML sequence specifies ``horizon`` and/or ``l_past``, these
        settings are applied before the predictors are rebuilt. This ensures the
        benchmark is evaluated with the same model configuration that was stored
        together with the sequence.
        """

        self.reset()

        settings_changed = False
        if horizon is not None and horizon != self.horizon:
            self.horizon = horizon
            settings_changed = True
        if l_past is not None and l_past != self.l_past:
            self.l_past = l_past
            settings_changed = True

        if settings_changed:
            self._rebuild_predictors()
            self._prepare_next_predictions()

        self.input_origin = origin
        self.loaded_sequence_description = description
        self._simulation_queue = deque(bits)

        settings_suffix = self._loaded_settings_suffix(horizon=horizon, l_past=l_past)
        self.status_message = f"Loaded {len(bits)} bits: {description}{settings_suffix}"

    def _loaded_settings_suffix(
        self, *, horizon: int | None, l_past: int | None
    ) -> str:
        """Return a compact status-message suffix for loaded sequence settings."""

        parts: list[str] = []
        if horizon is not None:
            parts.append(f"h={self.horizon}")
        if l_past is not None:
            parts.append(f"L={self.l_past}")

        if not parts:
            return ""

        return f" ({', '.join(parts)})"

    def load_sequence_file(self, path: str) -> LoadedBitSequence:
        """Load a .txt/.yaml sequence file and queue it as benchmark data."""

        loaded = load_bit_sequence(path)
        self.start_benchmark_sequence(
            list(loaded.bits),
            origin="loaded",
            description=loaded.description,
            horizon=loaded.horizon,
            l_past=loaded.l_past,
        )
        return loaded

    def save_current_user_sequence(self, username: str) -> str:
        """Save the current manual bit sequence as a YAML user sequence."""

        if self.input_origin != "manual":
            raise ValueError("Only manually entered sequences can be saved.")
        if self.simulation_running:
            raise ValueError("Cannot save while a benchmark sequence is still running.")

        output_path = save_user_sequence(
            bits=self.bits,
            username=username,
            horizon=self.horizon,
            l_past=self.l_past,
            randomness_score=self.randomness_score(),
            model_scores=self.model_score_snapshot(),
        )
        self.status_message = f"Saved sequence: {output_path.name}"
        return str(output_path)

    def model_score_snapshot(self) -> list[dict[str, object]]:
        """Return serializable model-score metadata for YAML output."""

        snapshot: list[dict[str, object]] = []
        for predictor, score in zip(self.predictors, self.scores, strict=True):
            snapshot.append(
                {
                    "name": predictor.name,
                    "correct": score.correct,
                    "evaluated": score.evaluated,
                    "accuracy": score.accuracy,
                }
            )
        return snapshot

    def randomness_score(self) -> float | None:
        """Return the score where 1 means random-like and 0 means predictable."""

        accuracies = [
            score.accuracy for score in self.scores if score.accuracy is not None
        ]
        if not accuracies:
            return None

        best_accuracy = max(accuracies)
        predictability = max(0.0, 2.0 * (best_accuracy - 0.5))
        return max(0.0, min(1.0, 1.0 - predictability))

    def _rebuild_predictors(self) -> None:
        self.predictors = build_predictors(self.l_past)
        for predictor in self.predictors:
            predictor.reset()
        self.scores = [PredictorScore() for _ in self.predictors]
        self.revealed_predictions = []
        self._pending_next_predictions = []

    def _prepare_next_predictions(self) -> None:
        context = self._context_for_next_prediction()
        self._pending_next_predictions = [
            PendingPrediction(
                prediction=predictor.predict(context),
                context_at_prediction_time=context,
            )
            for predictor in self.predictors
        ]

    def _context_for_next_prediction(self) -> tuple[Bit, ...]:
        context_length = max(0, len(self.bits) - self.horizon + 1)
        return tuple(self.bits[:context_length])
