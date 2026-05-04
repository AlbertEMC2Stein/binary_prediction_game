"""Loading and storing named binary benchmark sequences."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from binary_prediction_game import config
from binary_prediction_game.models.base import Bit


class SequenceLoadError(ValueError):
    """Raised when a bit-sequence file cannot be parsed or validated."""


class UsernameValidationError(ValueError):
    """Raised when a username is not permitted for storing a sequence."""


@dataclass(frozen=True)
class LoadedBitSequence:
    """Validated binary sequence loaded from disk.

    ``horizon`` and ``l_past`` are optional because plain text files and some
    built-in YAML files may only contain a sequence. When present, they describe
    the model settings that should be used when benchmarking this sequence.
    """

    bits: tuple[Bit, ...]
    description: str
    path: Path
    origin: str = "loaded"
    horizon: int | None = None
    l_past: int | None = None


_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_BIT_PATTERN = re.compile(r"^[01]+$")
_SUPPORTED_SUFFIXES = {".txt", ".yaml", ".yml"}


def validate_bit_string(raw_sequence: object) -> str:
    """Validate and normalize a single-line bit string."""

    if not isinstance(raw_sequence, str):
        raise SequenceLoadError("The sequence must be stored as a string.")

    sequence = raw_sequence.strip()
    if not sequence:
        raise SequenceLoadError("The sequence is empty.")

    if not _BIT_PATTERN.fullmatch(sequence):
        raise SequenceLoadError("The sequence may only contain the characters 0 and 1.")

    return sequence


def bits_from_string(sequence: str) -> tuple[Bit, ...]:
    """Convert a validated bit string into integer bits."""

    return tuple(int(character) for character in sequence)


def bit_string_from_bits(bits: Iterable[Bit]) -> str:
    """Convert integer bits into a compact bit string."""

    characters: list[str] = []
    for bit in bits:
        if bit not in (0, 1):
            raise SequenceLoadError(f"Invalid bit {bit!r}; expected 0 or 1.")
        characters.append(str(bit))
    if not characters:
        raise SequenceLoadError("Cannot store an empty sequence.")
    return "".join(characters)


def load_bit_sequence(path: str | Path) -> LoadedBitSequence:
    """Load a validated bit sequence from .txt, .yaml, or .yml."""

    sequence_path = Path(path)
    if not sequence_path.exists():
        raise SequenceLoadError(f"File does not exist: {sequence_path}")
    if sequence_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise SequenceLoadError("Only .txt, .yaml and .yml files are supported.")

    horizon = None
    l_past = None

    if sequence_path.suffix.lower() == ".txt":
        sequence = validate_bit_string(sequence_path.read_text(encoding="utf-8"))
        description = sequence_path.stem
    else:
        payload = _read_yaml(sequence_path)
        if not isinstance(payload, dict):
            raise SequenceLoadError("The YAML file must contain a mapping/object.")

        sequence = validate_bit_string(payload.get("sequence"))
        description = _optional_string(
            payload.get("description"), default=sequence_path.stem
        )
        horizon = _optional_int_in_range(
            _first_present(payload, "horizon", "h"),
            name="horizon",
            minimum=config.HORIZON_MIN,
            maximum=config.HORIZON_MAX,
        )
        l_past = _optional_int_in_range(
            _first_present(payload, "l_past", "L", "l"),
            name="l_past",
            minimum=config.L_PAST_MIN,
            maximum=config.L_PAST_MAX,
        )

    return LoadedBitSequence(
        bits=bits_from_string(sequence),
        description=description,
        path=sequence_path,
        origin="loaded",
        horizon=horizon,
        l_past=l_past,
    )


def _first_present(payload: dict[str, Any], *keys: str) -> object:
    """Return the first non-None value among several possible YAML keys."""

    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _optional_int_in_range(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    """Validate an optional integer YAML setting."""

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, int):
        raise SequenceLoadError(f"{name} must be an integer if provided.")

    if not minimum <= value <= maximum:
        raise SequenceLoadError(
            f"{name} must lie in [{minimum}, {maximum}], got {value}."
        )

    return value


def list_builtin_sequences() -> list[Path]:
    """Return available built-in sequence files."""

    return _list_sequence_files_in_folder(config.BUILTIN_SEQUENCE_DIR)


def list_user_sequences_by_leaderboard() -> list[Path]:
    """Return saved user sequence files ordered by the leaderboard ranking."""

    paths: list[Path] = []
    seen: set[Path] = set()

    for row in load_leaderboard():
        path = row.get("_path")
        if not isinstance(path, Path):
            continue

        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue

        seen.add(resolved)
        paths.append(path)

    return paths


def _list_sequence_files_in_folder(folder: Path) -> list[Path]:
    """Return supported sequence files from a single folder."""

    if not folder.exists():
        return []

    paths: list[Path] = []
    seen: set[Path] = set()

    for suffix in sorted(_SUPPORTED_SUFFIXES):
        for path in sorted(folder.glob(f"*{suffix}")):
            resolved = path.resolve()
            if resolved in seen:
                continue

            seen.add(resolved)
            paths.append(path)

    return paths


def list_available_sequences() -> list[Path]:
    """Return available built-in and user sequence files."""

    return [
        *list_builtin_sequences(),
        *list_user_sequences_by_leaderboard(),
    ]


def bits_needed_for_store(bits_count: int) -> int:
    """Return how many additional manual bits are required before saving."""

    return max(0, config.MIN_BITS_REQUIRED_FOR_STORE - bits_count)


def validate_username(username: str) -> str:
    """Validate a username for use inside saved file names."""

    normalized = username.strip()
    if not normalized:
        raise UsernameValidationError("Please enter a username before saving.")
    if not _USERNAME_PATTERN.fullmatch(normalized):
        raise UsernameValidationError(
            "Only letters, digits, underscores and hyphens are allowed."
        )
    if len(normalized) > config.USERNAME_MAX_LENGTH:
        raise UsernameValidationError(
            f"Usernames cannot be longer than {config.USERNAME_MAX_LENGTH} characters."
        )

    lower_name = normalized.lower()
    for bad_word in _load_bad_words(config.USERNAME_BAD_WORDS_FILE):
        if bad_word and bad_word in lower_name:
            censored_word = bad_word[0] + "*" * (len(bad_word) - 2) + bad_word[-1]
            raise UsernameValidationError(
                f"This username is not permitted ({censored_word})."
            )

    return normalized


def save_user_sequence(
    *,
    bits: Iterable[Bit],
    username: str,
    horizon: int,
    l_past: int,
    randomness_score: float | None,
    model_scores: list[dict[str, Any]],
) -> Path:
    """Store a user-created sequence and associated model metadata as YAML."""

    sequence = bit_string_from_bits(bits)
    sequence_length = len(sequence)
    if sequence_length < config.MIN_BITS_REQUIRED_FOR_STORE:
        raise SequenceLoadError(
            f"At least {config.MIN_BITS_REQUIRED_FOR_STORE} bits are required."
        )

    safe_username = validate_username(username)
    score_value = _score_for_filename(randomness_score)
    timestamp = datetime.now().strftime("%d%m%y_%H%M")

    config.USER_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.USER_SEQUENCE_DIR / (
        f"h{horizon}_L{l_past}_{score_value:03d}_{sequence_length}_"
        f"{safe_username}_{timestamp}.yaml"
    )

    payload = {
        "description": f"User sequence by {safe_username}",
        "origin": "user",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "username": safe_username,
        "horizon": horizon,
        "l_past": l_past,
        "sequence_length": sequence_length,
        "randomness_score": randomness_score,
        "model_scores": model_scores,
        "sequence": sequence,
    }

    _write_yaml(output_path, payload)
    return output_path


def load_leaderboard() -> list[dict[str, Any]]:
    """Load leaderboard rows from saved user sequence YAML files."""

    folder = config.USER_SEQUENCE_DIR

    leaderboard: list[dict[str, Any]] = []
    seen: set[Path] = set()
    if not folder.exists():
        return leaderboard
    for path in folder.glob("*.yaml"):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            data = _read_yaml(path)
        except Exception:
            continue

        if isinstance(data, dict) and _is_leaderboard_entry(data):
            row = dict(data)
            row["_path"] = path
            leaderboard.append(row)

    return sorted(
        leaderboard,
        key=lambda row: (
            int(row.get("horizon", 0)),
            -int(row.get("l_past", 0)),
            -float(row.get("randomness_score", 0.0) or 0.0),
            -int(row.get("sequence_length", 0)),
        ),
    )


def _is_leaderboard_entry(data: dict[str, Any]) -> bool:
    """Return whether a YAML payload has enough data for the leaderboard."""

    required_keys = {
        "username",
        "horizon",
        "l_past",
        "sequence_length",
        "randomness_score",
    }
    return required_keys.issubset(data.keys())


def _read_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as error:
        raise SequenceLoadError(
            "Loading YAML files requires PyYAML. Install it with: pip install PyYAML"
        ) from error

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as error:
        raise SequenceLoadError(
            "Saving YAML files requires PyYAML. Install it with: pip install PyYAML"
        ) from error

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)


def _optional_string(value: object, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise SequenceLoadError("The description must be a string if provided.")
    return value.strip() or default


def _load_bad_words(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()

    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().lower()
        if stripped and not stripped.startswith("#"):
            words.append(stripped)
    return tuple(words)


def _score_for_filename(randomness_score: float | None) -> int:
    if randomness_score is None:
        return 0
    clamped = max(0.0, min(1.0, randomness_score))
    return int(round(100.0 * clamped))
