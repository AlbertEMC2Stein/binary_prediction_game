"""Static configuration for the GUI prototype."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


# This function determines the root directory for runtime data, which can differ between development and frozen (packaged) modes.
def _runtime_root() -> Path:
    """Return the folder where runtime data should live."""

    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()

        # macOS .app:
        # dist/TdM.app/Contents/MacOS/TdM
        # We want dist/data next to TdM.app.
        if (
            sys.platform == "darwin"
            and executable.parent.name == "MacOS"
            and executable.parent.parent.name == "Contents"
        ):
            return executable.parents[2].parent

        # Windows/Linux:
        # dist/TdM/TdM.exe or dist/TdM/TdM
        # We want dist/TdM/data.
        return executable.parent

    # Development mode:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _runtime_root()

# GUI settings
WINDOW_TITLE = "Binary Prediction Game"
WINDOW_SIZE = (1440, 1000)
MIN_WINDOW_SIZE = (1200, 950)
TARGET_FPS = 60

# UI layout settings
INPUT_TAPE_VISIBLE_BITS = 100
TAPE_HORIZONTAL_PADDING = 25
TAPE_LABEL_GAP = 10
TAPE_LABEL_WIDTH = 190
TAPE_CELL_WIDTH = 15
TAPE_PREDICTION_CELL_SIZE = 11

CONTEXT_INDICATOR_RADIUS = 2
CONTEXT_INDICATOR_Y_OFFSET = 5
CONTEXT_ARROW_HEIGHT = 5
CONTEXT_ARROW_HEAD_SIZE = 2

SEQUENCE_DROPDOWN_MAX_OPTIONS = 8

# Simulation settings
DEFAULT_SIMULATION_STEPS = 1000
SIMULATION_STEPS_PER_FRAME = 1

# Prediction settings
HORIZON_MIN = 1
HORIZON_MAX = 5
HORIZON_DEFAULT = 1

L_PAST_MIN = 1
L_PAST_MAX = 20
L_PAST_DEFAULT = 10

# Preset n-gram lengths
PRESET_NGRAMS = (1, 2, 5)

# Data storage settings
DATA_ROOT = PROJECT_ROOT / "data"
BUILTIN_SEQUENCE_DIR = DATA_ROOT / "built-in-sequences"
USER_SEQUENCE_DIR = DATA_ROOT / "user-sequences"
SEQUENCE_SOURCE_DIRS = (BUILTIN_SEQUENCE_DIR, USER_SEQUENCE_DIR)

# Constraints for user-generated sequences and usernames
MIN_BITS_REQUIRED_FOR_STORE = 250
USERNAME_ALLOWED_CHARS = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)
USERNAME_BAD_WORDS_FILE = DATA_ROOT / "bad_words.txt"
USERNAME_MAX_LENGTH = 20

# Leaderboard display settings
LEADERBOARD_MAX_ROWS = 12


@dataclass(frozen=True)
class FontSpec:
    """Font sizes used by the GUI."""

    tiny: int = 14
    small: int = 18
    regular: int = 22
    large: int = 32
    huge: int = 81
