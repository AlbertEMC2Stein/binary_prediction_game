# RPTU: Tag der Mathematik 2026 - Binary Prediction Game

Interactive Pygame demonstration for teaching randomness, stochastic prediction, and binary sequence modelling.

Users enter a sequence of `0` and `1` inputs. Several models try to predict the next visible bit while only receiving a horizon-adjusted part of the past. The application displays the revealed predictions, model accuracies, a randomness score, benchmark sequence loading, and a leaderboard for saved user sequences.

## Contents

- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Run from source](#run-from-source)
- [Ubuntu/Linux notes](#ubuntulinux-notes)
- [Data folders](#data-folders)
- [Sequence file formats](#sequence-file-formats)
- [Controls](#controls)
- [Packaging on Ubuntu/Linux](#packaging-on-ubuntulinux)
- [Building Windows and macOS versions](#building-windows-and-macos-versions)
- [Cleaning the source tree](#cleaning-the-source-tree)
- [Troubleshooting](#troubleshooting)

## Project layout

Expected layout:

```text
Binary-Prediction-Game/
    README.md
    pyproject.toml
    run_game.sh
    scripts/
        bpg_entry.py
    data/
        bad_words.txt
        built-in-sequences/
        user-sequences/
    src/
        binary_prediction_game/
            __init__.py
            __main__.py
            app.py
            config.py
            game_state.py
            sequence_io.py
            models/
                __init__.py
                base.py
                eSPA.py
                factory.py
                frequency.py
                hopfield.py
                ngram.py
                sklearn_predictors.py
                utils.py
            ui/
                components.py
                layout.py
                theme.py
```

`data/built-in-sequences/` contains predefined benchmark sequences.

`data/user-sequences/` contains saved user attempts and leaderboard entries.

## Requirements

Recommended Python version:

```text
Python >= 3.11
```

The main runtime dependencies are:

```text
pygame
numpy
scikit-learn
PyYAML
```

They should be listed in `pyproject.toml`, for example:

```toml
[project]
dependencies = [
    "pygame>=2.5.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.4.0",
    "PyYAML>=6.0.0",
]
```

## Run from source

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m binary_prediction_game
```

Alternatively, if `run_game.sh` is present and executable:

```bash
chmod +x run_game.sh
./run_game.sh
```

## Ubuntu/Linux notes

On some Ubuntu installations, Pygame/scientific Python libraries may load the wrong `libstdc++.so.6`. If the app only starts with `LD_PRELOAD`, use the provided launcher script:

```bash
./run_game.sh
```

The script should contain something like:

```bash
#!/usr/bin/env bash
set -e

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python -m binary_prediction_game
```

If this path does not exist on your system, search for the correct library path:

```bash
find /usr -name libstdc++.so.6 2>/dev/null
```

Then adjust `run_game.sh` accordingly.

## Data folders

The application expects the following writable data structure:

```text
data/
    bad_words.txt
    built-in-sequences/
    user-sequences/
```

`bad_words.txt` should contain one forbidden username fragment per line. Empty lines and comment lines starting with `#` are ignored.

`built-in-sequences/` is for predefined benchmark files distributed with the program.

`user-sequences/` is for saved user results. Files in this folder are also used for the leaderboard and for the user-sequence section of the dropdown.

## Sequence file formats

### Plain text

A `.txt` sequence file contains only one sequence of bits:

```text
0100010111010010110
```

Only `0` and `1` are allowed.

### YAML

A `.yaml` or `.yml` sequence file can contain metadata:

```yaml
description: Example benchmark sequence
horizon: 1
l_past: 10
sequence: "0100010111010010110"
```

`horizon` and `l_past` are optional. If present, the application uses those values before benchmarking the sequence.

Saved user sequences are stored as YAML because they include additional metadata such as username, score, model settings, sequence length, and model accuracies.

## Controls

- Press `0` or `1` to add manual input bits.
- `horizon h` controls how far into the future the models need to predict.
- `context L` controls the context length of L-dependent models.
- `h` and `L` can only be changed before the first input after reset.
- **Reset** clears the current sequence and rebuilds all models.
- **RNG sequence** resets the game and feeds a pseudo-random benchmark sequence.
- Drag-and-drop `.txt`, `.yaml`, or `.yml` files onto the input panel to benchmark them.
- Use the sequence dropdown to load built-in or saved user sequences.
- Use **Save** after enough manual bits have been entered to store a user attempt.
- Use **Leaderboard** to display saved user results.

Undo is intentionally unsupported.

## Packaging on Ubuntu/Linux

PyInstaller is the recommended packaging tool. Build on the same operating system family that you want to distribute to.

Install build dependencies in a clean virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyinstaller
```

Build a folder-based Linux distribution:

```bash
python -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --name Binary-Prediction-Game \
  --paths src \
  --collect-all sklearn \
  --collect-all scipy \
  scripts/bpg_entry.py
```

Copy the writable data folder next to the executable:

```bash
cp -R data dist/Binary-Prediction-Game/data
```

Test the packaged app:

```bash
./dist/Binary-Prediction-Game/Binary-Prediction-Game
```

Create an archive for distribution:

```bash
tar -czf Binary-Prediction-Game-linux-x86_64.tar.gz -C dist Binary-Prediction-Game
```

Distribute the whole `Binary-Prediction-Game/` folder or the `.tar.gz` archive, not just the executable file. The application needs the adjacent `data/` folder for built-in sequences, saved user sequences, and the bad-word list.

## Building Windows and macOS versions

PyInstaller is not a cross-compiler. A Windows `.exe` should be built on Windows, and a macOS `.app` should be built on macOS.

For Windows, ask a colleague with Windows to run roughly:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyinstaller
python -m PyInstaller `
  --noconfirm `
  --clean `
  --onedir `
  --windowed `
  --name Binary-Prediction-Game `
  --paths src `
  --collect-all sklearn `
  --collect-all scipy `
  scripts\bpg_entry.py
xcopy /E /I /Y data dist\Binary-Prediction-Game\data
```

For macOS, ask a colleague with macOS to run roughly:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyinstaller
python -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --windowed \
  --name Binary-Prediction-Game \
  --osx-bundle-identifier de.rptu.Binary-Prediction-Game.binarypredictiongame \
  --paths src \
  --collect-all sklearn \
  --collect-all scipy \
  scripts/bpg_entry.py
cp -R data dist/data
```

The macOS output should be distributed together with the adjacent `data/` folder.

## Cleaning the source tree

Do not commit or distribute generated files:

```bash
rm -rf .venv
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
rm -rf build dist *.spec .pytest_cache .mypy_cache .ruff_cache
```

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.py[cod]
build/
dist/
*.spec
.pytest_cache/
.mypy_cache/
.ruff_cache/
```

Keep `data/built-in-sequences/` and `data/bad_words.txt` under version control. Decide whether `data/user-sequences/` should be versioned. For a live event, it is usually better to keep the folder but ignore generated user files.

## Troubleshooting

### `ModuleNotFoundError`

Reinstall the package in editable mode:

```bash
source .venv/bin/activate
python -m pip install -e .
```

### Pygame opens no window or crashes on Ubuntu

Try the launcher script with `LD_PRELOAD`:

```bash
./run_game.sh
```

### Packaged app cannot find built-in sequences

Make sure the folder structure is:

```text
dist/Binary-Prediction-Game/Binary-Prediction-Game
dist/Binary-Prediction-Game/data/built-in-sequences/
dist/Binary-Prediction-Game/data/user-sequences/
dist/Binary-Prediction-Game/data/bad_words.txt
```

### Packaged app starts but saving fails

Check that `data/user-sequences/` exists and is writable.