"""Pygame GUI for the binary prediction game."""

from __future__ import annotations

import sys
import os
import subprocess
from pathlib import Path
from typing import Any

import pygame

from binary_prediction_game import config
from binary_prediction_game.game_state import GameState, RevealedPrediction
from binary_prediction_game.sequence_io import (
    SequenceLoadError,
    UsernameValidationError,
    list_builtin_sequences,
    list_user_sequences_by_leaderboard,
    load_leaderboard,
)
from binary_prediction_game.ui.components import (
    Button,
    Dropdown,
    Fonts,
    NumberInput,
    draw_panel,
)
from binary_prediction_game.ui.layout import compute_layout
from binary_prediction_game.ui.theme import DEFAULT_THEME, Theme


class BinaryPredictionGui:
    """Pygame application containing the visual prototype."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)

        self.screen = pygame.display.set_mode(config.WINDOW_SIZE, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.theme = DEFAULT_THEME
        self.state = GameState()
        self.fonts = self._load_fonts()
        self.layout = compute_layout(config.WINDOW_SIZE)
        self.resize_timer = None
        self.pending_resize_size = None

        self.active_popup: str | None = None
        self.save_username = ""
        self.save_error_message = ""
        self.leaderboard_rows: list[dict[str, Any]] = []

        self._ensure_sequence_directories()

        self.reset_button = Button(self.layout.reset_button, "Reset", self.state.reset)
        self.rerun_button = Button(
            self.layout.rerun_button,
            "Re-run",
            self.state.rerun_current_sequence,
            enabled=False,
        )
        self.leaderboard_button = Button(
            self.layout.leaderboard_button,
            "Leaderboard",
            self._open_leaderboard_popup,
        )
        self.save_button = Button(
            self.layout.save_button,
            "0/250 bits left",
            self._open_save_popup,
            enabled=False,
        )
        self.rng_button = Button(
            self.layout.rng_button,
            "RNG sequence",
            self.state.rng_simulation,
        )
        self.open_folder_button = Button(
            self.layout.open_folder_button,
            "browse",
            self._open_sequence_folder,
        )
        self.sequence_dropdown = Dropdown(
            self.layout.sequence_dropdown,
            "Load sequence...",
            self._load_selected_sequence,
            max_visible_options=config.SEQUENCE_DROPDOWN_MAX_OPTIONS,
        )
        self._refresh_sequence_options()

        self.horizon_input = NumberInput(
            self.layout.horizon_input,
            "horizon h",
            self.state.horizon,
            config.HORIZON_MIN,
            config.HORIZON_MAX,
            self._set_horizon,
        )
        self.l_past_input = NumberInput(
            self.layout.l_past_input,
            "context L",
            self.state.l_past,
            config.L_PAST_MIN,
            config.L_PAST_MAX,
            self._set_l_past,
        )

    def run(self) -> None:
        """Start the main event loop."""

        while True:
            for event in pygame.event.get():
                self._handle_event(event)

            self._update()
            self._draw()
            pygame.display.flip()
            self.clock.tick(config.TARGET_FPS)

    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        if event.type == pygame.VIDEORESIZE:
            if self.resize_timer is not None:
                self.resize_timer.cancel()

            self.resize_timer = pygame.time.set_timer(pygame.USEREVENT, 200)
            self.pending_resize_size = event.size
            return

        if event.type == pygame.USEREVENT and self.pending_resize_size is not None:
            self._handle_resize(self.pending_resize_size)
            self.pending_resize_size = None
            return

        if self.active_popup is not None:
            self._handle_popup_event(event)
            return

        if event.type == pygame.DROPFILE:
            self._load_dropped_sequence(event.file)
            return

        self.reset_button.handle_event(event)
        self.rerun_button.handle_event(event)
        self.leaderboard_button.handle_event(event)
        self.save_button.handle_event(event)
        self.rng_button.handle_event(event)
        self.open_folder_button.handle_event(event)
        self.sequence_dropdown.handle_event(event)
        self.horizon_input.handle_event(event)
        self.l_past_input.handle_event(event)

        if event.type == pygame.KEYDOWN and not self.state.simulation_running:
            if event.key == pygame.K_0:
                self.state.append_bit(0)
            elif event.key == pygame.K_1:
                self.state.append_bit(1)

    def _handle_resize(self, size: tuple[int, int]) -> None:
        width = max(config.MIN_WINDOW_SIZE[0], size[0])
        height = max(config.MIN_WINDOW_SIZE[1], size[1])

        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.layout = compute_layout((width, height))

        self.reset_button.rect = self.layout.reset_button
        self.rerun_button.rect = self.layout.rerun_button
        self.leaderboard_button.rect = self.layout.leaderboard_button
        self.save_button.rect = self.layout.save_button
        self.rng_button.rect = self.layout.rng_button
        self.open_folder_button.rect = self.layout.open_folder_button
        self.sequence_dropdown.set_rect(self.layout.sequence_dropdown)
        self.horizon_input.set_rect(self.layout.horizon_input)
        self.l_past_input.set_rect(self.layout.l_past_input)

    def _load_dropped_sequence(self, path: str) -> None:
        """Load a dropped .txt/.yaml bit sequence into benchmark playback."""

        if not self.layout.input_panel.collidepoint(pygame.mouse.get_pos()):
            self.state.status_message = "Drop sequence files onto the input panel."
            return

        try:
            loaded = self.state.load_sequence_file(path)
        except SequenceLoadError as error:
            self.state.status_message = f"Could not load sequence: {error}"
        else:
            self.sequence_dropdown.selected_label = self._sequence_label(loaded.path)
            self._sync_setting_inputs_from_state()
            self.state.status_message = (
                f"Loaded {len(loaded.bits)} bits from {loaded.path.name}"
            )

    def _update(self) -> None:
        if self.state.simulation_running:
            self.state.advance_simulation()

        controls_enabled = not self.state.controls_locked and self.active_popup is None
        benchmark_controls_enabled = not self.state.simulation_running
        self.horizon_input.set_enabled(controls_enabled)
        self.l_past_input.set_enabled(controls_enabled)
        self.rng_button.enabled = (
            benchmark_controls_enabled and not self.sequence_dropdown.is_open
        )
        self.rerun_button.enabled = (
            benchmark_controls_enabled and len(self.state.bits) > 0
        )
        self.open_folder_button.enabled = self.active_popup is None
        self.sequence_dropdown.set_enabled(benchmark_controls_enabled)
        self.leaderboard_button.enabled = self.active_popup is None
        self._update_save_button()

    def _draw(self) -> None:
        self.screen.fill(self.theme.background)
        self._draw_input_tape(self.layout.input_tape, self.theme)
        self._draw_prediction_tape(self.layout.prediction_tape, self.theme)
        self._draw_score_panel(self.layout.score_panel, self.theme)
        self._draw_sequence_panel(self.layout.sequence_panel, self.theme)
        self._draw_input_panel(self.layout.input_panel, self.theme)
        self._draw_legend_panel(self.layout.legend_panel, self.theme)
        self._draw_plot_panel(self.layout.plot_panel, self.theme)
        self.sequence_dropdown.draw_options(self.screen, self.fonts, self.theme)

        if self.active_popup is not None:
            self._draw_popup_overlay()

    def _draw_input_tape(self, rect: pygame.Rect, theme: Theme) -> None:
        draw_panel(self.screen, rect, theme)
        label = self.fonts.small.render("Input tape", True, theme.text_muted)
        self.screen.blit(
            label, (rect.left + config.TAPE_HORIZONTAL_PADDING, rect.top + 8)
        )

        columns = self._visible_tape_columns(rect)
        start_x = self._tape_start_x(rect)
        visible_bits = self.state.bits[-columns:]

        if not visible_bits:
            placeholder = self.fonts.mono_regular.render(
                "Waiting for input...", True, theme.text_muted
            )
            self.screen.blit(
                placeholder, placeholder.get_rect(midleft=(start_x, rect.centery + 2))
            )
            return

        visible_start_index = len(self.state.bits) - len(visible_bits)

        for column, bit in enumerate(visible_bits):
            cell = self._tape_cell_rect(start_x, rect.centery, column)
            rendered = self.fonts.mono_regular.render(str(bit), True, theme.text)
            self.screen.blit(rendered, rendered.get_rect(center=cell.center))

        self._draw_context_indicator(
            rect=rect,
            theme=theme,
            start_x=start_x,
            columns=columns + 1,
            visible_start_index=visible_start_index,
        )

    def _draw_prediction_tape(self, rect: pygame.Rect, theme: Theme) -> None:
        draw_panel(
            self.screen,
            rect,
            theme,
            title="Prediction tape",
            fonts=self.fonts,
        )

        names = self.state.active_model_names()

        row_top = rect.top + 38
        row_height = 16
        start_x = self._tape_start_x(rect)
        columns = self._visible_tape_columns(rect)
        visible_start_index = max(0, len(self.state.bits) - columns)
        tape_width = (columns + 1) * config.TAPE_CELL_WIDTH

        for idx, name_text in enumerate(names):
            y = row_top + idx * (row_height + 7)
            name = self.fonts.tiny.render(name_text, True, theme.text_muted)
            name_rect = name.get_rect(
                midright=(start_x - config.TAPE_LABEL_GAP, y + row_height // 2)
            )

            self.screen.blit(name, name_rect)

            tape_rect = pygame.Rect(start_x, y, tape_width, row_height)
            pygame.draw.rect(self.screen, theme.panel_dark, tape_rect, border_radius=4)
            self._draw_prediction_cells(
                tape_rect,
                columns,
                theme,
                model_index=idx,
                visible_start_index=visible_start_index,
            )

    def _draw_prediction_cells(
        self,
        rect: pygame.Rect,
        columns: int,
        theme: Theme,
        *,
        model_index: int,
        visible_start_index: int,
    ) -> None:
        y = rect.centery - config.TAPE_PREDICTION_CELL_SIZE // 2
        for column in range(columns):
            cell_center_x = (
                rect.left
                + column * config.TAPE_CELL_WIDTH
                + config.TAPE_CELL_WIDTH // 2
            )
            cell = pygame.Rect(
                cell_center_x - config.TAPE_PREDICTION_CELL_SIZE // 2,
                y,
                config.TAPE_PREDICTION_CELL_SIZE,
                config.TAPE_PREDICTION_CELL_SIZE,
            )

            global_index = visible_start_index + column
            revealed = self._revealed_prediction_at(global_index, model_index)
            if revealed is None:
                pygame.draw.rect(
                    self.screen,
                    theme.transparent_grid,
                    cell,
                    border_radius=2,
                )
                continue

            color = theme.correct if revealed.is_correct else theme.wrong
            pygame.draw.rect(self.screen, color, cell, border_radius=2)

            rendered = self.fonts.mono_tiny.render(
                str(revealed.prediction.bit), True, theme.panel_dark
            )
            self.screen.blit(
                rendered, rendered.get_rect(center=(cell.centerx, cell.centery - 0.5))
            )

    def _revealed_prediction_at(
        self, global_index: int, model_index: int
    ) -> RevealedPrediction | None:
        if not 0 <= global_index < len(self.state.revealed_predictions):
            return None

        row = self.state.revealed_predictions[global_index]
        if not 0 <= model_index < len(row):
            return None

        return row[model_index]

    def _draw_context_indicator(
        self,
        *,
        rect: pygame.Rect,
        theme: Theme,
        start_x: int,
        columns: int,
        visible_start_index: int,
    ) -> None:
        """Draw the marker for the context end used by the current horizon."""

        context_index = len(self.state.bits) - self.state.horizon
        if context_index < 0:
            return

        context_column = context_index - visible_start_index
        if not 0 <= context_column < columns:
            return

        context_cell = self._tape_cell_rect(start_x, rect.centery, context_column)
        dot_center = (
            context_cell.centerx + 1,
            context_cell.bottom + config.CONTEXT_INDICATOR_Y_OFFSET,
        )

        pygame.draw.circle(
            self.screen,
            theme.wrong,
            dot_center,
            config.CONTEXT_INDICATOR_RADIUS,
        )

        self._draw_context_arrow(
            rect=rect,
            theme=theme,
            start_x=start_x,
            from_column=context_column,
            to_column=context_column + self.state.horizon,
            columns=columns,
        )

    def _draw_context_arrow(
        self,
        *,
        rect: pygame.Rect,
        theme: Theme,
        start_x: int,
        from_column: int,
        to_column: int,
        columns: int,
    ) -> None:
        """Draw a small curved arrow from the marker to the next tape cell."""

        from_cell = self._tape_cell_rect(start_x, rect.centery, from_column)

        if to_column < columns:
            to_cell = self._tape_cell_rect(start_x, rect.centery, to_column)
        else:
            to_cell = self._tape_cell_rect(start_x, rect.centery, from_column)
            to_cell.x += config.TAPE_CELL_WIDTH * (to_column - from_column)

        start = (
            from_cell.centerx + config.CONTEXT_INDICATOR_RADIUS - 1,
            from_cell.bottom
            + config.CONTEXT_INDICATOR_Y_OFFSET
            + config.CONTEXT_ARROW_HEIGHT
            - 1,
        )
        end = (
            to_cell.centerx + config.CONTEXT_INDICATOR_RADIUS - 1,
            to_cell.bottom,
        )

        arc_rect = pygame.Rect(start[0], end[1], end[0] - start[0], start[1] - end[1])
        pygame.draw.arc(self.screen, theme.wrong, arc_rect, -3.141, 0.3, 1)

        head_size = config.CONTEXT_ARROW_HEAD_SIZE
        pygame.draw.polygon(
            self.screen,
            theme.wrong,
            [
                (end[0] - head_size, end[1]),
                (end[0] - 2 * head_size, end[1] + head_size),
                (end[0] + head_size, end[1] + head_size),
            ],
        )

    def _visible_tape_columns(self, rect: pygame.Rect) -> int:
        available_width = (
            rect.width - config.TAPE_LABEL_WIDTH - config.TAPE_HORIZONTAL_PADDING
        )
        columns = available_width // config.TAPE_CELL_WIDTH
        return max(1, min(config.INPUT_TAPE_VISIBLE_BITS, columns))

    def _tape_start_x(self, rect: pygame.Rect) -> int:
        return rect.left + config.TAPE_LABEL_WIDTH

    def _tape_cell_rect(self, start_x: int, center_y: int, column: int) -> pygame.Rect:
        return pygame.Rect(
            start_x + column * config.TAPE_CELL_WIDTH,
            center_y - config.TAPE_CELL_WIDTH // 2,
            config.TAPE_CELL_WIDTH,
            config.TAPE_CELL_WIDTH,
        )

    def _draw_score_panel(self, rect: pygame.Rect, theme: Theme) -> None:
        draw_panel(self.screen, rect, theme, title="Randomness score", fonts=self.fonts)

        score = self.state.randomness_score()
        if score is None:
            main_text = "--"
            subtitle_text = "available after the first input"
        else:
            main_text = f"{score:.2f}"
            subtitle_text = "1 = random-like, 0 = predictable"

        score_text = self.fonts.huge.render(main_text, True, self._score_color(score))
        score_rect = score_text.get_rect(center=(rect.centerx, rect.top + 78))
        self.screen.blit(score_text, score_rect)

        subtitle = self.fonts.tiny.render(subtitle_text, True, theme.text_muted)
        subtitle_rect = subtitle.get_rect(center=(rect.centerx, score_rect.bottom + 18))
        self.screen.blit(subtitle, subtitle_rect)

        best_accuracy = self._best_accuracy()
        if best_accuracy is not None:
            detail = self.fonts.tiny.render(
                f"best model accuracy: {100.0 * best_accuracy:.1f}%",
                True,
                theme.text_muted,
            )
            self.screen.blit(
                detail,
                detail.get_rect(center=(rect.centerx, self.save_button.rect.top - 15)),
            )

        self.leaderboard_button.draw(self.screen, self.fonts, theme)
        self.save_button.draw(self.screen, self.fonts, theme)

    def _draw_sequence_panel(self, rect: pygame.Rect, theme: Theme) -> None:
        draw_panel(
            self.screen, rect, theme, title="Benchmark sequences", fonts=self.fonts
        )

        hint = self.fonts.tiny.render(
            "Drop .txt/.yaml files onto the input panel.",
            True,
            theme.text_muted,
        )
        self.screen.blit(hint, (rect.left + 14, rect.top + 30))

        self.sequence_dropdown.draw(self.screen, self.fonts, theme)
        self.rng_button.draw(self.screen, self.fonts, theme)
        self.open_folder_button.draw(self.screen, self.fonts, theme)

    def _open_sequence_folder(self) -> None:
        """Open the runtime data folder in the operating system's file browser."""

        config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
        config.BUILTIN_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
        config.USER_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

        try:
            if sys.platform.startswith("win"):
                os.startfile(str(config.DATA_ROOT))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(config.DATA_ROOT)])
            else:
                subprocess.Popen(["xdg-open", str(config.DATA_ROOT)])
        except OSError as error:
            self.state.status_message = f"Could not open data folder: {error}"
        else:
            self.state.status_message = f"Opened folder: {config.DATA_ROOT}"

    def _draw_input_panel(self, rect: pygame.Rect, theme: Theme) -> None:
        draw_panel(self.screen, rect, theme, title="", fonts=self.fonts)

        self.reset_button.draw(self.screen, self.fonts, theme)
        self.rerun_button.draw(self.screen, self.fonts, theme)
        self.horizon_input.draw(self.screen, self.fonts, theme)
        self.l_past_input.draw(self.screen, self.fonts, theme)

        latest = self.state.latest_bit
        if latest is None:
            main_text = "0 / 1"
            color = theme.text_muted
        else:
            main_text = str(latest)
            color = theme.text

        main_surface = self.fonts.huge.render(main_text, True, color)
        main_rect = main_surface.get_rect(center=(rect.centerx, rect.centery + 22))
        self.screen.blit(main_surface, main_rect)

        status_lines = self._split_status_message(self.state.status_message)

        line_height = self.fonts.small.get_linesize()
        total_height = len(status_lines) * line_height

        start_y = rect.bottom - 18 - total_height

        for line_index, line in enumerate(status_lines):
            status_surface = self.fonts.small.render(line, True, theme.text_muted)
            status_rect = status_surface.get_rect(
                midtop=(rect.centerx, start_y + line_index * line_height)
            )
            self.screen.blit(status_surface, status_rect)

        if self.state.controls_locked:
            locked = self.fonts.tiny.render(
                "h and L locked until reset",
                True,
                theme.warning,
            )
            self.screen.blit(
                locked,
                (self.horizon_input.rect.left, self.horizon_input.rect.bottom + 32),
            )

    def _split_status_message(
        self,
        message: str,
        max_line_length: int = 35,
        max_total_length: int = 70,
    ) -> list[str]:
        """Split a status message into at most two display lines."""

        message = message.strip()

        if len(message) > max_total_length:
            message = message[: max_total_length - 3].rstrip() + "..."

        if len(message) <= max_line_length:
            return [message]

        break_point = message.rfind(" ", 0, max_line_length + 1)

        if break_point <= 0:
            break_point = max_line_length

        first_line = message[:break_point].rstrip()
        second_line = message[break_point:].lstrip()

        return [first_line, second_line]

    def _draw_legend_panel(self, rect: pygame.Rect, theme: Theme) -> None:
        draw_panel(self.screen, rect, theme, title="Models", fonts=self.fonts)

        names = self.state.active_model_names()
        latest = self.state.latest_revealed_predictions()
        y = rect.top + 42

        sorted_indices = list(range(len(names)))
        if self.state.scores:
            model_accuracies = [
                score.accuracy if score.accuracy is not None else -1.0
                for score in self.state.scores
            ]
            sorted_indices = sorted(
                range(len(names)),
                key=lambda i: model_accuracies[i],
                reverse=True,
            )

        for idx in range(len(names)):
            mapped_idx = sorted_indices[idx]

            row_color = theme.model_colors[mapped_idx % len(theme.model_colors)]
            number = self.fonts.tiny.render(f"{idx + 1}.", True, theme.text_muted)
            label = self.fonts.tiny.render(names[mapped_idx], True, row_color)
            self.screen.blit(number, (rect.left + 16, y))
            self.screen.blit(label, (rect.left + 42, y))

            prediction_box = pygame.Rect(rect.right - 72, y - 2, 22, 18)
            revealed = latest[mapped_idx] if mapped_idx < len(latest) else None
            if revealed is None:
                box_fill = theme.panel_dark
                box_border = theme.border_muted
                prediction_text = "-"
                prediction_color = theme.text_muted
            else:
                box_fill = theme.correct if revealed.is_correct else theme.wrong
                box_border = box_fill
                prediction_text = str(revealed.prediction.bit)
                prediction_color = theme.panel_dark

            pygame.draw.rect(self.screen, box_fill, prediction_box, border_radius=4)
            pygame.draw.rect(
                self.screen,
                box_border,
                prediction_box,
                width=1,
                border_radius=4,
            )
            pred_surface = self.fonts.tiny.render(
                prediction_text, True, prediction_color
            )
            self.screen.blit(
                pred_surface, pred_surface.get_rect(center=prediction_box.center)
            )

            accuracy = (
                self.state.scores[mapped_idx].accuracy
                if mapped_idx < len(self.state.scores)
                else None
            )
            accuracy_text = "--" if accuracy is None else f"{100.0 * accuracy:.0f}%"
            acc_surface = self.fonts.tiny.render(accuracy_text, True, theme.text_muted)
            self.screen.blit(
                acc_surface, acc_surface.get_rect(midleft=(rect.right - 44, y + 7))
            )

            y += 25

        footer = self.fonts.tiny.render(
            "box = latest revealed prediction, % = cumulative accuracy",
            True,
            theme.text_muted,
        )
        self.screen.blit(footer, (rect.left + 16, rect.bottom - 28))

    def _draw_plot_panel(self, rect: pygame.Rect, theme: Theme) -> None:
        draw_panel(
            self.screen,
            rect,
            theme,
            title="Accuracy plot",
            fonts=self.fonts,
        )

        plot = rect.inflate(-50, -60)
        plot.y += 16
        pygame.draw.rect(self.screen, theme.panel_dark, plot)
        pygame.draw.rect(self.screen, theme.border_muted, plot, width=1)

        # top and bottom label should be rotated by 90° so that they are readable when looking at the plot
        top_label = self.fonts.tiny.render("50%", True, theme.text_muted)
        bottom_label = self.fonts.tiny.render("100%", True, theme.text_muted)
        top_label = pygame.transform.rotate(top_label, 90)
        bottom_label = pygame.transform.rotate(bottom_label, 90)

        self.screen.blit(
            top_label,
            (plot.left - top_label.get_width(), plot.top),
        )
        self.screen.blit(
            bottom_label,
            (
                plot.left - bottom_label.get_width(),
                plot.bottom - bottom_label.get_height(),
            ),
        )

        for fraction in (0.0, 0.5, 1.0):
            y = plot.top + int(fraction * plot.height)
            pygame.draw.line(
                self.screen,
                theme.transparent_grid,
                (plot.left, y),
                (plot.right, y),
                1,
            )

        if not self.state.bits:
            note = self.fonts.small.render(
                "model accuracy curves will be drawn here",
                True,
                theme.text_muted,
            )
            self.screen.blit(note, note.get_rect(center=plot.center))
            return

        max_points = max(2, plot.width)
        for idx, score in enumerate(self.state.scores):
            history = score.accuracy_history[-max_points:]
            if not history:
                continue

            color = theme.model_colors[idx % len(theme.model_colors)]
            points = self._accuracy_points(history, plot)
            if len(points) == 1:
                pygame.draw.circle(self.screen, color, points[0], 3)
            else:
                pygame.draw.lines(self.screen, color, False, points, 2)

    def _draw_popup_overlay(self) -> None:
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 175))
        self.screen.blit(overlay, (0, 0))

        if self.active_popup == "leaderboard":
            self._draw_leaderboard_popup()
        elif self.active_popup == "save":
            self._draw_save_popup()

    def _draw_leaderboard_popup(self) -> None:
        theme = self.theme
        rect = self._popup_rect(width=760, height=540)
        draw_panel(self.screen, rect, theme, title="Leaderboard", fonts=self.fonts)

        close_rect = self._leaderboard_close_rect(rect)
        self._draw_modal_button(close_rect, "Close", enabled=True)

        if not self.leaderboard_rows:
            message = self.fonts.small.render(
                "No saved user sequences yet.", True, theme.text_muted
            )
            self.screen.blit(message, message.get_rect(center=rect.center))
            return

        headers = ["#", "user", "score", "h", "L", "bits", "created"]
        x_positions = [
            rect.left + 28,
            rect.left + 70,
            rect.left + 270,
            rect.left + 360,
            rect.left + 410,
            rect.left + 465,
            rect.left + 550,
        ]
        header_y = rect.top + 62
        for text, x in zip(headers, x_positions, strict=True):
            header = self.fonts.tiny.render(text, True, theme.text_muted)
            self.screen.blit(header, (x, header_y))

        line_y = header_y + 24
        pygame.draw.line(
            self.screen,
            theme.border_muted,
            (rect.left + 24, line_y),
            (rect.right - 24, line_y),
            1,
        )

        for rank, row in enumerate(
            self.leaderboard_rows[: config.LEADERBOARD_MAX_ROWS], start=1
        ):
            y = line_y + 12 + (rank - 1) * 30
            score = float(row.get("randomness_score", 0.0) or 0.0)
            values = [
                str(rank),
                str(row.get("username", "-")),
                f"{score:.2f}",
                str(row.get("horizon", "-")),
                str(row.get("l_past", "-")),
                str(row.get("sequence_length", "-")),
                self._format_created_at(row.get("created_at")),
            ]
            for value, x in zip(values, x_positions, strict=True):
                color = (
                    self._score_color(score) if value == f"{score:.2f}" else theme.text
                )
                rendered = self.fonts.tiny.render(value, True, color)
                self.screen.blit(rendered, (x, y))

    def _draw_save_popup(self) -> None:
        theme = self.theme
        rect = self._popup_rect(width=560, height=300)
        draw_panel(self.screen, rect, theme, title="Save sequence", fonts=self.fonts)

        description = self.fonts.small.render(
            f"Enter a username for the leaderboard entry (max. {config.USERNAME_MAX_LENGTH} characters).",
            True,
            theme.text_muted,
        )
        self.screen.blit(description, (rect.left + 26, rect.top + 58))

        input_rect = self._save_username_rect(rect)
        pygame.draw.rect(self.screen, theme.panel_dark, input_rect, border_radius=8)
        pygame.draw.rect(
            self.screen, theme.border, input_rect, width=2, border_radius=8
        )

        value = self.save_username or "A-Z, a-z, 0-9, - and _ only"
        color = theme.text if self.save_username else theme.text_muted
        rendered = self.fonts.small.render(value, True, color)
        self.screen.blit(
            rendered,
            rendered.get_rect(midleft=(input_rect.left + 12, input_rect.centery)),
        )

        if self.save_error_message:
            error = self.fonts.tiny.render(self.save_error_message, True, theme.wrong)
            self.screen.blit(error, (input_rect.left, input_rect.bottom + 12))

        save_rect, cancel_rect = self._save_popup_button_rects(rect)
        self._draw_modal_button(save_rect, "Save", enabled=bool(self.save_username))
        self._draw_modal_button(cancel_rect, "Cancel", enabled=True)

    def _draw_modal_button(
        self, rect: pygame.Rect, label: str, *, enabled: bool
    ) -> None:
        theme = self.theme
        hovered = enabled and rect.collidepoint(pygame.mouse.get_pos())
        if not enabled:
            fill = theme.disabled
            border = theme.border_muted
            text_color = theme.text_muted
        elif hovered:
            fill = theme.panel_light
            border = theme.accent
            text_color = theme.text
        else:
            fill = theme.panel
            border = theme.border
            text_color = theme.text

        pygame.draw.rect(self.screen, fill, rect, border_radius=8)
        pygame.draw.rect(self.screen, border, rect, width=2, border_radius=8)
        text = self.fonts.small.render(label, True, text_color)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _handle_popup_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._close_popup()
            return

        if self.active_popup == "leaderboard":
            self._handle_leaderboard_popup_event(event)
        elif self.active_popup == "save":
            self._handle_save_popup_event(event)

    def _handle_leaderboard_popup_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return

        rect = self._popup_rect(width=760, height=540)
        if self._leaderboard_close_rect(rect).collidepoint(event.pos):
            self._close_popup()

    def _handle_save_popup_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.save_username = self.save_username[:-1]
                self.save_error_message = ""
            elif event.key == pygame.K_RETURN:
                self._submit_save_popup()
            elif event.unicode and event.unicode in config.USERNAME_ALLOWED_CHARS:
                self.save_username += event.unicode
                self.save_error_message = ""
            return

        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return

        rect = self._popup_rect(width=560, height=300)
        save_rect, cancel_rect = self._save_popup_button_rects(rect)
        if save_rect.collidepoint(event.pos):
            self._submit_save_popup()
        elif cancel_rect.collidepoint(event.pos):
            self._close_popup()

    def _submit_save_popup(self) -> None:
        try:
            self.state.save_current_user_sequence(self.save_username)
        except (ValueError, SequenceLoadError, UsernameValidationError) as error:
            self.save_error_message = str(error)
            return

        self._refresh_sequence_options()
        self._close_popup()

    def _popup_rect(self, *, width: int, height: int) -> pygame.Rect:
        screen_rect = self.screen.get_rect()
        return (
            pygame.Rect(0, 0, width, height)
            .clamp(screen_rect)
            .move(
                (screen_rect.width - width) // 2,
                (screen_rect.height - height) // 2,
            )
        )

    def _leaderboard_close_rect(self, popup_rect: pygame.Rect) -> pygame.Rect:
        return pygame.Rect(popup_rect.right - 124, popup_rect.bottom - 56, 96, 36)

    def _save_username_rect(self, popup_rect: pygame.Rect) -> pygame.Rect:
        return pygame.Rect(
            popup_rect.left + 26, popup_rect.top + 104, popup_rect.width - 52, 42
        )

    def _save_popup_button_rects(
        self, popup_rect: pygame.Rect
    ) -> tuple[pygame.Rect, pygame.Rect]:
        save_rect = pygame.Rect(popup_rect.right - 236, popup_rect.bottom - 58, 96, 38)
        cancel_rect = pygame.Rect(
            popup_rect.right - 126, popup_rect.bottom - 58, 96, 38
        )
        return save_rect, cancel_rect

    def _open_leaderboard_popup(self) -> None:
        self.leaderboard_rows = load_leaderboard()
        self.active_popup = "leaderboard"
        self.sequence_dropdown.is_open = False

    def _open_save_popup(self) -> None:
        if not self.state.save_eligible:
            return
        self.save_username = ""
        self.save_error_message = ""
        self.active_popup = "save"
        self.sequence_dropdown.is_open = False

    def _close_popup(self) -> None:
        self.active_popup = None
        self.save_error_message = ""

    def _accuracy_points(
        self, accuracy_history: list[float], plot: pygame.Rect
    ) -> list[tuple[int, int]]:
        if len(accuracy_history) == 1:
            x_values = [plot.left]
        else:
            x_values = [
                plot.left + int(i * plot.width / (len(accuracy_history) - 1))
                for i in range(len(accuracy_history))
            ]

        points: list[tuple[int, int]] = []
        for x, accuracy in zip(x_values, accuracy_history, strict=True):
            clamped_accuracy = max(0.5, min(1.0, accuracy))
            y_fraction = (clamped_accuracy - 0.5) / 0.5
            y = plot.top + int(y_fraction * plot.height)
            points.append((x, y))
        return points

    def _best_accuracy(self) -> float | None:
        accuracies = [
            score.accuracy for score in self.state.scores if score.accuracy is not None
        ]
        if not accuracies:
            return None
        return max(accuracies)

    def _set_horizon(self, value: int) -> None:
        self.state.set_horizon(value)
        self.horizon_input.value = self.state.horizon

    def _set_l_past(self, value: int) -> None:
        self.state.set_l_past(value)
        self.l_past_input.value = self.state.l_past

    def _load_selected_sequence(self, value: object) -> None:
        if value is None:
            return

        path = Path(value)
        try:
            loaded = self.state.load_sequence_file(str(path))
        except SequenceLoadError as error:
            self.state.status_message = f"Could not load sequence: {error}"
            return

        self.sequence_dropdown.selected_label = self._sequence_label(loaded.path)
        self._sync_setting_inputs_from_state()
        self.state.status_message = (
            f"Loaded {len(loaded.bits)} bits from {loaded.path.name}"
        )

    def _sync_setting_inputs_from_state(self) -> None:
        """Keep the visible h/L number inputs aligned with game-state settings."""

        self.horizon_input.value = self.state.horizon
        self.l_past_input.value = self.state.l_past

    def _refresh_sequence_options(self) -> None:
        builtin_paths = list_builtin_sequences()
        user_paths = list_user_sequences_by_leaderboard()

        options: list[tuple[str, object]] = [
            (self._sequence_label(path), path) for path in builtin_paths
        ]

        if builtin_paths and user_paths:
            options.append(Dropdown.separator())

        options.extend((self._sequence_label(path), path) for path in user_paths)

        self.sequence_dropdown.set_options(options)

    def _sequence_label(self, path: Path) -> str:
        try:
            # relative = path.resolve().relative_to(config.DATA_ROOT.resolve())
            # return str(relative)
            # only keep filename to avoid cluttering
            filename = path.stem
            return filename
        except ValueError:
            return path.name

    def _update_save_button(self) -> None:
        if self.state.save_eligible and self.active_popup is None:
            self.save_button.label = "Save"
            self.save_button.enabled = True
            return

        self.save_button.enabled = False
        if self.state.input_origin != "manual":
            self.save_button.label = "Save"
            return

        remaining = self.state.remaining_bits_until_save
        self.save_button.label = f"Input {remaining} more bits"

    def _score_color(self, score: float | None) -> tuple[int, int, int]:
        """Interpolate score color from theme.wrong at 0 to theme.correct at 1."""

        if score is None:
            return self.theme.text

        clamped = max(0.0, min(1.0, score))
        red = self.theme.wrong
        green = self.theme.correct
        return tuple(
            int(red[index] + clamped * (green[index] - red[index]))
            for index in range(3)
        )

    def _format_created_at(self, created_at: object) -> str:
        if not isinstance(created_at, str):
            return "-"
        return created_at.replace("T", " ")[:16]

    def _ensure_sequence_directories(self) -> None:
        config.BUILTIN_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
        config.USER_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_fonts(self) -> Fonts:
        spec = config.FontSpec()
        return Fonts(
            tiny=pygame.font.SysFont("arial", spec.tiny),
            small=pygame.font.SysFont("arial", spec.small),
            regular=pygame.font.SysFont("arial", spec.regular),
            large=pygame.font.SysFont("arial", spec.large),
            huge=pygame.font.SysFont("arial", spec.huge),
            mono_tiny=pygame.font.SysFont("consolas", spec.tiny),
            mono_small=pygame.font.SysFont("consolas", spec.small),
            mono_regular=pygame.font.SysFont("consolas", spec.regular),
            mono_large=pygame.font.SysFont("consolas", spec.large),
        )


def main() -> None:
    """Entry point for the Pygame GUI prototype."""

    BinaryPredictionGui().run()
