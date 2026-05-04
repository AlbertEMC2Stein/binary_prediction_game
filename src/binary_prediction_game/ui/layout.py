"""Layout helpers for the fixed GUI prototype."""

from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass(frozen=True)
class AppLayout:
    """Collection of major rectangles in the UI."""

    input_tape: pygame.Rect
    prediction_tape: pygame.Rect
    score_panel: pygame.Rect
    sequence_panel: pygame.Rect
    input_panel: pygame.Rect
    legend_panel: pygame.Rect
    plot_panel: pygame.Rect
    reset_button: pygame.Rect
    leaderboard_button: pygame.Rect
    save_button: pygame.Rect
    rng_button: pygame.Rect
    sequence_dropdown: pygame.Rect
    horizon_input: pygame.Rect
    l_past_input: pygame.Rect


def compute_layout(window_size: tuple[int, int]) -> AppLayout:
    """Compute all UI rectangles from the current window size."""

    width, height = window_size
    margin = 24
    gap = 5

    input_tape = pygame.Rect(margin, margin, width - 2 * margin, 54)
    prediction_tape = pygame.Rect(
        margin, input_tape.bottom + gap, width - 2 * margin, 250
    )

    middle_top = prediction_tape.bottom + gap
    plot_height = max(150, int(height * 0.2))
    plot_panel = pygame.Rect(
        margin, height - margin - plot_height, width - 2 * margin, plot_height
    )

    middle_height = max(180, plot_panel.top - middle_top - gap)
    score_width = max(230, int(width * 0.3))
    legend_width = max(270, int(width * 0.27))

    score_height = max(190, int(middle_height * 0.6))
    score_panel = pygame.Rect(margin, middle_top, score_width, score_height)
    sequence_panel = pygame.Rect(
        margin,
        score_panel.bottom + gap,
        score_width,
        max(130, middle_height - score_height - gap),
    )
    legend_panel = pygame.Rect(
        width - margin - legend_width, middle_top, legend_width, middle_height
    )
    input_panel = pygame.Rect(
        score_panel.right + gap,
        middle_top,
        legend_panel.left - score_panel.right - 2 * gap,
        middle_height,
    )

    control_y = input_panel.top + 18
    button_width = 120
    button_height = 42
    reset_button = pygame.Rect(
        input_panel.left + 18, control_y, button_width, button_height
    )

    number_width = 92
    number_height = 42
    l_past_input = pygame.Rect(
        input_panel.right - 18 - number_width, control_y, number_width, number_height
    )
    horizon_input = pygame.Rect(
        l_past_input.left - 18 - number_width, control_y, number_width, number_height
    )

    score_button_y = score_panel.bottom - 52
    score_button_width = max(98, (score_panel.width - 42) // 2)
    leaderboard_button = pygame.Rect(
        score_panel.left + 14,
        score_button_y,
        score_button_width,
        36,
    )
    save_button = pygame.Rect(
        leaderboard_button.right + 14,
        score_button_y,
        score_panel.right - leaderboard_button.right - 28,
        36,
    )

    sequence_dropdown = pygame.Rect(
        sequence_panel.left + 14,
        sequence_panel.top + 58,
        sequence_panel.width - 28,
        36,
    )
    rng_button = pygame.Rect(
        sequence_panel.left + 14,
        sequence_dropdown.bottom + 10,
        sequence_panel.width - 28,
        38,
    )

    return AppLayout(
        input_tape=input_tape,
        prediction_tape=prediction_tape,
        score_panel=score_panel,
        sequence_panel=sequence_panel,
        input_panel=input_panel,
        legend_panel=legend_panel,
        plot_panel=plot_panel,
        reset_button=reset_button,
        leaderboard_button=leaderboard_button,
        save_button=save_button,
        rng_button=rng_button,
        sequence_dropdown=sequence_dropdown,
        horizon_input=horizon_input,
        l_past_input=l_past_input,
    )
