"""Visual theme for the GUI prototype."""

from __future__ import annotations

from dataclasses import dataclass

Color = tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    """Shared color palette."""

    background: Color = (19, 24, 33)
    panel: Color = (31, 39, 52)
    panel_light: Color = (43, 54, 72)
    panel_dark: Color = (15, 20, 28)
    panel_hover: Color = (125, 135, 152)
    border: Color = (104, 119, 144)
    border_muted: Color = (70, 82, 102)
    text: Color = (235, 239, 246)
    text_muted: Color = (158, 169, 188)
    disabled: Color = (83, 93, 110)
    accent: Color = (115, 162, 255)
    correct: Color = (90, 196, 128)
    wrong: Color = (232, 96, 96)
    warning: Color = (245, 189, 86)
    transparent_grid: Color = (55, 65, 82)

    model_colors: tuple[Color, ...] = (
        (115, 162, 255),
        (90, 196, 128),
        (232, 96, 96),
        (245, 189, 86),
        (255, 161, 115),
        (162, 115, 255),
        (96, 232, 232),
        (189, 245, 189),
        (255, 115, 162),
        (115, 255, 162),
    )


DEFAULT_THEME = Theme()
