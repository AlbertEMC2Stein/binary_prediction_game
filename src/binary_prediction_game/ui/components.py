"""Small reusable Pygame UI components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pygame

from binary_prediction_game.ui.theme import Theme


@dataclass
class Fonts:
    """Loaded pygame fonts used by the UI."""

    tiny: pygame.font.Font
    small: pygame.font.Font
    regular: pygame.font.Font
    large: pygame.font.Font
    huge: pygame.font.Font
    mono_tiny: pygame.font.Font
    mono_small: pygame.font.Font
    mono_regular: pygame.font.Font
    mono_large: pygame.font.Font


class Button:
    """Simple clickable button."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        callback: Callable[[], None],
        *,
        enabled: bool = True,
    ) -> None:
        self.rect = rect
        self.label = label
        self.callback = callback
        self.enabled = enabled
        self.is_hovered = False

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle mouse motion and clicks."""

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.enabled and self.rect.collidepoint(event.pos):
                self.callback()

    def draw(self, surface: pygame.Surface, fonts: Fonts, theme: Theme) -> None:
        """Draw the button."""

        if not self.enabled:
            fill = theme.disabled
            border = theme.border_muted
        elif self.is_hovered:
            fill = theme.panel_light
            border = theme.accent
        else:
            fill = theme.panel
            border = theme.border

        pygame.draw.rect(surface, fill, self.rect, border_radius=8)
        pygame.draw.rect(surface, border, self.rect, width=2, border_radius=8)

        text = fonts.small.render(
            self.label, True, theme.text if self.enabled else theme.text_muted
        )
        text_rect = text.get_rect(center=self.rect.center)
        surface.blit(text, text_rect)


class NumberInput:
    """Integer number input controlled by down/up arrow buttons."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        value: int,
        min_value: int,
        max_value: int,
        on_change: Callable[[int], None],
        *,
        enabled: bool = True,
    ) -> None:
        self.rect = rect
        self.label = label
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.on_change = on_change
        self.enabled = enabled
        self.left_arrow = pygame.Rect(rect.left, rect.top, 24, rect.height)
        self.right_arrow = pygame.Rect(rect.right - 24, rect.top, 24, rect.height)
        self.is_left_hovered = False
        self.is_right_hovered = False

    def set_rect(self, rect: pygame.Rect) -> None:
        """Update geometry after a window resize."""

        self.rect = rect
        self.left_arrow = pygame.Rect(rect.left, rect.top, 24, rect.height)
        self.right_arrow = pygame.Rect(rect.right - 24, rect.top, 24, rect.height)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable user interaction."""

        self.enabled = enabled

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle arrow clicks."""

        if event.type == pygame.MOUSEMOTION:
            self.is_left_hovered = self.left_arrow.collidepoint(event.pos)
            self.is_right_hovered = self.right_arrow.collidepoint(event.pos)
            return

        if not self.enabled:
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.left_arrow.collidepoint(event.pos):
                self._change_by(-1)
            elif self.right_arrow.collidepoint(event.pos):
                self._change_by(1)

    def draw(self, surface: pygame.Surface, fonts: Fonts, theme: Theme) -> None:
        """Draw the number input."""

        label = fonts.small.render(
            self.label, True, theme.text_muted if self.enabled else theme.disabled
        )
        surface.blit(label, (self.rect.left, self.rect.bottom + 6))

        fill = theme.panel if self.enabled else theme.panel_dark
        border = theme.border if self.enabled else theme.border_muted
        pygame.draw.rect(surface, fill, self.rect, border_radius=8)
        pygame.draw.rect(surface, border, self.rect, width=2, border_radius=8)

        self._draw_arrow_button(
            surface, fonts, theme, self.left_arrow, "-", self.is_left_hovered
        )
        self._draw_arrow_button(
            surface, fonts, theme, self.right_arrow, "+", self.is_right_hovered
        )

        color = theme.text if self.enabled else theme.text_muted
        value_text = fonts.regular.render(str(self.value), True, color)
        value_rect = value_text.get_rect(center=self.rect.center)
        surface.blit(value_text, value_rect)

    def _draw_arrow_button(
        self,
        surface: pygame.Surface,
        fonts: Fonts,
        theme: Theme,
        rect: pygame.Rect,
        symbol: str,
        hovered: bool,
    ) -> None:
        if self.enabled and hovered:
            pygame.draw.rect(surface, theme.panel_light, rect, border_radius=6)

        color = theme.text if self.enabled else theme.text_muted
        text = fonts.regular.render(symbol, True, color)
        surface.blit(text, text.get_rect(center=rect.center))

    def _change_by(self, delta: int) -> None:
        new_value = max(self.min_value, min(self.max_value, self.value + delta))
        if new_value != self.value:
            self.value = new_value
            self.on_change(new_value)


def draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    theme: Theme,
    *,
    title: str | None = None,
    fonts: Fonts | None = None,
) -> None:
    """Draw a rounded panel with an optional title."""

    pygame.draw.rect(surface, theme.panel, rect, border_radius=10)
    pygame.draw.rect(surface, theme.border_muted, rect, width=2, border_radius=10)

    if title is not None and fonts is not None:
        title_surface = fonts.small.render(title, True, theme.text_muted)
        surface.blit(title_surface, (rect.left + 14, rect.top + 10))


class Dropdown:
    """Minimal single-select dropdown for short option lists."""

    SEPARATOR = object()

    @classmethod
    def separator(cls) -> tuple[str, object]:
        """Return a non-selectable separator row."""

        return ("", cls.SEPARATOR)

    def __init__(
        self,
        rect: pygame.Rect,
        placeholder: str,
        on_select: Callable[[object], None],
        *,
        enabled: bool = True,
        max_visible_options: int = 8,
    ) -> None:
        self.rect = rect
        self.placeholder = placeholder
        self.on_select = on_select
        self.enabled = enabled
        self.max_visible_options = max_visible_options
        self.options: list[tuple[str, object]] = []
        self.selected_label: str | None = None
        self.is_open = False
        self.is_hovered = False

    def set_rect(self, rect: pygame.Rect) -> None:
        """Update geometry after a resize."""

        self.rect = rect

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable user interaction."""

        self.enabled = enabled
        if not enabled:
            self.is_open = False

    def set_options(self, options: list[tuple[str, object]]) -> None:
        """Replace the available options."""

        self.options = options
        selectable_labels = {
            label
            for label, value in options
            if value is not None and value is not self.SEPARATOR
        }

        if self.selected_label not in selectable_labels:
            self.selected_label = None

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle mouse interaction."""

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            return

        if not self.enabled:
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_open = not self.is_open
                return

            if self.is_open:
                selected = self._option_at(event.pos)
                if selected is not None:
                    label, value = selected
                    self.is_open = False
                    if value is not None:
                        self.selected_label = label
                        self.on_select(value)
                    return

                self.is_open = False

    def draw(self, surface: pygame.Surface, fonts: Fonts, theme: Theme) -> None:
        """Draw the collapsed dropdown."""

        if not self.enabled:
            fill = theme.panel_dark
            border = theme.border_muted
            text_color = theme.text_muted
        elif self.is_hovered or self.is_open:
            fill = theme.panel_light
            border = theme.accent
            text_color = theme.text
        else:
            fill = theme.panel
            border = theme.border
            text_color = theme.text

        pygame.draw.rect(surface, fill, self.rect, border_radius=8)
        pygame.draw.rect(surface, border, self.rect, width=2, border_radius=8)

        label = self.selected_label or self.placeholder
        text = fonts.tiny.render(
            _ellipsize(label, fonts.tiny, self.rect.width - 42), True, text_color
        )
        surface.blit(
            text, text.get_rect(midleft=(self.rect.left + 12, self.rect.centery))
        )

        arrow = fonts.small.render("▼" if not self.is_open else "▲", True, text_color)
        surface.blit(
            arrow, arrow.get_rect(center=(self.rect.right - 18, self.rect.centery))
        )

    def draw_options(self, surface: pygame.Surface, fonts: Fonts, theme: Theme) -> None:
        """Draw the open option list above other UI elements."""

        if not self.is_open:
            return

        option_layout = self._visible_option_layout()

        if not option_layout:
            option_layout = [
                (
                    ("No sequences found", None),
                    pygame.Rect(
                        self.rect.left,
                        self.rect.bottom + 4,
                        self.rect.width,
                        self.rect.height,
                    ),
                )
            ]

        list_top = self.rect.bottom + 4
        list_height = sum(option_rect.height for _, option_rect in option_layout)
        list_rect = pygame.Rect(
            self.rect.left,
            list_top,
            self.rect.width,
            list_height,
        )

        pygame.draw.rect(surface, theme.panel_dark, list_rect, border_radius=8)
        pygame.draw.rect(surface, theme.border, list_rect, width=2, border_radius=8)

        mouse_pos = pygame.mouse.get_pos()

        for (label, value), option_rect in option_layout:
            if value is self.SEPARATOR:
                y = option_rect.centery
                pygame.draw.line(
                    surface,
                    theme.border_muted,
                    (option_rect.left + 12, y),
                    (option_rect.right - 12, y),
                    1,
                )
                continue

            if value is not None and option_rect.collidepoint(mouse_pos):
                pygame.draw.rect(
                    surface, theme.panel_hover, option_rect, border_radius=6
                )

            text_color = theme.text if value is not None else theme.text_muted
            text = fonts.tiny.render(
                _ellipsize(label, fonts.tiny, option_rect.width - 20),
                True,
                text_color,
            )
            surface.blit(
                text,
                text.get_rect(midleft=(option_rect.left + 10, option_rect.centery)),
            )

    def _option_at(self, pos: tuple[int, int]) -> tuple[str, object] | None:
        for option, option_rect in self._visible_option_layout():
            _, value = option

            if value is self.SEPARATOR:
                continue

            if option_rect.collidepoint(pos):
                return option

        return None

    def _visible_option_layout(self) -> list[tuple[tuple[str, object], pygame.Rect]]:
        visible_options = self.options[: self.max_visible_options]

        option_layout: list[tuple[tuple[str, object], pygame.Rect]] = []
        current_y = self.rect.bottom + 4

        for option in visible_options:
            _, value = option
            option_height = self._option_height(value)

            option_rect = pygame.Rect(
                self.rect.left,
                current_y,
                self.rect.width,
                option_height,
            )
            option_layout.append((option, option_rect))

            current_y += option_height

        return option_layout

    def _option_height(self, value: object) -> int:
        if value is self.SEPARATOR:
            return 10

        return self.rect.height


def _ellipsize(text: str, font: pygame.font.Font, max_width: int) -> str:
    """Shorten text to fit into max_width pixels."""

    if font.size(text)[0] <= max_width:
        return text

    suffix = "..."
    available = max(0, max_width - font.size(suffix)[0])
    shortened = ""
    for character in text:
        if font.size(shortened + character)[0] > available:
            break
        shortened += character
    return shortened + suffix
