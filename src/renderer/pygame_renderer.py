"""pygame renderer + input.

Renders glyphs and UI text to a pygame window and reports abstract actions
to the game loop.
"""

from __future__ import annotations

from collections import deque

from .base import Renderer


class PygameRenderer(Renderer):
    def __init__(self) -> None:
        self._pygame = None
        self._screen = None
        self._font = None
        self._cell_w = 12
        self._cell_h = 20

        # Screen size in text cells; roomy enough for map + sidebar + menus.
        self._cols = 120
        self._rows = 40

        self._bg = (14, 16, 20)
        self._default_fg = (230, 230, 230)
        self._class_colors = {
            "default": (230, 230, 230),
            "wall": (90, 140, 230),
            "stairs": (120, 220, 240),
            "friendly": (120, 220, 120),
            "enemy": (230, 110, 110),
            "valuable": (245, 215, 110),
        }

        self._keydown_to_action = {}
        self._keyup_to_action = {}
        self._space_held = False
        self._space_initial_delay_ms = 180
        self._space_repeat_interval_ms = 70
        self._next_space_repeat_ms = 0
        self._pending_actions: deque[str] = deque()

    def setup(self) -> None:
        import pygame

        pygame.init()
        pygame.display.set_caption("Tales of Derision")

        self._pygame = pygame
        self._font = pygame.font.SysFont("DejaVu Sans Mono", 20)
        sample = self._font.render("M", True, self._default_fg)
        self._cell_w = max(10, sample.get_width())
        self._cell_h = max(16, sample.get_height())

        self._screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

        self._keydown_to_action = {
            pygame.K_w: "move_up",
            pygame.K_s: "move_down",
            pygame.K_a: "move_left",
            pygame.K_d: "move_right",
            pygame.K_i: "open_inventory",
            pygame.K_ESCAPE: "open_pause_menu",
            pygame.K_RETURN: "menu_select",
            pygame.K_KP_ENTER: "menu_select",
            pygame.K_SPACE: "confirm_action",
        }
        self._keyup_to_action = {
            pygame.K_w: "release_up",
            pygame.K_s: "release_down",
            pygame.K_a: "release_left",
            pygame.K_d: "release_right",
        }

    def teardown(self) -> None:
        if self._pygame is None:
            return
        self._pygame.quit()

    def clear(self) -> None:
        if self._screen is not None:
            self._screen.fill(self._bg)

    def _blit_text(self, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
        if self._screen is None or self._font is None:
            return
        surface = self._font.render(text, True, color)
        self._screen.blit(surface, (x * self._cell_w, y * self._cell_h))

    def draw_glyph(self, x: int, y: int, glyph: str) -> None:
        self._blit_text(x, y, glyph, self._default_fg)

    def draw_glyph_classified(self, x: int, y: int, glyph: str, classification: str) -> None:
        color = self._class_colors.get(classification, self._default_fg)
        self._blit_text(x, y, glyph, color)

    def draw_text(self, x: int, y: int, text: str) -> None:
        self._blit_text(x, y, text, self._default_fg)

    def present(self) -> None:
        if self._pygame is not None:
            self._pygame.display.flip()

    def poll_action(self) -> str | None:
        if self._pygame is None:
            return None

        while True:
            if self._pending_actions:
                return self._pending_actions.popleft()

            events = self._pygame.event.get()
            now = self._pygame.time.get_ticks()

            for event in events:
                if event.type == self._pygame.QUIT:
                    return "quit"

                if event.type == self._pygame.KEYDOWN:
                    if event.key == self._pygame.K_SPACE:
                        if not self._space_held:
                            self._space_held = True
                            self._next_space_repeat_ms = now + self._space_initial_delay_ms
                        self._pending_actions.append("confirm_action")
                        continue

                    if event.key in self._keydown_to_action:
                        self._pending_actions.append(self._keydown_to_action[event.key])

                if event.type == self._pygame.KEYUP:
                    if event.key == self._pygame.K_SPACE:
                        self._space_held = False
                        continue

                    if event.key in self._keyup_to_action:
                        self._pending_actions.append(self._keyup_to_action[event.key])

            if self._pending_actions:
                return self._pending_actions.popleft()

            if self._space_held and now >= self._next_space_repeat_ms:
                self._next_space_repeat_ms = now + self._space_repeat_interval_ms
                return "confirm_action"

            self._pygame.time.wait(8)
