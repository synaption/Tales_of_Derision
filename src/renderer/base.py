"""Renderer interface.

Everything the game needs from a display backend lives here. Implement this
for curses today, and for tcod / pygame / raylib / opengl later without
changing any game or system code.
"""
from abc import ABC, abstractmethod


class Renderer(ABC):
    def __enter__(self) -> "Renderer":
        self.setup()
        return self

    def __exit__(self, *exc) -> None:
        self.teardown()

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def teardown(self) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        """Erase the frame before drawing."""

    @abstractmethod
    def draw_glyph(self, x: int, y: int, glyph: str) -> None:
        """Draw a single character at map cell (x, y)."""

    def draw_glyph_classified(self, x: int, y: int, glyph: str, classification: str) -> None:
        """Draw a glyph with semantic classification (wall, enemy, etc).

        Renderers that do not support color/class styling can ignore
        classification and fall back to plain glyph drawing.
        """
        self.draw_glyph(x, y, glyph)

    @abstractmethod
    def draw_text(self, x: int, y: int, text: str) -> None:
        """Draw a UI string (status line, messages)."""

    @abstractmethod
    def present(self) -> None:
        """Flush the frame to the screen."""

    @abstractmethod
    def poll_action(self) -> str | None:
        """Block for input and return an abstract action name.

        Actions are backend-independent strings: 'move_up', 'move_down',
        'move_left', 'move_right', 'quit', or None for an unrecognised key.
        """
