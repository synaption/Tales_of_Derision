"""Renderer interface.

Everything the game needs from a display backend lives here. The shipped
runtime uses pygame, and this seam keeps systems/test code backend-agnostic.
"""
from abc import ABC, abstractmethod


# --- Remembered-tile ("fog of war") tone ------------------------------------
# A tile the player has seen but no longer has line of sight to is drawn from
# memory: partly desaturated (each colour pulled toward its OWN luminance -- not
# a flat grey, which would wash darks out and lift blacks) then dimmed by a plain
# brightness multiply (so black stays black). These tuning values live here, on
# the neutral renderer seam, so both the backend-agnostic draw logic
# (``memory_color``) and the pygame backend (which applies the identical
# grayscale-blend + multiply to on-screen regions and sprites) share one source
# of truth. Turn these down for a subtler fade, up for a starker one.
MEMORY_DESATURATE = 0.4   # 0 keeps full colour, 1 collapses to pure luminance grey
MEMORY_DIM = 0.55         # brightness multiplier applied after desaturating
MEMORY_FLOOR_FG = (150, 150, 150)  # stand-in for the (colourless) floor tile


def memory_color(rgb: tuple[int, int, int] | None) -> tuple[int, int, int]:
    """Desaturate and dim a colour into its 'remembered' tone: pull it toward its
    own luminance by ``MEMORY_DESATURATE`` then multiply brightness by
    ``MEMORY_DIM``. A colourless input (``None``, e.g. the plain floor) is treated
    as neutral grey first so it, too, reads as dim memory rather than disappearing.
    Mirrors the pygame region/sprite transform so both draw paths agree."""
    r, g, b = rgb if rgb is not None else MEMORY_FLOOR_FG
    lum = 0.3 * r + 0.59 * g + 0.11 * b

    def blend(channel: float) -> int:
        desaturated = channel + (lum - channel) * MEMORY_DESATURATE
        return max(0, min(255, int(desaturated * MEMORY_DIM)))

    return (blend(r), blend(g), blend(b))


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
    def draw_glyph(
        self,
        x: int,
        y: int,
        glyph: str,
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
    ) -> None:
        """Draw a single character at map cell (x, y)."""

    def draw_glyph_classified(
        self,
        x: int,
        y: int,
        glyph: str,
        classification: str,
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        force_glyph: bool = False,
    ) -> None:
        """Draw a glyph with semantic classification (wall, enemy, etc).

        Renderers that do not support color/class styling can ignore
        classification and fall back to plain glyph drawing. ``force_glyph``
        asks the renderer to draw the literal glyph rather than any sprite the
        classification maps to (used for status identifiers like "~").
        """
        self.draw_glyph(x, y, glyph, fg=fg, bg=bg)

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
