from __future__ import annotations

from renderer.base import Renderer


class FakeRenderer(Renderer):
    """Headless renderer test double that captures draw output in memory."""

    def __init__(self):
        self.glyphs: dict[tuple[int, int], str] = {}
        self.classified_glyphs: dict[tuple[int, int], tuple[str, str]] = {}
        self.glyph_colors: dict[tuple[int, int], tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]] = {}
        self.forced_glyphs: dict[tuple[int, int], bool] = {}
        self.text: list[tuple[int, int, str]] = []
        self.present_calls = 0
        self.setup_calls = 0
        self.teardown_calls = 0

    def setup(self) -> None:
        self.setup_calls += 1

    def teardown(self) -> None:
        self.teardown_calls += 1

    def clear(self) -> None:
        self.glyphs = {}
        self.classified_glyphs = {}
        self.glyph_colors = {}
        self.forced_glyphs = {}
        self.text = []

    def draw_glyph(
        self,
        x: int,
        y: int,
        glyph: str,
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
    ) -> None:
        self.glyphs[(x, y)] = glyph
        self.glyph_colors[(x, y)] = (fg, bg)

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
        self.glyphs[(x, y)] = glyph
        self.classified_glyphs[(x, y)] = (glyph, classification)
        self.glyph_colors[(x, y)] = (fg, bg)
        self.forced_glyphs[(x, y)] = force_glyph

    def draw_text(self, x: int, y: int, text: str) -> None:
        self.text.append((x, y, text))

    def present(self) -> None:
        self.present_calls += 1

    def poll_action(self) -> str | None:
        return None
