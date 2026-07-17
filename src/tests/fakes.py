from __future__ import annotations

from renderer.base import Renderer


class FakeRenderer(Renderer):
    """Headless renderer test double that captures draw output in memory."""

    def __init__(self):
        self.glyphs: dict[tuple[int, int], str] = {}
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
        self.text = []

    def draw_glyph(self, x: int, y: int, glyph: str) -> None:
        self.glyphs[(x, y)] = glyph

    def draw_text(self, x: int, y: int, text: str) -> None:
        self.text.append((x, y, text))

    def present(self) -> None:
        self.present_calls += 1

    def poll_action(self) -> str | None:
        return None
