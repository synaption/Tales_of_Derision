"""pygame renderer + input.

Renders map/entity glyphs with optional tileset sprites and UI text with a
separate high-resolution font scale.
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

from .base import Renderer


class PygameRenderer(Renderer):
    def __init__(self, options: dict | None = None) -> None:
        self._options = options or {}

        self._pygame = None
        self._screen = None
        self._font = None

        self._tile_size = 16
        self._tile_scale = 1.0
        self._ui_scale = 1.0
        self._ui_font_size = 16

        # Render-cell size used by map and glyph positioning.
        self._cell_w = 16
        self._cell_h = 16

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

        project_root = Path(__file__).resolve().parents[2]
        self._tile_config_path = project_root / "gfx" / "tilesets" / "pygame_tileset_config.json"
        self._sheet_cache: dict[str, object] = {}
        self._glyph_tiles: dict[str, object] = {}
        self._class_tiles: dict[str, object] = {}

    def apply_options(self, options: dict) -> None:
        self._options = dict(options)
        if self._pygame is None:
            return

        self._tile_scale = self._coerce_scale(self._options.get("tile_scale", 2.0))
        self._ui_scale = self._coerce_scale(self._options.get("ui_scale", 1.0))

        self._cell_w = max(1, int(round(self._tile_size * self._tile_scale)))
        self._cell_h = max(1, int(round(self._tile_size * self._tile_scale)))

        self._ui_font_size = max(8, int(round(16 * self._ui_scale)))
        self._font = self._pygame.font.SysFont("DejaVu Sans Mono", self._ui_font_size)

        fullscreen = bool(self._options.get("fullscreen", False))
        if fullscreen:
            self._screen = self._pygame.display.set_mode((0, 0), self._pygame.FULLSCREEN)
        else:
            window_w = max(640, self._cols * self._cell_w)
            window_h = max(480, self._rows * self._cell_h)
            self._screen = self._pygame.display.set_mode((window_w, window_h))

        self._sheet_cache = {}
        self._glyph_tiles = {}
        self._class_tiles = {}
        self._load_tileset_config()

    @staticmethod
    def _coerce_scale(value: object) -> float:
        try:
            scale = float(value)
        except (TypeError, ValueError):
            return 1.0
        if scale < 0.5:
            return 0.5
        if scale > 4.0:
            return 4.0
        return scale

    def _load_tile(self, sheet_path: str, tile_x: int, tile_y: int):
        if self._pygame is None:
            return None

        sheet = self._sheet_cache.get(sheet_path)
        if sheet is None:
            try:
                sheet = self._pygame.image.load(sheet_path).convert_alpha()
                self._sheet_cache[sheet_path] = sheet
            except Exception:
                return None

        rect = self._pygame.Rect(
            tile_x * self._tile_size,
            tile_y * self._tile_size,
            self._tile_size,
            self._tile_size,
        )
        if rect.right > sheet.get_width() or rect.bottom > sheet.get_height():
            return None

        tile = self._pygame.Surface((self._tile_size, self._tile_size), self._pygame.SRCALPHA)
        tile.blit(sheet, (0, 0), rect)

        if self._cell_w != self._tile_size or self._cell_h != self._tile_size:
            tile = self._pygame.transform.scale(tile, (self._cell_w, self._cell_h))

        return tile

    def _load_tileset_config(self) -> None:
        if self._pygame is None:
            return
        if not self._tile_config_path.exists():
            return

        try:
            payload = json.loads(self._tile_config_path.read_text(encoding="utf-8"))
        except Exception:
            return

        if not isinstance(payload, dict):
            return

        tile_size = payload.get("tile_size")
        if isinstance(tile_size, int) and tile_size > 0:
            self._tile_size = tile_size
            self._cell_w = max(1, int(round(self._tile_size * self._tile_scale)))
            self._cell_h = max(1, int(round(self._tile_size * self._tile_scale)))

        default_sheet = payload.get("default_sheet")
        if not isinstance(default_sheet, str):
            default_sheet = ""

        def resolve_sheet(sheet_value: str) -> str:
            sheet_value = sheet_value or default_sheet
            if not sheet_value:
                return ""
            sheet_path = Path(sheet_value)
            if not sheet_path.is_absolute():
                sheet_path = Path(__file__).resolve().parents[2] / sheet_path
            return str(sheet_path)

        self._glyph_tiles = {}
        glyph_payload = payload.get("glyphs", {})
        if isinstance(glyph_payload, dict):
            for glyph, spec in glyph_payload.items():
                if not isinstance(glyph, str) or not isinstance(spec, dict):
                    continue
                sheet_path = resolve_sheet(str(spec.get("sheet", "")))
                tx = spec.get("x")
                ty = spec.get("y")
                if not sheet_path or not isinstance(tx, int) or not isinstance(ty, int):
                    continue
                tile = self._load_tile(sheet_path, tx, ty)
                if tile is not None:
                    self._glyph_tiles[glyph] = tile

        self._class_tiles = {}
        class_payload = payload.get("classifications", {})
        if isinstance(class_payload, dict):
            for classification, spec in class_payload.items():
                if not isinstance(classification, str) or not isinstance(spec, dict):
                    continue
                sheet_path = resolve_sheet(str(spec.get("sheet", "")))
                tx = spec.get("x")
                ty = spec.get("y")
                if not sheet_path or not isinstance(tx, int) or not isinstance(ty, int):
                    continue
                tile = self._load_tile(sheet_path, tx, ty)
                if tile is not None:
                    self._class_tiles[classification] = tile

    def setup(self) -> None:
        import pygame

        pygame.init()
        pygame.display.set_caption("Tales of Derision")

        self._pygame = pygame
        self.apply_options(self._options)

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
        if self._screen is not None and glyph in self._glyph_tiles:
            self._screen.blit(self._glyph_tiles[glyph], (x * self._cell_w, y * self._cell_h))
            return
        self._blit_text(x, y, glyph, self._default_fg)

    def draw_glyph_classified(self, x: int, y: int, glyph: str, classification: str) -> None:
        if self._screen is not None:
            if glyph in self._glyph_tiles:
                self._screen.blit(self._glyph_tiles[glyph], (x * self._cell_w, y * self._cell_h))
                return
            if classification in self._class_tiles:
                self._screen.blit(self._class_tiles[classification], (x * self._cell_w, y * self._cell_h))
                return
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
