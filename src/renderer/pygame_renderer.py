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
        self._ui_cell_w = 8
        self._ui_cell_h = 16

        # Screen size in text cells; roomy enough for map + sidebar + menus.
        self._cols = 120
        self._rows = 40

        self._bg = (14, 16, 20)
        self._default_fg = (230, 230, 230)
        self._panel_bg = (24, 28, 36)
        self._panel_border = (88, 108, 132)
        self._splitter = (130, 160, 190)
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
        self._grid_cols = self._cols
        self._grid_rows = self._rows
        self._ui_cols = self._cols
        self._ui_rows = self._rows
        self._sidebar_width_px: int | None = None
        self._sidebar_width_ratio = 0.22
        self._dragging_sidebar = False
        self._splitter_hit_slop_px = 8
        self._cursor_kind = "arrow"
        self._mouse_visible = True
        self._last_mouse_activity_ms = 0
        self._mouse_hide_delay_ms = 2400

    def apply_options(self, options: dict) -> None:
        self._options = dict(options)
        if self._pygame is None:
            return

        self._tile_scale = self._coerce_scale(self._options.get("tile_scale", 1.5))
        self._ui_scale = self._coerce_scale(self._options.get("ui_scale", 1.0))

        self._cell_w = max(1, int(round(self._tile_size * self._tile_scale)))
        self._cell_h = max(1, int(round(self._tile_size * self._tile_scale)))

        self._ui_font_size = max(8, int(round(16 * self._ui_scale)))
        self._font = self._pygame.font.SysFont("DejaVu Sans Mono", self._ui_font_size)
        self._ui_cell_w, self._ui_cell_h = self._font.size("M")
        self._ui_cell_w = max(1, self._ui_cell_w)
        self._ui_cell_h = max(1, self._ui_cell_h)
        try:
            ratio = float(self._options.get("sidebar_width_ratio", 0.22))
        except (TypeError, ValueError):
            ratio = 0.22
        self._sidebar_width_ratio = min(0.5, max(0.14, ratio))

        fullscreen = bool(self._options.get("fullscreen", False))
        if fullscreen:
            self._screen = self._pygame.display.set_mode((0, 0), self._pygame.FULLSCREEN)
        else:
            window_w = max(640, self._cols * self._cell_w)
            window_h = max(480, self._rows * self._cell_h)
            self._screen = self._pygame.display.set_mode((window_w, window_h))

        if self._screen is not None:
            screen_w = self._screen.get_width()
            screen_h = self._screen.get_height()
            if self._sidebar_width_px is None:
                self._sidebar_width_px = int(round(screen_w * self._sidebar_width_ratio))
            self._sidebar_width_px = max(180, min(int(screen_w * 0.6), self._sidebar_width_px))
            self._grid_cols = max(1, screen_w // self._cell_w)
            self._grid_rows = max(1, screen_h // self._cell_h)
            self._ui_cols = max(1, screen_w // self._ui_cell_w)
            self._ui_rows = max(1, screen_h // self._ui_cell_h)

        self._sheet_cache = {}
        self._glyph_tiles = {}
        self._class_tiles = {}
        self._load_tileset_config()

    def get_grid_size(self) -> tuple[int, int]:
        return (self._grid_cols, self._grid_rows)

    def get_ui_grid_size(self) -> tuple[int, int]:
        return (self._ui_cols, self._ui_rows)

    def get_screen_size_px(self) -> tuple[int, int]:
        if self._screen is None:
            return (self._cols * self._cell_w, self._rows * self._cell_h)
        return (self._screen.get_width(), self._screen.get_height())

    def get_tile_cell_size_px(self) -> tuple[int, int]:
        return (self._cell_w, self._cell_h)

    def get_ui_cell_size_px(self) -> tuple[int, int]:
        return (self._ui_cell_w, self._ui_cell_h)

    def get_sidebar_width_px(self, default: int = 320) -> int:
        if self._sidebar_width_px is None:
            return default
        return self._sidebar_width_px

    def get_sidebar_width_cells(self, default: int = 28) -> int:
        if self._sidebar_width_px is None:
            return default
        return max(14, int(round(self._sidebar_width_px / max(1, self._cell_w))))

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

    def _blit_text(
        self,
        x: int,
        y: int,
        text: str,
        color: tuple[int, int, int],
        max_width_px: int | None = None,
    ) -> None:
        if self._screen is None or self._font is None:
            return
        surface = self._font.render(text, True, color)
        if max_width_px is not None:
            if max_width_px <= 0:
                return
            if surface.get_width() > max_width_px:
                clip = self._pygame.Rect(0, 0, max_width_px, surface.get_height())
                surface = surface.subsurface(clip)
        self._screen.blit(surface, (x * self._ui_cell_w, y * self._ui_cell_h))

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

    def draw_text_clipped(self, x: int, y: int, text: str, max_cells: int) -> None:
        if max_cells <= 0:
            return
        max_width_px = max(1, max_cells * self._ui_cell_w - 4)
        self._blit_text(x, y, text, self._default_fg, max_width_px=max_width_px)

    def text_columns_for_cells(self, width_cells: int, padding_px: int = 4) -> int:
        if width_cells <= 0:
            return 1
        usable_px = max(1, width_cells * self._ui_cell_w - max(0, padding_px))
        if self._font is None:
            return max(1, usable_px // max(1, self._ui_cell_w))
        char_w, _char_h = self._font.size("M")
        return max(1, usable_px // max(1, char_w))

    def draw_panel(self, x: int, y: int, width: int, height: int, title: str | None = None) -> None:
        if self._pygame is None or self._screen is None:
            return
        if width <= 0 or height <= 0:
            return

        px = x * self._ui_cell_w
        py = y * self._ui_cell_h
        pw = width * self._ui_cell_w
        ph = height * self._ui_cell_h

        panel_rect = self._pygame.Rect(px, py, pw, ph)
        self._pygame.draw.rect(self._screen, self._panel_bg, panel_rect)
        self._pygame.draw.rect(self._screen, self._panel_border, panel_rect, width=2)

        if title:
            # Draw title text slightly inset so it does not collide with border corners.
            self._blit_text(x + 1, y, f"[{title}]", self._panel_border)

    def present(self) -> None:
        if self._pygame is not None:
            self._pygame.display.flip()

    def _ensure_mouse_visible(self, now_ms: int) -> None:
        if self._pygame is None:
            return
        self._last_mouse_activity_ms = now_ms
        if not self._mouse_visible:
            self._pygame.mouse.set_visible(True)
            self._mouse_visible = True

    def _hide_mouse_if_idle(self, now_ms: int) -> None:
        if self._pygame is None:
            return
        if self._dragging_sidebar:
            return
        if not self._mouse_visible:
            return
        if now_ms - self._last_mouse_activity_ms >= self._mouse_hide_delay_ms:
            self._pygame.mouse.set_visible(False)
            self._mouse_visible = False

    def _sidebar_geometry_px(self) -> tuple[int, int, int, int]:
        if self._screen is None:
            return (0, 0, 0, 0)
        screen_w = self._screen.get_width()
        screen_h = self._screen.get_height()
        sidebar_px = self._sidebar_width_px
        if sidebar_px is None:
            sidebar_px = int(round(screen_w * self._sidebar_width_ratio))
        sidebar_cells = max(14, int(round(sidebar_px / max(1, self._ui_cell_w))))
        snapped_width_px = sidebar_cells * self._ui_cell_w
        left_px = max(0, screen_w - snapped_width_px)
        return (left_px, 0, snapped_width_px, screen_h)

    def _set_cursor(self, kind: str) -> None:
        if self._pygame is None:
            return
        if kind == self._cursor_kind:
            return
        if kind == "resize":
            self._pygame.mouse.set_cursor(self._pygame.SYSTEM_CURSOR_SIZEWE)
            self._cursor_kind = "resize"
            return
        self._pygame.mouse.set_cursor(self._pygame.SYSTEM_CURSOR_ARROW)
        self._cursor_kind = "arrow"

    def _update_cursor_for_splitter(self) -> None:
        if self._screen is None or self._pygame is None:
            return
        if self._dragging_sidebar:
            self._set_cursor("resize")
            return

        left_px, _top_px, _width_px, _height_px = self._sidebar_geometry_px()
        splitter_x = left_px
        mouse_x, _mouse_y = self._pygame.mouse.get_pos()
        if abs(mouse_x - splitter_x) <= self._splitter_hit_slop_px:
            self._set_cursor("resize")
        else:
            self._set_cursor("arrow")

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

                if event.type in {self._pygame.MOUSEMOTION, self._pygame.MOUSEBUTTONDOWN, self._pygame.MOUSEBUTTONUP}:
                    self._ensure_mouse_visible(now)

                if event.type == self._pygame.MOUSEBUTTONDOWN and event.button == 1 and self._screen is not None:
                    left_px, _top_px, _width_px, _height_px = self._sidebar_geometry_px()
                    splitter_x = left_px
                    if abs(event.pos[0] - splitter_x) <= self._splitter_hit_slop_px:
                        self._dragging_sidebar = True
                        self._set_cursor("resize")
                        continue

                if event.type == self._pygame.MOUSEBUTTONUP and event.button == 1:
                    self._dragging_sidebar = False
                    self._update_cursor_for_splitter()
                    continue

                if event.type == self._pygame.MOUSEMOTION and self._dragging_sidebar and self._screen is not None:
                    screen_w = self._screen.get_width()
                    new_sidebar_px = screen_w - event.pos[0]
                    min_px = max(14 * self._ui_cell_w, int(screen_w * 0.14))
                    max_px = int(screen_w * 0.6)
                    self._sidebar_width_px = max(min_px, min(max_px, new_sidebar_px))
                    self._sidebar_width_ratio = self._sidebar_width_px / max(1, screen_w)
                    self._options["sidebar_width_ratio"] = self._sidebar_width_ratio
                    self._pending_actions.append("ui_layout_changed")
                    self._set_cursor("resize")
                    continue

                if event.type == self._pygame.MOUSEMOTION:
                    self._update_cursor_for_splitter()
                    continue

                if event.type == self._pygame.KEYDOWN:
                    if event.key == self._pygame.K_SPACE:
                        if not self._space_held:
                            self._space_held = True
                            self._next_space_repeat_ms = now + self._space_initial_delay_ms
                        self._pending_actions.append("confirm_action")
                        continue

                    if event.key in {self._pygame.K_EQUALS, self._pygame.K_KP_PLUS}:
                        self._pending_actions.append("tile_scale_up")
                        continue
                    if event.key in {self._pygame.K_MINUS, self._pygame.K_KP_MINUS}:
                        self._pending_actions.append("tile_scale_down")
                        continue

                    if event.key in self._keydown_to_action:
                        self._pending_actions.append(self._keydown_to_action[event.key])
                    self._ensure_mouse_visible(now)

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

            self._update_cursor_for_splitter()
            self._hide_mouse_if_idle(now)

            self._pygame.time.wait(8)
