"""pygame renderer + input.

Renders map/entity glyphs with optional tileset sprites and UI text with a
separate high-resolution font scale.
"""

from __future__ import annotations

import csv
from collections import deque
import json
from pathlib import Path
import sys

from .base import Renderer


# Under pygbag the browser canvas fills the page via CSS (width/height: 100%),
# so the pygame surface resolution must match the visible canvas or the browser
# scales it and the game looks wrong-sized.
IS_WEB = sys.platform == "emscripten"


def _apply_web_canvas_style() -> None:
    """Tell the browser to scale the canvas with nearest-neighbor sampling.

    The canvas is CSS-scaled to fit the page; the browser's default smoothing
    blurs upscaled pixels and text. `image-rendering: pixelated` keeps them crisp.
    """
    try:
        import platform as _platform  # pygbag-injected; has .window.canvas on web

        _platform.window.canvas.style.imageRendering = "pixelated"
    except Exception:
        pass


def _web_display_size(default: tuple[int, int] = (1280, 720)) -> tuple[int, int]:
    """Visible browser canvas size (CSS pixels) for the pygbag build.

    Falls back to a sane default off-web or if the JS bridge is unavailable.
    """
    try:
        import platform as _platform  # pygbag injects a `window` proxy here

        width = int(_platform.window.innerWidth)
        height = int(_platform.window.innerHeight)
        if width > 0 and height > 0:
            return (width, height)
    except Exception:
        pass
    return default


_DEFAULT_ACTION_KEYBINDS: dict[str, list[str]] = {
    "move_up": ["w"],
    "move_down": ["s"],
    "move_left": ["a"],
    "move_right": ["d"],
    "confirm_action": ["space"],
    "menu_select": ["enter", "kp_enter"],
    "open_menu": ["tab"],
    "open_inventory": ["i"],
    "open_status": ["c"],
    "look": ["l"],
    "sleep": ["r"],
    "open_pause_menu": ["esc"],
    "tile_scale_up": ["equals", "kp_plus"],
    "tile_scale_down": ["minus", "kp_minus"],
}

_LEGACY_KEYBIND_ALIASES = {
    "up": "move_up",
    "down": "move_down",
    "left": "move_left",
    "right": "move_right",
}

_KEY_NAME_ALIASES = {
    "esc": "escape",
    "enter": "return",
    "plus": "equals",
    "+": "equals",
    "-": "minus",
    "numpad_enter": "kp_enter",
    "numpad_plus": "kp_plus",
    "numpad_minus": "kp_minus",
}

_MOVE_TO_RELEASE_ACTION = {
    "move_up": "release_up",
    "move_down": "release_down",
    "move_left": "release_left",
    "move_right": "release_right",
}


def _normalise_keybind_values(raw_value: object) -> list[str | int]:
    if isinstance(raw_value, (str, int)):
        return [raw_value]
    if isinstance(raw_value, list):
        values: list[str | int] = []
        for item in raw_value:
            if isinstance(item, (str, int)):
                values.append(item)
        return values
    return []


def _key_code_from_binding(pygame_module: object, binding: str | int) -> int | None:
    if isinstance(binding, int):
        return binding

    key_name = binding.strip().lower()
    if not key_name:
        return None

    key_name = _KEY_NAME_ALIASES.get(key_name, key_name)

    for attr_name in (f"K_{key_name}", f"K_{key_name.upper()}"):
        key_code = getattr(pygame_module, attr_name, None)
        if isinstance(key_code, int):
            return key_code

    pygame_key_module = getattr(pygame_module, "key", None)
    key_code_func = getattr(pygame_key_module, "key_code", None)
    if callable(key_code_func):
        try:
            return int(key_code_func(key_name))
        except Exception:
            return None

    return None


def _build_key_mappings(
    pygame_module: object,
    options: dict | None = None,
) -> tuple[dict[int, str], dict[int, str]]:
    resolved_action_bindings: dict[str, list[str | int]] = {
        action: list(bindings)
        for action, bindings in _DEFAULT_ACTION_KEYBINDS.items()
    }
    customised_actions: set[str] = set()

    raw_keybinds = None
    if isinstance(options, dict):
        raw_keybinds = options.get("keybinds")

    if isinstance(raw_keybinds, dict):
        for raw_action, raw_bindings in raw_keybinds.items():
            if not isinstance(raw_action, str):
                continue

            canonical_action = raw_action.strip().lower()
            canonical_action = _LEGACY_KEYBIND_ALIASES.get(canonical_action, canonical_action)
            if canonical_action not in resolved_action_bindings:
                continue

            normalised_bindings = _normalise_keybind_values(raw_bindings)
            if normalised_bindings:
                resolved_action_bindings[canonical_action] = normalised_bindings
                customised_actions.add(canonical_action)

    keydown_to_action: dict[int, str] = {}
    keyup_to_action: dict[int, str] = {}
    for action_name in _DEFAULT_ACTION_KEYBINDS:
        for binding in resolved_action_bindings[action_name]:
            key_code = _key_code_from_binding(pygame_module, binding)
            if key_code is None:
                continue

            if key_code in keydown_to_action:
                previous_action = keydown_to_action[key_code]
                previous_is_custom = previous_action in customised_actions
                current_is_custom = action_name in customised_actions
                if previous_is_custom and not current_is_custom:
                    continue

            keydown_to_action[key_code] = action_name
            release_action = _MOVE_TO_RELEASE_ACTION.get(action_name)
            if release_action is not None:
                keyup_to_action[key_code] = release_action
            else:
                keyup_to_action.pop(key_code, None)

    return keydown_to_action, keyup_to_action


def _coerce_rgb(value: object, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if (
        isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(channel, int) for channel in value)
    ):
        return tuple(max(0, min(255, int(channel))) for channel in value)
    return fallback


def _rgb_to_hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


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

        self._bg = (0, 0, 0)
        self._default_fg = (224, 224, 224)
        self._panel_bg = (0, 0, 0)
        self._panel_border = (118, 118, 118)
        self._panel_header = (22, 22, 22)
        self._menu_backdrop = (0, 0, 0)
        self._menu_scanline = (10, 10, 10)
        self._splitter = (96, 96, 96)
        self._class_colors = {
            "default": (224, 224, 224),
            "wall": (122, 122, 122),
            "stairs": (186, 186, 186),
            "friendly": (240, 240, 240),
            "enemy": (202, 202, 202),
            "valuable": (214, 214, 214),
        }

        self._keydown_to_action = {}
        self._keyup_to_action = {}
        self._confirm_keys: set[int] = set()
        self._confirm_held_key: int | None = None
        self._confirm_initial_delay_ms = 180
        self._confirm_repeat_interval_ms = 70
        self._next_confirm_repeat_ms = 0
        self._pending_actions: deque[str] = deque()

        project_root = Path(__file__).resolve().parents[2]
        self._tile_config_path = project_root / "gfx" / "tilesets" / "pygame_tileset_config.json"
        self._fallback_sheet_path = str(project_root / "gfx" / "tilesets" / "Bisasam_16x16.png")
        self._fallback_tile_size = 16
        self._fallback_sheet_columns = 16
        self._fallback_sheet_rows = 16
        self._fallback_tint_with_fg = True
        self._fallback_fill_bg = True
        self._sheet_cache: dict[str, object] = {}
        self._glyph_tiles: dict[str, object] = {}
        self._class_tiles: dict[str, object] = {}
        self._tile_index_lookup: dict[str, tuple[str, int, int, int | None]] = {}
        self._autotile_tiles: dict[str, dict[int, object]] = {"wall": {}, "water": {}, "ledges": {}}
        self._fallback_tiles: dict[
            tuple[str, str, tuple[int, int, int], tuple[int, int, int], int, int],
            object,
        ] = {}
        self._tinted_tiles: dict[tuple[int, tuple[int, int, int]], object] = {}
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
        # Cached game-scene snapshot reused as a static menu backdrop (see
        # capture_backdrop). Avoids a full esper.process() render per menu frame.
        self._backdrop_snapshot = None
        # Cached fully-drawn map; walking blits regions of it instead of
        # re-drawing every visible tile (see capture_map_surface).
        self._map_surface = None

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
        if IS_WEB:
            # Browser "fullscreen" isn't a real display mode. Match the surface to
            # the visible canvas so 1 surface px == 1 CSS px; otherwise the browser
            # rescales an oversized surface and the whole game looks tiny.
            window_w, window_h = _web_display_size()
            self._screen = self._pygame.display.set_mode((window_w, window_h))
            _apply_web_canvas_style()
        elif fullscreen:
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
        self._tile_index_lookup = {}
        self._autotile_tiles = {"wall": {}, "water": {}, "ledges": {}}
        self._fallback_tiles = {}
        self._tinted_tiles = {}
        # Screen was just re-created (set_mode); cached surfaces are now the
        # wrong size / cell scale.
        self._backdrop_snapshot = None
        self._map_surface = None
        self._load_tileset_config()

        self._keydown_to_action, self._keyup_to_action = _build_key_mappings(self._pygame, self._options)
        self._confirm_keys = {
            key_code
            for key_code, action_name in self._keydown_to_action.items()
            if action_name == "confirm_action"
        }
        if not self._confirm_keys and hasattr(self._pygame, "K_SPACE"):
            self._confirm_keys = {self._pygame.K_SPACE}

        if self._confirm_held_key is not None and self._confirm_held_key not in self._confirm_keys:
            self._confirm_held_key = None

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

    def _load_tile(
        self,
        sheet_path: str,
        tile_x: int,
        tile_y: int,
        tile_size: int | None = None,
    ):
        if self._pygame is None:
            return None

        sample_size = tile_size if isinstance(tile_size, int) and tile_size > 0 else self._tile_size

        sheet = self._sheet_cache.get(sheet_path)
        if sheet is None:
            try:
                sheet = self._pygame.image.load(sheet_path).convert_alpha()
                self._sheet_cache[sheet_path] = sheet
            except Exception:
                return None

        rect = self._pygame.Rect(
            tile_x * sample_size,
            tile_y * sample_size,
            sample_size,
            sample_size,
        )
        if rect.right > sheet.get_width() or rect.bottom > sheet.get_height():
            return None

        tile = self._pygame.Surface((sample_size, sample_size), self._pygame.SRCALPHA)
        tile.blit(sheet, (0, 0), rect)

        if self._cell_w != sample_size or self._cell_h != sample_size:
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

        sheet_aliases: dict[str, str] = {}
        raw_sheet_aliases = payload.get("sheet_paths")
        if isinstance(raw_sheet_aliases, dict):
            for alias_name, alias_value in raw_sheet_aliases.items():
                if not isinstance(alias_name, str) or not isinstance(alias_value, str):
                    continue
                resolved = resolve_sheet(alias_value)
                if resolved:
                    sheet_aliases[alias_name] = resolved

        def resolve_sheet_alias(sheet_value: str) -> str:
            sheet_value = (sheet_value or "").strip()
            if not sheet_value:
                return resolve_sheet(sheet_value)

            if sheet_value in sheet_aliases:
                return sheet_aliases[sheet_value]

            if tile_index_path is not None:
                index_dir = tile_index_path.parent
                for suffix in ("_transparent.png", ".png"):
                    candidate = index_dir / f"{sheet_value}{suffix}"
                    if candidate.exists():
                        return str(candidate)

            return resolve_sheet(sheet_value)

        tile_index_path: Path | None = None
        raw_tile_index = payload.get("tile_index_csv")
        if isinstance(raw_tile_index, str) and raw_tile_index.strip():
            candidate = Path(raw_tile_index.strip())
            if not candidate.is_absolute():
                candidate = Path(__file__).resolve().parents[2] / candidate
            tile_index_path = candidate
        elif default_sheet:
            default_sheet_path = Path(resolve_sheet(default_sheet))
            inferred = default_sheet_path.parent / "tile_index.csv"
            if inferred.exists():
                tile_index_path = inferred

        sheet_column_cache: dict[tuple[str, int], int] = {}

        def sheet_columns(sheet_name: str, sample_size: int | None = None) -> int | None:
            sheet_path = resolve_sheet_alias(sheet_name)
            if not sheet_path:
                return None

            tile_px = sample_size if isinstance(sample_size, int) and sample_size > 0 else self._tile_size
            key = (sheet_path, tile_px)
            cached = sheet_column_cache.get(key)
            if cached is not None:
                return cached

            sheet = self._sheet_cache.get(sheet_path)
            if sheet is None:
                try:
                    sheet = self._pygame.image.load(sheet_path).convert_alpha()
                    self._sheet_cache[sheet_path] = sheet
                except Exception:
                    return None

            cols = sheet.get_width() // max(1, tile_px)
            if cols <= 0:
                return None

            sheet_column_cache[key] = cols
            return cols

        tile_index: dict[str, tuple[str, int, int, int | None]] = {}
        if tile_index_path is not None and tile_index_path.exists():
            try:
                with tile_index_path.open("r", encoding="utf-8", newline="") as index_file:
                    for row in csv.DictReader(index_file):
                        if not isinstance(row, dict):
                            continue

                        sample_size: int | None = None
                        width_raw = row.get("w")
                        try:
                            if width_raw is not None and str(width_raw).strip():
                                parsed = int(width_raw)
                                if parsed > 0:
                                    sample_size = parsed
                        except (TypeError, ValueError):
                            sample_size = None

                        legacy_tile_id = str(row.get("tile_id", "")).strip()
                        legacy_sheet_name = str(row.get("sheet", "")).strip()
                        legacy_col = row.get("col")
                        legacy_row = row.get("row")
                        if legacy_tile_id and legacy_sheet_name:
                            try:
                                tile_x = int(legacy_col)
                                tile_y = int(legacy_row)
                            except (TypeError, ValueError):
                                continue
                            tile_index[legacy_tile_id] = (legacy_sheet_name, tile_x, tile_y, sample_size)
                            tile_number = tile_y * max(1, sheet_columns(legacy_sheet_name, sample_size) or 1) + tile_x
                            tile_index[f"{legacy_sheet_name}:{tile_number}"] = (
                                legacy_sheet_name,
                                tile_x,
                                tile_y,
                                sample_size,
                            )
                            tile_name = str(row.get("tile_name", row.get("cv_label", ""))).strip().lower()
                            if tile_name:
                                tile_index[f"{legacy_sheet_name}/{tile_name}"] = (
                                    legacy_sheet_name,
                                    tile_x,
                                    tile_y,
                                    sample_size,
                                )
                            continue

                        simple_sheet_name = str(row.get("tilesheet", row.get("sheet", ""))).strip()
                        if not simple_sheet_name:
                            continue

                        tile_number_raw = row.get("tile_number")
                        try:
                            tile_number = int(str(tile_number_raw).strip())
                        except (TypeError, ValueError):
                            continue

                        cols = sheet_columns(simple_sheet_name, sample_size)
                        if cols is None or cols <= 0:
                            continue

                        tile_x = tile_number % cols
                        tile_y = tile_number // cols
                        tile_index[f"{simple_sheet_name}:{tile_number}"] = (
                            simple_sheet_name,
                            tile_x,
                            tile_y,
                            sample_size,
                        )
                        tile_index[f"{simple_sheet_name}_{tile_y:02d}_{tile_x:02d}"] = (
                            simple_sheet_name,
                            tile_x,
                            tile_y,
                            sample_size,
                        )

                        tile_name = str(row.get("tile_name", "")).strip().lower()
                        if tile_name:
                            tile_index[f"{simple_sheet_name}/{tile_name}"] = (
                                simple_sheet_name,
                                tile_x,
                                tile_y,
                                sample_size,
                            )
            except Exception:
                tile_index = {}

        self._tile_index_lookup = dict(tile_index)
        self._autotile_tiles = {"wall": {}, "water": {}, "ledges": {}}
        for key, indexed in tile_index.items():
            if "/" not in key:
                continue

            sheet_name, tile_name = key.split("/", 1)
            tile_name = tile_name.strip().lower()

            base_name: str | None = None
            for base in ("wall", "water", "ledges"):
                if tile_name.startswith(base):
                    base_name = base
                    break
            if base_name is None:
                continue

            suffix = tile_name[len(base_name) :]
            if not suffix.isdigit():
                continue

            mask_value = int(suffix)
            if mask_value < 0 or mask_value > 255:
                continue

            indexed_sheet, tile_x, tile_y, sample_size = indexed
            sheet_path = resolve_sheet_alias(indexed_sheet or sheet_name)
            if not sheet_path:
                continue

            tile = self._load_tile(sheet_path, tile_x, tile_y, sample_size)
            if tile is None:
                continue

            self._autotile_tiles[base_name][mask_value] = tile

        def load_tile_from_spec(spec: dict) -> object | None:
            tile_id = spec.get("tile_id")
            if isinstance(tile_id, str):
                indexed = tile_index.get(tile_id.strip())
                if indexed is not None:
                    sheet_name, tile_x, tile_y, sample_size = indexed
                    sheet_path = resolve_sheet_alias(sheet_name)
                    if sheet_path:
                        tile = self._load_tile(sheet_path, tile_x, tile_y, sample_size)
                        if tile is not None:
                            return tile

            sheet_name = str(spec.get("sheet", "")).strip()

            tile_number = spec.get("tile_number")
            if isinstance(tile_number, int) and sheet_name:
                indexed = tile_index.get(f"{sheet_name}:{tile_number}")
                if indexed is not None:
                    indexed_sheet, tile_x, tile_y, sample_size = indexed
                    sheet_path = resolve_sheet_alias(indexed_sheet)
                    if sheet_path:
                        tile = self._load_tile(sheet_path, tile_x, tile_y, sample_size)
                        if tile is not None:
                            return tile
                cols = sheet_columns(sheet_name, None)
                if cols is not None and cols > 0:
                    tile_x = tile_number % cols
                    tile_y = tile_number // cols
                    sheet_path = resolve_sheet_alias(sheet_name)
                    if sheet_path:
                        tile = self._load_tile(sheet_path, tile_x, tile_y, None)
                        if tile is not None:
                            return tile

            tile_name = spec.get("tile_name")
            if isinstance(tile_name, str) and sheet_name:
                indexed = tile_index.get(f"{sheet_name}/{tile_name.strip().lower()}")
                if indexed is not None:
                    indexed_sheet, tile_x, tile_y, sample_size = indexed
                    sheet_path = resolve_sheet_alias(indexed_sheet)
                    if sheet_path:
                        tile = self._load_tile(sheet_path, tile_x, tile_y, sample_size)
                        if tile is not None:
                            return tile

            sheet_path = ""
            raw_sheet = spec.get("sheet")
            if isinstance(raw_sheet, str):
                sheet_path = resolve_sheet_alias(raw_sheet)
            elif default_sheet:
                sheet_path = resolve_sheet(default_sheet)

            tile_x = spec.get("x")
            tile_y = spec.get("y")
            if not isinstance(tile_x, int) or not isinstance(tile_y, int):
                tile_x = spec.get("col")
                tile_y = spec.get("row")

            sample_size = spec.get("tile_size")
            sample_size_value = sample_size if isinstance(sample_size, int) and sample_size > 0 else None

            if not sheet_path or not isinstance(tile_x, int) or not isinstance(tile_y, int):
                return None
            return self._load_tile(sheet_path, tile_x, tile_y, sample_size_value)

        self._glyph_tiles = {}
        glyph_payload = payload.get("glyphs", {})
        if isinstance(glyph_payload, dict):
            for glyph, spec in glyph_payload.items():
                if not isinstance(glyph, str) or not isinstance(spec, dict):
                    continue
                tile = load_tile_from_spec(spec)
                if tile is not None:
                    self._glyph_tiles[glyph] = tile

        self._class_tiles = {}
        class_payload = payload.get("classifications", {})
        if isinstance(class_payload, dict):
            for classification, spec in class_payload.items():
                if not isinstance(classification, str) or not isinstance(spec, dict):
                    continue
                tile = load_tile_from_spec(spec)
                if tile is not None:
                    self._class_tiles[classification] = tile

        fallback_payload = payload.get("fallback")
        if isinstance(fallback_payload, dict):
            fallback_sheet = fallback_payload.get("sheet")
            if isinstance(fallback_sheet, str) and fallback_sheet.strip():
                resolved = resolve_sheet(fallback_sheet.strip())
                if resolved:
                    self._fallback_sheet_path = resolved

            tile_size_value = fallback_payload.get("tile_size")
            if isinstance(tile_size_value, int) and tile_size_value > 0:
                self._fallback_tile_size = tile_size_value

            col_value = fallback_payload.get("columns")
            if isinstance(col_value, int) and col_value > 0:
                self._fallback_sheet_columns = col_value

            row_value = fallback_payload.get("rows")
            if isinstance(row_value, int) and row_value > 0:
                self._fallback_sheet_rows = row_value

            if "tint_with_fg" in fallback_payload:
                self._fallback_tint_with_fg = bool(fallback_payload.get("tint_with_fg"))

            if "use_bg_fill" in fallback_payload:
                self._fallback_fill_bg = bool(fallback_payload.get("use_bg_fill"))

    def setup(self) -> None:
        pygame = __import__("pygame")

        pygame.init()
        pygame.display.set_caption("Tales of Derision")

        self._pygame = pygame
        self.apply_options(self._options)

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

    @staticmethod
    def _style_lookup_key(token: str, fg: tuple[int, int, int], bg: tuple[int, int, int]) -> str:
        return f"{token}|{_rgb_to_hex(fg)}|{_rgb_to_hex(bg)}"

    def _fill_glyph_background(self, x: int, y: int, bg: tuple[int, int, int]) -> None:
        if self._pygame is None or self._screen is None:
            return
        rect = self._pygame.Rect(x * self._cell_w, y * self._cell_h, self._cell_w, self._cell_h)
        self._pygame.draw.rect(self._screen, bg, rect)

    def _blit_world_tile(
        self,
        x: int,
        y: int,
        tile: object,
        bg: tuple[int, int, int] | None = None,
    ) -> None:
        if self._pygame is None or self._screen is None:
            return

        fill_color = _coerce_rgb(bg, self._bg) if bg is not None else self._bg
        rect = self._pygame.Rect(x * self._cell_w, y * self._cell_h, self._cell_w, self._cell_h)
        self._pygame.draw.rect(self._screen, fill_color, rect)
        self._screen.blit(tile, (x * self._cell_w, y * self._cell_h))

    def _load_fallback_tile(
        self,
        glyph: str,
        fg: tuple[int, int, int],
        bg: tuple[int, int, int],
    ):
        if self._pygame is None:
            return None
        if not self._fallback_sheet_path:
            return None

        glyph_char = glyph[0] if glyph else " "
        cache_key = (self._fallback_sheet_path, glyph_char, fg, bg, self._cell_w, self._cell_h)
        cached = self._fallback_tiles.get(cache_key)
        if cached is not None:
            return cached

        columns = max(1, self._fallback_sheet_columns)
        rows = max(1, self._fallback_sheet_rows)
        max_glyphs = columns * rows

        codepoint = ord(glyph_char)
        if codepoint < 0 or codepoint >= max_glyphs:
            codepoint = ord("?") % max_glyphs

        tile_x = codepoint % columns
        tile_y = (codepoint // columns) % rows

        tile = self._load_tile(
            self._fallback_sheet_path,
            tile_x,
            tile_y,
            self._fallback_tile_size,
        )
        if tile is None:
            return None

        tile = tile.copy()
        if self._fallback_tint_with_fg:
            tint = self._pygame.Surface((tile.get_width(), tile.get_height()), self._pygame.SRCALPHA)
            tint.fill((fg[0], fg[1], fg[2], 255))
            tile.blit(tint, (0, 0), special_flags=self._pygame.BLEND_RGBA_MULT)

        if self._fallback_fill_bg:
            with_bg = self._pygame.Surface((tile.get_width(), tile.get_height()), self._pygame.SRCALPHA)
            with_bg.fill((bg[0], bg[1], bg[2], 255))
            with_bg.blit(tile, (0, 0))
            tile = with_bg

        self._fallback_tiles[cache_key] = tile
        return tile

    def _tint_tile(self, tile: object, fg: tuple[int, int, int]):
        if self._pygame is None:
            return tile

        tint_key = (id(tile), fg)
        cached = self._tinted_tiles.get(tint_key)
        if cached is not None:
            return cached

        tinted = tile.copy()
        tint_surface = self._pygame.Surface((tile.get_width(), tile.get_height()), self._pygame.SRCALPHA)
        tint_surface.fill((fg[0], fg[1], fg[2], 255))
        tinted.blit(tint_surface, (0, 0), special_flags=self._pygame.BLEND_RGBA_MULT)
        self._tinted_tiles[tint_key] = tinted
        return tinted

    def _resolve_tile_surface(
        self,
        glyph: str,
        classification: str | None,
        fg: tuple[int, int, int],
        bg: tuple[int, int, int],
    ) -> tuple[object | None, str]:
        style_key = self._style_lookup_key(glyph, fg, bg)
        tile = self._glyph_tiles.get(style_key)
        if tile is not None:
            return (tile, "styled")

        tile = self._glyph_tiles.get(glyph)
        if tile is None:
            tile = None
        else:
            return (tile, "glyph")

        if classification is not None:
            class_style_key = self._style_lookup_key(classification, fg, bg)
            tile = self._class_tiles.get(class_style_key)
            if tile is not None:
                return (tile, "styled")

            tile = self._class_tiles.get(classification)
            if tile is None:
                tile = None
            else:
                return (tile, "class")

            default_style_key = self._style_lookup_key("default", fg, bg)
            tile = self._class_tiles.get(default_style_key)
            if tile is not None:
                return (tile, "styled")

            tile = self._class_tiles.get("default")
            if tile is None:
                tile = None
            else:
                return (tile, "class")

        fallback_tile = self._load_fallback_tile(glyph, fg, bg)
        if fallback_tile is not None:
            return (fallback_tile, "fallback")
        return (None, "none")

    def draw_glyph(
        self,
        x: int,
        y: int,
        glyph: str,
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
    ) -> None:
        explicit_fg = fg is not None
        resolved_fg = _coerce_rgb(fg, self._default_fg)
        resolved_bg = _coerce_rgb(bg, self._bg)

        if bg is not None:
            self._fill_glyph_background(x, y, resolved_bg)

        tile, source = self._resolve_tile_surface(glyph, None, resolved_fg, resolved_bg)
        if self._screen is not None and tile is not None:
            draw_tile = tile
            if explicit_fg and source in {"glyph", "class"}:
                draw_tile = self._tint_tile(tile, resolved_fg)
            tile_bg = resolved_bg if bg is not None else None
            self._blit_world_tile(x, y, draw_tile, tile_bg)
            return

        self._blit_text(x, y, glyph, resolved_fg)

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
        explicit_fg = fg is not None
        if fg is None:
            resolved_fg = self._class_colors.get(classification, self._default_fg)
        else:
            resolved_fg = _coerce_rgb(fg, self._default_fg)
        resolved_bg = _coerce_rgb(bg, self._bg)

        if bg is not None:
            self._fill_glyph_background(x, y, resolved_bg)

        # force_glyph renders the literal glyph (e.g. a status identifier like
        # "~"): resolve with no classification so it can't fall back to the
        # classification's sprite, which would otherwise mask the glyph.
        resolve_classification = None if force_glyph else classification
        tile, source = self._resolve_tile_surface(glyph, resolve_classification, resolved_fg, resolved_bg)
        if self._screen is not None and tile is not None:
            draw_tile = tile
            if explicit_fg and source in {"glyph", "class"}:
                draw_tile = self._tint_tile(tile, resolved_fg)
            tile_bg = resolved_bg if bg is not None else None
            self._blit_world_tile(x, y, draw_tile, tile_bg)
            return

        self._blit_text(x, y, glyph, resolved_fg)

    def draw_autotile_variant(
        self,
        x: int,
        y: int,
        base_name: str,
        mask_value: int,
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
    ) -> bool:
        normalized_base = (base_name or "").strip().lower()
        bank = self._autotile_tiles.get(normalized_base, {})
        if not bank:
            return False

        tile = bank.get(mask_value)
        if tile is None:
            tile = bank.get(0)

        if tile is None:
            best_key = min(
                bank.keys(),
                key=lambda cand: (
                    (mask_value & ~cand).bit_count(),
                    (cand & ~mask_value).bit_count(),
                    -(mask_value & cand).bit_count(),
                    cand,
                ),
            )
            tile = bank.get(best_key)

        if tile is None:
            return False

        draw_tile = tile
        if fg is not None:
            draw_tile = self._tint_tile(tile, _coerce_rgb(fg, self._default_fg))

        resolved_bg = _coerce_rgb(bg, self._bg)
        tile_bg = resolved_bg if bg is not None else None
        self._blit_world_tile(x, y, draw_tile, tile_bg)
        return True

    def draw_ui_glyph(
        self,
        x: int,
        y: int,
        glyph: str,
        classification: str = "default",
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        cell_span: int = 2,
    ) -> bool:
        if self._pygame is None or self._screen is None:
            return False

        explicit_fg = fg is not None
        if fg is None:
            resolved_fg = self._class_colors.get(classification, self._default_fg)
        else:
            resolved_fg = _coerce_rgb(fg, self._default_fg)
        resolved_bg = _coerce_rgb(bg, self._bg)

        span = max(1, int(cell_span))
        slot_x = x * self._ui_cell_w
        slot_y = y * self._ui_cell_h
        slot_w = max(1, span * self._ui_cell_w)
        slot_h = max(1, self._ui_cell_h)

        if bg is not None:
            bg_rect = self._pygame.Rect(slot_x, slot_y, slot_w, slot_h)
            self._pygame.draw.rect(self._screen, resolved_bg, bg_rect)

        tile, source = self._resolve_tile_surface(glyph, classification, resolved_fg, resolved_bg)
        if tile is None:
            return False

        if explicit_fg and source in {"glyph", "class"}:
            tile = self._tint_tile(tile, resolved_fg)

        icon_size = max(1, min(slot_w, slot_h))
        icon = tile
        if tile.get_width() != icon_size or tile.get_height() != icon_size:
            icon = self._pygame.transform.scale(tile, (icon_size, icon_size))

        icon_x = slot_x + max(0, (slot_w - icon_size) // 2)
        icon_y = slot_y + max(0, (slot_h - icon_size) // 2)
        self._screen.blit(icon, (icon_x, icon_y))
        return True

    def draw_text(self, x: int, y: int, text: str) -> None:
        self._blit_text(x, y, text, self._default_fg)

    def draw_text_tinted(self, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
        self._blit_text(x, y, text, color)

    def fill_cells(self, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
        if self._pygame is None or self._screen is None:
            return
        if width <= 0 or height <= 0:
            return

        px = x * self._ui_cell_w
        py = y * self._ui_cell_h
        pw = width * self._ui_cell_w
        ph = height * self._ui_cell_h
        rect = self._pygame.Rect(px, py, pw, ph)
        self._pygame.draw.rect(self._screen, color, rect)

    def draw_menu_backdrop(self) -> None:
        if self._pygame is None or self._screen is None:
            return

        self._screen.fill(self._menu_backdrop)
        step = max(2, self._ui_cell_h // 2)
        for py in range(0, self._screen.get_height(), step):
            self._pygame.draw.line(
                self._screen,
                self._menu_scanline,
                (0, py),
                (self._screen.get_width(), py),
                width=1,
            )

    def draw_overlay(self, color: tuple[int, int, int], alpha: int = 144) -> None:
        if self._pygame is None or self._screen is None:
            return

        clamped_alpha = max(0, min(255, int(alpha)))
        overlay = self._pygame.Surface(
            (self._screen.get_width(), self._screen.get_height()),
            self._pygame.SRCALPHA,
        )
        overlay.fill((color[0], color[1], color[2], clamped_alpha))
        self._screen.blit(overlay, (0, 0))

    def capture_backdrop(self) -> None:
        """Snapshot the current screen so menus can reuse it as a static
        backdrop instead of re-rendering the whole game every keypress."""
        if self._screen is not None:
            self._backdrop_snapshot = self._screen.copy()

    def has_backdrop(self) -> bool:
        return self._backdrop_snapshot is not None

    def blit_backdrop(self) -> bool:
        if self._screen is None or self._backdrop_snapshot is None:
            return False
        self._screen.blit(self._backdrop_snapshot, (0, 0))
        return True

    def invalidate_backdrop(self) -> None:
        self._backdrop_snapshot = None

    def build_map_surface(self, cols: int, rows: int, draw_callback) -> None:
        """Render the whole map once to an off-screen, world-sized surface.

        ``draw_callback`` draws every tile at its WORLD cell position (vx==wx,
        vy==wy). Caching in world coordinates (not view coordinates) keeps the
        cache valid when the camera scrolls at higher zoom -- later frames blit
        the visible region with the scroll offset applied (see blit_map_region).
        """
        if self._pygame is None:
            return
        surface = self._pygame.Surface((max(1, cols) * self._cell_w, max(1, rows) * self._cell_h))
        surface.fill(self._bg)
        saved_screen = self._screen
        self._screen = surface
        try:
            draw_callback()
        finally:
            self._screen = saved_screen
        self._map_surface = surface

    def has_map_surface(self) -> bool:
        return self._map_surface is not None

    def invalidate_map_surface(self) -> None:
        self._map_surface = None

    def redraw_map_cells(self, cells, draw_cell) -> bool:
        """Repaint just ``cells`` (world (x, y) tuples) on the cached map surface,
        leaving the rest untouched. ``draw_cell(wx, wy)`` renders one tile at its
        world cell. Cheap incremental update for a handful of edited tiles, versus
        rebuilding the whole world surface. No-op (returns False) if no surface is
        cached yet."""
        if self._pygame is None or self._map_surface is None:
            return False
        saved_screen = self._screen
        self._screen = self._map_surface
        try:
            for wx, wy in cells:
                rect = self._pygame.Rect(
                    wx * self._cell_w, wy * self._cell_h, self._cell_w, self._cell_h
                )
                self._map_surface.fill(self._bg, rect)
                draw_cell(wx, wy)
        finally:
            self._screen = saved_screen
        return True

    def blit_map_region(
        self,
        world_x: int,
        world_y: int,
        w_cells: int,
        h_cells: int,
        origin_x: int = 0,
        origin_y: int = 0,
    ) -> bool:
        """Blit a world-cell block of the cached map to the screen, offset by the
        current view origin (camera scroll). pygame clips a negative/overflowing
        destination automatically."""
        if self._screen is None or self._map_surface is None:
            return False
        src = self._pygame.Rect(
            world_x * self._cell_w,
            world_y * self._cell_h,
            max(0, w_cells) * self._cell_w,
            max(0, h_cells) * self._cell_h,
        )
        dest = ((world_x - origin_x) * self._cell_w, (world_y - origin_y) * self._cell_h)
        self._screen.blit(self._map_surface, dest, src)
        return True

    def fill_cell_bg(self, vx: int, vy: int) -> None:
        """Paint a single map cell with the background (used to hide FOV shadows)."""
        if self._screen is None:
            return
        rect = self._pygame.Rect(vx * self._cell_w, vy * self._cell_h, self._cell_w, self._cell_h)
        self._screen.fill(self._bg, rect)

    def draw_cursor(
        self,
        vx: int,
        vy: int,
        color: tuple[int, int, int] = (255, 236, 100),
    ) -> None:
        """Outline a single map cell to mark the "look" cursor. Non-destructive
        (an outline box), so whatever occupies the cell stays visible inside it."""
        if self._pygame is None or self._screen is None:
            return
        rect = self._pygame.Rect(vx * self._cell_w, vy * self._cell_h, self._cell_w, self._cell_h)
        thickness = max(2, self._cell_w // 8)
        self._pygame.draw.rect(self._screen, color, rect, width=thickness)

    def _bubble_dims(self, text: str, indicator: str) -> tuple[int, int, int, int, int, int, int, int, int]:
        """Geometry for a speech bubble holding ``text`` (+ optional ``indicator``):
        ``(body_w, body_h, tail, pad_x, pad_y, border, gap, text_w, text_h)`` in
        world pixels. Shared by ``measure_world_label`` (layout) and
        ``draw_world_label`` (drawing) so the two never drift."""
        tw, th = self._font.size(text) if self._font is not None else (len(text) * 6, 10)
        iw = self._font.size(indicator)[0] if (indicator and self._font is not None) else 0
        gap = max(3, self._font.size(" ")[0]) if (indicator and self._font is not None) else 0
        pad_x = max(4, self._cell_w // 5)
        pad_y = max(3, self._cell_h // 6)
        border = max(2, self._cell_w // 14)
        tail = max(4, self._cell_h // 5)
        body_w = tw + gap + iw + pad_x * 2
        body_h = th + pad_y * 2
        return body_w, body_h, tail, pad_x, pad_y, border, gap, tw, th

    def measure_world_label(self, text: str, indicator: str = "") -> tuple[int, int, int]:
        """The ``(body_w, body_h, tail)`` pixel footprint of a bubble, so the caller
        can lay bubbles out without overlap before drawing them."""
        body_w, body_h, tail, *_ = self._bubble_dims(text, indicator)
        return body_w, body_h, tail

    def draw_world_label(
        self,
        vx: int,
        vy: int,
        text: str,
        fg: tuple[int, int, int] | None = None,
        indicator: str = "",
        indicator_color: tuple[int, int, int] | None = None,
        lift: int = 0,
        alpha: int = 255,
        bg: tuple[int, int, int] = (40, 40, 46),
    ) -> None:
        """Draw a speech bubble (translucent dark-gray body, dark outline, a tail
        pointing down-right at the top-right corner of the speaker's tile) centred
        above map cell ``(vx, vy)``, in world/tile pixel space so it aligns with the
        character on that tile (the UI text grid is a different size).

        ``text`` is the white gibberish; ``indicator`` (e.g. ``"++"``) trails it in
        ``indicator_color``. ``lift`` raises the whole bubble by that many pixels so
        the caller can stack bubbles that would otherwise overlap (a newer bubble is
        drawn lower and shoves older ones up). ``alpha`` (0-255) fades the whole
        bubble uniformly, for a smooth fade-out near end of life."""
        pygame = self._pygame
        if pygame is None or self._screen is None or self._font is None or not text or alpha <= 0:
            return

        outline = (18, 18, 22, 225)
        fill = (*_coerce_rgb(bg, (40, 40, 46)), 175)  # translucent dark gray
        text_surf = self._font.render(text, True, _coerce_rgb(fg, (245, 245, 245)))  # white gibberish
        ind_surf = None
        if indicator:
            ind_surf = self._font.render(indicator, True, _coerce_rgb(indicator_color, (245, 245, 245)))

        body_w, body_h, tail, pad_x, pad_y, border, gap, tw, _th = self._bubble_dims(text, indicator)
        radius = max(3, self._cell_w // 8)

        cx = vx * self._cell_w + self._cell_w // 2
        bx = cx - body_w // 2
        by = vy * self._cell_h - body_h - tail - lift  # always above the tile, raised by lift

        # Compose on a per-pixel-alpha surface so the bubble is translucent (and can
        # be faded as a whole), then blit it once. Local coords leave a margin plus
        # room for the tail below the body.
        margin = border + 1
        surf = pygame.Surface((body_w + margin * 2, body_h + tail + margin * 2), pygame.SRCALPHA)
        body = pygame.Rect(margin, margin, body_w, body_h)

        # The tail tip lands on the top-right corner of the speaker's tile; its base
        # sits on the bubble's bottom edge, so it slants down-right toward the tile.
        tile_corner_x = (vx + 1) * self._cell_w
        tail_x = margin + (tile_corner_x - bx)
        tail_x = max(margin + tail + border, min(margin + body_w - tail - border, tail_x))
        tip = (tail_x, margin + body_h + tail)
        base_y = margin + body_h - border
        fill_base_y = margin + body_h - border * 2
        fill_tip = (tip[0], tip[1] - border)

        # 1. dark outline (tail then body), 2. translucent fill (body then tail),
        # so the fills merge across the tail base with no visible seam.
        pygame.draw.polygon(surf, outline, [tip, (tail_x - tail, base_y), (tail_x + tail, base_y)])
        pygame.draw.rect(surf, outline, body, border_radius=radius)
        pygame.draw.rect(
            surf, fill, body.inflate(-border * 2, -border * 2),
            border_radius=max(2, radius - border),
        )
        pygame.draw.polygon(
            surf, fill,
            [fill_tip, (tail_x - tail + border, fill_base_y), (tail_x + tail - border, fill_base_y)],
        )

        # Text/indicator go onto the same surface so the fade multiply below covers
        # them too.
        surf.blit(text_surf, (margin + pad_x, margin + pad_y))
        if ind_surf is not None:
            surf.blit(ind_surf, (margin + pad_x + tw + gap, margin + pad_y))

        if alpha < 255:
            # Scale every pixel's alpha by alpha/255, fading the whole bubble evenly.
            surf.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)

        self._screen.blit(surf, (bx - margin, by - margin))

    def set_map_clip(self, w_cells: int, h_cells: int) -> None:
        """Restrict drawing to the map viewport (top-left w_cells x h_cells) so a
        composite map blit can't spill past the viewport into the sidebar."""
        if self._screen is not None:
            self._screen.set_clip(
                self._pygame.Rect(0, 0, max(0, w_cells) * self._cell_w, max(0, h_cells) * self._cell_h)
            )

    def clear_clip(self) -> None:
        if self._screen is not None:
            self._screen.set_clip(None)

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

        title_surface = None
        if title and self._font is not None:
            title_surface = self._font.render(f"[{title}]", True, (236, 236, 236))

        header_rect = None
        if ph > 6:
            header_padding_y = max(3, self._ui_cell_h // 4)
            if title_surface is not None:
                desired_header_h = title_surface.get_height() + (header_padding_y * 2)
            else:
                desired_header_h = self._ui_cell_h + (header_padding_y * 2)
            header_h = max(4, min(ph - 4, desired_header_h))
            header_rect = self._pygame.Rect(px + 2, py + 2, max(1, pw - 4), header_h)
            self._pygame.draw.rect(self._screen, self._panel_header, header_rect)

        self._pygame.draw.rect(self._screen, self._panel_border, panel_rect, width=2)

        inner_top = py + 4
        if header_rect is not None:
            inner_top = max(inner_top, header_rect.bottom + 2)
        inner_rect = self._pygame.Rect(
            px + 4,
            inner_top,
            max(0, pw - 8),
            max(0, ph - (inner_top - py) - 4),
        )
        if inner_rect.width > 2 and inner_rect.height > 2:
            self._pygame.draw.rect(self._screen, self._panel_border, inner_rect, width=1)

        if title_surface is not None:
            if header_rect is not None:
                text_x = header_rect.x + 8
                text_ink_rect = title_surface.get_bounding_rect()
                if text_ink_rect.width <= 0 or text_ink_rect.height <= 0:
                    text_ink_rect = title_surface.get_rect()
                available_h = max(0, header_rect.height - text_ink_rect.height)
                text_y = header_rect.y + ((available_h + 1) // 2) - text_ink_rect.y
            else:
                text_x = px + 8
                text_y = py + 2
            self._screen.blit(title_surface, (text_x, text_y))

    def present(self) -> None:
        if self._pygame is not None:
            self._pygame.display.flip()

    def save_screenshot(self, output_path: str | Path) -> None:
        if self._pygame is None or self._screen is None:
            raise RuntimeError("renderer not initialized")

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._pygame.image.save(self._screen, str(target))

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

    def _poll_action_once(self) -> str | None:
        if self._pygame is None:
            return None

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
                if event.key in self._confirm_keys:
                    if self._confirm_held_key != event.key:
                        self._confirm_held_key = event.key
                        self._next_confirm_repeat_ms = now + self._confirm_initial_delay_ms
                    self._pending_actions.append("confirm_action")
                    continue

                if event.key in self._keydown_to_action:
                    self._pending_actions.append(self._keydown_to_action[event.key])
                self._ensure_mouse_visible(now)

            if event.type == self._pygame.KEYUP:
                if self._confirm_held_key == event.key:
                    self._confirm_held_key = None
                    continue

                if event.key in self._keyup_to_action:
                    self._pending_actions.append(self._keyup_to_action[event.key])

        if self._pending_actions:
            return self._pending_actions.popleft()

        if self._confirm_held_key is not None and now >= self._next_confirm_repeat_ms:
            self._next_confirm_repeat_ms = now + self._confirm_repeat_interval_ms
            return "confirm_action"

        self._update_cursor_for_splitter()
        self._hide_mouse_if_idle(now)
        return None

    def poll_action(self) -> str | None:
        if self._pygame is None:
            return None

        while True:
            action = self._poll_action_once()
            if action is not None:
                return action

            self._pygame.time.wait(8)

    def poll_action_nonblocking(self) -> str | None:
        return self._poll_action_once()
