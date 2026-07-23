"""The render system: turn the ECS world into glyphs for the ``Renderer``.

``RenderProcessor`` owns field-of-view, remembered-tile (fog-of-war) memory, wall/
water autotiling, the section camera, the sidebar/log/status line, and the real-time
status-identifier animation. Split out of ``systems`` because it is by far the
largest single system and the only one that talks to the display seam; it imports the
map/time/status helpers it needs from ``systems``. Import it via ``systems`` (which
re-exports it), never directly, so the module-load order stays sound.

Perf invariants (see wiki/Performance.md): never transform the whole cached world
surface per frame; repaint only dirty cells and the on-screen region.
"""
from __future__ import annotations

import textwrap
import time
from collections.abc import Callable

import esper

from components import Enemy, Friendly, NPC, Name, Needs, Player, Position, Renderable, Vision
from game_map import GameMap
from renderer.base import Renderer
from content.effects import STATUS_BASE_SECONDS, effect_display
from systems import (
    _ACTION_DELTAS, _ARROW_TO_WORD, _AUTOTILE_DIRECTION_MASKS,
    _AUTOTILE_DIRECTION_OFFSETS, _DIR_TO_ARROW, _DOOR_BROWN, _MIN_SECTION_H,
    _MIN_SECTION_W, _NIGHT_TINT, _NearbyEntry, _RENDER_SECTION_H, _RENDER_SECTION_W,
    _WALL_BROWN, _WATER_BLUE, _WINDOW_CYAN, _memory_color, _pull_turn_events,
    _rects_overlap, active_bubbles, active_statuses, bubble_alpha, format_datetime,
    is_memorable_scenery, night_overlay_alpha, world_clock,
)


class RenderProcessor(esper.Processor):
    """Draws the map, then all Renderable entities on top of it."""

    def __init__(self, renderer: Renderer, game_map: GameMap):
        self.renderer = renderer
        self.game_map = game_map
        self.sidebar_x = game_map.width + 2
        self.sidebar_width = 22
        self._view_origin_x = 0
        self._view_origin_y = 0
        self._view_width = game_map.width
        self._view_height = game_map.height
        # "Look" mode overlay: a world-cell cursor to outline and a status line to
        # show while examining. Both are None during normal play (set/cleared by
        # the turn loop's look handler).
        self.look_cursor: tuple[int, int] | None = None
        self.look_info: str | None = None
        self._status_y = game_map.height
        self._message_log: list[str] = ["You enter the area."]
        self._max_log_lines = 200
        self._last_player_pos: tuple[int, int] | None = None
        self._seen_entity_ids: set[int] = set()
        self._visible_tiles: set[tuple[int, int]] = set()
        # Tile memory ("fog of war"): every tile ever in view is "explored" and
        # keeps being drawn -- desaturated -- once it drops out of line of sight.
        # ``_tile_memory`` remembers the last static scenery (tree, furniture,
        # ...) seen on a tile so it can be redrawn from memory; NPCs and loot are
        # deliberately never stored, so they vanish the moment they leave view.
        self._explored_tiles: set[tuple[int, int]] = set()
        self._tile_memory: dict[
            tuple[int, int],
            tuple[str, str, tuple[int, int, int] | None, tuple[int, int, int] | None],
        ] = {}
        # Last-seen terrain char per explored tile. The living world edits the map
        # out of sight (villagers raise walls, etc.); remembered terrain must show
        # what the player last *saw*, not the current map, so a wall built in an
        # unseen area doesn't appear in memory until it's actually seen.
        self._seen_terrain: dict[tuple[int, int], str] = {}
        # Per-view memory geometry, recomputed whenever the player moves (mirrors
        # the FOV memo below): the desaturated region to blit, the never-seen
        # holes inside it to black out, the explored-but-unseen cells, the
        # remembered scenery to overlay on them, and the terrain overrides (cells
        # whose live terrain has since diverged from what was last seen).
        self._memory_bbox: tuple[int, int, int, int] | None = None
        self._memory_holes: tuple[tuple[int, int], ...] = ()
        self._memory_cells: frozenset[tuple[int, int]] = frozenset()
        self._memory_scenery: tuple[
            tuple[tuple[int, int], tuple[str, str, tuple[int, int, int] | None, tuple[int, int, int] | None]],
            ...,
        ] = ()
        self._memory_terrain_overrides: tuple[tuple[tuple[int, int], str], ...] = ()
        # The rendered remembered-tile layer is cached in the renderer (a viewport
        # blit per frame instead of a grayscale fade). It only needs rebuilding
        # when the memory geometry changes (move / map edit) or the view resizes.
        self._memory_cache_dirty = True
        self._memory_cache_dims: tuple[int, int] | None = None
        # Render caches (big win on web, where every re-render is main-thread work
        # that can stutter audio). Field-of-view only changes when the player
        # moves; wall autotile masks never change on a static map.
        self._visible_cache_key: tuple[int | None, int, int] | None = None
        self._wall_mask_cache: dict[tuple[int, int], int] = {}
        self._water_mask_cache: dict[tuple[int, int], int] = {}
        # Last map revision drawn. When the map mutates at runtime (a wall or door
        # gets built), the cached map surface, autotile masks, and FOV memo all go
        # stale, so we drop them and let the next frame rebuild.
        self._map_revision = getattr(game_map, "revision", 0)
        # Per-player-position FOV memo: pos -> (visible set, view bbox, shadow
        # cells). Lets a walking step blit one cached map region + a few shadow
        # fills instead of re-drawing every visible tile.
        self._visible_bbox: tuple[int, int, int, int] | None = None
        self._shadow_cells: tuple[tuple[int, int], ...] = ()
        self._fov_cache: dict[
            tuple[int | None, int, int],
            tuple[set[tuple[int, int]], tuple[int, int, int, int] | None, tuple[tuple[int, int], ...]],
        ] = {}
        # Only the fixed-size render section (40x20) the player stands in is
        # drawn; the camera snaps to it and the view/FOV are clipped to its
        # bounds. The map is a grid of these sections (a 120x60 world is 3x3 of
        # them; the 360x180 ocean world is 9x9). Sectioning only engages on a map
        # big enough for it to matter; tiny test maps keep the centred camera.
        self._section_w = _RENDER_SECTION_W
        self._section_h = _RENDER_SECTION_H
        self._sections_enabled = (
            game_map.width >= 3 * _MIN_SECTION_W and game_map.height >= 3 * _MIN_SECTION_H
        )
        self._section_bounds: tuple[int, int, int, int] | None = None
        self._current_section: tuple[int, int] | None = None
        # Wall clock driving the swimming glyph bob: it flips every second of
        # real time, independent of turns (so an idle swimmer still shimmers).
        # Injectable so tests can pin the time.
        self._clock: Callable[[], float] = time.monotonic

    @staticmethod
    def _clamp_sign(value: int) -> int:
        if value < 0:
            return -1
        if value > 0:
            return 1
        return 0

    def _direction_arrow(self, source: Position, target: Position) -> str:
        dx = self._clamp_sign(target.x - source.x)
        dy = self._clamp_sign(target.y - source.y)
        return _DIR_TO_ARROW[(dx, dy)]

    def _neighbor_mask(self, x: int, y: int, tile_char: str) -> int:
        connected: dict[int, bool] = {}
        for direction, (dx, dy) in _AUTOTILE_DIRECTION_OFFSETS.items():
            nx = x + dx
            ny = y + dy
            connected[direction] = self.game_map.in_bounds(nx, ny) and self.game_map.tile_at(nx, ny) == tile_char

        mask_value = 0
        if connected.get(2, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[2]  # N
        if connected.get(4, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[4]  # E
        if connected.get(6, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[6]  # S
        if connected.get(8, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[8]  # W

        if connected.get(1, False) and connected.get(2, False) and connected.get(8, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[1]  # NW
        if connected.get(3, False) and connected.get(2, False) and connected.get(4, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[3]  # NE
        if connected.get(7, False) and connected.get(6, False) and connected.get(8, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[7]  # SW
        if connected.get(5, False) and connected.get(6, False) and connected.get(4, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[5]  # SE

        return mask_value

    def _append_message(self, text: str) -> None:
        self._message_log.append(text)
        if len(self._message_log) > self._max_log_lines:
            self._message_log = self._message_log[-self._max_log_lines :]

    def _describe_movement(
        self,
        action: str | None,
        player_pos: Position | None,
        suppress_wall_message: bool,
    ) -> None:
        if action not in _ACTION_DELTAS or player_pos is None:
            return

        if self._last_player_pos is None:
            self._last_player_pos = (player_pos.x, player_pos.y)
            return

        moved = self._last_player_pos != (player_pos.x, player_pos.y)
        if not moved and not suppress_wall_message:
            dx, dy = _ACTION_DELTAS[action]
            target_x = player_pos.x + dx
            target_y = player_pos.y + dy
            if not self.game_map.is_walkable(target_x, target_y):
                self._append_message("You bump into a wall.")

        self._last_player_pos = (player_pos.x, player_pos.y)

    def _collect_nearby_objects(
        self,
        player_pos: Position,
    ) -> list[_NearbyEntry]:
        nearby: list[_NearbyEntry] = []

        for ent, (pos, rend) in esper.get_components(Position, Renderable):
            if pos.x == player_pos.x and pos.y == player_pos.y:
                continue
            if (pos.x, pos.y) not in self._visible_tiles:
                continue

            dx = pos.x - player_pos.x
            dy = pos.y - player_pos.y
            if max(abs(dx), abs(dy)) != 1:
                continue

            arrow = self._direction_arrow(player_pos, pos)
            name = "Unknown"
            if esper.has_component(ent, Name):
                name = esper.component_for_entity(ent, Name).value

            classification = "default"
            if esper.has_component(ent, Friendly):
                classification = "friendly"
            elif esper.has_component(ent, Enemy):
                classification = "enemy"

            nearby.append(
                (
                    ent,
                    rend.glyph,
                    arrow,
                    name,
                    abs(dx) + abs(dy),
                    max(abs(dx), abs(dy)),
                    classification,
                    rend.fg,
                    rend.bg,
                )
            )

        nearby.sort(key=lambda item: (item[4], item[5], item[3]))
        return nearby

    def _compute_visible_tiles(self, player_ent: int | None, player_pos: Position | None) -> set[tuple[int, int]]:
        if player_pos is None:
            return set()

        radius = max(self.game_map.width // 2, self.game_map.height // 2)
        if player_ent is not None and esper.has_component(player_ent, Vision):
            radius = esper.component_for_entity(player_ent, Vision).radius

        # Clip the scan (and thus visibility) to the current section: other
        # sections are simulated but never drawn.
        if self._section_bounds is not None:
            sx, sy, sw, sh = self._section_bounds
            x_lo, x_hi = sx, sx + sw
            y_lo, y_hi = sy, sy + sh
        else:
            x_lo, x_hi = 0, self.game_map.width
            y_lo, y_hi = 0, self.game_map.height

        visible: set[tuple[int, int]] = set()
        origin = (player_pos.x, player_pos.y)
        for y in range(y_lo, y_hi):
            for x in range(x_lo, x_hi):
                if max(abs(x - player_pos.x), abs(y - player_pos.y)) > radius:
                    continue
                if self.game_map.has_line_of_sight(origin, (x, y)):
                    visible.add((x, y))
        return visible

    @staticmethod
    def _fov_bbox_and_shadows(
        visible: set[tuple[int, int]],
    ) -> tuple[tuple[int, int, int, int] | None, tuple[tuple[int, int], ...]]:
        """Bounding box of the visible set plus the cells inside that box that
        are *not* visible (wall shadows). Used to composite the map as one blit
        of the box followed by a few shadow blackouts."""
        if not visible:
            return None, ()
        xs = [c[0] for c in visible]
        ys = [c[1] for c in visible]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        shadows = tuple(
            (x, y)
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if (x, y) not in visible
        )
        return bbox, shadows

    def _compute_memory_geometry(self) -> None:
        """Recompute the remembered-tile geometry for the current view. Cheap set
        work over the viewport (at most one render section), refreshed only when
        the player moves -- the same cadence as the FOV memo -- so a walking step
        stays a handful of region blits.

        Uses ``_explored_tiles`` as of the *previous* frame (currently-visible
        tiles are folded into it after drawing), which is exactly right: a tile
        only needs remembering once it is no longer lit.
        """
        ox, oy = self._view_origin_x, self._view_origin_y
        vw, vh = self._view_width, self._view_height
        explored = self._explored_tiles
        visible = self._visible_tiles

        explored_in_view: list[tuple[int, int]] = []
        memory_cells: set[tuple[int, int]] = set()
        for vy in range(vh):
            wy = oy + vy
            for vx in range(vw):
                wx = ox + vx
                cell = (wx, wy)
                if cell in explored:
                    explored_in_view.append(cell)
                    if cell not in visible:
                        memory_cells.add(cell)

        if not explored_in_view:
            self._memory_bbox = None
            self._memory_holes = ()
            self._memory_cells = frozenset()
            self._memory_scenery = ()
            self._memory_terrain_overrides = ()
            self._memory_cache_dirty = True
            return

        xs = [c[0] for c in explored_in_view]
        ys = [c[1] for c in explored_in_view]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        self._memory_bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        explored_set = set(explored_in_view)
        self._memory_holes = tuple(
            (x, y)
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if (x, y) not in explored_set
        )
        self._memory_cells = frozenset(memory_cells)
        # Precompute the (few) remembered-scenery overlays so the render path
        # doesn't scan the whole explored region for the handful with scenery.
        self._memory_scenery = tuple(
            (cell, self._tile_memory[cell])
            for cell in memory_cells
            if cell in self._tile_memory
        )
        # Terrain overrides: unseen cells whose live terrain has diverged from
        # what was last seen (e.g. a villager raised a wall out of sight). These
        # get overdrawn with the remembered char so memory shows the old terrain.
        overrides: list[tuple[tuple[int, int], str]] = []
        for cell in memory_cells:
            seen = self._seen_terrain.get(cell)
            if seen is not None and seen != self.game_map.tile_at(cell[0], cell[1]):
                overrides.append((cell, seen))
        self._memory_terrain_overrides = tuple(overrides)
        # The remembered layer's contents just changed, so its render cache is stale.
        self._memory_cache_dirty = True

    def _update_tile_memory(self, seen_scenery: dict[tuple[int, int], tuple]) -> None:
        """Fold this frame's field of view into the persistent tile memory: mark
        every visible tile explored, record its current terrain char (so memory
        tracks what was last seen, not later out-of-sight edits), then record the
        static scenery seen there (from ``seen_scenery``, gathered during the
        entity draw pass) or clear stale memory if it is gone (e.g. a felled tree).
        """
        self._explored_tiles |= self._visible_tiles
        for cell in self._visible_tiles:
            self._seen_terrain[cell] = self.game_map.tile_at(cell[0], cell[1])
            appearance = seen_scenery.get(cell)
            if appearance is not None:
                self._tile_memory[cell] = appearance
            else:
                self._tile_memory.pop(cell, None)

    def _draw_map_tile(
        self,
        r,
        vx: int,
        vy: int,
        wx: int,
        wy: int,
        draw_autotile_variant,
        dim: bool = False,
        tile_char: str | None = None,
    ) -> None:
        """Draw the base terrain tile at world cell (wx, wy) to view cell (vx, vy).
        With ``dim`` the tile is toned into its remembered (desaturated) colour --
        used by renderers that can't composite a whole desaturated map surface
        (the test double); the pygame path desaturates the cached surface instead.
        ``tile_char`` overrides the terrain char (last-seen terrain for remembered
        cells the live map has since changed); autotile masks still read the live
        neighbours, which is fine for the rare divergent cell.
        """
        tone = _memory_color if dim else (lambda colour: colour)
        tile = tile_char if tile_char is not None else self.game_map.tile_at(wx, wy)
        if tile == self.game_map.WALL and callable(draw_autotile_variant):
            wall_mask = self._wall_mask_cache.get((wx, wy))
            if wall_mask is None:
                wall_mask = self._neighbor_mask(wx, wy, self.game_map.WALL)
                self._wall_mask_cache[(wx, wy)] = wall_mask
            if bool(draw_autotile_variant(vx, vy, "wall", wall_mask, fg=tone(_WALL_BROWN), bg=None)):
                return
        if tile == self.game_map.WATER and callable(draw_autotile_variant):
            water_mask = self._water_mask_cache.get((wx, wy))
            if water_mask is None:
                water_mask = self._neighbor_mask(wx, wy, self.game_map.WATER)
                self._water_mask_cache[(wx, wy)] = water_mask
            if bool(draw_autotile_variant(vx, vy, "water", water_mask, fg=tone(_WATER_BLUE), bg=None)):
                return
        classification = "wall" if tile == self.game_map.WALL else "default"
        if tile == self.game_map.WALL:
            tile_fg: tuple[int, int, int] | None = _WALL_BROWN
        elif tile == self.game_map.WATER:
            tile_fg = _WATER_BLUE
        elif tile == self.game_map.DOOR:
            tile_fg = _DOOR_BROWN
        elif tile == self.game_map.WINDOW:
            tile_fg = _WINDOW_CYAN
        else:
            tile_fg = None
        # ``_memory_color`` maps a colourless floor (None) to a neutral dim grey,
        # so remembered floor stays faintly visible rather than turning black.
        draw_fg = _memory_color(tile_fg) if dim else tile_fg
        r.draw_glyph_classified(vx, vy, tile, classification, fg=draw_fg, bg=None)

    def _draw_memory_glyph(
        self,
        r,
        vx: int,
        vy: int,
        glyph: str,
        classification: str,
        fg: tuple[int, int, int] | None,
        bg: tuple[int, int, int] | None,
    ) -> None:
        """Draw a remembered scenery glyph desaturated. Prefers the renderer's
        sprite-aware ``draw_glyph_memory`` (desaturates the whole tile); falls
        back to a memory-toned foreground for the plain test double."""
        draw_memory = getattr(r, "draw_glyph_memory", None)
        if callable(draw_memory):
            draw_memory(vx, vy, glyph, classification, fg=fg, bg=bg)
            return
        r.draw_glyph_classified(vx, vy, glyph, classification, fg=_memory_color(fg), bg=bg)

    def _draw_memory_tile(self, r, vx: int, vy: int, wx: int, wy: int, draw_autotile_variant) -> None:
        """Draw one remembered tile (used by the per-tile fallback path): the
        desaturated last-seen terrain, then any static scenery last seen there."""
        self._draw_map_tile(
            r, vx, vy, wx, wy, draw_autotile_variant, dim=True,
            tile_char=self._seen_terrain.get((wx, wy)),
        )
        appearance = self._tile_memory.get((wx, wy))
        if appearance is not None:
            glyph, classification, fg, bg = appearance
            self._draw_memory_glyph(r, vx, vy, glyph, classification, fg, bg)

    # SDL's common maximum texture dimension. A cached world surface wider or taller
    # than this can't live on the GPU and, as a software surface, balloons into
    # gigabytes (a 1400x800 world at 24px/cell is 33600x19200 ~ 2.6 GB), so past this
    # we skip the whole-map cache and draw the viewport per tile instead.
    _MAX_WORLD_SURFACE_PX = 16384

    def _world_surface_too_large(self, r) -> bool:
        """True when a whole-map cached surface would exceed the safe texture size,
        so ``_render_map_layer`` should fall back to per-viewport tile drawing."""
        get_cell = getattr(r, "get_tile_cell_size_px", None)
        cell_w = cell_h = 24
        if callable(get_cell):
            try:
                cell_w, cell_h = get_cell()
            except Exception:
                pass
        return (
            self.game_map.width * cell_w > self._MAX_WORLD_SURFACE_PX
            or self.game_map.height * cell_h > self._MAX_WORLD_SURFACE_PX
        )

    def _render_map_layer(self, r, draw_autotile_variant) -> None:
        """Draw the map for the current FOV. When the renderer supports a cached
        map surface, a step becomes one region blit + a few shadow fills instead
        of re-drawing every visible tile (the dominant walking cost on web)."""
        has_map_surface = getattr(r, "has_map_surface", None)
        build_map_surface = getattr(r, "build_map_surface", None)
        blit_map_region = getattr(r, "blit_map_region", None)
        fill_cell_bg = getattr(r, "fill_cell_bg", None)
        can_composite = all(
            callable(fn) for fn in (has_map_surface, build_map_surface, blit_map_region, fill_cell_bg)
        ) and not self._world_surface_too_large(r)

        ox, oy = self._view_origin_x, self._view_origin_y

        if not can_composite:
            # Fallback: per-tile draw of just the viewport. Used by the test double
            # renderer, and by worlds too large to cache as one surface (the
            # archipelago) -- the whole-map surface is only a walking-speed cache, so
            # dropping it just costs a few extra per-tile draws inside the FOV each
            # frame, never a giant off-screen allocation. Lit tiles show in full
            # colour; explored-but-unseen tiles are drawn from memory (desaturated
            # terrain + last-seen scenery).
            for vy in range(self._view_height):
                wy = oy + vy
                for vx in range(self._view_width):
                    wx = ox + vx
                    if (wx, wy) in self._visible_tiles:
                        self._draw_map_tile(r, vx, vy, wx, wy, draw_autotile_variant)
                    elif (wx, wy) in self._explored_tiles:
                        self._draw_memory_tile(r, vx, vy, wx, wy, draw_autotile_variant)
            return

        if not has_map_surface():
            # One-time: render every tile fully lit to a world-sized off-screen
            # surface. World coords stay valid when the camera scrolls at zoom.
            def _draw_full_map() -> None:
                for wy in range(self.game_map.height):
                    for wx in range(self.game_map.width):
                        self._draw_map_tile(r, wx, wy, wx, wy, draw_autotile_variant)

            build_map_surface(self.game_map.width, self.game_map.height, _draw_full_map)

        if self._visible_bbox is None and self._memory_bbox is None:
            return

        # The remembered-tile layer (faded terrain + holes + overrides + scenery)
        # is expensive to draw (a grayscale fade) but only changes on move / map
        # edit, so it is rendered once into a viewport-sized cache and blitted each
        # frame. Feature-detected: renderers lacking the cache just fall back to
        # black beyond the FOV.
        apply_memory_fade = getattr(r, "apply_memory_fade", None)
        capture_memory_layer = getattr(r, "capture_memory_layer", None)
        has_memory_cache = getattr(r, "has_memory_cache", None)
        blit_memory_cache = getattr(r, "blit_memory_cache", None)
        blit_memory_cache_cell = getattr(r, "blit_memory_cache_cell", None)
        can_remember = all(
            callable(fn)
            for fn in (
                apply_memory_fade,
                capture_memory_layer,
                has_memory_cache,
                blit_memory_cache,
                blit_memory_cache_cell,
            )
        )

        # Rebuild the cache only when the remembered layer changed, the view
        # resized, or the renderer dropped the cache (e.g. after a zoom rebuild).
        vw, vh = self._view_width, self._view_height
        use_memory_cache = can_remember and self._memory_bbox is not None
        if use_memory_cache:
            if (
                self._memory_cache_dirty
                or self._memory_cache_dims != (vw, vh)
                or not has_memory_cache()
            ):
                self._memory_cache_dirty = False
                self._memory_cache_dims = (vw, vh)
                capture_memory_layer(
                    vw,
                    vh,
                    lambda: self._draw_memory_terrain_layer(
                        r, blit_map_region, apply_memory_fade, fill_cell_bg, draw_autotile_variant
                    ),
                )

        # Screen was cleared at the top of process(); composite the map (offset by
        # the scroll origin). Clip to the viewport so a big region blit can't spill
        # into the sidebar/status area when a box is larger than the visible map.
        set_map_clip = getattr(r, "set_map_clip", None)
        clear_clip = getattr(r, "clear_clip", None)
        clipped = callable(set_map_clip) and callable(clear_clip)
        if clipped:
            set_map_clip(self._view_width, self._view_height)
        try:
            # 1. The cached remembered layer (all explored-but-unseen cells).
            if use_memory_cache:
                blit_memory_cache()
            # 2. The lit field of view on top.
            if self._visible_bbox is not None:
                bx, by, bw, bh = self._visible_bbox
                blit_map_region(bx, by, bw, bh, ox, oy)
                # Shadow cells sit inside the FOV box but are blocked from view;
                # the rectangular blit just painted them lit, so restore the
                # explored ones from the cached memory layer and black out the rest.
                for wx, wy in self._shadow_cells:
                    if use_memory_cache and (wx, wy) in self._explored_tiles:
                        blit_memory_cache_cell(wx - ox, wy - oy)
                    else:
                        fill_cell_bg(wx - ox, wy - oy)
        finally:
            if clipped:
                clear_clip()

    def _draw_memory_terrain_layer(
        self, r, blit_map_region, apply_memory_fade, fill_cell_bg, draw_autotile_variant
    ) -> None:
        """Draw the whole remembered-tile layer at view coords: faded last-seen
        terrain over the explored region, never-seen holes blacked out, terrain
        overrides (cells the live map changed out of sight) and static scenery.
        Rendered into the memory cache (see ``capture_memory_layer``); every cell
        is inside the viewport by construction."""
        ox, oy = self._view_origin_x, self._view_origin_y

        # Faded last-seen terrain over the explored region, then black holes.
        if self._memory_bbox is not None:
            mbx, mby, mbw, mbh = self._memory_bbox
            blit_map_region(mbx, mby, mbw, mbh, ox, oy)
            apply_memory_fade(mbx - ox, mby - oy, mbw, mbh)
            for wx, wy in self._memory_holes:
                fill_cell_bg(wx - ox, wy - oy)

        # Terrain overrides: the region blit above showed *live* terrain, so unseen
        # cells the map has since changed (a wall raised out of sight) are redrawn
        # from the last-seen char and faded, hiding the change until it is seen.
        for (wx, wy), seen_char in self._memory_terrain_overrides:
            if (wx, wy) in self._visible_tiles:
                continue
            view_xy = self._world_to_view(wx, wy)
            if view_xy is None:
                continue
            self._draw_map_tile(
                r, view_xy[0], view_xy[1], wx, wy, draw_autotile_variant, tile_char=seen_char
            )
            apply_memory_fade(view_xy[0], view_xy[1], 1, 1)

        # Remembered scenery (trees, furniture, ...) on top of the faded terrain.
        for (wx, wy), appearance in self._memory_scenery:
            view_xy = self._world_to_view(wx, wy)
            if view_xy is None:
                continue
            glyph, classification, fg, bg = appearance
            self._draw_memory_glyph(r, view_xy[0], view_xy[1], glyph, classification, fg, bg)

    def _update_sighting_events(self, nearby: list[tuple[int, str, str, str, int, int, int, int]]) -> None:
        newly_seen = [item for item in nearby if item[0] not in self._seen_entity_ids]

        for ent_id, _glyph, arrow, name, _md, _cd, _x, _y in newly_seen[:2]:
            direction = _ARROW_TO_WORD.get(arrow, "nearby")
            self._append_message(f"You notice {name} to the {direction}.")
            self._seen_entity_ids.add(ent_id)

    def _collect_visible_npcs(
        self,
        player_pos: Position,
    ) -> list[tuple[int, str, str, str, int, int, int, int]]:
        visible: list[tuple[int, str, str, str, int, int, int, int]] = []

        for ent, (pos, rend, _npc) in esper.get_components(Position, Renderable, NPC):
            if (pos.x, pos.y) == (player_pos.x, player_pos.y):
                continue
            if (pos.x, pos.y) not in self._visible_tiles:
                continue

            dx = pos.x - player_pos.x
            dy = pos.y - player_pos.y
            arrow = self._direction_arrow(player_pos, pos)
            name = "Unknown"
            if esper.has_component(ent, Name):
                name = esper.component_for_entity(ent, Name).value

            visible.append((ent, rend.glyph, arrow, name, abs(dx) + abs(dy), max(abs(dx), abs(dy)), pos.x, pos.y))

        visible.sort(key=lambda item: (item[4], item[5], item[3]))
        return visible

    def _draw_sidebar(self, player_pos: Position | None) -> None:
        r = self.renderer
        draw_text_clipped = getattr(r, "draw_text_clipped", None)
        draw_text_tinted = getattr(r, "draw_text_tinted", None)
        fill_cells = getattr(r, "fill_cells", None)
        draw_ui_glyph = getattr(r, "draw_ui_glyph", None)

        def draw_line(
            x: int,
            y: int,
            text: str,
            width_cells: int,
            color: tuple[int, int, int] | None = None,
        ) -> None:
            clipped_text = text[: max(0, width_cells)]
            if color is not None and callable(draw_text_tinted):
                draw_text_tinted(x, y, clipped_text, color)
                return
            if callable(draw_text_clipped):
                draw_text_clipped(x, y, text, width_cells)
                return
            r.draw_text(x, y, clipped_text)

        panel_x = max(0, self.sidebar_x)
        panel_y = 0
        panel_w = max(14, self.sidebar_width)
        panel_h = max(14, self._grid_h)

        x0 = panel_x + 2
        content_width = max(6, panel_w - 4)
        wrap_width = content_width
        get_text_cols = getattr(r, "text_columns_for_cells", None)
        if callable(get_text_cols):
            cols = get_text_cols(content_width)
            if isinstance(cols, int):
                wrap_width = max(1, cols)

        nearby_data: list[_NearbyEntry] = []
        if player_pos is not None:
            visible_npcs = self._collect_visible_npcs(player_pos)
            self._update_sighting_events(visible_npcs)
            nearby_data = self._collect_nearby_objects(player_pos)

        nearby_lines: list[str] = []

        if player_pos is None:
            nearby_lines.append("none")
        else:
            if not nearby_data:
                nearby_lines.append("none")
            else:
                for _ent_id, glyph, arrow, name, _mdist, _cdist, _classification, _fg, _bg in nearby_data:
                    line = f"{glyph} {arrow} {name}"
                    nearby_lines.append(line)

        wrapped_nearby: list[str] = []
        for line in nearby_lines:
            wrapped = textwrap.wrap(line, width=wrap_width, break_long_words=True, break_on_hyphens=False)
            if wrapped:
                wrapped_nearby.extend(wrapped)
            else:
                wrapped_nearby.append("")

        panel_gap = 1
        min_box_h = 6
        nearby_content_needed = max(1, len(wrapped_nearby))
        nearby_h_needed = nearby_content_needed + 4
        max_nearby_h = max(min_box_h, panel_h - min_box_h - panel_gap)
        nearby_h = max(min_box_h, min(max_nearby_h, nearby_h_needed))
        log_h = panel_h - nearby_h - panel_gap
        if log_h < min_box_h:
            log_h = min_box_h
            nearby_h = max(min_box_h, panel_h - log_h - panel_gap)

        nearby_y = panel_y
        log_y = nearby_y + nearby_h + panel_gap

        draw_panel = getattr(r, "draw_panel", None)
        has_custom_panels = callable(draw_panel)
        if has_custom_panels:
            draw_panel(panel_x, nearby_y, panel_w, nearby_h, title="NEARBY")
            draw_panel(panel_x, log_y, panel_w, log_h, title="LOG")
        else:
            def draw_ascii_panel(px: int, py: int, pw: int, ph: int, title: str) -> None:
                top = "+" + ("-" * max(0, pw - 2)) + "+"
                r.draw_text(px, py, top)
                for row in range(1, max(1, ph - 1)):
                    r.draw_text(px, py + row, "|" + (" " * max(0, pw - 2)) + "|")
                if ph > 1:
                    r.draw_text(px, py + ph - 1, top)
                r.draw_text(px + 2, py, title)

            draw_ascii_panel(panel_x, nearby_y, panel_w, nearby_h, "NEARBY")
            draw_ascii_panel(panel_x, log_y, panel_w, log_h, "LOG")

        if callable(fill_cells) and not has_custom_panels:
            header_bg = (18, 18, 18)
            fill_cells(panel_x + 1, nearby_y + 1, max(1, panel_w - 2), 1, header_bg)
            fill_cells(panel_x + 1, log_y + 1, max(1, panel_w - 2), 1, header_bg)

        header_color = (236, 236, 236)
        text_color = (214, 214, 214)
        muted_color = (168, 168, 168)

        if not has_custom_panels:
            draw_line(x0, nearby_y + 1, "[NEARBY]", content_width, color=header_color)
            nearby_content_y = nearby_y + 3
            nearby_content_h = max(1, nearby_h - 4)
        else:
            nearby_content_y = nearby_y + 2
            nearby_content_h = max(1, nearby_h - 3)

        if callable(draw_ui_glyph) and nearby_data:
            icon_cells = 2
            text_x = x0 + icon_cells + 1
            text_width = max(1, content_width - icon_cells - 1)
            for idx, entry in enumerate(nearby_data[:nearby_content_h]):
                _ent_id, glyph, arrow, name, _mdist, _cdist, classification, fg, bg = entry
                row_y = nearby_content_y + idx
                icon_drawn = bool(draw_ui_glyph(
                    x0,
                    row_y,
                    glyph,
                    classification=classification,
                    fg=fg,
                    bg=bg,
                    cell_span=icon_cells,
                ))
                if icon_drawn:
                    line_text = f"{arrow} {name}"
                else:
                    line_text = f"{glyph} {arrow} {name}"
                draw_line(text_x, row_y, line_text, text_width, color=text_color)
        else:
            for idx, line in enumerate(wrapped_nearby[:nearby_content_h]):
                draw_line(x0, nearby_content_y + idx, line, content_width, color=text_color)

        if not has_custom_panels:
            draw_line(x0, log_y + 1, "[LOG]", content_width, color=header_color)
            log_content_y = log_y + 3
            log_content_h = max(1, log_h - 4)
        else:
            log_content_y = log_y + 2
            log_content_h = max(1, log_h - 3)

        log_messages: list[str] = []
        for message in self._message_log:
            wrapped = textwrap.wrap(message, width=wrap_width, break_long_words=True, break_on_hyphens=False)
            if wrapped:
                log_messages.extend(wrapped)
            else:
                log_messages.append("")

        if not log_messages:
            log_messages = ["none"]

        visible_log = log_messages[-log_content_h:]
        for idx, line in enumerate(visible_log):
            color = muted_color if line == "none" else text_color
            draw_line(x0, log_content_y + idx, line, content_width, color=color)

    def _compute_layout(self, player_pos: Position | None) -> None:
        get_screen_px = getattr(self.renderer, "get_screen_size_px", None)
        get_tile_cell_px = getattr(self.renderer, "get_tile_cell_size_px", None)
        get_ui_grid = getattr(self.renderer, "get_ui_grid_size", None)
        get_ui_cell_px = getattr(self.renderer, "get_ui_cell_size_px", None)
        get_sidebar_px = getattr(self.renderer, "get_sidebar_width_px", None)

        if (
            callable(get_screen_px)
            and callable(get_tile_cell_px)
            and callable(get_ui_grid)
            and callable(get_ui_cell_px)
            and callable(get_sidebar_px)
        ):
            screen_w, screen_h = get_screen_px()
            tile_w, tile_h = get_tile_cell_px()
            ui_cols, ui_rows = get_ui_grid()
            ui_cell_w, ui_cell_h = get_ui_cell_px()

            tile_w = max(1, tile_w)
            tile_h = max(1, tile_h)
            ui_cell_w = max(1, ui_cell_w)
            ui_cell_h = max(1, ui_cell_h)

            status_px = ui_cell_h
            sidebar_px = max(14 * ui_cell_w, int(get_sidebar_px(22 * ui_cell_w)))
            sidebar_px = min(max(sidebar_px, 14 * ui_cell_w), max(14 * ui_cell_w, screen_w - 8 * tile_w))

            map_px_w = max(8 * tile_w, screen_w - sidebar_px)
            map_px_h = max(5 * tile_h, screen_h - status_px)

            self._view_width = min(self.game_map.width, max(8, map_px_w // tile_w))
            self._view_height = min(self.game_map.height, max(5, map_px_h // tile_h))

            self._grid_w = max(1, ui_cols)
            self._grid_h = max(1, ui_rows)
            self.sidebar_width = max(14, sidebar_px // ui_cell_w)
            self.sidebar_x = max(0, self._grid_w - self.sidebar_width)
            self._status_y = max(0, self._grid_h - 1)

            max_ox = max(0, self.game_map.width - self._view_width)
            max_oy = max(0, self.game_map.height - self._view_height)
            if player_pos is None:
                self._view_origin_x = 0
                self._view_origin_y = 0
                return

            self._view_origin_x = min(max_ox, max(0, player_pos.x - self._view_width // 2))
            self._view_origin_y = min(max_oy, max(0, player_pos.y - self._view_height // 2))
            return

        grid_w = self.game_map.width + self.sidebar_width + 3
        grid_h = self.game_map.height + 1
        get_grid = getattr(self.renderer, "get_grid_size", None)
        if callable(get_grid):
            w, h = get_grid()
            if isinstance(w, int) and isinstance(h, int):
                grid_w = max(20, w)
                grid_h = max(8, h)

        get_sidebar_w = getattr(self.renderer, "get_sidebar_width_cells", None)
        if callable(get_sidebar_w):
            w = get_sidebar_w(self.sidebar_width)
            if isinstance(w, int):
                self.sidebar_width = max(12, min(w, max(12, grid_w - 8)))

        self._grid_w = grid_w
        self._grid_h = grid_h

        self.sidebar_x = max(0, grid_w - self.sidebar_width)
        self._view_width = min(self.game_map.width, max(8, self.sidebar_x))
        self._view_height = min(self.game_map.height, max(5, grid_h - 1))
        self._status_y = max(0, grid_h - 1)

        max_ox = max(0, self.game_map.width - self._view_width)
        max_oy = max(0, self.game_map.height - self._view_height)
        if player_pos is None:
            self._view_origin_x = 0
            self._view_origin_y = 0
            return

        self._view_origin_x = min(max_ox, max(0, player_pos.x - self._view_width // 2))
        self._view_origin_y = min(max_oy, max(0, player_pos.y - self._view_height // 2))

    def _world_to_view(self, wx: int, wy: int) -> tuple[int, int] | None:
        vx = wx - self._view_origin_x
        vy = wy - self._view_origin_y
        if vx < 0 or vy < 0 or vx >= self._view_width or vy >= self._view_height:
            return None
        return (vx, vy)

    def view_bounds(self) -> tuple[int, int, int, int]:
        """(origin_x, origin_y, width, height) of the currently drawn viewport in
        world cells. Used by the look cursor to stay on-screen."""
        return (self._view_origin_x, self._view_origin_y, self._view_width, self._view_height)

    def tile_is_visible(self, x: int, y: int) -> bool:
        """Whether world cell (x, y) is in the player's current field of view."""
        return (x, y) in self._visible_tiles

    def _apply_section_camera(self, player_pos: Position | None) -> None:
        """Lock the camera to the render section the player occupies. Sets
        ``_section_bounds`` (used to clip FOV) and clamps the view origin inside
        that section, so only the current section is ever drawn. Emits a log line
        when the player crosses into a new section."""
        if player_pos is None or not self._sections_enabled:
            self._section_bounds = None
            return

        sw, sh = self._section_w, self._section_h
        cols = max(1, self.game_map.width // sw)
        rows = max(1, self.game_map.height // sh)
        col = min(cols - 1, max(0, player_pos.x // sw))
        row = min(rows - 1, max(0, player_pos.y // sh))
        sec_ox, sec_oy = col * sw, row * sh
        # The last column/row absorbs any remainder when the map isn't evenly
        # divisible by the section size, so the whole map stays covered.
        sec_w = self.game_map.width - sec_ox if col == cols - 1 else sw
        sec_h = self.game_map.height - sec_oy if row == rows - 1 else sh
        self._section_bounds = (sec_ox, sec_oy, sec_w, sec_h)

        if self._current_section is not None and self._current_section != (col, row):
            self._append_message("You cross into a new area.")
        self._current_section = (col, row)

        # Never render wider/taller than the section; clamp the origin inside it.
        self._view_width = min(self._view_width, sec_w)
        self._view_height = min(self._view_height, sec_h)
        max_ox = sec_ox + sec_w - self._view_width
        max_oy = sec_oy + sec_h - self._view_height
        self._view_origin_x = min(max_ox, max(sec_ox, player_pos.x - self._view_width // 2))
        self._view_origin_y = min(max_oy, max(sec_oy, player_pos.y - self._view_height // 2))

    def _status_appearance(
        self,
        ent: int,
        pos: Position,
        glyph: str,
        fg: tuple[int, int, int] | None,
    ) -> tuple[str, tuple[int, int, int] | None, bool]:
        """The (glyph, colour, is_status_identifier) to draw for a character this
        instant. With no active status it's just its own tile; with statuses it
        cycles through the base tile then each status identifier in turn, on a
        wall clock, so the animation plays even while idle. The bool marks the
        status-identifier frames, which the renderer must draw as literal glyphs
        (not the character's classification sprite)."""
        statuses = active_statuses(self.game_map, ent, pos)
        if not statuses:
            return glyph, fg, False

        # (glyph, fg, seconds, is_status_identifier)
        frames: list[tuple[str, tuple[int, int, int] | None, float, bool]] = [
            (glyph, fg, STATUS_BASE_SECONDS, False)
        ]
        for name in statuses:
            id_glyph, id_fg, seconds = effect_display(name)
            frames.append((id_glyph, id_fg if id_fg is not None else fg, seconds, True))

        total = sum(frame[2] for frame in frames)
        if total <= 0:
            return glyph, fg, False

        elapsed = self._clock() % total
        cursor = 0.0
        for frame_glyph, frame_fg, seconds, is_status in frames:
            cursor += seconds
            if elapsed < cursor:
                return frame_glyph, frame_fg, is_status
        last = frames[-1]
        return last[0], last[1], last[3]

    def _invalidate_map_caches_if_changed(self) -> None:
        """Refresh map-derived caches when the map has been edited since last
        frame, so a newly built wall/door/window shows up immediately.

        Field-of-view can change anywhere, so its memo is always dropped. The
        cached map *surface*, though, is repainted incrementally -- only the
        edited cells (and their neighbours, whose autotile masks depend on them)
        -- because re-rendering the whole world surface on every single tile edit
        is hundreds of milliseconds on a large map (an NPC building a house edits
        many tiles in quick succession)."""
        revision = getattr(self.game_map, "revision", 0)
        if revision == self._map_revision:
            return
        self._map_revision = revision
        self._fov_cache.clear()
        self._visible_cache_key = None

        consume = getattr(self.game_map, "consume_dirty_tiles", None)
        redraw = getattr(self.renderer, "redraw_map_cells", None)
        has_surface = getattr(self.renderer, "has_map_surface", None)
        dirty = consume() if callable(consume) else None

        if dirty and callable(redraw) and callable(has_surface) and has_surface():
            # An edited cell and its 8 neighbours are the only cells whose drawn
            # appearance can change (autotile masks read neighbours).
            affected: set[tuple[int, int]] = set()
            for x, y in dirty:
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if self.game_map.in_bounds(nx, ny):
                            affected.add((nx, ny))
            for cell in affected:
                self._wall_mask_cache.pop(cell, None)
                self._water_mask_cache.pop(cell, None)
            draw_autotile_variant = getattr(self.renderer, "draw_autotile_variant", None)
            redraw(
                affected,
                lambda wx, wy: self._draw_map_tile(
                    self.renderer, wx, wy, wx, wy, draw_autotile_variant
                ),
            )
            return

        # No incremental path (test double, or surface not built yet): fall back
        # to a full invalidate and let the next frame rebuild.
        self._wall_mask_cache.clear()
        self._water_mask_cache.clear()
        invalidate_surface = getattr(self.renderer, "invalidate_map_surface", None)
        if callable(invalidate_surface):
            invalidate_surface()

    def process(self, action: str | None = None) -> None:
        r = self.renderer
        r.clear()

        self._invalidate_map_caches_if_changed()

        # Just locate the player; a whole-world position/name index was built
        # here every frame but nothing read it -- a needless full-entity scan
        # that hurt once the ocean added a couple thousand fish/seaweed.
        player_pos: Position | None = None
        player_ent: int | None = None
        for ent, (pos, _player) in esper.get_components(Position, Player):
            player_pos = pos
            player_ent = ent
            break

        if player_pos is not None and self._last_player_pos is None:
            self._last_player_pos = (player_pos.x, player_pos.y)

        events = _pull_turn_events()
        for event in events:
            self._append_message(event)

        self._describe_movement(action, player_pos, suppress_wall_message=bool(events))
        self._compute_layout(player_pos)
        self._apply_section_camera(player_pos)

        # Only recompute field-of-view when the player actually moved; the LOS
        # raycast over every tile is one of the heaviest per-frame costs. Results
        # are memoized per position (the map is static, so FOV is deterministic).
        vis_key = None if player_pos is None else (player_ent, player_pos.x, player_pos.y)
        if vis_key != self._visible_cache_key:
            self._visible_cache_key = vis_key
            if vis_key is None:
                self._visible_tiles = set()
                self._visible_bbox = None
                self._shadow_cells = ()
            else:
                cached = self._fov_cache.get(vis_key)
                if cached is None:
                    visible = self._compute_visible_tiles(player_ent, player_pos)
                    cached = (visible, *self._fov_bbox_and_shadows(visible))
                    self._fov_cache[vis_key] = cached
                self._visible_tiles, self._visible_bbox, self._shadow_cells = cached
            # Field of view changed, so the remembered region around it has too.
            # (Unlike FOV this can't be memoized per position: the explored set
            # keeps growing, so a revisited tile may recall more than last time.)
            self._compute_memory_geometry()
        draw_autotile_variant = getattr(r, "draw_autotile_variant", None)

        self._render_map_layer(r, draw_autotile_variant)

        # draw tuple: (vx, vy, glyph, classification, fg, bg, force_glyph)
        DrawData = tuple[int, int, str, str, tuple[int, int, int] | None, tuple[int, int, int] | None, bool]
        player_draw: DrawData | None = None
        character_draws: list[DrawData] = []
        # Static scenery seen this frame, keyed by tile -- folded into tile memory
        # after drawing so it can be recalled once the tile leaves view. Gathered
        # here to piggyback on the entity scan rather than sweep every entity twice.
        seen_scenery: dict[tuple[int, int], tuple[str, str, tuple[int, int, int] | None, tuple[int, int, int] | None]] = {}
        for ent, (pos, rend) in esper.get_components(Position, Renderable):
            is_player = esper.has_component(ent, Player)
            is_character = is_player or esper.has_component(ent, NPC)
            if (pos.x, pos.y) not in self._visible_tiles and not is_player:
                continue

            if (pos.x, pos.y) in self._visible_tiles and is_memorable_scenery(ent):
                seen_scenery[(pos.x, pos.y)] = (rend.glyph, "default", rend.fg, rend.bg)

            view_xy = self._world_to_view(pos.x, pos.y)
            if view_xy is None and not is_player:
                continue

            classification = "default"
            if is_player or esper.has_component(ent, Friendly):
                classification = "friendly"
            elif esper.has_component(ent, Enemy):
                classification = "enemy"

            if is_character:
                glyph, fg, force_glyph = self._status_appearance(ent, pos, rend.glyph, rend.fg)
            else:
                glyph, fg, force_glyph = rend.glyph, rend.fg, False

            if is_player:
                if view_xy is not None:
                    player_draw = (view_xy[0], view_xy[1], glyph, classification, fg, rend.bg, force_glyph)
                continue

            if view_xy is not None:
                draw_data = (view_xy[0], view_xy[1], glyph, classification, fg, rend.bg, force_glyph)
                if is_character:
                    character_draws.append(draw_data)
                else:
                    r.draw_glyph_classified(
                        draw_data[0], draw_data[1], draw_data[2], draw_data[3],
                        fg=draw_data[4], bg=draw_data[5], force_glyph=draw_data[6],
                    )

        for draw_data in character_draws:
            r.draw_glyph_classified(
                draw_data[0], draw_data[1], draw_data[2], draw_data[3],
                fg=draw_data[4], bg=draw_data[5], force_glyph=draw_data[6],
            )

        if player_draw is not None:
            r.draw_glyph_classified(
                player_draw[0], player_draw[1], player_draw[2], player_draw[3],
                fg=player_draw[4], bg=player_draw[5], force_glyph=player_draw[6],
            )

        # Commit this frame's sightings to tile memory (marks tiles explored and
        # records/refreshes their scenery), ready to be recalled next time they
        # fall out of view.
        self._update_tile_memory(seen_scenery)

        self._draw_sidebar(player_pos)

        # Dim the world when night draws in (a translucent wash over everything
        # already drawn). Skipped on renderers without the overlay hook (tests).
        clock = world_clock()
        overlay_alpha = night_overlay_alpha(clock)
        draw_overlay = getattr(r, "draw_overlay", None)
        if overlay_alpha > 0 and callable(draw_overlay):
            draw_overlay(_NIGHT_TINT, overlay_alpha)

        # Speech bubbles float above characters in world space (aligned to the
        # tile grid, not the UI text grid), drawn on top of the night wash so they
        # stay legible. Skipped on renderers without the hook (the test double).
        draw_world_label = getattr(r, "draw_world_label", None)
        measure_world_label = getattr(r, "measure_world_label", None)
        get_cell_px = getattr(r, "get_tile_cell_size_px", None)
        if callable(draw_world_label):
            now = self._clock()
            bubbles = active_bubbles(now)
            if callable(measure_world_label) and callable(get_cell_px):
                # Lay bubbles out newest-first: each one takes the lowest vertical
                # slot (nearest its character) that doesn't overlap an already-placed
                # bubble, lifting up a row at a time until it fits. So a newer bubble
                # sits below and shoves older/neighbouring ones up -- no two overlap.
                cell_w, cell_h = get_cell_px()
                cell_w, cell_h = max(1, cell_w), max(1, cell_h)
                placed: list[tuple[int, int, int, int]] = []
                for bubble in reversed(bubbles):
                    view = self._world_to_view(bubble.x, bubble.y)
                    if view is None:
                        continue
                    vx, vy = view
                    body_w, body_h, tail = measure_world_label(bubble.text, bubble.indicator)
                    step = body_h + max(2, cell_h // 8)
                    x0 = vx * cell_w + cell_w // 2 - body_w // 2
                    lift = 0
                    for _ in range(64):  # bounded so a dense pile can't spin forever
                        top = vy * cell_h - body_h - tail - lift
                        rect = (x0, top, x0 + body_w, top + body_h)
                        if not any(_rects_overlap(rect, other) for other in placed):
                            break
                        lift += step
                    placed.append(rect)
                    draw_world_label(
                        vx, vy, bubble.text,
                        indicator=bubble.indicator,
                        indicator_color=bubble.indicator_color,
                        lift=lift,
                        alpha=bubble_alpha(bubble, now),
                    )
            else:
                # No measurement hook (test double): draw without overlap layout.
                for bubble in bubbles:
                    view = self._world_to_view(bubble.x, bubble.y)
                    if view is not None:
                        draw_world_label(
                            view[0], view[1], bubble.text,
                            indicator=bubble.indicator,
                            indicator_color=bubble.indicator_color,
                            alpha=bubble_alpha(bubble, now),
                        )

        # The look cursor rides on top of the night wash so it stays bright while
        # examining. Only drawn when the cursor cell is inside the viewport.
        if self.look_cursor is not None:
            draw_cursor = getattr(r, "draw_cursor", None)
            cursor_view = self._world_to_view(*self.look_cursor)
            if cursor_view is not None and callable(draw_cursor):
                draw_cursor(cursor_view[0], cursor_view[1])

        draw_text_clipped = getattr(r, "draw_text_clipped", None)
        time_text = f"{format_datetime(clock)}  "
        needs_text = ""
        if player_ent is not None and esper.has_component(player_ent, Needs):
            needs = esper.component_for_entity(player_ent, Needs)
            needs_text = (
                f"Hunger {int(needs.hunger)}%  Thirst {int(needs.thirst)}%  "
                f"Tired {int(needs.tiredness)}%    "
            )
        # While looking, the examine line replaces the usual hint bar so the
        # player can read what's under the cursor and which actions it affords.
        if self.look_info:
            status_line = self.look_info
        else:
            status_line = f"{time_text}{needs_text}I inv  C status  L look  R sleep  Esc menu"
        if callable(draw_text_clipped):
            draw_text_clipped(0, self._status_y, status_line, self._grid_w)
        else:
            r.draw_text(0, self._status_y, status_line)
        r.present()


