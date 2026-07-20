"""ECS processors (systems).

esper 3.x uses module-level state and esper.Processor subclasses whose
process() receives whatever args are passed to esper.process().
"""
from collections.abc import Callable
import textwrap

import esper

from components import BlocksMovement, Corpse, Enemy, Friendly, Name, NPC, Player, Position, Renderable, Vision
from game_map import GameMap
from renderer.base import Renderer

_ACTION_DELTAS = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
    "move_up_left": (-1, -1),
    "move_up_right": (1, -1),
    "move_down_left": (-1, 1),
    "move_down_right": (1, 1),
}

_DIR_TO_ARROW = {
    (-1, -1): "↖",
    (0, -1): "↑",
    (1, -1): "↗",
    (-1, 0): "←",
    (0, 0): "•",
    (1, 0): "→",
    (-1, 1): "↙",
    (0, 1): "↓",
    (1, 1): "↘",
}

_ARROW_TO_WORD = {
    "↖": "northwest",
    "↑": "north",
    "↗": "northeast",
    "←": "west",
    "•": "here",
    "→": "east",
    "↙": "southwest",
    "↓": "south",
    "↘": "southeast",
}

_WALL_BROWN = (124, 88, 56)

_AUTOTILE_DIRECTION_OFFSETS = {
    1: (-1, -1),
    2: (0, -1),
    3: (1, -1),
    4: (1, 0),
    5: (1, 1),
    6: (0, 1),
    7: (-1, 1),
    8: (-1, 0),
}

_AUTOTILE_DIRECTION_MASKS = {
    1: 1,    # NW
    2: 2,    # N
    3: 4,    # NE
    4: 16,   # E
    5: 128,  # SE
    6: 64,   # S
    7: 32,   # SW
    8: 8,    # W
}

_TURN_EVENTS: list[str] = []

_NearbyEntry = tuple[
    int,
    str,
    str,
    str,
    int,
    int,
    str,
    tuple[int, int, int] | None,
    tuple[int, int, int] | None,
]


def _push_turn_event(text: str) -> None:
    _TURN_EVENTS.append(text)


def _pull_turn_events() -> list[str]:
    events = list(_TURN_EVENTS)
    _TURN_EVENTS.clear()
    return events


class MovementProcessor(esper.Processor):
    """Applies a movement action to every player-controlled entity."""

    def __init__(
        self,
        game_map: GameMap,
        on_melee_attack: Callable[[], None] | None = None,
        on_enemy_death: Callable[[], None] | None = None,
    ):
        self.game_map = game_map
        self.on_melee_attack = on_melee_attack
        self.on_enemy_death = on_enemy_death

    def process(self, action: str | None = None) -> None:
        delta = _ACTION_DELTAS.get(action)
        if delta is None:
            return
        dx, dy = delta

        occupied = {
            (pos.x, pos.y): ent
            for ent, (pos, _blocks) in esper.get_components(Position, BlocksMovement)
        }

        for _ent, (pos, _player) in esper.get_components(Position, Player):
            nx, ny = pos.x + dx, pos.y + dy
            target_occupied = (nx, ny) in occupied and occupied[(nx, ny)] != _ent
            if self.game_map.is_walkable(nx, ny):
                if not target_occupied:
                    pos.x, pos.y = nx, ny
                    continue

                target_ent = occupied[(nx, ny)]
                target_name = "Unknown"
                if esper.has_component(target_ent, Name):
                    target_name = esper.component_for_entity(target_ent, Name).value

                if esper.has_component(target_ent, Enemy):
                    _push_turn_event(f"You attack {target_name}.")
                    if self.on_melee_attack is not None:
                        self.on_melee_attack()
                    corpse_name = f"Corpse of {target_name}"
                    esper.delete_entity(target_ent, immediate=True)
                    esper.create_entity(Position(nx, ny), Renderable("%"), Name(corpse_name), Corpse())
                    if self.on_enemy_death is not None:
                        self.on_enemy_death()
                    continue

                if esper.has_component(target_ent, Friendly):
                    _push_turn_event(f"{target_name} blocks your way. Press Enter to interact.")
                    continue

                _push_turn_event("Something blocks your way.")


class NpcAiProcessor(esper.Processor):
    """Makes NPCs chase the player when they have line of sight."""

    def __init__(self, game_map: GameMap):
        self.game_map = game_map

    def _find_player_position(self) -> tuple[int, int] | None:
        for _ent, (pos, _player) in esper.get_components(Position, Player):
            return (pos.x, pos.y)
        return None

    def process(self, action: str | None = None) -> None:
        if action not in _ACTION_DELTAS:
            return

        player_xy = self._find_player_position()
        if player_xy is None:
            return

        occupied = {
            (pos.x, pos.y): ent
            for ent, (pos, _blocks) in esper.get_components(Position, BlocksMovement)
        }

        for ent, (pos, _npc, _enemy) in esper.get_components(Position, NPC, Enemy):
            vision_radius = 8
            if esper.has_component(ent, Vision):
                vision_radius = esper.component_for_entity(ent, Vision).radius

            dx = player_xy[0] - pos.x
            dy = player_xy[1] - pos.y
            if max(abs(dx), abs(dy)) > vision_radius:
                continue
            if not self.game_map.has_line_of_sight((pos.x, pos.y), player_xy):
                continue

            blocked_tiles = {
                tile_xy
                for tile_xy, occ_ent in occupied.items()
                if occ_ent not in {ent}
            }
            path = self.game_map.find_path((pos.x, pos.y), player_xy, blocked_tiles=blocked_tiles)
            if not path:
                continue

            next_x, next_y = path[0]
            if (next_x, next_y) == player_xy:
                continue

            if (next_x, next_y) in occupied and occupied[(next_x, next_y)] != ent:
                continue

            old_xy = (pos.x, pos.y)
            pos.x, pos.y = next_x, next_y
            occupied.pop(old_xy, None)
            occupied[(next_x, next_y)] = ent


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
        self._status_y = game_map.height
        self._message_log: list[str] = ["You enter the area."]
        self._max_log_lines = 200
        self._last_player_pos: tuple[int, int] | None = None
        self._seen_entity_ids: set[int] = set()
        self._visible_tiles: set[tuple[int, int]] = set()

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

        visible: set[tuple[int, int]] = set()
        origin = (player_pos.x, player_pos.y)
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                if max(abs(x - player_pos.x), abs(y - player_pos.y)) > radius:
                    continue
                if self.game_map.has_line_of_sight(origin, (x, y)):
                    visible.add((x, y))
        return visible

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

    def _draw_sidebar(
        self,
        player_pos: Position | None,
        _entity_lookup: dict[tuple[int, int], tuple[str, str]],
    ) -> None:
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

    def process(self, action: str | None = None) -> None:
        r = self.renderer
        r.clear()

        entity_lookup: dict[tuple[int, int], tuple[str, str]] = {}
        player_pos: Position | None = None
        player_ent: int | None = None

        for ent, (pos, rend) in esper.get_components(Position, Renderable):
            name = "Unknown"
            if esper.has_component(ent, Name):
                name = esper.component_for_entity(ent, Name).value
            entity_lookup[(pos.x, pos.y)] = (rend.glyph, name)
            if esper.has_component(ent, Player):
                player_pos = pos
                player_ent = ent

        if player_pos is not None and self._last_player_pos is None:
            self._last_player_pos = (player_pos.x, player_pos.y)

        events = _pull_turn_events()
        for event in events:
            self._append_message(event)

        self._describe_movement(action, player_pos, suppress_wall_message=bool(events))
        self._compute_layout(player_pos)

        self._visible_tiles = self._compute_visible_tiles(player_ent, player_pos)
        draw_autotile_variant = getattr(r, "draw_autotile_variant", None)

        for vy in range(self._view_height):
            wy = self._view_origin_y + vy
            for vx in range(self._view_width):
                wx = self._view_origin_x + vx
                if (wx, wy) in self._visible_tiles:
                    tile = self.game_map.tile_at(wx, wy)
                    if tile == self.game_map.WALL and callable(draw_autotile_variant):
                        wall_mask = self._neighbor_mask(wx, wy, self.game_map.WALL)
                        drew = bool(
                            draw_autotile_variant(
                                vx,
                                vy,
                                "wall",
                                wall_mask,
                                fg=_WALL_BROWN,
                                bg=None,
                            )
                        )
                        if drew:
                            continue

                    classification = "wall" if tile == self.game_map.WALL else "default"
                    tile_fg = _WALL_BROWN if classification == "wall" else None
                    r.draw_glyph_classified(vx, vy, tile, classification, fg=tile_fg, bg=None)
                else:
                    r.draw_glyph(vx, vy, " ")

        player_draw: tuple[int, int, str, str, tuple[int, int, int] | None, tuple[int, int, int] | None] | None = None
        character_draws: list[
            tuple[int, int, str, str, tuple[int, int, int] | None, tuple[int, int, int] | None]
        ] = []
        for ent, (pos, rend) in esper.get_components(Position, Renderable):
            is_player = esper.has_component(ent, Player)
            is_character = is_player or esper.has_component(ent, NPC)
            if (pos.x, pos.y) not in self._visible_tiles and not is_player:
                continue

            view_xy = self._world_to_view(pos.x, pos.y)
            if view_xy is None and not is_player:
                continue

            classification = "default"
            if is_player or esper.has_component(ent, Friendly):
                classification = "friendly"
            elif esper.has_component(ent, Enemy):
                classification = "enemy"

            if is_player:
                if view_xy is not None:
                    player_draw = (view_xy[0], view_xy[1], rend.glyph, classification, rend.fg, rend.bg)
                continue

            if view_xy is not None:
                draw_data = (view_xy[0], view_xy[1], rend.glyph, classification, rend.fg, rend.bg)
                if is_character:
                    character_draws.append(draw_data)
                else:
                    r.draw_glyph_classified(
                        draw_data[0],
                        draw_data[1],
                        draw_data[2],
                        draw_data[3],
                        fg=draw_data[4],
                        bg=draw_data[5],
                    )

        for draw_data in character_draws:
            r.draw_glyph_classified(
                draw_data[0],
                draw_data[1],
                draw_data[2],
                draw_data[3],
                fg=draw_data[4],
                bg=draw_data[5],
            )

        if player_draw is not None:
            r.draw_glyph_classified(
                player_draw[0],
                player_draw[1],
                player_draw[2],
                player_draw[3],
                fg=player_draw[4],
                bg=player_draw[5],
            )

        self._draw_sidebar(player_pos, entity_lookup)

        draw_text_clipped = getattr(r, "draw_text_clipped", None)
        status_line = "I inventory  Esc menu  +/- tile scale"
        if callable(draw_text_clipped):
            draw_text_clipped(0, self._status_y, status_line, self._grid_w)
        else:
            r.draw_text(0, self._status_y, status_line)
        r.present()
