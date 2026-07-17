"""ECS processors (systems).

esper 3.x uses module-level state and esper.Processor subclasses whose
process() receives whatever args are passed to esper.process().
"""
import esper

from components import Name, Player, Position, Renderable
from game_map import GameMap
from renderer.base import Renderer

_ACTION_DELTAS = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
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


class MovementProcessor(esper.Processor):
    """Applies a movement action to every player-controlled entity."""

    def __init__(self, game_map: GameMap):
        self.game_map = game_map

    def process(self, action: str | None = None) -> None:
        delta = _ACTION_DELTAS.get(action)
        if delta is None:
            return
        dx, dy = delta
        for _ent, (pos, _player) in esper.get_components(Position, Player):
            nx, ny = pos.x + dx, pos.y + dy
            if self.game_map.is_walkable(nx, ny):
                pos.x, pos.y = nx, ny


class RenderProcessor(esper.Processor):
    """Draws the map, then all Renderable entities on top of it."""

    def __init__(self, renderer: Renderer, game_map: GameMap):
        self.renderer = renderer
        self.game_map = game_map
        self.sidebar_x = game_map.width + 2
        self.sidebar_width = 28
        self._message_log: list[str] = ["You enter the area."]
        self._max_log_lines = 200
        self._last_player_pos: tuple[int, int] | None = None
        self._visible_entity_keys: set[tuple[str, int, int]] = set()

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

    def _append_message(self, text: str) -> None:
        self._message_log.append(text)
        if len(self._message_log) > self._max_log_lines:
            self._message_log = self._message_log[-self._max_log_lines :]

    def _describe_movement(self, action: str | None, player_pos: Position | None) -> None:
        if action not in _ACTION_DELTAS or player_pos is None:
            return

        if self._last_player_pos is None:
            self._last_player_pos = (player_pos.x, player_pos.y)
            return

        moved = self._last_player_pos != (player_pos.x, player_pos.y)
        if not moved:
            self._append_message("You bump into a wall.")

        self._last_player_pos = (player_pos.x, player_pos.y)

    def _collect_nearby_objects(
        self,
        player_pos: Position,
    ) -> list[tuple[str, str, str, int, int, int, int]]:
        nearby: list[tuple[str, str, str, int, int, int, int]] = []
        radius = max(self.game_map.width // 2, self.game_map.height // 2)

        for ent, (pos, rend) in esper.get_components(Position, Renderable):
            if pos.x == player_pos.x and pos.y == player_pos.y:
                continue

            dx = pos.x - player_pos.x
            dy = pos.y - player_pos.y
            if max(abs(dx), abs(dy)) > radius:
                continue

            arrow = self._direction_arrow(player_pos, pos)
            name = "Unknown"
            if esper.has_component(ent, Name):
                name = esper.component_for_entity(ent, Name).value
            nearby.append((rend.glyph, arrow, name, abs(dx) + abs(dy), max(abs(dx), abs(dy),), pos.x, pos.y))

        nearby.sort(key=lambda item: (item[3], item[4], item[2]))
        return nearby

    def _update_sighting_events(self, nearby: list[tuple[str, str, str, int, int, int, int]]) -> None:
        current_keys = {(name, x, y) for _g, _a, name, _md, _cd, x, y in nearby}
        newly_seen = [item for item in nearby if (item[2], item[5], item[6]) not in self._visible_entity_keys]

        for _glyph, arrow, name, _md, _cd, _x, _y in newly_seen[:2]:
            direction = _ARROW_TO_WORD.get(arrow, "nearby")
            self._append_message(f"You notice {name} to the {direction}.")

        self._visible_entity_keys = current_keys

    def _draw_sidebar(
        self,
        player_pos: Position | None,
        _entity_lookup: dict[tuple[int, int], tuple[str, str]],
    ) -> None:
        r = self.renderer
        x0 = self.sidebar_x
        y = 0

        nearby_data: list[tuple[str, str, str, int, int, int, int]] = []
        if player_pos is not None:
            nearby_data = self._collect_nearby_objects(player_pos)
            self._update_sighting_events(nearby_data)

        r.draw_text(x0, y, "== NEARBY ==")
        y += 1

        max_nearby_lines = max(0, self.game_map.height - y - 2)
        nearby_lines_drawn = 0

        if player_pos is None:
            if max_nearby_lines > 0:
                r.draw_text(x0, y, "none")
                y += 1
                nearby_lines_drawn = 1
        else:
            if not nearby_data:
                if max_nearby_lines > 0:
                    r.draw_text(x0, y, "none")
                    y += 1
                    nearby_lines_drawn = 1
            else:
                for glyph, arrow, name, _mdist, _cdist, _x, _y in nearby_data[:max_nearby_lines]:
                    line = f"{glyph} {arrow} {name}"
                    r.draw_text(x0, y, line[: self.sidebar_width])
                    y += 1
                    nearby_lines_drawn += 1

        if nearby_lines_drawn > 0:
            y += 1

        r.draw_text(x0, y, "== LOG ==")
        y += 1

        log_height = max(1, self.game_map.height - y)
        messages = self._message_log[-log_height:]
        for idx, message in enumerate(messages):
            r.draw_text(x0, y + idx, message[: self.sidebar_width])

    def process(self, action: str | None = None) -> None:
        r = self.renderer
        r.clear()

        entity_lookup: dict[tuple[int, int], tuple[str, str]] = {}
        player_pos: Position | None = None

        for ent, (pos, rend) in esper.get_components(Position, Renderable):
            name = "Unknown"
            if esper.has_component(ent, Name):
                name = esper.component_for_entity(ent, Name).value
            entity_lookup[(pos.x, pos.y)] = (rend.glyph, name)
            if esper.has_component(ent, Player):
                player_pos = pos

        if player_pos is not None and self._last_player_pos is None:
            self._last_player_pos = (player_pos.x, player_pos.y)

        self._describe_movement(action, player_pos)

        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                r.draw_glyph(x, y, self.game_map.tile_at(x, y))

        for _ent, (pos, rend) in esper.get_components(Position, Renderable):
            r.draw_glyph(pos.x, pos.y, rend.glyph)

        self._draw_sidebar(player_pos, entity_lookup)

        r.draw_text(0, self.game_map.height, "move: arrows/hjkl/wasd   menu: esc")
        r.present()
