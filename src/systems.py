"""ECS processors (systems).

esper 3.x uses module-level state and esper.Processor subclasses whose
process() receives whatever args are passed to esper.process().
"""
from collections.abc import Callable

import esper

from components import BlocksMovement, Corpse, Dialogue, Enemy, Friendly, Name, NPC, Player, Position, Renderable, Vision
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

_TURN_EVENTS: list[str] = []


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
                    pos.x, pos.y = nx, ny
                    continue

                if esper.has_component(target_ent, Dialogue):
                    line = esper.component_for_entity(target_ent, Dialogue).line
                    _push_turn_event(f"{target_name} says: \"{line}\"")
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
        self.sidebar_width = 28
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
            self._append_message("You bump into a wall.")

        self._last_player_pos = (player_pos.x, player_pos.y)

    def _collect_nearby_objects(
        self,
        player_pos: Position,
    ) -> list[tuple[int, str, str, str, int, int, int, int]]:
        nearby: list[tuple[int, str, str, str, int, int, int, int]] = []

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
            nearby.append((ent, rend.glyph, arrow, name, abs(dx) + abs(dy), max(abs(dx), abs(dy),), pos.x, pos.y))

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
        x0 = self.sidebar_x
        y = 0

        nearby_data: list[tuple[int, str, str, str, int, int, int, int]] = []
        if player_pos is not None:
            visible_npcs = self._collect_visible_npcs(player_pos)
            self._update_sighting_events(visible_npcs)
            nearby_data = self._collect_nearby_objects(player_pos)

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
                for _ent_id, glyph, arrow, name, _mdist, _cdist, _x, _y in nearby_data[:max_nearby_lines]:
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

        self._visible_tiles = self._compute_visible_tiles(player_ent, player_pos)

        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                if (x, y) in self._visible_tiles:
                    tile = self.game_map.tile_at(x, y)
                    classification = "wall" if tile == self.game_map.WALL else "default"
                    r.draw_glyph_classified(x, y, tile, classification)
                else:
                    r.draw_glyph(x, y, " ")

        player_draw: tuple[int, int, str, str] | None = None
        for ent, (pos, rend) in esper.get_components(Position, Renderable):
            is_player = esper.has_component(ent, Player)
            if (pos.x, pos.y) not in self._visible_tiles and not is_player:
                continue

            classification = "default"
            if is_player or esper.has_component(ent, Friendly):
                classification = "friendly"
            elif esper.has_component(ent, Enemy):
                classification = "enemy"

            if is_player:
                player_draw = (pos.x, pos.y, rend.glyph, classification)
                continue

            r.draw_glyph_classified(pos.x, pos.y, rend.glyph, classification)

        if player_draw is not None:
            r.draw_glyph_classified(player_draw[0], player_draw[1], player_draw[2], player_draw[3])

        self._draw_sidebar(player_pos, entity_lookup)

        r.draw_text(0, self.game_map.height, "move: arrows/hjkl/wasd   inventory: i   menu: esc")
        r.present()
