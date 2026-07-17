"""ECS processors (systems).

esper 3.x uses module-level state and esper.Processor subclasses whose
process() receives whatever args are passed to esper.process().
"""
import os

import esper

from components import Player, Position, Renderable
from game_map import GameMap
from renderer.base import Renderer

_ACTION_DELTAS = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
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
        full_commit = os.environ.get("GAME_COMMIT", "dev")
        self.build_label = f"build: {full_commit[:12]}"

    def process(self, action: str | None = None) -> None:
        r = self.renderer
        r.clear()

        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                r.draw_glyph(x, y, self.game_map.tile_at(x, y))

        for _ent, (pos, rend) in esper.get_components(Position, Renderable):
            r.draw_glyph(pos.x, pos.y, rend.glyph)

        r.draw_text(0, self.game_map.height, "move: arrows/hjkl/wasd   menu: esc")
        r.draw_text(0, self.game_map.height + 1, self.build_label)
        r.present()
