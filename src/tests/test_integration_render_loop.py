from __future__ import annotations

import esper
import pytest

from components import Player, Position, Renderable
from game_map import GameMap
from systems import MovementProcessor, RenderProcessor
from fakes import FakeRenderer


pytestmark = pytest.mark.headless_renderer


def _setup_world(width: int = 10, height: int = 6) -> tuple[GameMap, FakeRenderer, Position]:
    game_map = GameMap(width, height)
    renderer = FakeRenderer()

    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(RenderProcessor(renderer, game_map), priority=0)

    player_pos = Position(width // 2, height // 2)
    esper.create_entity(player_pos, Renderable("@"), Player())

    return game_map, renderer, player_pos


def test_initial_tick_renders_map_player_and_status_line() -> None:
    game_map, renderer, player_pos = _setup_world()

    esper.process(None)

    assert renderer.present_calls == 1
    assert renderer.glyphs[(0, 0)] == game_map.WALL
    assert renderer.glyphs[(1, 1)] == game_map.FLOOR
    assert renderer.glyphs[(player_pos.x, player_pos.y)] == "@"
    assert (0, game_map.height, "move: arrows/hjkl/wasd   menu: esc") in renderer.text


def test_move_action_updates_position_and_rendered_player_location() -> None:
    _game_map, renderer, player_pos = _setup_world()

    start = (player_pos.x, player_pos.y)
    esper.process("move_right")

    assert (player_pos.x, player_pos.y) == (start[0] + 1, start[1])
    assert renderer.glyphs[(player_pos.x, player_pos.y)] == "@"


def test_player_does_not_move_through_wall() -> None:
    _game_map, _renderer, player_pos = _setup_world(width=8, height=6)

    player_pos.x = 1
    start = (player_pos.x, player_pos.y)

    esper.process("move_left")

    assert (player_pos.x, player_pos.y) == start
