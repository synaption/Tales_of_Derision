from __future__ import annotations

import esper
import pytest

from components import Name, Player, Position, Renderable
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
    esper.create_entity(player_pos, Renderable("@"), Name("You"), Player())
    esper.create_entity(Position(player_pos.x + 2, player_pos.y), Renderable("g"), Name("Goblin"))

    return game_map, renderer, player_pos


def test_initial_tick_renders_map_player_and_status_line() -> None:
    game_map, renderer, player_pos = _setup_world()

    esper.process(None)

    assert renderer.present_calls == 1
    assert renderer.glyphs[(0, 0)] == game_map.WALL
    assert renderer.glyphs[(1, 1)] == game_map.FLOOR
    assert renderer.glyphs[(player_pos.x, player_pos.y)] == "@"
    assert (0, game_map.height, "move: arrows/hjkl/wasd   menu: esc") in renderer.text
    sidebar_x = game_map.width + 2
    assert (sidebar_x, 0, "== NEARBY ==") in renderer.text
    assert any(text.startswith("g → Goblin") for _x, _y, text in renderer.text)
    assert any(text == "== LOG ==" for _x, _y, text in renderer.text)


def test_nearby_section_is_dynamic_and_log_starts_after_items() -> None:
    game_map = GameMap(30, 14)
    renderer = FakeRenderer()

    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(RenderProcessor(renderer, game_map), priority=0)

    player_pos = Position(15, 7)
    esper.create_entity(player_pos, Renderable("@"), Name("You"), Player())
    esper.create_entity(Position(16, 7), Renderable("g"), Name("Goblin"))
    esper.create_entity(Position(14, 7), Renderable("o"), Name("Orc"))
    esper.create_entity(Position(15, 8), Renderable("r"), Name("Rat"))

    esper.process(None)

    sidebar_x = game_map.width + 2
    nearby_header_y = next(y for x, y, text in renderer.text if x == sidebar_x and text == "== NEARBY ==")
    log_header_y = next(y for x, y, text in renderer.text if x == sidebar_x and text == "== LOG ==")
    nearby_rows = [
        y
        for x, y, text in renderer.text
        if x == sidebar_x and nearby_header_y < y < log_header_y and text and "==" not in text
    ]

    assert len(nearby_rows) >= 3
    assert log_header_y == max(nearby_rows) + 2


def test_move_action_updates_position_and_rendered_player_location() -> None:
    _game_map, renderer, player_pos = _setup_world()

    start = (player_pos.x, player_pos.y)
    esper.process("move_right")

    assert (player_pos.x, player_pos.y) == (start[0] + 1, start[1])
    assert renderer.glyphs[(player_pos.x, player_pos.y)] == "@"


def test_blocked_movement_adds_log_event() -> None:
    _game_map, renderer, player_pos = _setup_world(width=8, height=20)

    player_pos.x = 1
    esper.process(None)
    esper.process("move_left")

    assert any(text == "You bump into a wall." for _x, _y, text in renderer.text)


def test_player_does_not_move_through_wall() -> None:
    _game_map, _renderer, player_pos = _setup_world(width=8, height=6)

    player_pos.x = 1
    start = (player_pos.x, player_pos.y)

    esper.process("move_left")

    assert (player_pos.x, player_pos.y) == start
