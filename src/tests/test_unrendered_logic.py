from __future__ import annotations

from pathlib import Path

import esper
import pytest

from components import Player, Position
from game_map import GameMap
from main import _action_from_held_keys
from persistence import load_game, save_game
from systems import MovementProcessor

pytestmark = pytest.mark.unrendered


def test_gamemap_has_walls_on_border_and_floor_inside() -> None:
    game_map = GameMap(8, 6)

    assert game_map.tile_at(0, 0) == game_map.WALL
    assert game_map.tile_at(7, 5) == game_map.WALL
    assert game_map.tile_at(1, 1) == game_map.FLOOR


def test_movement_processor_moves_player_without_renderer() -> None:
    game_map = GameMap(10, 6)
    player_pos = Position(3, 3)
    esper.create_entity(player_pos, Player())
    esper.add_processor(MovementProcessor(game_map), priority=1)

    esper.process("move_right")
    esper.process("move_up")

    assert (player_pos.x, player_pos.y) == (4, 2)


def test_movement_processor_supports_diagonal_player_move() -> None:
    game_map = GameMap(10, 10)
    player_pos = Position(3, 3)
    esper.create_entity(player_pos, Player())
    esper.add_processor(MovementProcessor(game_map), priority=1)

    esper.process("move_down_right")

    assert (player_pos.x, player_pos.y) == (4, 4)


def test_npc_ai_can_move_diagonally_toward_player() -> None:
    from components import BlocksMovement, Enemy, NPC, Vision
    from systems import NpcAiProcessor

    game_map = GameMap(20, 20)
    player_pos = Position(12, 12)
    npc_pos = Position(6, 6)

    esper.create_entity(player_pos, Player(), BlocksMovement(), Vision(12))
    esper.create_entity(npc_pos, NPC(), Enemy(), BlocksMovement(), Vision(12))

    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(NpcAiProcessor(game_map), priority=0)

    esper.process("move_down_right")

    assert (npc_pos.x, npc_pos.y) == (7, 7)


def test_action_from_held_keys_supports_cardinal_and_diagonal() -> None:
    held: set[str] = set()

    held.add("move_up")
    action = _action_from_held_keys(held)
    assert action == "move_up"

    held.add("move_left")
    action = _action_from_held_keys(held)
    assert action == "move_up_left"

    held.discard("move_left")
    held.add("move_right")
    action = _action_from_held_keys(held)
    assert action == "move_up_right"

    held.discard("move_up")
    held.add("move_down")
    held.discard("move_right")
    held.add("move_left")
    action = _action_from_held_keys(held)
    assert action == "move_down_left"

    held.discard("move_left")
    held.add("move_right")
    held.discard("move_down")
    action = _action_from_held_keys(held)
    assert action == "move_right"


def test_action_from_held_keys_cancels_opposite_axis() -> None:
    held = {"move_up", "move_down", "move_left"}
    action = _action_from_held_keys(held)
    assert action == "move_left"


def test_load_game_returns_fallback_when_file_missing(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing_save.json"

    game_map, player_pos = load_game(missing_file, fallback_width=9, fallback_height=7)

    assert (game_map.width, game_map.height) == (9, 7)
    assert (player_pos.x, player_pos.y) == (4, 3)


def test_save_and_load_roundtrip_without_renderer(tmp_path: Path) -> None:
    game_map = GameMap(12, 8)
    save_file = tmp_path / "roundtrip_save.json"
    start_pos = Position(5, 4)

    save_game(game_map, save_file, start_pos)
    loaded_map, loaded_pos = load_game(save_file, fallback_width=2, fallback_height=2)

    assert (loaded_map.width, loaded_map.height) == (12, 8)
    assert (loaded_pos.x, loaded_pos.y) == (5, 4)


def test_line_of_sight_blocked_by_wall() -> None:
    game_map = GameMap(12, 7)
    game_map.tiles[3][5] = game_map.WALL

    assert game_map.has_line_of_sight((3, 3), (9, 3)) is False


def test_find_path_routes_around_obstacle() -> None:
    game_map = GameMap(9, 7)
    game_map.tiles[3][4] = game_map.WALL

    path = game_map.find_path((2, 3), (6, 3))

    assert path
    assert path[-1] == (6, 3)
    assert (4, 3) not in path


def test_large_default_map_contains_buildings() -> None:
    game_map = GameMap(40, 20)

    assert game_map.tile_at(4, 3) == game_map.WALL
    assert game_map.tile_at(12, 8) == game_map.WALL
    assert game_map.tile_at(8, 8) == game_map.FLOOR

    assert game_map.tile_at(26, 5) == game_map.WALL
    assert game_map.tile_at(35, 11) == game_map.WALL
    assert game_map.tile_at(30, 11) == game_map.FLOOR
