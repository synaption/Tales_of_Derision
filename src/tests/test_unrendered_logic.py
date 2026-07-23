from __future__ import annotations

from pathlib import Path

import esper
import pytest

from components import Enemy, NPC, Player, Position
from game_map import GameMap
from interactions import _action_from_held_keys
from worldgen import _setup_world
from persistence import load_game, save_game
from systems import MovementProcessor

pytestmark = pytest.mark.unrendered


def test_gamemap_has_walls_on_border_and_floor_inside() -> None:
    game_map = GameMap(8, 6)

    assert game_map.tile_at(0, 0) == game_map.WALL
    assert game_map.tile_at(7, 5) == game_map.WALL
    assert game_map.tile_at(1, 1) == game_map.FLOOR


def test_world_map_sets_land_island_in_a_vast_ocean() -> None:
    # A map twice the land in each dimension becomes the ocean world: a centred
    # 120x60 land island ringed by open sea (the eight surrounding sections).
    game_map = GameMap(360, 180)

    assert game_map.has_ocean is True
    assert (game_map.land_x0, game_map.land_y0) == (120, 60)
    assert (game_map.land_w, game_map.land_h) == (120, 60)

    # Land centre is dry ground; the sea far outside the island is open ocean.
    assert game_map.tile_at(180, 90) == game_map.FLOOR
    assert game_map.is_ocean(10, 10) is True
    # The island's edge is a water coastline (not ocean, since it's inside the
    # land rectangle), and only the map's outermost ring stays wall.
    assert game_map.tile_at(120, 90) == game_map.WATER
    assert game_map.is_ocean(120, 90) is False
    assert game_map.tile_at(0, 0) == game_map.WALL


def test_small_maps_stay_a_plain_walled_room() -> None:
    game_map = GameMap(40, 20)
    assert game_map.has_ocean is False
    assert game_map.tile_at(0, 0) == game_map.WALL
    assert game_map.tile_at(20, 10) == game_map.FLOOR


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


def test_action_from_held_keys_prefers_most_recent_opposite_key_when_provided() -> None:
    # Simulates holding S+D, then pressing A before releasing D.
    held = {"move_down", "move_right", "move_left"}
    order = {
        "move_up": -1,
        "move_down": 1,
        "move_right": 2,
        "move_left": 3,
    }

    action = _action_from_held_keys(held, order)
    assert action == "move_down_left"


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


def _bfs_path_len(game_map: GameMap, start, goal) -> int:
    """Reference shortest 8-way path length via plain BFS -- the ground truth A*
    must match. Kept local to the test so it can't drift from the production
    heuristic search it's checking."""
    from collections import deque

    if start == goal:
        return 0
    seen = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            return seen[cur]
        for nxt in game_map.neighbors_8(*cur):
            if nxt not in seen and game_map.is_walkable(*nxt):
                seen[nxt] = seen[cur] + 1
                q.append(nxt)
    return seen.get(goal, -1)


def test_find_path_is_optimal_like_bfs_through_a_maze() -> None:
    # A* must return a path of the *same length* BFS would; the heuristic and the
    # tie-break only change which equal-length route is chosen, never its cost.
    import random

    rng = random.Random(1234)
    game_map = GameMap(30, 20)
    for _ in range(120):  # scatter interior walls to force real routing
        x, y = rng.randint(1, 28), rng.randint(1, 18)
        if (x, y) not in ((1, 1), (28, 18)):
            game_map.tiles[y][x] = game_map.WALL

    start, goal = (1, 1), (28, 18)
    path = game_map.find_path(start, goal)
    expected = _bfs_path_len(game_map, start, goal)

    if expected == -1:
        assert path == []
    else:
        assert path and path[-1] == goal
        assert start not in path
        assert len(path) == expected  # optimal, matches BFS


def test_find_path_takes_diagonals_at_unit_cost() -> None:
    # Open ground: (1,1)->(5,5) is four diagonal steps, so a shortest path is
    # length 4 (Chebyshev), not the 8 a 4-directional search would need.
    game_map = GameMap(9, 9)
    path = game_map.find_path((1, 1), (5, 5))
    assert len(path) == 4
    assert path[-1] == (5, 5)


def test_find_path_returns_empty_when_goal_is_walled_off() -> None:
    game_map = GameMap(9, 7)
    for y in range(1, 6):  # wall off the right half completely
        game_map.tiles[y][5] = game_map.WALL
    assert game_map.find_path((2, 3), (7, 3)) == []


def test_large_default_map_contains_buildings() -> None:
    game_map = GameMap(40, 20)

    assert game_map.tile_at(4, 3) == game_map.WALL
    assert game_map.tile_at(12, 8) == game_map.WALL
    assert game_map.tile_at(8, 8) == game_map.DOOR  # carved doorway

    assert game_map.tile_at(26, 5) == game_map.WALL
    assert game_map.tile_at(35, 11) == game_map.WALL
    assert game_map.tile_at(30, 11) == game_map.DOOR


def test_setup_world_rat_flood_spawns_rat_on_every_walkable_tile() -> None:
    game_map = GameMap(12, 8)
    player_pos = Position(6, 4)

    rats_spawned = _setup_world(game_map, player_pos, rat_flood=True)

    expected_rat_positions = {
        (x, y)
        for y in range(game_map.height)
        for x in range(game_map.width)
        if game_map.is_walkable(x, y) and (x, y) != (player_pos.x, player_pos.y)
    }

    rat_positions = {
        (pos.x, pos.y)
        for _ent, (pos, _npc, _enemy) in esper.get_components(Position, NPC, Enemy)
        if (pos.x, pos.y) != (player_pos.x, player_pos.y)
    }

    assert rats_spawned == len(expected_rat_positions)
    assert rat_positions == expected_rat_positions
