"""Goal maps: one multi-source flood answers "where is the nearest X" for everyone.

The old shape picked the nearest resource by straight-line distance and then flooded
the island to *that* tile, so a dozen creatures choosing a dozen different trees cost
a dozen island-wide BFS floods. A goal map seeds every tree at distance 0 in a single
flood; the value at any tile is its walking distance to the nearest one, and stepping
downhill walks there for the price of eight dict lookups.

These tests pin the three things that makes true -- one flood serves everyone, the
winner is nearest by *walking* rather than by straight line, and the flood is rebuilt
only when the thing it maps actually changes -- plus the watched/unwatched split that
decides where it's used at all.
"""
from __future__ import annotations

from collections.abc import Callable
import itertools

import ecs
import pytest

from components import (
    BlocksMovement, Deer, Diet, NPC, Needs, Player, Position, Tree, WorldClock,
)
from game_map import GameMap
from regions import region_at
from systems import NpcAiProcessor
import spatial

pytestmark = pytest.mark.unrendered


def _no_background_pump() -> Callable[[], float]:
    """A wall clock already past any deadline, so the processor's own background
    pump can never add region-turns behind the test's back."""
    counter = itertools.count()
    return lambda: next(counter) * 1000.0


def _open_world(width: int = 360, height: int = 60) -> GameMap:
    """Open floor across more than one region, with the player parked far to the
    west so the eastern end of the map is unwatched."""
    ecs.clear_database()
    spatial.detach()
    game_map = GameMap(width, height)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            game_map.tiles[y][x] = game_map.FLOOR
    ecs.create_entity(WorldClock(turn=0))
    ecs.create_entity(Position(2, 30), Player())
    return game_map


def _processor(game_map: GameMap) -> NpcAiProcessor:
    proc = NpcAiProcessor(game_map, wall_clock=_no_background_pump())
    proc._player_xy = (2, 30)
    return proc


def _deer(x: int, y: int, hunger: float = 90.0) -> tuple[int, Position]:
    pos = Position(x, y)
    ent = ecs.create_entity(
        pos, NPC(), Deer(), Diet("herbivore"), Needs(hunger=hunger, thirst=10.0)
    )
    return ent, pos


def _count_floods(game_map: GameMap) -> tuple[list[int], Callable[[], None]]:
    """Count every island flood, however many sources it was seeded from. Returns
    the counter list (one entry per flood, holding its seed count) and a hook that
    puts the real method back."""
    counted: list[int] = []
    original = game_map.distance_field_from

    def spy(sources):
        source_list = list(sources)
        counted.append(len(source_list))
        return original(source_list)

    game_map.distance_field_from = spy
    return counted, lambda: setattr(game_map, "distance_field_from", original)


# --- the flood itself -------------------------------------------------------


def test_a_multi_source_flood_measures_distance_to_the_nearest_source() -> None:
    game_map = _open_world(60, 30)

    field = game_map.distance_field_from([(10, 15), (50, 15)])

    assert field[(10, 15)] == 0 and field[(50, 15)] == 0
    assert field[(12, 15)] == 2, "two steps from the western source"
    assert field[(48, 15)] == 2, "and two from the eastern one, in the same field"
    assert field[(30, 15)] == 20, "the midpoint is equidistant from both"


def test_one_source_gives_exactly_the_old_single_source_field() -> None:
    """``distance_field`` now delegates here, so the two must not have drifted."""
    game_map = _open_world(60, 30)

    assert game_map.distance_field((20, 15)) == game_map.distance_field_from([(20, 15)])


def test_unreachable_sources_simply_are_not_in_the_field() -> None:
    """A wall-off is expressed by absence, which is what replaced the per-creature
    same-region filter: you cannot step toward what isn't in your field."""
    game_map = _open_world(60, 30)
    for y in range(0, 30):
        game_map.tiles[y][30] = game_map.WALL  # split the room in two

    field = game_map.distance_field_from([(50, 15)])

    assert (50, 15) in field
    assert (10, 15) not in field, "the far side of the wall can't reach the source"


# --- one flood, many creatures ----------------------------------------------


def test_a_whole_crowd_shares_one_flood() -> None:
    game_map = _open_world()
    for x in range(40, 60, 2):
        ecs.create_entity(Position(x, 40), Tree(wood=5), BlocksMovement())
    herd = [_deer(x, 20) for x in range(20, 32, 2)]
    proc = _processor(game_map)
    region = region_at(game_map, 20, 20)
    floods, restore = _count_floods(game_map)
    try:
        for ent, pos in herd:
            proc._graze(ent, pos, ecs.component_for_entity(ent, Needs), _trees(), {}, region)
    finally:
        restore()

    assert len(herd) == 6
    assert len(floods) == 1, "six grazers, one flood -- that is the whole point"


def test_the_flood_is_reused_across_turns() -> None:
    game_map = _open_world()
    for x in (40, 44, 48):
        ecs.create_entity(Position(x, 40), Tree(wood=99), BlocksMovement())
    ent, pos = _deer(20, 20)
    needs = ecs.component_for_entity(ent, Needs)
    proc = _processor(game_map)
    region = region_at(game_map, 20, 20)
    floods, restore = _count_floods(game_map)
    try:
        for _ in range(10):
            proc._graze(ent, pos, needs, _trees(), {}, region)
    finally:
        restore()

    assert len(floods) == 1, "ten turns of walking on one flood"


def _trees() -> list[tuple[tuple[int, int], int]]:
    return [
        ((p.x, p.y), ent) for ent, (p, _t) in ecs.get_components(Position, Tree)
    ]


# --- nearest by walking, not by straight line -------------------------------


def test_the_nearest_source_is_the_nearest_to_walk_to() -> None:
    """A tree just over a wall is close as the crow flies and miles away on foot.
    A goal map can't be fooled by that -- its numbers *are* walking distance."""
    game_map = _open_world(120, 40)
    # A pocket walled off except for a door at the far end, with a tree inside.
    for y in range(10, 21):
        game_map.tiles[y][30] = game_map.WALL
    for x in range(30, 46):
        game_map.tiles[10][x] = game_map.WALL
        game_map.tiles[20][x] = game_map.WALL
    ecs.create_entity(Position(32, 15), Tree(wood=5), BlocksMovement())  # 4 tiles away
    ecs.create_entity(Position(20, 30), Tree(wood=5), BlocksMovement())  # 15 tiles away
    ent, pos = _deer(28, 15)
    proc = _processor(game_map)

    proc._graze(ent, pos, ecs.component_for_entity(ent, Needs), _trees(), {})

    assert pos.x <= 28, "it set off for the open tree, not the one behind the wall"


def test_an_equal_distance_tie_prefers_the_straight_step() -> None:
    game_map = _open_world(60, 30)
    ecs.create_entity(Position(40, 15), Tree(wood=5), BlocksMovement())
    ent, pos = _deer(20, 15)
    proc = _processor(game_map)

    proc._graze(ent, pos, ecs.component_for_entity(ent, Needs), _trees(), {})

    assert (pos.x, pos.y) == (21, 15), "straight east, not an equal-cost diagonal"


# --- when the flood is rebuilt ----------------------------------------------


def test_a_passing_creature_does_not_cost_a_re_flood() -> None:
    """The per-kind index version is the whole reason goal maps are affordable:
    the map of where the trees are must not die because a deer crossed a seam."""
    game_map = _open_world()
    ecs.create_entity(Position(40, 40), Tree(wood=99), BlocksMovement())
    ent, pos = _deer(20, 20)
    needs = ecs.component_for_entity(ent, Needs)
    wanderer, wander_pos = _deer(118, 20)
    proc = _processor(game_map)
    region = region_at(game_map, 20, 20)
    proc._graze(ent, pos, needs, _trees(), {}, region)  # build the map

    floods, restore = _count_floods(game_map)
    try:
        for step in range(6):  # walk the other deer over the region seam at x=120
            spatial.moved(wanderer, (wander_pos.x, wander_pos.y), (wander_pos.x + 1, wander_pos.y))
            wander_pos.x += 1
            proc._graze(ent, pos, needs, _trees(), {}, region)
    finally:
        restore()

    assert not floods, "nothing about the trees changed"


def test_felling_a_tree_does_rebuild_the_map() -> None:
    game_map = _open_world()
    doomed = ecs.create_entity(Position(40, 40), Tree(wood=99), BlocksMovement())
    for x in (44, 48):
        ecs.create_entity(Position(x, 40), Tree(wood=99), BlocksMovement())
    ent, pos = _deer(20, 20)
    needs = ecs.component_for_entity(ent, Needs)
    proc = _processor(game_map)
    region = region_at(game_map, 20, 20)
    proc._graze(ent, pos, needs, _trees(), {}, region)

    floods, restore = _count_floods(game_map)
    try:
        ecs.delete_entity(doomed, immediate=True)
        proc._graze(ent, pos, needs, _trees(), {}, region)
    finally:
        restore()

    assert len(floods) == 1, "the map of where the trees are is out of date"


# --- the watched / unwatched split ------------------------------------------


def test_an_unwatched_creature_does_not_build_a_goal_map() -> None:
    """Out in the lagging world a flood is bought over and over for a handful of
    uses, so those creatures keep the cheap scan-and-walk instead."""
    game_map = _open_world()
    ecs.create_entity(Position(300, 40), Tree(wood=99), BlocksMovement())
    ent, pos = _deer(280, 20)
    proc = _processor(game_map)
    assert proc._compactable((280, 20)), "the test creature must be out of sight"

    floods, restore = _count_floods(game_map)
    try:
        proc._graze(ent, pos, ecs.component_for_entity(ent, Needs), _trees(), {})
    finally:
        restore()

    assert not floods
    assert (pos.x, pos.y) != (280, 20), "but it still walked toward the tree"


def test_both_paths_agree_on_which_tree_a_creature_picks() -> None:
    """Watched and unwatched are two implementations of one behaviour, so a
    creature must not change its mind about which tree to eat just because the
    player walked away from it.

    They differ by one tile at the end and that is fine: a watched creature stops
    beside the tree, while a compacted walk can finish standing on its tile (tree
    tiles are walkable; it is the entity that blocks). Both then eat the same tree.
    """
    trees = [(40, 40), (60, 20), (26, 24)]
    nearest = (26, 24)

    def run(player_at: tuple[int, int]) -> tuple[int, int]:
        game_map = _open_world()
        for _e, (p, _pl) in ecs.get_components(Position, Player):
            p.x, p.y = player_at
        for tx, ty in trees:
            ecs.create_entity(Position(tx, ty), Tree(wood=99), BlocksMovement())
        ent, pos = _deer(20, 20)
        proc = NpcAiProcessor(game_map, wall_clock=_no_background_pump())
        proc._player_xy = player_at
        for _ in range(6):
            proc._graze(ent, pos, ecs.component_for_entity(ent, Needs), _trees(), {})
        return (pos.x, pos.y)

    watched = run((20, 20))
    unwatched = run((350, 55))
    assert max(abs(watched[0] - nearest[0]), abs(watched[1] - nearest[1])) <= 1
    assert max(abs(unwatched[0] - nearest[0]), abs(unwatched[1] - nearest[1])) <= 1
