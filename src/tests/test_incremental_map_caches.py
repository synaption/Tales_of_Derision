"""A tile edit is patched into the map's caches, never re-flooded wholesale.

Raising a wall used to invalidate everything the island knew about itself: its
connected-region labels (a 7200-tile flood) and its enclosed rooms (another one).
Villagers edit tiles constantly, so that was one of the two dominant per-turn
costs in the game -- roughly 16 ms per wall laid.

The map now patches those caches around the edited tile and only re-floods when
the edit could genuinely have changed which tiles reach which. That is a
correctness claim, so these tests check it the only way worth trusting: make
edits, then compare the incrementally-maintained answer against one computed
from scratch. If the two ever disagree, the optimization is wrong.
"""
from __future__ import annotations

import random

import pytest

from game_map import GameMap

pytestmark = pytest.mark.unrendered


def _fresh_labels(game_map: GameMap) -> dict[tuple[int, int], int | None]:
    """Every tile's region id, computed with all caches thrown away."""
    game_map._island_region_cache.clear()
    game_map._island_region_rev.clear()
    game_map._flood_mask_cache.clear()
    return {
        (x, y): game_map.region_of(x, y)
        for y in range(game_map.height)
        for x in range(game_map.width)
    }


def _live_labels(game_map: GameMap) -> dict[tuple[int, int], int | None]:
    """Every tile's region id as the incrementally-patched caches report it."""
    return {
        (x, y): game_map.region_of(x, y)
        for y in range(game_map.height)
        for x in range(game_map.width)
    }


def _same_partition(
    live: dict[tuple[int, int], int | None], fresh: dict[tuple[int, int], int | None]
) -> bool:
    """Whether two labellings describe the same regions.

    The *numbers* are arbitrary -- a patched cache keeps its old ids while a
    rebuild renumbers from one -- so what has to match is the partition: the same
    tiles walkable, and two tiles share a region in one iff they do in the other.
    """
    if {xy for xy, r in live.items() if r is None} != {
        xy for xy, r in fresh.items() if r is None
    }:
        return False
    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for xy, label in live.items():
        other = fresh[xy]
        if label is None:
            continue
        if mapping.setdefault(label, other) != other:
            return False
        if reverse.setdefault(other, label) != label:
            return False
    return True


def _room_world() -> GameMap:
    """A plain walled room -- small enough to compare every tile, and the layout
    the incremental paths have to get right for test maps too."""
    return GameMap(40, 24, land_rect=None)


def test_a_wall_in_the_open_does_not_relabel_the_island() -> None:
    """The whole point: the common edit is proven harmless locally, so the
    cached labels survive it."""
    game_map = _room_world()
    game_map.region_of(5, 5)  # warm the cache
    before = game_map._island_region_cache[0]

    assert game_map.set_tile(10, 10, game_map.WALL)

    assert game_map._island_region_cache.get(0) is before, (
        "a wall with room to walk around it cannot split anything"
    )
    assert game_map.region_of(10, 10) is None, "the wall itself is not walkable"
    assert _same_partition(_live_labels(game_map), _fresh_labels(game_map))


def test_sealing_a_room_off_really_does_split_it() -> None:
    """The conservative half: when an edit does cut the map in two, the labels
    must say so -- a villager on one side must not think it can walk to the
    other."""
    game_map = _room_world()
    for y in range(1, 23):
        game_map.set_tile(20, y, game_map.WALL)

    assert not game_map.same_region((10, 12), (30, 12)), (
        "a wall across the room separates the two halves"
    )
    assert _same_partition(_live_labels(game_map), _fresh_labels(game_map))


def test_cutting_a_door_reconnects_the_halves() -> None:
    """And the reverse: opening a tile merges the components it joins."""
    game_map = _room_world()
    for y in range(1, 23):
        game_map.set_tile(20, y, game_map.WALL)
    assert not game_map.same_region((10, 12), (30, 12))

    game_map.set_tile(20, 12, game_map.DOOR)

    assert game_map.same_region((10, 12), (30, 12)), "the door is a way through"
    assert _same_partition(_live_labels(game_map), _fresh_labels(game_map))


def test_a_walkability_neutral_edit_keeps_every_cached_field() -> None:
    """Swapping a floor for a door changes nothing about what reaches what, so
    connectivity-keyed caches (flow fields, paths) must not be invalidated."""
    game_map = _room_world()
    game_map.region_of(5, 5)
    before_rev = game_map.connectivity_revision(5, 5)

    game_map.set_tile(10, 10, game_map.DOOR)  # was floor: both are walkable

    assert game_map.connectivity_revision(5, 5) == before_rev
    assert _same_partition(_live_labels(game_map), _fresh_labels(game_map))


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_edits_agree_with_a_from_scratch_relabel(seed: int) -> None:
    """The real proof. Hundreds of random walls, doors and demolitions, then a
    tile-by-tile comparison against a map that cached nothing. Any divergence
    means an NPC could believe in a route that isn't there (or miss one that is).
    """
    rng = random.Random(seed)
    game_map = _room_world()
    game_map.region_of(5, 5)  # warm the caches so the patched paths are exercised

    for _ in range(300):
        x = rng.randrange(1, game_map.width - 1)
        y = rng.randrange(1, game_map.height - 1)
        tile = rng.choice(
            [game_map.WALL, game_map.WALL, game_map.WATER, game_map.WINDOW,
             game_map.FLOOR, game_map.DOOR]
        )
        game_map.set_tile(x, y, tile)
        # Ask about a tile every edit, so the cache is repeatedly rebuilt and
        # re-patched rather than only consulted at the end.
        game_map.region_of(x, y)

    live = _live_labels(game_map)
    assert _same_partition(live, _fresh_labels(game_map))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_edits_keep_enclosed_rooms_correct(seed: int) -> None:
    """Same proof for house detection: the incrementally-updated room list must
    match what a full re-scan of the island finds."""
    rng = random.Random(seed)
    game_map = _room_world()
    game_map.enclosed_rooms()

    for _ in range(200):
        x = rng.randrange(1, game_map.width - 1)
        y = rng.randrange(1, game_map.height - 1)
        game_map.set_tile(x, y, rng.choice(
            [game_map.WALL, game_map.WALL, game_map.FLOOR, game_map.DOOR]
        ))
        game_map.enclosed_rooms()

    live = {frozenset(room) for room in game_map.enclosed_rooms()}
    fresh = {frozenset(room) for room in game_map.find_enclosed_rooms()}
    assert live == fresh


def test_a_bounded_flood_matches_the_full_one_inside_its_window() -> None:
    """A flow field flooded only as far as its user needs must give exactly the
    same distances there -- the bound is an optimization, not an approximation of
    the values it does report."""
    game_map = _blank_room()
    goal = (20, 12)

    full = game_map.distance_field(goal)
    bounded = game_map.distance_field(goal, max_radius=5)

    for y in range(game_map.height):
        for x in range(game_map.width):
            inside = max(abs(x - goal[0]), abs(y - goal[1])) <= 5
            if inside and full.at(x, y) is not None:
                assert bounded.at(x, y) == full.at(x, y), f"{(x, y)} differs"


def test_a_bounded_flood_stops_at_its_window() -> None:
    """And past the window there is simply nothing -- the caller falls back the
    same way it does for a goal it cannot reach at all."""
    game_map = _blank_room()

    bounded = game_map.distance_field((20, 12), max_radius=3)

    assert bounded.at(20, 12) == 0
    assert bounded.at(23, 12) == 3
    assert bounded.at(30, 12) is None, "well outside the window"
    assert (30, 12) not in bounded


def _blank_room() -> GameMap:
    """A walled room of bare floor -- no default building, no lakes -- so a test
    can watch exactly one house come into existence."""
    game_map = _room_world()
    for y in range(1, game_map.height - 1):
        for x in range(1, game_map.width - 1):
            game_map.tiles[y][x] = game_map.FLOOR
    return game_map


def test_building_a_cabin_is_detected_room_by_room() -> None:
    """The edit sequence that actually happens in play: a villager lays a cabin
    wall by wall, and the interior becomes a house the moment the door goes in."""
    game_map = _blank_room()
    assert game_map.enclosed_rooms() == []

    for x in range(5, 12):
        game_map.set_tile(x, 5, game_map.WALL)
        game_map.set_tile(x, 11, game_map.WALL)
    for y in range(5, 12):
        game_map.set_tile(5, y, game_map.WALL)
        game_map.set_tile(11, y, game_map.WALL)
    assert game_map.enclosed_rooms() == [], "sealed shut, it is not a house yet"

    game_map.set_tile(5, 8, game_map.DOOR)

    rooms = game_map.enclosed_rooms()
    assert len(rooms) == 1
    assert (8, 8) in rooms[0]
    assert {frozenset(r) for r in rooms} == {
        frozenset(r) for r in game_map.find_enclosed_rooms()
    }
