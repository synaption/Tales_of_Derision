"""Unrendered tests for houses: detection, furnishing, residency/claiming,
NPC building, and player crafting/placement."""
from __future__ import annotations

import esper
import pytest

from components import (
    Bed,
    BlocksMovement,
    BuildPlan,
    Chest,
    Furniture,
    Home,
    Inventory,
    NPC,
    Name,
    Needs,
    Player,
    Position,
    Renderable,
    Resident,
    Stove,
)
from game_map import GameMap
from items import WOOD, WOOD_DOOR, WOOD_WALL, WOOD_WINDOW, craft_cost, is_placeable, placed_tile
from main import _craft_item, _place_buildable_at
from systems import (
    HousingProcessor,
    NpcAiProcessor,
    bed_owner,
    furnish_house,
    house_is_owned,
    houses_for,
    owned_bed_of,
    set_bed_owner,
    set_house_ownership,
)

pytestmark = pytest.mark.unrendered


def _map_with_cabin(ox: int = 4, oy: int = 3, w: int = 6, h: int = 5, door=(2, 4)) -> GameMap:
    """A blank map with a single walled cabin (top-left at (ox, oy)) and a door."""
    game_map = GameMap(30, 18)
    # Clear any procedurally carved buildings so only our cabin is enclosed.
    for y in range(1, game_map.height - 1):
        for x in range(1, game_map.width - 1):
            game_map.tiles[y][x] = game_map.FLOOR
    for dy in range(h):
        for dx in range(w):
            x, y = ox + dx, oy + dy
            if dx in (0, w - 1) or dy in (0, h - 1):
                game_map.tiles[y][x] = game_map.WALL
    dx, dy = door
    game_map.tiles[oy + dy][ox + dx] = game_map.DOOR
    return game_map


# --- Tile rules ------------------------------------------------------------


def test_door_is_passable_but_window_blocks_movement() -> None:
    game_map = GameMap(24, 14)
    game_map.tiles[6][10] = game_map.DOOR
    game_map.tiles[6][11] = game_map.WINDOW

    assert game_map.is_walkable(10, 6)  # door: walk through
    assert game_map.is_passable(10, 6)
    assert not game_map.is_walkable(11, 6)  # window blocks movement
    assert not game_map.is_passable(11, 6)
    # Both stay transparent -- only walls block sight.
    assert game_map.has_line_of_sight((9, 6), (12, 6))


def test_set_tile_bumps_revision_and_guards_the_border() -> None:
    game_map = GameMap(24, 14)
    start = game_map.revision

    assert game_map.set_tile(5, 5, game_map.WALL) is True
    assert game_map.revision == start + 1
    assert game_map.set_tile(5, 5, game_map.WALL) is False  # no-op, same tile
    assert game_map.revision == start + 1
    assert game_map.set_tile(0, 0, game_map.FLOOR) is False  # border protected


# --- House detection -------------------------------------------------------


def test_find_enclosed_rooms_detects_a_walled_cabin_with_a_door() -> None:
    game_map = _map_with_cabin()
    rooms = game_map.find_enclosed_rooms()
    assert len(rooms) == 1
    interior = rooms[0]
    assert (5, 4) in interior  # an inside floor tile
    assert (4, 3) not in interior  # the wall corner is not interior


def test_a_room_with_no_door_is_not_a_house() -> None:
    game_map = _map_with_cabin(ox=4, oy=3, door=(2, 4))
    # Seal the doorway (abs (6, 7)) with a wall -> fully closed box, no house.
    game_map.tiles[7][6] = game_map.WALL
    assert game_map.find_enclosed_rooms() == []


# --- Furnishing ------------------------------------------------------------


def test_furnish_house_keeps_the_bed_reachable_from_the_door() -> None:
    # Blocking furniture must never seal the bed off, or a villager could never
    # walk in to sleep in it.
    game_map = _map_with_cabin()
    interior = game_map.find_enclosed_rooms()[0]
    bed_xy = furnish_house(game_map, interior)
    assert bed_xy is not None

    blocked = {(p.x, p.y) for _e, (p, _b) in esper.get_components(Position, BlocksMovement)}
    door_adjacent = [
        t for t in interior
        if any(game_map.tile_at(nx, ny) == game_map.DOOR for nx, ny in game_map.neighbors_4(t[0], t[1]))
    ]
    start = door_adjacent[0]
    reachable = start == bed_xy or bool(game_map.find_path(start, bed_xy, blocked_tiles=blocked))
    assert reachable, "the bed must be reachable from the doorway past the furniture"


def test_furnish_house_places_all_furniture_and_returns_the_bed() -> None:
    game_map = _map_with_cabin()
    interior = game_map.find_enclosed_rooms()[0]

    bed_xy = furnish_house(game_map, interior)
    assert bed_xy in interior

    beds = list(esper.get_components(Bed))
    stoves = list(esper.get_components(Stove))
    chests = list(esper.get_components(Chest))
    furniture_kinds = {esper.component_for_entity(e, Furniture).kind for e, _ in esper.get_components(Furniture)}
    assert len(beds) == 1
    assert len(stoves) == 1  # the "oven" is the Stove cooking station
    assert len(chests) == 1
    assert furniture_kinds == {"table", "wardrobe", "bookshelf"}


# --- Residency / claiming --------------------------------------------------


def test_resident_claims_an_unowned_furnished_house() -> None:
    game_map = _map_with_cabin()
    interior = game_map.find_enclosed_rooms()[0]
    bed_xy = furnish_house(game_map, interior)

    villager = esper.create_entity(
        Position(15, 9), NPC(), Resident(), BlocksMovement(), Name("Villager")
    )
    HousingProcessor(game_map).process("wait")

    assert esper.has_component(villager, Home)
    home = esper.component_for_entity(villager, Home)
    assert (home.x, home.y) == bed_xy
    assert house_is_owned(interior)


def test_bed_ownership_helpers() -> None:
    person = esper.create_entity(Name("Person"))
    bed = esper.create_entity(Position(3, 3), Bed())

    assert bed_owner(bed) is None
    assert owned_bed_of(person) is None

    set_bed_owner(bed, person)
    assert bed_owner(bed) == person
    assert owned_bed_of(person) == bed

    # A house whose owner no longer exists counts as unowned again.
    esper.delete_entity(person, immediate=True)
    assert bed_owner(bed) is None


def test_set_house_ownership_marks_bed_and_chest_only_inside() -> None:
    owner = esper.create_entity(Name("Owner"))
    bed = esper.create_entity(Position(3, 3), Bed())
    chest = esper.create_entity(Position(4, 3), Chest(), Inventory(items=[]))
    outside_bed = esper.create_entity(Position(9, 9), Bed())

    set_house_ownership(frozenset({(3, 3), (4, 3)}), owner)

    assert bed_owner(bed) == owner
    assert bed_owner(chest) == owner  # containers belong to the house owner too
    assert bed_owner(outside_bed) is None  # a bed outside the house is untouched


def test_claiming_a_house_makes_its_bed_and_chest_belong_to_the_resident() -> None:
    game_map = _map_with_cabin()
    interior = game_map.find_enclosed_rooms()[0]
    furnish_house(game_map, interior)  # places a bed and a chest inside
    villager = esper.create_entity(Position(15, 9), NPC(), Resident(), BlocksMovement(), Name("V"))

    HousingProcessor(game_map).process("wait")

    bed = next(e for e, (p, _b) in esper.get_components(Position, Bed) if (p.x, p.y) in interior)
    chest = next(e for e, (p, _c) in esper.get_components(Position, Chest) if (p.x, p.y) in interior)
    assert bed_owner(bed) == villager
    assert bed_owner(chest) == villager


def test_resident_that_already_owns_a_house_is_left_alone() -> None:
    game_map = _map_with_cabin()
    interior = game_map.find_enclosed_rooms()[0]
    furnish_house(game_map, interior)  # an unowned, furnished house is available

    villager = esper.create_entity(Position(3, 2), NPC(), Resident(), BlocksMovement(), Name("V"))
    own_bed = esper.create_entity(Position(2, 2), Bed())
    set_bed_owner(own_bed, villager)  # it already owns a home

    HousingProcessor(game_map).process("wait")

    # It neither grabs the free house nor starts building a second one.
    assert not esper.has_component(villager, BuildPlan)
    assert not house_is_owned(interior)


def test_villager_will_not_claim_a_house_someone_else_owns() -> None:
    game_map = _map_with_cabin()
    interior = game_map.find_enclosed_rooms()[0]
    furnish_house(game_map, interior)
    bed_ent = next(e for e, (p, _b) in esper.get_components(Position, Bed) if (p.x, p.y) in interior)
    owner = esper.create_entity(Player(), Name("Owner"))
    set_bed_owner(bed_ent, owner)  # e.g. the player's house

    villager = esper.create_entity(Position(15, 9), NPC(), Resident(), BlocksMovement(), Name("V"))
    HousingProcessor(game_map).process("wait")

    assert bed_owner(bed_ent) == owner  # ownership unchanged
    assert not esper.has_component(villager, Home)  # it didn't move in
    assert esper.has_component(villager, BuildPlan)  # it builds its own instead


def test_two_residents_claim_two_different_houses() -> None:
    game_map = GameMap(30, 18)
    for y in range(1, game_map.height - 1):
        for x in range(1, game_map.width - 1):
            game_map.tiles[y][x] = game_map.FLOOR
    # Two separate cabins.
    for ox in (3, 15):
        for dy in range(5):
            for dx in range(6):
                x, y = ox + dx, 3 + dy
                if dx in (0, 5) or dy in (0, 4):
                    game_map.tiles[y][x] = game_map.WALL
        game_map.tiles[3 + 4][ox + 2] = game_map.DOOR
    for interior in game_map.find_enclosed_rooms():
        furnish_house(game_map, interior)

    v1 = esper.create_entity(Position(2, 12), NPC(), Resident(), BlocksMovement(), Name("A"))
    v2 = esper.create_entity(Position(25, 12), NPC(), Resident(), BlocksMovement(), Name("B"))
    HousingProcessor(game_map).process("wait")

    home1 = esper.component_for_entity(v1, Home)
    home2 = esper.component_for_entity(v2, Home)
    assert (home1.x, home1.y) != (home2.x, home2.y)  # they took different beds


def test_homeless_resident_with_no_house_gets_a_build_plan() -> None:
    game_map = GameMap(30, 18)  # blank, no carved buildings survive detection anyway
    for y in range(1, game_map.height - 1):
        for x in range(1, game_map.width - 1):
            game_map.tiles[y][x] = game_map.FLOOR
    villager = esper.create_entity(
        Position(15, 9), NPC(), Resident(), BlocksMovement(), Name("Builder")
    )

    HousingProcessor(game_map).process("wait")
    assert esper.has_component(villager, BuildPlan)
    plan = esper.component_for_entity(villager, BuildPlan)
    assert plan.remaining  # has pieces to build
    assert plan.interior  # and an interior to furnish later


def test_npc_builds_a_piece_when_it_has_wood_and_is_adjacent() -> None:
    game_map = GameMap(30, 18)
    for y in range(1, game_map.height - 1):
        for x in range(1, game_map.width - 1):
            game_map.tiles[y][x] = game_map.FLOOR
    processor = NpcAiProcessor(game_map)

    # A builder standing next to a single pending wall piece, holding wood.
    villager = esper.create_entity(
        Position(10, 10), NPC(), Needs(hunger=0.0, thirst=0.0, tiredness=0.0),
        Inventory(items=[WOOD]), BlocksMovement(), Name("Builder"),
        BuildPlan(remaining=[(11, 10, game_map.WALL)], interior=[(12, 10)], bed=(12, 10)),
    )

    processor.process("wait")
    assert game_map.tile_at(11, 10) == game_map.WALL  # placed the wall
    assert esper.component_for_entity(villager, Inventory).items == []  # spent the wood
    assert not esper.has_component(villager, BuildPlan)  # plan complete -> furnished
    assert esper.has_component(villager, Home)  # and moved in


def test_npc_builds_a_whole_cabin_from_a_blueprint() -> None:
    from systems import _blueprint_tiles

    game_map = GameMap(30, 20)
    for y in range(1, game_map.height - 1):
        for x in range(1, game_map.width - 1):
            game_map.tiles[y][x] = game_map.FLOOR
    processor = NpcAiProcessor(game_map)

    build, interior = _blueprint_tiles(game_map, 12, 8)
    # A builder with ample wood and no competing needs (all rates zeroed).
    builder = esper.create_entity(
        Position(6, 10), NPC(), Resident(),
        Needs(hunger=0.0, thirst=0.0, tiredness=0.0, hunger_rate=0.0, thirst_rate=0.0, tiredness_rate=0.0),
        Inventory(items=[WOOD] * 40), BlocksMovement(), Name("Builder"),
        BuildPlan(remaining=list(build), interior=list(interior), bed=interior[0]),
    )

    for _ in range(400):
        processor.process("wait")
        if not esper.has_component(builder, BuildPlan):
            break

    # The cabin got built (an enclosed house now exists), the builder moved in,
    # and it did not self-trap: it never got stuck holding an unfinishable plan.
    assert not esper.has_component(builder, BuildPlan)
    assert esper.has_component(builder, Home)
    assert len(game_map.find_enclosed_rooms()) == 1


# --- Player crafting / placement -------------------------------------------


def test_craft_item_consumes_wood_and_yields_a_placeable() -> None:
    player = esper.create_entity(Player(), Inventory(items=[WOOD, WOOD, WOOD]))
    message = _craft_item(player, WOOD_WALL)  # costs 2 wood

    items = esper.component_for_entity(player, Inventory).items
    assert items.count(WOOD) == 3 - craft_cost(WOOD_WALL)
    assert WOOD_WALL in items
    assert "craft" in message.lower()


def test_craft_item_refuses_without_enough_wood() -> None:
    player = esper.create_entity(Player(), Inventory(items=[WOOD]))
    message = _craft_item(player, WOOD_DOOR)  # costs 3 wood
    assert WOOD_DOOR not in esper.component_for_entity(player, Inventory).items
    assert "need" in message.lower()


def test_place_buildable_sets_the_tile_and_consumes_the_item() -> None:
    game_map = GameMap(24, 14)
    player = esper.create_entity(Position(5, 5), Player(), Inventory(items=[WOOD_WALL]))

    message = _place_buildable_at(player, game_map, WOOD_WALL, (6, 5))
    assert game_map.tile_at(6, 5) == game_map.WALL
    assert esper.component_for_entity(player, Inventory).items == []
    assert "build" in message.lower()


def test_place_buildable_refuses_an_occupied_tile() -> None:
    game_map = GameMap(24, 14)
    player = esper.create_entity(Position(5, 5), Player(), Inventory(items=[WOOD_WALL]))
    esper.create_entity(Position(6, 5), Name("Rock"), BlocksMovement())

    message = _place_buildable_at(player, game_map, WOOD_WALL, (6, 5))
    assert game_map.tile_at(6, 5) == game_map.FLOOR  # unchanged
    assert WOOD_WALL in esper.component_for_entity(player, Inventory).items
    assert "in the way" in message.lower()


def test_placeable_item_helpers() -> None:
    assert is_placeable(WOOD_WALL) and is_placeable(WOOD_DOOR) and is_placeable(WOOD_WINDOW)
    assert not is_placeable(WOOD)
    assert placed_tile(WOOD_DOOR) == GameMap.DOOR
    assert placed_tile(WOOD_WINDOW) == GameMap.WINDOW
    assert placed_tile("Bread") is None


def test_same_region_splits_across_water() -> None:
    game_map = GameMap(20, 12)  # small: no procedural water/buildings, one open room
    assert game_map.same_region((3, 3), (16, 8))

    for y in range(1, 11):  # carve a river straight down the middle
        game_map.set_tile(10, y, game_map.WATER)

    assert game_map.region_of(3, 6) is not None
    assert game_map.region_of(3, 6) != game_map.region_of(16, 6)
    assert not game_map.same_region((3, 6), (16, 6))  # opposite banks
    assert game_map.same_region((3, 6), (4, 7))  # same bank


def test_villager_will_not_claim_a_house_across_water() -> None:
    # A furnished house whose interior sits on the far bank of a river the
    # villager cannot cross must NOT be claimed (it would strand the villager).
    game_map = GameMap(20, 12)
    for y in range(1, 11):
        game_map.tiles[y][10] = game_map.WALL  # solid wall splits the map in two
    # A one-tile "house" on the right bank with a bed.
    esper.create_entity(Position(15, 6), Renderable("="), Name("Bed"), Bed())
    interior = frozenset({(15, 6)})

    villager = esper.create_entity(Position(4, 6), NPC(), Resident(), BlocksMovement(), Name("V"))

    # Patch detection to return our hand-made interior, then run housing.
    proc = HousingProcessor(game_map)
    proc._cache = {"revision": game_map.revision, "houses": [interior]}
    proc.process("wait")

    # It couldn't reach the far-bank house, so it starts building instead.
    assert not esper.has_component(villager, Home)
    assert esper.has_component(villager, BuildPlan)


def test_houses_for_caches_until_the_map_changes() -> None:
    game_map = _map_with_cabin()
    cache: dict = {}
    first = houses_for(game_map, cache)
    assert houses_for(game_map, cache) is first  # cached, same object

    game_map.set_tile(20, 10, game_map.WALL)  # map changed -> recompute
    assert houses_for(game_map, cache) is not first
