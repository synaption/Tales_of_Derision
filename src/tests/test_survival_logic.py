"""Unrendered tests for the survival loop: needs, chopping, drinking, cooking,
and eating. These exercise pure ECS/data logic with no live renderer."""
from __future__ import annotations

import esper
import pytest

from components import (
    BlocksMovement,
    Corpse,
    Enemy,
    Inventory,
    Meat,
    Name,
    NPC,
    Needs,
    Player,
    Position,
    Renderable,
    Stove,
    Tree,
    Vision,
    Well,
)
from game_map import GameMap
from items import (
    COOKED_MEAT,
    RAW_MEAT,
    WOOD,
    cook_meat,
    hunger_restored,
    is_cooked_meat,
    is_raw_meat,
)
from main import (
    _apply_consumable,
    _chop_tree,
    _cook_at_stove,
    _drink_from_well,
    _find_adjacent_feature,
)
from components import Deer, Diet
from systems import MovementProcessor, NeedsProcessor, NpcAiProcessor, _pull_turn_events

pytestmark = pytest.mark.unrendered


def _map_with_water(water_x: int, water_y: int) -> GameMap:
    game_map = GameMap(24, 14)
    game_map.tiles[water_y][water_x] = GameMap.WATER
    return game_map


def _make_player(x: int = 5, y: int = 5, **components: object) -> int:
    return esper.create_entity(Position(x, y), Player(), *components.values())


def test_needs_processor_ticks_only_on_movement_turns() -> None:
    player = esper.create_entity(Needs(hunger=0.0, thirst=0.0))
    processor = NeedsProcessor()

    processor.process("move_up")
    needs = esper.component_for_entity(player, Needs)
    assert needs.hunger == pytest.approx(1.0)
    assert needs.thirst == pytest.approx(1.4)

    # A menu refresh (action=None) or non-move action must not advance needs.
    processor.process(None)
    processor.process("open_inventory")
    assert needs.hunger == pytest.approx(1.0)
    assert needs.thirst == pytest.approx(1.4)


def test_needs_are_clamped_to_max() -> None:
    player = esper.create_entity(Needs(hunger=99.5, thirst=99.9, max_value=100.0))
    NeedsProcessor().process("move_down")
    needs = esper.component_for_entity(player, Needs)
    assert needs.hunger == pytest.approx(100.0)
    assert needs.thirst == pytest.approx(100.0)


def test_chop_tree_yields_wood_and_falls_when_exhausted() -> None:
    player = esper.create_entity(Position(5, 5), Player(), Inventory(items=[]))
    tree = esper.create_entity(Position(6, 5), Tree(wood=2))

    first = _chop_tree(tree, player)
    inventory = esper.component_for_entity(player, Inventory)
    assert inventory.items == [WOOD]
    assert esper.entity_exists(tree)
    assert "wood" in first.lower()

    second = _chop_tree(tree, player)
    assert inventory.items == [WOOD, WOOD]
    assert not esper.entity_exists(tree)
    assert "fell" in second.lower()


def test_drink_from_well_quenches_thirst() -> None:
    player = esper.create_entity(Position(5, 5), Player(), Needs(thirst=60.0))
    well = esper.create_entity(Position(5, 4), Well())

    message = _drink_from_well(well, player)
    assert esper.component_for_entity(player, Needs).thirst == 0.0
    assert "quenched" in message.lower()


def test_cook_at_stove_requires_wood_and_meat() -> None:
    player = esper.create_entity(Position(5, 5), Player(), Inventory(items=[RAW_MEAT]))
    stove = esper.create_entity(Position(5, 6), Stove())

    # Missing wood: nothing consumed.
    message = _cook_at_stove(stove, player)
    inventory = esper.component_for_entity(player, Inventory)
    assert inventory.items == [RAW_MEAT]
    assert "wood" in message.lower()

    inventory.items.append(WOOD)
    message = _cook_at_stove(stove, player)
    assert inventory.items == [COOKED_MEAT]
    assert "cook" in message.lower()


def test_apply_consumable_eats_food_and_returns_none_for_gear() -> None:
    player = esper.create_entity(
        Position(5, 5),
        Player(),
        Inventory(items=["Rusty Sword", COOKED_MEAT]),
        Needs(hunger=50.0),
    )

    # Index 0 is a weapon -> not consumable.
    assert _apply_consumable(player, 0) is None

    message = _apply_consumable(player, 1)
    inventory = esper.component_for_entity(player, Inventory)
    needs = esper.component_for_entity(player, Needs)
    assert COOKED_MEAT not in inventory.items
    assert needs.hunger < 50.0
    assert "eat" in message.lower()


def test_apply_consumable_drinks_and_clamps_thirst_at_zero() -> None:
    player = esper.create_entity(
        Position(5, 5),
        Player(),
        Inventory(items=["Waterskin"]),
        Needs(thirst=10.0),
    )

    message = _apply_consumable(player, 0)
    needs = esper.component_for_entity(player, Needs)
    assert esper.component_for_entity(player, Inventory).items == []
    assert needs.thirst == 0.0
    assert "drink" in message.lower()


def test_find_adjacent_feature_targets_faced_tile() -> None:
    esper.create_entity(Position(5, 5), Player())
    well = esper.create_entity(Position(5, 4), Well())

    assert _find_adjacent_feature("move_up", Well) == well
    assert _find_adjacent_feature("move_down", Well) is None


def test_meat_helpers_recognise_named_meat() -> None:
    assert is_raw_meat("Rat Meat")
    assert is_raw_meat(RAW_MEAT)
    assert not is_raw_meat("Cooked Rat Meat")
    assert is_cooked_meat("Cooked Goblin Meat")

    assert cook_meat("Rat Meat") == "Cooked Rat Meat"
    assert cook_meat(RAW_MEAT) == COOKED_MEAT

    # Raw meat barely feeds you; cooking it makes it far more filling.
    assert hunger_restored("Rat Meat") < hunger_restored("Cooked Rat Meat")


def test_cook_named_meat_keeps_the_creature_name() -> None:
    player = esper.create_entity(
        Position(5, 5), Player(), Inventory(items=["Goblin Meat", WOOD])
    )
    stove = esper.create_entity(Position(5, 6), Stove())

    message = _cook_at_stove(stove, player)
    assert esper.component_for_entity(player, Inventory).items == ["Cooked Goblin Meat"]
    assert "goblin meat" in message.lower()


def test_corpse_yields_creature_specific_meat() -> None:
    game_map = GameMap(12, 8)
    esper.add_processor(MovementProcessor(game_map), priority=1)

    esper.create_entity(
        Position(6, 4),
        Renderable("r"),
        Name("Cave Rat"),
        NPC(),
        Enemy(),
        BlocksMovement(),
        Vision(6),
        Meat("Rat Meat"),
    )
    esper.create_entity(
        Position(5, 4), Renderable("@"), Name("You"), Player(), BlocksMovement()
    )

    esper.process("move_right")

    corpse_ent = next(ent for ent, (_corpse,) in esper.get_components(Corpse))
    corpse_inventory = esper.component_for_entity(corpse_ent, Inventory)
    assert corpse_inventory.items == ["Rat Meat"]


def test_needs_processor_warns_only_for_the_player() -> None:
    esper.create_entity(Player(), Needs(hunger=79.0))  # will cross 80% this turn
    esper.create_entity(NPC(), Needs(hunger=79.0))  # NPC also crosses, but silent

    _pull_turn_events()  # clear any residue
    NeedsProcessor().process("move_up")
    events = _pull_turn_events()

    # Exactly one warning (the player's), not two.
    assert sum("hunger" in text.lower() for text in events) == 1


def test_npc_needs_still_tick() -> None:
    npc = esper.create_entity(NPC(), Needs(hunger=0.0, thirst=0.0))
    NeedsProcessor().process("move_up")
    needs = esper.component_for_entity(npc, Needs)
    assert needs.hunger > 0.0
    assert needs.thirst > 0.0


def test_water_tiles_block_movement_but_not_sight() -> None:
    game_map = _map_with_water(10, 6)
    assert game_map.is_water(10, 6)
    assert not game_map.is_walkable(10, 6)
    # Water is transparent -- line of sight passes across it.
    assert game_map.has_line_of_sight((8, 6), (12, 6))


def test_clear_water_around_restores_floor() -> None:
    game_map = _map_with_water(10, 6)
    game_map.clear_water_around(10, 6, radius=1)
    assert not game_map.is_water(10, 6)
    assert game_map.is_walkable(10, 6)


def test_thirsty_deer_drinks_from_adjacent_water() -> None:
    game_map = _map_with_water(11, 6)
    processor = NpcAiProcessor(game_map)  # shore tiles precomputed here
    deer = esper.create_entity(
        Position(10, 6), NPC(), Deer(), Diet("herbivore"),
        Needs(thirst=90.0, hunger=10.0), BlocksMovement(),
    )

    processor.process("move_up")
    assert esper.component_for_entity(deer, Needs).thirst == 0.0


def test_hungry_deer_grazes_adjacent_tree_and_depletes_it() -> None:
    game_map = GameMap(24, 14)
    processor = NpcAiProcessor(game_map)
    tree = esper.create_entity(Position(11, 6), Tree(wood=2), BlocksMovement())
    deer = esper.create_entity(
        Position(10, 6), NPC(), Deer(), Diet("herbivore"),
        Needs(hunger=90.0, thirst=10.0), BlocksMovement(),
    )

    processor.process("move_up")
    assert esper.component_for_entity(deer, Needs).hunger < 90.0
    assert esper.component_for_entity(tree, Tree).wood == 1


def test_hungry_carnivore_hunts_adjacent_deer_into_meat() -> None:
    game_map = GameMap(24, 14)
    processor = NpcAiProcessor(game_map)
    prey = esper.create_entity(
        Position(11, 6), NPC(), Deer(), Diet("herbivore"), Meat("Deer Meat"),
        Needs(), BlocksMovement(), Name("Deer"),
    )
    predator = esper.create_entity(
        Position(10, 6), NPC(), Enemy(), Diet("carnivore"),
        Needs(hunger=90.0, thirst=10.0), BlocksMovement(), Name("Goblin"),
    )

    processor.process("move_up")

    assert not esper.entity_exists(prey)
    assert esper.component_for_entity(predator, Needs).hunger < 90.0
    corpse_loot = [
        esper.component_for_entity(ent, Inventory).items
        for ent, (_corpse,) in esper.get_components(Corpse)
    ]
    assert corpse_loot == [["Deer Meat"]]


def test_is_passable_allows_water_but_not_walls() -> None:
    game_map = _map_with_water(10, 6)
    assert game_map.is_passable(10, 6)  # water: swimmable
    assert not game_map.is_passable(0, 0)  # border wall
    assert not game_map.is_walkable(10, 6)  # but NPCs still avoid water


def test_player_can_swim_into_water_but_walls_still_block() -> None:
    game_map = _map_with_water(6, 5)
    esper.add_processor(MovementProcessor(game_map), priority=1)
    player_pos = Position(5, 5)
    esper.create_entity(player_pos, Renderable("@"), Name("You"), Player(), BlocksMovement())

    esper.process("move_right")  # into water at (6, 5)
    assert (player_pos.x, player_pos.y) == (6, 5)
    assert game_map.is_water(player_pos.x, player_pos.y)

    # Turn the tile to the right into a wall and confirm it blocks.
    game_map.tiles[5][7] = GameMap.WALL
    esper.process("move_right")
    assert (player_pos.x, player_pos.y) == (6, 5)


def test_hungry_deer_steps_toward_distant_tree() -> None:
    game_map = GameMap(24, 14)
    processor = NpcAiProcessor(game_map)
    esper.create_entity(Position(18, 6), Tree(wood=3), BlocksMovement())
    deer = esper.create_entity(
        Position(10, 6), NPC(), Deer(), Diet("herbivore"),
        Needs(hunger=90.0, thirst=10.0), BlocksMovement(),
    )

    processor.process("move_up")
    pos = esper.component_for_entity(deer, Position)
    # Moved one step closer along x, not yet adjacent.
    assert pos.x == 11 and pos.y == 6
