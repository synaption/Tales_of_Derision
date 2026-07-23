from __future__ import annotations

import pytest
import esper

from components import Corpse, Equipment, Inventory, Name, NPC, Player, Position, Renderable
from interactions import (
    _equip_inventory_item,
    _find_interaction_npc,
    _find_interaction_corpse,
    _infer_slot_for_item,
    _list_trade_entries,
    _loot_item_from_corpse,
    _trade_item,
    _unequip_slot,
)

pytestmark = pytest.mark.unrendered


def test_infer_slot_for_item_uses_name_keywords() -> None:
    assert _infer_slot_for_item("Rusty Sword") == "main hand"
    assert _infer_slot_for_item("Traveler Tunic") == "chest"
    assert _infer_slot_for_item("Iron Shield") == "off hand"
    assert _infer_slot_for_item("Unknown Curio") is None


def test_equip_inventory_item_moves_item_and_swaps_previous() -> None:
    inv = Inventory(items=["Rusty Sword", "Jagged Dagger"])
    equip = Equipment(slots={"main hand": "Wooden Club"})

    msg = _equip_inventory_item(inv, equip, 0)

    assert equip.slots["main hand"] == "Rusty Sword"
    assert inv.items == ["Jagged Dagger", "Wooden Club"]
    assert "Equipped Rusty Sword" in msg


def test_unequip_slot_moves_equipped_item_back_to_inventory() -> None:
    inv = Inventory(items=["Bandage"])
    equip = Equipment(slots={"chest": "Traveler Tunic"})

    msg = _unequip_slot(inv, equip, "chest")

    assert equip.slots["chest"] is None
    assert inv.items == ["Bandage", "Traveler Tunic"]
    assert "Unequipped Traveler Tunic" in msg


def test_trade_entries_include_equipped_and_inventory_items() -> None:
    ent = esper.create_entity(
        Name("You"),
        Inventory(items=["Apple"]),
        Equipment(slots={"main hand": "Rusty Sword", "chest": None}),
    )

    entries = _list_trade_entries(ent)
    labels = {(entry.kind, entry.item_name, entry.slot_name) for entry in entries}

    assert ("inventory", "Apple", None) in labels
    assert ("equipment", "Rusty Sword", "main hand") in labels


def test_trade_can_move_equipped_item_between_actors() -> None:
    player_ent = esper.create_entity(
        Name("You"),
        Inventory(items=[]),
        Equipment(slots={"main hand": "Rusty Sword"}),
    )
    npc_ent = esper.create_entity(
        Name("Villager"),
        Inventory(items=[]),
        Equipment(slots={"main hand": None}),
    )

    entry = _list_trade_entries(player_ent)[0]
    message = _trade_item(player_ent, npc_ent, entry)

    player_equipment = esper.component_for_entity(player_ent, Equipment)
    npc_inventory = esper.component_for_entity(npc_ent, Inventory)
    assert player_equipment.slots["main hand"] is None
    assert npc_inventory.items == ["Rusty Sword"]
    assert message == "You traded Rusty Sword to Villager."


def test_find_interaction_corpse_supports_directional_and_idle_lookup() -> None:
    esper.create_entity(Position(5, 5), Player())
    corpse_here = esper.create_entity(
        Position(5, 5),
        Corpse(),
        Name("Corpse Here"),
        Inventory(items=["Coin"]),
    )
    corpse_right = esper.create_entity(
        Position(6, 5),
        Corpse(),
        Name("Corpse Right"),
        Inventory(items=["Pebble"]),
    )
    corpse_up = esper.create_entity(
        Position(5, 4),
        Corpse(),
        Name("Corpse Up"),
        Inventory(items=["String"]),
    )

    assert _find_interaction_corpse(None) == corpse_here
    assert _find_interaction_corpse("move_right") == corpse_right
    assert _find_interaction_corpse("move_left") is None
    assert _find_interaction_corpse("move_down") is None


def test_find_interaction_npc_only_targets_aimed_tile_or_current_tile() -> None:
    esper.create_entity(Position(5, 5), Player())
    villager_up = esper.create_entity(
        Position(5, 4),
        NPC(),
        Name("Villager Up"),
    )
    esper.create_entity(
        Position(6, 5),
        Corpse(),
        Name("Corpse Right"),
        Inventory(items=["Pebble"]),
    )

    assert _find_interaction_npc("move_up") == villager_up
    assert _find_interaction_npc("move_right") is None
    assert _find_interaction_npc(None) is None


def test_loot_item_from_corpse_transfers_item_and_keeps_bones_glyph() -> None:
    player_ent = esper.create_entity(
        Name("You"),
        Position(5, 5),
        Player(),
        Inventory(items=[]),
    )
    corpse_ent = esper.create_entity(
        Name("Corpse of Goblin"),
        Position(6, 5),
        Renderable("x"),
        Corpse(),
        Inventory(items=["Copper Coin"]),
    )

    entry = _list_trade_entries(corpse_ent)[0]
    message = _loot_item_from_corpse(corpse_ent, player_ent, entry)

    player_inventory = esper.component_for_entity(player_ent, Inventory)
    corpse_renderable = esper.component_for_entity(corpse_ent, Renderable)
    assert player_inventory.items == ["Copper Coin"]
    assert esper.entity_exists(corpse_ent)
    assert corpse_renderable.glyph == "x"
    assert _find_interaction_corpse("move_right") is None
    assert "You looted Copper Coin from Corpse of Goblin." in message
