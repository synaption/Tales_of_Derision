from __future__ import annotations

import pytest
import esper

from components import Equipment, Inventory, Name
from main import _equip_inventory_item, _infer_slot_for_item, _list_trade_entries, _trade_item, _unequip_slot

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
