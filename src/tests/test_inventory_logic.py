from __future__ import annotations

import pytest

from components import Equipment, Inventory
from main import _equip_inventory_item, _infer_slot_for_item, _unequip_slot

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
