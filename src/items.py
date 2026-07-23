"""Backwards-compatible façade over :mod:`content.items`.

Items moved into the content registry (see ``content/items.py`` and
``wiki/Content-and-Mods.md``). This module re-exports the same public names so
existing imports (``from items import WOOD, hunger_restored, ...``) keep working.
Prefer importing from ``content.items`` in new code.
"""
from __future__ import annotations

from content.items import (  # noqa: F401  (re-exported for backwards compatibility)
    BERRIES,
    COOKED_MEAT,
    COOKED_MEAT_HUNGER,
    CRAFTING_RECIPES,
    DRINK_VALUES,
    EAT_VALUES,
    ItemDef,
    PLACEABLE_TILES,
    RAW_MEAT,
    RAW_MEAT_HUNGER,
    WOOD,
    WOOD_DOOR,
    WOOD_WALL,
    WOOD_WINDOW,
    cook_meat,
    craft_cost,
    default_equipment_slots,
    hunger_restored,
    is_consumable,
    is_cooked_meat,
    is_placeable,
    is_raw_meat,
    item_def,
    placed_tile,
    register_item,
    thirst_restored,
)
