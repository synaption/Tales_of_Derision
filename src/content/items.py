"""Item definitions.

Items live in ``Inventory`` as plain **name strings** (see ``components.Inventory``);
this module gives each name a declarative ``ItemDef`` (food/drink value, equip slot,
placeable tile, craft cost, ...) and the shared lookup helpers the turn loop and the
ECS processors agree on. Keeping items as strings means nothing about the inventory
plumbing changes -- the registry just gives each string a definition.

Data path: ``register_item(ItemDef(...))``. Python path: nothing special is needed --
an item is data; behaviour lives in the systems that read these helpers.

``src/items.py`` re-exports this module's public API for backwards compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field

RGB = tuple[int, int, int]

# --- Canonical item names (constants, so a rename happens in one place) ------
RAW_MEAT = "Raw Meat"        # generic fallback when a creature has no named meat
COOKED_MEAT = "Cooked Meat"  # what cooking generic Raw Meat yields
WOOD = "Wood"

WOOD_WALL = "Wall"
WOOD_DOOR = "Door"
WOOD_WINDOW = "Window"

BERRIES = "Berries"

# Hunger points restored by cooked vs raw meat. Cooking is the payoff: cooked meat
# is far more filling (and safe); raw meat barely helps.
COOKED_MEAT_HUNGER = 45.0
RAW_MEAT_HUNGER = 8.0

# Meat is named per creature ("Rat Meat", ...) so it is recognised by *shape* rather
# than a table: a raw meat is any "... Meat"; cooking prefixes "Cooked ".
_MEAT_SUFFIX = " Meat"
_COOKED_PREFIX = "Cooked "


@dataclass
class ItemDef:
    """One item's data. Only ``name`` is required; every capability is opt-in, so a
    plain material (``Wood``) and an edible (``Bread``) share one shape."""
    name: str
    eat: float | None = None            # hunger points restored when eaten
    drink: float | None = None          # thirst points restored when drunk
    equip_slot: str | None = None       # equipment slot this fits, if any
    placeable_tile: str | None = None   # map tile char it becomes when placed
    craft_cost: int | None = None       # wood pieces to craft one, if craftable
    visual: tuple[str, RGB] | None = None
    on_consume: str | None = None       # effect id applied on eat/drink (potions, ...)
    tags: tuple[str, ...] = field(default_factory=tuple)


_ITEMS: dict[str, ItemDef] = {}


def register_item(defn: ItemDef) -> ItemDef:
    """Register (or override) an item definition by name."""
    _ITEMS[defn.name] = defn
    return defn


def item_def(name: str) -> ItemDef | None:
    """The registered definition for ``name``, or ``None`` if unregistered (dynamic
    names such as per-creature meats are handled by shape, not the registry)."""
    return _ITEMS.get(name)


def all_items() -> list[ItemDef]:
    return list(_ITEMS.values())


# --- Core item catalogue ----------------------------------------------------
for _defn in (
    ItemDef(WOOD, tags=("material",)),
    ItemDef("Bread", eat=30.0),
    ItemDef("Apple", eat=15.0),
    ItemDef(BERRIES, eat=12.0),
    ItemDef("Waterskin", drink=40.0),
    ItemDef("Water", drink=35.0),
    ItemDef(WOOD_WALL, placeable_tile="#", craft_cost=2, tags=("buildable",)),
    ItemDef(WOOD_WINDOW, placeable_tile="o", craft_cost=2, tags=("buildable",)),
    ItemDef(WOOD_DOOR, placeable_tile="+", craft_cost=3, tags=("buildable",)),
):
    register_item(_defn)


# --- Shape-based meat helpers (dynamic per-creature names) -------------------
def is_cooked_meat(item_name: str) -> bool:
    return item_name.startswith(_COOKED_PREFIX) and item_name.endswith(_MEAT_SUFFIX)


def is_raw_meat(item_name: str) -> bool:
    return item_name.endswith(_MEAT_SUFFIX) and not is_cooked_meat(item_name)


def cook_meat(item_name: str) -> str:
    """The cooked form of a raw meat. ``"Raw Meat" -> "Cooked Meat"`` and
    ``"Rat Meat" -> "Cooked Rat Meat"``."""
    base = item_name[len("Raw "):] if item_name.startswith("Raw ") else item_name
    return f"{_COOKED_PREFIX}{base}"


# --- Registry-backed lookups (query live, so mods' items work too) ----------
def hunger_restored(item_name: str) -> float | None:
    """Hunger points restored by eating ``item_name``, or ``None`` if inedible."""
    if is_cooked_meat(item_name):
        return COOKED_MEAT_HUNGER
    if is_raw_meat(item_name):
        return RAW_MEAT_HUNGER
    defn = _ITEMS.get(item_name)
    return defn.eat if defn is not None else None


def thirst_restored(item_name: str) -> float | None:
    """Thirst points restored by drinking ``item_name``, or ``None`` if not a drink."""
    defn = _ITEMS.get(item_name)
    return defn.drink if defn is not None else None


def is_consumable(item_name: str) -> bool:
    return hunger_restored(item_name) is not None or thirst_restored(item_name) is not None


def is_placeable(item_name: str) -> bool:
    defn = _ITEMS.get(item_name)
    return defn is not None and defn.placeable_tile is not None


def placed_tile(item_name: str) -> str | None:
    """The map tile a placeable item becomes, or ``None`` if it isn't placeable."""
    defn = _ITEMS.get(item_name)
    return defn.placeable_tile if defn is not None else None


def craft_cost(item_name: str) -> int | None:
    """Wood pieces needed to craft ``item_name``, or ``None`` if not craftable."""
    defn = _ITEMS.get(item_name)
    return defn.craft_cost if defn is not None else None


def default_equipment_slots() -> dict[str, str | None]:
    """A fresh, empty equipment layout. Returned per call (never a shared dict)."""
    return {
        "head": None,
        "chest": None,
        "hands": None,
        "legs": None,
        "feet": None,
        "main hand": None,
        "off hand": None,
        "ring": None,
    }


# --- Legacy snapshot tables (core items only; kept for direct readers) -------
# Functions above query the live registry so mod items work; these dicts are a
# convenience snapshot of the core catalogue at import time.
EAT_VALUES: dict[str, float] = {d.name: d.eat for d in _ITEMS.values() if d.eat is not None}
DRINK_VALUES: dict[str, float] = {d.name: d.drink for d in _ITEMS.values() if d.drink is not None}
PLACEABLE_TILES: dict[str, str] = {
    d.name: d.placeable_tile for d in _ITEMS.values() if d.placeable_tile is not None
}
CRAFTING_RECIPES: dict[str, int] = {
    d.name: d.craft_cost for d in _ITEMS.values() if d.craft_cost is not None
}
