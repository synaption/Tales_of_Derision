"""Item name constants and survival-item lookup tables.

Items in this game are plain strings living in ``Inventory.items`` (see
``components.Inventory``). Both the turn loop (``main.py``) and the ECS
processors (``systems.py``) need to agree on the survival item names, so the
canonical strings and their effects live here in one renderer-agnostic module
that is trivial to unit test.
"""
from __future__ import annotations

# Material / food item names. Use these constants instead of bare string
# literals so a rename only happens in one place.
RAW_MEAT = "Raw Meat"  # generic fallback when a creature has no named meat
COOKED_MEAT = "Cooked Meat"  # what cooking generic Raw Meat yields
WOOD = "Wood"

# --- Buildable items --------------------------------------------------------
# Craftable pieces the player carries and then *places* on a faced tile, turning
# it into the matching map tile. NPCs build the same pieces straight from wood.
WOOD_WALL = "Wall"
WOOD_DOOR = "Door"
WOOD_WINDOW = "Window"

# Placeable item -> the GameMap tile char it becomes when placed. The tile chars
# are duplicated here (rather than imported from GameMap) to keep this module
# renderer/map agnostic and trivially testable; they must stay in sync.
PLACEABLE_TILES: dict[str, str] = {
    WOOD_WALL: "#",
    WOOD_DOOR: "+",
    WOOD_WINDOW: "o",
}

# Crafting recipes: placeable item -> wood pieces it costs to craft one. A door
# (hinges, a latch) costs more than a plain wall panel.
CRAFTING_RECIPES: dict[str, int] = {
    WOOD_WALL: 2,
    WOOD_WINDOW: 2,
    WOOD_DOOR: 3,
}


def is_placeable(item_name: str) -> bool:
    """True when the item is a buildable piece the player can place on a tile."""
    return item_name in PLACEABLE_TILES


def placed_tile(item_name: str) -> str | None:
    """The map tile a placeable item becomes, or ``None`` if it isn't placeable."""
    return PLACEABLE_TILES.get(item_name)


def craft_cost(item_name: str) -> int | None:
    """Wood pieces needed to craft ``item_name``, or ``None`` if not craftable."""
    return CRAFTING_RECIPES.get(item_name)

# Meat is named per creature ("Rat Meat", "Goblin Meat", ...) rather than a fixed
# item, so these are recognised by shape rather than by a lookup table. A raw
# meat is any "... Meat" item; cooking prefixes "Cooked ".
_MEAT_SUFFIX = " Meat"
_COOKED_PREFIX = "Cooked "

# Hunger points restored by eating cooked vs. raw meat. Cooking is the payoff:
# cooked meat is far more filling (and safe); raw meat barely helps.
COOKED_MEAT_HUNGER = 45.0
RAW_MEAT_HUNGER = 8.0

BERRIES = "Berries"

# Non-meat edibles and their hunger values.
EAT_VALUES: dict[str, float] = {
    "Bread": 30.0,
    "Apple": 15.0,
    BERRIES: 12.0,
}

# How much thirst (points) a drinkable item removes when consumed. A well is the
# renewable source; a waterskin is the portable one (consumed on use for now).
DRINK_VALUES: dict[str, float] = {
    "Waterskin": 40.0,
    "Water": 35.0,
}


def is_cooked_meat(item_name: str) -> bool:
    return item_name.startswith(_COOKED_PREFIX) and item_name.endswith(_MEAT_SUFFIX)


def is_raw_meat(item_name: str) -> bool:
    return item_name.endswith(_MEAT_SUFFIX) and not is_cooked_meat(item_name)


def cook_meat(item_name: str) -> str:
    """Return the cooked form of a raw meat. ``"Raw Meat" -> "Cooked Meat"`` and
    ``"Rat Meat" -> "Cooked Rat Meat"``."""
    base = item_name[len("Raw "):] if item_name.startswith("Raw ") else item_name
    return f"{_COOKED_PREFIX}{base}"


def hunger_restored(item_name: str) -> float | None:
    """Hunger points restored by eating ``item_name``, or ``None`` if inedible."""
    if is_cooked_meat(item_name):
        return COOKED_MEAT_HUNGER
    if is_raw_meat(item_name):
        return RAW_MEAT_HUNGER
    return EAT_VALUES.get(item_name)


def thirst_restored(item_name: str) -> float | None:
    """Thirst points restored by drinking ``item_name``, or ``None`` if not a drink."""
    return DRINK_VALUES.get(item_name)


def is_consumable(item_name: str) -> bool:
    """True when the item can be eaten or drunk."""
    return hunger_restored(item_name) is not None or thirst_restored(item_name) is not None
