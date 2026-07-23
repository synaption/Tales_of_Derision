"""Component **kits** -- reusable bundles of components that apply to many entities.

A kit returns a fresh ``list`` of component instances (never shared between
entities). Prefabs compose from kits by name (see ``content.registry``), so a new
creature is usually just "a base glyph + a kit + a couple of extras" rather than a
hand-written component list. This is the concrete form of "reusable chunks of code
that apply to many entities".
"""
from __future__ import annotations

from collections.abc import Callable

from components import (
    Attributes, BlocksMovement, Deer, Dialogue, Diet, Enemy, Equipment, Family,
    Fish, Friendly, Gender, Inventory, Meat, NPC, Needs, Personality,
    Relationships, Resident, Vision,
)
from content.items import RAW_MEAT, default_equipment_slots

# The fake NPC language every villager speaks (see wiki/Gibberish.md).
GIBBERISH_LINE = "##!/$*~# GH01^@"


def predator_kit(
    *,
    meat: str = RAW_MEAT,
    vision: int = 8,
    dexterity: int | None = None,
    enemy: bool = True,
    hunger: float = 0.0,
    thirst: float = 0.0,
) -> list:
    """A carnivore that hunts prey and eats raw (rats, goblins). ``dexterity`` adds
    an ``Attributes`` only when given, so a quick creature acts more often in the
    action economy."""
    comps: list = [
        NPC(), Diet("carnivore"), Meat(meat), Vision(vision), BlocksMovement(),
        Needs(hunger=hunger, thirst=thirst),
    ]
    if enemy:
        comps.append(Enemy())
    if dexterity is not None:
        comps.append(Attributes(dexterity=dexterity))
    return comps


def grazer_kit(
    *,
    meat: str = "Deer Meat",
    vision: int = 8,
    hunger: float = 0.0,
    thirst: float = 0.0,
) -> list:
    """A wild herbivore that grazes trees and drinks water (deer). Prey for
    carnivores and the player."""
    return [
        NPC(), Deer(), Diet("herbivore"), Vision(vision), BlocksMovement(),
        Meat(meat), Needs(hunger=hunger, thirst=thirst),
    ]


def fish_kit(*, hunger: float = 25.0) -> list:
    """A sea creature driven by ``FishAiProcessor`` (water-only, never the land
    pathfinder). Never thirsty; its only drive is hunger, which sends it grazing
    seaweed."""
    return [Fish(), Needs(hunger=hunger, thirst=0.0, thirst_rate=0.0, tiredness_rate=0.0)]


def person_kit(*, gender: str, traits: list[str], surname: str) -> list:
    """A villager 'cook': gathers meat + wood and cooks before eating, seeks a home,
    and has a personality that drives who it befriends. ``Age``/``Name``/``Renderable``
    are added by the caller (they need the clock / a chosen name)."""
    return [
        Gender(gender),
        Family(surname=surname),
        NPC(),
        Friendly(),
        Dialogue(GIBBERISH_LINE),
        BlocksMovement(),
        Inventory(items=["Bread", "Waterskin"]),
        Equipment(slots=default_equipment_slots()),
        Diet("cook"),
        Needs(),
        Resident(),
        Personality(traits=list(traits)),
        Relationships(),
    ]


# Kit registry: name -> builder. Prefabs reference kits by name; mods can add more.
KITS: dict[str, Callable[..., list]] = {
    "predator": predator_kit,
    "grazer": grazer_kit,
    "fish": fish_kit,
    "person": person_kit,
}


def register_kit(name: str, builder: Callable[..., list]) -> None:
    KITS[name] = builder


def build_kit(name: str, kwargs: dict) -> list:
    """Invoke kit ``name`` with ``kwargs``, returning fresh components."""
    builder = KITS.get(name)
    if builder is None:
        raise KeyError(f"unknown kit {name!r}")
    return builder(**kwargs)
