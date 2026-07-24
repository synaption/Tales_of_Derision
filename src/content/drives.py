"""Declarative NPC drive definitions.

The AI processor owns execution (pathfinding, item mutation, social side effects),
but the decision ladder is data: each drive names cheap triggers, a weight, the
on-screen action handler, and optional closed-form resolve metadata for future
low-detail catch-up.  This mirrors the prefab registry's "Python that reads like
data" style while keeping the selector deliberately dumb and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


TriggerSpec = tuple[Any, ...]


@dataclass(frozen=True)
class WeightModifier:
    factor: float
    trigger: TriggerSpec


@dataclass(frozen=True)
class DriveWeight:
    base: float
    modifiers: tuple[WeightModifier, ...] = ()


@dataclass(frozen=True)
class DriveResolve:
    requires: tuple[TriggerSpec, ...] = ()
    success: float = 1.0
    effect: tuple[tuple[Any, ...], ...] = ()


@dataclass(frozen=True)
class DriveDef:
    id: str
    allow: tuple[TriggerSpec, ...]
    weight: DriveWeight
    act: str
    resolve: DriveResolve | None = None
    mode: str = "highest_weight"


_DRIVES: dict[str, DriveDef] = {}


def _trigger(spec: str | tuple[Any, ...]) -> TriggerSpec:
    if isinstance(spec, tuple):
        return spec
    return (spec,)


def _weight(spec: dict | int | float | DriveWeight) -> DriveWeight:
    if isinstance(spec, DriveWeight):
        return spec
    if isinstance(spec, (int, float)):
        return DriveWeight(float(spec))
    modifiers = tuple(
        WeightModifier(float(mod["factor"]), _trigger(mod["trigger"]))
        for mod in spec.get("modifiers", ())
    )
    return DriveWeight(float(spec.get("base", 0.0)), modifiers)


def _resolve(spec: dict | DriveResolve | None) -> DriveResolve | None:
    if spec is None or isinstance(spec, DriveResolve):
        return spec
    return DriveResolve(
        requires=tuple(_trigger(req) for req in spec.get("requires", ())),
        success=float(spec.get("success", 1.0)),
        effect=tuple(tuple(effect) for effect in spec.get("effect", ())),
    )


def register_drive(defn: DriveDef) -> None:
    _DRIVES[defn.id] = defn


def drive(def_id: str, *, allow=(), weight=0.0, act: str, resolve=None, mode="highest_weight") -> DriveDef:
    return DriveDef(
        id=def_id,
        allow=tuple(_trigger(item) for item in allow),
        weight=_weight(weight),
        act=act,
        resolve=_resolve(resolve),
        mode=mode,
    )


def get_drive(drive_id: str) -> DriveDef:
    return _DRIVES[drive_id]


def all_drives() -> tuple[DriveDef, ...]:
    return tuple(_DRIVES.values())


# Strictly spaced bases preserve the previous if/elif priority ladder exactly:
# sleep > thirst > hunger/eat > build > social > hostile chase.  Randomness is
# intentionally absent here so time travel remains seed-deterministic; future
# weighted-random tiers should draw only from world_rng streams keyed by logical
# turn and entity.
register_drive(drive(
    "sleep",
    allow=(("tiredness_at_least", 70.0),),
    weight={"base": 600.0},
    act="seek_sleep",
    resolve={"effect": (("asleep", "set", True),)},
))
register_drive(drive(
    "drink",
    allow=(("thirst_at_least", 55.0), ("thirst_beats_hunger",)),
    weight={"base": 500.0},
    act="seek_water",
    resolve={
        "requires": (("region_has_shore",),),
        "success": 0.95,
        "effect": (("thirst", "set", 0.0),),
    },
))
register_drive(drive(
    "eat_inventory",
    allow=(("hunger_at_least", 55.0), ("has_inventory_food",)),
    weight={"base": 410.0},
    act="eat_from_inventory",
    resolve={"effect": (("hunger", "sub", 50.0),)},
))
register_drive(drive(
    "forage_berries",
    allow=(("hunger_at_least", 55.0), ("diet_in", "herbivore", "cook"), ("region_has_bushes",)),
    weight={"base": 405.0},
    act="forage_berries",
    resolve={"requires": (("region_has_bushes",),), "success": 0.9, "effect": (("hunger", "sub", 45.0),)},
))
register_drive(drive(
    "graze",
    allow=(("hunger_at_least", 55.0), ("has_diet", "herbivore"), ("region_has_trees",)),
    weight={"base": 400.0, "modifiers": ({"factor": 1.5, "trigger": ("is_starving",)},)},
    act="graze",
    resolve={"requires": (("region_has_trees",),), "success": 0.9, "effect": (("hunger", "sub", 45.0),)},
))
register_drive(drive(
    "hunt_or_scavenge",
    allow=(("hunger_at_least", 55.0), ("has_diet", "carnivore"), ("region_has_meat",)),
    weight={"base": 400.0},
    act="seek_food",
))
register_drive(drive(
    "cook",
    allow=(("hunger_at_least", 55.0), ("has_diet", "cook")),
    weight={"base": 390.0},
    act="feed_cook",
))
register_drive(drive(
    "build_house",
    allow=(("should_build",),),
    weight={"base": 200.0},
    act="work_blueprints",
    resolve={"requires": (("region_has_blueprint",),), "effect": (("blueprint_progress", "add_per_turn", 1),)},
))
register_drive(drive(
    "socialize",
    allow=(("can_socialize",),),
    weight={"base": 100.0},
    act="socialize",
))
register_drive(drive(
    "chase_player",
    allow=(("has_player",), ("is_enemy",)),
    weight={"base": 10.0},
    act="chase_player",
))
