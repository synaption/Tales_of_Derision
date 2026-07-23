"""Status effects: a registry that unifies the status-identifier display and the
(future) per-turn behaviour of ailments/buffs.

Each ``EffectDef`` carries the display an on-tile status identifier animates
(glyph/colour/seconds -- read by the renderer) plus an optional ``component`` that
marks it, an optional ``detector`` for effects derived from world state, and optional
``on_apply/on_tick/on_remove`` Python handlers for behaviour.

This replaces the old ad-hoc ``OnFire`` handling + ``_STATUS_DISPLAY``/``_STATUS_ORDER``
tables in ``systems``. Registration order is display order.

Data path: ``register_effect(EffectDef(..., glyph=..., seconds=...))``. Python path:
supply ``on_tick``/``on_apply`` for behaviour.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import esper

from components import Asleep, OnFire, Position

RGB = tuple[int, int, int]

# How long the character's own tile shows before the status identifiers cycle.
STATUS_BASE_SECONDS = 1.0

# A world-state detector answers "is this effect active on ``ent`` at ``pos``?" for
# effects that aren't marked by a component (e.g. swimming = standing on water).
Detector = Callable[[object, int, Position], bool]


@dataclass
class EffectDef:
    id: str
    glyph: str                     # status-identifier glyph shown in the tile cycle
    seconds: float                 # how long the identifier shows
    label: str                     # human-readable name (status/examine screens)
    fg: RGB | None = None          # identifier colour, or None to keep the char's own
    component: type | None = None  # component that marks this effect (e.g. OnFire)
    detector: Detector | None = None
    on_apply: Callable[[int], None] | None = None
    on_tick: Callable[[int], None] | None = None
    on_remove: Callable[[int], None] | None = None


# Insertion order == display order (dict preserves it).
_EFFECTS: dict[str, EffectDef] = {}


def register_effect(defn: EffectDef) -> EffectDef:
    _EFFECTS[defn.id] = defn
    return defn


def effect(effect_id: str) -> EffectDef | None:
    return _EFFECTS.get(effect_id)


def effects_in_order() -> list[EffectDef]:
    return list(_EFFECTS.values())


def effect_display(effect_id: str) -> tuple[str, RGB | None, float]:
    """``(glyph, fg, seconds)`` for the status-identifier animation."""
    defn = _EFFECTS[effect_id]
    return (defn.glyph, defn.fg, defn.seconds)


def effect_label(effect_id: str) -> str:
    defn = _EFFECTS.get(effect_id)
    if defn is not None:
        return defn.label
    return effect_id.replace("_", " ").capitalize()


def active_effects(game_map, ent: int, pos: Position) -> list[str]:
    """The effect ids currently affecting ``ent``, in display order. Component-marked
    effects check the component; derived effects run their detector."""
    active: list[str] = []
    for defn in _EFFECTS.values():
        if defn.component is not None and esper.has_component(ent, defn.component):
            active.append(defn.id)
        elif defn.detector is not None and defn.detector(game_map, ent, pos):
            active.append(defn.id)
    return active


def apply_effect(ent: int, effect_id: str) -> None:
    """Apply an effect: attach its marker component (if any) and run ``on_apply``."""
    defn = _EFFECTS.get(effect_id)
    if defn is None:
        return
    if defn.component is not None and not esper.has_component(ent, defn.component):
        esper.add_component(ent, defn.component())
    if defn.on_apply is not None:
        defn.on_apply(ent)


def remove_effect(ent: int, effect_id: str) -> None:
    defn = _EFFECTS.get(effect_id)
    if defn is None:
        return
    if defn.component is not None and esper.has_component(ent, defn.component):
        esper.remove_component(ent, defn.component)
    if defn.on_remove is not None:
        defn.on_remove(ent)


class EffectsProcessor(esper.Processor):
    """Ticks the ``on_tick`` of every component-marked effect once per real turn.
    A no-op until an effect registers behaviour (e.g. fire that burns); the seam is
    here so it can be added as pure data + a handler without touching the loop."""

    def process(self, action: str | None = None) -> None:
        if action is None:
            return  # menu refreshes advance nothing
        for defn in _EFFECTS.values():
            if defn.on_tick is None or defn.component is None:
                continue
            for ent, _ in esper.get_components(defn.component):
                defn.on_tick(ent)


# --- Core effects -----------------------------------------------------------
# Order here is the tile-cycle order: own glyph -> swimming -> on fire -> asleep.
def _is_swimming(game_map, _ent: int, pos: Position) -> bool:
    return game_map.is_water(pos.x, pos.y)


register_effect(EffectDef("swimming", glyph="~", seconds=0.5, label="Swimming", detector=_is_swimming))
register_effect(EffectDef("on_fire", glyph="F", seconds=0.5, label="On fire", fg=(224, 74, 44), component=OnFire))
register_effect(EffectDef("asleep", glyph="Z", seconds=0.6, label="Asleep", fg=(150, 170, 220), component=Asleep))
