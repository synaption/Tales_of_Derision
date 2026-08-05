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

import ecs

from components import Asleep, OnFire, Player, Position
from regions import _UNBOUNDED
import spatial

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
        if defn.component is not None and ecs.has_component(ent, defn.component):
            active.append(defn.id)
        elif defn.detector is not None and defn.detector(game_map, ent, pos):
            active.append(defn.id)
    return active


def apply_effect(ent: int, effect_id: str) -> None:
    """Apply an effect: attach its marker component (if any) and run ``on_apply``."""
    defn = _EFFECTS.get(effect_id)
    if defn is None:
        return
    if defn.component is not None and not ecs.has_component(ent, defn.component):
        ecs.add_component(ent, defn.component())
    if defn.on_apply is not None:
        defn.on_apply(ent)


def remove_effect(ent: int, effect_id: str) -> None:
    defn = _EFFECTS.get(effect_id)
    if defn is None:
        return
    if defn.component is not None and ecs.has_component(ent, defn.component):
        ecs.remove_component(ent, defn.component)
    if defn.on_remove is not None:
        defn.on_remove(ent)


class EffectsProcessor(ecs.Processor):
    """Ticks the ``on_tick`` of every component-marked effect once per turn *of the
    region the affected thing is standing in*.

    A no-op until an effect registers behaviour (e.g. fire that burns); the seam is
    here so it can be added as pure data + a handler without touching the loop. It
    is region-scoped from the outset so that when behaviour does arrive, a fire
    burning on a far island burns on that island's turns -- in the background pump,
    on region entry, or during sleep -- and never on the player's keypress.
    """

    def __init__(self, game_map=None) -> None:
        # With a map, effects belong to a region's turn (see ``register_region_step``).
        # Without one -- a processor built directly in a unit test -- they tick
        # world-wide per turn, the original behaviour.
        self.game_map = game_map
        self._scheduler_driven = False

    def register_region_step(self, scheduler) -> None:
        """Make effects part of a region's turn."""
        scheduler.register("effects", self.advance_region, idle_turns=self._idle_turns)
        self._scheduler_driven = True

    @staticmethod
    def _idle_turns(_region_id) -> int:
        """How many upcoming turns this step provably has nothing to do for.

        While no registered effect declares an ``on_tick`` there is nothing to run
        on any turn, so a lagging region need not be visited on this step's
        account at all. Once one does, its behaviour is per-turn by definition and
        every turn has to be lived -- unless the effect itself grows a bulk form.

        Without this the step would pin every region to one turn at a time, which
        is what a step that offers neither hook means (see
        ``RegionScheduler.register``) -- and a no-op would be the thing preventing
        an empty ocean region from skipping a thousand turns of nothing.
        """
        return 0 if EffectsProcessor._tickable() else _UNBOUNDED

    @staticmethod
    def _tickable():
        return [
            defn for defn in _EFFECTS.values()
            if defn.on_tick is not None and defn.component is not None
        ]

    def advance_region(self, region_id) -> None:
        """One region-turn of every active effect on that region's entities."""
        if self.game_map is None:
            return
        tickable = self._tickable()
        if not tickable:
            return  # nothing declares behaviour: don't even ask the index
        index = spatial.ensure(self.game_map)
        for defn in tickable:
            for ent in sorted(index.entities_in(region_id)):
                if ecs.entity_exists(ent) and ecs.has_component(ent, defn.component):
                    defn.on_tick(ent)

    def process(self, action: str | None = None) -> None:
        if action is None:
            return  # menu refreshes advance nothing
        for defn in self._tickable():
            if self._scheduler_driven:
                # Every creature's effects tick with its own region; the player's
                # tick here, because the player's turn *is* the turn.
                for ent, _ in ecs.get_components(defn.component, Player):
                    defn.on_tick(ent)
                continue
            for ent, _ in ecs.get_components(defn.component):
                defn.on_tick(ent)


# --- Core effects -----------------------------------------------------------
# Order here is the tile-cycle order: own glyph -> swimming -> on fire -> asleep.
def _is_swimming(game_map, _ent: int, pos: Position) -> bool:
    return game_map.is_water(pos.x, pos.y)


register_effect(EffectDef("swimming", glyph="~", seconds=0.5, label="Swimming", detector=_is_swimming))
register_effect(EffectDef("on_fire", glyph="F", seconds=0.5, label="On fire", fg=(224, 74, 44), component=OnFire))
register_effect(EffectDef("asleep", glyph="Z", seconds=0.6, label="Asleep", fg=(150, 170, 220), component=Asleep))
