"""The time-based action economy: how long an action takes, in world time units.

Time is measured in **time units (TU)**. One baseline action (an average creature
taking a plain step) costs ``BASE_ACTION_COST`` TU, and a day is
``BASE_ACTION_COST`` x the old per-turn day length. Turn order is decided by *when*
each actor's action completes: the scheduler always runs whoever's ``next_time`` is
smallest, then pushes it forward by the action's cost.

Everything here defaults so that an all-baseline world (average dexterity, unit
action weights) behaves exactly like the old lockstep model -- one action per
``BASE_ACTION_COST`` TU -- so the switch is invisible until speeds are tuned.
"""
from __future__ import annotations

import esper

from components import Attributes

# One baseline action = 100 TU. Kept round and >1 so speeds can vary finely (a
# faster actor pays less, a slower one more) while staying integer.
BASE_ACTION_COST = 100

# Per-action time-cost multipliers (x BASE_ACTION_COST). Empty (all 1.0) for now so
# the baseline is preserved; add entries to make specific actions slower/faster
# (e.g. a heavier attack, a slow chop). Unknown actions default to 1.0.
ACTION_WEIGHTS: dict[str, float] = {}

# How much each point of dexterity above the 10 average speeds an actor up.
SPEED_PER_DEXTERITY = 0.03
# Floor so a very clumsy actor still acts (and we never divide by ~0).
_MIN_SPEED = 0.1


def action_weight(action: str | None) -> float:
    """The cost multiplier for an action type (1.0 unless tuned in ACTION_WEIGHTS)."""
    if action is None:
        return 1.0
    return ACTION_WEIGHTS.get(action, 1.0)


def actor_speed(ent: int) -> float:
    """How fast ``ent`` acts, as a multiplier on the base rate. Average dexterity
    (10) is 1.0; higher is faster. Entities without ``Attributes`` act at baseline."""
    if not esper.has_component(ent, Attributes):
        return 1.0
    dexterity = esper.component_for_entity(ent, Attributes).dexterity
    return max(_MIN_SPEED, 1.0 + (dexterity - 10) * SPEED_PER_DEXTERITY)


def action_cost(ent: int, action: str | None) -> int:
    """Time units ``ent`` spends performing ``action``. Baseline (average
    dexterity, unit weight) is exactly ``BASE_ACTION_COST``, so an all-baseline
    world keeps today's one-action-per-turn cadence."""
    return max(1, round(BASE_ACTION_COST * action_weight(action) / actor_speed(ent)))
