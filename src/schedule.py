"""Deterministic turn scheduler for the action economy.

Turn order is decided by *completion time*: whoever's next action lands soonest
acts next. Rather than keep a separate heap that could drift out of sync with the
world, the schedule lives on the entities themselves -- each actor's
``Actor.next_time`` -- and we pick the minimum by ``(next_time, entity id)``. The
entity-id tie-break makes equal-time turns fully deterministic, which is what lets a
fixed seed plus a fixed input sequence reproduce the world exactly.

(An O(n) scan over actors is plenty while the near cast is small and can never
desync from the ECS; far-away crowds stay cheap through the region scheduler, not
this queue. A heap can replace the scan later without changing this interface.)
"""
from __future__ import annotations

import esper

from action import action_cost
from components import Actor


def schedule_actor(ent: int, at_time: int = 0) -> Actor:
    """Give ``ent`` an ``Actor`` scheduled to act at ``at_time``. Idempotent: an
    existing ``Actor`` is simply re-timed."""
    if esper.has_component(ent, Actor):
        actor = esper.component_for_entity(ent, Actor)
        actor.next_time = at_time
        actor.last_acted = at_time
        return actor
    actor = Actor(next_time=at_time, last_acted=at_time)
    esper.add_component(ent, actor)
    return actor


def next_actor() -> int | None:
    """The entity that acts next -- the smallest ``(next_time, entity id)`` among
    all actors -- or ``None`` if nothing is scheduled."""
    best_key: tuple[int, int] | None = None
    best_ent: int | None = None
    for ent, (actor,) in esper.get_components(Actor):
        key = (actor.next_time, ent)
        if best_key is None or key < best_key:
            best_key = key
            best_ent = ent
    return best_ent


def actor_time(ent: int) -> int:
    """When ``ent`` is next scheduled to act (its ``Actor.next_time``)."""
    return esper.component_for_entity(ent, Actor).next_time


def complete_action(ent: int, action: str | None) -> int:
    """Charge ``ent`` for taking ``action``: stamp that it acted at its scheduled
    time and push its next turn out by the action's cost. Returns the cost spent."""
    actor = esper.component_for_entity(ent, Actor)
    cost = action_cost(ent, action)
    actor.last_acted = actor.next_time
    actor.next_time += cost
    return cost
