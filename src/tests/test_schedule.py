"""The deterministic turn scheduler: ordering, tie-breaks, cost charging, and the
emergent property that quicker actors take proportionally more turns.
"""
from __future__ import annotations

import esper
import pytest

from action import BASE_ACTION_COST
from components import Actor, Attributes
from schedule import actor_time, complete_action, next_actor, schedule_actor

pytestmark = pytest.mark.unrendered


def test_next_actor_is_none_when_nothing_scheduled() -> None:
    assert next_actor() is None


def test_next_actor_picks_smallest_time() -> None:
    a = esper.create_entity()
    b = esper.create_entity()
    schedule_actor(a, at_time=50)
    schedule_actor(b, at_time=10)
    assert next_actor() == b


def test_ties_break_by_entity_id_deterministically() -> None:
    first = esper.create_entity()
    second = esper.create_entity()
    schedule_actor(first, at_time=0)
    schedule_actor(second, at_time=0)
    # Same time -> the lower entity id (created first) goes first.
    assert next_actor() == first


def test_complete_action_advances_by_baseline_cost() -> None:
    ent = esper.create_entity(Attributes(dexterity=10))
    schedule_actor(ent, at_time=300)
    cost = complete_action(ent, "move_up")
    assert cost == BASE_ACTION_COST
    assert actor_time(ent) == 300 + BASE_ACTION_COST
    # last_acted records the time it just acted at (its old scheduled time).
    assert esper.component_for_entity(ent, Actor).last_acted == 300


def test_schedule_actor_is_idempotent() -> None:
    ent = esper.create_entity()
    schedule_actor(ent, at_time=5)
    schedule_actor(ent, at_time=99)
    assert actor_time(ent) == 99


def _run_until(horizon: int) -> list[int]:
    """Pop actors until the next one to act would land at/after ``horizon``.
    Returns the sequence of entities that acted (for determinism checks)."""
    order: list[int] = []
    while True:
        ent = next_actor()
        if ent is None or actor_time(ent) >= horizon:
            return order
        order.append(ent)
        complete_action(ent, "move_up")


def test_quicker_actor_takes_more_turns_over_a_span() -> None:
    slow = esper.create_entity(Attributes(dexterity=10))
    fast = esper.create_entity(Attributes(dexterity=25))
    schedule_actor(slow, at_time=0)
    schedule_actor(fast, at_time=0)

    order = _run_until(horizon=2000)
    assert order.count(fast) > order.count(slow)


def test_scheduler_sequence_is_deterministic() -> None:
    # Build the same two-actor race twice; the popped order must be identical.
    def build_and_run() -> list[int]:
        esper.clear_database()
        a = esper.create_entity(Attributes(dexterity=12))
        b = esper.create_entity(Attributes(dexterity=17))
        schedule_actor(a, at_time=0)
        schedule_actor(b, at_time=0)
        # Record relative order (a-first vs b-first) rather than raw ids, since ids
        # are assigned fresh each build.
        return [0 if ent == a else 1 for ent in _run_until(horizon=1500)]

    assert build_and_run() == build_and_run()
