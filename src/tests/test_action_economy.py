"""Action-economy cost model: baseline preservation and dexterity-driven speed.

These lock in the invariant that an all-baseline world (average dexterity, unit
action weights) costs exactly ``BASE_ACTION_COST`` per action -- so switching the
turn model from lockstep to the scheduler is invisible until speeds are tuned.
"""
from __future__ import annotations

import esper
import pytest

from action import BASE_ACTION_COST, action_cost, action_weight, actor_speed
from components import Attributes, Needs, Player, WorldClock
from systems import NeedsProcessor, TimeProcessor, world_clock

pytestmark = pytest.mark.unrendered


def test_actor_without_attributes_is_baseline_speed() -> None:
    ent = esper.create_entity()
    assert actor_speed(ent) == 1.0


def test_average_dexterity_is_baseline_speed() -> None:
    ent = esper.create_entity(Attributes(dexterity=10))
    assert actor_speed(ent) == 1.0


def test_baseline_action_costs_exactly_base_cost() -> None:
    ent = esper.create_entity(Attributes(dexterity=10))
    assert action_cost(ent, "move_up") == BASE_ACTION_COST
    assert action_cost(ent, "wait") == BASE_ACTION_COST
    # Unknown/None actions still fall back to the unit weight.
    assert action_cost(ent, None) == BASE_ACTION_COST


def test_higher_dexterity_acts_faster_costs_less() -> None:
    quick = esper.create_entity(Attributes(dexterity=20))
    assert actor_speed(quick) > 1.0
    assert action_cost(quick, "move_up") < BASE_ACTION_COST


def test_lower_dexterity_acts_slower_costs_more() -> None:
    slow = esper.create_entity(Attributes(dexterity=5))
    assert actor_speed(slow) < 1.0
    assert action_cost(slow, "move_up") > BASE_ACTION_COST


def test_speed_has_a_floor_so_cost_stays_positive() -> None:
    clumsy = esper.create_entity(Attributes(dexterity=-1000))
    assert actor_speed(clumsy) >= 0.1
    assert action_cost(clumsy, "move_up") >= 1


def test_action_weight_defaults_to_one() -> None:
    assert action_weight("anything") == 1.0
    assert action_weight(None) == 1.0


# --- End-to-end: the clock advances by the player's action cost -------------

def test_baseline_player_advances_clock_by_base_cost_per_action() -> None:
    esper.create_entity(WorldClock(turn=0))
    esper.create_entity(Player(), Attributes(dexterity=10))
    time_proc = TimeProcessor()

    for _ in range(5):
        time_proc.process("move_up")

    # Baseline cadence preserved: 5 actions == 5 * BASE_ACTION_COST of world time.
    assert world_clock().turn == 5 * BASE_ACTION_COST


def test_quicker_player_consumes_less_world_time_per_action() -> None:
    esper.create_entity(WorldClock(turn=0))
    esper.create_entity(Player(), Attributes(dexterity=25))

    TimeProcessor().process("move_up")

    # A nimble player's action still passes time, but less than a baseline turn.
    assert 0 < world_clock().turn < BASE_ACTION_COST


def test_baseline_needs_accrue_one_rate_per_baseline_turn() -> None:
    esper.create_entity(WorldClock(turn=0))
    esper.create_entity(Player(), Attributes(dexterity=10))
    ent = esper.create_entity(Needs(hunger=0.0, hunger_rate=1.0))

    time_proc = TimeProcessor()
    needs_proc = NeedsProcessor()
    turns = 4
    for _ in range(turns):
        time_proc.process("move_up")   # advance world time by the action's cost
        needs_proc.process("move_up")  # accrue needs for the elapsed span

    # Needs are time-based, but a baseline turn accrues exactly the old per-turn
    # amount, so the switch is invisible at baseline speed.
    assert esper.component_for_entity(ent, Needs).hunger == pytest.approx(turns * 1.0)
