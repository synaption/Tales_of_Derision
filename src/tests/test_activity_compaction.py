"""Compacted activities: an unwatched NPC settles a whole errand in one turn.

Outside the full-simulation box nobody can watch a creature spend its turns, only
the outcome. So an activity that is N region-turns of the same unobservable
repetition is settled in a single region-turn and the actor is billed the whole N.
It ends up where and how it would have, on the turn it would have -- what
disappears is N-1 rounds of re-ranking drives and re-scanning surroundings.

``ai`` offers two mechanisms and this pins both:

* **Billing the time** (``_charge_activity`` / ``_run_compacted``) -- the charge
  overdraws ``Actor.energy`` and the energy loop won't call the NPC again until
  the following region-turns have paid it off. Used for hauling, whose effects
  really do happen over those turns.
* **Settling the turns** (``Settled``) -- the effect is applied in closed form up
  front and per-turn systems skip the entity so they can't live those turns
  twice. Used for sleep, which is pure arithmetic.

Companion to ``test_travel_compaction``, which covers the third user, walking.
"""
from __future__ import annotations

from collections.abc import Callable
import itertools

import esper
import pytest

from action import BASE_ACTION_COST
from components import (
    Actor, Asleep, Blueprint, Home, Inventory, NPC, Needs, Player, Position,
    Resident, Settled, Tree, WorldClock,
)
from game_map import GameMap
from items import WOOD
from regions import region_at
from systems import (
    NeedsProcessor, NpcAiProcessor, TimeProcessor, _SLEEP_RECOVERY,
    create_construction_site, go_to_sleep, settle_sleep, sleep_turns_needed,
    wake_up, world_clock,
)
# ``ai`` is imported through ``systems`` (see its module docstring); importing it
# first would hit the cycle. These tuning constants live there.
from ai import _COMPACTED_ACTIVITY_TURNS, _COMPACTED_SLEEP_TURNS
import spatial

pytestmark = pytest.mark.unrendered


def _no_background_pump() -> Callable[[], float]:
    """A wall clock already past any deadline, so the processor's own background
    pump can never add region-turns behind the test's back."""
    counter = itertools.count()
    return lambda: next(counter) * 1000.0


def _world(width: int = 360, height: int = 60) -> GameMap:
    """Open floor wide enough to hold more than one region, so a creature can
    stand well beyond the full-sim box -- which is what every map in the real game
    looks like and no single-region test map ever does."""
    esper.clear_database()
    spatial.detach()
    game_map = GameMap(width, height)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            game_map.tiles[y][x] = game_map.FLOOR
    esper.create_entity(WorldClock(turn=0))
    esper.create_entity(Position(2, 30), Player())
    return game_map


def _processor(game_map: GameMap) -> NpcAiProcessor:
    return NpcAiProcessor(game_map, wall_clock=_no_background_pump())


def _tired(tiredness: float = 90.0) -> Needs:
    """A sleepy creature that also gets hungry and thirsty, so a compacted sleep
    has all three needs to settle rather than just the one."""
    return Needs(
        hunger=10.0, thirst=10.0, tiredness=tiredness,
        hunger_rate=0.5, thirst_rate=0.25, tiredness_rate=1.0,
    )


# --- the shared mechanism ---------------------------------------------------


def test_a_repeating_activity_runs_many_turns_and_is_billed_for_them() -> None:
    game_map = _world()
    pos = Position(200, 30)
    ent = esper.create_entity(pos, NPC())
    proc = _processor(game_map)
    proc._player_xy = (2, 30)

    turns = itertools.count()

    def five_turns_of_work() -> bool:
        return next(turns) < 5

    assert proc._run_compacted(ent, pos, _COMPACTED_ACTIVITY_TURNS, five_turns_of_work)

    # The region-turn it was decided in pays for the first; the other four are
    # charged here, leaving the actor exactly that far in arrears.
    assert esper.component_for_entity(ent, Actor).energy == -4 * BASE_ACTION_COST


def test_a_repeating_activity_stops_at_the_cap() -> None:
    """An activity settled in one decision is a decision made without noticing
    anything that happened during it, so a long one is broken into legs."""
    game_map = _world()
    pos = Position(200, 30)
    ent = esper.create_entity(pos, NPC())
    proc = _processor(game_map)
    proc._player_xy = (2, 30)

    done = itertools.count()
    proc._run_compacted(ent, pos, _COMPACTED_ACTIVITY_TURNS, lambda: next(done) or True)

    charged = -esper.component_for_entity(ent, Actor).energy / BASE_ACTION_COST
    assert charged == _COMPACTED_ACTIVITY_TURNS - 1


def test_a_repeating_activity_stops_on_the_threshold_of_the_watched_world() -> None:
    """Nothing is ever fast-forwarded in front of the player: an activity that
    carries its actor into the full-sim box hands the rest back to per-turn sim."""
    game_map = _world()
    pos = Position(200, 30)
    ent = esper.create_entity(pos, NPC())
    proc = _processor(game_map)
    proc._player_xy = (2, 30)

    def walk_home() -> bool:
        pos.x -= 60  # the second of these lands inside the box around the player
        return True

    proc._run_compacted(ent, pos, _COMPACTED_ACTIVITY_TURNS, walk_home)

    assert proc._far_from_player((200, 30)) and not proc._far_from_player((80, 30))
    assert pos.x == 80, "it should have stopped the moment it became watchable"
    assert esper.component_for_entity(ent, Actor).energy == -1 * BASE_ACTION_COST


# --- sleeping through the night ---------------------------------------------


def test_settling_a_sleep_matches_living_it_a_turn_at_a_time() -> None:
    """The closed form and the per-turn accrual must never drift apart -- this is
    the whole licence for replacing N turns of ``_accrue`` with arithmetic."""
    game_map = _world()
    needs_lived = _tired()
    needs_settled = _tired()
    turns = sleep_turns_needed(needs_lived)
    sleeper = esper.create_entity(Position(200, 30), NPC(), Asleep(), needs_lived)
    processor = NeedsProcessor(game_map)

    for _ in range(turns):
        processor._accrue(sleeper, needs_lived, scale=1.0, night=True)
    settle_sleep(needs_settled, turns)

    assert needs_settled.tiredness == pytest.approx(needs_lived.tiredness)
    assert needs_settled.hunger == pytest.approx(needs_lived.hunger)
    assert needs_settled.thirst == pytest.approx(needs_lived.thirst)
    assert needs_lived.tiredness == 0.0, "and it really is rested by then"


def test_an_unwatched_npc_sleeps_the_whole_night_in_one_turn() -> None:
    game_map = _world()
    needs = _tired()
    pos = Position(200, 30)
    ent = esper.create_entity(pos, NPC(), needs)
    proc = _processor(game_map)
    proc._player_xy = (2, 30)
    expected_turns = sleep_turns_needed(needs)

    assert proc._seek_sleep(ent, pos, needs, {}) is True

    assert esper.has_component(ent, Asleep)
    assert needs.tiredness == 0.0, "the whole rest was taken in this one turn"
    settled = esper.component_for_entity(ent, Settled)
    assert (settled.activity, settled.turns) == ("sleep", expected_turns)


def test_a_settled_sleeper_does_not_get_hungry_twice() -> None:
    """The receipt exists precisely so the per-turn systems don't re-live turns a
    compacted activity already applied."""
    game_map = _world()
    needs = _tired()
    pos = Position(200, 30)
    ent = esper.create_entity(pos, NPC(), needs)
    proc = _processor(game_map)
    proc._player_xy = (2, 30)
    proc._seek_sleep(ent, pos, needs, {})
    hunger_after_settling = needs.hunger

    needs_processor = NeedsProcessor(game_map)
    for _ in range(5):
        needs_processor.advance_region(region_at(game_map, pos.x, pos.y))

    assert needs.hunger == hunger_after_settling
    assert esper.component_for_entity(ent, Settled).turns == sleep_turns_needed(_tired()) - 5


def test_a_sleeper_wakes_on_exactly_the_turn_it_would_have() -> None:
    """Compaction moves work, not time: settling the night must not shorten it."""
    game_map = _world()
    needs = _tired()
    pos = Position(200, 30)
    ent = esper.create_entity(pos, NPC(), needs)
    region = region_at(game_map, pos.x, pos.y)
    proc = _processor(game_map)
    proc._player_xy = (2, 30)
    expected_turns = sleep_turns_needed(needs)

    proc._seek_sleep(ent, pos, needs, {})

    needs_processor = NeedsProcessor(game_map)
    woke_on = None
    for turn in range(1, _COMPACTED_SLEEP_TURNS + 2):
        needs_processor.advance_region(region)
        if not esper.has_component(ent, Asleep):
            woke_on = turn
            break

    assert woke_on == expected_turns, "same night, same dawn"
    assert not esper.has_component(ent, Settled), "the receipt is spent"


def test_a_sleeper_the_player_can_watch_sleeps_a_turn_at_a_time() -> None:
    game_map = _world()
    needs = _tired()
    pos = Position(20, 30)
    ent = esper.create_entity(pos, NPC(), needs)
    proc = _processor(game_map)
    proc._player_xy = (2, 30)
    assert not proc._far_from_player((20, 30)), "the test NPC must be on screen"

    proc._seek_sleep(ent, pos, needs, {})

    assert esper.has_component(ent, Asleep)
    assert not esper.has_component(ent, Settled)
    assert needs.tiredness == 90.0, "no arithmetic shortcut where it can be seen"


def test_waking_early_cancels_the_rest_of_a_settled_sleep() -> None:
    """Something shakes the sleeper awake mid-night. Leaving the receipt behind
    would make an awake creature invisible to the per-turn systems."""
    game_map = _world()
    needs = _tired()
    pos = Position(200, 30)
    ent = esper.create_entity(pos, NPC(), needs)
    proc = _processor(game_map)
    proc._player_xy = (2, 30)
    proc._seek_sleep(ent, pos, needs, {})

    wake_up(ent, game_map)

    assert not esper.has_component(ent, Settled)
    hunger_before = needs.hunger
    NeedsProcessor(game_map).advance_region(region_at(game_map, pos.x, pos.y))
    assert needs.hunger > hunger_before, "it is living its own turns again"


def test_a_homeless_creature_camps_and_settles_the_night_where_it_stands() -> None:
    game_map = _world()
    needs = _tired()
    pos = Position(200, 30)
    ent = esper.create_entity(pos, NPC(), needs)
    proc = _processor(game_map)
    proc._player_xy = (2, 30)

    proc._seek_sleep(ent, pos, needs, {})

    assert esper.component_for_entity(ent, Asleep).in_camp is True
    assert esper.has_component(ent, Settled)
    assert (pos.x, pos.y) == (200, 30), "it camped rather than walking anywhere"


def test_an_npc_at_home_settles_the_night_in_its_bed() -> None:
    game_map = _world()
    needs = _tired()
    pos = Position(200, 30)
    ent = esper.create_entity(pos, NPC(), needs, Home(200, 30))
    proc = _processor(game_map)
    proc._player_xy = (2, 30)

    proc._seek_sleep(ent, pos, needs, {})

    assert esper.component_for_entity(ent, Asleep).in_camp is False
    assert esper.component_for_entity(ent, Settled).turns == sleep_turns_needed(_tired())


# --- hauling wood to a blueprint --------------------------------------------


def _builder_and_site(game_map: GameMap, at: tuple[int, int], site: tuple[int, int]):
    """A resident with wood on its back, standing near a fresh construction site."""
    create_construction_site(game_map, site)
    pos = Position(*at)
    ent = esper.create_entity(
        pos, NPC(), Resident(),
        Needs(hunger=0.0, thirst=0.0, tiredness=0.0,
              hunger_rate=0.0, thirst_rate=0.0, tiredness_rate=0.0),
        Inventory(items=[WOOD] * 40),
    )
    return ent, pos


def _stocked_count() -> int:
    return sum(1 for _e, (bp,) in esper.get_components(Blueprint) if bp.stocked)


def test_an_unwatched_builder_hauls_a_whole_round_trip_in_one_turn() -> None:
    game_map = _world()
    ent, pos = _builder_and_site(game_map, at=(190, 30), site=(200, 30))
    proc = _processor(game_map)
    proc._player_xy = (2, 30)
    assert proc._far_from_player((190, 30))

    assert proc._work_blueprints(ent, pos, [], {}) is True

    # Ten tiles of walking and a delivery, settled in the turn it was decided in.
    assert _stocked_count() > 0
    charged = -esper.component_for_entity(ent, Actor).energy / BASE_ACTION_COST
    assert charged >= 10, "the walk to the site and the drop-off are both billed"


def test_a_compacted_haul_supplies_more_pieces_than_a_single_turn_would() -> None:
    """The point of compacting the errand: one decision covers the whole trip out
    to the site, where a single turn only gets a step closer to it."""
    game_map = _world()
    ent, pos = _builder_and_site(game_map, at=(190, 30), site=(200, 30))
    proc = _processor(game_map)
    proc._player_xy = (2, 30)  # unwatched: the whole trip runs now
    proc._work_blueprints(ent, pos, [], {})
    compacted = _stocked_count()

    esper.clear_database()
    spatial.detach()
    game_map = _world()
    ent, pos = _builder_and_site(game_map, at=(190, 30), site=(200, 30))
    proc = _processor(game_map)
    proc._player_xy = (189, 30)  # watched: one turn's work only
    proc._work_blueprints(ent, pos, [], {})

    assert _stocked_count() == 0, "a watched builder is still only walking there"
    assert compacted > 0


def test_a_watched_builder_still_hauls_a_turn_at_a_time() -> None:
    game_map = _world()
    ent, pos = _builder_and_site(game_map, at=(20, 30), site=(30, 30))
    proc = _processor(game_map)
    proc._player_xy = (2, 30)
    assert not proc._far_from_player((20, 30))

    proc._work_blueprints(ent, pos, [], {})

    assert not esper.has_component(ent, Actor) or (
        esper.component_for_entity(ent, Actor).energy == 0.0
    ), "on-screen work is charged by the turn loop, not billed ahead"


def test_a_compacted_haul_cannot_chop_a_felled_tree_twice() -> None:
    """The region's tree list is only rebuilt between region-turns. A haul that
    fells several trees inside one turn must not keep harvesting the stumps."""
    game_map = _world()
    ent, pos = _builder_and_site(game_map, at=(200, 30), site=(240, 30))
    inventory = esper.component_for_entity(ent, Inventory)
    inventory.items.clear()
    proc = _processor(game_map)
    proc._player_xy = (2, 30)
    # One tree beside the builder, with one log in it.
    tree_ent = esper.create_entity(Position(201, 30), Tree(wood=1))
    trees = [((201, 30), tree_ent)]

    for _ in range(5):
        proc._gather_wood(ent, pos, inventory, trees, {})

    assert not esper.entity_exists(tree_ent), "the tree really was felled"
    assert inventory.items.count(WOOD) == 1, "one log in the tree, one log taken"


def test_hauling_never_delivers_to_the_same_piece_twice() -> None:
    """``_haul_to_ghosts`` consumes ``unstocked`` as it supplies pieces, so a
    compacted round trip picks up where the last delivery left off instead of
    dropping a second log on a piece that already has one."""
    game_map = _world()
    ent, pos = _builder_and_site(game_map, at=(200, 30), site=(200, 30))
    proc = _processor(game_map)
    proc._player_xy = (2, 30)
    ghosts = proc._reachable_ghosts(pos)
    unstocked = {gxy: g_ent for g_ent, gxy, bp in ghosts if not bp.stocked}
    wood_before = esper.component_for_entity(ent, Inventory).items.count(WOOD)
    pieces = len(unstocked)

    while proc._haul_to_ghosts(ent, pos, unstocked, [], {}):
        pass

    wood_spent = wood_before - esper.component_for_entity(ent, Inventory).items.count(WOOD)
    assert wood_spent == pieces, "one log per piece, never two"
    assert not unstocked


# --- the player's own sleep -------------------------------------------------
#
# The player used to be the one sleeper in the world that lived its night a turn
# at a time: several hundred `esper.process` calls, each redrawing the frame and
# re-running every system, to apply arithmetic. It now settles the rest exactly
# as an NPC does. These pin that the shortcut costs the player the same night.


def _sleeping_player(tiredness: float = 90.0) -> tuple[GameMap, int, Needs]:
    """A tired player in a world with a clock and a needs processor -- the two
    things a sleep has to move correctly."""
    game_map = _world()
    player_ent = next(ent for ent, _ in esper.get_component(Player))
    needs = _tired(tiredness)
    esper.add_component(player_ent, needs)
    esper.add_processor(TimeProcessor(), priority=2)
    esper.add_processor(NeedsProcessor(game_map), priority=0)
    return game_map, player_ent, needs


def test_the_player_sleeps_the_whole_night_in_one_turn() -> None:
    """The point of the change: one turn of play, a whole night of world time."""
    import ui

    game_map, player_ent, needs = _sleeping_player()
    expected_turns = sleep_turns_needed(needs)
    started_at = world_clock().turn

    ui._sleep_player(None, in_camp=True, game_map=game_map)

    assert not esper.has_component(player_ent, Asleep), "and the player is up again"
    assert needs.tiredness == 0.0, "rested"
    assert world_clock().turn == started_at + expected_turns * BASE_ACTION_COST, (
        "the night really passed, in world time"
    )


def test_the_players_compacted_sleep_costs_exactly_what_living_it_would() -> None:
    """The equivalence that licenses the shortcut: same clock, same needs as the
    turn-at-a-time loop it replaces."""
    import ui

    # Live it: one turn per call, waking when the tiredness is paid off.
    game_map, player_ent, lived = _sleeping_player()
    started_at = world_clock().turn
    go_to_sleep(player_ent, in_camp=True, game_map=game_map)
    turns = 0
    while esper.has_component(player_ent, Asleep) and turns < 400:
        esper.process("wait")
        turns += 1
    lived_clock = world_clock().turn - started_at
    lived_needs = (lived.tiredness, lived.hunger, lived.thirst)

    # Settle it.
    game_map, player_ent, settled = _sleeping_player()
    started_at = world_clock().turn
    ui._sleep_player(None, in_camp=True, game_map=game_map)
    settled_clock = world_clock().turn - started_at

    assert settled_clock == lived_clock, "the same amount of world time passed"
    assert settled.tiredness == pytest.approx(lived_needs[0])
    assert settled.hunger == pytest.approx(lived_needs[1])
    assert settled.thirst == pytest.approx(lived_needs[2])


def test_waking_does_not_charge_the_night_a_second_time() -> None:
    """The clock leaps a whole night, so the turn after waking must charge only
    itself -- otherwise the player wakes rested and immediately starves."""
    import ui

    game_map, player_ent, needs = _sleeping_player()
    ui._sleep_player(None, in_camp=True, game_map=game_map)
    hunger_on_waking = needs.hunger

    esper.process("wait")

    assert needs.hunger == pytest.approx(hunger_on_waking + needs.hunger_rate), (
        "one turn's hunger for one turn awake"
    )
