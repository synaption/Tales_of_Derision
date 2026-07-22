"""Unrendered tests for the region grid + RegionScheduler (regions.py), and
the multi-region behaviour it gives NpcAiProcessor/FishAiProcessor once a map
is bigger than a single 120x60 simulation region -- every current gameplay map
in src/tests/*.py fits in one region, so those existing tests never exercise
this at all."""
from __future__ import annotations

from collections.abc import Callable
import itertools

import esper
import pytest

from components import Diet, NPC, Needs, Player, Position, Tree, WorldClock
from game_map import GameMap
from regions import RegionScheduler, all_region_ids, in_region_with_margin, region_at, region_grid_size
from systems import FishAiProcessor, NpcAiProcessor, world_clock

pytestmark = pytest.mark.unrendered


def _no_background_pump() -> Callable[[], float]:
    """A fake wall clock whose very first ``pump_background`` deadline is
    already expired by the time it's checked, so a processor's *own*
    background pump can never sneak in extra region-catch-up work behind a
    test's back -- these tests drive ``catch_up_region``/``catch_up_all``
    explicitly instead, to pin down exactly what should and shouldn't be
    caught up yet."""
    counter = itertools.count()
    return lambda: next(counter) * 1000.0


# --- grid math ---------------------------------------------------------


def test_region_grid_collapses_to_one_region_on_a_small_map() -> None:
    game_map = GameMap(30, 18)
    assert region_grid_size(game_map) == (30, 18, 1, 1)
    assert all_region_ids(game_map) == [(0, 0)]
    assert region_at(game_map, 29, 17) == (0, 0)


def test_region_grid_splits_a_wide_map_into_a_2x1_grid() -> None:
    game_map = GameMap(240, 60)
    assert region_grid_size(game_map) == (120, 60, 2, 1)
    assert region_at(game_map, 10, 30) == (0, 0)
    assert region_at(game_map, 200, 30) == (1, 0)


def test_in_region_with_margin_covers_the_region_and_a_border_band() -> None:
    game_map = GameMap(240, 60)
    # Just past the seam, within the margin -- still visible from (0, 0).
    assert in_region_with_margin(game_map, (0, 0), 125, 30, margin=8) is True
    # Too far past the seam to be within the margin.
    assert in_region_with_margin(game_map, (0, 0), 135, 30, margin=8) is False


# --- RegionScheduler -----------------------------------------------------


def test_advance_region_runs_steps_in_order_and_increments_the_cursor() -> None:
    game_map = GameMap(30, 18)
    scheduler = RegionScheduler(game_map, 0)
    calls: list[tuple[str, tuple[int, int]]] = []
    scheduler.register("a", lambda region_id: calls.append(("a", region_id)))
    scheduler.register("b", lambda region_id: calls.append(("b", region_id)))

    scheduler.advance_region((0, 0))

    assert calls == [("a", (0, 0)), ("b", (0, 0))]
    assert scheduler.region_turn[(0, 0)] == 1


def test_catch_up_region_replays_every_missing_turn_in_order() -> None:
    game_map = GameMap(240, 60)
    scheduler = RegionScheduler(game_map, 0)
    seen: list[int] = []
    scheduler.register("record", lambda region_id: seen.append(scheduler.region_turn[region_id]))

    scheduler.catch_up_region((1, 0), target_turn=4)

    # Replayed turn by turn (0, 1, 2, 3) -- never skipped or batched.
    assert seen == [0, 1, 2, 3]
    assert scheduler.region_turn[(1, 0)] == 4


def test_catch_up_all_brings_every_region_up_to_date() -> None:
    game_map = GameMap(240, 60)
    scheduler = RegionScheduler(game_map, 0)
    scheduler.register("noop", lambda region_id: None)

    scheduler.catch_up_all(target_turn=3)

    assert scheduler.region_turn == {(0, 0): 3, (1, 0): 3}


def test_pump_background_prefers_the_nearest_lagging_region_over_the_stalest() -> None:
    game_map = GameMap(480, 60)  # a 4x1 grid: (0, 0) .. (3, 0)
    scheduler = RegionScheduler(game_map, 5)
    # (2, 0) is only a little behind but close to the player; (3, 0) is far
    # more behind (never touched) but twice as far away. Distance must win.
    scheduler.region_turn[(2, 0)] = 3
    scheduler.region_turn[(3, 0)] = 0
    advanced: list[tuple[int, int]] = []
    scheduler.register("record", lambda region_id: advanced.append(region_id))

    # Fake wall clock: lets exactly one pump iteration run before "time" is up.
    times = iter([0.0, 0.0, 100.0])
    scheduler.pump_background(
        budget_seconds=1.0,
        player_region=(1, 0),
        target_turn=5,
        wall_clock=lambda: next(times),
    )

    assert advanced == [(2, 0)]


def test_next_turn_for_advances_by_at_least_one_without_a_moving_clock() -> None:
    """A processor constructed directly in a unit test (no ``TimeProcessor``
    advancing a real ``WorldClock``) must still tick forward by exactly one
    turn per call, matching an unpartitioned processor -- see
    ``NpcAiProcessor``/``FishAiProcessor``'s single-region test coverage."""
    game_map = GameMap(30, 18)
    scheduler = RegionScheduler(game_map, 0)
    region_id = (0, 0)

    assert scheduler.next_turn_for(region_id, observed_turn=0) == 1
    scheduler.advance_region(region_id)
    assert scheduler.next_turn_for(region_id, observed_turn=0) == 2

    # A genuinely advancing clock wins once it's ahead of the region's cursor.
    assert scheduler.next_turn_for(region_id, observed_turn=50) == 50


# --- NpcAiProcessor on a multi-region map --------------------------------
# GameMap(240, 60) is a plain walled room (too short to become the ocean
# world) split into a 2x1 grid of 120x60 regions: (0, 0) covers x in [0, 120),
# (1, 0) covers x in [120, 240).


def test_npc_outside_the_players_region_does_not_act_until_caught_up() -> None:
    game_map = GameMap(240, 60)
    esper.create_entity(Position(10, 30), Player())
    npc = esper.create_entity(Position(200, 30), NPC(), Needs(hunger=90.0), Diet("herbivore"))
    esper.create_entity(Position(201, 30), Tree(wood=5))

    processor = NpcAiProcessor(game_map, wall_clock=_no_background_pump())
    processor.process("wait")

    pos = esper.component_for_entity(npc, Position)
    needs = esper.component_for_entity(npc, Needs)
    assert (pos.x, pos.y) == (200, 30)  # hasn't acted -- its region never ran
    assert needs.hunger == 90.0

    far_region = processor.scheduler.region_at(200, 30)
    processor.scheduler.catch_up_region(far_region, processor.scheduler.region_turn[far_region] + 1)

    needs = esper.component_for_entity(npc, Needs)
    assert needs.hunger < 90.0  # grazed once its own region was actually run


def test_entering_a_new_region_replays_every_missed_turn_at_once() -> None:
    esper.create_entity(WorldClock(turn=0, day_length=240))
    game_map = GameMap(240, 60)
    player = esper.create_entity(Position(10, 30), Player())

    processor = NpcAiProcessor(game_map, wall_clock=_no_background_pump())
    region_a = processor.scheduler.region_at(10, 30)
    region_b = processor.scheduler.region_at(200, 30)

    clock = world_clock()
    for turn in range(1, 6):
        clock.turn = turn
        processor.process("wait")
    assert processor.scheduler.region_turn[region_a] == 5
    assert processor.scheduler.region_turn[region_b] == 0  # never entered yet

    # The player walks into region B -- it must come up fully to date in this
    # one call, not a bounded burst.
    player_pos = esper.component_for_entity(player, Position)
    player_pos.x, player_pos.y = 200, 30
    clock.turn = 6
    processor.process("wait")

    assert processor.scheduler.region_turn[region_b] == 6


def test_npc_near_a_region_seam_still_reaches_a_resource_just_across_it() -> None:
    game_map = GameMap(240, 60)
    esper.create_entity(Position(10, 30), Player())  # same region as the NPC below
    npc = esper.create_entity(Position(119, 30), NPC(), Needs(hunger=90.0), Diet("herbivore"))
    esper.create_entity(Position(125, 30), Tree(wood=5))  # 6 tiles into the next region

    processor = NpcAiProcessor(game_map)
    processor.process("wait")

    pos = esper.component_for_entity(npc, Position)
    # Stepped toward the tree despite it sitting in a different simulation
    # region -- without the border margin this tree would be invisible and
    # the NPC would never move.
    assert pos.x > 119


def test_sleep_catches_up_every_region_not_just_the_players() -> None:
    """Mirrors what ``main._sleep_player`` does after waking: call
    ``catch_up_all`` on every region-aware processor's scheduler."""
    esper.create_entity(WorldClock(turn=0, day_length=240))
    game_map = GameMap(240, 60)
    esper.create_entity(Position(10, 30), Player())
    npc = esper.create_entity(Position(200, 30), NPC(), Needs(hunger=90.0), Diet("herbivore"))
    esper.create_entity(Position(201, 30), Tree(wood=5))

    npc_ai = NpcAiProcessor(game_map, wall_clock=_no_background_pump())
    fish_ai = FishAiProcessor(game_map, clock=_no_background_pump())
    esper.add_processor(npc_ai, priority=0)
    esper.add_processor(fish_ai, priority=0)

    # The player stays in region A for ten turns; region B (the NPC's) racks
    # up ten turns of debt.
    clock = world_clock()
    for turn in range(1, 11):
        clock.turn = turn
        npc_ai.process("wait")

    far_region = npc_ai.scheduler.region_at(200, 30)
    assert npc_ai.scheduler.region_turn[far_region] == 0
    needs = esper.component_for_entity(npc, Needs)
    assert needs.hunger == 90.0  # not simulated yet

    for processor in (esper.get_processor(NpcAiProcessor), esper.get_processor(FishAiProcessor)):
        processor.scheduler.catch_up_all(clock.turn)

    assert npc_ai.scheduler.region_turn[far_region] == 10
    needs = esper.component_for_entity(npc, Needs)
    assert needs.hunger < 90.0  # grazed once its region was actually simulated
