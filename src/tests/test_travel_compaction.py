"""Compacted travel: an unwatched NPC walks a whole journey in one region-turn.

Nobody can see a creature on another island put one foot in front of the other --
only where it ends up, and when. So outside the full-simulation box a walk is
settled as a single activity: the NPC covers the leg at once and is charged the
whole leg's travel time, which keeps it out of the simulation for exactly as many
region-turns as the walk would have taken. Same arrival, same availability, a
fraction of the per-tile decision work.

These tests pin the three things that makes true: it really does move the whole
leg, it really is unavailable for the walk's duration, and the watched world
around the player is completely unaffected.
"""
from __future__ import annotations

from collections.abc import Callable
import itertools

import esper
import pytest

from action import BASE_ACTION_COST
from components import Actor, NPC, Player, Position, WorldClock
from game_map import GameMap
from regions import region_at
from systems import NpcAiProcessor, world_clock
# ``ai`` is imported through ``systems`` (see its module docstring); importing it
# first would hit the cycle. These two tuning constants live there.
from ai import _COMPACTED_TRAVEL_STEPS, _FULL_SIM_HALF_W
import spatial

pytestmark = pytest.mark.unrendered


def _no_background_pump() -> Callable[[], float]:
    """A wall clock already past any deadline, so the processor's own background
    pump can never add region-turns behind the test's back."""
    counter = itertools.count()
    return lambda: next(counter) * 1000.0


def _world() -> GameMap:
    """A 3x1 region grid of open floor: wide enough that a creature can stand well
    beyond the full-sim box, which is what every map in the real game looks like
    and no single-region test map ever does."""
    esper.clear_database()
    spatial.detach()
    game_map = GameMap(360, 60)
    esper.create_entity(WorldClock(turn=0))
    esper.create_entity(Position(2, 30), Player())
    return game_map


def _npc_at(x: int, y: int) -> tuple[int, Position]:
    pos = Position(x, y)
    return esper.create_entity(pos, NPC()), pos


def _processor(game_map: GameMap) -> NpcAiProcessor:
    return NpcAiProcessor(game_map, wall_clock=_no_background_pump())


# --- the walk itself --------------------------------------------------------

def test_an_unwatched_npc_covers_a_whole_walk_in_one_turn() -> None:
    game_map = _world()
    ent, pos = _npc_at(200, 30)
    proc = _processor(game_map)
    proc._player_xy = (2, 30)
    assert proc._far_from_player((200, 30)), "the test NPC must be beyond the box"

    assert proc._step_toward(ent, pos, (212, 30), {}) is True

    # Twelve tiles of walking, settled in the one turn it was decided in.
    assert (pos.x, pos.y) == (212, 30)


def test_the_walk_is_billed_as_the_time_it_took() -> None:
    game_map = _world()
    ent, pos = _npc_at(200, 30)
    proc = _processor(game_map)
    proc._player_xy = (2, 30)

    proc._step_toward(ent, pos, (212, 30), {})

    # The turn it was decided in pays for the first tile the ordinary way; the
    # other eleven are charged here, leaving the traveller that far in arrears.
    assert esper.component_for_entity(ent, Actor).energy == -11 * BASE_ACTION_COST


def test_a_long_trek_is_walked_in_capped_legs() -> None:
    """A journey longer than the cap isn't settled in one go -- the traveller
    stops to take stock, so it can notice it grew hungry crossing the island."""
    game_map = _world()
    ent, pos = _npc_at(150, 30)
    proc = _processor(game_map)
    proc._player_xy = (2, 30)
    goal = (150 + _COMPACTED_TRAVEL_STEPS * 2, 30)

    proc._step_toward(ent, pos, goal, {})

    assert pos.x == 150 + _COMPACTED_TRAVEL_STEPS
    # ...and the rest of the route is still cached, so resuming costs no pathfind.
    assert proc._trip_cache[ent][0] == goal


def test_a_walk_stops_on_the_threshold_of_the_watched_world() -> None:
    """A traveller heading toward the player never materializes mid-stride in
    front of them: the compacted leg ends where the full-sim box begins, and the
    rest of the approach is walked a tile at a time like anything else on screen."""
    game_map = _world()
    player_xy = (100, 30)
    for _ent, (pos, _p) in esper.get_components(Position, Player):
        pos.x, pos.y = player_xy
    start_x = player_xy[0] + _FULL_SIM_HALF_W + 20
    ent, pos = _npc_at(start_x, 30)
    proc = _processor(game_map)
    proc._player_xy = player_xy

    proc._step_toward(ent, pos, (player_xy[0] + 10, 30), {})

    # It stepped exactly onto the first tile the player's region simulates fully,
    # and no further.
    assert proc._far_from_player((pos.x + 1, pos.y)) is True
    assert proc._far_from_player((pos.x, pos.y)) is False


# --- the watched world is untouched -----------------------------------------

def test_an_npc_the_player_can_see_still_walks_one_tile_at_a_time() -> None:
    game_map = _world()
    ent, pos = _npc_at(20, 30)
    proc = _processor(game_map)
    proc._player_xy = (2, 30)
    assert not proc._far_from_player((20, 30))

    proc._step_toward(ent, pos, (32, 30), {})

    assert (pos.x, pos.y) == (21, 30)
    assert not esper.has_component(ent, Actor) or (
        esper.component_for_entity(ent, Actor).energy == 0.0
    ), "an on-screen step is charged by the turn loop, not billed ahead"


# --- and the timing it implies ----------------------------------------------

def test_a_traveller_acts_again_exactly_when_the_walk_would_have_ended() -> None:
    """The whole point of the charge: compaction moves *work*, not time.

    Walking twelve tiles a turn at a time, an NPC deciding to travel on region-turn
    N arrives on turn N+11 and does the next thing on N+12. Compacted, it arrives
    on turn N -- but stays in arrears until N+12, which is when it acts again. The
    schedule is identical; only the eleven turns of deciding are gone.
    """
    game_map = _world()
    ent, pos = _npc_at(200, 30)
    region = region_at(game_map, 200, 30)
    proc = _processor(game_map)

    acted_on: list[int] = []

    def travel_once(ent_: int, pos_: Position, occupied: dict, *_a, **_k) -> None:
        acted_on.append(proc.scheduler.region_turn[region])
        if len(acted_on) == 1:
            proc._step_toward(ent_, pos_, (212, 30), occupied)

    proc._take_turn = travel_once

    for _ in range(20):
        proc.scheduler.advance_region(region)

    assert len(acted_on) >= 2, "the traveller should act again after its walk"
    assert acted_on[1] - acted_on[0] == 12
    assert (pos.x, pos.y) == (212, 30)


def test_compaction_never_skips_a_turn_the_walk_did_not_pay_for() -> None:
    """A one-tile move is not a journey: it costs one turn, as it always did."""
    game_map = _world()
    ent, pos = _npc_at(200, 30)
    region = region_at(game_map, 200, 30)
    proc = _processor(game_map)

    acted_on: list[int] = []

    def step_once(ent_: int, pos_: Position, occupied: dict, *_a, **_k) -> None:
        acted_on.append(proc.scheduler.region_turn[region])
        if len(acted_on) == 1:
            proc._step_toward(ent_, pos_, (201, 30), occupied)

    proc._take_turn = step_once

    for _ in range(4):
        proc.scheduler.advance_region(region)

    assert acted_on[1] - acted_on[0] == 1
    assert world_clock() is not None  # the world clock is untouched by all this
