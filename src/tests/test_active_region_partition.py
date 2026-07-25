"""The active/inactive split, enforced.

The rule: ``esper.process(action)`` simulates the map tile the player is standing
in and nothing else. Everywhere else moves only in the places allowed to move it
-- the idle pump, region-entry catch-up, and sleep. These tests hold the turn path
to that, so a future system that quietly scans the world (the thing that used to
make a hundred islands cost a hundred times a keypress) fails here rather than in
a profile months later.
"""
from __future__ import annotations

import esper
import pytest

from components import Needs, NPC, Player, Position
from config import ACTIVE_REGION_CATCHUP_STEPS_PER_INPUT
from game_map import GameMap, archipelago_size
from regions import region_at
from rng import set_world_rng
from systems import (
    FishAiProcessor, HousingProcessor, MovementProcessor, NeedsProcessor,
    NpcAiProcessor, ReproductionProcessor, TimeProcessor, TreeGrowthProcessor,
)
from content.effects import EffectsProcessor
from worldgen import _setup_world
import spatial

pytestmark = pytest.mark.headless_renderer


def _world(grid: int = 2) -> GameMap:
    esper.clear_database()
    spatial.detach()
    set_world_rng(0x7A1E5)
    width, height = archipelago_size(grid)
    game_map = GameMap(width, height, layout="islands")
    _setup_world(game_map, Position(width // 2, height // 2))
    esper.add_processor(TimeProcessor(), priority=2)
    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(HousingProcessor(game_map, live_region_only=True), priority=0)
    npc_ai = NpcAiProcessor(
        game_map, max_entry_catchup_advances=ACTIVE_REGION_CATCHUP_STEPS_PER_INPUT
    )
    esper.add_processor(npc_ai, priority=0)
    esper.add_processor(
        FishAiProcessor(game_map, max_entry_catchup_advances=ACTIVE_REGION_CATCHUP_STEPS_PER_INPUT),
        priority=0,
    )
    needs = NeedsProcessor(game_map)
    esper.add_processor(needs, priority=0)
    effects = EffectsProcessor(game_map)
    esper.add_processor(effects, priority=0)
    esper.add_processor(TreeGrowthProcessor(game_map), priority=0)
    esper.add_processor(ReproductionProcessor(game_map), priority=0)
    # Wired the way game._register_processors wires it: needs and effects are steps
    # on the region scheduler, not per-turn world passes.
    needs.register_region_step(npc_ai.scheduler)
    effects.register_region_step(npc_ai.scheduler)
    return game_map


def _player_region(game_map: GameMap):
    for _ent, (pos, _p) in esper.get_components(Position, Player):
        return region_at(game_map, pos.x, pos.y)
    raise AssertionError("the world should have a player")


def _npc_positions(game_map: GameMap) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    """entity -> (its region, its tile)."""
    return {
        ent: (region_at(game_map, pos.x, pos.y), (pos.x, pos.y))
        for ent, (pos, _npc) in esper.get_components(Position, NPC)
    }


def test_a_turn_moves_the_players_region_and_leaves_the_rest_still() -> None:
    game_map = _world()
    active = _player_region(game_map)
    before = _npc_positions(game_map)

    for _ in range(25):
        esper.process("wait")

    after = _npc_positions(game_map)
    moved_here = 0
    for ent, (region, tile) in before.items():
        if ent not in after:
            continue  # died; deaths are a simulation result, not a position
        new_tile = after[ent][1]
        if region == active:
            moved_here += tile != new_tile
        else:
            assert tile == new_tile, (
                f"entity {ent} in sleeping region {region} moved during a turn"
            )
    assert moved_here, "the player's own region should be simulating"


def test_a_turn_only_ticks_needs_in_the_players_region() -> None:
    game_map = _world()
    active = _player_region(game_map)
    before = {
        ent: needs.hunger
        for ent, (needs, pos) in esper.get_components(Needs, Position)
    }
    regions = {
        ent: region_at(game_map, pos.x, pos.y)
        for ent, (_needs, pos) in esper.get_components(Needs, Position)
    }

    for _ in range(25):
        esper.process("wait")

    ticked_here = 0
    for ent, (needs,) in esper.get_components(Needs):
        if ent not in before:
            continue
        if regions[ent] == active:
            ticked_here += needs.hunger != before[ent]
        else:
            assert needs.hunger == before[ent], (
                f"entity {ent} in a sleeping region got hungrier during a turn"
            )
    assert ticked_here, "the player's own region should be getting hungry"


def test_only_the_active_regions_clock_advances_on_a_turn() -> None:
    """The turn path advances one region's simulation cursor: the player's."""
    game_map = _world()
    active = _player_region(game_map)
    npc_ai = esper.get_processor(NpcAiProcessor)
    before = dict(npc_ai.scheduler.region_turn)

    for _ in range(10):
        esper.process("wait")

    after = npc_ai.scheduler.region_turn
    assert after[active] > before[active]
    for region_id, turn in before.items():
        if region_id != active:
            assert after[region_id] == turn, f"{region_id} simulated during a turn"


def test_a_caught_up_region_gets_as_hungry_as_the_turns_it_missed() -> None:
    """Needs belong to a region's turn, not the world's.

    A sleeping region's hunger doesn't tick during a keypress -- but it isn't lost
    either. When the region is finally simulated (idle pump, region entry, sleep)
    it accrues one turn of hunger for every turn it owed, so its people wake up as
    hungry as they should be rather than as hungry as they were when you left.
    """
    game_map = _world()
    active = _player_region(game_map)
    for _ in range(30):
        esper.process("wait")

    asleep_elsewhere = {
        ent: needs.hunger
        for ent, (needs, pos) in esper.get_components(Needs, Position)
        if region_at(game_map, pos.x, pos.y) != active
    }
    assert asleep_elsewhere, "the world should have people on other islands"

    npc_ai = esper.get_processor(NpcAiProcessor)
    target = max(npc_ai.scheduler.region_turn.values())
    for region_id, turn in list(npc_ai.scheduler.region_turn.items()):
        if turn < target:
            npc_ai.scheduler.catch_up_region(region_id, target)

    got_hungry = sum(
        1
        for ent, (needs,) in esper.get_components(Needs)
        if ent in asleep_elsewhere and needs.hunger > asleep_elsewhere[ent]
    )
    assert got_hungry, "catching a region up should charge it the hunger it missed"


def test_catch_up_is_where_a_sleeping_region_advances() -> None:
    """...and the debt it built up is payable off the turn path -- which is what
    the idle pump, region entry and sleep all call."""
    game_map = _world()
    active = _player_region(game_map)
    for _ in range(10):
        esper.process("wait")
    npc_ai = esper.get_processor(NpcAiProcessor)
    target = max(npc_ai.scheduler.region_turn.values())
    sleeping = [r for r, t in npc_ai.scheduler.region_turn.items() if r != active and t < target]
    assert sleeping, "sleeping regions should have fallen behind"

    for region_id in sleeping:
        npc_ai.scheduler.catch_up_region(region_id, target)

    assert all(npc_ai.scheduler.region_turn[r] == target for r in sleeping)
