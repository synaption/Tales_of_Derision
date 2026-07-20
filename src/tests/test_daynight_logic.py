"""Unrendered tests for the day/night cycle and the tiredness/sleep loop."""
from __future__ import annotations

import esper
import pytest

from components import (
    Asleep,
    BlocksMovement,
    Camp,
    Home,
    Name,
    NPC,
    Needs,
    Player,
    Position,
    Tree,
    WorldClock,
)
from game_map import GameMap
from systems import (
    NeedsProcessor,
    NpcAiProcessor,
    TimeProcessor,
    go_to_sleep,
    is_night,
    night_overlay_alpha,
    time_phase,
    wake_up,
    world_clock,
    _pull_turn_events,
)

pytestmark = pytest.mark.unrendered


# --- Clock / phases --------------------------------------------------------


def test_time_processor_advances_clock_only_on_turn_actions() -> None:
    esper.create_entity(WorldClock(turn=0, day_length=100))
    processor = TimeProcessor()

    processor.process("move_up")
    assert world_clock().turn == 1

    # A menu refresh (None) or a non-turn action must not advance time.
    processor.process(None)
    processor.process("open_inventory")
    assert world_clock().turn == 1


def test_phase_and_night_track_the_day_fraction() -> None:
    clock = WorldClock(turn=0, day_length=100)

    clock.turn = 30  # 0.30 -> daytime
    assert time_phase(clock) == "Day"
    assert not is_night(clock)

    clock.turn = 80  # 0.80 -> night
    assert time_phase(clock) == "Night"
    assert is_night(clock)


def test_missing_clock_reads_as_plain_day() -> None:
    assert time_phase(None) == "Day"
    assert not is_night(None)
    assert night_overlay_alpha(None) == 0


def test_night_overlay_is_darker_than_dusk() -> None:
    night = WorldClock(turn=90, day_length=100)  # 0.90 -> Night
    dusk = WorldClock(turn=60, day_length=100)  # 0.60 -> Dusk
    day = WorldClock(turn=30, day_length=100)  # 0.30 -> Day
    assert night_overlay_alpha(night) > night_overlay_alpha(dusk) > night_overlay_alpha(day)
    assert night_overlay_alpha(day) == 0


# --- Tiredness -------------------------------------------------------------


def test_tiredness_rises_faster_at_night() -> None:
    clock = WorldClock(turn=30, day_length=100)  # daytime
    esper.create_entity(clock)
    processor = NeedsProcessor()

    day_ent = esper.create_entity(Needs(tiredness=0.0))
    processor.process("move_up")
    day_gain = esper.component_for_entity(day_ent, Needs).tiredness

    clock.turn = 80  # now night
    night_ent = esper.create_entity(Needs(tiredness=0.0))
    processor.process("move_up")
    night_gain = esper.component_for_entity(night_ent, Needs).tiredness

    assert night_gain > day_gain > 0.0


def test_tiredness_warns_only_for_the_player() -> None:
    esper.create_entity(Player(), Needs(tiredness=49.7))  # crosses 50% this turn
    esper.create_entity(NPC(), Needs(tiredness=49.7))  # also crosses, but silent

    _pull_turn_events()
    NeedsProcessor().process("move_up")
    events = _pull_turn_events()

    assert sum("tired" in text.lower() for text in events) == 1


# --- Sleeping --------------------------------------------------------------


def test_sleeping_recovers_tiredness_and_wakes_when_rested() -> None:
    player = esper.create_entity(Player(), Needs(tiredness=5.0, hunger=0.0), Asleep())
    processor = NeedsProcessor()

    processor.process("wait")  # 5 -> 2, still asleep
    needs = esper.component_for_entity(player, Needs)
    assert needs.tiredness == pytest.approx(2.0)
    assert esper.has_component(player, Asleep)
    assert needs.hunger > 0.0  # hunger still creeps up while asleep

    processor.process("wait")  # 2 -> 0, wakes
    assert esper.component_for_entity(player, Needs).tiredness == 0.0
    assert not esper.has_component(player, Asleep)


def test_go_to_sleep_pitches_a_camp_and_waking_breaks_it() -> None:
    ent = esper.create_entity(Position(8, 8), Name("Wanderer"))

    go_to_sleep(ent, in_camp=True)
    assert esper.component_for_entity(ent, Asleep).in_camp is True
    camps = [(pos.x, pos.y) for _e, (pos, _c) in esper.get_components(Position, Camp)]
    assert camps == [(8, 8)]

    wake_up(ent)
    assert not esper.has_component(ent, Asleep)
    assert list(esper.get_components(Position, Camp)) == []


def test_sleeping_at_home_does_not_pitch_a_camp() -> None:
    ent = esper.create_entity(Position(3, 3))
    go_to_sleep(ent, in_camp=False)
    assert esper.has_component(ent, Asleep)
    assert list(esper.get_components(Position, Camp)) == []


# --- NPC sleep AI ----------------------------------------------------------


def test_tired_npc_walks_toward_its_home() -> None:
    game_map = GameMap(24, 14)
    processor = NpcAiProcessor(game_map)
    npc = esper.create_entity(
        Position(10, 6), NPC(), Needs(tiredness=80.0), Home(15, 6), BlocksMovement(), Name("Villager")
    )

    processor.process("wait")
    pos = esper.component_for_entity(npc, Position)
    assert (pos.x, pos.y) == (11, 6)  # one step closer to home, not yet asleep
    assert not esper.has_component(npc, Asleep)


def test_tired_npc_sleeps_when_standing_on_its_home() -> None:
    game_map = GameMap(24, 14)
    processor = NpcAiProcessor(game_map)
    npc = esper.create_entity(
        Position(15, 6), NPC(), Needs(tiredness=80.0), Home(15, 6), BlocksMovement(), Name("Villager")
    )

    processor.process("wait")
    assert esper.component_for_entity(npc, Asleep).in_camp is False


def test_homeless_tired_npc_camps_where_it_stands() -> None:
    game_map = GameMap(24, 14)
    processor = NpcAiProcessor(game_map)
    npc = esper.create_entity(
        Position(10, 6), NPC(), Needs(tiredness=80.0), BlocksMovement(), Name("Deer")
    )

    processor.process("wait")
    assert esper.component_for_entity(npc, Asleep).in_camp is True
    camps = [(pos.x, pos.y) for _e, (pos, _c) in esper.get_components(Position, Camp)]
    assert camps == [(10, 6)]


def test_asleep_npc_skips_its_turn_even_when_hungry() -> None:
    game_map = GameMap(24, 14)
    processor = NpcAiProcessor(game_map)
    esper.create_entity(Position(11, 6), Tree(wood=3), BlocksMovement())
    npc = esper.create_entity(
        Position(10, 6), NPC(), Asleep(), Home(20, 6),
        Needs(hunger=90.0, tiredness=90.0), BlocksMovement(), Name("Villager"),
    )

    processor.process("wait")
    pos = esper.component_for_entity(npc, Position)
    assert (pos.x, pos.y) == (10, 6)  # did not move to forage or seek home
    assert esper.has_component(npc, Asleep)


def test_full_stack_sleep_cycle_advances_time_and_wakes_the_player() -> None:
    # Mirrors the turn loop's fast-forward: the clock ticks and needs recover
    # each processed turn until the sleeping player wakes rested.
    esper.create_entity(WorldClock(turn=0, day_length=240))
    esper.add_processor(TimeProcessor(), priority=2)
    esper.add_processor(NeedsProcessor(), priority=0)
    player = esper.create_entity(Player(), Needs(tiredness=30.0), Asleep())

    for _ in range(200):
        esper.process("wait")
        if not esper.has_component(player, Asleep):
            break

    assert not esper.has_component(player, Asleep)
    assert esper.component_for_entity(player, Needs).tiredness == 0.0
    assert world_clock().turn > 0  # time really passed during the rest
