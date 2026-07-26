"""Unrendered tests for the calendar (year/month/week/day + clock time) and the
tree growth cycle (saplings sprouting near trees and maturing after a year)."""
from __future__ import annotations

import esper
import pytest

from components import BerryBush, BlocksMovement, Inventory, Name, Player, Position, Renderable, Sapling, Tree, WorldClock
from game_map import GameMap
from systems import (
    TreeGrowthProcessor,
    _BERRY_REGROW_DAYS,
    _DAYS_PER_YEAR,
    calendar,
    format_datetime,
    pick_berries,
)

pytestmark = pytest.mark.unrendered


# --- Calendar --------------------------------------------------------------


def test_calendar_breaks_the_clock_into_year_month_week_day() -> None:
    clock = WorldClock(turn=0, day_length=10)

    # Day 0 -> Year 1, Month 1, Week 1, Day 1.
    assert calendar(clock)[:4] == (1, 1, 1, 1)

    clock.turn = 7 * 10  # day 7 (0-indexed) -> Week 2, Day 1
    assert calendar(clock)[:4] == (1, 1, 2, 1)

    clock.turn = 28 * 10  # day 28 -> next month (Month 2, Week 1, Day 1)
    assert calendar(clock)[:4] == (1, 2, 1, 1)

    clock.turn = _DAYS_PER_YEAR * 10  # 112 days -> Year 2
    assert calendar(clock)[:4] == (2, 1, 1, 1)


def test_calendar_reports_clock_time_within_the_day() -> None:
    clock = WorldClock(turn=0, day_length=24)  # 1 turn == 1 hour
    clock.turn = 6  # a quarter through the day
    _y, _mo, _w, _d, hour, minute = calendar(clock)
    assert (hour, minute) == (6, 0)


def test_format_datetime_is_compact_and_handles_no_clock() -> None:
    clock = WorldClock(turn=0, day_length=10)
    text = format_datetime(clock)
    assert text.startswith("Y1 M1 W1 D1")
    assert format_datetime(None) == "Day"


# --- Tree growth -----------------------------------------------------------

_DAY_LEN = 10


def _advance_a_day(processor, clock, *, from_turn: int) -> None:
    """Prime the processor's day baseline at ``from_turn`` then step one day, so
    exactly one daily forest pass runs."""
    clock.turn = from_turn
    processor.process("wait")  # establishes the baseline day (no pass)
    clock.turn = from_turn + clock.day_length
    processor.process("wait")  # a new day -> one forest pass


def test_saplings_sprout_on_open_outdoor_ground() -> None:
    game_map = GameMap(24, 14)  # too small to carve houses -> all floor is outdoor
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    processor = TreeGrowthProcessor(game_map, rng=lambda: 0.0)  # every roll sprouts

    _advance_a_day(processor, clock, from_turn=0)

    saplings = list(esper.get_components(Sapling))
    assert saplings, "expected saplings to sprout on open ground"
    for ent, (sapling,) in saplings:
        pos = esper.component_for_entity(ent, Position)
        assert game_map.tile_at(pos.x, pos.y) == game_map.FLOOR
        assert sapling.planted_turn == clock.turn


def test_saplings_do_not_sprout_indoors() -> None:
    # A 40x20 map carves houses; their interior floor is indoors and must stay
    # sapling-free even when every roll would otherwise sprout.
    game_map = GameMap(40, 20)
    interiors = set().union(*game_map.find_enclosed_rooms())
    assert interiors  # sanity: the map really has enclosed houses
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    processor = TreeGrowthProcessor(game_map, rng=lambda: 0.0)

    _advance_a_day(processor, clock, from_turn=0)

    sapling_tiles = {
        (esper.component_for_entity(e, Position).x, esper.component_for_entity(e, Position).y)
        for e, _c in esper.get_components(Sapling)
    }
    assert sapling_tiles.isdisjoint(interiors)


def test_no_saplings_when_the_daily_roll_never_fires() -> None:
    game_map = GameMap(24, 14)
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    processor = TreeGrowthProcessor(game_map, rng=lambda: 1.0)  # never below the chance

    _advance_a_day(processor, clock, from_turn=0)
    assert list(esper.get_components(Sapling)) == []


def test_growth_only_happens_on_a_new_day() -> None:
    game_map = GameMap(24, 14)
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    processor = TreeGrowthProcessor(game_map, rng=lambda: 0.0)

    processor.process("wait")  # first call just sets the baseline day
    clock.turn = _DAY_LEN - 1  # still day 0
    processor.process("wait")
    assert list(esper.get_components(Sapling)) == []  # no new day yet -> nothing


def test_trees_can_die() -> None:
    game_map = GameMap(24, 14)
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    t1 = esper.create_entity(Position(6, 6), Renderable("T"), Name("Tree"), Tree(), BlocksMovement())
    t2 = esper.create_entity(Position(9, 9), Renderable("T"), Name("Tree"), Tree(), BlocksMovement())
    # rng = 0 is below the death chance, so every tree dies on the daily pass.
    processor = TreeGrowthProcessor(game_map, rng=lambda: 0.0)

    _advance_a_day(processor, clock, from_turn=0)
    assert not esper.entity_exists(t1)
    assert not esper.entity_exists(t2)


def test_sapling_matures_into_a_tree_after_a_year() -> None:
    game_map = GameMap(24, 14)
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    sapling = esper.create_entity(
        Position(8, 8), Renderable("t"), Name("Sapling"), Sapling(planted_turn=0)
    )
    processor = TreeGrowthProcessor(game_map, rng=lambda: 1.0)  # suppress sprouts/deaths

    # A daily pass before it is a year old: still a sapling.
    _advance_a_day(processor, clock, from_turn=100 * _DAY_LEN)
    assert esper.has_component(sapling, Sapling)
    assert not esper.has_component(sapling, Tree)

    # A daily pass after a full year (112 days) has elapsed: it becomes a tree.
    clock.turn = _DAYS_PER_YEAR * _DAY_LEN
    processor.process("wait")
    assert not esper.has_component(sapling, Sapling)
    assert esper.has_component(sapling, Tree)
    assert esper.has_component(sapling, BlocksMovement)
    assert esper.component_for_entity(sapling, Renderable).glyph == "T"
    assert esper.component_for_entity(sapling, Name).value == "Tree"


def test_sapling_maturation_waits_while_its_tile_is_occupied() -> None:
    game_map = GameMap(24, 14)
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    sapling = esper.create_entity(
        Position(8, 8), Renderable("t"), Name("Sapling"), Sapling(planted_turn=0)
    )
    # The player stands on the sapling tile -- it must not turn into a tree under
    # them (that would trap them in a wall of wood).
    esper.create_entity(Position(8, 8), Player(), BlocksMovement())
    processor = TreeGrowthProcessor(game_map, rng=lambda: 1.0)

    _advance_a_day(processor, clock, from_turn=_DAYS_PER_YEAR * _DAY_LEN)
    assert esper.has_component(sapling, Sapling)  # deferred
    assert not esper.has_component(sapling, Tree)


# --- Berry bushes ----------------------------------------------------------


def test_bush_sapling_matures_into_a_berry_bush() -> None:
    game_map = GameMap(24, 14)
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    seedling = esper.create_entity(
        Position(8, 8), Renderable(","), Name("Bush Seedling"), Sapling(planted_turn=0, kind="bush")
    )
    processor = TreeGrowthProcessor(game_map, rng=lambda: 1.0)  # suppress sprouts/deaths

    _advance_a_day(processor, clock, from_turn=_DAYS_PER_YEAR * _DAY_LEN)
    assert not esper.has_component(seedling, Sapling)
    assert esper.has_component(seedling, BerryBush)
    assert esper.has_component(seedling, BlocksMovement)
    assert esper.component_for_entity(seedling, BerryBush).has_berries is True
    assert esper.component_for_entity(seedling, Name).value == "Berry Bush"


def test_picking_berries_marks_the_bush_bare_and_stamps_the_time() -> None:
    clock = WorldClock(turn=500, day_length=_DAY_LEN)
    esper.create_entity(clock)
    bush = esper.create_entity(Position(8, 8), Renderable("%"), Name("Berry Bush"), BerryBush())

    assert pick_berries(bush, clock) is True
    b = esper.component_for_entity(bush, BerryBush)
    assert b.has_berries is False
    assert b.harvested_turn == 500
    # A bare bush yields nothing until it regrows.
    assert pick_berries(bush, clock) is False


def test_bush_regrows_berries_after_seven_days() -> None:
    game_map = GameMap(24, 14)
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    bush = esper.create_entity(Position(8, 8), Renderable("%"), Name("Berry Bush"), BerryBush())
    pick_berries(bush, clock)  # harvested at turn 0
    processor = TreeGrowthProcessor(game_map, rng=lambda: 1.0)

    # A daily pass on day 6 (only six days after harvest): not ripe yet.
    _advance_a_day(processor, clock, from_turn=5 * _DAY_LEN)
    assert esper.component_for_entity(bush, BerryBush).has_berries is False

    # Seven days after harvest: a fresh crop.
    clock.turn = _BERRY_REGROW_DAYS * _DAY_LEN
    processor.process("wait")
    assert esper.component_for_entity(bush, BerryBush).has_berries is True


def test_bush_picked_midway_through_a_day_still_regrows_on_the_seventh_day() -> None:
    """Deadlines are counted in whole days from the day they were set, so the
    hour a bush was picked at doesn't push its crop into an eighth day. Picking
    almost always happens mid-day in play, and the queue files by day, so this is
    the case that would drift if the rounding were wrong."""
    game_map = GameMap(24, 14)
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    bush = esper.create_entity(Position(8, 8), Renderable("%"), Name("Berry Bush"), BerryBush())
    processor = TreeGrowthProcessor(game_map, rng=lambda: 1.0)
    processor.process("wait")  # baseline day 0
    clock.turn = _DAY_LEN // 2  # picked halfway through day 0
    pick_berries(bush, clock)

    for day in range(1, _BERRY_REGROW_DAYS):
        clock.turn = day * _DAY_LEN
        processor.process("wait")
        assert esper.component_for_entity(bush, BerryBush).has_berries is False

    clock.turn = _BERRY_REGROW_DAYS * _DAY_LEN
    processor.process("wait")
    assert esper.component_for_entity(bush, BerryBush).has_berries is True


def test_bush_saplings_can_sprout() -> None:
    game_map = GameMap(24, 14)
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    # Sprouting draws twice per sapling: first the gap to the next sprouting
    # tile (0.0 -> the very next one), then what kind it is. A kind roll above
    # the tree's share of the combined chance makes every sprout a bush.
    from systems import _DAILY_SPROUT_CHANCE, _DAILY_BUSH_SPROUT_CHANCE
    tree_share = _DAILY_SPROUT_CHANCE / (_DAILY_SPROUT_CHANCE + _DAILY_BUSH_SPROUT_CHANCE)
    rolls = iter([0.0, tree_share + 0.01] * 10_000)
    processor = TreeGrowthProcessor(game_map, rng=lambda: next(rolls))

    _advance_a_day(processor, clock, from_turn=0)
    kinds = {esper.component_for_entity(e, Sapling).kind for e, _c in esper.get_components(Sapling)}
    assert kinds == {"bush"}


# --- Counting instead of scanning ------------------------------------------
#
# The daily pass must cost what actually grew, not what could have. These pin
# that down directly, because it is the property the whole design rests on and
# nothing else in the suite would notice it regressing.


def test_a_quiet_day_costs_a_handful_of_rolls_not_one_per_tile() -> None:
    game_map = GameMap(24, 14)  # ~250 outdoor ground tiles
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    for x in range(4, 12):
        esper.create_entity(Position(x, 3), Renderable("T"), Name("Tree"), Tree(), BlocksMovement())
    calls = 0

    def counting_rng() -> float:
        nonlocal calls
        calls += 1
        return 0.999999  # nothing sprouts, nothing dies

    processor = TreeGrowthProcessor(game_map, rng=counting_rng)
    _advance_a_day(processor, clock, from_turn=0)

    # One draw for tree deaths, one for bush deaths, one for sprouting: the day
    # asks each distribution "how many?" once and is told "none".
    assert calls <= 4, f"a quiet day should not roll per tile or per plant (rolled {calls})"
    assert list(esper.get_components(Sapling)) == []


def test_sprouting_costs_two_rolls_per_sapling_regardless_of_map_size() -> None:
    """The same forced-sprout run on a bigger map plants proportionally more but
    still spends exactly two random numbers per sapling -- the gap draw and the
    kind draw. A per-tile scan would scale with the tiles instead."""
    counts = {}
    for width, height in ((24, 14), (48, 28)):
        esper.clear_database()
        game_map = GameMap(width, height)
        clock = WorldClock(turn=0, day_length=_DAY_LEN)
        esper.create_entity(clock)
        calls = 0

        def counting_rng() -> float:
            nonlocal calls
            calls += 1
            return 0.0  # every trial is a hit

        processor = TreeGrowthProcessor(game_map, rng=counting_rng)
        processor._cap = 10  # stop well short of the map, so tiles aren't the limit
        _advance_a_day(processor, clock, from_turn=0)
        counts[(width, height)] = (calls, len(list(esper.get_components(Sapling))))

    for _size, (calls, saplings) in counts.items():
        assert saplings == 10  # the cap, on both maps
        assert calls <= 2 * saplings + 4
    assert counts[(24, 14)][0] == counts[(48, 28)][0]


def test_death_count_tracks_the_population_not_a_per_plant_roll() -> None:
    game_map = GameMap(24, 14)
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    trees = [
        esper.create_entity(Position(x, 3), Renderable("T"), Name("Tree"), Tree(), BlocksMovement())
        for x in range(4, 12)
    ]
    # A gap roll worth exactly two failures, so the deaths land two trees apart:
    # the pass never asks the other six trees anything.
    from systems import _DAILY_DEATH_CHANCE
    two_apart = 1.0 - (1.0 - _DAILY_DEATH_CHANCE) ** 2
    rolls = iter([two_apart] * 100)
    processor = TreeGrowthProcessor(game_map, rng=lambda: next(rolls))
    processor._cap = 0  # no sprouting to interleave

    _advance_a_day(processor, clock, from_turn=0)

    dead = [i for i, e in enumerate(sorted(trees)) if not esper.entity_exists(e)]
    assert dead == [2, 5]  # skip 2, hit; skip 2, hit; the next gap runs off the end


def test_growth_respects_the_soft_cap() -> None:
    game_map = GameMap(24, 14)
    clock = WorldClock(turn=0, day_length=_DAY_LEN)
    esper.create_entity(clock)
    esper.create_entity(Position(10, 6), Renderable("T"), Name("Tree"), Tree(), BlocksMovement())
    # rng between the death chance and the sprout chance: the tree survives, and
    # sprouting *would* fire -- but the cap (already met by the tree) blocks it.
    processor = TreeGrowthProcessor(game_map, rng=lambda: 0.00007)
    processor._cap = 1

    _advance_a_day(processor, clock, from_turn=0)
    assert list(esper.get_components(Sapling)) == []
