"""Wild populations kept as per-region counts.

The model has to hold two things at once: the *population* is real and evolves
against its food, while the *individuals* are a convenience that only exists
where someone can see them. These tests pin both -- that dissolving a region
into a number and materialising it back doesn't quietly lose animals, and that
the day model actually responds to how much there is to eat.
"""
from __future__ import annotations

import ecs
import pytest

from components import Name, Position, Renderable, Seaweed, Tree
from game_map import GameMap, archipelago_size
from regions import RegionId, all_region_ids, region_at
from rng import set_world_rng
import spatial
import wildlife
from content.loader import load_all_content
from wildlife import FISH, Species, WildlifeStocks

pytestmark = pytest.mark.unrendered


# A species with a habitat we control, so the model can be tested on a plain
# test map instead of needing real ocean geometry. It eats Trees, which are
# trivial to place, and lives anywhere.
def _anywhere(game_map: GameMap, x: int, y: int) -> bool:
    return True


def _species(**overrides) -> Species:
    base = dict(
        name="minnow",
        component=Seaweed,      # any marker component works; Seaweed is inert
        prefab="seaweed",
        habitat=_anywhere,
        food=Tree,
        food_per_unit=1,
        bites_per_day=1.0,
        forage_efficiency=1.0,
        starve_chance=0.0,
        breed_chance=0.0,
        density_cap=1.0,
        predators=(),
        predation_chance=0.0,
    )
    base.update(overrides)
    return Species(**base)


def _stocks(game_map: GameMap, species: Species) -> WildlifeStocks:
    load_all_content()  # materialising spawns from the prefab registry
    wildlife.detach()
    return WildlifeStocks(game_map, species=(species,))


def _only_region(game_map: GameMap) -> RegionId:
    regions = all_region_ids(game_map)
    assert len(regions) == 1, "test maps are meant to be a single region"
    return regions[0]


# --- stocks vs entities ----------------------------------------------------


def test_a_stocked_region_holds_a_population_with_no_entities() -> None:
    game_map = GameMap(24, 14)
    species = _species()
    stocks = _stocks(game_map, species)
    region = _only_region(game_map)

    stocks.stock(region, species).count = 40

    assert stocks.population(region, species) == 40
    assert len(spatial.ensure(game_map).of_kind(region, species.component)) == 0


def test_materialise_then_dissolve_conserves_the_population() -> None:
    game_map = GameMap(24, 14)
    species = _species()
    stocks = _stocks(game_map, species)
    region = _only_region(game_map)
    stocks.stock(region, species).count = 25

    stocks.sync_residency({region})
    materialised = len(spatial.ensure(game_map).of_kind(region, species.component))
    assert materialised == 25, "every stocked animal should become an entity"
    assert stocks.population(region, species) == 25

    stocks.sync_residency(set())
    assert len(spatial.ensure(game_map).of_kind(region, species.component)) == 0
    assert stocks.population(region, species) == 25, "dissolving must not lose any"


def test_seeding_adopts_world_gen_animals_and_stocks_them_down() -> None:
    game_map = GameMap(24, 14)
    species = _species()
    stocks = _stocks(game_map, species)
    region = _only_region(game_map)
    for i in range(7):
        ecs.create_entity(Position(2 + i, 3), Renderable("f"), Name("Minnow"), Seaweed())

    stocks.seed_from_world()

    assert stocks.population(region, species) == 7   # counted
    assert len(spatial.ensure(game_map).of_kind(region, species.component)) == 0  # not resident


# --- the day model ---------------------------------------------------------


def _place_food(game_map: GameMap, count: int) -> None:
    placed = 0
    for y in range(1, game_map.height - 1):
        for x in range(1, game_map.width - 1):
            if placed >= count:
                return
            if game_map.tiles[y][x] != game_map.FLOOR:
                continue
            ecs.create_entity(Position(x, y), Renderable("T"), Name("Tree"), Tree())
            placed += 1


def test_a_fed_population_breeds_up_toward_its_density_cap() -> None:
    game_map = GameMap(40, 20)
    species = _species(breed_chance=0.2, density_cap=1.0)
    stocks = _stocks(game_map, species)
    region = _only_region(game_map)
    stocks.stock(region, species).count = 10
    _place_food(game_map, 200)  # far more than 10 animals can eat

    for day in range(1, 30):
        stocks.advance_day(region, day)

    assert stocks.population(region, species) > 10, "a fed population should grow"


def test_a_starving_population_dies_back() -> None:
    game_map = GameMap(40, 20)
    species = _species(starve_chance=0.5, breed_chance=0.0)
    stocks = _stocks(game_map, species)
    region = _only_region(game_map)
    stocks.stock(region, species).count = 60
    # No food at all -> fed == 0 -> the full starvation rate applies.

    for day in range(1, 20):
        stocks.advance_day(region, day)

    assert stocks.population(region, species) < 60


def test_population_settles_at_what_the_food_supply_can_feed() -> None:
    """The point of the whole model: how many animals a region carries follows
    what there is to eat, not what it started with.

    Food is topped back up to ``ration`` each day, standing in for the seaweed
    the flora pass regrows -- against a *fixed* larder every population starves
    to nothing, which says nothing interesting.
    """
    ration = 20
    game_map = GameMap(40, 20)
    species = _species(starve_chance=0.5, breed_chance=0.05, forage_efficiency=1.0)
    stocks = _stocks(game_map, species)
    region = _only_region(game_map)
    stocks.stock(region, species).count = 80

    for day in range(1, 60):
        missing = ration - spatial.component_population(Tree)
        if missing > 0:
            _place_food(game_map, missing)
        stocks.advance_day(region, day)

    survivors = stocks.population(region, species)
    # It should land near what a day's ration feeds (one bite each), not at the
    # 80 it started with and not at zero.
    assert 0 < survivors < 80, f"expected a die-back toward the food, got {survivors}"
    assert survivors <= ration * 2, f"settled at {survivors} on a ration of {ration}"


def test_eating_removes_the_food_it_ate() -> None:
    game_map = GameMap(40, 20)
    species = _species(food_per_unit=1, bites_per_day=1.0, forage_efficiency=1.0)
    stocks = _stocks(game_map, species)
    region = _only_region(game_map)
    stocks.stock(region, species).count = 5
    _place_food(game_map, 40)
    before = spatial.component_population(Tree)

    stocks.advance_day(region, 1)

    assert spatial.component_population(Tree) == before - 5


def test_a_region_with_no_animals_costs_nothing_and_stays_empty() -> None:
    game_map = GameMap(24, 14)
    species = _species(breed_chance=1.0)
    stocks = _stocks(game_map, species)
    region = _only_region(game_map)
    _place_food(game_map, 50)

    for day in range(1, 10):
        stocks.advance_day(region, day)

    assert stocks.population(region, species) == 0, "nothing breeds out of nothing"


# --- determinism -----------------------------------------------------------


def _run_days(days: list[int], start: int = 30) -> int:
    """Advance one region over ``days`` (in the given order) and report the
    population. Used to prove the answer doesn't depend on the order."""
    ecs.clear_database()
    spatial.detach()
    set_world_rng(4242)
    game_map = GameMap(40, 20)
    species = _species(starve_chance=0.4, breed_chance=0.1)
    stocks = _stocks(game_map, species)
    region = _only_region(game_map)
    stocks.stock(region, species).count = start
    _place_food(game_map, 25)
    for day in days:
        stocks.advance_day(region, day)
    return stocks.population(region, species)


def test_the_same_seed_and_days_give_the_same_population() -> None:
    assert _run_days(list(range(1, 21))) == _run_days(list(range(1, 21)))


def test_a_days_outcome_does_not_depend_on_when_it_was_simulated() -> None:
    """Batch independence, the rule ``RegionScheduler`` documents. A region's
    randomness comes from ``rng.region_day_rng(seed, region, day)``, so day 7 is
    day 7 whether it was caught up in one burst or a day at a time -- which is
    what lets the idle pump advance however many regions its budget allows.
    """
    ecs.clear_database()
    spatial.detach()
    set_world_rng(99)
    game_map = GameMap(40, 20)
    species = _species(starve_chance=0.4, breed_chance=0.0)
    stocks = _stocks(game_map, species)
    region = _only_region(game_map)
    stocks.stock(region, species).count = 50
    _place_food(game_map, 10)

    seen = []
    for day in (1, 2, 3):
        stocks.advance_day(region, day)
        seen.append(stocks.population(region, species))

    # Replaying day 2 from the same population must reproduce day 2's outcome,
    # because the draw is keyed on the day rather than on a running stream.
    stocks.stock(region, species).count = seen[0]
    stocks.advance_day(region, 2)
    assert stocks.population(region, species) == seen[1]


# --- fish on a real ocean --------------------------------------------------
#
# These build a real archipelago and use the real ``FISH`` species, but drive
# residency explicitly rather than reading it off where world-gen happened to
# put the villagers: a small test archipelago packs an island into nearly every
# region, so *everywhere* would be live and the interesting case would never
# arise. At shipping size (143 regions) two are live and the rest are numbers.


def _ocean_world():
    from worldgen import _setup_world

    ecs.clear_database()
    spatial.detach()
    wildlife.detach()
    set_world_rng(0x7A1E5)
    width, height = archipelago_size(3)
    game_map = GameMap(width, height, layout="islands")
    _setup_world(game_map, Position(width // 2, height // 2))
    return game_map, wildlife.ensure(game_map)


def _biggest_shoal(game_map, stocks) -> RegionId:
    return max(
        all_region_ids(game_map), key=lambda r: stocks.population(r, FISH)
    )


def test_world_gen_fish_survive_being_stocked_down() -> None:
    """Every fish world-gen scattered is still in the world afterwards -- as a
    number in all but the one region left live."""
    from components import Fish

    game_map, stocks = _ocean_world()
    before = stocks.world_population(FISH)
    assert before > 0, "the sea should have fish"

    keep = _biggest_shoal(game_map, stocks)
    stocks.sync_residency({keep})

    assert stocks.world_population(FISH) == before, "stocking down must lose none"
    index = spatial.ensure(game_map)
    for region_id in all_region_ids(game_map):
        if region_id != keep:
            assert not index.of_kind(region_id, Fish), "only the live region holds entities"
    assert spatial.component_population(Fish) < before, "the rest are numbers now"


def test_entering_a_region_materialises_its_shoal() -> None:
    from components import Fish

    game_map, stocks = _ocean_world()
    target = _biggest_shoal(game_map, stocks)
    stocks.sync_residency(set())            # nobody anywhere: all numbers
    expected = stocks.stock(target, FISH).count
    assert expected > 0

    stocks.sync_residency({target})         # somebody walks in

    got = len(spatial.ensure(game_map).of_kind(target, Fish))
    assert got == expected, f"entering should surface all {expected} fish, saw {got}"


def test_the_fish_step_skips_a_region_whose_shoal_is_a_number() -> None:
    """The whole performance case, held directly. A stocked region reports an
    unbounded idle span, so the scheduler pays its debt in one visit instead of
    replaying a night of random walks -- see ``FishAiProcessor._idle_turns``."""
    from components import Fish
    from regions import _UNBOUNDED
    from systems import FishAiProcessor

    game_map, stocks = _ocean_world()
    live = _biggest_shoal(game_map, stocks)
    stocks.sync_residency({live})
    processor = FishAiProcessor(game_map)
    index = spatial.ensure(game_map)

    stocked = [
        r for r in all_region_ids(game_map)
        if r != live and not index.of_kind(r, Fish)
    ]
    assert stocked, "expected regions holding no fish entities"
    for region_id in stocked:
        assert processor._idle_turns(region_id) == _UNBOUNDED
        assert processor.scheduler.jump_limit(region_id) > 1
    # The live one still runs a turn at a time -- there are real fish to swim.
    assert processor._idle_turns(live) == 1


# --- one day cursor --------------------------------------------------------


def test_the_population_model_rides_the_floras_day_cursor() -> None:
    """Animals eat plants, so the two day models cannot keep separate cursors.

    They briefly did, and it was wrong in a way only long catch-ups showed: each
    was caught up to completion in turn, so a region could grow a month of
    seaweed with nothing grazing it and *then* graze a month with nothing
    growing, landing somewhere the interleaved simulation never would. The fix is
    that there is only one cursor -- the flora's -- and the population is a step
    on it. This holds that wiring, because the failure is invisible over one day.
    """
    from components import WorldClock
    from systems import TreeGrowthProcessor

    game_map = GameMap(40, 20)
    clock = WorldClock(turn=0, day_length=10)
    ecs.create_entity(clock)
    flora = TreeGrowthProcessor(game_map, rng=lambda: 1.0)
    animals = wildlife.WildlifeProcessor(game_map)
    animals.register_on(flora)

    seen: list[tuple[RegionId, int]] = []
    flora.register_day_step("spy", lambda r, d: seen.append((r, d)))

    clock.turn = 0
    flora.process("wait")          # baseline
    clock.turn = 5 * 10            # five days elapse in one go
    flora.process("wait")

    region = _only_region(game_map)
    assert seen == [(region, d) for d in range(1, 6)], (
        "every day the flora ran, the population must have run too, in order"
    )
    # And the population step really is the wildlife one, not just the spy.
    assert any(name == "wildlife" for name, _fn in flora._day_steps)


def test_a_wildlife_processor_keeps_no_day_cursor_of_its_own() -> None:
    """The invariant behind the test above: a second cursor is the bug."""
    game_map = GameMap(24, 14)
    animals = wildlife.WildlifeProcessor(game_map)
    assert not hasattr(animals, "_region_day")
    assert not hasattr(animals, "catch_up_all_wildlife")
    assert not hasattr(animals, "pump_wildlife")
