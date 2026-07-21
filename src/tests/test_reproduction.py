"""Unrendered tests for aging, courtship, sex, pregnancy, marriage, and birth:
the adult threshold, the private/consensual mating rules (men once per day, women
unbounded), the 1% pregnancy roll, weddings that merge two people into one
household, and the ReproductionProcessor delivering babies at term."""
from __future__ import annotations

import esper
import pytest

from components import (
    Age,
    Bed,
    Family,
    Friendly,
    Gender,
    Home,
    Name,
    Owned,
    Personality,
    Position,
    Pregnant,
    Relationships,
    Resident,
)
from game_map import GameMap
import systems
from systems import (
    HousingProcessor,
    ReproductionProcessor,
    WAIT_ACTION,
    WorldClock,
    age_years,
    born_turn_for_age,
    is_adult,
    is_private,
    try_marry,
    try_mate,
)

pytestmark = pytest.mark.unrendered

_DAY = 240  # day_length used throughout


def _clock(turn: int = 1_000_000) -> WorldClock:
    return WorldClock(turn=turn, day_length=_DAY)


def _person(
    gender: str,
    clock: WorldClock,
    *,
    age: float = 25.0,
    x: int = 5,
    y: int = 5,
    traits: list[str] | None = None,
    surname: str = "Vale",
) -> int:
    given = "Man" if gender == "male" else "Maid"
    return esper.create_entity(
        Position(x, y),
        Name(f"{given} {surname}"),
        Gender(gender),
        Age(born_turn_for_age(clock, age)),
        Friendly(),
        Family(surname=surname),
        Personality(traits=traits or []),
        Relationships(),
    )


def _set_mutual(a: int, b: int, score: float) -> None:
    esper.component_for_entity(a, Relationships).scores[b] = score
    esper.component_for_entity(b, Relationships).scores[a] = score


# --- Aging -----------------------------------------------------------------


def test_age_years_and_adulthood_threshold() -> None:
    clock = _clock()
    grown = _person("female", clock, age=25.0)
    child = _person("male", clock, age=6.0, x=1, y=1)

    assert age_years(grown, clock) == pytest.approx(25.0, abs=0.01)
    assert age_years(child, clock) == pytest.approx(6.0, abs=0.01)
    assert is_adult(grown, clock)
    assert not is_adult(child, clock)


# --- Marriage --------------------------------------------------------------


def test_try_marry_weds_at_threshold_sharing_home_and_surname() -> None:
    clock = _clock()
    husband = _person("male", clock, surname="Stone")
    wife = _person("female", clock, surname="Brook", x=6, y=5)
    esper.add_component(husband, Home(9, 9))
    _set_mutual(husband, wife, 90.0)

    assert try_marry(husband, wife, clock)

    assert esper.component_for_entity(husband, Family).spouse == wife
    assert esper.component_for_entity(wife, Family).spouse == husband
    # Wife takes the husband's surname, name and family record both.
    assert esper.component_for_entity(wife, Family).surname == "Stone"
    assert esper.component_for_entity(wife, Name).value == "Maid Stone"
    # They share a bed.
    wife_home = esper.component_for_entity(wife, Home)
    assert (wife_home.x, wife_home.y) == (9, 9)


def test_try_marry_needs_mutual_high_friendship() -> None:
    clock = _clock()
    a = _person("male", clock)
    b = _person("female", clock, x=6, y=5)
    # One-sided adoration isn't enough: the weaker direction gates it.
    esper.component_for_entity(a, Relationships).scores[b] = 95.0
    esper.component_for_entity(b, Relationships).scores[a] = 40.0

    assert not try_marry(a, b, clock)
    assert esper.component_for_entity(a, Family).spouse is None


def test_marriage_needs_two_adults_of_opposite_gender() -> None:
    clock = _clock()
    man = _person("male", clock)
    boy = _person("male", clock, age=8.0, x=6, y=5)
    _set_mutual(man, boy, 95.0)
    assert not try_marry(man, boy, clock)  # same gender + a child


# --- Sex, privacy, and cooldowns -------------------------------------------


def _lovers(clock: WorldClock) -> tuple[int, int]:
    man = _person("male", clock, x=5, y=5)
    woman = _person("female", clock, x=6, y=5)
    _set_mutual(man, woman, 70.0)  # above the lovers threshold, below marriage
    return man, woman


def _sentients() -> list[tuple[int, Position]]:
    return [(e, p) for e, (p, _pers) in esper.get_components(Position, Personality)]


def test_sex_requires_privacy() -> None:
    clock = _clock()
    man, woman = _lovers(clock)
    # Alone: it happens (force conception with rng=0).
    assert try_mate(man, woman, clock.turn, clock, _sentients(), rng=lambda: 0.0)
    assert esper.has_component(woman, Pregnant)

    # Reset and add a witness within the privacy radius: no dice.
    esper.remove_component(woman, Pregnant)
    esper.component_for_entity(man, systems.Mating).last_turn = -10_000
    _person("male", clock, x=7, y=5, surname="Nosy")  # a bystander two tiles away
    assert not is_private(man, woman, _sentients())
    assert not try_mate(man, woman, clock.turn, clock, _sentients(), rng=lambda: 0.0)


def test_men_mate_once_per_day_women_unbounded() -> None:
    clock = _clock(turn=100 * _DAY)  # some whole day
    man, woman = _lovers(clock)

    assert try_mate(man, woman, clock.turn, clock, _sentients(), rng=lambda: 1.0)
    # Same day, a turn later: the man is spent.
    assert not try_mate(man, woman, clock.turn + 1, clock, _sentients(), rng=lambda: 1.0)
    # Next day: he can again.
    next_day = _clock(turn=101 * _DAY)
    assert try_mate(man, woman, next_day.turn, next_day, _sentients(), rng=lambda: 1.0)


def test_woman_is_unbounded_across_partners_in_a_day() -> None:
    # A woman has no daily limit: two different men in one day both succeed. Move
    # each partner in turn so the pair is always alone (privacy respected).
    clock = _clock(turn=200 * _DAY)
    w = _person("female", clock, x=5, y=5)
    m1 = _person("male", clock, x=6, y=5, surname="One")
    m2 = _person("male", clock, x=40, y=40, surname="Two")  # far away for now
    _set_mutual(w, m1, 70.0)
    _set_mutual(w, m2, 70.0)

    assert try_mate(w, m1, clock.turn, clock, _sentients(), rng=lambda: 1.0)
    # Shuffle: m1 leaves, m2 arrives -- the woman has no cooldown.
    esper.component_for_entity(m1, Position).x = 41
    esper.component_for_entity(m2, Position).x = 6
    esper.component_for_entity(m2, Position).y = 5
    assert try_mate(w, m2, clock.turn, clock, _sentients(), rng=lambda: 1.0)


def test_pregnancy_only_on_a_low_roll() -> None:
    clock = _clock()
    man, woman = _lovers(clock)
    # A high roll (>= 1%): no pregnancy.
    assert try_mate(man, woman, clock.turn, clock, _sentients(), rng=lambda: 0.5)
    assert not esper.has_component(woman, Pregnant)


# --- Birth -----------------------------------------------------------------


def test_reproduction_processor_delivers_at_term() -> None:
    clock = WorldClock(turn=0, day_length=_DAY)
    esper.create_entity(clock)
    father = esper.create_entity(Name("Pa Vale"), Gender("male"), Family(surname="Vale"))
    mother = esper.create_entity(
        Position(10, 10),
        Name("Ma Vale"),
        Gender("female"),
        Family(surname="Vale"),
        Home(10, 10),
        Pregnant(conceived_turn=0, father=father),
    )

    proc = ReproductionProcessor()
    term_turns = systems._GESTATION_DAYS * _DAY

    # A day before term: baseline pass, no birth.
    clock.turn = term_turns
    proc.process(WAIT_ACTION)
    assert esper.has_component(mother, Pregnant)

    # The next day rolls over past term: the baby arrives.
    clock.turn = term_turns + _DAY
    proc.process(WAIT_ACTION)
    assert not esper.has_component(mother, Pregnant)

    babies = [
        (e, fam)
        for e, (fam, _pos) in esper.get_components(Family, Position)
        if fam.parents
    ]
    assert len(babies) == 1
    baby, fam = babies[0]
    assert fam.surname == "Vale"
    assert set(fam.parents) == {father, mother}
    assert baby in esper.component_for_entity(mother, Family).children
    assert baby in esper.component_for_entity(father, Family).children
    assert esper.component_for_entity(baby, Gender).value in ("male", "female")
    # Lives with mother; not a resident (won't build its own cabin as a baby).
    assert esper.has_component(baby, Home)
    assert not esper.has_component(baby, Resident)


# --- Spouses share a bed ---------------------------------------------------


def test_housing_moves_a_resident_in_with_their_housed_spouse() -> None:
    esper.create_entity(WorldClock(turn=0, day_length=_DAY))
    game_map = GameMap(30, 18)

    owner = esper.create_entity(Name("Owner"), Home(4, 4), Family(surname="Vale"))
    bed = esper.create_entity(Position(4, 4), Bed(), Owned(owner))
    assert systems.owned_bed_of(owner) == bed

    spouse = esper.create_entity(
        Position(20, 12),
        Name("Spouse"),
        Family(surname="Vale", spouse=owner),
        Resident(),
    )
    esper.component_for_entity(owner, Family).spouse = spouse

    HousingProcessor(game_map).process(WAIT_ACTION)

    spouse_home = esper.component_for_entity(spouse, Home)
    assert (spouse_home.x, spouse_home.y) == (4, 4)
