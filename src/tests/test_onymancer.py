"""Unrendered tests for the onymancer (procedural name generation) and the family
ties it wires onto the starting villagers."""
from __future__ import annotations

import esper
import pytest

from components import Family, Gender, Name, Personality
from game_map import GameMap
from components import Position
from worldgen import _setup_world
from onymancer import Onymancer, make_onymancer

pytestmark = pytest.mark.unrendered


def test_same_seed_conjures_the_same_names() -> None:
    a = make_onymancer(1234)
    b = make_onymancer(1234)
    names_a = [a.given_name("female") for _ in range(20)]
    names_b = [b.given_name("female") for _ in range(20)]
    assert names_a == names_b


def test_different_seeds_diverge() -> None:
    a = [make_onymancer(1).given_name("male") for _ in range(10)]
    b = [make_onymancer(2).given_name("male") for _ in range(10)]
    assert a != b


def test_names_are_non_empty_capitalized_words() -> None:
    ony = make_onymancer(7)
    for gender in ("male", "female", None):
        for _ in range(30):
            name = ony.given_name(gender)
            assert name and name[0].isupper()
            assert name.isalpha()


def test_surname_and_full_name_shape() -> None:
    ony = make_onymancer(99)
    surname = ony.surname()
    assert surname and surname[0].isupper()

    given, fam, full = ony.full_name("female", surname="Ashford")
    assert fam == "Ashford"
    assert full == f"{given} Ashford"

    # With no surname supplied the onymancer coins one.
    _given, coined, _full = ony.full_name("male")
    assert coined and coined[0].isupper()


def test_male_and_female_pools_differ() -> None:
    # Not a strict rule -- just a sanity check that the gendered ending bias makes
    # the two streams diverge rather than being identical.
    ony_m = make_onymancer(42)
    ony_f = make_onymancer(42)
    males = [ony_m.given_name("male") for _ in range(50)]
    females = [ony_f.given_name("female") for _ in range(50)]
    assert males != females


def test_monogender_none_draws_without_error() -> None:
    ony = Onymancer(make_onymancer(5).rng)
    assert ony.given_name(None).isalpha()


# --- Family ties on the starting village -----------------------------------


def _villager_families() -> list[tuple[int, Family]]:
    return [
        (ent, esper.component_for_entity(ent, Family))
        for ent, (_pers, fam) in esper.get_components(Personality, Family)
    ]


def test_setup_world_gives_every_villager_a_name_gender_and_family() -> None:
    _setup_world(GameMap(40, 20), Position(20, 10))
    villagers = _villager_families()
    assert villagers  # some spawned
    for ent, _fam in villagers:
        assert esper.has_component(ent, Name)
        assert esper.has_component(ent, Gender)
        gender = esper.component_for_entity(ent, Gender)
        assert gender.value in ("male", "female")


def test_spouse_links_are_reciprocal() -> None:
    _setup_world(GameMap(40, 20), Position(20, 10))
    spouses = [(ent, fam) for ent, fam in _villager_families() if fam.spouse is not None]
    assert spouses  # at least one couple
    for ent, fam in spouses:
        partner_fam = esper.component_for_entity(fam.spouse, Family)
        assert partner_fam.spouse == ent
        # Spouses share the household surname.
        assert partner_fam.surname == fam.surname


def test_parent_and_child_links_are_reciprocal_and_share_a_surname() -> None:
    _setup_world(GameMap(40, 20), Position(20, 10))
    children = [(ent, fam) for ent, fam in _villager_families() if fam.parents]
    assert children  # a family with kids exists
    for child, fam in children:
        assert len(fam.parents) == 2
        for parent in fam.parents:
            parent_fam = esper.component_for_entity(parent, Family)
            assert child in parent_fam.children
            assert parent_fam.surname == fam.surname
