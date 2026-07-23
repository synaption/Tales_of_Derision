"""Determinism tests: the world RNG is seed-reproducible, its substreams are
isolated, worldgen is identical for a given seed, and no simulation module reaches
for the global ``random`` module (which would break reproducibility / time travel).
"""
from __future__ import annotations

from pathlib import Path
import re

import esper
import pytest

import ai
import interactions
import main
import render
import systems
import worldgen
from components import Name, Position
from game_map import GameMap
from worldgen import _setup_world
from rng import WorldRng, new_seed, set_world_rng, world_rng

pytestmark = pytest.mark.unrendered


# --- WorldRng unit behaviour ------------------------------------------------

def test_same_seed_same_stream_sequence() -> None:
    a = WorldRng(42)
    b = WorldRng(42)
    seq_a = [a.stream("flora").random() for _ in range(20)]
    seq_b = [b.stream("flora").random() for _ in range(20)]
    assert seq_a == seq_b


def test_streams_are_independent() -> None:
    """Draws in one domain must not shift another domain's sequence: a stream's
    output is the same whether or not a sibling stream was drawn first."""
    solo = WorldRng(7)
    solo_seq = [solo.stream("ai").random() for _ in range(10)]

    interleaved = WorldRng(7)
    _ = [interleaved.stream("flora").random() for _ in range(100)]
    inter_seq = [interleaved.stream("ai").random() for _ in range(10)]

    assert solo_seq == inter_seq


def test_different_seeds_diverge() -> None:
    a = [WorldRng(1).stream("ai").random() for _ in range(10)]
    b = [WorldRng(2).stream("ai").random() for _ in range(10)]
    assert a != b


def test_int_seed_is_stable_and_named() -> None:
    assert WorldRng(99).int_seed("names") == WorldRng(99).int_seed("names")
    assert WorldRng(99).int_seed("names") != WorldRng(99).int_seed("newborn")


def test_set_and_get_world_rng_roundtrip() -> None:
    set_world_rng(123)
    assert world_rng().seed == 123


def test_new_seed_produces_ints() -> None:
    assert isinstance(new_seed(), int)


# --- Worldgen reproducibility ----------------------------------------------

def _worldgen_snapshot() -> list[tuple[str, int, int]]:
    return sorted(
        (name.value, pos.x, pos.y)
        for _ent, (pos, name) in esper.get_components(Position, Name)
    )


def test_worldgen_is_reproducible_for_a_seed() -> None:
    set_world_rng(0xABCDEF)
    _setup_world(GameMap(60, 30), Position(30, 15))
    first = _worldgen_snapshot()

    esper.clear_database()

    set_world_rng(0xABCDEF)
    _setup_world(GameMap(60, 30), Position(30, 15))
    second = _worldgen_snapshot()

    assert first == second
    assert first  # sanity: the world actually populated


def test_worldgen_varies_by_seed() -> None:
    set_world_rng(1)
    _setup_world(GameMap(60, 30), Position(30, 15))
    first = _worldgen_snapshot()

    esper.clear_database()

    set_world_rng(2)
    _setup_world(GameMap(60, 30), Position(30, 15))
    second = _worldgen_snapshot()

    assert first != second


# --- Grep-gate: no global ``random`` in simulation modules ------------------

_BARE_RANDOM_CALL = re.compile(r"(?:^|[^.\w])random\.[A-Za-z_]")
_IMPORT_RANDOM = re.compile(r"^\s*(?:import random\b|from random import)", re.MULTILINE)


@pytest.mark.parametrize("module", [systems, main, ai, render, worldgen, interactions])
def test_no_global_random_in_sim_modules(module) -> None:
    """Simulation randomness must flow through ``rng.world_rng()``. A stray
    ``import random`` / ``random.foo`` reintroduces process-global RNG state and
    silently breaks seed reproducibility."""
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert not _IMPORT_RANDOM.search(source), (
        f"{module.__name__} imports the global random module"
    )
    assert not _BARE_RANDOM_CALL.search(source), (
        f"{module.__name__} calls the global random module directly"
    )
