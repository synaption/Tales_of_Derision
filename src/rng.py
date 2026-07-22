"""Deterministic world RNG: one master seed, many isolated named substreams.

Every bit of *simulation* randomness must flow through here so a world is fully
reproducible from its seed -- the foundation for reproducible saves and, later,
time travel (a saved seed + input log reconstructs the world). Do NOT call the
global ``random`` module from simulation code (``systems.py`` / ``main.py``); draw
from ``world_rng().stream(name)`` instead.

Streams are isolated by name so adding a draw in one domain never shifts another
domain's sequence -- e.g. a new social roll can't perturb worldgen or flora.
"""
from __future__ import annotations

import random
import secrets


class WorldRng:
    """A master seed plus lazily-created, named ``random.Random`` substreams."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._streams: dict[str, random.Random] = {}

    def stream(self, name: str) -> random.Random:
        """The substream for ``name`` (created on first use), seeded from a stable
        string derived from the master seed. ``random.Random`` hashes string seeds
        with SHA-512, so this is reproducible across processes regardless of
        ``PYTHONHASHSEED`` (unlike the builtin ``hash()``)."""
        rng = self._streams.get(name)
        if rng is None:
            rng = random.Random(f"{self.seed}:{name}")
            self._streams[name] = rng
        return rng

    def int_seed(self, name: str) -> int:
        """A stable integer seed derived from ``(master seed, name)`` -- for helpers
        that take an ``int`` seed of their own (e.g. ``make_onymancer``)."""
        return random.Random(f"{self.seed}:int:{name}").getrandbits(64)


# Fallback so code/tests that never call ``set_world_rng`` are still deterministic
# (and never touch an uninitialised RNG).
_FALLBACK_SEED = 0x7A1E5D  # "TALESD"-ish; any fixed value works
_world_rng: WorldRng | None = None


def set_world_rng(seed: int) -> WorldRng:
    """Install the active world RNG for a freshly (re)generated world."""
    global _world_rng
    _world_rng = WorldRng(seed)
    return _world_rng


def world_rng() -> WorldRng:
    """The active world RNG, auto-initialised to a fixed fallback seed if unset."""
    global _world_rng
    if _world_rng is None:
        _world_rng = WorldRng(_FALLBACK_SEED)
    return _world_rng


def new_seed() -> int:
    """A fresh, unpredictable 64-bit world seed for a brand-new game. Seed
    *generation* may be nondeterministic; the world it produces is fully
    reproducible from the returned value (which the save stores)."""
    return secrets.randbits(64)
