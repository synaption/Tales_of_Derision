"""Deterministic world RNG: one master seed, many isolated named substreams.

Every bit of *simulation* randomness must flow through here so a world is fully
reproducible from its seed -- the foundation for reproducible saves and, later,
time travel (a saved seed + input log reconstructs the world). Do NOT call the
global ``random`` module from simulation code (``systems.py`` / ``game.py``); draw
from ``world_rng().stream(name)`` instead.

Streams are isolated by name so adding a draw in one domain never shifts another
domain's sequence -- e.g. a new social roll can't perturb worldgen or flora.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from math import log1p
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


# --- counting instead of scanning ------------------------------------------


def bernoulli_hits(n: int, p: float, rng: Callable[[], float]) -> Iterator[int]:
    """The indices of the successes among ``n`` independent Bernoulli(``p``)
    trials, drawing one random number **per success** instead of one per trial.

    This is the primitive behind "count it, don't scan for it". A per-tile or
    per-animal chance over N candidates is a binomial, and the *gap* between
    consecutive successes is geometric, so the index of the next success can be
    sampled directly:

        skip = floor( log(U) / log(1 - p) )   failures, then a hit

    Same process and therefore the same distribution as rolling every trial --
    it *is* the same Bernoulli process, only sampled by its waiting time -- but
    the cost is the number of things that actually happen, not the number that
    might have. It turns a region's daily growth from O(tiles) into O(sprouts)
    and its daily die-off from O(animals) into O(deaths).

    ``U`` is taken as ``1 - rng()`` so an injected test RNG keeps its natural
    meaning at the extremes: ``rng() == 0.0`` (a roll below any chance) makes
    every trial a hit, and ``rng() == 1.0`` (above any chance) makes none.
    """
    if n <= 0 or p <= 0.0:
        return
    if p >= 1.0:
        yield from range(n)
        return
    log_survive = log1p(-p)  # negative; the log-probability of one failure
    i = 0
    while i < n:
        u = rng()
        if u >= 1.0:
            # log1p(-1) is -inf: the gap to the next hit is unbounded, so the
            # remaining trials all fail. (Also keeps log1p out of its domain.)
            return
        i += int(log1p(-u) / log_survive)
        if i >= n:
            return
        yield i
        i += 1


def binomial(n: int, p: float, rng: Callable[[], float]) -> int:
    """How many of ``n`` trials succeed at probability ``p`` -- ``bernoulli_hits``
    when only the count is wanted (how many animals died, not which). Costs one
    random number per success, so the common "none of them" answer is one draw."""
    return sum(1 for _ in bernoulli_hits(n, p, rng))


def region_day_rng(base_seed: int, region: tuple[int, int], day: int) -> random.Random:
    """A private RNG for one region's one day.

    Population and growth models are advanced per region per day, but *when* that
    happens is not deterministic -- the idle pump advances whichever region is
    nearest and however many the time budget allows, which differs on a faster
    machine. Deriving the randomness from (region, day) instead of drawing from a
    shared stream makes a region-day's outcome independent of the order and the
    batching, which is the property ``regions.RegionScheduler`` asks for.
    """
    mixed = (
        base_seed
        ^ (region[0] * 0x9E3779B97F4A7C15)
        ^ (region[1] * 0xC2B2AE3D27D4EB4F)
        ^ (day * 0x165667B19E3779F9)
    )
    return random.Random(mixed & 0xFFFFFFFFFFFFFFFF)
