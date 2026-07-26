"""Where everything is: a live index of the world's entities bucketed by
simulation region, so no system ever has to scan the whole world.

Why this exists
---------------
The region scheduler (``regions.py``) decides *when* a region simulates. It
never solved *what to iterate*: every system still reached for
``esper.get_components(...)``, which walks every entity in the world no matter
where the player is. At one island that is a few hundred entities and invisible;
at a hundred islands it is ~85k, and those scans -- not pathfinding -- are what
make a keypress cost more on a big world than a small one.

This module is the answer: one authoritative index, built once when the world is
created and maintained incrementally afterwards, that can answer "what is in
region R" in constant time. With it, an active-region turn costs what one region
holds, not what the world holds, which is the whole point of partitioning.

Strict partitioning
-------------------
The index is what makes the active/inactive split enforceable rather than
aspirational: ``entities_in(region)`` hands a system exactly its region's
entities, so a per-turn system *cannot* accidentally touch a sleeping region's
entities. Inactive regions are simulated only where the design says they may be
-- ``simulate_idle``, ``_catch_up_entered_region_cooperatively``, and sleep.

Keeping it true
---------------
Three things can invalidate a bucket, and each has a hook:

* an entity **moves**       -> ``moved(ent, old_xy, new_xy)`` (the three step sites)
* an entity's **kind changes** (a sapling matures into a tree, someone falls
  asleep)                   -> ``reclassify(ent)``
* an entity is **created or destroyed** -> nothing to call: creation and deletion
  both change esper's own entity population, which the index checks (an O(1)
  ``len``) on every read and repairs by rebuilding. So a missed hook can only
  ever cost one rebuild, never a wrong answer. Direct ``esper.create_entity``
  calls -- including every one in the tests -- stay correct for free.

Indexes are per map and built on demand (``ensure``), so every caller can count on
having one -- systems get a single regional code path instead of a regional path
plus a whole-world fallback. A processor constructed directly in a unit test
simply causes its map to be indexed on first use.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
import weakref

import esper

from components import (
    Bed, BerryBush, BlocksMovement, Blueprint, Camp, Corpse, Deer, Fish, Needs, NPC,
    Personality, Player, Position, Resident, Sapling, Seaweed, Stove, Tree,
)
from game_map import GameMap
from regions import RegionId, region_at

# The component types worth bucketing: every system that used to scan the world
# for one of these now asks the index for its region's share instead. Kinds are
# recomputed for an entity only when it is added or reclassified -- never per turn.
_KINDS: tuple[type, ...] = (
    NPC, Personality, Deer, Fish, Corpse, Tree, BerryBush, Stove, Seaweed, Needs,
    Resident, Player, Bed, Sapling, Blueprint, Camp,
)


class _CreationCounter:
    """Remembers the last entity id esper handed out.

    esper has no "something was created" signal, and no cheap one can be derived
    from its tables (a create plus a delete leaves every length unchanged). So the
    id source itself is wrapped, once -- the whole of the hook, and the thing that
    lets every ``esper.create_entity`` in the game and in the tests stay untouched
    while the index still notices them. Ids are issued in order, so "the last id"
    also names exactly which entities are new since any earlier look.
    """

    def __init__(self, inner, last_id: int) -> None:
        self._inner = inner
        self.last_id = last_id

    def __iter__(self) -> "_CreationCounter":
        return self

    def __next__(self) -> int:
        self.last_id = next(self._inner)
        return self.last_id


def _last_entity_id() -> int:
    """The highest entity id esper has issued. Installs the counter on first ask,
    seeding it from the ids already out there (this may be a world that was built
    before anything asked)."""
    counter = esper._entity_count
    if not isinstance(counter, _CreationCounter):
        counter = _CreationCounter(counter, max(esper._entities, default=0))
        esper._entity_count = counter
    return counter.last_id


class SpatialIndex:
    """Every positioned entity, bucketed by region and by kind.

    ``blockers`` is kept as a plain ``(x, y) -> entity`` dict of the *static*
    blockers (everything that blocks movement and isn't a creature), maintained
    in place instead of rescanned: the AI treats it as a map overlay and reads it
    by tile, so a dict is exactly the right shape -- it just must never be
    rebuilt from a world scan again.
    """

    def __init__(self, game_map: GameMap) -> None:
        self.game_map = game_map
        self._by_region: dict[RegionId, set[int]] = {}
        self._by_kind: dict[tuple[RegionId, type], set[int]] = {}
        self._region_of: dict[int, RegionId] = {}
        self._kinds_of: dict[int, tuple[type, ...]] = {}
        self.blockers: dict[tuple[int, int], int] = {}
        # The same blocker map read the other way (entity -> its tile), so removing
        # one costs a lookup rather than a walk over every blocker in the world.
        self._blocker_tile: dict[int, tuple[int, int]] = {}
        # Bumped whenever a region's *membership* changes -- something appears,
        # disappears, changes kind, or crosses in or out. Walking about inside one
        # region doesn't count. Systems cache per-region work against this, so a
        # cache rebuilds when its region actually changed rather than on a timer.
        self._version: dict[RegionId, int] = {}
        # The same counter, but per (region, kind). A cache over one kind of thing
        # -- the AI's tree/bush goal maps -- must not be thrown away every time a
        # deer wanders across a seam, which is what the region-wide counter above
        # would do. Trees change when a tree grows or is felled, and that is all
        # this moves for.
        self._kind_version: dict[tuple[RegionId, type], int] = {}
        # esper's population as of the last sync. Creation/deletion anywhere moves
        # this, which is how the index notices work it wasn't told about.
        self._population: tuple[int, int, int] = (-1, -1, -1)
        self.rebuilds = 0  # diagnostics; a healthy live game leaves this alone
        self.rebuild()

    # --- population tracking ---------------------------------------------

    @staticmethod
    def _population_now() -> tuple[int, int, int]:
        """A signature that changes whenever an entity is created or destroyed, in
        O(1).

        The last id issued has to be in it, not just how many are alive:
        a turn where a deer dies and its corpse appears leaves the population
        exactly as it was, and an index that only watched the total would keep
        handing out the dead deer. ``_dead_entities`` matters too -- a deferred
        delete lands there first and only leaves ``_entities`` at the next
        ``esper.process``.
        """
        return (_last_entity_id(), len(esper._entities), len(esper._dead_entities))

    def sync(self) -> None:
        """Fold in the entities created or destroyed since the last look.

        Almost always a single tuple comparison. When something did change, the
        work is proportional to *what changed*, not to the world: esper hands out
        entity ids in order, so the ids created since last time are exactly the
        next run of integers, and only a death forces a pass over what we hold.
        A full rebuild is the last resort (a new or cleared world).
        """
        signature = self._population_now()
        if signature == self._population:
            return
        last_id, alive, dead = signature
        prev_last_id, prev_alive, prev_dead = self._population
        if last_id < prev_last_id or prev_last_id < 0:
            self.rebuild()  # the world was cleared or replaced underneath us
            return

        positioned = esper._components.get(Position, set())
        for ent in range(prev_last_id + 1, last_id + 1):
            if ent in positioned:
                self._insert(ent, esper.component_for_entity(ent, Position))
        if (alive - prev_alive) != (last_id - prev_last_id) or dead != prev_dead:
            # Something died. Which is only answerable by comparing what we hold
            # against what still lives -- but only on the turns a death happened.
            for ent in [e for e in self._region_of if e not in positioned]:
                self._forget(ent)
        self._population = signature

    def rebuild(self) -> None:
        """One full scan. Runs at world creation, and whenever the population
        changed without a hook -- never on a steady-state turn."""
        self.rebuilds += 1
        self._by_region = {}
        self._by_kind = {}
        self._region_of = {}
        self._kinds_of = {}
        self.blockers = {}
        self._blocker_tile = {}
        for ent, (pos,) in esper.get_components(Position):
            self._insert(ent, pos)
        self._population = self._population_now()

    # --- maintenance ------------------------------------------------------

    def _kinds_for(self, ent: int) -> tuple[type, ...]:
        return tuple(kind for kind in _KINDS if esper.has_component(ent, kind))

    def _is_static_blocker(self, ent: int) -> bool:
        """Static blockers are the furniture of the world -- trees, walls-in-a-box,
        wells, ovens, beds. Creatures block too, but they move every turn and the
        AI deliberately ignores creature-on-creature collision, so they are left
        out (see ``NpcAiProcessor._advance_region``)."""
        return (
            esper.has_component(ent, BlocksMovement)
            and not esper.has_component(ent, NPC)
            and not esper.has_component(ent, Player)
        )

    def _bump(self, region: RegionId, kinds: Iterable[type] = ()) -> None:
        self._version[region] = self._version.get(region, 0) + 1
        for kind in kinds:
            key = (region, kind)
            self._kind_version[key] = self._kind_version.get(key, 0) + 1

    def _insert(self, ent: int, pos: Position) -> None:
        region = region_at(self.game_map, pos.x, pos.y)
        kinds = self._kinds_for(ent)
        self._bump(region, kinds)
        self._region_of[ent] = region
        self._by_region.setdefault(region, set()).add(ent)
        self._kinds_of[ent] = kinds
        for kind in kinds:
            self._by_kind.setdefault((region, kind), set()).add(ent)
        if self._is_static_blocker(ent):
            self.blockers[(pos.x, pos.y)] = ent
            self._blocker_tile[ent] = (pos.x, pos.y)

    def _forget(self, ent: int) -> None:
        """Drop a destroyed entity from every bucket it was in."""
        region = self._region_of.pop(ent, None)
        if region is None:
            return
        kinds = self._kinds_of.pop(ent, ())
        self._bump(region, kinds)
        self._by_region.get(region, set()).discard(ent)
        for kind in kinds:
            self._by_kind.get((region, kind), set()).discard(ent)
        tile = self._blocker_tile.pop(ent, None)
        if tile is not None and self.blockers.get(tile) == ent:
            del self.blockers[tile]

    def moved(self, ent: int, old_xy: tuple[int, int], new_xy: tuple[int, int]) -> None:
        """Record a step. Called by the three places that move an entity."""
        if old_xy == new_xy:
            return
        old_region = self._region_of.get(ent)
        if old_region is None:
            return  # not indexed (created since the last sync); the sync will place it
        new_region = region_at(self.game_map, *new_xy)
        if self.blockers.get(old_xy) == ent:
            del self.blockers[old_xy]
            self.blockers[new_xy] = ent
            self._blocker_tile[ent] = new_xy
        if new_region == old_region:
            return
        kinds = self._kinds_of.get(ent, ())
        self._bump(old_region, kinds)
        self._bump(new_region, kinds)
        self._by_region.get(old_region, set()).discard(ent)
        self._by_region.setdefault(new_region, set()).add(ent)
        for kind in kinds:
            self._by_kind.get((old_region, kind), set()).discard(ent)
            self._by_kind.setdefault((new_region, kind), set()).add(ent)
        self._region_of[ent] = new_region

    def reclassify(self, ent: int) -> None:
        """Re-read an entity's kinds after a component was added or removed at
        runtime (a sapling maturing into a tree, a bush being picked)."""
        region = self._region_of.get(ent)
        if region is None:
            return
        was = self._kinds_of.get(ent, ())
        self._bump(region, was)
        for kind in was:
            self._by_kind.get((region, kind), set()).discard(ent)
        if not esper.entity_exists(ent) or not esper.has_component(ent, Position):
            self._kinds_of.pop(ent, None)
            self._forget(ent)
            return
        kinds = self._kinds_for(ent)
        # Only the kinds it gained; the ones it lost were bumped just above.
        self._bump(region, (kind for kind in kinds if kind not in was))
        self._kinds_of[ent] = kinds
        for kind in kinds:
            self._by_kind.setdefault((region, kind), set()).add(ent)
        pos = esper.component_for_entity(ent, Position)
        if self._is_static_blocker(ent):
            self.blockers[(pos.x, pos.y)] = ent
            self._blocker_tile[ent] = (pos.x, pos.y)
        elif self.blockers.get((pos.x, pos.y)) == ent:
            del self.blockers[(pos.x, pos.y)]
            self._blocker_tile.pop(ent, None)

    # --- queries ----------------------------------------------------------

    def entities_in(self, region_id: RegionId) -> set[int]:
        self.sync()
        return self._by_region.get(region_id, set())

    def of_kind(self, region_id: RegionId, kind: type) -> set[int]:
        """The entities of one kind standing in one region."""
        self.sync()
        return self._by_kind.get((region_id, kind), set())

    def region_of(self, ent: int) -> RegionId | None:
        self.sync()
        return self._region_of.get(ent)

    def neighborhood_version(self, region_id: RegionId) -> tuple[int, ...]:
        """A cache key for work that reads a region *and* its eight neighbours --
        which is what any region-scoped search does, since goal searches are
        widened across seams by ``_REGION_BORDER_MARGIN``. Changes exactly when
        one of those nine regions gains, loses or reclassifies an entity.
        """
        self.sync()
        cx, cy = region_id
        return tuple(
            self._version.get((x, y), 0)
            for y in range(cy - 1, cy + 2)
            for x in range(cx - 1, cx + 2)
        )

    def kind_neighborhood_version(self, region_id: RegionId, kind: type) -> tuple[int, ...]:
        """``neighborhood_version`` narrowed to one kind: changes exactly when one
        of the nine regions gains, loses or reclassifies an entity **of that
        kind**. What a cache over a single resource type wants, so a passing deer
        doesn't invalidate the map of where the trees are."""
        self.sync()
        cx, cy = region_id
        return tuple(
            self._kind_version.get(((x, y), kind), 0)
            for y in range(cy - 1, cy + 2)
            for x in range(cx - 1, cx + 2)
        )

    def blocker_at(self, x: int, y: int) -> int | None:
        """The static blocker on a tile, in O(1) -- the whole reason a player step
        no longer builds a dict of every blocking entity in the world."""
        self.sync()
        return self.blockers.get((x, y))

    def components(self, region_id: RegionId, *kinds: type) -> Iterator[tuple[int, tuple]]:
        """``esper.get_components``, scoped to one region.

        Yields ``(entity, (component, ...))`` for the region's entities that have
        every requested component, in ascending entity order so iteration is as
        deterministic as esper's own.
        """
        self.sync()
        if not kinds:
            return
        indexed = [k for k in kinds if k in _KINDS]
        if indexed:
            candidates: set[int] | None = None
            for kind in indexed:
                bucket = self._by_kind.get((region_id, kind), set())
                candidates = set(bucket) if candidates is None else (candidates & bucket)
            candidates = candidates or set()
        else:
            candidates = set(self._by_region.get(region_id, set()))
        for ent in sorted(candidates):
            if not esper.entity_exists(ent):
                continue
            if all(esper.has_component(ent, kind) for kind in kinds):
                yield ent, tuple(esper.component_for_entity(ent, kind) for kind in kinds)

    # --- diagnostics ------------------------------------------------------

    def audit(self) -> list[str]:
        """Differences between the index and a fresh whole-world scan. Empty means
        the index is telling the truth; used by the tests to prove the incremental
        maintenance stays correct across a long simulation."""
        problems: list[str] = []
        truth_region: dict[int, RegionId] = {}
        truth_blockers: dict[tuple[int, int], int] = {}
        for ent, (pos,) in esper.get_components(Position):
            truth_region[ent] = region_at(self.game_map, pos.x, pos.y)
            if self._is_static_blocker(ent):
                truth_blockers[(pos.x, pos.y)] = ent
        if truth_region != self._region_of:
            missing = set(truth_region) - set(self._region_of)
            extra = set(self._region_of) - set(truth_region)
            misplaced = {
                e for e in set(truth_region) & set(self._region_of)
                if truth_region[e] != self._region_of[e]
            }
            problems.append(
                f"regions differ: {len(missing)} missing, {len(extra)} stale, "
                f"{len(misplaced)} in the wrong region"
            )
        if truth_blockers != self.blockers:
            problems.append(
                f"blockers differ: index {len(self.blockers)} vs world {len(truth_blockers)}"
            )
        for (region, kind), bucket in self._by_kind.items():
            for ent in bucket:
                if not esper.entity_exists(ent):
                    problems.append(f"{kind.__name__} bucket {region} holds dead entity {ent}")
                    break
                if not esper.has_component(ent, kind):
                    problems.append(f"{kind.__name__} bucket {region} holds unrelated {ent}")
                    break
                if truth_region.get(ent) != region:
                    problems.append(f"{kind.__name__} bucket {region} holds misplaced {ent}")
                    break
        return problems


# An index is meaningful only for the map whose geometry it bucketed by, so they
# are kept per map and handed out against one. Weak keys: an index disappears with
# the world it describes, which is what keeps one test's map (or a previous
# session's) from ever answering questions about another's.
_INDEXES: "weakref.WeakKeyDictionary[GameMap, SpatialIndex]" = weakref.WeakKeyDictionary()


def ensure(game_map: GameMap) -> SpatialIndex:
    """The index for ``game_map``, built (one scan) if this is the first ask.

    Every caller can rely on getting one, so systems have a single code path
    rather than a regional path plus a whole-world fallback. Building on demand is
    what makes that safe for a processor constructed directly in a unit test.
    """
    index = _INDEXES.get(game_map)
    if index is None:
        index = SpatialIndex(game_map)
        _INDEXES[game_map] = index
    return index


def attach(game_map: GameMap) -> SpatialIndex:
    """Index a freshly generated world, up front rather than on first use."""
    return ensure(game_map)


def detach() -> None:
    """Forget every index. Tests use this to start from a clean world."""
    _INDEXES.clear()


def index_for(game_map: GameMap) -> SpatialIndex | None:
    """The index for ``game_map`` if one exists, without building one."""
    return _INDEXES.get(game_map)


def moved(ent: int, old_xy: tuple[int, int], new_xy: tuple[int, int]) -> None:
    """Tell every live index that an entity stepped. Called from the three places
    anything moves; a no-op for an entity no index has heard of."""
    for index in list(_INDEXES.values()):
        index.moved(ent, old_xy, new_xy)


def reclassify(ent: int) -> None:
    """Tell every live index that an entity's components changed."""
    for index in list(_INDEXES.values()):
        index.reclassify(ent)


def component_population(component_type: type) -> int:
    """How many entities carry ``component_type``, in O(1).

    A change signal for the handful of things the index can't bucket because they
    have no position (construction sites). Cheap enough to check every turn, which
    is the point: it turns "rescan the world in case something changed" into
    "rebuild only when something did".
    """
    return len(esper._components.get(component_type, ()))


