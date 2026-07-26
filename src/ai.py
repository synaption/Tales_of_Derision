"""Creature AI: the per-turn brains for land NPCs (``NpcAiProcessor``) and sea life
(``FishAiProcessor``).

Both drive their creatures through the region scheduler (far regions lag and catch
up nearest-first; see wiki/World-Simulation.md) rather than simulating the whole map
every turn. Split out of ``systems`` because together they are ~1200 lines and share
the region-simulation machinery; they import the behaviour helpers (social, family,
sleep, construction, combat, time) from ``systems``. Import via ``systems`` (which
re-exports them), never directly, to keep module-load order sound.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import replace

import esper

from components import (
    Actor, Asleep, BerryBush, BlocksMovement, Blueprint, Corpse, Deer, Diet, DriveProfile,
    Enemy, Fish, Friendly, Home, Inventory, NPC, Needs, Personality, Player, Position,
    Relationships, Resident, Seaweed, Settled, Stove, Tree, Vision, WorldClock,
)
from game_map import GameMap
from content.drives import DriveDef, all_drives
from regions import RegionId, RegionScheduler, all_region_ids, in_region_with_margin
import spatial
from action import BASE_ACTION_COST, action_cost
from items import WOOD, cook_meat, hunger_restored, is_cooked_meat, is_raw_meat
from rng import world_rng
from systems import (
    _DRINK_RESTORE, _EXHAUSTED_THRESHOLD, _FEED_RESTORE, _FISH_BACKGROUND_BUDGET,
    _FISH_GRAZE_RESTORE, _FISH_SIGHT, _FISH_WANDER_CHANCE, _FORAGE_THRESHOLD,
    _GRAZE_RESTORE, _HAUL_BATCH, _MAX_ACTIONS_PER_REGION_TURN, _NPC_BACKGROUND_BUDGET,
    _OSC_GUARD, _PATH_FIELD_REFRESH_CALLS, _REGION_BORDER_MARGIN, _SLEEP_THRESHOLD,
    _SOCIAL_COOLDOWN, _SOCIAL_DISTANCE_PENALTY, _SOCIAL_SIGHT, _STATIC_SNAPSHOT_REFRESH_CALLS,
    _TURN_ACTIONS, _WORLD_SNAPSHOT_REFRESH_CALLS, _chebyshev, _current_region_turn,
    _set_blueprint_stocked, friendship, go_to_sleep, interact, owned_bed_of,
    pick_berries, raise_blueprint, settle_sleep, slay_entity, sleep_turns_needed,
    try_marry, try_mate, world_clock,
)

# How far the occupant-aware fallback pathfind may search around a creature. Big
# enough to round any local cluster of blockers, small enough that the search is
# cheap; long-range travel rides the cached flow field, not this fallback.
_LOCAL_PATH_RADIUS = 12

# Full-simulation box (width x height) centred on the player. An NPC inside it
# gets full, occupant-aware, flow-field pathfinding every turn; one outside walks
# a cached concrete route (``_far_path_step``) and pays a real pathfind only on a
# cache miss -- no per-turn flow-field builds for the distant world. Sized to the
# player's current map-tile plus its neighbours.
_FULL_SIM_HALF_W = 80   # 160 wide
_FULL_SIM_HALF_H = 40   # 80 tall

# How many tiles of a trip an unwatched NPC may walk in a single region-turn.
# Outside the full-simulation box nobody can see the walk, only its outcome, so a
# journey is settled as one *activity*: the NPC arrives and is charged the whole
# journey's time (see ``_far_travel``), instead of being woken to re-decide its
# life once per tile. The cap exists so a very long trek still re-evaluates its
# drives a few times -- an NPC shouldn't cross an island without noticing it grew
# hungry on the way -- and so the time debt a single decision can incur is bounded.
_COMPACTED_TRAVEL_STEPS = 32

# ---------------------------------------------------------------------------
# Compacted activities
# ---------------------------------------------------------------------------
# Outside the full-simulation box nobody can watch an NPC spend its turns; only
# the outcome is ever observed. So an activity that is N region-turns of the same
# unobservable repetition -- walking a road, hauling load after load of wood,
# sleeping a night through -- is *settled* in a single region-turn and the actor
# is then billed the whole N. The NPC ends up where and how it would have ended
# up, on the turn it would have got there; what disappears is N-1 rounds of
# re-ranking its drives and re-scanning its surroundings, which is the expensive
# part.
#
# There are exactly two mechanisms, and a new activity picks whichever fits:
#
#  1. **Bill the time** -- ``_charge_activity(ent, turns)``. This is the whole
#     scheduler: the charge overdraws ``Actor.energy``, and ``_advance_region``'s
#     energy loop simply doesn't call the NPC again until the following
#     region-turns have paid the balance off. That *is* "busy for the next N
#     turns", with no second clock, no busy flag and no queue to keep in sync.
#     Needs keep accruing throughout, so the activity is genuinely tiring.
#     Used by travel and hauling, whose per-turn effects really do happen over
#     those turns.
#
#  2. **Settle the turns** -- apply the whole activity's effect in closed form and
#     tag the entity ``Settled(activity, turns)``. Per-turn systems skip a
#     ``Settled`` entity and burn one turn off the receipt instead, so they can't
#     live those turns twice. Used by sleep, which is pure arithmetic and whose
#     "busy" state is already the ``Asleep`` tag.
#
# For repeating activities the generic driver is ``_run_compacted``: hand it the
# activity's ordinary one-turn body and it runs turns of it until the body says
# there's nothing left to do, then bills the difference. An activity written that
# way never has to know it's being compacted.
#
# Every activity has a ``_COMPACTED_*`` cap, all for the same reason: an activity
# settled in one decision is a decision made without noticing anything that
# happens during it, so a long one is broken into legs that re-decide.

# How many turns of a repeating activity ``_run_compacted`` may fold into one
# region-turn. Deliberately smaller than the travel cap: each of these turns can
# contain a whole travel leg of its own, so a hauling round trip already spans
# far more than eight tiles.
_COMPACTED_ACTIVITY_TURNS = 8

# The longest sleep settled as one activity. A full night is ~34 turns (100
# tiredness at _SLEEP_RECOVERY = 3 a turn); the cap only bites on a creature that
# lies down utterly spent, and keeps a single decision from writing off a whole
# day in which the world around it moved on.
_COMPACTED_SLEEP_TURNS = 120


class NpcAiProcessor(esper.Processor):
    """Drives NPC behaviour each turn.

    Priority order per creature:
      1. Satisfy an urgent need -- drink at water, or (by diet) graze a tree or
         hunt prey.
      2. Otherwise, hostiles chase the player when they can see them.

    Water tiles and the walkable "shore" tiles beside them are precomputed once
    (the map is static) so a thirsty animal is a cheap nearest-lookup plus one
    pathfind, not a full-map rescan every turn.
    """

    def __init__(
        self,
        game_map: GameMap,
        wall_clock: Callable[[], float] | None = None,
        max_entry_catchup_advances: int | None = None,
    ):
        self.game_map = game_map
        self._max_entry_catchup_advances = max_entry_catchup_advances
        self._shore_tiles: list[tuple[int, int]] = self._compute_shore_tiles()
        self._wall_clock = wall_clock if wall_clock is not None else time.monotonic
        # The player's tile as of the region-advance in flight. An NPC within the
        # full-simulation box around it (``_far_from_player``) gets full, occupant-
        # aware, per-turn pathfinding; one beyond it walks a cached concrete route
        # and re-pathfinds only on a cache miss. ``None`` (no player: unit tests,
        # whole-world catch-up) means "full sim for everyone" -- exact, fully
        # pathfound behaviour.
        self._player_xy: tuple[int, int] | None = None
        # ent -> (goal, path, cursor) for an NPC currently beyond the box. ``path``
        # is start-inclusive, so ``path[cursor]`` is where the NPC stands and
        # ``path[cursor + 1]`` its next step. Reused every turn with no pathfinding
        # until the goal changes, the route runs out, the NPC drifts off it, or a
        # tile on it stops being walkable (see ``_far_path_step``). Cleared on a
        # world switch so a recycled entity id can never inherit a stale route.
        self._trip_cache: dict[
            int, tuple[tuple[int, int], list[tuple[int, int]], int]
        ] = {}
        self._trip_cache_world: str | None = None
        self.scheduler = RegionScheduler(game_map, _current_region_turn())
        self.scheduler.register("npc_ai", self._advance_region)
        # Shore tiles bucketed by simulation region (like the resource snapshot), so
        # a thirsty NPC scans only nearby shores, not every shore in the world -- the
        # flat list is O(all shores) per drink and dominates at archipelago scale.
        # Shores are static (drawn from the fixed water layout), so this is built once.
        self._shore_by_region: dict[RegionId, list[tuple[int, int]]] = {}
        for shore in self._shore_tiles:
            self._shore_by_region.setdefault(
                self.scheduler.region_at(shore[0], shore[1]), []
            ).append(shore)
        # What a region's NPCs can see, gathered from the spatial index and kept
        # per region -- never from a world scan. This replaced a pair of snapshots
        # that rebuilt every entity bucket in the world on a timer: correct, but
        # its cost was the world's population, so simulating one island's turn got
        # slower every time another island was added.
        #
        # Two caches, because the two halves go stale for different reasons:
        #  * static (trees, ovens, shore) can only change when its nine regions
        #    gain or lose something, so it is keyed on the index's version of that
        #    neighbourhood -- exact, and usually untouched for hundreds of turns.
        #  * dynamic (deer, corpses, berry bushes, people) drifts as things move
        #    and ripen, so it is refreshed every _WORLD_SNAPSHOT_REFRESH_CALLS
        #    advances of *this* region -- the same staleness the world-wide
        #    snapshot allowed, now paid per region instead of per world.
        self._static_region_cache: dict[RegionId, tuple[tuple[int, ...], tuple[list, ...]]] = {}
        self._dynamic_region_cache: dict[
            RegionId, tuple[tuple[int, ...], int, tuple[list, ...]]
        ] = {}
        # goal xy -> (connectivity revision key, the flow field itself). Shared
        # across every NPC heading to the same goal, not per-entity -- a distance field
        # rooted at a (largely static) goal stays valid for any traveller
        # approaching it from anywhere, so many NPCs reuse the one flood.
        self._field_cache: dict[
            tuple[int, int], tuple[tuple[int | None, int], dict[tuple[int, int], int]]
        ] = {}
        # Goal maps (see ``_goal_map``): (kind, region) -> (key, source tiles,
        # multi-source distance field). One flood answers "where is the nearest
        # tree/shore/bush" for every creature in the region at once, instead of a
        # flood per creature per chosen target tile.
        self._goal_map_cache: dict[
            tuple[str, RegionId, int | None],
            tuple[tuple, dict[tuple[int, int], int], dict[tuple[int, int], int]],
        ] = {}
        # Per-map-revision memo for walkable connected-region ids. Many drives
        # filter dozens of local resource candidates with same-region checks;
        # caching the labels for each tile turns those scans into dict lookups
        # and avoids repeatedly touching GameMap's island flood-fill cache.
        self._region_of_cache_revision = -1
        self._region_of_cache: dict[tuple[int, int], int | None] = {}
        # ent -> the tile it stood on at the start of its previous turn. Used to
        # forbid an immediate one-tile reversal (see ``_advance_region``), which
        # is the only way an NPC ends up flip-flopping between two tiles forever.
        self._prev_turn_pos: dict[int, tuple[int, int]] = {}
        # Debug/inspection hook for data-driven AI: ent -> per-drive allow/weight
        # records from its most recent decision. Tests and a future UI panel can
        # read this without re-running any triggers.
        self.last_decisions: dict[int, list[dict]] = {}

    def _compute_shore_tiles(self) -> list[tuple[int, int]]:
        shore: list[tuple[int, int]] = []
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                if not self.game_map.is_walkable(x, y):
                    continue
                if any(self.game_map.is_water(nx, ny) for nx, ny in self.game_map.neighbors_8(x, y)):
                    shore.append((x, y))
        return shore

    def _find_player_position(self) -> tuple[int, int] | None:
        for _ent, (pos, _player) in esper.get_components(Position, Player):
            return (pos.x, pos.y)
        return None

    @staticmethod
    def _nearest(origin: tuple[int, int], candidates: list[tuple[int, int]]) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        best_dist = None
        for cand in candidates:
            dist = _chebyshev(origin, cand)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = cand
        return best

    def _distance_field_for(self, goal: tuple[int, int]) -> dict[tuple[int, int], int]:
        """A cached flow field to ``goal`` (see ``GameMap.distance_field``).

        ``distance_field`` is a pure function of the goal and the walkable
        connected component that contains it. The archipelago's components never
        cross the water between islands, so an edit on another island cannot
        affect this field. Cache by ``GameMap.connectivity_revision`` instead of
        the global map revision to avoid rebuilding every flow field whenever an
        unrelated off-screen villager raises a wall.
        """
        connectivity_rev = self.game_map.connectivity_revision(goal[0], goal[1])
        cached = self._field_cache.get(goal)
        if cached is not None:
            cached_rev, field = cached
            if cached_rev == connectivity_rev:
                return field
        field = self.game_map.distance_field(goal)
        self._field_cache[goal] = (connectivity_rev, field)
        return field

    # --- Goal maps ----------------------------------------------------------
    #
    # A *goal map* is one multi-source distance field per (resource kind, region):
    # every tree (or shore tile, or ripe bush) around the region is seeded at
    # distance 0 in a single flood, so the value at any tile is its walking
    # distance to the nearest one and stepping downhill walks to it.
    #
    # It replaces the older shape -- pick the nearest candidate by straight-line
    # distance, then flood the island to *that* tile -- which cost a full island
    # BFS per creature per chosen target, because a dozen creatures choosing a
    # dozen different trees is a dozen different goals. One flood now serves the
    # whole region's population and every creature in it steps for the price of
    # eight dict lookups. It is also more truthful: the winner is nearest by
    # *walking*, not by straight line, and a source across a river or behind a
    # wall simply isn't in the field -- which is the same thing ``_reachable``
    # was doing with an O(sources) same-region scan per creature per turn.

    def _goal_map(
        self,
        kind: str,
        region_id: RegionId,
        island: int | None,
        build_sources: Callable[[], Iterable[tuple[tuple[int, int], int]]],
        index_kind: type | None,
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
        """``(source tile -> entity, distance field)`` for ``kind`` around
        ``region_id``, on the walkable component ``island``.

        Seeds are filtered to ``island`` first. A region's widened source list can
        straddle several islands, and seeding them all would flood every one of
        them in a single pass -- measured at four islands' worth of tiles per
        flood, four times the work for three islands' worth of answers nobody
        standing here can walk to. This is the same "don't fixate on food across
        the water" rule ``_reachable`` enforced, except it is now paid once per
        flood instead of once per creature per turn.

        Keyed on the index's **per-kind** neighbourhood version (``index_kind``;
        terrain sources like the shore have none and key on the map alone) plus
        the map revision. So the flood is rebuilt exactly when a source of this
        kind appears or disappears nearby, or a tile edit changes what can reach
        what -- and never merely because some deer crossed a seam, which is what
        the region-wide version would have meant.

        A source being *depleted* rather than removed (a tree losing a log) moves
        neither key, so the map holds still while creatures work through it.

        The tile -> entity map falls out of the same pass, and is what lets a
        creature interact with what it arrived at without searching for it.
        """
        key = (
            self._index().kind_neighborhood_version(region_id, index_kind)
            if index_kind is not None
            else (),
            self.game_map.revision,
        )
        cached = self._goal_map_cache.get((kind, region_id, island))
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        sources = {
            xy: source_ent
            for xy, source_ent in build_sources()
            if self._region_of_cached(xy) == island
        }
        field = self.game_map.distance_field_from(sources)
        self._goal_map_cache[(kind, region_id, island)] = (key, sources, field)
        return sources, field

    def _adjacent_source(
        self, pos: Position, sources: dict[tuple[int, int], int]
    ) -> tuple[int, int] | None:
        """A source tile the creature is standing next to, or ``None``. Eight dict
        lookups, where finding the nearest candidate used to be a scan of every
        resource in the region followed by a distance sort."""
        for nxy in self.game_map.neighbors_8(pos.x, pos.y):
            if nxy in sources:
                return nxy
        return None

    def _step_down(
        self,
        ent: int,
        pos: Position,
        field: dict[tuple[int, int], int],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Take one step down ``field`` -- toward the nearest source of whatever
        the map was built for. Returns True if it moved.

        Ranks every downhill neighbour rather than only the best, and skips ones a
        live occupant blocks, for the same reason ``_greedy_step_toward`` does: a
        crowded resource shouldn't pin a creature in place when the second-best
        step is just as good and already in the same field.
        """
        here = field.get((pos.x, pos.y))
        if here is None:
            return False  # nothing of this kind is reachable from where we stand
        candidates = [
            (dist, nxy)
            for nxy in self.game_map.neighbors_8(pos.x, pos.y)
            if (dist := field.get(nxy)) is not None and dist < here
        ]
        # Sort on the distance alone, so equal-distance neighbours keep
        # ``neighbors_8`` order -- same tie-break as ``_greedy_step_toward``, which
        # prefers the straight step over the diagonal that costs the same.
        candidates.sort(key=lambda c: c[0])
        for _dist, nxy in candidates:
            if nxy not in occupied or occupied[nxy] == ent:
                self._commit_step(ent, pos, nxy, occupied)
                return True
        return False

    def _go_to_nearest(
        self,
        ent: int,
        pos: Position,
        kind: str,
        build_sources: Callable[[], Iterable[tuple[tuple[int, int], int]]],
        occupied: dict[tuple[int, int], int],
        region_id: RegionId | None = None,
        index_kind: type | None = None,
        on_source: bool = False,
    ) -> tuple[tuple[int, int] | None, int | None, bool]:
        """Head for the nearest ``kind``. Returns ``(tile, entity, acted)``.

        ``tile``/``entity`` are the source the creature has reached, or ``None``
        if it hasn't yet. ``acted`` says whether the turn was spent -- False means
        nothing of this kind is reachable and it couldn't even step, so the drive
        should fail and let the next one have the turn.

        Arrival is *beside* a source by default -- you chop a tree or pick a bush
        from an adjacent tile, and its own tile is usually blocked anyway. Pass
        ``on_source`` for the kinds you have to stand on, like a shore tile.

        **Watched creatures use the goal map; unwatched ones don't**, and that
        split is measured rather than aesthetic. A goal map costs one island-wide
        flood and then serves every creature that shares it for eight dict lookups
        a step -- an enormous win in the player's own region, which is simulated
        every single turn by everyone standing in it (per-turn cost fell 2-13x).
        Out in the lagging world a region gets one turn at a time and its trees
        churn as its creatures eat them, so the same flood was bought over and
        over for a handful of uses: 75 tree floods for ~205 uses, 2 ms each, and
        whole-world catch-up went *up* 47%. There the old shape -- scan the
        region's own short list for the nearest, walk it on a cached route -- is
        simply cheaper, and nobody is watching the difference anyway.
        """
        if region_id is None:
            region_id = self.scheduler.region_at(pos.x, pos.y)
        island = self._region_of_cached((pos.x, pos.y))
        if self._compactable((pos.x, pos.y)):
            return self._go_to_nearest_unwatched(
                ent, pos, island, build_sources, occupied, on_source
            )
        sources, field = self._goal_map(kind, region_id, island, build_sources, index_kind)
        if not sources:
            return None, None, False

        def reached() -> tuple[int, int] | None:
            here = (pos.x, pos.y)
            if on_source:
                return here if here in sources else None
            return self._adjacent_source(pos, sources)

        arrived = reached()
        if arrived is not None:
            return arrived, sources[arrived], True
        moved = self._step_down(ent, pos, field, occupied)
        arrived = reached()
        return (arrived, sources[arrived] if arrived is not None else None, moved)

    def _go_to_nearest_unwatched(
        self,
        ent: int,
        pos: Position,
        island: int | None,
        build_sources: Callable[[], Iterable[tuple[tuple[int, int], int]]],
        occupied: dict[tuple[int, int], int],
        on_source: bool,
    ) -> tuple[tuple[int, int] | None, int | None, bool]:
        """``_go_to_nearest`` for a creature nobody can see: pick the nearest
        source off the region's own list and walk to it on a cached route.

        No goal map out here -- see ``_go_to_nearest`` for why it doesn't pay. The
        list is already region-scoped, so this is a scan of what's nearby rather
        than of the world, and the walk itself is compacted by ``_step_toward``
        exactly as before.
        """
        here = (pos.x, pos.y)
        sources = [
            (xy, source_ent)
            for xy, source_ent in build_sources()
            if self._region_of_cached(xy) == island
            and not (on_source and xy != here and xy in occupied)
        ]
        if not sources:
            return None, None, False
        target_xy, target_ent = min(sources, key=lambda item: _chebyshev(here, item[0]))
        reach = 0 if on_source else 1
        if _chebyshev(here, target_xy) <= reach:
            return target_xy, target_ent, True
        return None, None, self._step_toward(ent, pos, target_xy, occupied)

    def _greedy_step_toward(
        self,
        ent: int,
        xy: tuple[int, int],
        goal: tuple[int, int],
        occupied: dict[tuple[int, int], int],
    ) -> tuple[int, int] | None:
        """The best available neighbour of ``xy`` toward ``goal`` per the
        cached flow field, or ``None`` if ``goal`` isn't in it (out of its
        walkable region, the field hasn't been built for it, or literally
        every closer neighbour is currently blocked).

        Ranks *every* closer neighbour, not just the single nearest one, and
        skips ones a live occupant blocks -- so a crowded goal (e.g. a
        builder's own other, not-yet-placed blueprint pieces sitting right
        next to this one) doesn't force an expensive fallback pathfind merely
        because the closest step happens to be taken; the second- or
        third-closest is usually just as good and is right here in the same
        cached field.
        """
        field = self._distance_field_for(goal)
        here = field.get(xy)
        if here is None:
            return None
        candidates = [
            (dist, nxy)
            for nxy in self.game_map.neighbors_8(xy[0], xy[1])
            if (dist := field.get(nxy)) is not None and dist < here
        ]
        candidates.sort(key=lambda c: c[0])
        for _dist, nxy in candidates:
            blocked_by_goal = nxy == goal and nxy in occupied
            blocked_by_other = nxy in occupied and occupied[nxy] != ent
            if not blocked_by_goal and not blocked_by_other:
                return nxy
        return None

    def _far_from_player(self, xy: tuple[int, int]) -> bool:
        """True when ``xy`` lies outside the full-simulation box centred on the
        player, so the NPC there should move by cached path rather than per-turn
        pathfinding. Always False when there is no player (unit tests, whole-world
        catch-up), which want exact, fully-pathfound behaviour."""
        p = self._player_xy
        if p is None:
            return False
        return abs(xy[0] - p[0]) > _FULL_SIM_HALF_W or abs(xy[1] - p[1]) > _FULL_SIM_HALF_H

    def _far_travel(
        self,
        ent: int,
        pos: Position,
        goal: tuple[int, int],
        occupied: dict[tuple[int, int], int],
    ) -> int:
        """Walk a whole leg of a trip toward ``goal`` in one region-turn, for an NPC
        beyond the full-sim box. Returns the number of tiles covered (0 = it didn't
        move); the caller charges the traveller for all of them.

        Out here the walk itself is unobservable -- nobody is watching an NPC on
        another island put one foot in front of the other -- so there is nothing to
        gain from spending a region-turn per tile, and a great deal to lose: every
        one of those turns re-ranks the creature's drives and re-scans its
        surroundings. Instead the journey is settled as a single activity. The NPC
        moves to the end of the leg and pays that leg's travel time, which keeps it
        out of the simulation for exactly as many region-turns as the walk would
        have taken. Same time spent, same arrival turn, a fraction of the work.

        The distant world doesn't need optimal, occupant-aware routing either: once
        a valid path to the goal is found it stays valid until a tile on it is
        edited, so the route is cached and a real ``find_path`` is paid only on a
        miss (once per trip, not per turn) -- unbounded, because it is that rare.
        Dynamic occupants are ignored out here (invisible off-screen and never
        recorded in ``occupied`` anyway), so only a static tile change can spoil a
        route mid-trip; the walk below stops short if it meets one.
        """
        xy = (pos.x, pos.y)
        path: list[tuple[int, int]] | None = None
        cursor = 0
        cached = self._trip_cache.get(ent)
        if cached is not None:
            c_goal, c_path, c_cursor = cached
            # Re-usable iff it's the same trip and the NPC is still on the route.
            if c_goal == goal and c_cursor + 1 < len(c_path) and c_path[c_cursor] == xy:
                path, cursor = c_path, c_cursor
        if path is None:
            route = self.game_map.find_path(xy, goal)
            if not route:
                self._trip_cache.pop(ent, None)
                return 0
            path, cursor = [xy, *route], 0

        end = cursor
        limit = min(len(path) - 1, cursor + _COMPACTED_TRAVEL_STEPS)
        while end < limit:
            nxt = path[end + 1]
            if not self.game_map.is_walkable(nxt[0], nxt[1]):
                break  # the world changed under the route; re-path next turn
            end += 1
            if not self._far_from_player(nxt):
                # This leg reaches the watched world. Stop on its threshold and
                # hand the rest of the trip back to full per-tile simulation, so
                # an NPC never materializes mid-stride in front of the player.
                break
        if end == cursor:
            self._trip_cache.pop(ent, None)
            return 0

        self._commit_step(ent, pos, path[end], occupied)
        if end + 1 < len(path):
            self._trip_cache[ent] = (goal, path, end)
        else:
            self._trip_cache.pop(ent, None)  # arrived
        return end - cursor

    def _step_toward(
        self,
        ent: int,
        pos: Position,
        goal: tuple[int, int],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Move one step toward ``goal``. Returns True if it moved.

        The common case is a cheap lookup in a cached, goal-rooted flow field
        (shared across every NPC currently heading to that goal) instead of a
        fresh BFS. If the goal is outside the cached field entirely, or every
        viable step toward it is currently blocked, this falls back to a
        one-off occupant-aware pathfind -- exactly what ran here before this
        cache existed.
        """
        # A non-walkable goal can never be reached (find_path can't discover
        # it either -- it's excluded during the BFS itself), so don't spend a
        # full flood or fallback pathfind finding that out the hard way. Comes
        # up for a stale/duplicate blueprint tile that's already the right type
        # (see ``_reachable_ghosts``) and cheaply guards any other goal kind that
        # could end up unreachable the same way.
        if not self.game_map.is_walkable(goal[0], goal[1]):
            return False

        xy = (pos.x, pos.y)
        if self._compactable(xy):
            # Beyond the full-sim box: no per-turn pathfinding, and no per-tile
            # turns. Walk the whole leg at once and bill the traveller for it.
            tiles = self._far_travel(ent, pos, goal, occupied)
            self._charge_activity(ent, tiles - 1)
            return tiles > 0

        step = self._greedy_step_toward(ent, xy, goal, occupied)
        if step is not None:
            self._commit_step(ent, pos, step, occupied)
            return True

        # Only blockers the bounded search can actually reach matter: ``find_path``
        # never expands past a Chebyshev window of _LOCAL_PATH_RADIUS around ``xy``,
        # so gathering the window's tiles (a few hundred dict lookups) is exactly
        # equivalent to -- and at archipelago scale thousands of times cheaper than --
        # iterating every static blocker in the world, which is what this did.
        blocked = self._blockers_within(xy, _LOCAL_PATH_RADIUS, occupied, ignore=ent)
        # Bound the occupant-aware fallback to a local window: it only needs to steer
        # around nearby blockers (the cached flow field already handles long-range
        # routing). This caps a fallback at O(radius^2) instead of flooding the whole
        # island -- the dominant on-screen per-turn cost at archipelago scale.
        path = self.game_map.find_path(xy, goal, blocked_tiles=blocked, max_radius=_LOCAL_PATH_RADIUS)
        if not path:
            return False
        next_x, next_y = path[0]
        if (next_x, next_y) == goal and (next_x, next_y) in occupied:
            # The goal tile itself is occupied (e.g. a tree/prey we path *to*);
            # don't step onto it -- the caller handles the adjacent interaction.
            return False
        if (next_x, next_y) in occupied and occupied[(next_x, next_y)] != ent:
            return False
        self._commit_step(ent, pos, (next_x, next_y), occupied)
        return True

    @staticmethod
    def _blockers_within(
        origin: tuple[int, int],
        radius: int,
        occupied: dict[tuple[int, int], int],
        ignore: int | None = None,
    ) -> set[tuple[int, int]]:
        """The occupied tiles inside a Chebyshev window -- the only ones a
        radius-bounded ``find_path`` can ever expand into."""
        ox, oy = origin
        found: set[tuple[int, int]] = set()
        for y in range(oy - radius, oy + radius + 1):
            for x in range(ox - radius, ox + radius + 1):
                occ = occupied.get((x, y))
                if occ is not None and occ != ignore:
                    found.add((x, y))
        return found

    @staticmethod
    def _commit_step(
        ent: int, pos: Position, next_xy: tuple[int, int], occupied: dict[tuple[int, int], int]
    ) -> None:
        # Just move the creature. ``occupied`` holds only static blockers now, so we
        # deliberately do NOT record movers in it -- creatures are free to share a
        # tile (invisible off-screen, harmless on-screen), and not tracking them is
        # what lets ``occupied`` be a cheap, rarely-rebuilt static cache.
        old_xy = (pos.x, pos.y)
        pos.x, pos.y = next_xy
        # One of the three places anything moves: tell the index, so no system ever
        # has to rescan the world to find out who is standing where (spatial.py).
        spatial.moved(ent, old_xy, next_xy)

    def _region_of_cached(self, xy: tuple[int, int]) -> int | None:
        if self._region_of_cache_revision != self.game_map.revision:
            self._region_of_cache.clear()
            self._region_of_cache_revision = self.game_map.revision
        if xy not in self._region_of_cache:
            self._region_of_cache[xy] = self.game_map.region_of(xy[0], xy[1])
        return self._region_of_cache[xy]

    def _same_region_cached(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        region = self._region_of_cached(a)
        return region is not None and region == self._region_of_cached(b)

    def _reachable(
        self, pos: Position, items: list[tuple[tuple[int, int], int]]
    ) -> list[tuple[tuple[int, int], int]]:
        """Keep only ``(xy, ent)`` targets in the same walkable region as ``pos``,
        so a creature never fixates on food/water across a river it can't cross."""
        here = (pos.x, pos.y)
        return [item for item in items if self._same_region_cached(here, item[0])]

    def _seek_water(
        self,
        ent: int,
        pos: Position,
        needs: Needs,
        occupied: dict[tuple[int, int], int],
        shore: list[tuple[int, int]],
        region_id: RegionId | None = None,
    ) -> bool:
        if any(self.game_map.is_water(nx, ny) for nx, ny in self.game_map.neighbors_8(pos.x, pos.y)):
            needs.thirst = max(0.0, needs.thirst - _DRINK_RESTORE)
            return True
        # Walk down the shore goal map. The field is rooted at every local shore
        # tile at once, so "the nearest bank I can actually walk to" is what
        # stepping downhill means -- no candidate list, no same-region filter, and
        # no fixating on a spot across the water that looked close in a straight
        # line. Shores don't move, so this map is built once and reused for good.
        # Blocked shore tiles (trees cluster by water) are still seeded: the animal
        # walks up beside one and the ``is_water`` check above lets it drink.
        _tile, _ent, acted = self._go_to_nearest(
            ent, pos, "shore", lambda: ((s, -1) for s in shore), occupied,
            region_id, on_source=True,
        )
        return acted

    def _graze(
        self,
        ent: int,
        pos: Position,
        needs: Needs,
        trees: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
        region_id: RegionId | None = None,
    ) -> bool:
        """Browse the nearest tree. Walks down the region's tree goal map rather
        than scanning every tree for the closest one."""
        target_xy, target_ent, acted = self._go_to_nearest(
            ent, pos, "trees", lambda: trees, occupied, region_id, index_kind=Tree
        )
        if target_xy is None:
            return acted
        if not self._take_wood_from_tree(target_ent, target_xy, occupied):
            return False  # the map named a stump; its version key already moved
        needs.hunger = max(0.0, needs.hunger - _GRAZE_RESTORE)
        return True

    @staticmethod
    def _take_wood_from_tree(
        tree_ent: int | None, xy: tuple[int, int], occupied: dict[tuple[int, int], int]
    ) -> bool:
        """Strip one wood off the tree at ``xy``; delete it once it's bare.
        Returns False if it has already gone (the goal map can be a turn stale)."""
        if tree_ent is None or not esper.entity_exists(tree_ent):
            return False
        if not esper.has_component(tree_ent, Tree):
            return False
        tree = esper.component_for_entity(tree_ent, Tree)
        tree.wood -= 1
        if tree.wood <= 0:
            esper.delete_entity(tree_ent, immediate=True)
            occupied.pop(xy, None)
        return True

    def _forage_berries(
        self,
        ent: int,
        pos: Position,
        needs: Needs,
        bushes: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
        clock: WorldClock | None,
        region_id: RegionId | None = None,
    ) -> bool:
        """Head to the nearest ripe berry bush and pick it clean. Bushes block
        their tile, so the forager eats from an adjacent one; the bush regrows a
        fresh crop days later (see ``TreeGrowthProcessor``).

        Takes ``clock`` rather than calling ``world_clock()`` itself: during a
        region catch-up burst this is an as-of clock for the turn being
        replayed, not the true "now" -- using the real clock here would let a
        bush regrow based on time that, for this region, hasn't happened yet.
        """
        target_xy, target_ent, acted = self._go_to_nearest(
            ent, pos, "bushes", lambda: bushes, occupied, region_id, index_kind=BerryBush
        )
        if target_xy is None:
            return acted
        if target_ent is None or not esper.entity_exists(target_ent):
            return False  # the map named a bush that has gone; the key already moved
        if pick_berries(target_ent, clock):
            needs.hunger = max(0.0, needs.hunger - _GRAZE_RESTORE)
        return True

    def _seek_food(
        self,
        ent: int,
        pos: Position,
        needs: Needs,
        prey: list[tuple[tuple[int, int], int]],
        corpses: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """A hungry meat-eater heads to the nearest food: an existing corpse it
        can scavenge, or a live deer it can hunt."""
        # (xy, kind, target_ent). Corpses are already filtered to ones with meat;
        # keep only what's actually reachable from here.
        corpses = self._reachable(pos, corpses)
        prey = self._reachable(pos, prey)
        candidates: list[tuple[tuple[int, int], str, int]] = [
            (xy, "corpse", corpse_ent) for xy, corpse_ent in corpses
        ]
        candidates.extend(
            (xy, "prey", prey_ent)
            for xy, prey_ent in prey
            if prey_ent != ent and esper.entity_exists(prey_ent)
        )
        if not candidates:
            return False

        target_xy, kind, target_ent = min(
            candidates, key=lambda c: _chebyshev((pos.x, pos.y), c[0])
        )
        distance = _chebyshev((pos.x, pos.y), target_xy)

        if kind == "prey":
            # Deer block their tile, so feed from an adjacent tile.
            if distance == 1:
                slay_entity(target_ent)  # kill and butcher; predator feeds
                occupied.pop(target_xy, None)
                needs.hunger = max(0.0, needs.hunger - _FEED_RESTORE)
                return True
            return self._step_toward(ent, pos, target_xy, occupied)

        # Corpses don't block, so "reach" is standing on or next to them.
        if distance <= 1:
            self._eat_from_corpse(target_ent, needs)
            return True
        return self._step_toward(ent, pos, target_xy, occupied)

    def _eat_from_corpse(self, corpse_ent: int, needs: Needs) -> None:
        if not esper.has_component(corpse_ent, Inventory):
            return
        inventory = esper.component_for_entity(corpse_ent, Inventory)
        for index, item in enumerate(inventory.items):
            if is_raw_meat(item) or is_cooked_meat(item):
                inventory.items.pop(index)
                needs.hunger = max(0.0, needs.hunger - _FEED_RESTORE)
                return

    def _eat_from_inventory(self, ent: int, needs: Needs) -> bool:
        """A hungry creature eats a *prepared* item it is carrying (cooked meat,
        bread, ...) before foraging. Raw meat is skipped -- meat must be cooked
        first. Returns True if it ate."""
        if not esper.has_component(ent, Inventory):
            return False
        inventory = esper.component_for_entity(ent, Inventory)
        for index, item in enumerate(inventory.items):
            if is_raw_meat(item):
                continue
            restored = hunger_restored(item)
            if restored is not None:
                inventory.items.pop(index)
                needs.hunger = max(0.0, needs.hunger - restored)
                return True
        return False

    # --- "cook" diet: the full loop a villager runs to feed itself ---------
    # get raw meat (hunt/scavenge into pack) -> get wood (chop a tree) ->
    # carry both to a stove and cook -> eat the cooked meat (via
    # _eat_from_inventory next turn). Each call advances one step of that plan.

    def _forage_meat(
        self,
        ent: int,
        pos: Position,
        inventory: Inventory,
        prey: list[tuple[tuple[int, int], int]],
        corpses: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Put raw meat in the pack: scavenge a corpse, or kill a deer (which
        leaves a corpse to scavenge next). Unlike a predator, does not eat here."""
        corpses = self._reachable(pos, corpses)
        prey = self._reachable(pos, prey)
        candidates: list[tuple[tuple[int, int], str, int]] = [
            (xy, "corpse", corpse_ent) for xy, corpse_ent in corpses
        ]
        candidates.extend(
            (xy, "prey", prey_ent)
            for xy, prey_ent in prey
            if prey_ent != ent and esper.entity_exists(prey_ent)
        )
        if not candidates:
            return False
        target_xy, kind, target_ent = min(candidates, key=lambda c: _chebyshev((pos.x, pos.y), c[0]))
        distance = _chebyshev((pos.x, pos.y), target_xy)

        if kind == "prey":
            if distance == 1:
                slay_entity(target_ent)  # leaves a corpse to butcher next turn
                occupied.pop(target_xy, None)
                return True
            return self._step_toward(ent, pos, target_xy, occupied)

        if distance <= 1:
            self._take_meat_from_corpse(target_ent, inventory)
            return True
        return self._step_toward(ent, pos, target_xy, occupied)

    def _take_meat_from_corpse(self, corpse_ent: int, inventory: Inventory) -> None:
        if not esper.has_component(corpse_ent, Inventory):
            return
        corpse_inventory = esper.component_for_entity(corpse_ent, Inventory)
        for index, item in enumerate(corpse_inventory.items):
            if is_raw_meat(item):
                corpse_inventory.items.pop(index)
                inventory.items.append(item)
                return

    def _gather_wood(
        self,
        ent: int,
        pos: Position,
        inventory: Inventory,
        trees: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
        region_id: RegionId | None = None,
    ) -> bool:
        """Fell one log off the nearest tree, walking down the tree goal map.

        ``_take_wood_from_tree`` returning False means the goal map named a tree
        that has already been felled -- it is only rebuilt when the region's tree
        list is, so a compacted haul that fells several in one turn walks on a map
        that is briefly a stump or two out of date. Taking no wood for that step
        is the honest answer; the next call sees a fresh map.
        """
        target_xy, tree_ent, acted = self._go_to_nearest(
            ent, pos, "trees", lambda: trees, occupied, region_id, index_kind=Tree
        )
        if target_xy is None:
            return acted
        if not self._take_wood_from_tree(tree_ent, target_xy, occupied):
            return False  # the map named a stump; its version key already moved
        inventory.items.append(WOOD)
        return True

    def _cook_at_stove(
        self,
        ent: int,
        pos: Position,
        inventory: Inventory,
        stoves: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
        region_id: RegionId | None = None,
    ) -> bool:
        target_xy, _stove_ent, acted = self._go_to_nearest(
            ent, pos, "stoves", lambda: stoves, occupied, region_id, index_kind=Stove
        )
        if target_xy is None:
            return acted
        raw = next((item for item in inventory.items if is_raw_meat(item)), None)
        if raw is not None and WOOD in inventory.items:
            inventory.items.remove(WOOD)
            inventory.items.remove(raw)
            inventory.items.append(cook_meat(raw))
        return True

    def _feed_cook(
        self,
        ent: int,
        pos: Position,
        prey: list[tuple[tuple[int, int], int]],
        corpses: list[tuple[tuple[int, int], int]],
        trees: list[tuple[tuple[int, int], int]],
        stoves: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        if not esper.has_component(ent, Inventory):
            return False
        inventory = esper.component_for_entity(ent, Inventory)
        needs = esper.component_for_entity(ent, Needs) if esper.has_component(ent, Needs) else None
        has_raw = any(is_raw_meat(item) for item in inventory.items)
        if not has_raw:
            if self._forage_meat(ent, pos, inventory, prey, corpses, occupied):
                return True
            # No reachable game or meat in this area: rather than starve, forage
            # food from the trees around (nuts/berries -- renewable, unlike deer).
            if needs is not None:
                return self._graze(ent, pos, needs, trees, occupied)
            return False
        if WOOD in inventory.items:
            return self._cook_at_stove(ent, pos, inventory, stoves, occupied)
        return self._gather_wood(ent, pos, inventory, trees, occupied)

    def _seek_sleep(
        self, ent: int, pos: Position, needs: Needs, occupied: dict[tuple[int, int], int]
    ) -> bool:
        """A tired NPC heads for bed. It prefers its home tile, walking there a
        step at a time; homeless creatures (and any too exhausted to make it
        home) camp where they stand. Lying down out of sight sleeps the whole
        rest through at once (``_bed_down``)."""
        home = None
        if esper.has_component(ent, Home):
            home = esper.component_for_entity(ent, Home)

        if home is None:
            return self._bed_down(ent, pos, needs, in_camp=True)

        home_xy = (home.x, home.y)
        if (pos.x, pos.y) == home_xy:
            return self._bed_down(ent, pos, needs, in_camp=False)

        if self._step_toward(ent, pos, home_xy, occupied):
            return True

        # Couldn't advance toward home. Camp where we stand if we're spent, or if
        # home is genuinely unreachable (blocked by water/walls) -- better to camp
        # and recover than to idle at a barrier, pinning tiredness and starving.
        if needs.tiredness >= _EXHAUSTED_THRESHOLD or not self._same_region_cached((pos.x, pos.y), home_xy):
            return self._bed_down(ent, pos, needs, in_camp=True)
        return False

    def _bed_down(self, ent: int, pos: Position, needs: Needs, in_camp: bool) -> bool:
        """Put an NPC to sleep, compacting the whole rest into this turn when
        nobody is watching.

        A night's sleep is the purest compactable activity there is: its only
        effects are the three needs, none of which depends on anything that
        happens during it, so the turns it takes can be run as arithmetic
        (``settle_sleep``) rather than as that many region-turns of accrual. The
        sleeper is then tagged ``Settled`` for exactly those turns, so the per-turn
        systems skip it and wake it on the turn it would have woken anyway.

        No energy charge here, unlike travel and hauling: ``Asleep`` already keeps
        a sleeper out of the AI's turn, and putting it in arrears on top would
        leave it standing around after waking.
        """
        go_to_sleep(ent, in_camp=in_camp, game_map=self.game_map)
        if self._compactable((pos.x, pos.y)):
            turns = min(_COMPACTED_SLEEP_TURNS, sleep_turns_needed(needs))
            settle_sleep(needs, turns)
            esper.add_component(ent, Settled(activity="sleep", turns=turns))
        return True

    def _ensure_inventory(self, ent: int) -> Inventory:
        if esper.has_component(ent, Inventory):
            return esper.component_for_entity(ent, Inventory)
        inventory = Inventory(items=[])
        esper.add_component(ent, inventory)
        return inventory

    def _actor_of(self, ent: int) -> Actor:
        """This NPC's action-economy bookkeeping (its per-region-turn energy),
        created on first use so creatures don't need one at spawn."""
        if esper.has_component(ent, Actor):
            return esper.component_for_entity(ent, Actor)
        actor = Actor()
        esper.add_component(ent, actor)
        return actor

    def _compactable(self, xy: tuple[int, int]) -> bool:
        """True when an NPC standing at ``xy`` may settle a whole activity in one
        region-turn -- that is, when nobody can see it happen.

        Today that is exactly "outside the full-simulation box", the same line
        that decides cached-route versus full pathfinding. With no player at all
        (unit tests, whole-world catch-up) nothing is compacted, so those paths
        keep running the exact, unabridged turn-at-a-time simulation.
        """
        return self._far_from_player(xy)

    def _charge_activity(self, ent: int, extra_turns: int) -> None:
        """Bill ``ent`` for the region-turns of a compacted activity beyond the
        one it is already being charged for.

        ``_advance_region`` grants one baseline action's worth of energy per
        region-turn and charges one action's cost per turn taken, so the first
        turn of any activity is already paid for the ordinary way. Overdrawing the
        account by the rest puts the actor in arrears, and the energy loop simply
        doesn't call it again until the following region-turns have paid the
        balance back off -- which is precisely "this NPC is busy for the next N
        turns", with no second clock to keep in sync. Needs keep accruing
        throughout (they belong to the region's turn, not the NPC's), so a long
        activity is genuinely tiring.
        """
        if extra_turns > 0:
            self._actor_of(ent).energy -= extra_turns * action_cost(ent, None)

    def _run_compacted(
        self, ent: int, pos: Position, cap: int, one_turn: Callable[[], bool]
    ) -> bool:
        """Run up to ``cap`` region-turns of a repeating activity in this one, and
        bill the actor for all of them. Returns True if it did anything.

        ``one_turn`` is the activity's ordinary single-turn body: it returns True
        when it did a turn's work and False when there is nothing left to do. That
        is the whole contract -- an activity written the normal way is compacted by
        being handed to this, and never has to know about it.

        The loop stops early if the activity walks its actor into the watched
        world, so nothing is ever fast-forwarded in front of the player: the
        remainder is handed straight back to ordinary per-turn simulation.
        """
        turns = 0
        while turns < cap and one_turn():
            turns += 1
            if not self._compactable((pos.x, pos.y)):
                break
        self._charge_activity(ent, turns - 1)
        return turns > 0

    def _should_build(self, ent: int) -> bool:
        """True when raising a home is this NPC's job right now: a resident that
        owns no bed. Such a villager pitches in on the nearest blueprint -- its
        own staked-out cabin or a neighbour's -- so building is shared labour."""
        return esper.has_component(ent, Resident) and owned_bed_of(ent) is None

    def _reachable_ghosts(self, pos: Position) -> list[tuple[int, tuple[int, int], Blueprint]]:
        """Every blueprint ghost in the same walkable region as ``pos`` -- the
        pieces this worker can actually get to. Anybody's ghosts count, so
        villagers converge on whatever proto-structure is nearest."""
        here = (pos.x, pos.y)
        out: list[tuple[int, tuple[int, int], Blueprint]] = []
        # A ghost this worker can walk to is on this island, so it is in this
        # region or one bordering it -- never on the far side of the sea, which is
        # what scanning every blueprint in the world was paying for.
        region = self.scheduler.region_at(pos.x, pos.y)
        for g_ent, (g_pos, bp) in self._widen_components(region, Blueprint):
            gxy = (g_pos.x, g_pos.y)
            if self.game_map.tile_at(gxy[0], gxy[1]) == bp.tile:
                continue  # already raised elsewhere; ignore this stale ghost
            if here == gxy or self._same_region_cached(here, gxy):
                out.append((g_ent, gxy, bp))
        return out

    def _work_blueprints(
        self,
        ent: int,
        pos: Position,
        trees: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Take one turn of work on the nearest reachable blueprint. While any
        reachable piece still lacks materials the worker **hauls wood** to it
        (lighting it up as its wood arrives); once the reachable pieces are all
        stocked it **raises** them into real tiles, a chunk a turn.

        Returns False when there's nothing this worker can actually do on the
        site right now -- no wood to haul *and* nothing stocked left to raise --
        so a builder who's run the woods dry drops out of build mode and gets on
        with other things instead of freezing beside a site it can't advance.
        Any pieces it already stocked still get raised (below), so the labour
        isn't wasted and the shell keeps rising as far as the materials reached;
        an unstocked piece stays a walkable gap, so raising the stocked ones
        never seals off the rest.

        Out of sight the hauling is compacted: supplying a piece is a round trip
        of pure travel -- out to the woods, a swing of the axe, back to the site,
        once per log -- and every one of those turns is the same unobservable
        errand. ``_run_compacted`` runs the whole trip now and bills the worker for
        the turns it really took, so a haul costs one decision instead of a dozen.
        """
        ghosts = self._reachable_ghosts(pos)
        if not ghosts:
            return False
        unstocked = {gxy: g_ent for g_ent, gxy, bp in ghosts if not bp.stocked}
        # Keep hauling while the woods can still supply the unfinished pieces;
        # _haul_to_ghosts returns False once there's no wood on hand and nothing
        # reachable to fell, at which point we raise whatever is already stocked.
        if unstocked:
            def haul() -> bool:
                return self._haul_to_ghosts(ent, pos, unstocked, trees, occupied)

            hauled = (
                self._run_compacted(ent, pos, _COMPACTED_ACTIVITY_TURNS, haul)
                if self._compactable((pos.x, pos.y))
                else haul()
            )
            if hauled:
                return True
        stocked = {gxy: g_ent for g_ent, gxy, bp in ghosts if bp.stocked}
        if stocked:
            return self._raise_nearby_ghost(ent, pos, stocked, occupied)
        return False

    def _haul_to_ghosts(
        self,
        ent: int,
        pos: Position,
        unstocked: dict[tuple[int, int], int],
        trees: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Carry wood to the proto-structure: gather a batch, walk it to the
        nearest ghost lacking materials, and drop it off -- each wood delivered
        lights one ghost up as "ready". One turn's worth of that per call.

        ``unstocked`` is consumed as pieces are supplied, so calling this again
        (which ``_work_blueprints`` does, to compact a whole round trip into one
        region-turn) picks up where the last delivery left off instead of
        re-delivering to a piece that already has its wood.
        """
        if not unstocked:
            return False  # everything reachable is supplied; nothing left to haul
        inventory = self._ensure_inventory(ent)
        wood = inventory.items.count(WOOD)
        batch = min(_HAUL_BATCH, len(unstocked))
        # Top up the load before making the trip, unless the woods are tapped out.
        if wood < batch and self._reachable(pos, trees):
            return self._gather_wood(ent, pos, inventory, trees, occupied)
        if wood == 0:
            return False  # nothing to carry and no reachable trees -- can't progress

        by_distance = sorted(unstocked, key=lambda xy: _chebyshev((pos.x, pos.y), xy))
        if _chebyshev((pos.x, pos.y), by_distance[0]) <= 1:
            stocked_any = False
            for gxy in by_distance:
                if WOOD not in inventory.items:
                    break
                inventory.items.remove(WOOD)
                _set_blueprint_stocked(unstocked.pop(gxy), True)
                stocked_any = True
            return stocked_any
        # Ghost tiles don't block, so the worker can walk right up to the site.
        return self._step_toward(ent, pos, by_distance[0], occupied)

    def _raise_nearby_ghost(
        self,
        ent: int,
        pos: Position,
        stocked: dict[tuple[int, int], int],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Raise a stocked ghost we're standing next to (a chunk a turn); a
        finished cabin furnishes itself. Otherwise approach the nearest ghost,
        treating still-standing ghosts as blocked so the worker stops beside them
        instead of on them (a raised wall's tile is unwalkable anyway)."""
        here = (pos.x, pos.y)
        for gxy, ghost in stocked.items():
            if _chebyshev(here, gxy) == 1:
                raise_blueprint(self.game_map, ghost)
                return True

        # Standing *on* a stocked ghost with none adjacent (it delivered wood to
        # this piece from here): a wall can't be raised under our own feet, so
        # step off onto any open tile and raise it from beside next turn. Without
        # this the nearest ghost is the one we're on -- distance 0 -- and every
        # "approach" below is a no-op step toward our own tile, pinning the
        # builder there for good.
        if here in stocked:
            for nxy in self.game_map.neighbors_8(here[0], here[1]):
                if (
                    nxy not in occupied
                    and nxy not in stocked
                    and self.game_map.is_walkable(nxy[0], nxy[1])
                ):
                    self._commit_step(ent, pos, nxy, occupied)
                    return True
            return False  # boxed in on our own ghost; nothing to do this turn

        target = min(stocked, key=lambda xy: _chebyshev(here, xy))
        added: list[tuple[int, int]] = []
        for gxy in stocked:
            if gxy not in occupied:
                occupied[gxy] = -1  # sentinel: not a real entity, just blocked
                added.append(gxy)
        moved = self._step_toward(ent, pos, target, occupied)
        for xy in added:
            if occupied.get(xy) == -1:
                del occupied[xy]
        return moved

    def _chase_player(
        self,
        ent: int,
        pos: Position,
        player_xy: tuple[int, int],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        vision_radius = 8
        if esper.has_component(ent, Vision):
            vision_radius = esper.component_for_entity(ent, Vision).radius
        if _chebyshev((pos.x, pos.y), player_xy) > vision_radius:
            return False
        if not self.game_map.has_line_of_sight((pos.x, pos.y), player_xy):
            return False
        # Line of sight crosses water/gaps a walker can't cross (it only blocks
        # on walls) -- so a hostile that can *see* the player across a river it
        # can't reach would otherwise retry a full, expensive "no path exists"
        # pathfind every single turn it keeps watching. Reachability is a
        # cheap, cached lookup (GameMap.region_of); check it before ever
        # calling into pathfinding.
        if not self._same_region_cached((pos.x, pos.y), player_xy):
            return False
        if _chebyshev((pos.x, pos.y), player_xy) == 1:
            return False  # adjacent: hold (player-facing combat is player-driven)
        return self._step_toward(ent, pos, player_xy, occupied)

    def _pick_social_partner(
        self,
        ent: int,
        pos: Position,
        sentients: list[tuple[int, Position]],
    ) -> tuple[int, Position] | None:
        """Who ``ent`` would most like to approach: the nearest awake, reachable
        being it can see, weighted so higher current friendship wins over a
        slightly closer stranger. Returns ``(entity, position)`` or ``None``."""
        rel = (
            esper.component_for_entity(ent, Relationships)
            if esper.has_component(ent, Relationships)
            else None
        )
        best: tuple[int, Position] | None = None
        best_score: float | None = None
        for other, other_pos in sentients:
            if other == ent or esper.has_component(other, Asleep):
                continue
            dist = _chebyshev((pos.x, pos.y), (other_pos.x, other_pos.y))
            if dist > _SOCIAL_SIGHT:
                continue
            if not self._same_region_cached((pos.x, pos.y), (other_pos.x, other_pos.y)):
                continue
            # Prefer friends: friendship pulls the score up, distance pushes it
            # down. A stranger (0 friendship) is still chosen when nobody
            # closer or friendlier is around.
            score = friendship(rel, other) - dist * _SOCIAL_DISTANCE_PENALTY
            if best_score is None or score > best_score:
                best_score = score
                best = (other, other_pos)
        return best

    def _socialize(
        self,
        ent: int,
        pos: Position,
        occupied: dict[tuple[int, int], int],
        sentients: list[tuple[int, Position]],
        turn: int,
        clock: WorldClock | None,
    ) -> bool:
        """A being seeks out someone to talk to, preferring beings it already
        likes. Interacts when adjacent (adjusting friendship + popping bubbles),
        otherwise steps toward the chosen partner. Honours a per-being cooldown so
        villagers don't chatter every turn.

        ``turn``/``clock`` are an as-of turn number and clock during a region
        catch-up burst, not necessarily the true "now" -- see ``_forage_berries``.
        """
        personality = esper.component_for_entity(ent, Personality)
        if turn - personality.last_social_turn < _SOCIAL_COOLDOWN:
            return False

        partner = self._pick_social_partner(ent, pos, sentients)
        if partner is None:
            return False
        partner_ent, partner_pos = partner

        if _chebyshev((pos.x, pos.y), (partner_pos.x, partner_pos.y)) == 1:
            interact(ent, partner_ent, turn)
            # Growing close turns friends into lovers and lovers into spouses: a
            # very close single pair weds; a close pair alone together may be
            # intimate. Both are no-ops unless the pair is adult, opposite-sex,
            # and friendly enough, so this fires only for genuine couples.
            try_marry(ent, partner_ent, clock)
            try_mate(ent, partner_ent, turn, clock, sentients)
            return True
        return self._step_toward(ent, pos, (partner_pos.x, partner_pos.y), occupied)

    # --- what a region can see -------------------------------------------

    def _index(self):
        return spatial.ensure(self.game_map)

    @staticmethod
    def _has_meat(ent: int) -> bool:
        if not esper.has_component(ent, Inventory):
            return False
        items = esper.component_for_entity(ent, Inventory).items
        return any(is_raw_meat(item) or is_cooked_meat(item) for item in items)

    @staticmethod
    def _has_berries(ent: int) -> bool:
        return esper.component_for_entity(ent, BerryBush).has_berries

    def _widen(
        self,
        index,
        region_id: RegionId,
        kind: type,
        predicate: Callable[[int], bool] | None = None,
        live_position: bool = False,
    ) -> list:
        """Every entity of one kind in ``region_id``, plus those within
        ``_REGION_BORDER_MARGIN`` of it in the eight neighbouring regions -- so a
        creature standing near a seam still sees a resource just across it.

        Nine index buckets are read; the world is not. Entities come out in id
        order so a region's turn plays out the same way every run, which the
        seeded-determinism guarantee depends on.
        """
        game_map = self.game_map
        margin = _REGION_BORDER_MARGIN
        cx, cy = region_id
        items: list = []
        for ny in range(cy - 1, cy + 2):
            for nx in range(cx - 1, cx + 2):
                own_region = (nx, ny) == region_id
                for ent in sorted(index.of_kind((nx, ny), kind)):
                    if not esper.entity_exists(ent) or not esper.has_component(ent, Position):
                        continue
                    pos = esper.component_for_entity(ent, Position)
                    if not own_region and not in_region_with_margin(
                        game_map, region_id, pos.x, pos.y, margin
                    ):
                        continue
                    if predicate is not None and not predicate(ent):
                        continue
                    items.append((ent, pos) if live_position else ((pos.x, pos.y), ent))
        return items

    def _widen_components(self, region_id: RegionId, kind: type):
        """``(entity, (Position, kind))`` for a region and its neighbours -- the
        regional form of ``esper.get_components(Position, kind)``."""
        index = self._index()
        cx, cy = region_id
        for ny in range(cy - 1, cy + 2):
            for nx in range(cx - 1, cx + 2):
                for ent in sorted(index.of_kind((nx, ny), kind)):
                    if not esper.entity_exists(ent) or not esper.has_component(ent, Position):
                        continue
                    yield ent, (
                        esper.component_for_entity(ent, Position),
                        esper.component_for_entity(ent, kind),
                    )

    def _static_region_items(self, region_id: RegionId) -> tuple[list, list, list]:
        """``(trees, stoves, shore)`` around a region. None of these move, so this
        is rebuilt only when the index says one of the nine regions involved gained
        or lost something -- not on a timer, and never from a world scan."""
        index = self._index()
        key = index.neighborhood_version(region_id)
        cached = self._static_region_cache.get(region_id)
        if cached is not None and cached[0] == key:
            return cached[1]
        items = (
            self._widen(index, region_id, Tree),
            self._widen(index, region_id, Stove),
            list(self._region_bucket(self._shore_by_region, region_id, lambda s: s)),
        )
        self._static_region_cache[region_id] = (key, items)
        return items

    def _dynamic_region_items(self, region_id: RegionId) -> tuple[list, list, list, list]:
        """``(prey, corpses, bushes, sentients)`` around a region.

        These drift without anything entering or leaving -- a deer wanders, a bush
        ripens -- so they are refreshed every ``_WORLD_SNAPSHOT_REFRESH_CALLS``
        advances of this region, exactly the staleness the old world-wide snapshot
        traded for, except the refresh now costs one region rather than one world.
        A membership change (a birth, a death, someone crossing the seam) refreshes
        it immediately regardless.
        """
        index = self._index()
        key = index.neighborhood_version(region_id)
        cached = self._dynamic_region_cache.get(region_id)
        if cached is not None and cached[0] == key and cached[1] > 0:
            self._dynamic_region_cache[region_id] = (key, cached[1] - 1, cached[2])
            return cached[2]
        items = (
            self._widen(index, region_id, Deer),
            self._widen(index, region_id, Corpse, predicate=self._has_meat),
            self._widen(index, region_id, BerryBush, predicate=self._has_berries),
            self._widen(index, region_id, Personality, live_position=True),
        )
        self._dynamic_region_cache[region_id] = (key, _WORLD_SNAPSHOT_REFRESH_CALLS, items)
        return items

    def _region_bucket(
        self, buckets: dict[RegionId, list], region_id: RegionId, xy_of: Callable[[object], tuple[int, int]]
    ) -> list:
        """The margin-widened items of a pre-bucketed *static tile* map (the shore
        list). Entity kinds go through ``_widen`` and the index instead."""
        game_map = self.game_map
        margin = _REGION_BORDER_MARGIN
        cx, cy = region_id
        items = list(buckets.get(region_id, ()))
        for nx in range(cx - 1, cx + 2):
            for ny in range(cy - 1, cy + 2):
                if (nx, ny) == region_id:
                    continue
                for item in buckets.get((nx, ny), ()):
                    x, y = xy_of(item)
                    if in_region_with_margin(game_map, region_id, x, y, margin):
                        items.append(item)
        return items

    def _advance_region(self, region_id: RegionId) -> None:
        """Run one turn of NPC AI for the NPCs standing in ``region_id``.

        Everything this turn reads comes from that region and its immediate
        neighbours (see ``_static_region_items``/``_dynamic_region_items``), so the
        cost of simulating an island is what that island holds -- not what the
        world holds. ``player_xy`` is fetched fresh every call regardless: it's a
        single cheap lookup, and hostile-chase distance is short-range enough that
        staleness there would actually be noticeable.
        """
        # The blocker map is the index's own live tile map: exact at all times, and
        # never rebuilt. It used to be a copy of every blocking entity in the world,
        # refreshed on a timer.
        occupied = self._index().blockers

        # This region's own logical turn: during a catch-up burst this replays
        # turn N, N+1, N+2 ... in order, each with its own as-of clock, so
        # turn-stamped state (berry regrowth, courtship cooldowns, ages) reads
        # correctly relative to *this* region's history, not the true "now".
        # The scheduler counts in whole region-turns; the world clock and every
        # turn-stamped value are in TU, so convert (one region-turn == one
        # baseline action == BASE_ACTION_COST TU) before stamping.
        logical_turn = (self.scheduler.region_turn[region_id] + 1) * BASE_ACTION_COST
        real_clock = world_clock()
        clock = replace(real_clock, turn=logical_turn) if real_clock is not None else None

        player_xy = self._find_player_position()
        # Per-NPC full-vs-cached-path movement keys off distance to this tile.
        self._player_xy = player_xy

        trees, stoves, shore = self._static_region_items(region_id)
        prey, corpses, bushes, sentients = self._dynamic_region_items(region_id)

        # NPCs acting this turn are exactly this region's own bucket (no
        # margin) -- a border-straddling NPC must belong to exactly one
        # region's turn, never both, or it would act twice. Read fresh from the
        # index every turn: who stands here is the one thing that must never be
        # stale, and it is a single bucket lookup.
        index = self._index()
        acting = [
            (ent, esper.component_for_entity(ent, Position))
            for ent in sorted(index.of_kind(region_id, NPC))
            if esper.entity_exists(ent) and esper.has_component(ent, Position)
        ]
        for ent, pos in acting:
            if not esper.entity_exists(ent):
                continue
            # Sleepers skip their turn; NeedsProcessor recovers and wakes them.
            if esper.has_component(ent, Asleep):
                continue

            # Anti-oscillation guard: forbid an immediate one-tile reversal back
            # onto the tile this NPC started its *previous* turn on. A productive
            # route never needs to reverse in one step (the flow field is
            # monotonic toward its goal); a two-tile ping-pong only arises when a
            # blocked higher drive (say, seeking water walled off by trees the
            # pathfinder sees straight through) hands off to a lower one pulling
            # the other way, turn after turn. Blocking that single tile for just
            # this NPC's turn -- via the same occupant sentinel the step code
            # already respects -- collapses the jitter into standing still, which
            # is what an NPC with nowhere better to go should do anyway.
            start_xy = (pos.x, pos.y)
            guard = self._prev_turn_pos.get(ent)
            if guard is not None and (guard == start_xy or guard in occupied):
                guard = None
            if guard is not None:
                occupied[guard] = _OSC_GUARD
            try:
                # Action economy: this region-turn is one baseline action's worth
                # of time, so grant BASE_ACTION_COST energy and let the NPC act as
                # many times as its speed allows -- a quicker creature (higher
                # dexterity, lower action cost) banks the surplus and acts again.
                # A baseline NPC (cost == BASE_ACTION_COST) acts exactly once,
                # preserving the old one-turn-per-region-turn cadence.
                actor = self._actor_of(ent)
                actor.energy += BASE_ACTION_COST
                cost = action_cost(ent, None)
                acted = 0
                while actor.energy >= cost and acted < _MAX_ACTIONS_PER_REGION_TURN:
                    self._take_turn(
                        ent, pos, occupied, player_xy, diet_buckets=(
                            trees, prey, corpses, stoves, bushes, sentients
                        ), shore=shore, logical_turn=logical_turn, clock=clock,
                        region_id=region_id,
                    )
                    actor.energy -= cost
                    acted += 1
                    if not esper.entity_exists(ent):
                        break
            finally:
                if guard is not None and occupied.get(guard) == _OSC_GUARD:
                    del occupied[guard]
                self._prev_turn_pos[ent] = start_xy

    def _drive_factor(self, ent: int, drive_id: str) -> float:
        if not esper.has_component(ent, DriveProfile):
            return 1.0
        return float(esper.component_for_entity(ent, DriveProfile).drives.get(drive_id, 1.0))

    def _has_inventory_food(self, ent: int) -> bool:
        if not esper.has_component(ent, Inventory):
            return False
        return any(
            not is_raw_meat(item) and hunger_restored(item) is not None
            for item in esper.component_for_entity(ent, Inventory).items
        )

    def _trigger_passes(self, ent: int, trigger: tuple, ctx: dict) -> bool:
        name, *args = trigger
        needs = ctx.get("needs")
        diet_kind = ctx.get("diet_kind")
        if name == "tiredness_at_least":
            return needs is not None and needs.tiredness >= float(args[0])
        if name == "thirst_at_least":
            return needs is not None and needs.thirst >= float(args[0])
        if name == "hunger_at_least":
            return needs is not None and needs.hunger >= float(args[0])
        if name == "thirst_beats_hunger":
            return needs is not None and needs.thirst >= needs.hunger
        if name == "has_diet":
            return diet_kind == args[0]
        if name == "diet_in":
            return diet_kind in args
        if name == "is_starving":
            return needs is not None and needs.hunger >= needs.max_value * 0.9
        if name == "has_inventory_food":
            return self._has_inventory_food(ent)
        if name == "region_has_trees":
            return bool(ctx["trees"])
        if name == "region_has_bushes":
            return bool(ctx["bushes"])
        if name == "region_has_meat":
            return bool(ctx["prey"] or ctx["corpses"])
        if name == "region_has_shore":
            return bool(ctx["shore"])
        if name == "should_build":
            return self._should_build(ent)
        if name == "can_socialize":
            return esper.has_component(ent, Personality) and esper.has_component(ent, Friendly)
        if name == "has_player":
            return ctx["player_xy"] is not None
        if name == "is_enemy":
            return esper.has_component(ent, Enemy)
        raise KeyError(f"unknown AI trigger {name!r}")

    def _score_drive(self, ent: int, drive: DriveDef, ctx: dict) -> tuple[float | None, dict]:
        factor = self._drive_factor(ent, drive.id)
        allow_results = [
            {"trigger": trigger, "passed": self._trigger_passes(ent, trigger, ctx)}
            for trigger in drive.allow
        ]
        if factor <= 0.0 or not all(item["passed"] for item in allow_results):
            return None, {
                "drive": drive.id, "factor": factor, "allow": allow_results, "weight": None,
            }
        weight = drive.weight.base * factor
        for modifier in drive.weight.modifiers:
            if self._trigger_passes(ent, modifier.trigger, ctx):
                weight *= modifier.factor
        return weight, {
            "drive": drive.id, "factor": factor, "allow": allow_results, "weight": weight,
        }

    def _rank_drives(self, ent: int, ctx: dict) -> list[DriveDef]:
        scored = []
        debug = []
        for drive in all_drives():
            weight, record = self._score_drive(ent, drive, ctx)
            debug.append(record)
            if weight is not None and weight > 0.0:
                scored.append((weight, drive.id, drive))
        scored.sort(key=lambda item: (-item[0], item[1]))
        self.last_decisions[ent] = debug
        return [drive for _weight, _id, drive in scored]

    def _execute_drive(self, drive: DriveDef, ent: int, pos: Position, occupied: dict, ctx: dict) -> bool:
        needs = ctx.get("needs")
        if drive.act == "seek_sleep" and needs is not None:
            return self._seek_sleep(ent, pos, needs, occupied)
        if drive.act == "seek_water" and needs is not None:
            return self._seek_water(ent, pos, needs, occupied, ctx["shore"], ctx["region_id"])
        if drive.act == "eat_from_inventory" and needs is not None:
            return self._eat_from_inventory(ent, needs)
        if drive.act == "forage_berries" and needs is not None:
            return self._forage_berries(
                ent, pos, needs, ctx["bushes"], occupied, ctx["clock"], ctx["region_id"]
            )
        if drive.act == "graze" and needs is not None:
            return self._graze(ent, pos, needs, ctx["trees"], occupied, ctx["region_id"])
        if drive.act == "seek_food" and needs is not None:
            return self._seek_food(ent, pos, needs, ctx["prey"], ctx["corpses"], occupied)
        if drive.act == "feed_cook":
            return self._feed_cook(ent, pos, ctx["prey"], ctx["corpses"], ctx["trees"], ctx["stoves"], occupied)
        if drive.act == "work_blueprints":
            return self._work_blueprints(ent, pos, ctx["trees"], occupied)
        if drive.act == "socialize":
            return self._socialize(ent, pos, occupied, ctx["sentients"], ctx["logical_turn"], ctx["clock"])
        if drive.act == "chase_player":
            self._chase_player(ent, pos, ctx["player_xy"], occupied)
            return True
        raise KeyError(f"unknown AI action {drive.act!r}")

    def _take_turn(
        self,
        ent: int,
        pos: Position,
        occupied: dict[tuple[int, int], int],
        player_xy: tuple[int, int] | None,
        diet_buckets: tuple,
        shore: list[tuple[int, int]],
        logical_turn: int,
        clock: "WorldClock | None",
        region_id: RegionId | None = None,
    ) -> None:
        """Select and execute one declarative drive for an NPC.

        The selector is deterministic highest-weight scoring. The old priority
        ladder is represented by data in ``content.drives`` with widely spaced
        base weights, so changing species/personality behavior no longer needs a
        new branch here.
        """
        trees, prey, corpses, stoves, bushes, sentients = diet_buckets
        needs = esper.component_for_entity(ent, Needs) if esper.has_component(ent, Needs) else None
        diet_kind = esper.component_for_entity(ent, Diet).kind if esper.has_component(ent, Diet) else None
        ctx = {
            "needs": needs, "diet_kind": diet_kind, "trees": trees, "prey": prey,
            "corpses": corpses, "stoves": stoves, "bushes": bushes,
            "sentients": sentients, "shore": shore, "player_xy": player_xy,
            "logical_turn": logical_turn, "clock": clock, "region_id": region_id,
        }
        for drive in self._rank_drives(ent, ctx):
            if self._execute_drive(drive, ent, pos, occupied, ctx):
                return

    def process(self, action: str | None = None) -> None:
        if action not in _TURN_ACTIONS:
            return

        # A recycled entity id must never inherit a previous world's cached route
        # or widened buckets (both hold entity ids / live Position refs).
        world = esper.current_world
        if world != self._trip_cache_world:
            self._trip_cache.clear()
            self._static_region_cache.clear()
            self._dynamic_region_cache.clear()
            self._trip_cache_world = world

        player_xy = self._find_player_position()
        player_region = (
            self.scheduler.region_at(player_xy[0], player_xy[1]) if player_xy is not None else None
        )

        if player_region is None:
            # No player (unit tests construct this processor directly): bring every
            # region up to date once, matching an unpartitioned pass. With no player
            # ``_far_from_player`` is always False, so every NPC is fully pathfound
            # -- exact, unpartitioned behaviour.
            for region_id in all_region_ids(self.game_map):
                self.scheduler.advance_region(region_id)
            return

        target_turn = self.scheduler.next_turn_for(player_region, _current_region_turn())

        # The player's own region gets first priority. In tests/default construction
        # this fully catches up; live play may cap the replay count so entering a
        # stale region does not block a walking frame for the whole debt.
        self.scheduler.catch_up_region(
            player_region, target_turn, max_advances=self._max_entry_catchup_advances
        )

        # Background: nudge the *nearest* other lagging regions along, closest
        # to the player first (never the stalest). Bounded, so it can never
        # stall input. Zero by default (see _NPC_BACKGROUND_BUDGET): an active
        # keypress simulates only the player's own region, and the main loop's
        # idle-time pump plus sleep/region-entry catch-up resolve the rest off
        # the keypress path.
        if _NPC_BACKGROUND_BUDGET > 0.0:
            self.scheduler.pump_background(
                _NPC_BACKGROUND_BUDGET, player_region, target_turn, self._wall_clock
            )


class FishAiProcessor(esper.Processor):
    """Swims the fish each turn. A fish is the aquatic mirror of a grazing deer:
    when hungry it eats seaweed it is next to, otherwise it drifts toward the
    nearest seaweed it can see, and failing that mills about at random.

    Fish move on water tiles only, so -- unlike the land creatures the
    ``NpcAiProcessor`` drives with ``find_path`` -- they step greedily between
    adjacent water cells and never strand themselves ashore. They are not tagged
    ``NPC``, so the land AI ignores them entirely.

    Uses a ``RegionScheduler`` (see ``regions.py``) so the player's own 120x60
    region is always fully live, distant regions pay down simulation debt in
    the background (nearest-to-player first), and either a region-entry or a
    sleep can force a region -- or the whole world -- fully up to date.
    """

    def __init__(
        self,
        game_map: GameMap,
        rng: Callable[[], float] | None = None,
        clock: Callable[[], float] | None = None,
        max_entry_catchup_advances: int | None = None,
    ):
        self.game_map = game_map
        self._max_entry_catchup_advances = max_entry_catchup_advances
        self._rng = rng if rng is not None else world_rng().stream("ai").random
        self._wall_clock = clock if clock is not None else time.monotonic
        self.scheduler = RegionScheduler(game_map, _current_region_turn())
        self.scheduler.register("fish", self._advance_region)
        # Which fish is on which tile, world-wide -- the one thing here that isn't
        # per-region, because a fish must not swim onto a neighbour resting in a
        # region nobody is simulating. Rebuilt only when the shoal gains or loses a
        # member; swimming updates it a tile at a time (see ``_move``).
        self._fish_tiles: dict[tuple[int, int], int] | None = None
        self._fish_population = -1
        # Per-region seaweed, cached against the index's version of the region.
        self._seaweed_by_region: dict[RegionId, tuple[tuple[int, ...], list]] = {}

    def _fish_occupied(self) -> dict[tuple[int, int], int]:
        """Tile -> fish, world-wide, so a fish never swims onto a resting neighbour
        even one in a region nobody is simulating. Rebuilt only when the shoal's
        population changes; fish that merely swam are moved tile to tile by
        ``_move``, which is the only thing that relocates one."""
        population = spatial.component_population(Fish)
        if self._fish_tiles is None or self._fish_population != population:
            self._fish_tiles = {}
            for ent, (pos, _f) in esper.get_components(Position, Fish):
                self._fish_tiles[(pos.x, pos.y)] = ent
            self._fish_population = population
        return self._fish_tiles

    def _region_fish(self, region_id: RegionId) -> list:
        """The fish swimming in one region, straight from the index."""
        index = spatial.ensure(self.game_map)
        return [
            (ent, esper.component_for_entity(ent, Position))
            for ent in sorted(index.of_kind(region_id, Fish))
            if esper.entity_exists(ent) and esper.has_component(ent, Position)
        ]

    def _region_seaweed(self, region_id: RegionId) -> list:
        """The seaweed growing in one region. Seaweed never moves, so this is
        cached until the region's membership changes (a frond eaten bare, a new
        one seeded)."""
        index = spatial.ensure(self.game_map)
        version = index.neighborhood_version(region_id)
        cached = self._seaweed_by_region.get(region_id)
        if cached is not None and cached[0] == version:
            return cached[1]
        fronds = [
            ((pos.x, pos.y), ent)
            for ent, pos in (
                (e, esper.component_for_entity(e, Position))
                for e in sorted(index.of_kind(region_id, Seaweed))
                if esper.entity_exists(e) and esper.has_component(e, Position)
            )
        ]
        self._seaweed_by_region[region_id] = (version, fronds)
        return fronds

    def _advance_region(self, region_id: RegionId) -> None:
        """One turn of fish for one region, reading only that region's fish and
        seaweed out of the index rather than re-bucketing every fish in the sea."""
        self._simulate_area(
            self._region_fish(region_id), self._region_seaweed(region_id), self._fish_occupied()
        )

    def process(self, action: str | None = None) -> None:
        # Fish move only on real turns -- never on idle/menu ticks -- so the
        # animation loop's rapid ``process(None)`` calls can never block.
        if action not in _TURN_ACTIONS:
            return

        player_region = None
        for _ent, (pos, _player) in esper.get_components(Position, Player):
            player_region = self.scheduler.region_at(pos.x, pos.y)
            break

        if player_region is None:
            # No player (unit tests construct this processor directly): bring
            # every region up to date once, matching an unpartitioned pass.
            for region_id in all_region_ids(self.game_map):
                self.scheduler.advance_region(region_id)
            return

        target_turn = self.scheduler.next_turn_for(player_region, _current_region_turn())

        # The player's own region gets first priority. In tests/default construction
        # this fully catches up; live play may cap the replay count so entering a
        # stale region does not block a walking frame for the whole debt.
        self.scheduler.catch_up_region(
            player_region, target_turn, max_advances=self._max_entry_catchup_advances
        )

        # Background: nudge the *nearest* other lagging regions along, closest
        # to the player first. Zero by default (see _FISH_BACKGROUND_BUDGET): an
        # active turn swims only the player's own region; distant shoals are
        # resolved by the idle-time pump, region entry (above), or sleep.
        if _FISH_BACKGROUND_BUDGET > 0.0:
            self.scheduler.pump_background(
                _FISH_BACKGROUND_BUDGET, player_region, target_turn, self._wall_clock
            )

    def _simulate_area(
        self,
        fish: list[tuple[int, Position]],
        seaweed: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> None:
        """Swim one area's fish for a single step against that area's seaweed."""
        seaweed_at = {xy: sw_ent for xy, sw_ent in seaweed}
        for ent, pos in fish:
            if not esper.entity_exists(ent):
                continue
            needs = (
                esper.component_for_entity(ent, Needs)
                if esper.has_component(ent, Needs)
                else None
            )
            hungry = needs is not None and needs.hunger >= _FORAGE_THRESHOLD

            if hungry:
                bite = next(
                    (
                        (nx, ny)
                        for nx, ny in self.game_map.neighbors_8(pos.x, pos.y)
                        if (nx, ny) in seaweed_at
                    ),
                    None,
                )
                if bite is not None:
                    self._graze_seaweed(seaweed_at[bite], bite, needs, seaweed_at, occupied)
                    continue
                target = self._nearest_seaweed(pos, seaweed)
                if target is not None and self._step_toward(ent, pos, target, occupied):
                    continue

            self._wander(ent, pos, occupied)

    def _water_neighbors(
        self, x: int, y: int, occupied: dict[tuple[int, int], int]
    ) -> list[tuple[int, int]]:
        return [
            (nx, ny)
            for nx, ny in self.game_map.neighbors_8(x, y)
            if self.game_map.is_water(nx, ny) and (nx, ny) not in occupied
        ]

    def _nearest_seaweed(
        self, pos: Position, seaweed: list[tuple[tuple[int, int], int]]
    ) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        best_dist: int | None = None
        for xy, _ent in seaweed:
            dist = _chebyshev((pos.x, pos.y), xy)
            if dist > _FISH_SIGHT:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = xy
        return best

    def _step_toward(
        self, ent: int, pos: Position, goal: tuple[int, int], occupied: dict[tuple[int, int], int]
    ) -> bool:
        """Greedily step to the open water neighbour nearest the goal."""
        options = self._water_neighbors(pos.x, pos.y, occupied)
        if not options:
            return False
        nx, ny = min(options, key=lambda xy: _chebyshev(xy, goal))
        self._move(ent, pos, nx, ny, occupied)
        return True

    def _wander(self, ent: int, pos: Position, occupied: dict[tuple[int, int], int]) -> None:
        if self._rng() >= _FISH_WANDER_CHANCE:
            return
        options = self._water_neighbors(pos.x, pos.y, occupied)
        if not options:
            return
        nx, ny = options[min(len(options) - 1, int(self._rng() * len(options)))]
        self._move(ent, pos, nx, ny, occupied)

    def _graze_seaweed(
        self,
        sw_ent: int,
        xy: tuple[int, int],
        needs: Needs,
        seaweed_at: dict[tuple[int, int], int],
        occupied: dict[tuple[int, int], int],
    ) -> None:
        needs.hunger = max(0.0, needs.hunger - _FISH_GRAZE_RESTORE)
        if esper.entity_exists(sw_ent) and esper.has_component(sw_ent, Seaweed):
            frond = esper.component_for_entity(sw_ent, Seaweed)
            frond.food -= 1
            if frond.food <= 0:
                esper.delete_entity(sw_ent, immediate=True)  # eaten bare
                seaweed_at.pop(xy, None)

    def _move(
        self, ent: int, pos: Position, nx: int, ny: int, occupied: dict[tuple[int, int], int]
    ) -> None:
        old_xy = (pos.x, pos.y)
        occupied.pop(old_xy, None)
        pos.x, pos.y = nx, ny
        occupied[(nx, ny)] = ent
        spatial.moved(ent, old_xy, (nx, ny))


