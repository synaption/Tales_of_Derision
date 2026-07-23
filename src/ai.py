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
from collections.abc import Callable
from dataclasses import replace

import esper

from components import (
    Actor, Asleep, BerryBush, BlocksMovement, Blueprint, Corpse, Deer, Diet, Enemy,
    Fish, Friendly, Home, Inventory, NPC, Needs, Personality, Player, Position,
    Relationships, Resident, Seaweed, Stove, Tree, Vision, WorldClock,
)
from game_map import GameMap
from regions import RegionId, RegionScheduler, all_region_ids, in_region_with_margin
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
    pick_berries, raise_blueprint, slay_entity, try_marry, try_mate, world_clock,
)

# An empty occupant map, passed to the greedy stepper for off-screen movement so it
# ignores dynamic collisions entirely (see ``_step_toward``). Never mutated.
_NO_OCCUPANTS: dict[tuple[int, int], int] = {}

# How far the occupant-aware fallback pathfind may search around a creature. Big
# enough to round any local cluster of blockers, small enough that the search is
# cheap; long-range travel rides the cached flow field, not this fallback.
_LOCAL_PATH_RADIUS = 12


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

    def __init__(self, game_map: GameMap, wall_clock: Callable[[], float] | None = None):
        self.game_map = game_map
        self._shore_tiles: list[tuple[int, int]] = self._compute_shore_tiles()
        self._wall_clock = wall_clock if wall_clock is not None else time.monotonic
        # The player's region + its 8 neighbours: the on-screen band that gets full,
        # dynamic pathfinding. Every other region is "off-screen" and moves with a
        # cheap static approximation (see ``_step_toward`` / ``_static_movement``).
        # ``None`` means "treat everything as on-screen" -- the safe default for
        # unit tests and whole-world catch-up (sleep), which want exact behaviour.
        self._near_regions: set[RegionId] | None = None
        # Set per region-advance: True when advancing an off-screen region.
        self._static_movement = False
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
        # A world-wide, region-bucketed snapshot (occupied tiles + every goal
        # kind + every NPC), rebuilt from scratch only every
        # _WORLD_SNAPSHOT_REFRESH_CALLS advances instead of on every single
        # one -- a catch-up burst (region-entry, sleep) advances many regions
        # many turns each, and re-scanning literally every entity in the world
        # for each of those individual turns is the dominant cost at any real
        # population (bounded staleness traded for that no longer happening).
        self._world_snapshot: dict[str, dict] | None = None
        self._world_snapshot_calls_left = 0
        # The near-static half of the snapshot (trees, stoves) is rebuilt far less
        # often -- see _STATIC_SNAPSHOT_REFRESH_CALLS -- since re-scanning the ~68k
        # trees every dynamic refresh was most of the snapshot cost.
        self._static_snapshot: dict[str, dict] | None = None
        self._static_snapshot_calls_left = 0
        # The static+dynamic halves merged into one dict, re-merged only when a half
        # is rebuilt (the halves hold live refs -- occupied is mutated in place, NPC
        # positions are live -- so the cached merge stays correct between rebuilds).
        self._merged_snapshot: dict[str, dict] | None = None
        # goal xy -> (region edit revision near goal, world edit revision, calls
        # left before a routine refresh, the flow field itself). Shared across
        # every NPC heading to the same goal, not per-entity -- a distance field
        # rooted at a (largely static) goal stays valid for any traveller
        # approaching it from anywhere, so many NPCs reuse the one flood.
        self._field_cache: dict[
            tuple[int, int], tuple[int, int, int, dict[tuple[int, int], int]]
        ] = {}
        # ent -> the tile it stood on at the start of its previous turn. Used to
        # forbid an immediate one-tile reversal (see ``_advance_region``), which
        # is the only way an NPC ends up flip-flopping between two tiles forever.
        self._prev_turn_pos: dict[int, tuple[int, int]] = {}

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

        ``distance_field`` is a pure function of the goal and the (static-per-
        turn) tile grid, so a cached field is valid exactly as long as no tile
        it could cover has changed. Two revisions gate reuse:

        * **World edit revision** (``GameMap.revision``, bumped on *any* tile
          edit anywhere). If it hasn't moved since the field was built, nothing
          in the world changed, so the field is provably identical -- reuse it
          indefinitely, no rebuild. This is the dominant case: during a
          catch-up burst the player stands still and NPCs edit no tiles, so one
          flood toward the player serves every chaser across the whole burst
          instead of being rebuilt dozens of times.
        * **Region edit revision** near the goal. When the world *has* been
          edited somewhere, the field spans more than the goal's own region
          cell, so a per-cell revision can't prove the edit missed it. We then
          fall back to a bounded-staleness reuse (``calls_left`` refreshes)
          exactly as before -- no regression for the living, self-editing world.
        """
        region_revision = self.game_map.region_edit_revision(goal[0], goal[1])
        world_revision = self.game_map.revision
        cached = self._field_cache.get(goal)
        if cached is not None:
            cached_region_rev, cached_world_rev, calls_left, field = cached
            if cached_region_rev == region_revision:
                if cached_world_rev == world_revision:
                    # No tile anywhere edited since the build -> byte-identical.
                    return field
                if calls_left > 0:
                    # Edited somewhere, but not in the goal's cell we can see;
                    # reuse under the staleness bound (keep the old world rev so
                    # the countdown still eventually rebuilds to pick it up).
                    self._field_cache[goal] = (
                        cached_region_rev, cached_world_rev, calls_left - 1, field
                    )
                    return field
        field = self.game_map.distance_field(goal)
        self._field_cache[goal] = (
            region_revision, world_revision, _PATH_FIELD_REFRESH_CALLS, field
        )
        return field

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
        if self._static_movement:
            # Off-screen: step along the cached *static* flow field only, ignoring
            # dynamic occupants and never doing a dynamic fallback pathfind. This
            # approximates how long the trip takes (one tile per turn along the
            # static shortest path) without the expensive occupant-aware search;
            # invisible creatures overlapping for a turn costs nothing on-screen.
            step = self._greedy_step_toward(ent, xy, goal, _NO_OCCUPANTS)
            if step is not None:
                self._commit_step(ent, pos, step, occupied)
                return True
            return False

        step = self._greedy_step_toward(ent, xy, goal, occupied)
        if step is not None:
            self._commit_step(ent, pos, step, occupied)
            return True

        blocked = {xy2 for xy2, occ_ent in occupied.items() if occ_ent != ent}
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
    def _commit_step(
        ent: int, pos: Position, next_xy: tuple[int, int], occupied: dict[tuple[int, int], int]
    ) -> None:
        # Just move the creature. ``occupied`` holds only static blockers now, so we
        # deliberately do NOT record movers in it -- creatures are free to share a
        # tile (invisible off-screen, harmless on-screen), and not tracking them is
        # what lets ``occupied`` be a cheap, rarely-rebuilt static cache.
        pos.x, pos.y = next_xy

    def _reachable(
        self, pos: Position, items: list[tuple[tuple[int, int], int]]
    ) -> list[tuple[tuple[int, int], int]]:
        """Keep only ``(xy, ent)`` targets in the same walkable region as ``pos``,
        so a creature never fixates on food/water across a river it can't cross."""
        return [item for item in items if self.game_map.same_region((pos.x, pos.y), item[0])]

    def _seek_water(
        self,
        ent: int,
        pos: Position,
        needs: Needs,
        occupied: dict[tuple[int, int], int],
        shore: list[tuple[int, int]],
    ) -> bool:
        if any(self.game_map.is_water(nx, ny) for nx, ny in self.game_map.neighbors_8(pos.x, pos.y)):
            needs.thirst = max(0.0, needs.thirst - _DRINK_RESTORE)
            return True
        # Drink from a shore tile we can actually stand on -- skip shores blocked
        # by a tree/creature (trees cluster by water), or the animal would fixate
        # on an unreachable spot and thrash on the bank without ever drinking.
        # ``shore`` is already this region's local shore bucket, so the same_region
        # filter runs over a handful of tiles, never every shore in the world.
        reachable = [
            s
            for s in shore
            if s not in occupied and self.game_map.same_region((pos.x, pos.y), s)
        ]
        target = self._nearest((pos.x, pos.y), reachable)
        if target is None:
            return False
        return self._step_toward(ent, pos, target, occupied)

    def _graze(
        self,
        ent: int,
        pos: Position,
        needs: Needs,
        trees: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        trees = self._reachable(pos, trees)
        if not trees:
            return False
        target_xy, target_ent = min(trees, key=lambda item: _chebyshev((pos.x, pos.y), item[0]))
        if _chebyshev((pos.x, pos.y), target_xy) == 1:
            needs.hunger = max(0.0, needs.hunger - _GRAZE_RESTORE)
            if esper.entity_exists(target_ent) and esper.has_component(target_ent, Tree):
                tree = esper.component_for_entity(target_ent, Tree)
                tree.wood -= 1
                if tree.wood <= 0:
                    esper.delete_entity(target_ent, immediate=True)  # grazed bare
                    occupied.pop(target_xy, None)
            return True
        return self._step_toward(ent, pos, target_xy, occupied)

    def _forage_berries(
        self,
        ent: int,
        pos: Position,
        needs: Needs,
        bushes: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
        clock: WorldClock | None,
    ) -> bool:
        """Head to the nearest ripe berry bush and pick it clean. Bushes block
        their tile, so the forager eats from an adjacent one; the bush regrows a
        fresh crop days later (see ``TreeGrowthProcessor``).

        Takes ``clock`` rather than calling ``world_clock()`` itself: during a
        region catch-up burst this is an as-of clock for the turn being
        replayed, not the true "now" -- using the real clock here would let a
        bush regrow based on time that, for this region, hasn't happened yet.
        """
        bushes = self._reachable(pos, bushes)
        if not bushes:
            return False
        target_xy, target_ent = min(bushes, key=lambda item: _chebyshev((pos.x, pos.y), item[0]))
        if _chebyshev((pos.x, pos.y), target_xy) == 1:
            if pick_berries(target_ent, clock):
                needs.hunger = max(0.0, needs.hunger - _GRAZE_RESTORE)
            return True
        return self._step_toward(ent, pos, target_xy, occupied)

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
    ) -> bool:
        trees = self._reachable(pos, trees)
        if not trees:
            return False
        target_xy, tree_ent = min(trees, key=lambda t: _chebyshev((pos.x, pos.y), t[0]))
        if _chebyshev((pos.x, pos.y), target_xy) == 1:
            inventory.items.append(WOOD)
            if esper.entity_exists(tree_ent) and esper.has_component(tree_ent, Tree):
                tree = esper.component_for_entity(tree_ent, Tree)
                tree.wood -= 1
                if tree.wood <= 0:
                    esper.delete_entity(tree_ent, immediate=True)
                    occupied.pop(target_xy, None)
            return True
        return self._step_toward(ent, pos, target_xy, occupied)

    def _cook_at_stove(
        self,
        ent: int,
        pos: Position,
        inventory: Inventory,
        stoves: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        stoves = self._reachable(pos, stoves)
        if not stoves:
            return False
        target_xy, _stove_ent = min(stoves, key=lambda s: _chebyshev((pos.x, pos.y), s[0]))
        if _chebyshev((pos.x, pos.y), target_xy) == 1:
            raw = next((item for item in inventory.items if is_raw_meat(item)), None)
            if raw is not None and WOOD in inventory.items:
                inventory.items.remove(WOOD)
                inventory.items.remove(raw)
                inventory.items.append(cook_meat(raw))
            return True
        return self._step_toward(ent, pos, target_xy, occupied)

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
        home) camp where they stand."""
        home = None
        if esper.has_component(ent, Home):
            home = esper.component_for_entity(ent, Home)

        if home is None:
            go_to_sleep(ent, in_camp=True)
            return True

        home_xy = (home.x, home.y)
        if (pos.x, pos.y) == home_xy:
            go_to_sleep(ent, in_camp=False)
            return True

        if self._step_toward(ent, pos, home_xy, occupied):
            return True

        # Couldn't advance toward home. Camp where we stand if we're spent, or if
        # home is genuinely unreachable (blocked by water/walls) -- better to camp
        # and recover than to idle at a barrier, pinning tiredness and starving.
        if needs.tiredness >= _EXHAUSTED_THRESHOLD or not self.game_map.same_region((pos.x, pos.y), home_xy):
            go_to_sleep(ent, in_camp=True)
            return True
        return False

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
        for g_ent, (g_pos, bp) in esper.get_components(Position, Blueprint):
            gxy = (g_pos.x, g_pos.y)
            if self.game_map.tile_at(gxy[0], gxy[1]) == bp.tile:
                continue  # already raised elsewhere; ignore this stale ghost
            if here == gxy or self.game_map.same_region(here, gxy):
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
        """
        ghosts = self._reachable_ghosts(pos)
        if not ghosts:
            return False
        unstocked = {gxy: g_ent for g_ent, gxy, bp in ghosts if not bp.stocked}
        # Keep hauling while the woods can still supply the unfinished pieces;
        # _haul_to_ghosts returns False once there's no wood on hand and nothing
        # reachable to fell, at which point we raise whatever is already stocked.
        if unstocked and self._haul_to_ghosts(ent, pos, unstocked, trees, occupied):
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
        lights one ghost up as "ready"."""
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
                _set_blueprint_stocked(unstocked[gxy], True)
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
        if not self.game_map.same_region((pos.x, pos.y), player_xy):
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
            if not self.game_map.same_region((pos.x, pos.y), (other_pos.x, other_pos.y)):
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

    def _build_static_snapshot(self) -> dict[str, dict]:
        """The near-static buckets -- trees, stoves, and the ``occupied`` blocker
        map -- bucketed by region. These barely change (trees only on a day
        boundary, stoves/furniture never), so they are rescanned only every
        _STATIC_SNAPSHOT_REFRESH_CALLS: the ~68k tree scan was the bulk of the old
        whole-snapshot cost.

        ``occupied`` deliberately holds only the STATIC blockers -- every
        ``BlocksMovement`` entity that is not a creature (no ``NPC``/``Player``).
        Moving creatures are ignored entirely: they never stack-collide with each
        other, which costs nothing off-screen and is invisible on-screen, and it
        means we never rescan all ~85k blockers every dynamic refresh just to track
        movers. What this still guarantees is the thing that matters -- nobody walks
        onto a tree/well/oven/furniture tile, and (with the wall/water/door tiles the
        map itself enforces) nobody reaches somewhere they shouldn't."""
        region_at = self.scheduler.region_at
        # Static blockers: every BlocksMovement entity that is not a creature. One
        # scan of all ~85k, but only every _STATIC_SNAPSHOT_REFRESH_CALLS.
        occupied: dict[tuple[int, int], int] = {
            (pos.x, pos.y): ent
            for ent, (pos, _b) in esper.get_components(Position, BlocksMovement)
            if not esper.has_component(ent, NPC) and not esper.has_component(ent, Player)
        }
        trees: dict[RegionId, list] = {}
        for ent, (pos, _t) in esper.get_components(Position, Tree):
            trees.setdefault(region_at(pos.x, pos.y), []).append(((pos.x, pos.y), ent))
        stoves: dict[RegionId, list] = {}
        for ent, (pos, _s) in esper.get_components(Position, Stove):
            stoves.setdefault(region_at(pos.x, pos.y), []).append(((pos.x, pos.y), ent))
        return {"trees": trees, "stoves": stoves, "occupied": occupied}

    def _build_dynamic_snapshot(self) -> dict[str, dict]:
        """The fast-changing buckets -- blocked tiles, moving prey/NPCs, appearing
        corpses, toggling berry bushes -- bucketed by each entity's *exact* region
        (margin is applied at lookup in ``_region_bucket``). Rescanned every
        _WORLD_SNAPSHOT_REFRESH_CALLS; the static half comes from
        ``_build_static_snapshot`` (which also owns ``occupied`` now)."""
        region_at = self.scheduler.region_at
        prey: dict[RegionId, list] = {}
        for ent, (pos, _d) in esper.get_components(Position, Deer):
            prey.setdefault(region_at(pos.x, pos.y), []).append(((pos.x, pos.y), ent))
        corpses: dict[RegionId, list] = {}
        for ent, (pos, _c, inv) in esper.get_components(Position, Corpse, Inventory):
            if any(is_raw_meat(item) or is_cooked_meat(item) for item in inv.items):
                corpses.setdefault(region_at(pos.x, pos.y), []).append(((pos.x, pos.y), ent))
        bushes: dict[RegionId, list] = {}
        for ent, (pos, bush) in esper.get_components(Position, BerryBush):
            if bush.has_berries:
                bushes.setdefault(region_at(pos.x, pos.y), []).append(((pos.x, pos.y), ent))
        # Sentients/NPCs keep the live Position object, not a frozen (x, y) --
        # they move every turn, so a snapshot reused for many calls must keep
        # reading their *current* position, not the one at snapshot time.
        sentients: dict[RegionId, list] = {}
        for ent, (pos, _p) in esper.get_components(Position, Personality):
            sentients.setdefault(region_at(pos.x, pos.y), []).append((ent, pos))
        npcs: dict[RegionId, list] = {}
        for ent, (pos, _npc) in esper.get_components(Position, NPC):
            npcs.setdefault(region_at(pos.x, pos.y), []).append((ent, pos))
        return {
            "prey": prey,
            "corpses": corpses,
            "bushes": bushes,
            "sentients": sentients,
            "npcs": npcs,
        }

    def _world_snapshot_for_this_call(self) -> dict[str, dict]:
        rebuilt = False
        if self._static_snapshot is None or self._static_snapshot_calls_left <= 0:
            self._static_snapshot = self._build_static_snapshot()
            self._static_snapshot_calls_left = _STATIC_SNAPSHOT_REFRESH_CALLS
            rebuilt = True
        else:
            self._static_snapshot_calls_left -= 1
        if self._world_snapshot is None or self._world_snapshot_calls_left <= 0:
            self._world_snapshot = self._build_dynamic_snapshot()
            self._world_snapshot_calls_left = _WORLD_SNAPSHOT_REFRESH_CALLS
            rebuilt = True
        else:
            self._world_snapshot_calls_left -= 1
        if rebuilt or self._merged_snapshot is None:
            self._merged_snapshot = {**self._static_snapshot, **self._world_snapshot}
        return self._merged_snapshot

    def _region_bucket(
        self, buckets: dict[RegionId, list], region_id: RegionId, xy_of: Callable[[object], tuple[int, int]]
    ) -> list:
        """This region's own bucketed items, widened by ``_REGION_BORDER_MARGIN``
        into the (up to 8) neighbouring regions' buckets -- so a creature near a
        seam still sees a resource one tile into the next region. Only the
        handful of neighbouring buckets are scanned, never the whole world."""
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
        """Run one turn of NPC AI for the NPCs standing in ``region_id``,
        using a periodically-refreshed world snapshot rather than rescanning
        every entity in the world on every single call (see
        ``_world_snapshot_for_this_call``). ``player_xy`` is fetched fresh
        every call regardless -- it's a single cheap lookup, not a full scan,
        and hostile-chase distance is short-range enough that staleness here
        would actually be noticeable.
        """
        # Off-screen regions move their creatures with a cheap static approximation
        # (no dynamic pathfinding / collision) -- see ``_step_toward``.
        self._static_movement = (
            self._near_regions is not None and region_id not in self._near_regions
        )
        snapshot = self._world_snapshot_for_this_call()
        occupied = snapshot["occupied"]

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

        xy_of_pair = lambda item: item[0]  # noqa: E731 -- ((x, y), ent) items
        xy_of_pos = lambda item: (item[1].x, item[1].y)  # noqa: E731 -- (ent, Position) items

        trees = self._region_bucket(snapshot["trees"], region_id, xy_of_pair)
        prey = self._region_bucket(snapshot["prey"], region_id, xy_of_pair)
        corpses = self._region_bucket(snapshot["corpses"], region_id, xy_of_pair)
        stoves = self._region_bucket(snapshot["stoves"], region_id, xy_of_pair)
        bushes = self._region_bucket(snapshot["bushes"], region_id, xy_of_pair)
        sentients = self._region_bucket(snapshot["sentients"], region_id, xy_of_pos)
        shore = self._region_bucket(self._shore_by_region, region_id, lambda s: s)

        # NPCs acting this turn are exactly this region's own bucket (no
        # margin) -- a border-straddling NPC must belong to exactly one
        # region's turn, never both, or it would act twice.
        for ent, pos in list(snapshot["npcs"].get(region_id, ())):
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
                    )
                    actor.energy -= cost
                    acted += 1
                    if not esper.entity_exists(ent):
                        break
            finally:
                if guard is not None and occupied.get(guard) == _OSC_GUARD:
                    del occupied[guard]
                self._prev_turn_pos[ent] = start_xy

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
    ) -> None:
        """One NPC's single-turn behaviour: the priority ladder of survival
        drives, then building, then leisure, then hostile pursuit. Split out of
        ``_advance_region`` so the per-turn anti-oscillation guard there can wrap
        it cleanly."""
        trees, prey, corpses, stoves, bushes, sentients = diet_buckets
        acted = False

        if esper.has_component(ent, Needs):
            needs = esper.component_for_entity(ent, Needs)
            diet_kind = None
            if esper.has_component(ent, Diet):
                diet_kind = esper.component_for_entity(ent, Diet).kind

            # Sleep is the strongest drive: a spent creature beds down before
            # it forages, preferring its home over camping.
            if needs.tiredness >= _SLEEP_THRESHOLD:
                acted = self._seek_sleep(ent, pos, needs, occupied)
            # Thirst wins ties -- a parched animal drinks before it eats.
            elif needs.thirst >= _FORAGE_THRESHOLD and needs.thirst >= needs.hunger:
                acted = self._seek_water(ent, pos, needs, occupied, shore)
            elif needs.hunger >= _FORAGE_THRESHOLD:
                # Eat prepared food already carried; otherwise forage by diet.
                if self._eat_from_inventory(ent, needs):
                    acted = True
                elif diet_kind == "herbivore":
                    # Ripe berries first (quick food); else graze a tree.
                    acted = self._forage_berries(ent, pos, needs, bushes, occupied, clock)
                    if not acted:
                        acted = self._graze(ent, pos, needs, trees, occupied)
                elif diet_kind == "carnivore":
                    # Predator: eats raw meat on the spot.
                    acted = self._seek_food(ent, pos, needs, prey, corpses, occupied)
                elif diet_kind == "cook":
                    # Villager: pick ripe berries when handy, else cook meat.
                    acted = self._forage_berries(ent, pos, needs, bushes, occupied, clock)
                    if not acted:
                        acted = self._feed_cook(ent, pos, prey, corpses, trees, stoves, occupied)

        # Building a house is the lowest-priority survival drive: a homeless
        # resident helps raise the nearest blueprint only once fed, watered,
        # and rested. Labour is shared -- several villagers can work one site.
        if not acted and self._should_build(ent):
            acted = self._work_blueprints(ent, pos, trees, occupied)

        # Leisure: a fed, rested, non-hostile being with a personality seeks
        # out company -- preferring beings it already likes. Lowest priority
        # of all, so survival and building always come first.
        if (
            not acted
            and esper.has_component(ent, Personality)
            and esper.has_component(ent, Friendly)
        ):
            acted = self._socialize(ent, pos, occupied, sentients, logical_turn, clock)

        if acted:
            return

        if player_xy is not None and esper.has_component(ent, Enemy):
            self._chase_player(ent, pos, player_xy, occupied)

    def process(self, action: str | None = None) -> None:
        if action not in _TURN_ACTIONS:
            return

        player_xy = self._find_player_position()
        player_region = (
            self.scheduler.region_at(player_xy[0], player_xy[1]) if player_xy is not None else None
        )

        if player_region is None:
            # No player (unit tests construct this processor directly): bring
            # every region up to date once, matching an unpartitioned pass.
            self._near_regions = None  # everything on-screen: exact behaviour
            for region_id in all_region_ids(self.game_map):
                self.scheduler.advance_region(region_id)
            return

        # The on-screen band: the player's region and its 8 neighbours get full,
        # dynamic simulation; everything else moves by the static approximation.
        px, py = player_region
        self._near_regions = {
            (px + dx, py + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        }

        target_turn = self.scheduler.next_turn_for(player_region, _current_region_turn())

        # The player's own region is always fully live -- this also covers
        # "just entered a new region": catch_up_region replays every turn the
        # region missed, in order, right here.
        self.scheduler.catch_up_region(player_region, target_turn)

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
    ):
        self.game_map = game_map
        self._rng = rng if rng is not None else world_rng().stream("ai").random
        self._wall_clock = clock if clock is not None else time.monotonic
        self.scheduler = RegionScheduler(game_map, _current_region_turn())
        self.scheduler.register("fish", self._advance_region)
        # Rebuilt from scratch only every _WORLD_SNAPSHOT_REFRESH_CALLS
        # advances rather than on every single one -- see the identical cache
        # on NpcAiProcessor for why a catch-up burst makes this matter.
        self._world_snapshot: dict[str, dict] | None = None
        self._world_snapshot_calls_left = 0
        # Seaweed is static (it never moves), so -- like the land AI's trees -- it is
        # rescanned only every _STATIC_SNAPSHOT_REFRESH_CALLS instead of every dynamic
        # refresh; only the moving fish are rebucketed often.
        self._seaweed_snapshot: dict[RegionId, list] | None = None
        self._seaweed_snapshot_calls_left = 0
        self._merged_snapshot: dict[str, dict] | None = None

    def _build_fish_snapshot(self) -> dict[str, dict]:
        region_at = self.scheduler.region_at
        fish_by_region: dict[RegionId, list] = {}
        occupied: dict[tuple[int, int], int] = {}
        for ent, (pos, _f) in esper.get_components(Position, Fish):
            fish_by_region.setdefault(region_at(pos.x, pos.y), []).append((ent, pos))
            occupied[(pos.x, pos.y)] = ent
        return {"fish": fish_by_region, "occupied": occupied}

    def _build_seaweed_snapshot(self) -> dict[RegionId, list]:
        region_at = self.scheduler.region_at
        seaweed_by_region: dict[RegionId, list] = {}
        for ent, (pos, _s) in esper.get_components(Position, Seaweed):
            seaweed_by_region.setdefault(region_at(pos.x, pos.y), []).append(((pos.x, pos.y), ent))
        return seaweed_by_region

    def _world_snapshot_for_this_call(self) -> dict[str, dict]:
        rebuilt = False
        if self._seaweed_snapshot is None or self._seaweed_snapshot_calls_left <= 0:
            self._seaweed_snapshot = self._build_seaweed_snapshot()
            self._seaweed_snapshot_calls_left = _STATIC_SNAPSHOT_REFRESH_CALLS
            rebuilt = True
        else:
            self._seaweed_snapshot_calls_left -= 1
        if self._world_snapshot is None or self._world_snapshot_calls_left <= 0:
            self._world_snapshot = self._build_fish_snapshot()
            self._world_snapshot_calls_left = _WORLD_SNAPSHOT_REFRESH_CALLS
            rebuilt = True
        else:
            self._world_snapshot_calls_left -= 1
        if rebuilt or self._merged_snapshot is None:
            self._merged_snapshot = {**self._world_snapshot, "seaweed": self._seaweed_snapshot}
        return self._merged_snapshot

    def _advance_region(self, region_id: RegionId) -> None:
        snapshot = self._world_snapshot_for_this_call()
        fish = list(snapshot["fish"].get(region_id, ()))
        seaweed = list(snapshot["seaweed"].get(region_id, ()))
        # Shared across the whole world so a fish never swims onto a resting
        # neighbour, even one in a region we're not simulating right now.
        occupied = snapshot["occupied"]
        self._simulate_area(fish, seaweed, occupied)

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

        # The player's own region is always fully live -- this also covers
        # "just entered a new region": catch_up_region replays every turn the
        # region missed, in order, right here.
        self.scheduler.catch_up_region(player_region, target_turn)

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
        occupied.pop((pos.x, pos.y), None)
        pos.x, pos.y = nx, ny
        occupied[(nx, ny)] = ent


