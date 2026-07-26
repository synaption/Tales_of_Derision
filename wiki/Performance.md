# Performance

Performance is a first-class goal: the target is thousands of tiles and thousands of
NPCs. There are two independent budgets — the **per-frame** render budget and the
**per-turn** simulation budget — and each has invariants that are easy to break by
accident. This page is the checklist; the [World Simulation](World-Simulation.md) and
[Game Map](Game-Map.md) pages cover the mechanisms.

## Render budget (per frame)

**Invariant: never transform the whole world map surface per frame.** The pygame
renderer caches the whole world as one off-screen `_map_surface`. On the 360×180 world
at 16px cells that surface is ~5760×2880 ≈ **16 Mpx**; a single full-surface transform
(e.g. grayscale + `BLEND_RGB_MULT`) costs ~230ms. Worse, the living-world sim edits map
tiles constantly (tree growth, construction), and each edit invalidates the derived
surface — so a per-frame whole-surface transform would rebuild most frames and make the
game unplayable.

Do instead:
- Composite the map by **blitting the visible region** of the cached surface
  (`blit_map_region`), offset by the camera; repaint only **dirty cells**
  (`redraw_map_cells` ← `GameMap.consume_dirty_tiles()`), never the whole surface.
- For per-tile visual effects (fog-of-war, tinting), modify only the **on-screen
  region**. Remembered/"fog" tiles fade via `apply_memory_fade` (grayscale-blend +
  multiply dim) rendered **once** into a viewport-sized cache
  (`capture_memory_layer` / `blit_memory_cache`), rebuilt only on move / map edit and
  just blitted on idle frames.
- Snapshot the scene once behind menus (`capture_backdrop`) instead of re-rendering
  every keypress.
- Clip the FOV/map blit to the viewport so it can't bleed into the sidebar when
  zoomed/scrolled.
- Small **per-sprite** transforms cached by `id(tile)` (`_desaturate_tile`) are fine —
  it's only the world-sized surface that must never be transformed per frame.

Keep the caches invalidated correctly whenever you touch tiles, scale, or camera
(`invalidate_map_surface` / `invalidate_backdrop` run on `set_mode`; the FOV/map caches
assume walls don't change mid-session).

## Simulation budget (per turn)

The lever is the synchronous per-turn `esper.process()` spike. **Profile the real
workload, not intuition** — a cProfile of a cross-region walk with a headless renderer
(so the map-surface build doesn't mask sim cost) is the trustworthy signal.
Whole-turn A/B timings are confounded because A*'s tie-break changes NPC routes and
the two runs diverge; trust divergence-proof metrics (per-call time, node counts).

Run the harness: `python3 scripts/profile_turns.py [turns] [--flood] [--render]`. It
builds the real 360×180 world and walks the player across section seams under
cProfile, sim-only by default. `--flood` spawns a cave rat on every walkable tile
(thousands of NPCs) for the scaling worst case.

Representative results (200 turns, ~2300 entities): **~19 ms/turn**, with cost spread
across `distance_field` (flow-field builds), `_compute_regions` / `find_enclosed_rooms`
(recomputed when the living world edits a tile and bumps `revision`), and the AI's
per-region bucket rebuild (`_region_bucket`). The flood case (~6500 NPCs) runs at
**~118 ms/turn**; after the flow-field-reuse fix below, the AI chase/step loop
(`_step_toward` greedy lookups + rare fallback pathfinds) dominates, not field builds.

**AI flow-field reuse (done).** Chasers heading to the same goal already share one
`distance_field` via `NpcAiProcessor._field_cache` (keyed by goal, not per-entity).
The remaining waste was that the cache's staleness countdown rebuilt that shared field
every few calls during catch-up bursts even when *nothing had changed* — a profiled
flood did **357** field builds over 40 turns, **261** of them redundant rebuilds of the
chase field for a standing-still player. Because `distance_field` is a pure function of
goal + tiles, the cache now also stores `GameMap.revision` (the global tile-edit
counter): if it hasn't moved, no tile anywhere changed, so the field is byte-identical
and is reused indefinitely (no rebuild). Only when the world *has* been edited does it
fall back to the per-region-revision + countdown hedge. Result: **357 → 13** builds
(one per distinct goal), flood **~166 → ~118 ms/turn**, behaviour byte-identical (the
skipped rebuilds would have produced the same field), no regression to the living world.

**Measured, not guessed:** an entity spatial index (per-tile buckets to replace the
O(n) `get_components(Position)` scans) was long deferred, because at one island the
profile said pathfinding dominated and the scans did not. That held until the world got
big: measured across 1 / 4 / 9 islands, the scans were what grew (`MovementProcessor`
rebuilt a dict of *every* blocking entity in the world on each keypress: 0.08 → 1.63
ms/turn), while the per-island work stayed flat. The index now exists — see
[`src/spatial.py`](../src/spatial.py) and "Regional entity index" below.

The BFS hot loops (`distance_field`, `_compute_regions`) inline their neighbour/bounds/
walkability tests the same way `find_path` does — that alone cut the normal case from
~32 to ~20 ms/turn with identical behaviour.

What has actually dominated, and the fixes already in place:
- **Build-site search** (`choose_build_site` / `_site_is_clear`) once cost 53–75% of
  per-turn time — a homeless villager with nowhere to build re-ran a ~14k-check
  outward scan every turn (99% failing). Fixed with a **ring-only scan** (test just the
  fresh perimeter at each radius) + a **back-off memo** (`HousingProcessor._no_site`,
  keyed by region cell + `region_edit_revision`): 20ms → 0.02ms/turn.
- **Pathfinding** was secondary. `find_path` is A* (Chebyshev heuristic + goal-biased
  tie-break, inlined hot loop): optimal like BFS but ~30× faster / ~50× fewer nodes.
  Cost now concentrates only in cross-region catch-up. Reuse **flow fields**
  (`distance_field`) across travellers/turns instead of re-pathing.

Remaining per-turn cost is spread thin (no single dominant): `distance_field` builds,
per-island connectivity relabels (`_compute_island_regions`, `find_enclosed_rooms`)
after a villager lays a wall, and region-geometry helpers
(`region_bounds`/`region_grid_size`/`in_region_with_margin`). All of these are bounded
by one island, so they no longer grow with the archipelago.

## Goal maps

Flow-field reuse solved "many creatures, one goal". It did nothing for the commoner
case — many creatures, *many* goals — because a per-goal field is a fresh island-wide
BFS for every tile anybody picks. Measured, that was **one ~6 ms island flood per
player turn, 60–70% of the whole turn, at every world size** (1.78 floods/turn at one
island; 1.17 at a hundred). Creatures mint new goals constantly: 113 cached fields for
15 deer, each choosing whichever tree looked nearest as it moved.

A **goal map** inverts it: `GameMap.distance_field_from` seeds *every* tree (or shore
tile, or ripe bush) at distance 0 in one flood, so stepping downhill from anywhere
walks to the nearest one. `NpcAiProcessor._goal_map` caches one per
`(kind, region, island)` and `_go_to_nearest` drives every resource-seeking drive off
it — replacing both the flood *and* the `_reachable` scan (an O(sources) same-region
filter each drive ran per creature per turn) with eight dict lookups.

Three things make it pay, and each was needed:

- **Per-kind invalidation.** `SpatialIndex.kind_neighborhood_version` narrows the
  index's version counter to one component type. Keyed on the region-wide counter
  instead, the map of where the trees are died every time a deer crossed a seam —
  a 40% miss rate.
- **Island scoping.** A region's widened source list can straddle several islands, and
  seeding them all floods every one: measured at 27 382 tiles per flood (four islands)
  where one island is 6 838. Seeds are filtered to the walking component the creature
  is standing on — the same rule `_reachable` enforced, now paid once per flood rather
  than per creature per turn.
- **Watched creatures only.** In the player's region a flood is shared by everyone
  standing there, every turn. In the lagging world a region gets one turn at a time
  and its trees churn as its creatures eat them, so the same flood is bought over and
  over for a handful of uses: 75 tree floods for ~205 uses (2 ms a use), and
  whole-world catch-up rose 47%. Out there `_go_to_nearest_unwatched` keeps the cheap
  scan-and-walk. (A demand-driven staleness policy was tried instead and measured
  *worse* — 228 floods against 109 — because a periodic refresh fires more often than
  real changes do.)

Everything improved, and per-turn cost is now flat in island count *and* small:

| islands | 1 | 9 | 25 | 100 |
|---|---|---|---|---|
| per player turn, before | 15.6 ms | 8.0 | 9.0 | 14.5 |
| per player turn, after | **3.6 ms** | **3.2** | **0.63** | **2.3** |
| enter a stale region, before | — | 153 ms | 156 | 165 |
| enter a stale region, after | — | **85 ms** | **88** | **98** |
| `catch_up_all`, before | — | 1.26 s | 3.28 | 16.03 |
| `catch_up_all`, after | — | **1.10 s** | **2.85** | **13.76** |

The floods that remain are single-source ones from `_step_toward` for the goals no
goal map covers yet — home, blueprints, prey, conversation partners — at 0.07–0.25
per turn. Those are the next candidates.

### Strict active/inactive partitioning

`esper.process(action)` simulates the region the player stands in and nothing else.
Everywhere else advances only where the design allows it to: `simulate_idle`'s
background pump, `_catch_up_entered_region_cooperatively` on region entry, and sleep.
`src/tests/test_active_region_partition.py` holds the turn path to that rule.

Needs and status effects are **registered steps on the region scheduler**
(`NeedsProcessor.register_region_step` / `EffectsProcessor.register_region_step`, wired
in `game._register_processors`), so they run wherever a region's turn actually runs: the
live region each turn, and everywhere else in the pump, on entry, and during sleep. That
closed a real gap — an inactive region's needs used to not tick *at all*, so villagers
caught up after a long absence replayed with the appetites they had when you walked
away and simply stood about. A region now accrues one baseline turn of hunger per
region-turn it owed. The player is the exception: their needs still tick against the
world clock in `process`, because a slow action has to make *them* proportionally
hungrier — an NPC has no slow actions, only region-turns.

## Scaling levers (in profile-justified order)

- **AI flow-field reuse** — *done* (see above): chasers share one goal-rooted
  `distance_field`, and a global-`revision` cache gate skips rebuilding it while no
  tile has changed, collapsing catch-up-burst rebuilds (357 → 13 builds in the flood).
- **Incremental region relabelling** — `_compute_regions` / `find_enclosed_rooms`
  recompute the whole map whenever a single tile edit bumps `revision`; relabel only
  the affected component instead.
- **Scheduler heap** — `next_actor()`'s O(n) scan → a heap (the interface already
  anticipates it; see [Action Economy](Action-Economy.md)).
- **Regional entity index** — *done* (`src/spatial.py`): every positioned entity
  bucketed by simulation region and by kind, built once at worldgen and maintained
  incrementally (three movement hooks; creation and deletion are noticed from esper's
  own id counter, so no call site has to remember). Systems ask for a region's
  entities instead of the world's. Measured over 300 turns after a 150-turn settle,
  at 9 islands: `MovementProcessor` 1.63 → 0.02 ms/turn, `NeedsProcessor` 0.15 → 0.03,
  `HousingProcessor` 0.88 → 0.47, whole turn 10.8 → 8.2 ms. What remains in a turn is
  bounded per-island tile work (`distance_field` floods), not entity scans.
- **Everything in memory at startup** — sounds, tiles, state (per `next.md`).

See [Roadmap](Roadmap.md) for sequencing.
