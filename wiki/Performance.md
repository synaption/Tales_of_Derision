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
O(n) `get_components(Position)` scans) was on the roadmap, but the profile shows those
scans are *not* the bottleneck — pathfinding is. So it isn't warranted yet.

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

Remaining per-turn cost is spread thin (no single dominant): per-region bucket rebuilds
(`_region_bucket`), `distance_field` builds, and region-geometry helpers
(`region_bounds`/`region_grid_size`/`in_region_with_margin`, called ~1.25M times over
300 turns). These are the next levers.

## Scaling levers (in profile-justified order)

- **AI flow-field reuse** — *done* (see above): chasers share one goal-rooted
  `distance_field`, and a global-`revision` cache gate skips rebuilding it while no
  tile has changed, collapsing catch-up-burst rebuilds (357 → 13 builds in the flood).
- **Incremental region relabelling** — `_compute_regions` / `find_enclosed_rooms`
  recompute the whole map whenever a single tile edit bumps `revision`; relabel only
  the affected component instead.
- **Scheduler heap** — `next_actor()`'s O(n) scan → a heap (the interface already
  anticipates it; see [Action Economy](Action-Economy.md)).
- **Entity spatial index** — per-tile buckets for `get_components(Position)` scans.
  *Deferred:* the profile shows these scans are not the bottleneck (pathfinding is);
  revisit only if a measurement says otherwise.
- **Everything in memory at startup** — sounds, tiles, state (per `next.md`).

See [Roadmap](Roadmap.md) for sequencing.
