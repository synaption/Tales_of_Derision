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
workload, not intuition** — a cProfile of a cross-region walk with a `FakeRenderer`
(so the map-surface build doesn't mask sim cost) is the trustworthy signal.
Whole-turn A/B timings are confounded because A*'s tie-break changes NPC routes and
the two runs diverge; trust divergence-proof metrics (per-call time, node counts).

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

## Scaling levers on the roadmap

- **Entity spatial index** — replace the O(n) `esper.get_components(Position)` scans in
  targeting/interaction/AI-occupancy with per-tile / per-region entity buckets. The
  biggest lever for thousands of NPCs.
- **Scheduler heap** — `next_actor()`'s O(n) scan → a heap (the interface already
  anticipates it; see [Action Economy](Action-Economy.md)).
- **Everything in memory at startup** — sounds, tiles, state (per `next.md`).

See [Roadmap](Roadmap.md) for sequencing.
