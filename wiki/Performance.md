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

The lever is the synchronous per-turn `ecs.process()` spike. **Profile the real
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

## Per-kind cache invalidation

`kind_neighborhood_version` turned out to matter well beyond goal maps.
`_static_region_items` (the region's trees and ovens) and `_dynamic_region_items`
(deer, corpses, bushes, people) both keyed on the index's **region-wide** version,
which moves whenever any entity anywhere in the nine-region neighbourhood appears,
dies or crosses a seam. A 40% miss rate, and `_static_region_items` alone was **31.6%
of whole-world catch-up** — 198 µs a call, 16 814 calls.

Keying each on the versions of the kinds it actually holds took catch-up at 25 islands
from **10.5 s to 8.3 s** and dropped it out of the profile entirely. Trees and ovens
change when trees and ovens change; a wandering deer has nothing to say about either.

Where catch-up time goes now (25 islands, 29 000 region-turns): `_take_turn` 70%, of
which `_step_toward`/`find_path` 25%, `_go_to_nearest` 22% and `_rank_drives` 17%.
That is real decision-making rather than bookkeeping — the next win there is fewer
decisions, not cheaper ones (see
[World Simulation](World-Simulation.md#analytic-catch-up-skipping-turns-nothing-happens-on)).

## Counting instead of scanning

The daily flora pass (`TreeGrowthProcessor`) was the last system whose cost was set
by how much world there is rather than by how much of it changed. Per region per
day it rolled once for every one of ~7200 outdoor ground tiles, once for every
ocean tile, and once for every plant, and it rebuilt a whole-world snapshot (five
`ecs.get_components` sweeps over ~85 000 entities, plus a set of every occupied
tile in the world) for each elapsed day. All of that to plant roughly one sapling.

Nothing about the *rules* required any of it. A per-tile chance over N tiles is a
binomial, and the gap between successes is geometric, so the day can ask the
distribution which tiles sprout and get the answer in one draw per sprout —
`_bernoulli_hits(n, p, rng)`. Deadlines (a sapling's year, a bush's seven days) are
known when they start, so they are filed under the day they come due instead of
being searched for. Populations and cap totals come from the spatial index in O(1).

| 1400×800, 143 regions | before | after |
|---|---|---|
| one day, whole world | 166 ms | **19.1 ms** |
| 200-day `catch_up_all_flora` | 15.0 s | **0.17 s** |

The 88× on catch-up is larger than the 8.7× on a single day because the old
whole-world snapshot was rebuilt per elapsed day; there is no snapshot now. What
remains is the cost of creating the plants themselves — nothing in the pass scales
with the size of the world any more.

### The tile check: `rooted`

Counting *which* tiles sprout still leaves the question the per-tile scan answered
for free — is that tile already taken? The first cut collected the region's
occupied tiles on demand, which left one term proportional to population and
promptly became ~70% of the pass.

The right answer is to ask per candidate and give up on a tile that's taken, which
needs an O(1) tile → entity lookup. `blocker_at` almost does it, but saplings and
seaweed occupy a tile *without blocking it*, so half the flora is invisible to it.
Hence `SpatialIndex.rooted` — the same tile-map machinery as `blockers`, for
plants. It is affordable for the reason `blockers` is: **nothing in it moves**, so
keeping it true costs entity creation and destruction, never a step. Measured on
`moved()` (the per-step hook): 615 ns → 655 ns, and it does not appear in a turn
profile at all.

Creatures are deliberately *not* consulted when sprouting. A sapling under a
passing villager is harmless — it doesn't block, and `_mature_saplings` won't let
it become a tree while anyone stands there — and checking for them is exactly what
would drag a region's population back into the pass.

The one bulk read left is `_region_blockers`, used when a sapling matures, which
genuinely does care about creatures. It fires about once a day in the few regions
that have land at all.

**It also fixed a latent bug.** Sprouting stamped `planted_turn` from the world
clock while the region replayed a day far behind it, so saplings planted during a
catch-up were dated to the present and could never reach maturity — 200 days of
simulation used to end with ~160 saplings and *fewer* trees than it started with.
Sprouts are now dated to the region's own day, and the same run ends with ~90
saplings and ~40 more trees. Over 1000 days the forest climbs steadily toward its
soft cap and `SpatialIndex.audit()` stays clean throughout.

## Populations instead of individuals

With flora counted rather than scanned, a night's sleep profiled at 3.93 s and
**74% of it was fish** — not because fish AI is expensive per fish (an NPC costs
76 µs a creature-turn, a fish 3.2 µs) but because there are 1121 of them against
16 NPCs, and sleep replayed every one for all 800 turns. ~900k fish-turns of
random-walking that nobody observed.

Two things were wrong with that. The cheap one: `FishAiProcessor` registered only
a `step` with the region scheduler — no `bulk`, no `idle_turns` — so per
[`jump_limit`](../src/regions.py) it was pinned to one turn at a time, the only
system to opt out of the skipping machinery while holding 99% of the entities.
The deep one: it was computing the wrong thing. The single durable output of all
that simulation was a seaweed count.

[wildlife.py](../src/wildlife.py) replaces it with per-region population stocks
(see [Systems](Systems.md#wildlifeprocessor-priority-0--populations-not-individuals)).
A region with no fish *entities* has nothing for the fish step to do, so
`_idle_turns` reports an unbounded span and the scheduler clears the whole
debt in one visit.

| a 800-turn night, 1400×800 | before | after |
|---|---|---|
| fish AI | 2.90 s (74%) | **0.024 s (1.7%)** |
| NPC AI | 0.97 s | 0.98 s |
| flora | 0.06 s | 0.06 s |
| wildlife day model | — | 0.38 s |
| **total** | **3.93 s** | **1.43 s** |

Fish catch-up is **120× faster**; a night is 2.7× faster overall, and NPC AI is
now the bottleneck at 68%. Entity count drops 32 766 → 31 658, and the per-turn
loop is unchanged (residency sync walks the couple of dozen *people*, not the 143
regions, and does not appear in a turn profile).

### What it costs, and the tuning it exposes

The day model is ~22 ms per world-day for 143 regions — the same order as flora,
and off the keypress path by the same route.

It also made an existing imbalance visible, which is the point of putting a real
number on `forage_efficiency`. Run for 448 days the sea overshoots and settles:
fish 1121 → 3788 by day 112, seaweed grazed 31 000 → 3 800 by day 168, then fish
fall back and both hold steady around **2400 fish / ~1800 seaweed** from day 224.
Textbook logistic overshoot, stable, no extinction.

### Spreading, and why the sea now recovers

That first cut collapsed monotonically: seaweed fell to ~6% of its world-gen
density and stayed there. The cause was structural — seaweed regrowth was a flat
per-ocean-tile chance *independent of standing stock*, so it could not accelerate
when grazed down, and the system settled wherever consumption met that constant.

Plants now **spread from what already stands** (`_spread_factor`, logistic), so
regrowth rises as a thinned reef recovers. That turns a one-way collapse into a
damped predator–prey oscillation that converges. Ten in-game years, 1400×800:

| day | fish | seaweed | trees |
|---|---|---|---|
| 1 | 1189 | 30 908 | 656 |
| 112 | 4501 | 14 928 | 654 |
| 224 | 837 | 4 086 | 693 |
| 336 | 1800 | 20 584 | 739 |
| 560 | 1724 | 9 939 | 849 |
| 840 | 1902 | 11 814 | 989 |
| 1120 | 1842 | 13 471 | 1112 |

Fish boom, overgraze, crash, and the reef comes back; the swings damp out to
roughly **1850 fish / 13 000 seaweed** while the forest grows steadily toward its
own cap. `_DAILY_SEAWEED_SPROUT_CHANCE` was doubled to 0.0012 as part of this:
spreading makes a *sparse* reef slower to recover, so the peak rate has to be
high enough that a healthy reef outgrows the shoal eating it.

Two things worth knowing:

- **World-gen over-stocks the sea.** It lays down ~31 000 seaweed against an
  equilibrium near 13 000, so the first ~6 in-game months always show a fish boom
  and a crash before things settle. Dropping the world-gen seaweed density
  (0.028 → ~0.013 of ocean tiles) would start the world at its own equilibrium
  and skip the opening swing — a one-line change in `_spawn_ocean_life`, left
  alone because the settling-in is arguably worth watching.
- **A bug the separation caught.** Giving seaweed a seedling stage as a third
  `Sapling.kind` silently stopped forests growing: `flora_total` counts saplings
  against the *land* cap, and 4000 young fronds instantly exceeded it. Hence
  `SeaSprout` as its own component — the two habitats have to be countable apart.
  Pinned by `test_young_seaweed_is_not_counted_against_the_land_flora_cap`.
- **And one in the audit.** `SpatialIndex.audit` was the only query that didn't
  `sync()` first, so it compared unrepaired state against the world. Since the
  index's contract is *repaired on read*, that measured a state no caller can
  observe and reported false staleness for any deletion after the last read —
  which is precisely what dissolving a shoal into a stock does. It read clean
  before only because a read always happened to precede it.

### Strict active/inactive partitioning

`ecs.process(action)` simulates the region the player stands in and nothing else.
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
  incrementally (three movement hooks; creation and deletion are noticed from the ECS's
  own id counter, so no call site has to remember). Systems ask for a region's
  entities instead of the world's. Measured over 300 turns after a 150-turn settle,
  at 9 islands: `MovementProcessor` 1.63 → 0.02 ms/turn, `NeedsProcessor` 0.15 → 0.03,
  `HousingProcessor` 0.88 → 0.47, whole turn 10.8 → 8.2 ms. What remains in a turn is
  bounded per-island tile work (`distance_field` floods), not entity scans.
- **Everything in memory at startup** — sounds, tiles, state (per `next.md`).

See [Roadmap](Roadmap.md) for sequencing.
