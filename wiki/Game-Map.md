# Game Map

The map is plain data, independent of any renderer, in
[src/game_map.py](../src/game_map.py). A tile is a single character (`#` wall, `.`
floor, `~` water, `+` door, `o` window). It carries the world layout **and** the
spatial algorithms systems rely on — pathfinding, reachability, and room detection.

## The ocean/island world

The real world map is `360 × 180` (`LAND_WIDTH*3 × LAND_HEIGHT*3`). A map at least
twice the land size in each dimension becomes an **island in a vast ocean**: the
central `120 × 60` land rectangle (floor, ringed by a water coastline) sits inside
open sea, with a hard wall only at the outermost world edge (`world_land_rect`).
Anything smaller (every test map) stays a **plain walled room** — one code path, so
tiny test maps keep the classic layout. Two elliptical **lakes** and a meandering
**river** are carved into the land (deterministically, no RNG), skipping a protected
zone around the player's spawn. Because the layout derives from width/height, a save
only needs to store the size to reconstruct the same island.

## Tile queries

| Method | Returns | Notes |
|--------|---------|-------|
| `in_bounds(x,y)` | `bool` | Inside the grid |
| `is_walkable(x,y)` | `bool` | Not wall/water/window — **NPC** pathfinding (land only) |
| `is_passable(x,y)` | `bool` | Not wall/window — the **player** can swim (water allowed) |
| `is_water(x,y)` / `is_ocean(x,y)` | `bool` | Water; open sea (outside the land rect) |
| `tile_at(x,y)` | `str` | The tile glyph, for rendering |
| `set_tile(x,y,tile)` | `bool` | Runtime edit (build/clear); never overwrites the border; bumps `revision` + records dirty/edit deltas |
| `clear_water_around(x,y,r)` | — | Safety: turn nearby water back to floor around a spawn |

## Change tracking (for caches)

Runtime edits (a wall raised, a door placed) must invalidate derived state without
forcing a global rebuild. `set_tile` maintains three signals:

- **`revision`** — bumps on any tile change; region labels and the renderer's cached
  world surface compare against it.
- **`consume_dirty_tiles()`** — the exact tiles edited since the last call, so the
  renderer repaints just those cells instead of the whole (large) surface. See
  [Performance](Performance.md).
- **`region_edit_revision(x,y)`** — per-120×60-region edit counters, so a cache scoped
  to one region invalidates only when a *nearby* edit lands (used by
  `HousingProcessor`'s build-site memo).

## Spatial algorithms

- **`find_path(start, goal, blocked=None)`** — 8-way unit-cost **A\*** with a
  Chebyshev heuristic and a goal-biased tie-break on equal `f`; neighbour/bounds/
  walkability inlined for the hot loop. Optimal like BFS but ~30× faster / ~50×
  fewer nodes on open ground. Returns the path excluding `start`, including `goal`
  (`[]` if unreachable).
- **`distance_field(goal)`** — BFS distances from `goal` to every walkable tile that
  can reach it: a **flow field** computed once and reused by many travellers over
  many turns (step to the lowest-valued neighbour to make progress) without a fresh
  pathfind.
- **`region_of(x,y)` / `same_region(a,b)`** — connected-component (8-connectivity)
  labels for walkable tiles, cached until the map changes. A cheap reachability test
  so NPCs never chase a resource across a river they can't cross. *(Distinct from the
  arithmetic simulation grid in [regions.py](../src/regions.py) — that one is a plain
  spatial bucket for scheduling, not a walkability label.)*
- **`find_enclosed_rooms(max_size=400)`** — flood-fill over floor tiles bounded by
  wall/window/door, returning interiors sealed off with at least one door: this is how
  a **house** is recognised (see [Gameplay](Gameplay.md)).
- **`line_points` / `has_line_of_sight`** — Bresenham line + wall-only sight blocking
  (water/windows stay transparent).
- **`neighbors_4` / `neighbors_8`** — bounded neighbour lists.

## Extending later

The design leaves room to grow without touching the render path — e.g. replacing the
bare tile char with a `Tile` dataclass (walkable/transparent/colours) while keeping
`tile_at` returning a glyph. See [Roadmap](Roadmap.md).
