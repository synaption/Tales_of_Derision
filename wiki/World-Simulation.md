# World Simulation

The whole `360 × 180` world stays alive, not just the section you're standing in —
but simulating every tile every turn would not scale. The **region scheduler** lets
far-away regions lag behind the current turn and pay that debt down later, **nearest
to the player first**, instead of every region-aware system rescanning the whole map
every turn. Defined in [src/regions.py](../src/regions.py); used chiefly by
`NpcAiProcessor` (and pioneered by `FishAiProcessor`).

## The region grid

The world is bucketed into `120 × 60` arithmetic cells (`region_at`,
`region_grid_size`, `region_bounds`). This is a plain **spatial bucket** for
scheduling — unrelated to `GameMap.region_of`, which is a topological *walkability*
label. A map smaller than one region (every test map) collapses to a single region,
so region-aware systems behave exactly as an unpartitioned global system would.

`in_region_with_margin` gives a region-scoped search a little visibility across the
seam, so a creature near a boundary doesn't lose sight of a resource one tile into the
next region.

## `RegionScheduler`

Owns each region's *"simulated up to turn N"* cursor and pays down debt.

- **`register(name, step)`** — add a per-region single-turn step (run in order).
- **`advance_region(region_id)`** — run every step for that region's next turn, then
  bump its cursor. Catch-up always **replays** turn N, N+1, N+2 … in strict order —
  never an analytic shortcut — so state one step builds (NPC positions, needs) stays
  consistent for the next. Different regions sit at different cursors at the same real
  moment; that's the whole point.
- **`catch_up_region(region_id, target_turn)`** — block until one region reaches a
  turn. **Used when you enter a region**: bring that whole 120×60 area fully up to
  date before it's shown/played.
- **`catch_up_all(target_turn)`** — bring the **entire world** up to date. **Used when
  you sleep** (the world fast-forwards through the night).
- **`pump_background(budget_seconds, player_region, target_turn, wall_clock)`** — spend
  a real-time budget advancing the **nearest** lagging region (Chebyshev distance to
  the player), never the stalest, until nothing lags or the budget runs out.

## Idle background pumping (the "more sim when idle" goal)

Idle time — the player thinking, or away from the keyboard — is exactly when there's
the most spare time to pay down simulation debt. The turn loop (`main.py`) polls input
on a short timeout; on each idle tick with no input it calls `pump_background` with a
budget that **ramps up** the longer nothing happens (`_idle_pump_budget`,
`_IDLE_PUMP_BASE_BUDGET` → `_IDLE_PUMP_MAX_BUDGET` by `_IDLE_PUMP_RAMP`). So a still
world quietly catches its distant regions up, and an active one spends the budget on
what's near.

This realises the design goals in `next.md`: *simulate the whole world, prioritise the
nearest tiles (not the stalest), bring a region fully up to date on entry, and bring
the whole world up to date on sleep.*

## Cost & correctness notes

- Catch-up cost concentrates on **region entry** (entering a lagging region replays
  every missed turn at once) — this is where pathfinding cost shows up. Kept in check
  by A* + reused flow fields (see [Game Map](Game-Map.md), [Performance](Performance.md)).
- A freshly built world starts every region "caught up" to its creation turn (there's
  no history to replay), which is always correct because per-region state isn't
  persisted across save/load yet — the world regenerates fresh from its seed.
- See [Action Economy](Action-Economy.md) for how region-turns grant `Actor.energy` so
  faster creatures act more often even in the far, region-simulated world.
