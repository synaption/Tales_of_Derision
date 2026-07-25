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
  bump its cursor. Steps run in order, so state one step builds (NPC positions,
  needs) stays consistent for the next. Different regions sit at different cursors at
  the same real moment; that's the whole point.
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
the most spare time to pay down simulation debt. The turn loop (`game.py`) polls input
on a short timeout; on each idle tick with no input it calls `pump_background` with a
budget that **ramps up** the longer nothing happens (`_idle_pump_budget`,
`_IDLE_PUMP_BASE_BUDGET` → `_IDLE_PUMP_MAX_BUDGET` by `_IDLE_PUMP_RAMP`). So a still
world quietly catches its distant regions up, and an active one spends the budget on
what's near.

This realises the design goals in `next.md`: *simulate the whole world, prioritise the
nearest tiles (not the stalest), bring a region fully up to date on entry, and bring
the whole world up to date on sleep.*

## Compacted activities

A region-turn is not the same thing as an NPC decision. Outside the full-simulation
box an activity that is N turns of the same unobservable repetition is settled in
**one** region-turn, and the actor is charged the whole N — so it sits out the turns
the activity consumed instead of being woken to re-decide its life once per tile,
per log, or per hour of sleep. Arrival turn and next action are identical either
way; the repeated *deciding* is what's gone.

Three activities use it today — travel, hauling wood to a blueprint, and sleeping —
through two mechanisms: billing the time against `Actor.energy`, or settling the
turns in closed form and marking the entity `Settled`. See
[Action Economy](Action-Economy.md#compacted-activities-one-turn-many-turns-worth-of-time)
for the mechanism, how to add a fourth, and the measurements (travel is a large win;
sleep and hauling measured as no change, and the section says why).

This is the counterpart to paying region debt down faster: the pump, region entry
and sleep all get cheaper per region-turn because the far world thinks less often,
not because it simulates less.

## Analytic shortcuts and what actually has to hold

A step may work out where N turns of something end up instead of living them one at
a time. Compacted activities are the first users; tree growth and other scan-shaped
systems are the obvious next ones.

The constraint is **not** "replay every turn". It is **batch independence**:

> the result must not depend on how the turns were divided up.

That is what keeps the world deterministic, because *how many turns a region
advances in one go is not deterministic*. `pump_background` spends a real-time
budget and live region entry is capped per input frame, so the same seed and the
same player inputs batch differently on a faster machine, or a busier one. A
shortcut whose answer changes with the batching would make the world change with the
frame rate. One that doesn't is indistinguishable from replaying, and free.

Two ways that bites in practice:

- **Take the length from the activity, not from the batch.** Settle a whole night's
  sleep because the sleeper needs N turns of it — never "however many turns the
  scheduler is handing me right now". This is not just theoretical: `x + rate * N`
  is not bit-identical to two halves added in turn, so a float need settled in two
  batches drifts from the same need settled in one.
- **Draw randomness as a function of (state, N)**, not once per call, or splitting a
  shortcut in two changes the RNG stream — and with it every downstream roll.
- **Date a replayed turn to when it happened, not to when it is replayed.** A
  lagging region's turns must read the world clock at *their* position, which is
  what `NpcAiProcessor._advance_region` and `NeedsProcessor._as_of_clock` both build
  from the region's own cursor. Reading the true clock instead means a region that
  fell a day behind accrues its night-time tiredness at the daytime rate — and how
  far it fell behind is a wall-clock quantity.

Compacted activities satisfy this by construction: each one is settled once, at its
own natural length (a whole walk, a whole round trip, a whole night), decided by the
actor rather than by the scheduler's budget.

The other half of determinism is unchanged: it is seeded (`rng.py`), and only player
actions may change the course of events — the goal that time travel and undo are
built on ([Roadmap](Roadmap.md)).

## Cost & correctness notes

- **Region entry draws frames on the wall clock, not per replayed turn.**
  `ACTIVE_REGION_CATCHUP_STEPS_PER_INPUT` is 1, so the replay stays interruptible at
  the finest grain — but that used to mean one full render per replayed region-turn.
  At 100 islands that was ~80% of the whole stall spent on frames nobody could see,
  and it grew with the debt: stepping into the next region cost 0.72 s after 100
  turns of play (250 frames) and 1.66 s after 300 (450 frames). Pacing the progress
  frames at `_CATCHUP_FRAME_SECONDS` (0.1 s) makes those 0.14 s / 2 frames and
  0.24 s / 3 frames. `esper.process(None)` is render-only, so the simulation is
  identical either way.
- Catch-up cost concentrates on **region entry** (entering a lagging region replays
  every missed turn at once) — this is where pathfinding cost shows up. Kept in check
  by A* + reused flow fields (see [Game Map](Game-Map.md), [Performance](Performance.md)).
- A freshly built world starts every region "caught up" to its creation turn (there's
  no history to replay), which is always correct because per-region state isn't
  persisted across save/load yet — the world regenerates fresh from its seed.
- See [Action Economy](Action-Economy.md) for how region-turns grant `Actor.energy` so
  faster creatures act more often even in the far, region-simulated world.
