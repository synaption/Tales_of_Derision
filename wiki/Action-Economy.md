# Action Economy

Turn order is decided by **completion time**, not a fixed round: whoever's next
action lands soonest acts next. Time is measured in **time units (TU)**. Defined in
[src/action.py](../src/action.py) and [src/schedule.py](../src/schedule.py); the
`WorldClock` and `Actor` components carry the state.

Everything defaults so an all-baseline world (average dexterity, unit action weights)
behaves exactly like the old one-action-per-turn lockstep — the switch is invisible
until speeds are tuned.

## Cost model (`action.py`)

- **`BASE_ACTION_COST = 100`** — one baseline action (an average creature taking a
  plain step). A day is `BASE_ACTION_COST × 240` TU, so `WorldClock.day_length` is
  24000.
- **`ACTION_WEIGHTS`** — per-action-type multipliers (empty today = all `1.0`). Add
  an entry to make a specific action slower/faster (a heavy attack, a slow chop).
- **`actor_speed(ent)`** — from `Attributes.dexterity`: `1.0 + (dex-10) *
  SPEED_PER_DEXTERITY`, floored at `_MIN_SPEED`. Average dexterity (10) = `1.0`;
  higher is faster. Entities without `Attributes` act at baseline.
- **`action_cost(ent, action) = round(BASE_ACTION_COST × weight / speed)`** — the TU
  the actor spends. A quick creature (a rat at `dexterity=16`) pays less and so acts
  more often.

## Scheduling (`schedule.py`)

The schedule lives on the entities themselves — each actor's **`Actor.next_time`** —
rather than a separate heap that could drift out of sync with the ECS.

- **`schedule_actor(ent, at_time=0)`** — give/retime an `Actor` (idempotent).
- **`next_actor()`** — the entity with the smallest `(next_time, entity_id)`. The
  entity-id tie-break makes equal-time turns fully deterministic — a fixed seed + a
  fixed input sequence reproduce the world exactly.
- **`complete_action(ent, action)`** — stamp `last_acted = next_time`, push
  `next_time += action_cost(...)`, return the cost.

`TimeProcessor` advances the `WorldClock` by the acting entity's cost, so effects that
accrue over time (needs) can use the exact elapsed span.

> **Note:** `next_actor()` is an O(n) scan today — fine while the near cast is small
> and it can never desync from the ECS. Far-away crowds stay cheap through the
> [region scheduler](World-Simulation.md), not this queue. A heap can replace the scan
> later without changing this interface — a [Roadmap](Roadmap.md) perf lever.

## How the two combine

- The **near** cast (on/around the player's section) runs on this completion-time
  queue — precise, deterministic turn order with real speed differences.
- The **far** world runs on the [region scheduler](World-Simulation.md), which grants
  each region-turn `BASE_ACTION_COST` of `Actor.energy` per NPC and spends the action
  cost per action — a quicker creature banks the surplus and acts again. Same cost
  model, cheaper bookkeeping for crowds.

## Compacted activities: one turn, many turns' worth of time

Outside the full-simulation box nobody can watch an NPC spend its turns; only the
outcome is ever observed. So an activity that is N region-turns of the same
unobservable repetition — walking a road, hauling load after load of wood, sleeping
a night through — is **settled in a single region-turn** and the actor is billed the
whole N. It ends up where and how it would have, on the turn it would have; what
disappears is N−1 rounds of re-ranking its drives and re-scanning its surroundings,
which is the expensive part.

Defined in [src/ai.py](../src/ai.py) under the *Compacted activities* banner.
`_compactable(xy)` is the gate — today exactly "outside the full-sim box", the same
line that picks cached-route over full pathfinding. With no player at all (unit
tests, whole-world catch-up) nothing is compacted, so those paths keep running the
exact, unabridged turn-at-a-time simulation.

### The two mechanisms

**1. Bill the time** — `_charge_activity(ent, turns)` overdraws `Actor.energy`, and
`_advance_region`'s energy loop won't call the NPC again until the following
region-turns have paid the balance off. That overdraft *is* the "busy" state: no
second clock, no busy flag, no queue. Needs keep accruing throughout, so the
activity is genuinely tiring. Used where the per-turn effects really do happen over
those turns.

For a *repeating* activity the generic driver is `_run_compacted(ent, pos, cap,
one_turn)`: hand it the activity's ordinary single-turn body — returns True when it
did a turn's work, False when there's nothing left — and it runs turns until the
body stops, then bills the difference. **An activity written the normal way is
compacted by being handed to this**, and never has to know about it.

**2. Settle the turns** — apply the whole effect in closed form and tag the entity
[`Settled(activity, turns)`](../src/components.py). Per-turn systems skip a
`Settled` entity and burn one turn off the receipt instead, so they can't live those
turns twice. Used where the activity is pure arithmetic.

Every activity has a `_COMPACTED_*` cap, all for the same reason: an activity
settled in one decision is a decision made without noticing anything that happened
during it, so a long one is broken into legs that re-decide.

### The three users today

| activity | mechanism | cap |
|---|---|---|
| travel (`_far_travel`) | bill the time | `_COMPACTED_TRAVEL_STEPS = 32` tiles |
| hauling to a blueprint (`_work_blueprints`) | bill the time, via `_run_compacted` | `_COMPACTED_ACTIVITY_TURNS = 8` turns |
| sleeping (`_bed_down`) | settle the turns | `_COMPACTED_SLEEP_TURNS = 120` turns |

**Travel.** An NPC that decides to walk covers the whole leg at once. Twelve tiles a
turn at a time means it decides on region-turn *N*, arrives on *N+11* and does the
next thing on *N+12*; compacted, it arrives on *N* but stays in arrears until
*N+12* — same next action, same turn. A leg **stops on the threshold of the full-sim
box**, so a creature walking toward the player never materializes mid-stride in
front of them; the last stretch is walked a tile at a time like anything else on
screen. `_run_compacted` stops on that same threshold.

**Hauling.** Supplying one blueprint piece is a round trip of pure travel — out to
the woods, a swing of the axe, back to the site, once per log. Each of those turns
is the same unobservable errand, so `_work_blueprints` runs the trip through
`_run_compacted`. Two things had to change for the per-turn body to be safe to
repeat inside one turn: `_haul_to_ghosts` now *consumes* its `unstocked` map as it
supplies pieces (or a second pass drops a second log on a piece that already has
one), and `_gather_wood` filters felled trees out of the region's cached tree list
(or a builder keeps harvesting a stump for free wood — the list is only rebuilt
between region-turns).

**Sleep.** A night's sleep is the purest compactable activity in the game: its only
effects are the three needs, none of which depends on anything that happens during
it. `settle_sleep` in [src/systems.py](../src/systems.py) is the closed form of
`NeedsProcessor._accrue`'s asleep branch — it lives beside `_accrue` precisely so
the two can't drift, and the suite pins them together. No energy charge here, unlike
the other two: `Asleep` already keeps a sleeper out of the AI's turn, and arrears on
top would leave it standing around after waking. Anything that wakes a sleeper early
(`wake_up`) cancels the receipt.

### What it's worth

**Travel: a large win.** On a 9-island world after 200 turns of walking (3300
region-turns of debt), catch-up **2.20 s vs 3.17 s** — 0.67 vs 0.95 ms per
region-turn. Per-turn cost while walking is unchanged (~2.8 ms); the player's own
region is never compacted. Time is conserved: action slots spent per energy grant is
1.02 with compaction and 1.02 without.

**Sleep and hauling: no measurable win.** Measured the same way over 11 000
region-turns, three runs each: **0.646 / 0.650 / 0.643 ms with, 0.645 / 0.647 /
0.652 ms without** — indistinguishable. Two reasons, both worth knowing before
reaching for compaction again:

- A sleeper already skips its turn in `_advance_region`, so all a sleeping
  NPC-turn ever cost was one `_accrue` (~3 µs against ~140 µs for a turn that
  acts). Compaction replaces 14 228 of those accruals with 14 228 receipt-burns and
  adds one `has_component(Settled)` check to every *awake* creature's turn — the
  two cancel.
- Hauling barely runs in the far world at all: 24 calls to `_haul_to_ghosts` over
  400 world-turns × 12 regions. `HousingProcessor` is live-region-only, so
  blueprints are only ever staked where the player is standing, and the villagers
  who want to build are all somewhere else.

The mechanism is kept because it is correct, cheap to hang new activities off, and
the numbers above say exactly where the next one should point: `_static_region_items`
rebuilds are 35% of catch-up (40% miss rate), and `graze` — a per-turn O(trees)
nearest-target scan — is another 36%.
