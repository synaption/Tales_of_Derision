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

## Compacted travel: one activity, many turns' worth of time

Out where nobody is watching, a walk is settled as a **single activity** rather than
a turn per tile. An NPC beyond the full-simulation box that decides to travel covers
the whole leg at once (`NpcAiProcessor._far_travel`, capped at
`_COMPACTED_TRAVEL_STEPS = 32` tiles) and is then billed for the tiles it didn't pay
for the ordinary way — `_charge_travel` simply overdraws its `Actor.energy`.

That overdraft *is* the "busy walking" state: the energy loop won't call the NPC
again until the following region-turns have paid the balance back off. No second
clock, no busy flag, no queue.

The schedule is unchanged by this. Twelve tiles a turn at a time means an NPC that
decides to travel on region-turn *N* arrives on *N+11* and does the next thing on
*N+12*; compacted, it arrives on *N* but stays in arrears until *N+12* — same next
action, same turn. What disappears is eleven rounds of re-ranking its drives and
re-scanning its surroundings, which is the expensive part.

Two deliberate limits:

- The cap keeps a long trek from being decided once and blindly executed — a
  traveller stops to take stock every 32 tiles, so it can notice it got hungry.
- A leg **stops on the threshold of the full-sim box**, so a creature walking toward
  the player never materializes mid-stride in front of them; the last stretch is
  walked a tile at a time like anything else on screen.

Measured on a 9-island world after 200 turns of walking (3300 region-turns of debt):
catch-up **2.20 s vs 3.17 s**, i.e. 0.67 vs 0.95 ms per region-turn. Per-turn cost
while walking is unchanged (~2.8 ms) — the player's own region is never compacted.
Time is conserved: action slots spent per energy grant is 1.02 with compaction and
1.02 without.
