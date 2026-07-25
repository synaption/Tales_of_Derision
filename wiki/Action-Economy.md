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
