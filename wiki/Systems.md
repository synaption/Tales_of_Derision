# Systems

Systems (called **processors** here) hold all behaviour. Each subclasses
`ecs.Processor` and implements `process(*args)`, receiving whatever was passed to
`ecs.process(...)`. Most live in [src/systems.py](../src/systems.py) alongside a
large set of free **helper functions** the processors and turn loop share; the two
largest were split into siblings — `RenderProcessor` into
[src/render.py](../src/render.py) and `NpcAiProcessor`/`FishAiProcessor` into
[src/ai.py](../src/ai.py). Both are re-exported through `systems`, so
`from systems import RenderProcessor` (etc.) still works; always import them via
`systems`, never directly, so module-load order stays sound.

Registration and priority are in [Architecture](Architecture.md#the-turn-loop).
Time-advancing systems tick only on a real action — the movement actions **plus**
`WAIT_ACTION` (`"wait"`); menu refreshes pass `None` and advance nothing.

## The processors

### `TimeProcessor` (priority 2)
Advances the `WorldClock` by the acting entity's action **cost** in time units, not
by 1 (see [Action Economy](Action-Economy.md)). Runs first so needs/AI read the
correct time of day, and emits a log line when the day's phase changes
(`_PHASE_MESSAGES`). Time-of-day, calendar, night detection, and age-from-clock all
derive from `WorldClock.turn` (`time_phase`, `is_night`, `calendar`,
`format_datetime`, `age_years`) — nothing is ticked per entity.

### `MovementProcessor` (priority 1)
Applies a movement action to the player. Walking is gated by `is_passable` (so the
player can **swim**); walking into a creature is a **bump attack** (`slay_entity`
shared with the AI), with optional melee/death SFX callbacks. Only `Position + Player`
entities move, so it generalises to co-op for free.

### `NpcAiProcessor` (priority 0) — the land brain
The single per-turn brain for every land `NPC` (~1000 lines; the biggest system). Per
creature, roughly in order:

1. **Urgent need** (≥ `_FORAGE_THRESHOLD`, thirst wins ties): drink at/route to
   water; when hungry, eat a *prepared* item first, else act by `Diet` —
   **herbivores** graze the nearest tree, **carnivores** hunt the nearest
   corpse/deer and eat raw, **cooks** run the full loop one step per turn
   (`_forage_meat` → `_gather_wood` → `_cook_at_stove` → eat next turn).
2. **Sleep** when tired (`_seek_sleep`, preferring `Home`).
3. **Build** when needs are met and a reachable blueprint exists (`_work_blueprints`
   → haul wood, raise stocked ghosts).
4. **Socialize** with a nearby compatible partner (`_socialize`).
5. **Chase** the player when hostile and in vision + line of sight (`_chase_player`).

Water tiles and their walkable shores are precomputed (the map is static), so a
thirsty animal is a cheap nearest-lookup plus one BFS, not a full-map rescan. A
predator can delete its prey mid-loop, so the processor snapshots the NPC list and
skips anything already killed this turn.

**Scaling:** this processor drives NPCs through the **region scheduler** — far
sections lag and catch up nearest-first instead of every NPC simulating every turn.
See [World Simulation](World-Simulation.md) (`_advance_region`, `_region_bucket`,
`_build_world_snapshot`).

### `FishAiProcessor` (priority 0) — the sea brain
The aquatic mirror: fish swim water-only (never the land pathfinder), wander, and
graze seaweed when hungry. It pioneered the region-bucketed simulation later
generalised into [regions.py](../src/regions.py).

### `NeedsProcessor` (priority 0)
Raises hunger/thirst/tiredness each real turn (tiredness faster at night via
`_NIGHT_TIREDNESS_MULTIPLIER`), and emits an escalating warning the first turn the
player crosses each 50/80/100% threshold. Eating/drinking/sleeping lower the values.
There is no HP system yet, so a maxed need currently warns rather than damages.

### `HousingProcessor` (priority 0)
Runs before the AI so a villager that just got a home can start heading there. Claims
the nearest reachable **unowned** house for a homeless `Resident`, merges spouses'
homes, and — when nothing is claimable — ensures a reachable **construction site**
exists to build one. Uses a back-off memo (`_no_site`, keyed by region cell +
edit-revision) so a villager with nowhere to build doesn't re-run the site search
every turn (a former hot spot — see [Performance](Performance.md)).

### `TreeGrowthProcessor` (priority 0)
Evaluated **once per in-game day**. Every plant in the world — trees, berry
bushes, seaweed — lives the same life:

**seedling → (matures) → grown plant → (rots, or is eaten)**

- `Sapling` on land (→ `Tree` after 112 days, or `BerryBush` per its `kind`);
  `SeaSprout` at sea (→ `Seaweed` after `_SEAWEED_MATURE_DAYS`, 7). They are two
  components rather than one so the land's soft cap and the sea's stay countable
  apart — a sea full of young fronds must not tell the forests they are full.
- Every grown plant can die at `_DAILY_DEATH_CHANCE`, seaweed included. Before,
  a frond could only ever leave the world down a fish.
- Berry bushes regrow their crop 7 days after being picked.

**Plants spread from what already stands.** The per-tile sprout chance is scaled
by `_spread_factor(standing, capacity)` — logistic, so growth is fastest at half
capacity and tails off at both ends: nothing to spread from when a region is
bare, nowhere to go when it is full. A floor (`_COLONISE_SHARE`, 5%) carries seed
in from elsewhere so a logged or grazed-out region can always come back; without
it, spreading could lock a region at zero forever.

Standing stock and capacity are both O(1) index reads, so this costs nothing and
sprouting stays a single binomial draw — see below.

**It does none of that by scanning.** Written literally those are four sweeps per
region per day — every tile, every plant, every sapling, every bush — so the
processor states the same rules two other ways:

- **Chance is counted, not rolled.** A per-tile or per-plant chance over N
  candidates is a binomial, and the gaps between successes are geometric, so
  `_bernoulli_hits(n, p, rng)` samples *which* trials succeed directly — one
  random number per success instead of one per trial. A region's 7200 ground
  tiles cost the ~1 sapling that sprouts on them, not 7200 rolls. It is the same
  Bernoulli process, so the distribution is unchanged (and the injected test RNG
  keeps its meaning: `0.0` makes every trial a hit, `1.0` none).
- **Deadlines are filed, not searched for.** A sapling's maturity day and a
  picked bush's regrow day are known the moment each is created, so they are
  filed under the day they come due (`_mature_due`, `_berry_due`). The daily pass
  pops that day's entries; no plant is ever asked whether its time has come.
  `pick_berries` is the only thing that makes a bush bare, so it is also the only
  place a regrow gets scheduled.

Populations come from the spatial index (`of_kind`, O(1)) and the world-wide
soft-cap totals from `spatial.component_population` (also O(1)), so not even the
caps need a scan. "Is this tile already taken?" is `SpatialIndex.rooted_at` —
a tile map of plants, needed because saplings and seaweed occupy a tile without
blocking it, and cheap because nothing in it ever moves. A candidate tile that is
taken is simply given up on. See
[Performance](Performance.md#counting-instead-of-scanning).

### `WildlifeProcessor` (priority 0) — populations, not individuals

Lives in [wildlife.py](../src/wildlife.py). Wild animals are **per-region counts**,
not thousands of persistent individuals walking the world. A species is in one of
two forms:

- **Stocked** — the normal case. A region holds an integer: "310 fish". No
  entities, no positions, no per-turn work. Population changes here, once a day.
- **Resident** — a region someone can watch (the player's, or one holding an NPC
  that might hunt) materialises its stock into real entities, which the per-turn
  AI swims or walks exactly as before.

At shipping size that is **1121 fish held as 13 entities** in two live regions.
Individuals are therefore not persistent — the fish you swam past yesterday is
not the same fish today. That is the trade: the *population* is what is being
modelled, and the population is what persists.

**Residency runs every turn** — the player crossing a seam must materialise the
sea before it is drawn, so it cannot lag. It is cheap: a walk over the couple of
dozen *people* in the world, not the hundred-odd regions.

**Population runs every day, on the flora's cursor, not its own.** Animals eat
plants, so the two day models must advance in lockstep; `register_on` makes the
population a step on `TreeGrowthProcessor`'s per-region day cursor. Given each its
own cursor — as they briefly had — a long catch-up could grow a month of seaweed
with nothing grazing it and then graze a month with nothing growing. See
[World Simulation](World-Simulation.md#coupled-day-models-need-one-cursor).

A day, per region:

```
demand    = count * bites_per_day
eaten     = min(demand, standing_food * forage_efficiency)
deaths   ~= starve_chance * (1 - eaten/demand)  +  predation_chance * predators
births   ~= breed_chance, if fed and under the region's density cap
```

Births and deaths are `rng.binomial` draws, so a day costs the animals that were
actually born or died. A species is a `Species` dataclass — adding one is a
declaration, not a code path. `predators` / `predation_chance` are the plug-in
point for the predator layer; `DEER` is declared but left resident everywhere
until deer scale with the land (see the note on it).

`forage_efficiency` is worth understanding: before this existed, fish "balanced"
against their food only because a wandering fish usually failed to *find* seaweed
within its sight radius. The ecology was propped up by a bad search. It is now an
explicit number.

### `ReproductionProcessor` (priority 0)
Delivers due pregnancies: after `_GESTATION_DAYS` it spawns a newborn beside the
mother, wires the family links reciprocally, names it via the onymancer, and removes
`Pregnant`. Courtship/marriage/mating themselves live in the social helpers.

### `RenderProcessor` (priority 0) — draws last
Draws the world every turn (~1200 lines; being split into a `render/` package). Key
responsibilities:

- **Field of view** memoized per player position; **remembered tiles** (fog of war)
  drawn dimmed from memory when out of sight.
- **Section camera** — only the player's 120×60 section is drawn; crossing an edge
  snaps the camera and logs "You cross into a new area." (`_apply_section_camera`).
- **Autotiling** — wall/water neighbour masks ([47-tile](47-tile_autotiling.md)),
  cached (static map).
- **Cached world surface** — the map is composited from a cached world-coordinate
  surface and blitted per frame, never re-drawn tile-by-tile (see
  [Performance](Performance.md)).
- **Sidebar / log / status line** — date+clock+phase, needs, nearby objects, message
  log.
- **Status-identifier animation** (`_status_appearance`) — real-time glyph cycling
  from `_STATUS_DISPLAY` / `_STATUS_ORDER`.

It talks only to the [`Renderer`](Renderers.md) interface, never to pygame.

## Free-function subsystems (shared helpers)

These aren't processors but are large behaviour clusters the processors and turn loop
call. They'll become the modules of the future `systems/` package:

- **Time/calendar** — `world_clock`, `time_phase`, `is_night`, `calendar`,
  `format_datetime`, `age_years`, `night_overlay_alpha`.
- **Sleep** — `go_to_sleep`, `wake_up`, `_camp_at`.
- **Housing/ownership** — `houses_for`, `bed_owner`, `set_bed_owner`, `owned_bed_of`,
  `house_is_owned`, `set_house_ownership`, `furnish_house`.
- **Construction** — `choose_build_site`, `create_construction_site`,
  `_spawn_blueprint`, `stock_blueprint`, `raise_blueprint`, `_complete_site`.
- **Social** — `_TRAITS`, `personality_warmth`, `interaction_delta`, `friendship`,
  `adjust_friendship`, `gibberish`, `spawn_speech_bubble`, `active_bubbles`,
  `interact`, `_react`.
- **Family/reproduction** — `are_courtship_eligible`, `try_marry`, `try_mate`,
  `_merge_homes`, `_adopt_surname`.
- **Combat/messages** — `slay_entity` (shared corpse creation), `queue_message` /
  `_pull_turn_events` (the turn-event log).

## Adding a system

1. Subclass `ecs.Processor` and implement `process(self, action=None)`.
2. Register it in `game.py` with a `priority` placing it correctly relative to
   Time (2), Movement (1), and Render (0).
3. Query the components it needs with `ecs.get_components(...)`; gate time-advancing
   work on `action in _TURN_ACTIONS`.

For a new *creature/item/effect*, you usually **don't** add a system — you add a
[prefab or effect](Content-and-Mods.md) and let the existing systems act on the
components it carries.
