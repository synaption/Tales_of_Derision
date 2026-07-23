# Gameplay

The player-facing manual: how the survival, building, and social loops actually
play. For the code behind each loop see [Systems](Systems.md); for controls and
saves see [Getting Started](Getting-Started.md).

## Survival (food, water, wood)

The player has **hunger**, **thirst**, and **tiredness** that rise every turn (a
step or a `Space` wait; shown on the bottom status line). Keep them down or you get
escalating warnings.

- **Water** — face the stone **well** (`O`) and press interact (Enter) to drink. A
  `Waterskin` in your inventory can also be drunk from the inventory screen.
- **Meat** — kill an enemy, face its **corpse**, press Enter to open the loot menu;
  every corpse yields meat named for the creature (`Rat Meat`, `Goblin Meat`, …).
  Raw meat barely fills you — cook it first.
- **Wood** — face a **tree** (`T`) and press Enter to chop a piece of `Wood`. The
  tree falls after a few chops.
- **Cooking** — face the iron **stove** (`#`) with `Wood` **and** any raw meat in
  your inventory and press Enter: the wood fuels a fire that cooks the meat into a
  filling `Cooked … Meat`.
- **Eating/drinking** — open the inventory (`I`), select a food or drink, press
  Enter (everything else equips as before).

Interaction always targets the tile you're **facing**: hold a movement key toward
the well/tree/stove/corpse, then press Enter (the same targeting used to talk to
NPCs). Every creature accumulates hunger/thirst/tiredness each turn, though only the
player's needs surface as on-screen warnings.

## Day/night cycle and sleep

Time flows with your turns. A full **day** runs dawn → day → dusk → night (the phase
leads the status line), and the world **visibly darkens** after dusk. Everyone must
**sleep once a day**: tiredness climbs each turn and **twice as fast at night**.

- **Sleep** — press `R` to rest. Next to your **bed** (`=`) you bed down at home;
  anywhere else you **set up camp** (a campfire `^`) and sleep there. Sleeping
  fast-forwards turns — the whole world keeps simulating through the night — until
  you wake rested, then breaks camp automatically.
- **NPCs sleep too.** A tired creature drops what it's doing and heads for bed,
  preferring its **home**: villagers walk back to their house; homeless wildlife
  camps where it stands. Sleepers skip their turn until their tiredness recovers.

**Calendar.** A **year is 4 months × 4 weeks × 7 days** (112 days). The status line
leads with the full date + clock, e.g. `Y1 M2 W3 D4 13:45 Day`. The clock, phase
boundaries, tiredness rates, and the night multiplier live at the top of
[src/systems.py](../src/systems.py) (`calendar`, `format_datetime`, `_PHASE_BOUNDS`,
`_NIGHT_TIREDNESS_MULTIPLIER`) and are easy to tune.

## Houses, residents, and building

A **house** is any floor area sealed by walls (`#`) and/or windows (`o`) whose only
opening is a **door** (`+`) — windows block movement but you (and light) see through
them. Enclosed rooms are found by a flood-fill in
[src/game_map.py](../src/game_map.py) (`find_enclosed_rooms`). Every house is
furnished with a **bed**, **oven**, **chest**, **table**, **wardrobe**, and
**bookshelf** (`furnish_house` in systems).

**Houses belong to people** via their **bed** and **chest** (an `Owned` marker
records the owner). Your bed by the well belongs to *you* from the start, so
villagers never claim it. A house with no owner (or a dead owner) is up for grabs.

**Respecting property.** Sleeping in someone else's bed or opening their chest (`n`)
pops a *"This belongs to <name>. Are you sure?"* prompt (defaults to **No**). Your
own and unowned property is used without a fuss.

**Residents move in.** A villager who owns no house claims the nearest reachable
**unowned** one and lives there. Once someone owns a house nobody else takes it.

**Blueprints are shared building sites** (blueprint → haul → raise):

1. **Stake out** — a homeless resident lays a preset cabin's whole footprint out as
   **blue-tinted ghost tiles** (you can walk through them).
2. **Haul** — workers carry **wood** from trees to the site; each ghost **brightens**
   as its wood arrives.
3. **Raise** — a stocked ghost becomes a real wall/door, **one chunk per turn**. The
   finished cabin is furnished and left **unowned** for the nearest homeless resident
   to claim.

**Anybody can work a blueprint** — a staked site belongs to no one, so several
villagers share the labour (a barn-raising). The `Blueprint`/`ConstructionSite`
components and the haul/raise logic live in
[src/systems.py](../src/systems.py) (`create_construction_site`, `raise_blueprint`,
`NpcAiProcessor._work_blueprints`). Nothing gets sealed into a wall: sites are staked
clear of trees/corpses/saplings, and anything caught under a raised wall is cleared
first.

**You build too.** Open the **Craft** tab (`Tab` menu) to craft `Wall`, `Window`,
and `Door` pieces from `Wood` (walls/windows cost 2, doors 3). From the **Inventory**
tab, select a piece and press a **direction** to lay a blueprint on that tile
(already stocked, bright blue). Then **face the ghost and press Enter** to raise it —
or leave it for a passing builder. Facing an *unstocked* ghost with `Wood` hauls a
piece in to stock it. Each haul/raise **spends a turn**. Seal a room with four walls
and a door and you've built a house. Recipes and the item/tile mapping live in
[src/items.py](../src/items.py).

## The living ecosystem

The map is a **3×3 grid of sections** of the 120×60 island (ringed by open ocean).
You occupy one section at a time; the camera shows just that section, and crossing an
edge moves you to the neighbour ("You cross into a new area."). The other eight
sections keep **fully simulating** every turn — deer graze and drink, predators hunt,
fish roam the sea — so the world lives while you're elsewhere. (How that stays cheap:
[World Simulation](World-Simulation.md).)

- **Deer** (`d`) graze trees when hungry and drink from lakes/rivers when thirsty;
  they're prey for carnivores and for you (walk into one to take it down for
  `Deer Meat`).
- **The forest grows and regrows.** Each day every open outdoor ground tile has a
  small chance to sprout a **seedling** — a tree (`t`→`T`, 0.01%/tile) or, rarer, a
  **berry bush** (`,`→`%`) — and every mature plant has half that chance to die. A
  seedling that survives a full **year** (112 days) matures. Odds live in
  `TreeGrowthProcessor`.
- **Berry bushes** (`%`) are renewable: face a ripe (red) bush and press Enter to
  pick `Berries`; the bare bush regrows a crop **7 days** later.
- **Fish** (`f`) swim the open sea and graze **seaweed** (`"`), the aquatic mirror of
  deer/trees (`FishAiProcessor`).

**Everyone feeds like you.** Villagers (**cooks**) scavenge meat, chop wood, cook at
a stove, then eat — never raw; if no game is reachable they forage from trees.
Predators (**goblins**, **cave rats**) eat raw on the spot. NPCs only pursue food,
water, and homes they can actually **walk to** (`region_of`/`same_region` in the game
map), so nobody strands itself across a river it can't cross.

## Swimming

Water blocks NPCs and line of sight, but *you* can wade in and swim across lakes and
rivers (walls still block). The player moves with `is_passable` (land + water); NPC
pathfinding uses `is_walkable` (land only).

## Status identifiers

A character's tile animates through its own glyph plus an identifier for each active
status, each for a set time, cycling in real time even while you stand still.
Swimming shows your tile (1s) then `~` (0.5s); sleeping shows `Z`; on fire shows a
red `F`. Statuses **stack sequentially**. Durations/identifiers live in
`_STATUS_DISPLAY` / `_STATUS_ORDER` in systems. Because the cycle is real-time, the
bob plays while idle (`_await_action_or_idle` in main polls on a short timeout).

## Reading creatures

Face a creature and press Enter: **friendlies** open a dialogue with a **Status**
option; **wild/hostile** creatures show a read-only **examine** panel (disposition,
hunger, thirst, tiredness, statuses including whether they're asleep).

## Menus

- Startup: Title Screen → Main Menu (`Continue`, `New Game`, `Quit`).
- **Player menu** (`Tab`): tabbed — `Inventory`, `Craft`, `Status`, with `Map`,
  `Journal`, `Skills` stubbed. `Tab` reopens on the last tab and cycles; `I`/`C` jump
  to Inventory/Status (and toggle closed); `Esc` closes.
- **Pause** (`Esc`): `Save Game`, `Options`, `Quit`. Options toggles `Fullscreen` and
  `Show FPS`; changes are written to the working options file.

> **Note:** survival state is not saved yet — the world regenerates from its seed
> each session (only map size, player position, and the seed are persisted). Survival
> item names and their food/water values live in [src/items.py](../src/items.py).
