# Components

Components are **plain data** attached to entities — no behaviour, no renderer
handles (a `Renderable` stores a glyph and colours, not a renderer object). All are
`@dataclass`es in [src/components.py](../src/components.py). Systems query by
component *presence*, so adding a component to an entity is how you give it a
capability. This page groups them by domain; the file is the source of truth.

## Core

| Component | Fields | Meaning |
|-----------|--------|---------|
| `Position` | `x, y` | Tile-grid coordinates |
| `Renderable` | `glyph, fg=None, bg=None` | The character + optional fg/bg colours to draw |
| `Player` | *(tag)* | The entity input controls |
| `NPC` | *(tag)* | AI-controlled |
| `Name` | `value` | Display name |
| `BlocksMovement` | *(tag)* | Occupies and blocks its tile |
| `Vision` | `radius=8` | Sight radius (FOV + AI awareness) |

## Disposition & combat

| Component | Fields | Meaning |
|-----------|--------|---------|
| `Enemy` | *(tag)* | Hostile to the player (chases on sight) |
| `Friendly` | *(tag)* | Non-hostile (opens dialogue, not examine) |
| `Corpse` | *(tag)* | A dead body (loot for meat) |
| `OnFire` | *(tag)* | On fire — drives a red `F` status identifier (no damage wired yet; becoming a registered [effect](Content-and-Mods.md)) |

## Survival, diet & food

| Component | Fields | Meaning |
|-----------|--------|---------|
| `Needs` | `hunger, thirst, tiredness` + per-turn `*_rate` + `max_value=100` | Rise each turn; 0 sated, `max_value` dire. Per-creature rates live here so different creatures tick differently without touching the processor |
| `Diet` | `kind="herbivore"` | `"herbivore"` grazes trees · `"carnivore"` hunts prey · `"cook"` gathers+cooks before eating |
| `Meat` | `name="Raw Meat"` | The meat item this creature's corpse yields |
| `Deer` | *(tag)* | Wild grazing prey (grazes trees, drinks water) |
| `Fish` | *(tag)* | Roams open sea, grazes seaweed (own AI, water-only) |
| `Seaweed` | `food=3` | Underwater plant fish graze; bites remaining |

## Flora & world features

| Component | Fields | Meaning |
|-----------|--------|---------|
| `Tree` | `wood=3` | Choppable; chops remaining before it falls |
| `Sapling` | `planted_turn=0, kind="tree"` | Young plant; matures after one year into a `Tree` or `BerryBush` |
| `BerryBush` | `has_berries=True, harvested_turn=None` | Ripe bush is pickable; regrows 7 days after harvest |
| `Well` | *(tag)* | Renewable drink source |
| `Stove` | *(tag)* | Wood + raw meat → cooked meat |
| `Bed` | *(tag)* | Sleep-at-home target; stands in for the house that owns it |
| `Chest` | *(tag)* | Storage (pairs with `Inventory`) |
| `Furniture` | `kind="furniture"` | Decorative furnishing (`table`, `wardrobe`, `bookshelf`) |

## Housing & construction

| Component | Fields | Meaning |
|-----------|--------|---------|
| `Home` | `x, y` | The tile a character returns to to sleep |
| `Owned` | `owner` | Marks a bed/house as belonging to an entity |
| `Resident` | *(tag)* | Wants a house: claims an unowned one, else builds |
| `Camp` | *(tag)* | Temporary campsite (broken on waking) |
| `Asleep` | `in_camp=False` | Sleeping; skips turns while tiredness recovers |
| `Blueprint` | `tile="#", stocked=False, site=None` | One ghost tile of a structure; `stocked` once its wood arrives |
| `ConstructionSite` | `pieces{}, interior[], bed` | A staked-out cabin: ghost pieces → real tiles, then furnished |

## People, family & social

| Component | Fields | Meaning |
|-----------|--------|---------|
| `Gender` | `value` | `"male"`/`"female"` (free string; onymancer + reproduction read it) |
| `Age` | `born_turn` | Age is *derived* from the clock, so nobody is ticked each turn |
| `Family` | `surname, spouse, parents[], children[]` | Household ties (kept reciprocal) |
| `Mating` | `last_turn=-10000` | Reproduction cooldown bookkeeping |
| `Pregnant` | `conceived_turn, father` | Carried until birth; `ReproductionProcessor` delivers |
| `Personality` | `traits[], last_social_turn` | Named traits drive sociability + friendship deltas |
| `Relationships` | `scores{}, pending{}` | Sims-like friendship in `[-100,100]`; `pending` gates the `+`/`-` indicator |
| `Dialogue` | `line` | Gibberish spoken line |

## Items & stats

| Component | Fields | Meaning |
|-----------|--------|---------|
| `Inventory` | `items[]` | Item **name strings** (see [Content & Mods](Content-and-Mods.md) / [items.py](../src/items.py)) |
| `Equipment` | `slots{}` | Slot name → equipped item name (or `None`) |
| `Attributes` | `strength … charisma` (all `10`) | Morrowind-style attributes; `dexterity` drives action speed |

## Time & scheduling

| Component | Fields | Meaning |
|-----------|--------|---------|
| `WorldClock` | `turn=0, day_length=24000` | **Singleton.** `turn` advances by an action's *cost* in time units (TU), not by 1. 240 baseline actions per day |
| `Actor` | `next_time, last_acted, energy` | Turn scheduling: whoever has the smallest `next_time` acts next; `energy` drives region-simulated NPCs. See [Action Economy](Action-Economy.md) |

## Adding a component

1. Add a `@dataclass` to [src/components.py](../src/components.py).
2. Attach it at creation (`esper.create_entity(...)`, or a [prefab](Content-and-Mods.md))
   or later with `esper.add_component(ent, MyComponent(...))`.
3. Query it from a system via `esper.get_components(MyComponent, ...)`.

Because behaviour is keyed off component presence, most new content needs **no new
component** — it reuses the ones above (a new predator is just another entity with
`NPC + Enemy + Diet("carnivore") + Meat + Needs + …`). Reach for a new component only
when you need genuinely new *data* the existing systems don't model. See
[Systems](Systems.md) for how processors read these, and
[Content & Mods](Content-and-Mods.md) for the kits that bundle them.
