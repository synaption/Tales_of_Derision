# Content & Mods

The content layer lives in [src/content/](../src/content/) and is covered by
[src/tests/test_content.py](../src/tests/test_content.py). Mods are a first-class
goal: anyone should be able to add or change content —
creatures, NPCs, items, effects — **either as readable data or as Python, the same way
the core game does it.** That's the design decision this layer encodes: one registry,
two authoring styles feeding it.

- **Data path** (readable/editable) — tune or add a *variant* of an existing feature by
  editing a declarative definition (a `PrefabDef`/`ItemDef`/`EffectDef`, JSON-droppable).
  No code, no new behaviour — just data the existing [systems](Systems.md) already act
  on.
- **Python path** (new behaviour) — add a genuinely new feature by writing a Python
  module that registers a factory / effect handler. Exactly how the core content is
  written, so a mod is not a second-class citizen.

The core game's own content is just the first "mod": it registers through the same
registry at startup.

## Why this works: behaviour is keyed off components

The existing ECS already does the hard part. A creature is a carnivore because it has
`Diet("carnivore")`, prey because it has `Deer`, a builder because it has `Resident`.
So **most new content needs no new system and no new component** — it's a new
*combination* of existing components. The content layer's job is to make assembling
those combinations declarative and reusable instead of hand-written
`esper.create_entity(...)` calls scattered across `main.py`.

## The pieces (`src/content/`)

### `registry.py` — prefabs
```python
@dataclass
class PrefabDef:
    id: str
    glyph: str
    name: str | None = None
    fg: RGB | None = None
    bg: RGB | None = None
    kits: list = []        # each entry is "kit_name" or ("kit_name", {kwargs})
    components: list = []  # extra components: a class, a factory, or an instance
    tags: tuple = ()

register_prefab(defn, prefab_id=None)  # a PrefabDef (data) OR a factory callable (Python)
build_components(id, x, y, **overrides) -> list      # pure, headless-testable
spawn(id, x, y, **overrides) -> int                  # esper.create_entity(...)
```
`register_prefab` accepts **either** a `PrefabDef` (data path) **or** a factory
`Callable[[int, int, dict], list]` (Python path, needs an explicit `prefab_id`) —
same registry, same `spawn`. A prefab's per-instance state is kept unshared: a
`components` entry that is a **class** or a **factory** is built fresh each spawn, and
an **instance** is deep-copied — so two goblins never alias one `Inventory`.

### `kits.py` — reusable component bundles
The "reusable chunks of code that apply to many entities." A kit returns a plain list
of components a prefab composes from:
```python
predator_kit(meat, dexterity, vision)  # NPC, Enemy, Diet("carnivore"), Meat, Vision, Attributes, Needs, BlocksMovement
person_kit(gender, traits, surname, age)  # Gender, Family, Personality, Relationships, Dialogue, Friendly, Diet("cook"), Resident, Inventory, Equipment, Needs, NPC, BlocksMovement
grazer_kit(...)   feature_kit(...)
```

### `effects.py` — status effects
```python
@dataclass
class EffectDef:
    id: str
    glyph: str               # status-identifier glyph in the tile cycle
    seconds: float           # how long the identifier shows
    label: str               # human-readable name (status/examine screens)
    fg: RGB | None = None     # identifier colour, or None to keep the char's own
    component: type | None = None    # marks the effect (e.g. OnFire)
    detector: Callable | None = None # for effects derived from world state (swimming)
    on_apply / on_tick / on_remove: Callable | None = None   # Python for behaviour

register_effect(EffectDef)
apply_effect(ent, id)   remove_effect(ent, id)   active_effects(game_map, ent, pos)
```
Effects come two ways: **component-marked** (fire, sleep — checked by component) or
**derived** (swimming — a `detector` reads world state). Simple ones are pure data (a
glyph + colour + seconds); behaviour supplies `on_tick`/`on_apply`. `EffectsProcessor`
ticks component effects each turn (a no-op until one declares `on_tick`), and the
[renderer](Renderers.md) reads the glyph/colour for the status-identifier animation —
replacing the old ad-hoc `OnFire` tag + `_STATUS_DISPLAY`/`_STATUS_ORDER` tables. The
`systems` names `active_statuses`/`status_label` are now thin aliases for
`active_effects`/`effect_label`.

### `items.py` — item definitions
```python
@dataclass
class ItemDef:
    name: str
    eat: float | None = None       # hunger restored
    drink: float | None = None     # thirst restored
    equip_slot: str | None = None
    placeable_tile: str | None = None
    craft_cost: int | None = None
    visual: tuple[str, RGB] | None = None
    on_consume: str | None = None  # effect id, for potions etc.
    tags: list[str] = []
```
Absorbs today's `items.py` lookup tables (`EAT_VALUES`, `DRINK_VALUES`,
`PLACEABLE_TILES`, `CRAFTING_RECIPES`) into one definition per item. Items remain
**name strings** inside `Inventory`, so nothing about the item plumbing changes — the
registry just gives each string a definition. The shape-based meat helpers
(`is_raw_meat`/`cook_meat`) stay, since meat is dynamically named per creature.

### `loader.py` — the mod seam
`load_all_content()` imports the core content modules (each self-registers on import)
**and** scans `content/data/*.json` (and later a user `mods/` dir), registering those
through the same `register_prefab`. Called once at startup and in a test fixture.

## Recipes: adding content

**A new creature (data):** add one `PrefabDef` (as in `content/creatures.py`) —
```python
register_prefab(PrefabDef(
    id="dire_rat", glyph="r", name="Dire Rat", fg=(150, 60, 60),
    kits=[("predator", dict(meat="Dire Rat Meat", vision=7, dexterity=18))],
))
```
then `spawn("dire_rat", x, y)`. The same thing as a JSON file under
`content/data/` (the non-coder path):
```json
[{"id": "dire_rat", "glyph": "r", "name": "Dire Rat", "fg": [150, 60, 60],
  "kits": [["predator", {"meat": "Dire Rat Meat", "vision": 7}]]}]
```

**A new item (data):** `register_item(ItemDef(name="Healing Herb", eat=5,
on_consume="regen"))`.

**A new effect (Python behaviour):** `register_effect(EffectDef(id="poison",
component=Poisoned, display=..., on_tick=poison_tick))`.

**A whole new behaviour:** write a system (see
[Systems: Adding a system](Systems.md#adding-a-system)) that acts on a component your
prefab attaches — the Python path, identical to how the core game adds features.

## Testing

Everything here is renderer-agnostic and headless-testable: `build_components(...)`
returns a plain list to assert on (no esper side effects), and `spawn`/`apply_effect`
drive the ECS directly. New tests cover prefab build/spawn, kit composition, effect
apply/tick, item lookups, and the JSON side-loader.
