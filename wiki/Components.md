# Components

Components are **plain data** attached to entities. They contain no behaviour and
no renderer handles — a `Renderable` stores a glyph character, not a renderer
object. Defined in [src/components.py](../src/components.py) as `@dataclass`es.

## Current components

### `Position`
```python
@dataclass
class Position:
    x: int
    y: int
```
Where the entity sits on the map grid, in tile coordinates.

### `Renderable`
```python
@dataclass
class Renderable:
    glyph: str
```
The single character used to draw the entity. Kept backend-neutral so any
renderer can decide how to turn a glyph into pixels. Colour/fg/bg would be added
here later.

### `Player`
```python
@dataclass
class Player:
    """Tag component: marks the entity the input controls."""
```
A **tag** — a component with no fields, used purely to mark entities. Systems
query `Position, Player` to act on "the things the human drives".

### Survival components

```python
@dataclass
class Needs:
    hunger: float = 0.0
    thirst: float = 0.0
    hunger_rate: float = 1.0
    thirst_rate: float = 1.4
    max_value: float = 100.0
```
Carried by anything that gets hungry/thirsty (currently the player). Values rise
each real turn via the `NeedsProcessor` (0 = sated, `max_value` = dire) and are
lowered by eating/drinking. See [Systems](Systems.md).

Every NPC also carries `Needs`, so the world is full of creatures that get
hungry and thirsty; only the player's needs are surfaced as log warnings.

```python
@dataclass
class Deer:              # tag: wild grazing prey

@dataclass
class Diet:
    kind: str = "herbivore"   # "herbivore" grazes trees; "carnivore" hunts prey
```
`Deer` are `NPC`s that graze trees and drink from lakes/rivers, and are hunted by
carnivores (and the player). `Diet` drives the needs-based AI in
[Systems](Systems.md): rats/goblins are carnivores, deer are herbivores.

```python
@dataclass
class Meat:
    name: str = "Raw Meat"   # e.g. Meat("Rat Meat"), Meat("Goblin Meat"), Meat("Deer Meat")
```
Set on creatures so their corpse yields creature-specific meat when butchered
(rats drop `Rat Meat`, goblins drop `Goblin Meat`); corpses fall back to generic
`Raw Meat` when a creature has no `Meat`. Meat is recognised by shape in
[src/items.py](../src/items.py) (`is_raw_meat` / `is_cooked_meat` / `cook_meat`),
so any `"... Meat"` cooks into `"Cooked ... Meat"`.

```python
@dataclass
class Tree:
    wood: int = 3      # chops remaining before it falls

@dataclass
class Well:            # tag: renewable water source
    ...

@dataclass
class Stove:           # tag: turns Wood + Raw Meat into a Cooked Meat
    ...
```
These are placed by `_spawn_environment_features` in
[src/main.py](../src/main.py) and interacted with by facing them and pressing the
`menu_select` key (Enter), the same targeting used for corpses/NPCs.

Survival **items** are plain strings in an `Inventory` (`"Wood"`, `"Raw Meat"`,
`"Cooked Meat"`, `"Waterskin"`, …). Their canonical names and food/drink values
live in [src/items.py](../src/items.py), shared by the turn loop and the ECS
processors so they never drift apart. A killed enemy's corpse always carries
`"Raw Meat"` to butcher.

## Adding a component

1. Add a `@dataclass` to [src/components.py](../src/components.py).
2. Attach it when creating an entity, or later with
   `esper.add_component(ent, MyComponent(...))`.
3. Query it from a system via `esper.get_components(MyComponent, ...)`.

Example — give something hit points:

```python
@dataclass
class Health:
    hp: int
    max_hp: int
```

```python
esper.create_entity(Position(5, 5), Renderable("g"), Health(10, 10))
```

See [Systems](Systems.md) for how processors read components, and
[Architecture](Architecture.md) for the ECS model.
