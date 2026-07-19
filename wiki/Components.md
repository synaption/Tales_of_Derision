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
