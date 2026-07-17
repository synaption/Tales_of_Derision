# Systems

Systems (esper calls them **processors**) hold all behaviour. Each subclasses
`esper.Processor` and implements `process(*args)`, which receives whatever was
passed to `esper.process(...)`. Defined in [`systems.py`](../systems.py).

## `MovementProcessor` (priority 1)

Applies a movement action to every player-controlled entity.

```python
class MovementProcessor(esper.Processor):
    def __init__(self, game_map):
        self.game_map = game_map

    def process(self, action=None):
        delta = _ACTION_DELTAS.get(action)   # {"move_up": (0,-1), ...}
        if delta is None:
            return
        dx, dy = delta
        for _ent, (pos, _player) in esper.get_components(Position, Player):
            nx, ny = pos.x + dx, pos.y + dy
            if self.game_map.is_walkable(nx, ny):
                pos.x, pos.y = nx, ny
```

- Only entities with **both** `Position` and `Player` move, so it generalises to
  co-op / multiple controlled entities for free.
- Movement is gated by [`GameMap.is_walkable`](Game-Map.md) — walls block.
- Unknown actions (e.g. the render-only frame) are ignored.

## `RenderProcessor` (priority 0)

Draws the world every turn.

```python
class RenderProcessor(esper.Processor):
    def __init__(self, renderer, game_map):
        self.renderer = renderer
        self.game_map = game_map

    def process(self, action=None):
        r = self.renderer
        r.clear()
        # 1. map tiles
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                r.draw_glyph(x, y, self.game_map.tile_at(x, y))
        # 2. entities on top
        for _ent, (pos, rend) in esper.get_components(Position, Renderable):
            r.draw_glyph(pos.x, pos.y, rend.glyph)
        # 3. UI
        r.draw_text(0, self.game_map.height, "move: arrows/hjkl/wasd   menu: esc")
        r.present()
```

It talks only to the [`Renderer`](Renderers.md) interface, never to curses.

## Priority and ordering

`esper.process(action)` calls each processor's `process(action)` in **descending
priority**. Registration in [`main.py`](../main.py):

```python
esper.add_processor(MovementProcessor(game_map), priority=1)  # runs first
esper.add_processor(RenderProcessor(renderer, game_map), priority=0)  # then draw
```

Move-before-draw means a keypress is reflected in the same frame.

## Adding a system

1. Subclass `esper.Processor` in [`systems.py`](../systems.py) and implement
   `process(self, action=None)`.
2. Register it in [`main.py`](../main.py) with a `priority` that places it
   correctly relative to movement and rendering.
3. Query the components it needs with `esper.get_components(...)`.

Example — a system that opens a door the player steps on would run at a priority
between movement (1) and render (0), e.g. `priority=1` alongside movement or a
fractional/interleaved value, reading `Position, Player` and mutating the
[map](Game-Map.md).

See [Architecture](Architecture.md) for the full turn loop.
