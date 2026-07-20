# Game Map

The map is plain data, independent of any renderer, in
[`game_map.py`](../game_map.py). A tile is currently just a character.

## `GameMap`

```python
class GameMap:
    WALL = "#"
    FLOOR = "."
    WATER = "~"

    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Bordered room: walls on the edge, floor inside.
        self.tiles = [[...] for y in range(height)]
        self._add_default_buildings()
        self._add_water_features()   # lakes + a meandering river
```

`tiles` is a `height × width` list of rows; index as `tiles[y][x]`.

## Methods

| Method | Returns | Notes |
|--------|---------|-------|
| `in_bounds(x, y)` | `bool` | Inside the grid rectangle |
| `is_walkable(x, y)` | `bool` | In bounds and **not** a wall or water — NPC pathfinding (land only) |
| `is_passable(x, y)` | `bool` | In bounds and **not** a wall (water allowed) — the player can **swim** |
| `is_water(x, y)` | `bool` | Tile is water (drink target for the survival AI) |
| `tile_at(x, y)` | `str` | The tile glyph, used by [rendering](Systems.md) |
| `clear_water_around(x, y, r)` | — | Safety: turn water back to floor near a spawn |

## Current shape

A `120 × 60` (3×3 of the original room) bordered map with a `#` wall edge and `.`
floor inside. Two elliptical **lakes** and a gently meandering vertical **river**
are carved as `~` water; a protected zone around the centre keeps the player's
spawn clear. Water **blocks movement but not line of sight** — the renderer draws
it with the Hexany `water` autotiles (see [Systems](Systems.md)).

## Extending later

The design intentionally leaves room to grow without touching the render path:

- **Tile as a type.** Replace the bare character with a `Tile` dataclass carrying
  `walkable`, `transparent`, and colours; keep `tile_at` returning a glyph for
  renderers and add `is_walkable`/`is_transparent` lookups for systems.
- **Procedural generation.** Swap the constructor's bordered-room fill for a
  dungeon generator producing the same `tiles` structure — nothing downstream
  changes.
- **Field of view.** Add a `transparent` flag and a visibility pass; the
  [RenderProcessor](Systems.md) would consult it before drawing a tile.

See [Roadmap](Roadmap.md) for the planned order.
