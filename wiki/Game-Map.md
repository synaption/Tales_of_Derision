# Game Map

The map is plain data, independent of any renderer, in
[`game_map.py`](../game_map.py). A tile is currently just a character.

## `GameMap`

```python
class GameMap:
    WALL = "#"
    FLOOR = "."

    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Bordered room: walls on the edge, floor inside.
        self.tiles = [[...] for y in range(height)]
```

`tiles` is a `height × width` list of rows; index as `tiles[y][x]`.

## Methods

| Method | Returns | Notes |
|--------|---------|-------|
| `in_bounds(x, y)` | `bool` | Inside the grid rectangle |
| `is_walkable(x, y)` | `bool` | In bounds **and** not a wall — used by [movement](Systems.md) |
| `tile_at(x, y)` | `str` | The tile glyph, used by [rendering](Systems.md) |

## Current shape

A single rectangular room: a `#` border with a `.` floor inside. The player
spawns at the centre and is blocked by the border walls.

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
