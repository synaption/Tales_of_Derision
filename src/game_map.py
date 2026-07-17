"""The map is plain data too, independent of any renderer.

A tile is just a character for now ('#' wall, '.' floor). Later this can grow
into a Tile dataclass (walkable, transparent, colours) without touching the
render path.
"""


class GameMap:
    WALL = "#"
    FLOOR = "."

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Bordered room: walls around the edge, floor inside.
        self.tiles = [
            [
                self.WALL
                if x == 0 or y == 0 or x == width - 1 or y == height - 1
                else self.FLOOR
                for x in range(width)
            ]
            for y in range(height)
        ]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.tiles[y][x] != self.WALL

    def tile_at(self, x: int, y: int) -> str:
        return self.tiles[y][x]
