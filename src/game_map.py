"""The map is plain data too, independent of any renderer.

A tile is just a character for now ('#' wall, '.' floor). Later this can grow
into a Tile dataclass (walkable, transparent, colours) without touching the
render path.
"""
from collections import deque


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
        self._add_default_buildings()

    def _carve_building(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        door: tuple[int, int],
    ) -> None:
        if not self.in_bounds(left, top) or not self.in_bounds(right, bottom):
            return
        if right - left < 2 or bottom - top < 2:
            return

        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                is_edge = x in {left, right} or y in {top, bottom}
                self.tiles[y][x] = self.WALL if is_edge else self.FLOOR

        door_x, door_y = door
        if self.in_bounds(door_x, door_y):
            self.tiles[door_y][door_x] = self.FLOOR

    def _add_default_buildings(self) -> None:
        if self.width < 30 or self.height < 16:
            return

        self._carve_building(left=4, top=3, right=12, bottom=8, door=(8, 8))
        self._carve_building(
            left=self.width - 14,
            top=5,
            right=self.width - 5,
            bottom=11,
            door=(self.width - 10, 11),
        )

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.tiles[y][x] != self.WALL

    def tile_at(self, x: int, y: int) -> str:
        return self.tiles[y][x]

    def neighbors_4(self, x: int, y: int) -> list[tuple[int, int]]:
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(nx, ny) for nx, ny in candidates if self.in_bounds(nx, ny)]

    def line_points(self, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        """Return Bresenham line points from start to end, inclusive."""
        x0, y0 = start
        x1, y1 = end

        points: list[tuple[int, int]] = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points

    def has_line_of_sight(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        if not self.in_bounds(start[0], start[1]) or not self.in_bounds(end[0], end[1]):
            return False

        points = self.line_points(start, end)
        for x, y in points[1:-1]:
            if self.tile_at(x, y) == self.WALL:
                return False
        return True

    def find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked_tiles: set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """Find a shortest 4-way path from start to goal using BFS.

        Returns a list of coordinates excluding start and including goal.
        Returns [] when no path exists.
        """
        if start == goal:
            return []
        if not self.in_bounds(start[0], start[1]) or not self.in_bounds(goal[0], goal[1]):
            return []

        blocked = blocked_tiles or set()
        queue: deque[tuple[int, int]] = deque([start])
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal:
                break

            for nxt in self.neighbors_4(current[0], current[1]):
                if nxt in came_from:
                    continue
                if not self.is_walkable(nxt[0], nxt[1]):
                    continue
                if nxt in blocked and nxt != goal:
                    continue
                came_from[nxt] = current
                queue.append(nxt)

        if goal not in came_from:
            return []

        path: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = goal
        while cur is not None and cur != start:
            path.append(cur)
            cur = came_from[cur]

        path.reverse()
        return path
