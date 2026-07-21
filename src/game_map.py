"""The map is plain data too, independent of any renderer.

A tile is just a character for now ('#' wall, '.' floor). Later this can grow
into a Tile dataclass (walkable, transparent, colours) without touching the
render path.
"""
from collections import deque


class GameMap:
    WALL = "#"
    FLOOR = "."
    WATER = "~"
    DOOR = "+"  # passable + transparent gap in a wall; marks a house entrance
    WINDOW = "o"  # blocks movement but is transparent (see-through) like a wall gap

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # ``revision`` bumps whenever a tile changes at runtime (building a wall,
        # a door, ...). Cache holders (the renderer's map surface, house
        # detection, connected-region labels) compare it to know when to rebuild.
        self.revision = 0
        self._regions: dict[tuple[int, int], int] = {}
        self._regions_revision = -1
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
        self._add_water_features()

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
            self.tiles[door_y][door_x] = self.DOOR

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

    def _spawn_safe_zone(self) -> tuple[int, int, int, int]:
        """A rectangle around the map centre kept clear of water so the player
        (who spawns near the centre) never starts stuck in a lake or river.
        Returns (min_x, min_y, max_x, max_y)."""
        cx, cy = self.width // 2, self.height // 2
        return (cx - 6, cy - 4, cx + 6, cy + 4)

    def _carve_lake(self, cx: int, cy: int, rx: int, ry: int) -> None:
        """Flood an ellipse of interior floor into water."""
        sx0, sy0, sx1, sy1 = self._spawn_safe_zone()
        for y in range(max(1, cy - ry), min(self.height - 1, cy + ry + 1)):
            for x in range(max(1, cx - rx), min(self.width - 1, cx + rx + 1)):
                if sx0 <= x <= sx1 and sy0 <= y <= sy1:
                    continue
                nx = (x - cx) / max(1, rx)
                ny = (y - cy) / max(1, ry)
                if nx * nx + ny * ny <= 1.0 and self.tiles[y][x] == self.FLOOR:
                    self.tiles[y][x] = self.WATER

    def _carve_river(self, start_x: int, width: int = 2) -> None:
        """Carve a gently wavering vertical river down the map, skipping the
        central spawn zone so it never seals the player in."""
        sx0, sy0, sx1, sy1 = self._spawn_safe_zone()
        x = start_x
        for y in range(1, self.height - 1):
            # Deterministic gentle meander (no RNG so maps stay reproducible).
            if (y // 3) % 2 == 0:
                x += 1
            else:
                x -= 1
            x = max(2, min(self.width - 3, x))
            for wx in range(x, x + width):
                if sx0 <= wx <= sx1 and sy0 <= y <= sy1:
                    continue
                if self.in_bounds(wx, y) and self.tiles[y][wx] == self.FLOOR:
                    self.tiles[y][wx] = self.WATER

    def _add_water_features(self) -> None:
        if self.width < 24 or self.height < 14:
            return
        self._carve_lake(cx=int(self.width * 0.22), cy=int(self.height * 0.72), rx=6, ry=4)
        self._carve_lake(cx=int(self.width * 0.80), cy=int(self.height * 0.28), rx=7, ry=5)
        self._carve_river(start_x=int(self.width * 0.62))

    def clear_water_around(self, x: int, y: int, radius: int = 1) -> None:
        """Turn any water in a square around (x, y) back into floor. Used as a
        final safety so an entity (the player) is never spawned onto water."""
        for ny in range(y - radius, y + radius + 1):
            for nx in range(x - radius, x + radius + 1):
                if self.in_bounds(nx, ny) and self.tiles[ny][nx] == self.WATER:
                    self.tiles[ny][nx] = self.FLOOR

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        # Walls, water, and windows all block movement; a door is a passable gap.
        # Water/windows stay transparent to sight (has_line_of_sight only blocks
        # on walls).
        return self.in_bounds(x, y) and self.tiles[y][x] not in (self.WALL, self.WATER, self.WINDOW)

    def is_water(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.tiles[y][x] == self.WATER

    def is_passable(self, x: int, y: int) -> bool:
        """Can a *swimming* actor (the player) move here? Everything but walls and
        windows, including water and doors. NPC pathfinding still uses
        ``is_walkable`` (land only), so animals stay ashore while the player can
        wade in to swim."""
        return self.in_bounds(x, y) and self.tiles[y][x] not in (self.WALL, self.WINDOW)

    def tile_at(self, x: int, y: int) -> str:
        return self.tiles[y][x]

    def set_tile(self, x: int, y: int, tile: str) -> bool:
        """Change a tile at runtime (building/clearing). Never overwrites the map
        border. Bumps ``revision`` on a real change so caches rebuild. Returns
        True when a change was applied."""
        if not self.in_bounds(x, y):
            return False
        if x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1:
            return False  # keep the world border intact
        if self.tiles[y][x] == tile:
            return False
        self.tiles[y][x] = tile
        self.revision += 1
        return True

    def _compute_regions(self) -> dict[tuple[int, int], int]:
        """Label every walkable tile with a connected-region id (8-connectivity,
        matching ``find_path``). Tiles separated by water/walls/windows land in
        different regions, so two tiles share a region iff a walking path exists
        between them (ignoring transient entity blockers)."""
        regions: dict[tuple[int, int], int] = {}
        region_id = 0
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in regions or not self.is_walkable(x, y):
                    continue
                region_id += 1
                queue: deque[tuple[int, int]] = deque([(x, y)])
                regions[(x, y)] = region_id
                while queue:
                    cx, cy = queue.popleft()
                    for nx, ny in self.neighbors_8(cx, cy):
                        if (nx, ny) not in regions and self.is_walkable(nx, ny):
                            regions[(nx, ny)] = region_id
                            queue.append((nx, ny))
        return regions

    def region_of(self, x: int, y: int) -> int | None:
        """The connected-region id of a walkable tile, or ``None`` if the tile
        isn't walkable. Regions are cached until the map changes."""
        if self._regions_revision != self.revision:
            self._regions = self._compute_regions()
            self._regions_revision = self.revision
        return self._regions.get((x, y))

    def same_region(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        """True when both tiles are walkable and mutually reachable on foot -- a
        cheap stand-in for ``find_path`` when you only need reachability."""
        ra = self.region_of(a[0], a[1])
        return ra is not None and ra == self.region_of(b[0], b[1])

    def find_enclosed_rooms(self, max_size: int = 400) -> list[frozenset[tuple[int, int]]]:
        """Return the interiors of enclosed rooms: floor regions sealed off by
        walls/windows whose only openings are doors.

        A region qualifies as a house interior when it (a) never touches the map
        border, (b) is bounded by wall/window/door tiles, and (c) is reachable
        from outside only through at least one door. The flood-fill spreads over
        floor tiles only, so walls, windows, water, and doors all act as
        boundaries -- a doorway stops the fill, which is what keeps a house's
        inside from leaking out into the open map.
        """
        rooms: list[frozenset[tuple[int, int]]] = []
        visited: set[tuple[int, int]] = set()

        for sy in range(self.height):
            for sx in range(self.width):
                if (sx, sy) in visited or self.tiles[sy][sx] != self.FLOOR:
                    continue

                region: list[tuple[int, int]] = []
                touches_border = False
                has_door = False
                queue: deque[tuple[int, int]] = deque([(sx, sy)])
                visited.add((sx, sy))

                while queue:
                    cx, cy = queue.popleft()
                    region.append((cx, cy))
                    if cx == 0 or cy == 0 or cx == self.width - 1 or cy == self.height - 1:
                        touches_border = True
                    for nx, ny in self.neighbors_4(cx, cy):
                        neighbor_tile = self.tiles[ny][nx]
                        if neighbor_tile == self.DOOR:
                            has_door = True
                            continue
                        if neighbor_tile != self.FLOOR:
                            continue
                        if (nx, ny) in visited:
                            continue
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                    if len(region) > max_size:
                        break

                if not touches_border and has_door and len(region) <= max_size:
                    rooms.append(frozenset(region))

        return rooms

    def neighbors_4(self, x: int, y: int) -> list[tuple[int, int]]:
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(nx, ny) for nx, ny in candidates if self.in_bounds(nx, ny)]

    def neighbors_8(self, x: int, y: int) -> list[tuple[int, int]]:
        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
            (x + 1, y + 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x - 1, y - 1),
        ]
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
        """Find a shortest 8-way path from start to goal using BFS.

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

            for nxt in self.neighbors_8(current[0], current[1]):
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
