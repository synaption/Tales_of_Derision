"""Verticality for a tile grid: hills, cliffs and stairs, as plain data.

No pygame, no OpenGL and no juice registry, for the same reason `juicefx.py`
has none of them: the interesting questions here -- *can you walk from this
tile to that one*, *how high is the ground under this pixel*, *which way is
downhill* -- are all answerable with no display, so they are all testable with
no display.

The model is deliberately the smallest one that reads as verticality:

* every tile has an integer **level**. Tiles at the same level are one
  continuous floor and you walk between them without noticing;
* the boundary between two levels is a **cliff**. You cannot walk up or down
  it, and it is drawn as a face rather than as a line, which is the whole
  reason a level is worth having at all;
* a **stair** is a single tile that ramps from its low neighbour to its high
  one, and is the only place a level change is legal.

That last point is what makes this a *design* rather than a height map. A
smooth height field would let you drift up a hill without ever deciding to,
and the grid has no way to draw the difference between a gentle slope and a
step. One tile, one level, one named way up.

Height is measured in **levels**, never in pixels. How tall a level looks is a
slider on the bench, and a model that knows the answer to that cannot be
re-tuned without being rewritten.

## The edge rule

Passability is decided at the *edge* between two tiles rather than by their
levels, which is what makes a staircase a staircase instead of a ramp you can
walk onto sideways:

    a plain tile at level L  ...  every edge is at height L
    a stair with base L, uphill d
        the edge facing d      ...  L + 1     (meets the high ground)
        the edge facing -d     ...  L         (meets the low ground)
        the two side edges     ...  L + 0.5   (meets nothing)

A step is legal when the two tiles agree about the height of the edge they
share. Nothing else needs saying: the side of a staircase is impassable
because half-levels never match, and a cliff is impassable because L and L+1
never match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Tile = Tuple[int, int]
Step = Tuple[int, int]

#: The characters a layout is written in. A digit is a level; an arrow is a
#: stair pointing *uphill*. The arrow is the direction you would be walking if
#: you were climbing it, which is the way round that makes a layout readable
#: as a picture of the room.
STAIR_CHARS: Dict[str, Step] = {
    ">": (1, 0),
    "<": (-1, 0),
    "v": (0, 1),
    "^": (0, -1),
}

#: A level is packed into a byte for the shader as `level * PACK_STRIDE`, so
#: seven is as high as a hill can get. Nothing here wants more -- three levels
#: of a 32px tile is already most of the screen -- and the ceiling is worth
#: having because it makes the texture encoding exact rather than approximate.
MAX_LEVEL = 7
PACK_STRIDE = 32

#: Which packed code each uphill direction gets. Zero means "not a stair", so
#: the codes start at one.
STAIR_CODES: Dict[Step, int] = {(1, 0): 1, (-1, 0): 2, (0, 1): 3, (0, -1): 4}


class TerrainError(ValueError):
    """A layout that does not describe a room anyone could walk around."""


@dataclass
class Terrain:
    """A height for every tile, and the rules that follow from it.

    Composed into the sim rather than inherited by it: `World` holds one (or
    holds `None`, which is a flat arena and the old behaviour exactly), asks it
    whether a step is legal, and otherwise never mentions it. The renderer asks
    the same object how high the ground is under a sprite. Neither one owns it.

    `enabled` is a live switch rather than a rebuild because the bench's whole
    method is turning a thing off and looking at the difference. Switching it
    off cannot strand anybody: a level never blocks a *tile*, only the edge
    between two, so the arena a flat bench sees is the arena it always had.
    """

    #: `levels[y][x]`. A stair stores the level of its *low* side; the ramp up
    #: to `level + 1` happens across the tile.
    levels: List[List[int]]
    #: (x, y) -> uphill direction, for the tiles that are stairs.
    stairs: Dict[Tile, Step] = field(default_factory=dict)
    #: Tiles that are masonry. Held because "which way is downhill" has to know
    #: not to answer "into that wall", and because a wall is never a hill.
    walls: Set[Tile] = field(default_factory=set)
    #: World pixels per tile. The one unit conversion this file knows about.
    tile: float = 32.0
    enabled: bool = True

    # -- shape --------------------------------------------------------------
    @property
    def width(self) -> int:
        return len(self.levels[0]) if self.levels else 0

    @property
    def height(self) -> int:
        return len(self.levels)

    @property
    def max_level(self) -> int:
        """The highest ground in the room, counting the top of a stair.

        A true ceiling rather than a fair guess, because the renderer divides
        by it: the floor shader only looks a fixed number of tile rows ahead
        for the surface visible at a fragment, so how tall a level is allowed
        to get depends on how many levels there are.
        """
        top = max((max(row) for row in self.levels), default=0)
        for (tx, ty) in self.stairs:
            top = max(top, self.levels[ty][tx] + 1)
        return top

    # -- queries ------------------------------------------------------------
    def level(self, tx: int, ty: int) -> int:
        """The level of a tile. Off the map is level zero, which matches the
        border being solid wall: nothing can stand there to disagree."""
        if not (0 <= tx < self.width and 0 <= ty < self.height):
            return 0
        return self.levels[ty][tx]

    def edge_height(self, tx: int, ty: int, dx: int, dy: int) -> float:
        """Height of this tile's ground at the edge facing `(dx, dy)`.

        See the module docstring: this one function is the whole passability
        rule, and every other movement question is two calls to it.
        """
        base = float(self.level(tx, ty))
        uphill = self.stairs.get((tx, ty))
        if uphill is None:
            return base
        if (dx, dy) == uphill:
            return base + 1.0
        if (dx, dy) == (-uphill[0], -uphill[1]):
            return base
        return base + 0.5              # the side of a staircase meets nothing

    def passable(self, ax: int, ay: int, bx: int, by: int) -> bool:
        """Whether a step between two adjacent tiles is legal on height alone.

        Says nothing about walls or about who is standing where -- those are
        the sim's questions and it already has answers for them. Switched off,
        every step is legal, which is a flat arena.
        """
        if not self.enabled:
            return True
        dx, dy = bx - ax, by - ay
        return abs(self.edge_height(ax, ay, dx, dy)
                   - self.edge_height(bx, by, -dx, -dy)) < 1e-6

    def lift_at(self, wx: float, wy: float) -> float:
        """Ground height under a world pixel, in levels. Continuous.

        Flat within a tile and flat across the join between two tiles of the
        same level, so the room is made of terraces rather than of dunes -- and
        then linear across a stair, which is what makes walking up one a smooth
        rise instead of a jump at the tile boundary. The renderer and the floor
        shader both implement exactly this, and a test holds them together.
        """
        if not self.enabled:
            return 0.0
        tx, ty = int(wx // self.tile), int(wy // self.tile)
        base = float(self.level(tx, ty))
        uphill = self.stairs.get((tx, ty))
        if uphill is None:
            return base
        fx = wx / self.tile - tx
        fy = wy / self.tile - ty
        along = {(1, 0): fx, (-1, 0): 1.0 - fx,
                 (0, 1): fy, (0, -1): 1.0 - fy}[uphill]
        return base + min(1.0, max(0.0, along))

    def downhill(self, wx: float, wy: float, probe: float = 0.0):
        """Which way the ground falls away under a world pixel, in levels.

        A one-sided gradient on purpose: only the neighbours that are *lower*
        than here contribute. A two-sided one would push a body lying at the
        foot of a cliff away from it, because the cliff is a huge gradient and
        the body is not on it -- that reads as the rock repelling corpses.
        Sampling only downwards means a body on a ledge slides off and a body
        under one lies still.

        Walls are skipped rather than treated as level zero, or every plateau
        that touches masonry would tip bodies into it.
        """
        if not self.enabled:
            return 0.0, 0.0
        probe = probe or self.tile * 0.5
        here = self.lift_at(wx, wy)
        gx = gy = 0.0
        for ox, oy in ((probe, 0.0), (-probe, 0.0), (0.0, probe), (0.0, -probe)):
            px, py = wx + ox, wy + oy
            if (int(px // self.tile), int(py // self.tile)) in self.walls:
                continue
            drop = here - self.lift_at(px, py)
            if drop > 0.0:
                gx += ox / probe * drop
                gy += oy / probe * drop
        return gx, gy

    def step_up(self, wx: float, wy: float, tx: int, ty: int) -> float:
        """How far a thing at (wx, wy) would have to climb to enter a tile.

        In levels, and never negative: dropping *into* a tile is free, which is
        the difference between a cliff and a wall. Used by the corpse physics,
        which is not on the grid and so cannot ask the edge rule -- a body
        sliding across the floor is at a pixel, not on a tile.
        """
        if not self.enabled or (tx, ty) in self.stairs:
            # A stair is never a barrier: it meets the low ground at its foot,
            # and something that arrives at it from the high side is walking
            # down, which costs nothing.
            return 0.0
        return max(0.0, float(self.level(tx, ty)) - self.lift_at(wx, wy))

    # -- the graphics card --------------------------------------------------
    def pack(self) -> bytes:
        """Two bytes a tile: the level, and the stair code. Row-major.

        Uploaded as a two-channel 8-bit texture and read back in the floor
        shader with `floor(v * 255 / 32 + 0.5)`, which is exact for every value
        either channel can hold. A float texture would be the obvious thing and
        would cost four times the bandwidth to carry three bits of information.
        """
        out = bytearray(self.width * self.height * 2)
        for ty in range(self.height):
            for tx in range(self.width):
                i = (ty * self.width + tx) * 2
                out[i] = min(MAX_LEVEL, self.levels[ty][tx]) * PACK_STRIDE
                uphill = self.stairs.get((tx, ty))
                out[i + 1] = STAIR_CODES[uphill] * PACK_STRIDE if uphill else 0
        return bytes(out)

    # -- construction -------------------------------------------------------
    @classmethod
    def parse(cls, rows: Iterable[str], grid: Sequence[Sequence[int]],
              tile: float = 32.0):
        """Build a terrain from a picture of it, checking that it makes sense.

        The layout is one character a tile -- see `STAIR_CHARS` -- and the
        checks are the point of doing it this way. A hand-drawn arena is the
        right way to author this (a generated one cannot promise the room is
        still walkable), and a hand-drawn arena is also exactly the kind of
        thing that ends up with a staircase leading into a wall. Every mistake
        that can be caught by reading the picture is caught here, at import.
        """
        levels: List[List[int]] = []
        stairs: Dict[Tile, Step] = {}
        walls = _wall_set(grid)
        rows = list(rows)
        if len(rows) != len(grid) or any(len(r) != len(grid[0]) for r in rows):
            raise TerrainError(
                f"layout is {len(rows)}x{len(rows[0]) if rows else 0}, "
                f"the map is {len(grid)}x{len(grid[0])}")

        for ty, row in enumerate(rows):
            out = []
            for tx, ch in enumerate(row):
                if ch in (".", "0"):
                    out.append(0)
                elif ch.isdigit():
                    out.append(int(ch))
                elif ch in STAIR_CHARS:
                    stairs[(tx, ty)] = STAIR_CHARS[ch]
                    out.append(0)             # filled in below, from its feet
                else:
                    raise TerrainError(f"unknown terrain character {ch!r} "
                                       f"at ({tx}, {ty})")
                if out[-1] > MAX_LEVEL:
                    raise TerrainError(f"level {out[-1]} at ({tx}, {ty}) is "
                                       f"above the {MAX_LEVEL} a byte holds")
            levels.append(out)

        terrain = cls(levels=levels, stairs=stairs, walls=walls, tile=tile)
        # A stair's base is not written down: it is whatever its low neighbour
        # stands at. Deriving it means a layout cannot disagree with itself,
        # which is a whole class of bug that never has to be looked for.
        for (tx, ty), (dx, dy) in stairs.items():
            low = terrain.level(tx - dx, ty - dy)
            high = terrain.level(tx + dx, ty + dy)
            if (tx, ty) in walls:
                raise TerrainError(f"the stair at ({tx}, {ty}) is inside a wall")
            if high != low + 1:
                raise TerrainError(
                    f"the stair at ({tx}, {ty}) climbs from {low} to {high}; "
                    f"a stair is exactly one level")
            levels[ty][tx] = low

        for ty in range(terrain.height):
            for tx in range(terrain.width):
                if levels[ty][tx] and (tx, ty) in walls:
                    raise TerrainError(f"({tx}, {ty}) is raised ground and a "
                                       f"wall at the same time")
        return terrain

    # -- reachability -------------------------------------------------------
    def reachable(self, grid: Sequence[Sequence[int]], start: Tile) -> Set[Tile]:
        """Every floor tile you can walk to from `start`, obeying the heights.

        Here rather than in a test because it is the question the *layout* has
        to answer, and a room with an unreachable corner is a bug in the room
        rather than in the code that reads it. `build_hills` runs it before it
        hands anything back.
        """
        seen = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in seen or (nx, ny) in self.walls:
                    continue
                if not self.passable(x, y, nx, ny):
                    continue
                seen.add((nx, ny))
                stack.append((nx, ny))
        return seen


def _wall_set(grid: Sequence[Sequence[int]]) -> Set[Tile]:
    return {(x, y) for y, row in enumerate(grid)
            for x, cell in enumerate(row) if cell}


# ---------------------------------------------------------------------------
# The arena's own hills
# ---------------------------------------------------------------------------
#
# Hand-drawn, for the same reason `rogue_juice.build_map` hand-places its
# pillars: this is the room the bench is looked at in, and it has a job. Two
# raised areas, placed where there was open floor to spare, each reachable two
# ways so that losing a route to a corpse pile is never how you get stuck; and
# one second-level shelf in the middle of the eastern one, because a single
# step up reads as a kerb and it takes two before anybody says "hill".
#
# A generator would be the obvious alternative and is the wrong tool. It cannot
# promise the room stays walkable, it cannot promise a stair does not end in a
# pillar, and every run of the bench would be a different arena -- which is the
# one thing a bench must never be, because two runs have to be comparable.

#: One character a tile, 40x20, matching `rogue_juice.build_map`.
#: `.` floor, digits raised ground, `< > ^ v` a stair pointing uphill.
ARENA = [
    "........................................",   # 0  wall
    "........................................",   # 1
    "........................................",   # 2
    "........................................",   # 3
    "........................................",   # 4
    "........................................",   # 5
    "........................................",   # 6
    "...111111...............................",   # 7
    "...111111...............................",   # 8
    "...111111<..............................",   # 9
    "...111111...............................",   # 10
    "...111111...........v...................",   # 11
    ".....^.............111111...............",   # 12
    "...................1122<1...............",   # 13
    "...................112211<..............",   # 14
    "...................111111...............",   # 15
    "........................................",   # 16
    "........................................",   # 17
    "........................................",   # 18
    "........................................",   # 19 wall
]


def build_hills(grid: Sequence[Sequence[int]], tile: float = 32.0,
                start: Tile = None) -> Terrain:
    """The arena's terrain, checked against the map it is laid over.

    Raises rather than quietly shipping a broken room: a stair into a pillar or
    a plateau nobody can reach is a mistake in `ARENA`, and the only moment it
    is cheap to find is the moment the layout is read.
    """
    terrain = Terrain.parse(ARENA, grid, tile=tile)
    if start is None:
        start = (len(grid[0]) // 2, len(grid) // 2)
    floor = {(x, y) for y, row in enumerate(grid)
             for x, cell in enumerate(row) if not cell}
    stranded = floor - terrain.reachable(grid, start)
    if stranded:
        raise TerrainError(
            f"{len(stranded)} floor tiles cannot be walked to from {start}, "
            f"nearest {min(stranded, key=lambda t: abs(t[0] - start[0]) + abs(t[1] - start[1]))}")
    return terrain
