"""Uniform-grid spatial hashes, for "what is near me" without the N squared.

Two of them, because the two questions have different shapes:

* :class:`SpatialHash` is rebuilt every frame from moving points -- the ants,
  the allies, the player. Target selection and projectile hits both ask it.
* :class:`StaticGrid` is built once from the city's boxes and then only read.
  Collision asks it, sixty times a second, for every moving body.

A city block is about 20m and a bug is about 2m, so a 12m cell keeps both
queries down to a handful of buckets. Neither structure allocates during a
query beyond the result list, which matters when the answer is "nothing" a
few thousand times a second.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Iterator


class SpatialHash:
    """A rebuilt-every-frame bucket grid over 2D points.

    Only X and Y are hashed. Altitude is not: a Wing Diver 60m up is still
    "at" the intersection she is above, and every consumer of this class
    wants that -- ants track the ground position of a flyer and only fail to
    reach her, which is the correct behaviour.
    """

    __slots__ = ("cell", "_buckets")

    def __init__(self, cell: float = 12.0) -> None:
        self.cell = cell
        self._buckets: dict[tuple[int, int], list[tuple[int, float, float]]] = defaultdict(list)

    def clear(self) -> None:
        self._buckets.clear()

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.cell), int(y // self.cell))

    def insert(self, ident: int, x: float, y: float) -> None:
        self._buckets[self._key(x, y)].append((ident, x, y))

    def query_radius(self, x: float, y: float, radius: float) -> Iterator[tuple[int, float]]:
        """Yield ``(ident, distance_squared)`` for every point within ``radius``.

        Unordered. Callers that want the nearest should take the min rather
        than sorting -- they almost always only want one.
        """
        c = self.cell
        r2 = radius * radius
        cx0, cy0 = int((x - radius) // c), int((y - radius) // c)
        cx1, cy1 = int((x + radius) // c), int((y + radius) // c)
        buckets = self._buckets
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                bucket = buckets.get((cx, cy))
                if not bucket:
                    continue
                for ident, px, py in bucket:
                    dx, dy = px - x, py - y
                    d2 = dx * dx + dy * dy
                    if d2 <= r2:
                        yield ident, d2

    def nearest(
        self,
        x: float,
        y: float,
        radius: float,
        accept: Iterable[int] | None = None,
    ) -> tuple[int, float]:
        """Closest point within ``radius``, or ``(-1, inf)``.

        ``accept``, if given, is a membership test applied to the identifier
        -- pass a set of enemy entities and the search filters as it goes
        rather than after.
        """
        best, best_d2 = -1, float("inf")
        for ident, d2 in self.query_radius(x, y, radius):
            if d2 < best_d2 and (accept is None or ident in accept):
                best, best_d2 = ident, d2
        return best, best_d2


class StaticGrid:
    """Build-once index of axis-aligned boxes, keyed by the cells they cover.

    Boxes are stored as ``(ident, x0, y0, x1, y1, top)``. They all start at
    ``z = 0``; the city has no overhangs, and collision is much cheaper for
    it.
    """

    __slots__ = ("cell", "_buckets", "boxes")

    def __init__(self, cell: float = 12.0) -> None:
        self.cell = cell
        self._buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.boxes: list[tuple[int, float, float, float, float, float]] = []

    def add_box(self, ident: int, x0: float, y0: float, x1: float, y1: float, top: float) -> None:
        index = len(self.boxes)
        self.boxes.append((ident, x0, y0, x1, y1, top))
        c = self.cell
        for cx in range(int(x0 // c), int(x1 // c) + 1):
            for cy in range(int(y0 // c), int(y1 // c) + 1):
                self._buckets[(cx, cy)].append(index)

    def query_aabb(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> Iterator[tuple[int, float, float, float, float, float]]:
        """Yield every stored box whose cells overlap the query rectangle.

        Cell overlap is a superset of box overlap, so callers still have to
        test properly -- they were going to anyway, to find the penetration
        depth.
        """
        c = self.cell
        seen: set[int] = set()
        boxes = self.boxes
        for cx in range(int(x0 // c), int(x1 // c) + 1):
            for cy in range(int(y0 // c), int(y1 // c) + 1):
                for index in self._buckets.get((cx, cy), ()):
                    if index not in seen:
                        seen.add(index)
                        yield boxes[index]

    def height_at(self, x: float, y: float) -> float:
        """Tallest roof directly over ``(x, y)``, or 0 for open street."""
        best = 0.0
        for _ident, x0, y0, x1, y1, top in self.query_aabb(x, y, x, y):
            if x0 <= x <= x1 and y0 <= y <= y1 and top > best:
                best = top
        return best
