"""The world's 120x60 "map tile" grid, and a scheduler that lets far-away
regions lag behind the current turn and pay that debt down later -- nearest to
the player first -- instead of every region-aware system simulating the whole
map every turn.

``FishAiProcessor`` pioneered this idea with a private area grid; this module
is the shared, generic version so other expensive systems (chiefly
``NpcAiProcessor``) can use the same scheme.
"""
from __future__ import annotations

from collections.abc import Callable

from game_map import GameMap, LAND_HEIGHT, LAND_WIDTH

RegionId = tuple[int, int]

REGION_W = LAND_WIDTH   # 120
REGION_H = LAND_HEIGHT  # 60


# A map's grid geometry depends only on its (static, never-resized) width and
# height, not its ever-changing tile content -- and region_at() below calls
# region_grid_size() on every single entity of every single region-turn
# advance, so recomputing it each time is pure waste at any real population.
_grid_size_cache: dict[tuple[int, int], tuple[int, int, int, int]] = {}


def region_grid_size(game_map: GameMap) -> tuple[int, int, int, int]:
    """``(area_w, area_h, cols, rows)`` for this map's region grid. A map
    smaller than one region (every test map today) collapses to a single
    region covering the whole map, so region-aware systems behave exactly as
    an unpartitioned global system would."""
    key = (game_map.width, game_map.height)
    cached = _grid_size_cache.get(key)
    if cached is not None:
        return cached
    area_w = max(1, min(REGION_W, game_map.width))
    area_h = max(1, min(REGION_H, game_map.height))
    cols = max(1, game_map.width // area_w)
    rows = max(1, game_map.height // area_h)
    result = (area_w, area_h, cols, rows)
    _grid_size_cache[key] = result
    return result


def region_at(game_map: GameMap, x: int, y: int) -> RegionId:
    """The arithmetic grid cell containing ``(x, y)``. Unrelated to
    ``GameMap.region_of``, which is a *topological* walkability-connectivity
    id -- this is a plain spatial bucket, used only to decide what to
    simulate now versus later."""
    area_w, area_h, cols, rows = region_grid_size(game_map)
    return (
        min(cols - 1, max(0, x // area_w)),
        min(rows - 1, max(0, y // area_h)),
    )


def all_region_ids(game_map: GameMap) -> list[RegionId]:
    _, _, cols, rows = region_grid_size(game_map)
    return [(cx, cy) for cy in range(rows) for cx in range(cols)]


def region_count(game_map: GameMap) -> int:
    _, _, cols, rows = region_grid_size(game_map)
    return cols * rows


def region_bounds(game_map: GameMap, region_id: RegionId) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` -- the half-open tile rectangle a region covers."""
    area_w, area_h, _cols, _rows = region_grid_size(game_map)
    cx, cy = region_id
    return (cx * area_w, cy * area_h, (cx + 1) * area_w, (cy + 1) * area_h)


def in_region_with_margin(
    game_map: GameMap, region_id: RegionId, x: int, y: int, margin: int
) -> bool:
    """True for tiles inside ``region_id``, or within ``margin`` tiles of its
    border (whether that's just inside or just outside it). Used to give a
    region-scoped goal search visibility across a seam, so a creature standing
    near a region boundary doesn't lose sight of a resource one tile into the
    next region -- something a global (unpartitioned) scan never had to worry
    about."""
    x0, y0, x1, y1 = region_bounds(game_map, region_id)
    return x0 - margin <= x < x1 + margin and y0 - margin <= y < y1 + margin


def _chebyshev(a: RegionId, b: RegionId) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class RegionScheduler:
    """Owns each region's "simulated up to turn N" cursor and pays down debt.

    Each registered step is a plain, region-scoped single turn of work; catch-
    up always replays turn N, then N+1, then N+2 ... in strict order -- never
    an analytic shortcut -- so state one step builds (e.g. NPC positions) is
    always consistent for the next step and the next turn. Different regions
    can sit at different turn cursors at the same real moment; that's the
    entire point (true global lockstep would mean simulating everything every
    turn, i.e. no optimization at all).
    """

    def __init__(self, game_map: GameMap, current_turn: int):
        self.game_map = game_map
        self._steps: list[tuple[str, Callable[[RegionId], None]]] = []
        # A freshly built world has no history to be behind on: every region
        # starts "caught up" to the turn it was created at, not zero -- else
        # the first pump would replay turns nothing ever actually lived
        # through (persistence today doesn't carry region state across a
        # save/load; the world regenerates fresh, so this is always correct).
        self.region_turn: dict[RegionId, int] = {
            region_id: current_turn for region_id in all_region_ids(game_map)
        }

    def register(self, name: str, step: Callable[[RegionId], None]) -> None:
        """Add a per-region simulation step, run in registration order."""
        self._steps.append((name, step))

    def region_at(self, x: int, y: int) -> RegionId:
        return region_at(self.game_map, x, y)

    def advance_region(self, region_id: RegionId) -> None:
        """Run every registered step for ``region_id``'s next turn."""
        for _name, step in self._steps:
            step(region_id)
        self.region_turn[region_id] = self.region_turn.get(region_id, 0) + 1

    def next_turn_for(self, region_id: RegionId, observed_turn: int) -> int:
        """The turn number *this* call represents for ``region_id``.

        Normally just ``observed_turn`` (the world clock, already advanced by
        ``TimeProcessor`` before region-aware processors run each real turn).
        Never less than one past this region's own cursor, though -- so a
        caller with no advancing world clock at all (a processor constructed
        directly in a unit test, with no ``TimeProcessor`` in the loop) still
        advances by exactly one turn per call, the same as an unpartitioned
        processor would."""
        return max(observed_turn, self.region_turn.get(region_id, observed_turn) + 1)

    def catch_up_region(self, region_id: RegionId, target_turn: int) -> None:
        """Block until ``region_id`` has been simulated up to ``target_turn``."""
        while self.region_turn.get(region_id, target_turn) < target_turn:
            self.advance_region(region_id)

    def catch_up_all(self, target_turn: int) -> None:
        """Block until every region has been simulated up to ``target_turn``."""
        for region_id in all_region_ids(self.game_map):
            self.catch_up_region(region_id, target_turn)

    def pump_background(
        self,
        budget_seconds: float,
        player_region: RegionId | None,
        target_turn: int,
        wall_clock: Callable[[], float],
    ) -> None:
        """Spend up to ``budget_seconds`` of real time advancing the *nearest*
        lagging region to ``player_region`` by one turn at a time -- never the
        stalest -- until either nothing lags or the budget runs out."""
        deadline = wall_clock() + budget_seconds
        while wall_clock() < deadline:
            lagging = [r for r, t in self.region_turn.items() if t < target_turn]
            if not lagging:
                return
            nearest = (
                lagging[0]
                if player_region is None
                else min(lagging, key=lambda r: _chebyshev(r, player_region))
            )
            self.advance_region(nearest)
