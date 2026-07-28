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
from dataclasses import dataclass

from game_map import GameMap, LAND_HEIGHT, LAND_WIDTH

RegionId = tuple[int, int]

# "As far as you like" for ``jump_limit`` -- a region with nothing in it can skip
# straight to the target turn, so the only real bound is the debt itself.
_UNBOUNDED = 1 << 30


@dataclass(frozen=True)
class _Step:
    """One registered per-region simulation step and its optional fast paths.
    See ``RegionScheduler.register``."""
    name: str
    step: Callable[[RegionId], None]
    bulk: Callable[[RegionId, int], None] | None = None
    idle_turns: Callable[[RegionId], int] | None = None

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

    Each registered step is a plain, region-scoped single turn of work, run in
    order, so state one step builds (e.g. NPC positions) is consistent for the
    next step and the next turn. Different regions can sit at different turn
    cursors at the same real moment; that's the entire point (true global
    lockstep would mean simulating everything every turn, i.e. no optimization
    at all).

    **Analytic shortcuts are allowed.** A step is free to work out where N turns
    of something end up rather than living them one at a time -- that is how
    compacted activities work (see the compaction notes in ``ai``). The rule a
    shortcut must obey is not "replay every turn", it is **batch independence**:

        the result must not depend on how the turns were divided up.

    That is what keeps the world deterministic, because how many turns a region
    advances in one go is *not*. ``pump_background`` spends a real-time budget
    and live catch-up is capped per input frame, so the same seed and the same
    player inputs will batch differently on a faster machine or a busier one. A
    shortcut whose answer changes with the batching would make the world change
    with the frame rate; one that doesn't is indistinguishable from replaying.

    Two ways that bites in practice:

    * **Length the activity chooses, not the batch it landed in.** Settle a
      whole night's sleep because the sleeper needs N turns of it -- never
      "however many turns the scheduler happens to be handing me right now".
      Float arithmetic makes the difference real as well as theoretical:
      ``x + rate * N`` is not bit-identical to two halves added in turn.
    * **Randomness must be drawn as a function of (state, N)**, not once per
      call, or splitting a shortcut in two changes the RNG stream.
    """

    def __init__(self, game_map: GameMap, current_turn: int):
        self.game_map = game_map
        self._steps: list[_Step] = []
        # A freshly built world has no history to be behind on: every region
        # starts "caught up" to the turn it was created at, not zero -- else
        # the first pump would replay turns nothing ever actually lived
        # through (persistence today doesn't carry region state across a
        # save/load; the world regenerates fresh, so this is always correct).
        self.region_turn: dict[RegionId, int] = {
            region_id: current_turn for region_id in all_region_ids(game_map)
        }
        # Diagnostics: region-turns actually replayed vs jumped over. The ratio is
        # what says whether a slow catch-up is a lot of cheap turns or a few
        # expensive ones, which are different problems with different fixes.
        self.simulated = 0
        self.skipped = 0

    def register(
        self,
        name: str,
        step: Callable[[RegionId], None],
        bulk: Callable[[RegionId, int], None] | None = None,
        idle_turns: Callable[[RegionId], int] | None = None,
    ) -> None:
        """Add a per-region simulation step, run in registration order.

        ``step`` runs one turn and is all a system needs to provide. The other two
        are how a step opts in to being **skipped ahead** (see ``jump_limit``):

        * ``bulk(region_id, turns)`` -- do those turns' worth of work in one call.
          A step with a bulk never limits how far the region may jump.
        * ``idle_turns(region_id)`` -- how many upcoming turns this step can prove
          require nothing beyond what its bulk does. A step with neither hook
          pins the region to one turn at a time, which is the old behaviour.
        """
        self._steps.append(_Step(name, step, bulk, idle_turns))

    def region_at(self, x: int, y: int) -> RegionId:
        return region_at(self.game_map, x, y)

    def advance_region(self, region_id: RegionId) -> None:
        """Run every registered step for ``region_id``'s next turn."""
        self.simulated += 1
        for entry in self._steps:
            entry.step(region_id)
        self.region_turn[region_id] = self.region_turn.get(region_id, 0) + 1

    def jump_limit(self, region_id: RegionId) -> int:
        """How many turns ``region_id`` may be advanced in one go right now.

        The smallest ``idle_turns`` any step reports, ignoring steps that can do a
        span in bulk, and 1 for any step that offers neither -- so a region jumps
        only as far as *every* step agrees nothing happens. This is the classic
        discrete-event move: rather than tick a world where nothing can change,
        skip to the next turn on which something can.

        Nothing about the outcome depends on how far it happens to jump, which is
        the batch-independence rule this class documents: a step is asked to skip
        only turns it has proved are empty for it, and the arithmetic a bulk does
        is a function of the span, not of how the span was chosen.
        """
        limit = _UNBOUNDED
        for entry in self._steps:
            if entry.idle_turns is not None:
                limit = min(limit, max(0, entry.idle_turns(region_id)))
            elif entry.bulk is None:
                limit = min(limit, 1)
            if limit <= 1:
                return 1
        return limit

    def advance_region_by(self, region_id: RegionId, turns: int) -> None:
        """Advance ``turns`` region-turns at once. Steps with a bulk hook do the
        whole span in one call; steps without one have already promised (via
        ``idle_turns``) that the span is empty for them, so they are simply not
        run. Callers must not pass more than ``jump_limit`` allows."""
        if turns <= 0:
            return
        self.skipped += turns
        for entry in self._steps:
            if entry.bulk is not None:
                entry.bulk(region_id, turns)
        self.region_turn[region_id] = self.region_turn.get(region_id, 0) + turns

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

    def catch_up_region(
        self, region_id: RegionId, target_turn: int, max_advances: int | None = None
    ) -> bool:
        """Advance ``region_id`` toward ``target_turn``.

        With ``max_advances=None`` this preserves the old blocking behaviour and
        returns only once the region is fully current. Passing a finite cap lets
        live play prioritize the player's input frame: it performs a deterministic
        number of replayed region-turns, returns whether the region is current,
        and leaves the remaining debt for later frames.

        Turns nothing can happen on are skipped rather than replayed (see
        ``jump_limit``), and a skip counts as one advance against ``max_advances``
        -- it is cheaper than a real turn, so paying for it as if it were one is
        conservative in the direction that protects the input frame.
        """
        advances = 0
        while self.region_turn.get(region_id, target_turn) < target_turn:
            if max_advances is not None and advances >= max_advances:
                return False
            gap = target_turn - self.region_turn.get(region_id, target_turn)
            jump = min(gap, self.jump_limit(region_id))
            if jump > 1:
                self.advance_region_by(region_id, jump)
            else:
                self.advance_region(region_id)
            advances += 1
        return True

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
        lagging region to ``player_region`` -- never the stalest -- until either
        nothing lags or the budget runs out.

        A turn at a time, except where a region can prove nothing happens on the
        next several (``jump_limit``), which the pump takes in one step: an empty
        stretch of ocean is brought fully current for the price of one visit."""
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
            gap = target_turn - self.region_turn.get(nearest, target_turn)
            jump = min(gap, self.jump_limit(nearest))
            if jump > 1:
                self.advance_region_by(nearest, jump)
            else:
                self.advance_region(nearest)
