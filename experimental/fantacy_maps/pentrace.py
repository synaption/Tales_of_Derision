"""Turn a finished drawing into the strokes that would have drawn it.

The input is a coverage mask -- a boolean array saying where the ink is -- and
the output is an ordered list of pen strokes, each a polyline with a nib width
at every point. Nothing here knows about pygame, OpenGL or a window: it is
arrays in and dataclasses out, so it can be tested without rendering anything.

The idea is the medial axis. A shape made by dragging a round nib around is
exactly the set of points within some distance of the path the nib took, so
recovering the path means finding the centre of the shape, and recovering the
nib means measuring how far the centre is from the edge. Those are two classical
transforms, and this module is essentially them plus the bookkeeping to turn
their output into something a pen can follow:

    mask -> distance transform -> how wide the nib was, everywhere
         -> thinning           -> where the nib went, one pixel wide
         -> tracing            -> that skeleton as polylines
         -> pruning, simplify  -> polylines worth drawing, cheaply
         -> ordering           -> a sequence a hand would plausibly use

The reconstruction is not pixel-exact and is not meant to be. A skeleton loses
the sharp ends of a shape -- thinning eats them before it stops -- so corners
come back slightly rounded, and that is the right trade for artwork that is
about to be drawn in wet ink by a round brush anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Chamfer weights. Using sqrt(2) for the diagonal rather than the integer 3-4
# mask keeps the distance -- which becomes the nib width -- within about 4% of
# true Euclidean, and the extra cost is nil since it is all vectorised.
ORTHOGONAL_STEP = 1.0
DIAGONAL_STEP = math.sqrt(2.0)


@dataclass
class TracedStroke:
    """One pen stroke recovered from a drawing, in mask pixels.

    `widths` is the nib radius at each point, so it is the same length as
    `points`. A stroke is stored as a plain polyline rather than a curve
    because that is what the pen consumes, and because the thing being traced
    was pixels to begin with.
    """

    points: list[tuple[float, float]]
    widths: list[float] = field(default_factory=list)

    @property
    def length(self) -> float:
        return sum(
            math.dist(a, b) for a, b in zip(self.points, self.points[1:])
        )

    @property
    def mean_width(self) -> float:
        return sum(self.widths) / len(self.widths) if self.widths else 0.0

    def reversed(self) -> "TracedStroke":
        return TracedStroke(self.points[::-1], self.widths[::-1])


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Distance from every set pixel to the nearest clear one.

    A chamfer transform run to a fixed point: each pass lets every pixel take
    the best of its own distance and its neighbours' plus one step, and the
    whole field settles once no pixel can improve. Everything outside the array
    counts as clear, so a shape running off the edge is treated as ending there.

    Iterating to convergence costs one pass per pixel of the widest stroke,
    which for sprite-sized artwork is a few dozen passes over a small array.
    """
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)

    far = float(mask.shape[0] + mask.shape[1])
    distance = np.where(mask, far, 0.0).astype(np.float32)

    while True:
        padded = np.pad(distance, 1, constant_values=0.0)
        best = np.minimum.reduce(
            [
                padded[0:-2, 1:-1] + ORTHOGONAL_STEP,
                padded[2:, 1:-1] + ORTHOGONAL_STEP,
                padded[1:-1, 0:-2] + ORTHOGONAL_STEP,
                padded[1:-1, 2:] + ORTHOGONAL_STEP,
                padded[0:-2, 0:-2] + DIAGONAL_STEP,
                padded[0:-2, 2:] + DIAGONAL_STEP,
                padded[2:, 0:-2] + DIAGONAL_STEP,
                padded[2:, 2:] + DIAGONAL_STEP,
            ]
        )
        improved = np.minimum(distance, best)
        if np.array_equal(improved, distance):
            return distance
        distance = improved


def _neighbourhood(mask: np.ndarray) -> list[np.ndarray]:
    """The eight neighbours of every pixel, clockwise from north.

    Returned as whole arrays rather than looked up per pixel: the thinning
    below is a handful of boolean expressions over these, which is what makes
    it fast enough to run on a 512-square sprite without a compiled helper.
    """
    padded = np.pad(mask, 1, constant_values=False)
    return [
        padded[0:-2, 1:-1],  # N
        padded[0:-2, 2:],  # NE
        padded[1:-1, 2:],  # E
        padded[2:, 2:],  # SE
        padded[2:, 1:-1],  # S
        padded[2:, 0:-2],  # SW
        padded[1:-1, 0:-2],  # W
        padded[0:-2, 0:-2],  # NW
    ]


def thin(mask: np.ndarray) -> np.ndarray:
    """Reduce a filled shape to a one-pixel-wide skeleton down its middle.

    Zhang-Suen: two alternating sub-iterations peel a layer off the boundary,
    deleting only pixels that are neither an end of a line nor a bridge holding
    one part of the shape to another. Those two guards are what stop it eating
    the figure entirely -- what is left when nothing more can be deleted is a
    curve through the middle with the shape's connectivity intact.
    """
    image = mask.astype(bool).copy()

    while True:
        changed = False
        for second_pass in (False, True):
            neighbours = _neighbourhood(image)
            north, north_east, east, south_east = neighbours[0:4]
            south, south_west, west, north_west = neighbours[4:8]

            # How many neighbours are set: two is a line, one is an end, and
            # more than six means the pixel is interior and not on a boundary.
            filled = sum(n.astype(np.uint8) for n in neighbours)

            # How many times the ring of neighbours goes from clear to set.
            # Exactly one means the set neighbours form a single arc, so the
            # pixel is not the only bridge between two arms of the shape.
            ring = neighbours + [neighbours[0]]
            crossings = sum(
                ((~ring[index]) & ring[index + 1]).astype(np.uint8)
                for index in range(8)
            )

            if second_pass:
                blocked = (north & east & west) | (north & south & west)
            else:
                blocked = (north & east & south) | (east & south & west)

            doomed = (
                image
                & (filled >= 2)
                & (filled <= 6)
                & (crossings == 1)
                & ~blocked
            )
            if doomed.any():
                image &= ~doomed
                changed = True

        if not changed:
            return image


def _adjacency(skeleton: np.ndarray) -> dict[tuple[int, int], list]:
    """The skeleton as a graph: every set pixel mapped to its set neighbours.

    Eight-connected, because that is what the thinning produces -- a diagonal
    run of pixels is a line, not a dotted one -- but with the diagonals that
    merely duplicate an orthogonal path taken back out. At a staircase corner
    all three pixels are mutually adjacent, and left alone that makes every
    corner of the drawing look like a three-way junction; on a sprite-sized
    figure it turns a few dozen real branches into several hundred fragments.
    Dropping the shortcut leaves the corner as what it is, a bend in one line.
    """
    rows, columns = np.nonzero(skeleton)
    pixels = set(zip(columns.tolist(), rows.tolist()))
    steps = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))

    def links(x: int, y: int) -> list[tuple[int, int]]:
        found = []
        for dx, dy in steps:
            if (x + dx, y + dy) not in pixels:
                continue
            redundant = dx and dy and (
                (x + dx, y) in pixels or (x, y + dy) in pixels
            )
            if not redundant:
                found.append((x + dx, y + dy))
        return found

    return {(x, y): links(x, y) for x, y in pixels}


def trace_polylines(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    """Break a skeleton into polylines, cut at every end and every junction.

    Each edge of the skeleton graph is walked exactly once, so a fork produces
    one polyline per arm rather than an arbitrary path through the fork, and a
    closed loop -- which has no end to start from -- is picked up on a second
    sweep that will start anywhere.
    """
    neighbours = _adjacency(skeleton)
    degree = {pixel: len(links) for pixel, links in neighbours.items()}
    walked: set[frozenset] = set()
    lines: list[list[tuple[int, int]]] = []

    def walk(start: tuple[int, int], step: tuple[int, int]) -> list:
        line = [start]
        previous, current = start, step
        while True:
            walked.add(frozenset((previous, current)))
            line.append(current)
            if degree[current] != 2:
                break
            onward = next(
                (p for p in neighbours[current] if p != previous), None
            )
            if onward is None or frozenset((current, onward)) in walked:
                break
            previous, current = current, onward
        return line

    # Sorted so the result does not depend on set iteration order, which would
    # make the stroke order -- and so the animation -- differ between runs.
    ordered = sorted(neighbours)
    for pixel in ordered:
        if degree[pixel] == 0:
            lines.append([pixel])  # a lone dot is a stroke that goes nowhere
        elif degree[pixel] != 2:
            for step in neighbours[pixel]:
                if frozenset((pixel, step)) not in walked:
                    lines.append(walk(pixel, step))

    for pixel in ordered:
        for step in neighbours[pixel]:
            if frozenset((pixel, step)) not in walked:
                lines.append(walk(pixel, step))

    return lines


def _heading(points: list, at_end: bool, reach: int = 4) -> tuple[float, float]:
    """A unit vector pointing out of one end of a polyline, into the junction.

    Taken over several points rather than the last two: on a pixel skeleton a
    single step is one of eight directions, which is far too coarse to tell a
    line continuing straight from one turning off.
    """
    if at_end:
        near, far = points[-1], points[max(0, len(points) - 1 - reach)]
    else:
        near, far = points[0], points[min(len(points) - 1, reach)]
    dx, dy = near[0] - far[0], near[1] - far[1]
    span = math.hypot(dx, dy)
    return (0.0, 0.0) if span == 0.0 else (dx / span, dy / span)


def chain_polylines(lines: list[list], max_turn: float = 75.0) -> list[list]:
    """Join polylines that meet and carry straight on into single strokes.

    Tracing cuts the skeleton at every junction, which is right for finding the
    branches but wrong for drawing them: a limb that a hand drew in one motion
    arrives as three pieces because two other lines happen to touch it. This
    puts them back together, pairing up the ends that meet at a junction by how
    little the pen would have to turn to go from one into the other.

    Greedy over all candidate pairs, least turn first, so the straightest
    continuation at a junction wins the pairing and the arms that really do
    branch off are left as their own strokes. `max_turn` is in degrees, and a
    pair that would bend more sharply than that is not a continuation at all.
    """
    limit = math.cos(math.radians(max_turn))
    lines = [list(line) for line in lines]

    # Every free end of every line, gathered by the pixel it sits on.
    ends: dict[tuple, list[tuple[int, bool]]] = {}
    for index, line in enumerate(lines):
        if len(line) < 2:
            continue
        ends.setdefault(tuple(line[0]), []).append((index, False))
        ends.setdefault(tuple(line[-1]), []).append((index, True))

    candidates = []
    for meeting in ends.values():
        for first in range(len(meeting)):
            for second in range(first + 1, len(meeting)):
                a_index, a_end = meeting[first]
                b_index, b_end = meeting[second]
                if a_index == b_index:
                    continue  # a line meeting itself would close a loop
                ax, ay = _heading(lines[a_index], a_end)
                bx, by = _heading(lines[b_index], b_end)
                # Straight on means B leaves the junction the way A arrived, so
                # the two headings -- both pointing inwards -- are opposed.
                straightness = -(ax * bx + ay * by)
                if straightness >= limit:
                    candidates.append(
                        (straightness, a_index, a_end, b_index, b_end)
                    )

    candidates.sort(key=lambda item: -item[0])
    joined: dict[tuple[int, bool], tuple[int, bool]] = {}
    for _, a_index, a_end, b_index, b_end in candidates:
        if (a_index, a_end) in joined or (b_index, b_end) in joined:
            continue
        joined[(a_index, a_end)] = (b_index, b_end)
        joined[(b_index, b_end)] = (a_index, a_end)

    # Walk the pairings into chains. Each line is used once, entered from
    # whichever of its ends is still free.
    spent = [False] * len(lines)
    chains = []
    for index, line in enumerate(lines):
        if spent[index] or len(line) < 2:
            continue
        # Start from an end that nothing joins on to, so a chain is walked from
        # its beginning rather than picked up in the middle.
        if (index, False) in joined and (index, True) in joined:
            continue
        chains.append(_follow_chain(lines, joined, spent, index))

    # Anything left is a closed ring of joins, with no free end to start at.
    for index, line in enumerate(lines):
        if not spent[index] and len(line) >= 2:
            chains.append(_follow_chain(lines, joined, spent, index))

    chains.extend(line for line in lines if len(line) < 2)
    return chains


def _follow_chain(lines, joined, spent, index: int) -> list:
    """Concatenate lines from `index` onwards through the pairing table."""
    at_end = (index, False) in joined  # enter from the joined end if there is one
    chain: list = []
    while True:
        spent[index] = True
        piece = lines[index][::-1] if at_end else lines[index]
        # The junction pixel itself belongs to both lines; keep one copy.
        chain.extend(piece[1:] if chain and chain[-1] == piece[0] else piece)
        onward = joined.get((index, not at_end))
        if onward is None or spent[onward[0]]:
            return chain
        index, at_end = onward


def simplify(points: list[tuple[float, float]], tolerance: float) -> list:
    """Drop points a straight line would have passed through anyway.

    Ramer-Douglas-Peucker, iteratively so a long skeleton branch cannot blow
    the stack. A traced branch has one point per pixel; at a tolerance under a
    pixel this typically removes nine in ten of them, which is worth doing
    because every survivor is a vertex the pen has to stop at.
    """
    if len(points) < 3:
        return list(points)

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    pending = [(0, len(points) - 1)]

    while pending:
        first, last = pending.pop()
        (x0, y0), (x1, y1) = points[first], points[last]
        dx, dy = x1 - x0, y1 - y0
        span = math.hypot(dx, dy)

        worst, worst_at = -1.0, first
        for index in range(first + 1, last):
            x, y = points[index]
            if span == 0.0:
                offset = math.dist((x, y), (x0, y0))
            else:
                # Twice the triangle's area over its base: the perpendicular
                # distance from the point to the chord.
                offset = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / span
            if offset > worst:
                worst, worst_at = offset, index

        if worst > tolerance:
            keep[worst_at] = True
            pending.append((first, worst_at))
            pending.append((worst_at, last))

    return [point for point, kept in zip(points, keep) if kept]


def order_strokes(strokes: list[TracedStroke]) -> list[TracedStroke]:
    """Sequence the strokes the way a hand would: never far from the last one.

    Greedy nearest-neighbour over stroke ends, starting from the longest stroke
    since that is usually the figure's main line. A stroke is flipped if its
    other end is the closer one, which is free -- a stroke has no inherent
    direction -- and stops the pen jumping back and forth across the page.

    Greedy is the right amount of effort here. This is the travelling salesman
    problem and the optimum is not worth chasing: the cost of a bad ordering is
    that the drawing looks slightly less purposeful, not that it looks wrong.
    """
    if not strokes:
        return []

    remaining = list(strokes)
    first = max(range(len(remaining)), key=lambda i: remaining[i].length)
    ordered = [remaining.pop(first)]

    while remaining:
        here = ordered[-1].points[-1]
        best, best_gap, flip = 0, math.inf, False
        for index, stroke in enumerate(remaining):
            head = math.dist(here, stroke.points[0])
            tail = math.dist(here, stroke.points[-1])
            if min(head, tail) < best_gap:
                best, best_gap, flip = index, min(head, tail), tail < head
        stroke = remaining.pop(best)
        ordered.append(stroke.reversed() if flip else stroke)

    return ordered


def trace_drawing(
    mask: np.ndarray,
    tolerance: float = 0.8,
    spur_factor: float = 1.4,
    width_gain: float = 1.0,
    max_turn: float = 75.0,
) -> list[TracedStroke]:
    """Recover the strokes that would draw `mask`, ready to hand to a pen.

    `tolerance` is how far, in pixels, a simplified polyline may stray from the
    traced one. `spur_factor` drops branches shorter than that many times their
    own width: thinning throws off short whiskers wherever the outline bulges,
    and they are artefacts of the shape rather than strokes anyone drew.
    `width_gain` scales every nib, for a brush whose soft edge does not cover
    quite as much as the geometry says it should. `max_turn` is how sharply the
    pen may bend and still be treated as carrying on through a junction.

    The order of the steps matters. Whiskers are pruned before the joining, so
    a stub cannot win a junction from the line that really continues through
    it; simplification comes after, so the joining sees the traced shape rather
    than an approximation of it.
    """
    mask = mask.astype(bool)
    distance = distance_transform(mask)
    skeleton = thin(mask)

    def width(point) -> float:
        x, y = point
        return max(
            0.5, float(distance[int(round(y)), int(round(x))]) * width_gain
        )

    def is_whisker(line) -> bool:
        if len(line) < 2:
            return False
        length = sum(math.dist(a, b) for a, b in zip(line, line[1:]))
        thickness = sum(width(p) for p in line) / len(line)
        return length < spur_factor * thickness

    lines = [line for line in trace_polylines(skeleton) if not is_whisker(line)]

    strokes = []
    for line in chain_polylines(lines, max_turn):
        points = simplify([(float(x), float(y)) for x, y in line], tolerance)
        strokes.append(TracedStroke(points, [width(p) for p in points]))

    return order_strokes(strokes)


def reveal_travel(
    mask: np.ndarray,
    strokes: list[TracedStroke],
    lift: float = 0.0,
    slack: float = 1.0,
) -> np.ndarray:
    """How far a pen has travelled when it first reaches each pixel of `mask`.

    This is what makes a traced animation exact. Redrawing a figure with a
    round brush can only ever approximate it, because the brush is not the tool
    that made it; but a pen sweeping along the traced path says *when* each
    pixel is arrived at, and the drawing can then uncover its own pixels in
    that order. The shape is the original's, to the pixel, and only the order it
    appears in comes from the trace.

    That also means overshoot is free. A nib reaching past the edge of the
    figure uncovers nothing, so the radius is deliberately generous -- `slack`
    pixels past the traced width -- and the only thing that could go wrong,
    a pixel the nib never comes near, is filled in afterwards from its
    neighbours.

    Travel is returned in pixels, not seconds, so that changing the pen's speed
    part way through a drawing takes effect at once. Between one stroke and the
    next the pen is charged for the distance it has to cross, as though it
    moved through the air at the same rate, plus `lift` for picking it up and
    putting it down. Charging the real gap rather than a flat pause is what
    keeps a figure of many short strokes from spending most of its time waiting:
    the ordering has already put the next stroke nearby, so the crossing is
    usually short, and where it is not the pause reads as the hand moving.
    """
    height, width = mask.shape
    travel = np.full(mask.shape, np.inf, dtype=np.float32)
    clock = 0.0

    def touch(x: float, y: float, radius: float, when: float) -> None:
        reach = int(math.ceil(radius))
        left, right = max(0, int(x) - reach), min(width, int(x) + reach + 1)
        top, bottom = max(0, int(y) - reach), min(height, int(y) + reach + 1)
        if left >= right or top >= bottom:
            return
        rows, columns = np.ogrid[top:bottom, left:right]
        under_nib = (columns - x) ** 2 + (rows - y) ** 2 <= radius * radius
        patch = travel[top:bottom, left:right]
        np.minimum(
            patch,
            np.where(under_nib & mask[top:bottom, left:right], when, np.inf),
            out=patch,
        )

    previous_end = None
    for stroke in strokes:
        if previous_end is not None:
            clock += math.dist(previous_end, stroke.points[0]) + lift
        previous_end = stroke.points[-1]

        nibs = list(zip(stroke.points, stroke.widths))
        if len(nibs) == 1:
            (x, y), radius = nibs[0]
            touch(x, y, radius + slack, clock)
        for (start, from_width), (end, to_width) in zip(nibs, nibs[1:]):
            span = math.dist(start, end)
            # Half-pixel steps, so the swept region has no gaps in it whatever
            # the nib is doing.
            steps = max(1, int(span * 2.0))
            for index in range(steps + 1):
                along = index / steps
                touch(
                    start[0] + (end[0] - start[0]) * along,
                    start[1] + (end[1] - start[1]) * along,
                    from_width + (to_width - from_width) * along + slack,
                    clock + span * along,
                )
            clock += span

    return _fill_unreached(travel, mask, clock)


def _fill_unreached(
    travel: np.ndarray, mask: np.ndarray, clock: float
) -> np.ndarray:
    """Give a time to pixels the nib never covered, from their neighbours.

    Thinning eats the sharp tip of a shape before it stops, and pruning drops
    the whiskers, so a few pixels of a figure lie outside everything the pen
    swept. They grow outwards from whatever was drawn nearest them, one pixel
    of travel per ring, which reads as the tip of a stroke finishing itself.
    """
    missing = mask & ~np.isfinite(travel)
    while missing.any():
        padded = np.pad(travel, 1, constant_values=np.inf)
        nearest = np.minimum.reduce(
            [
                padded[0:-2, 1:-1],
                padded[2:, 1:-1],
                padded[1:-1, 0:-2],
                padded[1:-1, 2:],
                padded[0:-2, 0:-2],
                padded[0:-2, 2:],
                padded[2:, 0:-2],
                padded[2:, 2:],
            ]
        )
        travel = np.where(missing, nearest + 1.0, travel)
        filled = missing & np.isfinite(travel)
        if not filled.any():
            # An island with nothing drawn anywhere near it: put it last.
            return np.where(missing, clock, travel)
        missing &= ~filled

    return travel


def bounding_box(strokes: list[TracedStroke]) -> tuple[float, float, float, float]:
    """The box the strokes cover, nibs included, as (left, top, right, bottom)."""
    if not strokes:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [x for s in strokes for x in (p[0] for p in s.points)]
    ys = [y for s in strokes for y in (p[1] for p in s.points)]
    reach = max(w for s in strokes for w in s.widths)
    return (min(xs) - reach, min(ys) - reach, max(xs) + reach, max(ys) + reach)
