"""The dungeon behind the roguelike workbench: the rules, with no pixels.

Nothing in here imports pygame, ModernGL or `inkfx`. A dungeon can be
generated, walked around, fought in and finished with no window and no
graphics driver anywhere in sight, which is what makes the whole game
testable without rendering it. `rogue.py` is the half that knows how any of
this looks; this half does not know that it looks like anything.

The layout is entity-component: entities are bare integers, components are
small dataclasses held in one dictionary per type, and the systems -- taking a
turn, hunting the player, seeing -- are plain functions over `World`. Nothing
inherits from anything.

Three algorithms carry most of the weight and are each worth naming:

    recursive shadowcasting   what the player can see, in `field_of_view`
    breadth-first flood       how far every cell is from the player, in
                              `approach_map`, so a monster's whole decision is
                              to look at its neighbours and step downhill
    rejection sampling        where the rooms go, in `generate_dungeon`
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Iterator

import numpy


# --- components -----------------------------------------------------------
#
# Data only. A component never knows which systems read it, so adding a system
# is adding a function and nothing else.


@dataclass(slots=True)
class Position:
    """Where something is, in grid cells."""

    x: int
    y: int

    @property
    def cell(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass(slots=True)
class Glyph:
    """Which character stands for this, and what it draws over.

    The code is a code page 437 index, which is also its ASCII value for
    everything below 128 -- so `ord("@")` is the player's tile in any of the
    square tilesets that follow that layout. Storing the number here rather
    than a name keeps the renderer from having to hold a table of its own; it
    is still only data, and nothing in this module draws it.
    """

    code: int
    layer: int = 0


@dataclass(slots=True)
class Blocker:
    """Tag: two of these cannot share a cell."""


@dataclass(slots=True)
class Health:
    current: int
    maximum: int

    @property
    def alive(self) -> bool:
        return self.current > 0


@dataclass(slots=True)
class Attack:
    power: int


@dataclass(slots=True)
class Brain:
    """Tag for something that takes its own turn, and how far it notices.

    `sight` is measured along the floor rather than through the rock: it is
    read off the same flood the step is taken down, so a monster two cells
    away through a wall and thirty around the corridor is thirty away. That is
    also what stops the whole level converging on the player at once.
    """

    sight: int = 8


@dataclass(slots=True)
class Lantern:
    """Tag for something that carries its own light, and how far it throws.

    Radius only. What a light looks like is the renderer's business, not the
    dungeon's, and at present every light in the place burns the same colour
    anyway -- they differ in how far they reach, which is this number.

    Nothing takes this off a corpse. A dropped torch goes on burning where it
    fell, which is both true and the more interesting of the two options.
    """

    radius: int = 5


@dataclass(slots=True)
class Name:
    text: str


@dataclass(slots=True)
class Player:
    """Tag: the one entity the keyboard drives."""


# --- the registry ---------------------------------------------------------


class World:
    """A component registry. Entities are integers and nothing else.

    Components live in one dictionary per type. That is what makes `query`
    cheap: a system asks for the handful of entities that have what it needs
    instead of walking every entity in the game and testing each one. It also
    means an entity is created, extended and destroyed without any type ever
    being declared -- a corpse is an entity that lost its `Brain`.
    """

    def __init__(self) -> None:
        self._next_id = 1
        self._stores: dict[type, dict[int, object]] = {}

    def spawn(self, *components: object) -> int:
        """Create an entity holding `components`; returns its id."""
        entity = self._next_id
        self._next_id += 1
        for component in components:
            self.add(entity, component)
        return entity

    def add(self, entity: int, component: object) -> None:
        self._stores.setdefault(type(component), {})[entity] = component

    def get(self, entity: int, kind: type):
        """The entity's component of that type, or None if it has none."""
        return self._stores.get(kind, {}).get(entity)

    def has(self, entity: int, *kinds: type) -> bool:
        return all(entity in self._stores.get(kind, {}) for kind in kinds)

    def remove(self, entity: int, kind: type) -> None:
        self._stores.get(kind, {}).pop(entity, None)

    def kill(self, entity: int) -> None:
        """Forget an entity entirely, whatever it was made of."""
        for store in self._stores.values():
            store.pop(entity, None)

    def store(self, kind: type) -> dict[int, object]:
        """The whole store for one component type, for a system that wants it."""
        return self._stores.setdefault(kind, {})

    def query(self, *kinds: type) -> Iterator[tuple]:
        """Every entity holding all of `kinds`, with those components.

        Driven from the rarest component rather than the first one named. One
        player and forty monsters is a very different scan depending on which
        end you start from, and the caller should not have to think about the
        order it lists them in.

        The keys are copied first so a system can kill what it is iterating
        over, which is exactly what a fight does.
        """
        if not kinds:
            return
        stores = [self._stores.get(kind, {}) for kind in kinds]
        if not all(stores):
            # Nothing has one of these at all, so nothing has all of them.
            return

        smallest = min(stores, key=len)
        for entity in list(smallest):
            found = [store.get(entity) for store in stores]
            if all(component is not None for component in found):
                yield (entity, *found)


# --- the map --------------------------------------------------------------

WALL = 0
FLOOR = 1

# Eight-way, so corridors can be cut across and a monster can round a corner
# in one turn like the player can.
NEIGHBOURS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)


@dataclass(frozen=True, slots=True)
class Room:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def cells(self) -> Iterator[tuple[int, int]]:
        for y in range(self.y, self.y + self.height):
            for x in range(self.x, self.x + self.width):
                yield (x, y)

    def overlaps(self, other: "Room", margin: int = 1) -> bool:
        """True if the two rooms touch, allowing `margin` cells of wall between."""
        return (
            self.x - margin < other.x + other.width
            and other.x - margin < self.x + self.width
            and self.y - margin < other.y + other.height
            and other.y - margin < self.y + self.height
        )


class Dungeon:
    """A grid of tiles, indexed `[x, y]` to match the rest of the codebase.

    Pygame's surface arrays are column-major, so keeping the map the same way
    round means a cell and the pixels it is drawn into are indexed alike and
    there is no axis swap hiding anywhere between the two halves of the game.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.tiles = numpy.full((width, height), WALL, dtype=numpy.uint8)
        self.rooms: list[Room] = []

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.tiles[x, y] != WALL

    def transparent(self, x: int, y: int) -> bool:
        """Whether sight passes through. Separate from `walkable` on purpose:
        a closed door or a window would want one and not the other."""
        return self.walkable(x, y)

    def carve(self, x: int, y: int) -> None:
        if self.in_bounds(x, y):
            self.tiles[x, y] = FLOOR


def generate_dungeon(
    width: int,
    height: int,
    rng: random.Random,
    attempts: int = 90,
    min_size: int = 4,
    max_size: int = 9,
) -> Dungeon:
    """Rooms joined by corridors, by rejection sampling.

    Rooms are proposed at random and kept only if they clear everything
    already placed. That is far simpler than packing them deliberately and,
    for a map this size, indistinguishable from it -- the failures cost a
    rectangle overlap test each.

    Each room is then joined to the one before it, which guarantees the whole
    map is connected without any connectivity check: room `n` reaches room
    `n-1` reaches ... reaches room 0.
    """
    dungeon = Dungeon(width, height)

    # One cell of solid rock is kept around the edge so no corridor runs along
    # the boundary and no room wall is missing.
    high_x = max(min_size, min(max_size, width - 4))
    high_y = max(min_size, min(max_size, height - 4))

    for _ in range(attempts):
        room_width = rng.randint(min_size, high_x)
        room_height = rng.randint(min_size, high_y)
        if room_width + 2 >= width or room_height + 2 >= height:
            continue
        room = Room(
            rng.randint(1, width - room_width - 2),
            rng.randint(1, height - room_height - 2),
            room_width,
            room_height,
        )
        if any(room.overlaps(placed) for placed in dungeon.rooms):
            continue

        for x, y in room.cells():
            dungeon.carve(x, y)
        if dungeon.rooms:
            _join(dungeon, dungeon.rooms[-1].center, room.center, rng)
        dungeon.rooms.append(room)

    if not dungeon.rooms:
        # A map too small for even one proposal to land still has to be
        # playable, so put a chamber in the middle of it by hand.
        room = Room(1, 1, max(1, width - 2), max(1, height - 2))
        for x, y in room.cells():
            dungeon.carve(x, y)
        dungeon.rooms.append(room)

    return dungeon


def _join(
    dungeon: Dungeon,
    start: tuple[int, int],
    end: tuple[int, int],
    rng: random.Random,
) -> None:
    """Cut an L-shaped corridor between two points, with the corner either way.

    Which leg goes first is a coin toss, which is the whole difference between
    a map whose corridors all turn the same way and one that reads as dug.
    """
    (x0, y0), (x1, y1) = start, end
    horizontal_first = rng.random() < 0.5

    for x in range(min(x0, x1), max(x0, x1) + 1):
        dungeon.carve(x, y0 if horizontal_first else y1)
    for y in range(min(y0, y1), max(y0, y1) + 1):
        dungeon.carve(x1 if horizontal_first else x0, y)


# --- seeing ---------------------------------------------------------------

# The eight octant transforms that let one piece of scanning code cover the
# whole circle: each row is (xx, xy, yx, yy), a 2x2 matrix mapping the octant
# being scanned onto the one the code is written for.
_OCTANTS = (
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (0, -1, 1, 0),
    (-1, 0, 0, 1),
    (-1, 0, 0, -1),
    (0, -1, -1, 0),
    (0, 1, -1, 0),
    (1, 0, 0, -1),
)


def field_of_view(
    dungeon: Dungeon,
    origin: tuple[int, int],
    radius: int,
    into: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """Recursive shadowcasting: the cells visible from `origin`.

    Each octant is swept row by row, carrying the pair of slopes that bound
    the part of it still lit. A wall narrows that wedge -- recursing on the
    piece to one side of it and carrying on with the piece to the other -- so
    the cost is the visible area rather than the area of the circle, and the
    result has none of the artefacts that casting a ray to every cell gives.

    One known permissiveness, shared with every octant-based version of this:
    where two walls meet at a corner, the cell diagonally past the corner is
    lit. The two walls sit on the boundaries of the octant that cell is in, so
    neither is ever scanned as an obstacle to it. It is cheap to live with --
    a corner you can just see round -- and expensive to remove, so it is left
    alone and pinned down by a test rather than quietly tolerated.

    Writes into `into` when one is supplied, so a per-turn call does not
    allocate a new array every time.
    """
    if into is None:
        visible = numpy.zeros((dungeon.width, dungeon.height), dtype=bool)
    else:
        visible = into
        visible[:] = False

    x, y = origin
    if dungeon.in_bounds(x, y):
        # You can always see your own square, lit or not.
        visible[x, y] = True

    for xx, xy, yx, yy in _OCTANTS:
        _cast_light(dungeon, visible, x, y, 1, 1.0, 0.0, radius, xx, xy, yx, yy)
    return visible


def _cast_light(
    dungeon: Dungeon,
    visible: numpy.ndarray,
    cx: int,
    cy: int,
    row: int,
    start: float,
    end: float,
    radius: int,
    xx: int,
    xy: int,
    yx: int,
    yy: int,
) -> None:
    """Sweep one octant from `row` outwards between two slopes."""
    if start < end:
        return

    radius_squared = radius * radius
    blocked = False
    next_start = start

    for distance in range(row, radius + 1):
        if blocked:
            break
        dy = -distance
        for dx in range(-distance, 1):
            # The slopes to this cell's leading and trailing corners. Corners,
            # not centres: a wall has to shadow the whole cell behind it.
            left_slope = (dx - 0.5) / (dy + 0.5)
            right_slope = (dx + 0.5) / (dy - 0.5)

            if start < right_slope:
                continue
            if end > left_slope:
                break

            x = cx + dx * xx + dy * xy
            y = cy + dx * yx + dy * yy
            if not dungeon.in_bounds(x, y):
                continue

            # Round the corners of the lit area off, so the edge of sight is a
            # circle rather than the square the scan is shaped like.
            if dx * dx + dy * dy <= radius_squared:
                visible[x, y] = True

            if blocked:
                if not dungeon.transparent(x, y):
                    next_start = right_slope
                    continue
                # Out the far side of a wall: the wedge resumes where it stopped.
                blocked = False
                start = next_start
            elif not dungeon.transparent(x, y) and distance < radius:
                # A new wall. Everything to its left is handled by recursing on
                # the narrower wedge; the scan carries on to its right.
                blocked = True
                _cast_light(
                    dungeon, visible, cx, cy, distance + 1,
                    start, left_slope, radius, xx, xy, yx, yy,
                )
                next_start = right_slope


# --- moving ---------------------------------------------------------------


def approach_map(
    dungeon: Dungeon,
    goals: list[tuple[int, int]],
) -> numpy.ndarray:
    """How many steps every walkable cell is from the nearest goal; -1 if none.

    A breadth-first flood outwards from the goals, which for a unit-cost grid
    is exactly Dijkstra and a great deal less machinery. It costs one sweep of
    the map per turn and in return every monster's entire decision becomes
    "which of my eight neighbours holds the smallest number" -- no per-monster
    search, no paths to store, and no two monsters ever computing the same
    route. Walking downhill from anywhere is a shortest path by construction.

    Occupancy is deliberately not baked in here. Closing the cells monsters
    are standing on would also close the cell each monster is reading its own
    distance from, and the whole dungeon would stand still; who is in the way
    belongs to the step, not to the map.
    """
    distance = numpy.full((dungeon.width, dungeon.height), -1, dtype=numpy.int32)
    frontier: deque[tuple[int, int]] = deque()

    for x, y in goals:
        if dungeon.walkable(x, y):
            distance[x, y] = 0
            frontier.append((x, y))

    while frontier:
        x, y = frontier.popleft()
        step = distance[x, y] + 1
        for dx, dy in NEIGHBOURS:
            nx, ny = x + dx, y + dy
            if not dungeon.walkable(nx, ny) or distance[nx, ny] >= 0:
                continue
            distance[nx, ny] = step
            frontier.append((nx, ny))

    return distance


# --- the game --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Species:
    """One kind of monster, as data rather than as a subclass.

    `lantern` is how far it carries a light, or 0 for something that does not
    carry one. It is what makes the bestiary worth being data: giving the
    ogre a brand is a number in this table, not a class.
    """

    name: str
    code: int
    health: int
    power: int
    weight: int
    lantern: int = 0


BESTIARY = (
    # Vermin go about in the dark. The two that use tools carry a light, which
    # is often how you know one is there before you can see it.
    Species("rat", ord("r"), health=3, power=1, weight=5),
    Species("kobold", ord("k"), health=5, power=2, weight=3, lantern=6),
    Species("goblin", ord("g"), health=8, power=3, weight=2),
    Species("ogre", ord("O"), health=14, power=5, weight=1, lantern=9),
)


@dataclass(slots=True)
class LightSource:
    """One lamp burning in the dungeon, and which cells it can see.

    `reach` is a shadowcast from `cell`, so a wall stops this light exactly
    where it would stop sight from the same spot. Each source carries its own,
    because two lamps in different places have different shadows and merging
    them would put light through walls.

    It is occlusion and nothing else -- where the light can get to, not how
    much of it arrives. How a lamp fades with distance is a property of light
    rather than of the dungeon, so it belongs to whatever is drawing this and
    is worked out from `cell` and `radius` there.
    """

    cell: tuple[int, int]
    radius: int
    reach: numpy.ndarray

PLAYER_GLYPH = ord("@")
WALL_GLYPH = ord("#")
FLOOR_GLYPH = ord(".")
CORPSE_GLYPH = ord("%")

MESSAGE_LIMIT = 4


class Game:
    """One dungeon, its inhabitants, and whose turn it is.

    The renderer is told what changed rather than left to work it out: `dirty`
    collects every cell whose appearance could have moved this turn and
    `fresh` the ones seen for the first time. Repainting the whole page for
    one step would be eight megapixels of surface and a full texture upload,
    against a few dozen cells for the same result.
    """

    def __init__(
        self,
        size: tuple[int, int],
        seed: int = 0,
        fov_radius: int = 9,
        monsters: int = 8,
        light_slots: int = 4,
    ) -> None:
        width, height = size
        self.rng = random.Random(seed)
        self.dungeon = generate_dungeon(width, height, self.rng)
        self.world = World()
        self.fov_radius = fov_radius
        # How many lamps can burn at once. Passed in rather than assumed: it
        # is a limit of whatever is drawing the dungeon -- the occlusion map
        # has one channel per lamp -- and not of the dungeon itself.
        self.light_slots = max(1, light_slots)
        self.turn = 0

        self.visible = numpy.zeros((width, height), dtype=bool)
        self.explored = numpy.zeros((width, height), dtype=bool)
        self.lights: list[LightSource] = []

        # What the renderer has not caught up with yet.
        self.dirty: set[tuple[int, int]] = set()
        self.fresh: set[tuple[int, int]] = set()

        self.messages: deque[str] = deque(maxlen=MESSAGE_LIMIT)
        self.messages_changed = True

        start = self.dungeon.rooms[0].center
        self.player = self.world.spawn(
            Position(*start),
            Glyph(PLAYER_GLYPH, layer=2),
            Blocker(),
            Health(30, 30),
            Attack(4),
            Name("you"),
            Player(),
        )
        self._populate(monsters)

        self.log("You unfold the map. The ink is still wet.")
        self._observe()

    # --- state a caller asks about ---------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        return (self.dungeon.width, self.dungeon.height)

    @property
    def alive(self) -> bool:
        health = self.world.get(self.player, Health)
        return health is not None and health.alive

    @property
    def player_position(self) -> tuple[int, int]:
        return self.world.get(self.player, Position).cell

    @property
    def health(self) -> Health:
        return self.world.get(self.player, Health)

    @property
    def monsters_left(self) -> int:
        return sum(1 for _ in self.world.query(Brain, Health))

    def log(self, text: str) -> None:
        self.messages.append(text)
        self.messages_changed = True

    def take_dirty(self) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Hand over the changed cells and start collecting again.

        `fresh` is a subset of `dirty`: cells being seen for the first time,
        which the renderer draws as ink that has only just gone on.
        """
        dirty, fresh = self.dirty, self.fresh
        self.dirty, self.fresh = set(), set()
        return dirty, fresh

    def glyph_at(self, x: int, y: int) -> int | None:
        """The topmost entity glyph in a cell, or None if nothing is standing there."""
        best: Glyph | None = None
        for _, position, glyph in self.world.query(Position, Glyph):
            if position.cell != (x, y):
                continue
            if best is None or glyph.layer > best.layer:
                best = glyph
        return None if best is None else best.code

    def blocker_at(self, x: int, y: int) -> int | None:
        for entity, position, _ in self.world.query(Position, Blocker):
            if position.cell == (x, y):
                return entity
        return None

    # --- taking a turn ----------------------------------------------------

    def act(self, dx: int, dy: int) -> bool:
        """Move or attack in a direction; `(0, 0)` waits. True if a turn passed.

        Walking into a wall is not a turn -- a misjudged keypress should not
        hand the dungeon a free round -- so the monsters only move when this
        returns True.
        """
        if not self.alive:
            return False

        if dx or dy:
            position = self.world.get(self.player, Position)
            target = (position.x + dx, position.y + dy)

            occupant = self.blocker_at(*target)
            if occupant is not None:
                self._strike(self.player, occupant)
            elif self.dungeon.walkable(*target):
                self._place(self.player, target)
            else:
                return False

        self.turn += 1
        self._monsters_act()
        self._observe()
        return True

    def _monsters_act(self) -> None:
        """Everything with a brain gets one step, downhill towards the player."""
        if not self.alive:
            return

        player_cell = self.player_position
        movers = list(self.world.query(Brain, Position, Health))
        if not movers:
            return

        distance = approach_map(self.dungeon, [player_cell])
        # Where the bodies are, kept up to date as they move, so a line of
        # monsters in a corridor shuffles forward one at a time instead of all
        # piling onto the same square.
        taken = {
            position.cell
            for _, position, _ in self.world.query(Position, Blocker)
            if position.cell != player_cell
        }

        for entity, brain, position, health in movers:
            if not health.alive:
                continue
            if not self.alive:
                # The player went down earlier in the round; the rest of the
                # dungeon does not get to keep hitting a corpse.
                break
            here = distance[position.x, position.y]
            if here < 0 or here > brain.sight:
                # Out of the flood entirely, or too far off to have noticed.
                continue
            if here == 1:
                self._strike(entity, self.player)
                continue

            step = self._downhill(distance, position.cell, taken)
            if step is not None:
                taken.discard(position.cell)
                taken.add(step)
                self._place(entity, step)

    def _downhill(
        self,
        distance: numpy.ndarray,
        cell: tuple[int, int],
        taken: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """The free neighbour closest to the goal, or None if none improves."""
        x, y = cell
        best = distance[x, y]
        chosen = None
        for dx, dy in NEIGHBOURS:
            nx, ny = x + dx, y + dy
            if not self.dungeon.in_bounds(nx, ny) or (nx, ny) in taken:
                continue
            value = distance[nx, ny]
            if 0 <= value < best:
                best = value
                chosen = (nx, ny)
        return chosen

    def _place(self, entity: int, cell: tuple[int, int]) -> None:
        """Move an entity, marking both cells it affected."""
        position = self.world.get(entity, Position)
        self._touch(position.cell)
        position.x, position.y = cell
        self._touch(cell)

    def _strike(self, attacker: int, defender: int) -> None:
        power = self.world.get(attacker, Attack)
        health = self.world.get(defender, Health)
        if power is None or health is None:
            return

        # A little spread, so the same fight is not the same every time.
        damage = max(1, power.power + self.rng.randint(-1, 1))
        health.current -= damage

        attacker_name = self.world.get(attacker, Name).text
        defender_name = self.world.get(defender, Name).text
        if attacker == self.player:
            self.log(f"You hit the {defender_name} for {damage}.")
        else:
            self.log(f"The {attacker_name} hits you for {damage}.")

        if health.alive:
            return

        position = self.world.get(defender, Position)
        if defender == self.player:
            self.log("You die. Press N for a fresh sheet.")
            self.world.remove(self.player, Blocker)
            self.world.get(self.player, Glyph).code = CORPSE_GLYPH
        else:
            self.log(f"The {defender_name} dies.")
            # A corpse is the same entity with the parts that acted taken off
            # it, which is the whole argument for composition in one line.
            self.world.remove(defender, Brain)
            self.world.remove(defender, Blocker)
            self.world.remove(defender, Attack)
            self.world.get(defender, Glyph).code = CORPSE_GLYPH
            self.world.get(defender, Glyph).layer = 1
            if self.monsters_left == 0:
                self.log("Nothing else stirs. The dungeon is yours.")
        self._touch(position.cell)

    # --- bookkeeping ------------------------------------------------------

    def _touch(self, cell: tuple[int, int]) -> None:
        if self.dungeon.in_bounds(*cell):
            self.dirty.add(cell)

    def _observe(self) -> None:
        """Recompute sight, and note every cell whose appearance changed."""
        was_visible = self.visible.copy()
        field_of_view(
            self.dungeon, self.player_position, self.fov_radius, self.visible
        )

        newly_seen = self.visible & ~self.explored
        self.explored |= self.visible

        for x, y in numpy.argwhere(newly_seen):
            self.fresh.add((int(x), int(y)))
        # Cells that lit up or went dark both have to be redrawn: one gains
        # ink, the other fades back to the remembered weight.
        for x, y in numpy.argwhere(was_visible != self.visible):
            self.dirty.add((int(x), int(y)))

        self._relight()

    def _relight(self) -> None:
        """Work out which lamps are burning, and which cells each can see.

        Slot 0 is always the player, whose lamp reaches exactly what the
        player can see -- the shadowcast already done above, so it is free.

        The rest go to other lights, and the test for whether one counts is
        not whether you can see the thing holding it. It is whether any of the
        light it throws lands somewhere you can see. A torch around a corner
        lights the far wall of the passage you are standing in, and you see
        that wall lit long before you see what is lighting it, which is most
        of the reason a light in someone else's hand is worth having at all.

        What is clipped away is only the part falling where the player cannot
        see: light on a cell you have no view of would be drawing you a room
        you have never been shown.
        """
        player_cell = self.player_position
        lights = [
            LightSource(player_cell, self.fov_radius, self.visible.copy())
        ]

        candidates = []
        for entity, position, lantern in self.world.query(Position, Lantern):
            if entity == self.player:
                continue
            span = max(
                abs(position.x - player_cell[0]), abs(position.y - player_cell[1])
            )
            # Nothing this lamp throws could possibly land inside the player's
            # sight, so it is not worth a shadowcast to find that out.
            if span > self.fov_radius + lantern.radius:
                continue

            reach = field_of_view(self.dungeon, position.cell, lantern.radius)
            reach &= self.visible
            if not reach.any():
                continue
            candidates.append((span, entity, position.cell, lantern.radius, reach))

        # Nearest first, so when there are more lamps than slots the ones that
        # are dropped are the ones contributing least.
        candidates.sort(key=lambda found: (found[0], found[1]))
        for _, _, cell, radius, reach in candidates[: self.light_slots - 1]:
            lights.append(LightSource(cell, radius, reach))

        self.lights = lights

    def _populate(self, count: int) -> None:
        """Scatter monsters through every room but the one the player is in."""
        rooms = self.dungeon.rooms[1:]
        if not rooms:
            return

        species = [
            kind for kind in BESTIARY for _ in range(kind.weight)
        ]  # a weighted bag, so `choice` respects how common each kind is
        occupied = {self.player_position}

        for _ in range(count):
            room = self.rng.choice(rooms)
            cell = (
                self.rng.randrange(room.x, room.x + room.width),
                self.rng.randrange(room.y, room.y + room.height),
            )
            if cell in occupied or not self.dungeon.walkable(*cell):
                continue
            occupied.add(cell)

            kind = self.rng.choice(species)
            monster = self.world.spawn(
                Position(*cell),
                Glyph(kind.code, layer=2),
                Blocker(),
                Health(kind.health, kind.health),
                Attack(kind.power),
                Brain(),
                Name(kind.name),
            )
            if kind.lantern:
                self.world.add(monster, Lantern(kind.lantern))
