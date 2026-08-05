"""Playable prototype for Project Nightshift.

Draws with the MSX-UnDeadPeopleEdition CP437 sheet on a low-resolution internal
surface, and runs the core underneath it: one persistent three-storey mansion
generated from a seed, raycast field of view with tile memory, the energy
scheduler, enemies that chase down a flow field, stance and cover, and a
key/fixture dependency chain you have to backtrack through to escape.

The mansion is generated once and never regenerates. Stairs move you between
storeys; doors you opened, enemies you put down, and puzzles you solved all stay
that way.

    python mockup.py                 # play; window auto-fits the desktop
    python mockup.py --cell 24       # 960x600, between the 16px 1x and 2x rungs
    python mockup.py --seed 18421    # fixed seed
    python mockup.py --out shot.png  # headless screenshot

Keys: WASD/arrows move (bump to attack, open a door, or finish a downed enemy),
shift+dir sprint, C crouch, E interact, space wait, F fire, R reload, Q first
aid, ? controls, F1 debug, N new mansion, Escape quit.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pygame

HERE = Path(__file__).parent
SHEET = HERE / "MSX-UnDeadPeopleEdition.png"

SHEET_CELL = 32          # native glyph size in the sheet
DEFAULT_CELL = 16
CELL = DEFAULT_CELL      # internal-surface cell size; pick_geometry may raise it at startup
COLS, ROWS = 40, 25      # internal surface in cells

# (cell, upscale) rungs in ascending window size. Cell size is the only way to
# land between 640x400 and 1280x800 without fractional scaling.
#   16,1 = 640x400   24,1 = 960x600   16,2 = 1280x800
#   16,3 = 1920x1200   16,4 = 2560x1600
SIZE_LADDER = [(16, 1), (24, 1), (16, 2), (16, 3), (16, 4)]

VIEW_W, VIEW_H = 26, 20  # map viewport, top-left at (0, 1)
MAP_X, MAP_Y = 0, 1
DIVIDER_X = 26
PANEL_X, PANEL_W = 27, 13
LOG_Y = 22

FLOOR_W, FLOOR_H = 56, 38
STOREYS = ("CELLAR", "GROUND", "UPPER")
GROUND = 1               # the storey you start on and escape from

# --- palette ---------------------------------------------------------------
# Desaturated, slightly sick. Everything else is this multiplied by light.
BLACK = (0, 0, 0)
UI_DIM = (104, 116, 100)
UI_TEXT = (172, 188, 156)
UI_HOT = (198, 78, 56)
UI_WARM = (196, 172, 116)
OTHER_STOREY = (58, 74, 96)   # cold blue: something on a floor that isn't yours

# --- tiles -----------------------------------------------------------------
# char -> (cp437 glyph, colour, walkable, opaque)
TILES = {
    "#": ("█", (114, 106, 94), False, True),    # wall
    ".": (".", (96, 92, 90), True, False),      # floorboards
    ",": ("░", (88, 78, 62), True, False),      # debris
    '"': ("░", (132, 58, 52), True, False),     # carpet
    "=": ("≡", (150, 110, 62), False, False),   # furniture -- low cover
    "H": ("▀", (146, 132, 104), False, False),  # railing -- low cover, see over
    "O": (" ", (0, 0, 0), False, False),        # open shaft down to the storey below
    "T": ("Φ", (150, 146, 132), False, True),   # statue
    "+": ("+", (158, 116, 62), False, True),    # closed door
    "'": ("'", (128, 98, 56), True, False),     # open door
    "L": ("╬", (176, 96, 72), False, True),     # locked door
    "*": ("☼", (236, 194, 112), True, False),   # candle
    "%": ("%", (124, 28, 30), True, False),     # gore
    "<": ("<", (172, 172, 182), True, False),   # stairs up
    ">": (">", (172, 172, 182), True, False),   # stairs down
    "Ω": ("Ω", (150, 180, 190), False, False),  # fixture: fuse box / safe
    "∩": ("∩", (214, 196, 150), True, False),   # the front door -- the way out
}

LOW_COVER = ("=", "H")   # crouch behind these

# char -> (cp437 glyph, colour, display name)
ITEMS = {
    "!": ("!", (198, 62, 58), "FIRST AID"),
    "/": ("/", (150, 152, 160), "CROWBAR"),
    "&": ("¶", (190, 180, 150), "CODE NOTE"),
    "k": ("§", (204, 172, 80), "BRASS KEY"),
    "i": ("§", (188, 196, 210), "IRON KEY"),
    "f": ("♣", (196, 156, 96), "FUSE"),
    "a": ("=", (188, 160, 96), "AMMO BOX"),
}

CANDLE_RADIUS = 4.0
LANTERN_RADIUS = 9.5
SIGHT_RADIUS = 11
MEMORY_LEVEL = 0.20
AMBIENT = 0.50        # dark-adjusted eyes: a visible tile is never pitch black
LIGHT_LEVELS = 7

# --- action costs (README table) -------------------------------------------
COST_WAIT = 50
COST_STANCE = 25
COST_MOVE = 100
COST_CROUCH_MOVE = 125
COST_SPRINT = 150
COST_DOOR = 75
COST_PICKUP = 50
COST_FIRE = 100
COST_MELEE = 100
COST_FINISH = 150
COST_RELOAD = 125
COST_HEAL = 150
COST_INTERACT = 100

BAYER = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
]

# Python's cp437 codec maps 0x00-0x1F to C0 controls, so the graphical glyphs
# that live down there need their indices spelled out.
LOW_GLYPHS = {"☼": 15, "¶": 20, "§": 21, "♣": 5, "↑": 24, "→": 26, "▲": 30}


# ---------------------------------------------------------------------------
# font
# ---------------------------------------------------------------------------
def _downsample(cell: pygame.Surface) -> pygame.Surface:
    """32 -> CELL px, snapped back to two tones.

    A plain smoothscale turns dense glyphs ('@', '%') into grey mush; averaging
    then thresholding keeps them reading as bitmaps at the smaller cell.
    """
    small = pygame.transform.smoothscale(cell, (CELL, CELL))
    out = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
    for y in range(CELL):
        for x in range(CELL):
            r, g, b, a = small.get_at((x, y))
            if a < 96:
                continue
            luma = (r * 3 + g * 6 + b) // 10
            tone = 255 if luma > 104 else 0
            out.set_at((x, y), (tone, tone, tone, 255))
    return out


class Font:
    """CP437 sheet sliced to CELL px, tinted on demand."""

    def __init__(self, path: Path) -> None:
        sheet = pygame.image.load(str(path)).convert_alpha()
        self._glyphs: list[pygame.Surface] = []
        for index in range(256):
            sx = (index % 16) * SHEET_CELL
            sy = (index // 16) * SHEET_CELL
            cell = sheet.subsurface(pygame.Rect(sx, sy, SHEET_CELL, SHEET_CELL))
            self._glyphs.append(_downsample(cell))
        self._cache: dict[tuple[int, tuple[int, int, int]], pygame.Surface] = {}

    def tinted(self, char: str, colour: tuple[int, int, int]) -> pygame.Surface:
        index = LOW_GLYPHS.get(char) or char.encode("cp437")[0]
        key = (index, colour)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._glyphs[index].copy()
            cached.fill((*colour, 255), special_flags=pygame.BLEND_RGB_MULT)
            self._cache[key] = cached
        return cached

    def blit(self, dest, char, cx, cy, colour) -> None:
        if char == " ":
            return
        dest.blit(self.tinted(char, colour), (cx * CELL, cy * CELL))

    def text(self, dest, string, cx, cy, colour) -> None:
        for offset, char in enumerate(string):
            self.blit(dest, char, cx + offset, cy, colour)


# ---------------------------------------------------------------------------
# actors
# ---------------------------------------------------------------------------
@dataclass
class Actor:
    glyph: str
    colour: tuple[int, int, int]
    name: str
    x: int
    y: int
    hp: int
    max_hp: int
    energy_gain: int
    damage: tuple[int, int]
    accuracy: float = 0.0
    sense: int = 0
    rises: bool = False      # goes down instead of dying, and gets back up
    energy: int = 0
    aware: bool = False
    target: tuple[int, int] | None = None
    patience: int = 0
    downed: bool = False
    rise_in: int = 0
    dead: bool = False

    @property
    def alive(self) -> bool:
        return not self.dead

    @property
    def acting(self) -> bool:
        """On its feet and able to take a turn."""
        return not self.dead and not self.downed

    @property
    def max_energy(self) -> int:
        return self.energy_gain * 2

    def gain(self) -> None:
        self.energy = min(self.energy + self.energy_gain, self.max_energy)

    def can_afford(self, cost: int) -> bool:
        return self.energy >= cost

    def spend(self, cost: int) -> None:
        if not self.can_afford(cost):
            raise ValueError(f"{self.name} cannot afford action cost {cost}")
        self.energy -= cost


def make_player(x: int, y: int) -> Actor:
    return Actor("@", (244, 238, 214), "YOU", x, y, 100, 100, 100, (7, 13))


ENEMY_KINDS = {
    # Shamblers do not die; they go down and get back up unless finished.
    "z": dict(glyph="z", colour=(132, 152, 92), name="SHAMBLER", hp=62,
              energy_gain=60, damage=(7, 13), accuracy=0.70, sense=7, rises=True),
    "d": dict(glyph="d", colour=(152, 112, 72), name="HOUND", hp=34,
              # 140 outran a sprint, which made contact with a hound unbreakable.
              energy_gain=120, damage=(5, 11), accuracy=0.78, sense=9, rises=False),
}


def make_enemy(kind: str, x: int, y: int) -> Actor:
    spec = ENEMY_KINDS[kind]
    return Actor(spec["glyph"], spec["colour"], spec["name"], x, y,
                 spec["hp"], spec["hp"], spec["energy_gain"], spec["damage"],
                 spec["accuracy"], spec["sense"], spec["rises"])


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
@dataclass
class Room:
    x: int
    y: int
    w: int
    h: int

    @property
    def centre(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def area(self) -> int:
        return self.w * self.h

    def cells(self):
        for y in range(self.y, self.y + self.h):
            for x in range(self.x, self.x + self.w):
                yield x, y

    def overlaps(self, other: "Room", pad: int = 1) -> bool:
        return (self.x - pad < other.x + other.w and other.x - pad < self.x + self.w
                and self.y - pad < other.y + other.h and other.y - pad < self.y + self.h)

    def ring(self):
        """The wall cells immediately surrounding the room interior."""
        for x in range(self.x - 1, self.x + self.w + 1):
            yield x, self.y - 1
            yield x, self.y + self.h
        for y in range(self.y, self.y + self.h):
            yield self.x - 1, y
            yield self.x + self.w, y


class Floor:
    """One storey. Generated once, then mutated in place for the whole run."""

    def __init__(self, rng: random.Random, depth: int) -> None:
        self.rng = rng
        self.depth = depth
        self.name = STOREYS[depth]
        self.tiles = [["#"] * FLOOR_W for _ in range(FLOOR_H)]
        self.items: dict[tuple[int, int], str] = {}
        self.rooms: list[Room] = []
        self.enemies: list[Actor] = []
        self.explored: set[tuple[int, int]] = set()
        self.visible: set[tuple[int, int]] = set()
        self.flow: dict[tuple[int, int], int] = {}
        self._carve()
        self._decorate()

    # -- terrain ------------------------------------------------------------
    def _carve(self) -> None:
        rng = self.rng
        for _ in range(400):
            if len(self.rooms) >= 14:
                break
            w, h = rng.randint(5, 11), rng.randint(4, 8)
            x, y = rng.randint(2, FLOOR_W - w - 3), rng.randint(2, FLOOR_H - h - 3)
            room = Room(x, y, w, h)
            if any(room.overlaps(other) for other in self.rooms):
                continue
            self.rooms.append(room)

        for room in self.rooms:
            for cx, cy in room.cells():
                self.tiles[cy][cx] = "."

        corridor: set[tuple[int, int]] = set()
        for previous, room in zip(self.rooms, self.rooms[1:]):
            ax, ay = previous.centre
            bx, by = room.centre
            if rng.random() < 0.5:
                legs = [(ax, bx, ay, True), (ay, by, bx, False)]
            else:
                legs = [(ay, by, ax, False), (ax, bx, by, True)]
            for start, end, fixed, horizontal in legs:
                step = 1 if end >= start else -1
                for value in range(start, end + step, step):
                    cx, cy = (value, fixed) if horizontal else (fixed, value)
                    if self.tiles[cy][cx] == "#":
                        corridor.add((cx, cy))
                    self.tiles[cy][cx] = "."

        # A corridor punching through a room's wall ring is a doorway -- but only
        # where it actually pierces the wall. A corridor merely running alongside
        # a room touches the ring too, and would otherwise become a row of doors.
        doors: set[tuple[int, int]] = set()
        for room in self.rooms:
            for cell in room.ring():
                if cell in corridor and self._is_doorway(*cell):
                    doors.add(cell)

        # Rooms two tiles apart pierce as two back-to-back doorways; keep one.
        for cell in sorted(doors):
            if any(n in doors for n in neighbours(cell)):
                doors.discard(cell)
                self.tiles[cell[1]][cell[0]] = "."

        for x, y in doors:
            self.tiles[y][x] = "+" if rng.random() < 0.65 else "'"

    def _is_doorway(self, x: int, y: int) -> bool:
        if not (0 < x < FLOOR_W - 1 and 0 < y < FLOOR_H - 1):
            return False
        open_h = self.tiles[y][x - 1] != "#" and self.tiles[y][x + 1] != "#"
        open_v = self.tiles[y - 1][x] != "#" and self.tiles[y + 1][x] != "#"
        wall_h = self.tiles[y][x - 1] == "#" and self.tiles[y][x + 1] == "#"
        wall_v = self.tiles[y - 1][x] == "#" and self.tiles[y + 1][x] == "#"
        return (open_h and wall_v) or (open_v and wall_h)

    def _decorate(self) -> None:
        rng = self.rng
        for index, room in enumerate(self.rooms):
            role = rng.choice(["hall", "study", "storage", "gallery", "plain", "plain"])
            cx, cy = room.centre
            if role == "hall":
                for x, y in room.cells():
                    if abs(x - cx) <= room.w // 4 and abs(y - cy) <= room.h // 4:
                        self.tiles[y][x] = '"'
            elif role == "study":
                for _ in range(rng.randint(3, 6)):
                    x, y = self._spot(room)
                    if self.tiles[y][x] == ".":
                        self.tiles[y][x] = "="
            elif role == "storage":
                for _ in range(rng.randint(4, 10)):
                    x, y = self._spot(room)
                    if self.tiles[y][x] == ".":
                        self.tiles[y][x] = ","
            elif role == "gallery":
                for x in range(room.x + 1, room.x + room.w - 1, 3):
                    if self.tiles[room.y][x] == ".":
                        self.tiles[room.y][x] = "T"

            if index and rng.random() < 0.4:
                for _ in range(8):
                    x, y = self._spot(room)
                    if self.tiles[y][x] == ".":
                        self.tiles[y][x] = "*"
                        break

    def _spot(self, room: Room) -> tuple[int, int]:
        return (self.rng.randint(room.x, room.x + room.w - 1),
                self.rng.randint(room.y, room.y + room.h - 1))

    def free_cell(self, exclude: set[tuple[int, int]] | None = None,
                  within: set[tuple[int, int]] | None = None) -> tuple[int, int] | None:
        exclude = exclude or set()
        for _ in range(400):
            room = self.rng.choice(self.rooms)
            cell = self._spot(room)
            if self.tiles[cell[1]][cell[0]] not in (".", ",", '"'):
                continue
            if cell in self.items or cell in exclude or self.actor_at(*cell):
                continue
            if within is not None and cell not in within:
                continue
            return cell
        return None

    # -- queries ------------------------------------------------------------
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < FLOOR_W and 0 <= y < FLOOR_H

    def walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and TILES[self.tiles[y][x]][2]

    def pathable(self, x: int, y: int) -> bool:
        """Walkable, plus closed doors -- anything that can open one may route through."""
        return self.walkable(x, y) or (self.in_bounds(x, y) and self.tiles[y][x] == "+")

    def opaque(self, x: int, y: int) -> bool:
        return not self.in_bounds(x, y) or TILES[self.tiles[y][x]][3]

    def is_cover(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.tiles[y][x] in LOW_COVER

    def actor_at(self, x: int, y: int) -> Actor | None:
        for enemy in self.enemies:
            if enemy.alive and enemy.x == x and enemy.y == y:
                return enemy
        return None

    def reachable(self, origin: tuple[int, int], blocked: set[tuple[int, int]] | None = None,
                  unlocked: bool = False) -> set[tuple[int, int]]:
        """Flood fill over pathable tiles.

        `blocked` treats cells as walls; `unlocked` models carrying the iron key,
        since opening a locked door turns it into floor you can walk through.
        """
        blocked = blocked or set()

        def passable(x: int, y: int) -> bool:
            if (x, y) in blocked:
                return False
            if self.pathable(x, y):
                return True
            return unlocked and self.in_bounds(x, y) and self.tiles[y][x] == "L"

        seen = {origin}
        queue = deque([origin])
        while queue:
            cell = queue.popleft()
            for n in neighbours(cell):
                if n not in seen and passable(*n):
                    seen.add(n)
                    queue.append(n)
        return seen


def neighbours(cell: tuple[int, int]):
    x, y = cell
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


# ---------------------------------------------------------------------------
# the mansion: one persistent world, generated once
# ---------------------------------------------------------------------------
@dataclass
class Fixture:
    """An interactable that gates progress. Solved once, stays solved."""
    kind: str
    depth: int
    x: int
    y: int
    needs_item: str | None = None
    needs_flags: tuple[str, ...] = ()
    grants_item: str | None = None
    grants_flag: str | None = None
    blurb: str = ""
    solved: bool = False


class Mansion:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.floors = [Floor(rng, depth) for depth in range(len(STOREYS))]
        self.stairs: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        self.fixtures: list[Fixture] = []
        self.atrium: Room | None = None
        self.start = self.floors[GROUND].rooms[0].centre
        self._link_stairs()
        self._cut_atrium()
        self._place_progression()
        self._populate()

    def floor(self, depth: int) -> Floor | None:
        return self.floors[depth] if 0 <= depth < len(self.floors) else None

    # -- vertical links -----------------------------------------------------
    def _link_stairs(self) -> None:
        ground = self.floors[GROUND]
        for other, glyph_here, glyph_there in ((GROUND - 1, ">", "<"), (GROUND + 1, "<", ">")):
            here = ground.free_cell(exclude={self.start})
            there = self.floors[other].free_cell()
            if here is None or there is None:
                continue
            ground.tiles[here[1]][here[0]] = glyph_here
            self.floors[other].tiles[there[1]][there[0]] = glyph_there
            self.stairs[(GROUND, *here)] = (other, *there)
            self.stairs[(other, *there)] = (GROUND, *here)

    def _cut_atrium(self) -> None:
        """Open the biggest ground-floor room up to the storey above.

        The upper storey gets a hole ringed by railings; standing at the rail you
        look down into the room below, and from below you can see who is up there.
        """
        ground, upper = self.floors[GROUND], self.floors[GROUND + 1]
        room = max(ground.rooms, key=lambda r: r.area)
        if room.w < 6 or room.h < 5:
            return
        hole = Room(room.x + 1, room.y + 1, room.w - 2, room.h - 2)

        for x, y in hole.cells():
            upper.tiles[y][x] = "O"
        for x, y in hole.ring():
            if not upper.in_bounds(x, y):
                continue
            # Walkway around the drop, with a rail on the inward edge.
            upper.tiles[y][x] = "H"
        for x, y in Room(hole.x - 2, hole.y - 2, hole.w + 4, hole.h + 4).ring():
            if upper.in_bounds(x, y) and upper.tiles[y][x] == "#":
                upper.tiles[y][x] = "."
        # Clear the ground-floor room so the drop looks onto open space.
        for x, y in hole.cells():
            if ground.tiles[y][x] in ("=", "T"):
                ground.tiles[y][x] = "."
        self.atrium = hole

    # -- progression --------------------------------------------------------
    def _place_progression(self) -> None:
        """Author the dependency chain, then place items so it is solvable.

        cellar: BRASS KEY + CODE NOTE (both loose)
          -> upper: safe wants CODE NOTE, grants IRON KEY and the SEAL flag
          -> upper: IRON KEY opens a locked wing holding the FUSE
          -> cellar: fuse box wants FUSE, grants the POWER flag
          -> ground: the front door wants POWER and SEAL
        """
        cellar, ground, upper = self.floors
        cellar_start = self._stair_cell(GROUND - 1)
        upper_start = self._stair_cell(GROUND + 1)

        # The fuse sits behind a door we lock; everything needed to reach the key
        # has to live outside that door, so gate first and place afterwards.
        # The sealed set has to be computed before the tile becomes "L", since a
        # locked door is not pathable and would hide the region it seals.
        sealed = self._lock_a_door(upper, upper_start)
        open_upper = upper.reachable(upper_start)

        spot = next((c for c in sorted(sealed) if upper.tiles[c[1]][c[0]] == "."), None)
        if spot is None:                              # no lockable door found
            spot = upper.free_cell(within=open_upper)
        if spot:
            upper.items[spot] = "f"

        for glyph in ("k", "&"):
            cell = cellar.free_cell()
            if cell:
                cellar.items[cell] = glyph

        safe = upper.free_cell(within=open_upper)
        if safe:
            upper.tiles[safe[1]][safe[0]] = "Ω"
            self.fixtures.append(Fixture(
                "SAFE", GROUND + 1, *safe, needs_item="CODE NOTE",
                grants_item="IRON KEY", grants_flag="SEAL",
                blurb="a wall safe, dial worn smooth"))

        box = cellar.free_cell()
        if box:
            cellar.tiles[box[1]][box[0]] = "Ω"
            self.fixtures.append(Fixture(
                "FUSEBOX", GROUND - 1, *box, needs_item="FUSE",
                grants_flag="POWER", blurb="the fuse box, one socket empty"))

        exit_cell = ground.free_cell(exclude={self.start})
        if exit_cell:
            ground.tiles[exit_cell[1]][exit_cell[0]] = "∩"
            self.fixtures.append(Fixture(
                "FRONTDOOR", GROUND, *exit_cell, needs_flags=("POWER", "SEAL"),
                blurb="the front door, dead bolts and a dead keypad"))

    def _stair_cell(self, depth: int) -> tuple[int, int]:
        for (d, x, y) in self.stairs:
            if d == depth:
                return (x, y)
        return self.floors[depth].rooms[0].centre

    def _lock_a_door(self, floor: Floor, origin: tuple[int, int]) -> set[tuple[int, int]]:
        """Lock a door that seals off a worthwhile chunk, and return what it seals."""
        doors = [(x, y) for y in range(FLOOR_H) for x in range(FLOOR_W)
                 if floor.tiles[y][x] in ("+", "'")]
        self.rng.shuffle(doors)
        whole = floor.reachable(origin)
        for door in doors:
            if door == origin:
                continue
            outside = floor.reachable(origin, blocked={door})
            sealed = whole - outside - {door}
            if 10 <= len(sealed) <= len(whole) // 2:
                floor.tiles[door[1]][door[0]] = "L"
                return sealed
        return set()

    def _populate(self) -> None:
        rng = self.rng
        for depth, floor in enumerate(self.floors):
            # A mansion you re-cross a dozen times cannot be packed: at 25 the
            # walk itself outpaced every medkit in the building.
            count = 3 + depth + rng.randint(0, 2)
            for _ in range(count):
                cell = floor.free_cell(exclude={self.start})
                if cell is None or (depth == GROUND and math.dist(cell, self.start) < 9):
                    continue
                floor.enemies.append(make_enemy("d" if rng.random() < 0.3 else "z", *cell))
            for _ in range(rng.randint(4, 9)):
                cell = floor.free_cell()
                if cell and floor.tiles[cell[1]][cell[0]] == ".":
                    floor.tiles[cell[1]][cell[0]] = "%"
            for glyph in "!!!/aa" + ("!" if depth != GROUND else ""):
                cell = floor.free_cell()
                if cell:
                    floor.items[cell] = glyph

    def solvable(self) -> bool:
        """Walk the dependency chain by reachability. A seed that fails is thrown away.

        Existence is not enough -- an item can generate inside a pocket the
        corridors never reached, so every step is checked against a flood fill
        from where the player actually is when they need it.
        """
        cellar, ground, upper = self.floors
        if {f.kind for f in self.fixtures} != {"SAFE", "FUSEBOX", "FRONTDOOR"}:
            return False

        def usable(cell: tuple[int, int], region: set[tuple[int, int]]) -> bool:
            # Fixtures are solid; you work them from an adjacent tile.
            return cell in region or any(n in region for n in neighbours(cell))

        ground_open = ground.reachable(self.start)
        landings = [(x, y) for (depth, x, y) in self.stairs if depth == GROUND]
        if len(landings) != 2 or any(cell not in ground_open for cell in landings):
            return False

        cellar_open = cellar.reachable(self._stair_cell(GROUND - 1))
        for glyph in ("k", "&"):
            cell = next((c for c, g in cellar.items.items() if g == glyph), None)
            if cell is None or cell not in cellar_open:
                return False

        upper_start = self._stair_cell(GROUND + 1)
        upper_open = upper.reachable(upper_start)
        upper_all = upper.reachable(upper_start, unlocked=True)

        safe = next(f for f in self.fixtures if f.kind == "SAFE")
        if not usable((safe.x, safe.y), upper_open):
            return False                       # the safe grants the key; it cannot be behind it

        fuse = next((c for c, g in upper.items.items() if g == "f"), None)
        if fuse is None or fuse not in upper_all:
            return False

        box = next(f for f in self.fixtures if f.kind == "FUSEBOX")
        door = next(f for f in self.fixtures if f.kind == "FRONTDOOR")
        return usable((box.x, box.y), cellar_open) and (door.x, door.y) in ground_open


# ---------------------------------------------------------------------------
# field of view and flow field
# ---------------------------------------------------------------------------
def compute_fov(floor: Floor, ox: int, oy: int, radius: int) -> set[tuple[int, int]]:
    """Raycast FOV. Coarse, but the viewport is small and it only runs on a turn."""
    visible = {(ox, oy)}
    for index in range(540):
        angle = index * math.tau / 540
        dx, dy = math.cos(angle) * 0.5, math.sin(angle) * 0.5
        x, y = ox + 0.5, oy + 0.5
        for _ in range(radius * 2):
            x += dx
            y += dy
            tx, ty = int(x), int(y)
            if not floor.in_bounds(tx, ty) or math.dist((tx, ty), (ox, oy)) > radius:
                break
            visible.add((tx, ty))
            if floor.opaque(tx, ty):
                break
    return visible


def compute_flow(floor: Floor, origin: tuple[int, int], limit: int = 22) -> dict[tuple[int, int], int]:
    """Breadth-first distance field to the player; enemies just walk downhill."""
    flow = {origin: 0}
    queue = deque([origin])
    while queue:
        cell = queue.popleft()
        distance = flow[cell]
        if distance >= limit:
            continue
        for n in neighbours(cell):
            if n in flow or not floor.pathable(*n):
                continue
            flow[n] = distance + 1
            queue.append(n)
    return flow


# ---------------------------------------------------------------------------
# world
# ---------------------------------------------------------------------------
@dataclass
class World:
    seed: int
    rng: random.Random
    mansion: Mansion
    player: Actor
    depth: int = GROUND
    turn: int = 0
    noise: int = 0
    crouched: bool = False
    ammo: int = 6
    ammo_reserve: int = 10
    flags: set[str] = field(default_factory=set)
    inventory: dict[str, int] = field(default_factory=lambda: {"FIRST AID": 2})
    messages: list[tuple[str, tuple[int, int, int]]] = field(default_factory=list)
    dead: bool = False
    escaped: bool = False

    @property
    def level(self) -> Floor:
        return self.mansion.floors[self.depth]

    def log(self, text: str, colour=UI_TEXT) -> None:
        self.messages.append((text.upper(), colour))
        del self.messages[:-2]

    def carried(self) -> int:
        return sum(self.inventory.values())

    def take(self, name: str) -> None:
        self.inventory[name] = self.inventory.get(name, 0) + 1

    def drop(self, name: str) -> None:
        if self.inventory.get(name):
            self.inventory[name] -= 1
            if not self.inventory[name]:
                del self.inventory[name]

    def fixture_at(self, depth: int, x: int, y: int) -> Fixture | None:
        for fixture in self.mansion.fixtures:
            if (fixture.depth, fixture.x, fixture.y) == (depth, x, y):
                return fixture
        return None


def new_world(seed: int) -> World:
    rng = random.Random(seed)
    for _ in range(12):
        mansion = Mansion(rng)
        if mansion.solvable():
            break
    else:
        raise RuntimeError(f"seed {seed}: could not lay out a solvable mansion")
    world = World(seed=seed, rng=rng, mansion=mansion,
                  player=make_player(*mansion.start))
    world.log("the door locks behind you. ? for controls.", UI_DIM)
    refresh_senses(world)
    return world


def refresh_senses(world: World) -> None:
    floor = world.level
    player = world.player
    floor.visible = compute_fov(floor, player.x, player.y, SIGHT_RADIUS)
    floor.explored |= floor.visible
    floor.flow = compute_flow(floor, (player.x, player.y))


# ---------------------------------------------------------------------------
# stance and cover
# ---------------------------------------------------------------------------
def behind_cover(floor: Floor, x: int, y: int) -> bool:
    return any(floor.is_cover(*n) for n in neighbours((x, y)))


def player_concealed(world: World) -> bool:
    """Crouched and tucked against something low: hard to see, hard to hit."""
    return world.crouched and behind_cover(world.level, world.player.x, world.player.y)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ranged_chance(world: World, target: Actor) -> float:
    distance = math.dist((target.x, target.y), (world.player.x, world.player.y))
    chance = 0.86 - 0.055 * max(0.0, distance - 2.0)
    if world.crouched:
        chance += 0.12                                   # braced
    if behind_cover(world.level, target.x, target.y):
        chance -= 0.22
    if target.downed:
        chance += 0.20
    return clamp(chance, 0.15, 0.95)


def melee_chance(world: World, target: Actor) -> float:
    if target.downed:
        return 1.0
    chance = 0.74
    if target.name == "HOUND":
        chance -= 0.16                                   # fast and low
    if world.crouched:
        chance -= 0.12                                   # awkward swing
    return clamp(chance, 0.2, 0.95)


def enemy_chance(world: World, enemy: Actor) -> float:
    chance = enemy.accuracy
    if player_concealed(world):
        chance -= 0.26
    elif world.crouched:
        chance -= 0.10
    return clamp(chance, 0.15, 0.95)


def sense_radius(world: World, enemy: Actor) -> float:
    if player_concealed(world):
        return enemy.sense * 0.45
    if world.crouched:
        return enemy.sense * 0.7
    return float(enemy.sense)


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------
def stain(floor: Floor, x: int, y: int) -> None:
    """Leave gore, but only on plain floor.

    Painting it over whatever was there deleted staircases, fixtures and doors
    when something died on one, which could strand a run permanently.
    """
    if floor.tiles[y][x] in (".", ",", '"'):
        floor.tiles[y][x] = "%"


def damage_roll(actor: Actor, rng: random.Random) -> int:
    return rng.randint(*actor.damage)


def make_noise(world: World, loudness: int, radius: int) -> None:
    """Sound off, pulling nearby enemies toward you.

    Loudness is the current volume, not a running total -- accumulating it pinned
    the meter at maximum after a dozen ordinary steps, so walking read as loud as
    gunfire and the HUD gauge told you nothing.
    """
    world.noise = max(world.noise, loudness)
    player = world.player
    for enemy in world.level.enemies:
        if enemy.acting and math.dist((enemy.x, enemy.y), (player.x, player.y)) <= radius:
            enemy.aware = True
            enemy.target = (player.x, player.y)
            enemy.patience = 12


def tick(world: World) -> None:
    """Hand out energy until somebody can act, then let the enemies spend it."""
    world.player.gain()
    for enemy in world.level.enemies:
        if enemy.acting:
            enemy.gain()
        elif enemy.downed and not enemy.dead:
            enemy.rise_in -= 1
            if enemy.rise_in <= 0:
                rise(world, enemy)
    world.turn += 1
    if world.turn % 2 == 0 and world.noise > 0:
        world.noise -= 1
    resolve_enemies(world)


def rise(world: World, enemy: Actor) -> None:
    enemy.downed = False
    enemy.hp = max(1, int(enemy.max_hp * 0.4))
    enemy.energy = 0
    enemy.aware = True
    enemy.target = (world.player.x, world.player.y)
    enemy.patience = 12
    if (enemy.x, enemy.y) in world.level.visible:
        world.log(f"the {enemy.name.lower()} gets back up.", UI_HOT)
    else:
        world.log("something drags itself upright.", UI_DIM)


def perform(world: World, cost: int, action) -> None:
    """Advance time until the player can afford the action, then take it."""
    player = world.player
    guard = 0
    while not player.can_afford(cost) and not world.dead:
        tick(world)
        guard += 1
        if guard > 64:
            break
    if world.dead:
        return
    player.spend(cost)
    action()
    refresh_senses(world)
    resolve_enemies(world)
    if world.player.hp <= 0:
        world.dead = True
        world.log("you die in the dark.", UI_HOT)


def try_move(world: World, dx: int, dy: int, sprint: bool = False) -> None:
    floor = world.level
    player = world.player
    tx, ty = player.x + dx, player.y + dy

    target = floor.actor_at(tx, ty)
    if target is not None:
        if target.downed:
            perform(world, COST_FINISH, lambda: finish(world, target))
        else:
            perform(world, COST_MELEE, lambda: melee(world, target))
        return

    if floor.in_bounds(tx, ty) and floor.tiles[ty][tx] == "+":
        perform(world, COST_DOOR, lambda: open_door(world, tx, ty))
        return

    if floor.in_bounds(tx, ty) and floor.tiles[ty][tx] == "L":
        perform(world, COST_DOOR, lambda: unlock_door(world, tx, ty))
        return

    if not floor.walkable(tx, ty):
        if floor.is_cover(tx, ty) and not world.crouched:
            world.log("too high to climb. crouch to use it as cover.", UI_DIM)
        return

    if sprint and world.crouched:
        world.log("you cannot sprint from a crouch.", UI_DIM)
        return

    cost = COST_SPRINT if sprint else (COST_CROUCH_MOVE if world.crouched else COST_MOVE)
    steps = 2 if sprint else 1
    perform(world, cost, lambda: walk(world, dx, dy, steps, sprint))


def walk(world: World, dx: int, dy: int, steps: int, sprint: bool) -> None:
    floor = world.level
    player = world.player
    for _ in range(steps):
        tx, ty = player.x + dx, player.y + dy
        if not floor.walkable(tx, ty) or floor.actor_at(tx, ty):
            break
        player.x, player.y = tx, ty
    if sprint:
        make_noise(world, 6, 9)
    elif not world.crouched:
        make_noise(world, 2, 3)
    pick_up(world)


def toggle_crouch(world: World) -> None:
    world.crouched = not world.crouched
    if world.crouched:
        cover = behind_cover(world.level, world.player.x, world.player.y)
        world.log("you drop into a crouch." + (" good cover here." if cover else ""),
                  UI_WARM if cover else UI_DIM)
    else:
        world.log("you straighten up.", UI_DIM)


def pick_up(world: World) -> None:
    floor = world.level
    cell = (world.player.x, world.player.y)
    glyph = floor.items.get(cell)
    if glyph is None:
        return
    name = ITEMS[glyph][2]
    if world.carried() >= 8 and glyph != "a":
        world.log(f"no room for the {name.lower()}.", UI_DIM)
        return

    # Stooping costs on top of the move that carried you here.
    world.player.energy = max(0, world.player.energy - COST_PICKUP)
    del floor.items[cell]
    if glyph == "a":
        world.ammo_reserve += 4
        world.log("you pocket four rounds.", UI_WARM)
    else:
        world.take(name)
        world.log(f"picked up {name.lower()}.", UI_WARM)


def open_door(world: World, x: int, y: int) -> None:
    world.level.tiles[y][x] = "'"
    make_noise(world, 4, 6)
    world.log("the hinges shriek.", UI_DIM)


def unlock_door(world: World, x: int, y: int) -> None:
    if world.inventory.get("IRON KEY"):
        world.level.tiles[y][x] = "'"
        world.log("the iron key turns. the wing is open.", UI_WARM)
    else:
        world.log("locked. a heavy iron keyway.", UI_HOT)


def melee(world: World, target: Actor) -> None:
    weapon = "CROWBAR" if world.inventory.get("CROWBAR") else None
    make_noise(world, 5, 6)
    if world.rng.random() >= melee_chance(world, target):
        world.log(f"you swing wide of the {target.name.lower()}.", UI_DIM)
        return
    bonus = 7 if weapon else 0
    target.hp -= damage_roll(world.player, world.rng) + bonus
    resolve_damage(world, target, "you strike the")


def finish(world: World, target: Actor) -> None:
    target.dead = True
    target.downed = False
    stain(world.level, target.x, target.y)
    make_noise(world, 5, 6)
    world.log(f"you finish the {target.name.lower()}. it stays down.", UI_WARM)


def resolve_damage(world: World, target: Actor, verb: str) -> None:
    if target.hp > 0:
        world.log(f"{verb} {target.name.lower()}.", UI_TEXT)
        return
    if target.rises:
        target.downed = True
        target.hp = 0
        target.rise_in = world.rng.randint(18, 34)
        world.log(f"the {target.name.lower()} goes down. it is not finished.", UI_WARM)
    else:
        target.dead = True
        stain(world.level, target.x, target.y)
        world.log(f"the {target.name.lower()} drops.", UI_WARM)


def fire(world: World) -> None:
    floor = world.level
    if world.ammo <= 0:
        world.log("the hammer falls on an empty chamber.", UI_HOT)
        return
    world.ammo -= 1
    make_noise(world, 10, 16)

    targets = [e for e in floor.enemies if e.alive and (e.x, e.y) in floor.visible]
    if not targets:
        world.log("you fire into the dark. nothing.", UI_DIM)
        return
    target = min(targets, key=lambda e: math.dist((e.x, e.y), (world.player.x, world.player.y)))
    if world.rng.random() >= ranged_chance(world, target):
        world.log(f"the shot goes wide. plaster and dust.", UI_DIM)
        return
    target.hp -= world.rng.randint(15, 26)
    resolve_damage(world, target, "the round tears into the")


def reload_weapon(world: World) -> None:
    if world.ammo >= 6:
        world.log("the cylinder is full.", UI_DIM)
        return
    if world.ammo_reserve <= 0:
        world.log("no rounds left.", UI_HOT)
        return
    loaded = min(6 - world.ammo, world.ammo_reserve)
    world.ammo += loaded
    world.ammo_reserve -= loaded
    world.log(f"you load {loaded} rounds.", UI_WARM)


def use_first_aid(world: World) -> None:
    if not world.inventory.get("FIRST AID"):
        world.log("nothing to bind the wound with.", UI_HOT)
        return
    world.drop("FIRST AID")
    world.player.hp = min(world.player.max_hp, world.player.hp + 40)
    world.log("you bind the wound. it holds.", UI_WARM)


def interact(world: World) -> None:
    """Stairs, fixtures, the way out -- whatever is under your feet."""
    floor = world.level
    player = world.player
    here = floor.tiles[player.y][player.x]

    link = world.mansion.stairs.get((world.depth, player.x, player.y))
    if link is not None:
        world.depth, player.x, player.y = link
        world.log(f"you take the stairs. {world.level.name.lower()}.", UI_WARM)
        return

    fixture = world.fixture_at(world.depth, player.x, player.y)
    if fixture is None:
        for cell in neighbours((player.x, player.y)):
            fixture = world.fixture_at(world.depth, *cell)
            if fixture:
                break
    if fixture is not None:
        solve(world, fixture)
        return

    # Shutting a door behind you is the one way to break contact: the pursuer
    # has to spend 75 shouldering it open again, which is your head start.
    for cx, cy in neighbours((player.x, player.y)):
        if floor.in_bounds(cx, cy) and floor.tiles[cy][cx] == "'" and not floor.actor_at(cx, cy):
            floor.tiles[cy][cx] = "+"
            make_noise(world, 3, 4)
            world.log("you pull the door shut.", UI_WARM)
            return
    world.log("nothing here to work with.", UI_DIM)


def solve(world: World, fixture: Fixture) -> None:
    if fixture.solved:
        world.log("already dealt with.", UI_DIM)
        return
    missing = [f for f in fixture.needs_flags if f not in world.flags]
    if missing:
        world.log(f"{fixture.blurb}. still dead: {', '.join(missing).lower()}.", UI_HOT)
        return
    if fixture.needs_item and not world.inventory.get(fixture.needs_item):
        world.log(f"{fixture.blurb}. you need the {fixture.needs_item.lower()}.", UI_HOT)
        return

    fixture.solved = True
    if fixture.needs_item:
        world.drop(fixture.needs_item)
    if fixture.grants_item:
        world.take(fixture.grants_item)
    if fixture.grants_flag:
        world.flags.add(fixture.grants_flag)

    if fixture.kind == "FRONTDOOR":
        world.escaped = True
        world.log("the bolts draw back. you are out.", UI_WARM)
    elif fixture.kind == "SAFE":
        world.log("the safe opens. an iron key, and a name you know.", UI_WARM)
    else:
        world.log("the fuse seats. somewhere, the house wakes up.", UI_WARM)


# ---------------------------------------------------------------------------
# enemy turns
# ---------------------------------------------------------------------------
def resolve_enemies(world: World) -> None:
    for enemy in world.level.enemies:
        guard = 0
        while enemy.acting and not world.dead and enemy_turn(world, enemy):
            guard += 1
            if guard > 8:
                break


def enemy_turn(world: World, enemy: Actor) -> bool:
    """Take one action. Returns False when the enemy banks energy instead.

    Banking rather than burning the surplus on a wait is what makes speed mean
    anything: a shambler gaining 60 has to sit out a tick to afford a 100 move.
    """
    floor = world.level
    player = world.player
    here = (enemy.x, enemy.y)
    distance = math.dist(here, (player.x, player.y))

    # Player FOV doubles as the enemy's: if you can see it, it can see you --
    # unless you are low and tucked in, which is what crouching buys.
    if here in floor.visible and distance <= sense_radius(world, enemy):
        enemy.aware = True
        enemy.target = (player.x, player.y)
        enemy.patience = 12

    if distance <= 1.5 and enemy.aware:
        if not enemy.can_afford(COST_MELEE):
            return False
        enemy.spend(COST_MELEE)
        if world.rng.random() >= enemy_chance(world, enemy):
            world.log(f"the {enemy.name.lower()} lunges and misses.", UI_DIM)
            return True
        player.hp -= damage_roll(enemy, world.rng)
        world.log(f"the {enemy.name.lower()} tears into you.", UI_HOT)
        if player.hp <= 0:
            world.dead = True
        return True

    step = None
    if enemy.aware:
        # Downhill on the flow field when the player is reachable, otherwise
        # grope toward the last known position.
        current = floor.flow.get(here)
        if current is not None:
            options = [(floor.flow.get(n, 999), n) for n in neighbours(here)
                       if floor.pathable(*n) and not floor.actor_at(*n)]
            options = [o for o in options if o[0] < current]
            if options:
                step = min(options)[1]
        if step is None and enemy.target:
            step = grope(floor, here, enemy.target)
    elif world.rng.random() < 0.6:
        step = world.rng.choice(list(neighbours(here)))

    if step is None or floor.actor_at(*step) or step == (player.x, player.y):
        if not enemy.can_afford(COST_WAIT):
            return False
        enemy.spend(COST_WAIT)
        return True

    if floor.tiles[step[1]][step[0]] == "+":
        # Doors are a delay for the enemy, not a wall -- and they stay open.
        if not enemy.can_afford(COST_DOOR):
            return False
        enemy.spend(COST_DOOR)
        floor.tiles[step[1]][step[0]] = "'"
        if step in floor.visible:
            world.log(f"the {enemy.name.lower()} shoulders a door open.", UI_HOT)
        else:
            world.log("a door swings open somewhere.", UI_DIM)
    elif floor.walkable(*step):
        if not enemy.can_afford(COST_MOVE):
            return False
        enemy.spend(COST_MOVE)
        enemy.x, enemy.y = step
    else:
        if not enemy.can_afford(COST_WAIT):
            return False
        enemy.spend(COST_WAIT)

    if enemy.aware:
        enemy.patience -= 1
        if enemy.patience <= 0:
            enemy.aware = False
            enemy.target = None
    return True


def grope(floor: Floor, here: tuple[int, int], target: tuple[int, int]):
    best = None
    for option in neighbours(here):
        if not floor.walkable(*option) or floor.actor_at(*option):
            continue
        score = math.dist(option, target)
        if best is None or score < best[0]:
            best = (score, option)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def scale_colour(colour, amount: float):
    return tuple(min(255, max(0, int(channel * amount))) for channel in colour)


def light_at(world: World, x: int, y: int, flicker: float) -> float:
    player = world.player
    total = 1.0 - (math.dist((x, y), (player.x, player.y)) / (LANTERN_RADIUS * flicker)) ** 1.15
    total = max(0.0, total)
    for cx, cy in candles(world.level):
        distance = math.dist((x, y), (cx, cy))
        if distance < CANDLE_RADIUS:
            total += 0.7 * (1.0 - distance / CANDLE_RADIUS)
    return min(1.0, total)


def candles(floor: Floor) -> list[tuple[int, int]]:
    cached = getattr(floor, "_candles", None)
    if cached is None:
        cached = [(x, y) for y in range(FLOOR_H) for x in range(FLOOR_W)
                  if floor.tiles[y][x] == "*"]
        floor._candles = cached
    return cached


def dither(value: float, x: int, y: int) -> float:
    """Ordered-dither a brightness into LIGHT_LEVELS bands."""
    threshold = (BAYER[y % 4][x % 4] + 0.5) / 16.0
    return min(1.0, math.floor(value * LIGHT_LEVELS + threshold) / LIGHT_LEVELS)


def camera(world: World) -> tuple[int, int]:
    cx = max(0, min(world.player.x - VIEW_W // 2, FLOOR_W - VIEW_W))
    cy = max(0, min(world.player.y - VIEW_H // 2, FLOOR_H - VIEW_H))
    return cx, cy


def shade(colour, lit: float):
    if lit < 0.05:
        grey = sum(colour) / 3.0
        return (int(grey * 0.7), int(grey * 0.8), int(grey * 1.0))
    warmth = lit * 0.3
    return tuple(int(c * (1.0 - warmth) + tint * warmth)
                 for c, tint in zip(colour, (255, 226, 186)))


def draw_map(surface, font: Font, world: World, flicker: float, debug: bool) -> None:
    floor = world.level
    below = world.mansion.floor(world.depth - 1)
    above = world.mansion.floor(world.depth + 1)
    ox, oy = camera(world)

    for sy in range(VIEW_H):
        for sx in range(VIEW_W):
            x, y = ox + sx, oy + sy
            if (x, y) not in floor.explored:
                continue
            seen = (x, y) in floor.visible

            if floor.tiles[y][x] == "O":
                if seen and below is not None:
                    draw_through(surface, font, below, x, y, MAP_X + sx, MAP_Y + sy)
                continue

            lit = max(light_at(world, x, y, flicker), AMBIENT) if seen else 0.0
            level_value = dither(lit if seen else MEMORY_LEVEL, x, y)
            if level_value <= 0.0:
                continue

            glyph, colour, _, _ = TILES[floor.tiles[y][x]]
            if (x, y) in floor.items and seen:
                glyph, colour, _ = ITEMS[floor.items[(x, y)]]
            colour = shade(colour, lit)
            font.blit(surface, glyph, MAP_X + sx, MAP_Y + sy, scale_colour(colour, level_value))

            # Standing under the atrium you can see who is on the balcony.
            if seen and above is not None and above.tiles[y][x] == "O":
                overhead = above.actor_at(x, y)
                if overhead and overhead.alive:
                    font.blit(surface, overhead.glyph, MAP_X + sx, MAP_Y + sy, OTHER_STOREY)

    for actor in [*floor.enemies, world.player]:
        if not actor.alive or (actor.x, actor.y) not in floor.visible:
            continue
        sx, sy = actor.x - ox, actor.y - oy
        if not (0 <= sx < VIEW_W and 0 <= sy < VIEW_H):
            continue
        lit = light_at(world, actor.x, actor.y, flicker)
        level_value = dither(max(lit, 0.45), actor.x, actor.y)
        glyph, colour = actor.glyph, actor.colour
        if actor.downed:
            glyph, colour = "%", (128, 46, 44)
        elif lit < 0.25 and actor is not world.player:
            # Outside the lantern things read as silhouettes, not sprites.
            colour = (int(colour[0] * 0.85), int(colour[1] * 0.6), int(colour[2] * 0.7))
        font.blit(surface, glyph, MAP_X + sx, MAP_Y + sy, scale_colour(colour, level_value))
        if debug and actor.aware and actor.acting:
            font.blit(surface, "!", MAP_X + sx, MAP_Y + sy - 1, UI_HOT)


def draw_through(surface, font: Font, below: Floor, x: int, y: int, sx: int, sy: int) -> None:
    """Look down a shaft: the storey underneath, cold and flattened."""
    glyph, _, _, _ = TILES[below.tiles[y][x]]
    if below.tiles[y][x] == "O":
        return
    font.blit(surface, glyph, sx, sy, scale_colour(OTHER_STOREY, 0.75))
    actor = below.actor_at(x, y)
    if actor and actor.alive:
        font.blit(surface, actor.glyph if not actor.downed else "%", sx, sy, OTHER_STOREY)


def bar(value: int, maximum: int, width: int = 10) -> str:
    filled = max(0, min(width, int(round(width * value / maximum))))
    return "█" * filled + "░" * (width - filled)


def threat_level(world: World) -> tuple[str, tuple[int, int, int]]:
    aware = sum(1 for e in world.level.enemies if e.acting and e.aware)
    if aware >= 3:
        return "HIGH", UI_HOT
    if aware:
        return "CLOSE", UI_WARM
    return "QUIET", UI_DIM


def draw_hud(surface, font: Font, world: World, tick_count: int, debug: bool) -> None:
    player = world.player
    font.text(surface, "NIGHTSHIFT".ljust(14) + world.level.name, 0, 0, UI_DIM)
    font.text(surface, f"SEED {world.seed}"[:PANEL_W], PANEL_X, 0, UI_DIM)

    for y in range(0, LOG_Y - 1):
        font.blit(surface, "│", DIVIDER_X, y, UI_DIM)
    font.text(surface, "─" * COLS, 0, LOG_Y - 1, UI_DIM)
    font.blit(surface, "┴", DIVIDER_X, LOG_Y - 1, UI_DIM)

    rule = "─" * PANEL_W
    threat, threat_colour = threat_level(world)
    cover = player_concealed(world)
    panel: list[tuple[str, tuple[int, int, int]]] = [
        ("   STATUS", UI_TEXT),
        (rule, UI_DIM),
        (f"HP {bar(player.hp, player.max_hp)}", (176, 62, 58)),
        (f"{player.hp:>7}/{player.max_hp}", UI_TEXT),
        (f"EN {bar(player.energy, player.max_energy)}", (120, 174, 168)),
        (f"{player.energy:>7}/{player.max_energy}", UI_TEXT),
        (rule, UI_DIM),
        (f"STANCE {'CROUCH' if world.crouched else ' STAND'}",
         UI_WARM if world.crouched else UI_DIM),
        (f"COVER  {'HIDDEN' if cover else '  OPEN'}", UI_WARM if cover else UI_DIM),
        (rule, UI_DIM),
        (f")  AMMO  {world.ammo}/ 6", UI_WARM if world.ammo else UI_HOT),
        (f"   SPARE  {world.ammo_reserve:>3}", UI_TEXT),
        (rule, UI_DIM),
        (f"CARRY    {world.carried()}/8", UI_DIM),
    ]
    for name, count in list(world.inventory.items())[:4]:
        prefix = f"{count}x" if count > 1 else "-"
        panel.append((f"{prefix} {name}"[:PANEL_W], UI_TEXT))
    panel.append((rule, UI_DIM))
    panel.append((f"THREAT {threat:>6}", threat_colour))

    for offset, (line, colour) in enumerate(panel[:LOG_Y - 2]):
        font.text(surface, line[:PANEL_W], PANEL_X, MAP_Y + offset, colour)

    for offset, (line, colour) in enumerate(world.messages[-2:]):
        font.text(surface, f"> {line}"[:COLS], 0, LOG_Y + offset, colour)

    if debug:
        floor = world.level
        up = sum(1 for e in floor.enemies if e.acting)
        down = sum(1 for e in floor.enemies if e.downed and not e.dead)
        footer = f"UP {up}  DOWN {down}  TURN {world.turn}  FLOW {len(floor.flow)}"
    else:
        power = "ON" if "POWER" in world.flags else "--"
        seal = "ON" if "SEAL" in world.flags else "--"
        footer = f"NOISE {bar(world.noise, 10, 5)}  POWER {power}  SEAL {seal}  ? KEYS"
    font.text(surface, footer[:COLS], 0, LOG_Y + 2, UI_DIM)

    if world.dead:
        banner(surface, font, "YOU DIED", "N FOR A NEW MANSION", tick_count)
    elif world.escaped:
        banner(surface, font, "YOU ARE OUT", f"{world.turn} TURNS", tick_count)


HELP_LINES = [
    ("WASD / ARROWS", "MOVE 100"),
    ("  + SHIFT", "SPRINT 150"),
    ("C  CROUCH", "25"),
    ("BUMP ENEMY", "ATTACK 100"),
    ("BUMP DOWNED", "FINISH 150"),
    ("BUMP DOOR", "OPEN 75"),
    ("E  INTERACT", "100"),
    ("F  FIRE", "100"),
    ("R  RELOAD", "125"),
    ("Q  FIRST AID", "150"),
    ("N", "NEW MANSION"),
    ("F1", "DEBUG OVERLAY"),
    ("ESC", "QUIT"),
]
# 13 entries + title + 2 rules + 2 notes = 18 lines, which is exactly VIEW_H - 2.
# Anything more gets silently clipped by draw_help.


def draw_help(surface, font: Font) -> None:
    inner = VIEW_W - 2
    body = [f"{left:<{inner - len(right)}}{right}" if left else "─" * inner
            for left, right in HELP_LINES]
    body = ["CONTROLS", "─" * inner, *body, "─" * inner,
            "CROUCH BY COVER TO HIDE.", "E ALSO SHUTS DOORS."]

    body = body[:VIEW_H - 2]          # never overflow the viewport
    height = len(body) + 2
    top = MAP_Y + max(0, (VIEW_H - height) // 2)
    surface.fill(BLACK, pygame.Rect(MAP_X * CELL, top * CELL, VIEW_W * CELL, height * CELL))

    font.text(surface, "┌" + "─" * inner + "┐", MAP_X, top, UI_DIM)
    for offset, line in enumerate(body, start=1):
        font.blit(surface, "│", MAP_X, top + offset, UI_DIM)
        font.blit(surface, "│", MAP_X + VIEW_W - 1, top + offset, UI_DIM)
        colour = UI_WARM if offset == 1 else UI_TEXT
        font.text(surface, line[:inner], MAP_X + 1, top + offset, colour)
    font.text(surface, "└" + "─" * inner + "┘", MAP_X, top + height - 1, UI_DIM)


def banner(surface, font: Font, title: str, hint: str, tick_count: int) -> None:
    y = MAP_Y + VIEW_H // 2 - 1
    font.text(surface, title, (VIEW_W - len(title)) // 2, y,
              UI_HOT if (tick_count // 20) % 2 == 0 else UI_WARM)
    font.text(surface, hint, (VIEW_W - len(hint)) // 2, y + 2, UI_DIM)


def make_scanlines(size) -> pygame.Surface:
    surface = pygame.Surface(size).convert()
    surface.fill((255, 255, 255))
    for y in range(0, size[1], 2):
        pygame.draw.line(surface, (206, 206, 212), (0, y), (size[0], y))
    return surface


def make_vignette(size) -> pygame.Surface:
    w, h = size
    surface = pygame.Surface(size).convert()
    cx, cy = w / 2, h / 2
    longest = math.hypot(cx, cy)
    for y in range(h):
        row = ((y - cy) / longest) ** 2
        for x in range(0, w, 8):
            distance = math.sqrt(row + ((x - cx) / longest) ** 2)
            amount = max(0.0, 1.0 - 0.40 * distance**1.8)
            shade_value = int(255 * amount)
            surface.fill((shade_value, shade_value, shade_value), (x, y, 8, 1))
    return surface


def make_grain(size, rng: random.Random) -> pygame.Surface:
    surface = pygame.Surface(size).convert()
    for y in range(size[1]):
        for x in range(0, size[0], 2):
            value = rng.randint(0, 9)
            surface.fill((value, value, value), (x, y, 2, 1))
    return surface


def render_frame(font, internal, scanlines, vignette, grain, world, tick_count,
                 debug=False, show_help=False) -> pygame.Surface:
    flicker = 1.0 + 0.045 * math.sin(tick_count * 0.19) + 0.02 * math.sin(tick_count * 0.71)

    internal.fill(BLACK)
    draw_map(internal, font, world, flicker, debug)
    draw_hud(internal, font, world, tick_count, debug)
    if show_help:
        draw_help(internal, font)

    offset = (tick_count * 7) % grain.get_height()
    internal.blit(grain, (0, -offset), special_flags=pygame.BLEND_RGB_ADD)
    internal.blit(grain, (0, grain.get_height() - offset), special_flags=pygame.BLEND_RGB_ADD)

    # The post-pass surfaces were built for the window, so they define the size.
    frame = pygame.transform.scale(internal, scanlines.get_size())
    frame.blit(scanlines, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
    frame.blit(vignette, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
    return frame


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------
DIRECTIONS = {
    pygame.K_LEFT: (-1, 0), pygame.K_a: (-1, 0),
    pygame.K_RIGHT: (1, 0), pygame.K_d: (1, 0),
    pygame.K_UP: (0, -1), pygame.K_w: (0, -1),
    pygame.K_DOWN: (0, 1), pygame.K_s: (0, 1),
}


def pick_geometry(cell: int | None, scale: int | None) -> tuple[int, int]:
    """Largest (cell, upscale) pair from SIZE_LADDER that fits the desktop.

    Upscale alone jumps 640 -> 1280 with nothing usable between, so the ladder
    varies the cell size too. Both stay whole numbers: nearest-neighbour only
    looks right at integer multiples.
    """
    if cell or scale:
        return cell or DEFAULT_CELL, scale or 1
    try:
        desktop_w, desktop_h = pygame.display.get_desktop_sizes()[0]
    except (AttributeError, IndexError, pygame.error):
        info = pygame.display.Info()
        desktop_w, desktop_h = info.current_w, info.current_h
    # Title bar, borders, and a taskbar all eat into what a window can claim.
    avail_w, avail_h = desktop_w - 32, desktop_h - 96
    best = SIZE_LADDER[0]
    for rung in SIZE_LADDER:
        if COLS * rung[0] * rung[1] <= avail_w and ROWS * rung[0] * rung[1] <= avail_h:
            best = rung
    return best


def is_help_key(event) -> bool:
    """'?' is shift+/ on US layouts but its own key elsewhere; accept either."""
    return getattr(event, "unicode", "") == "?" or event.key == pygame.K_QUESTION


def handle_key(world: World, event) -> World:
    if event.key == pygame.K_n:
        return new_world(world.rng.randrange(10**5))

    if world.dead or world.escaped:
        return world

    if event.key in DIRECTIONS:
        dx, dy = DIRECTIONS[event.key]
        try_move(world, dx, dy, bool(event.mod & pygame.KMOD_SHIFT))
    elif event.key == pygame.K_c:
        perform(world, COST_STANCE, lambda: toggle_crouch(world))
    elif event.key == pygame.K_e:
        perform(world, COST_INTERACT, lambda: interact(world))
    elif event.key == pygame.K_SPACE:
        perform(world, COST_WAIT, lambda: None)
    elif event.key == pygame.K_f:
        perform(world, COST_FIRE, lambda: fire(world))
    elif event.key == pygame.K_r:
        perform(world, COST_RELOAD, lambda: reload_weapon(world))
    elif event.key == pygame.K_q:
        perform(world, COST_HEAL, lambda: use_first_aid(world))
    return world


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Nightshift prototype")
    parser.add_argument("--out", type=Path, help="render one frame headless and exit")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scale", type=int, default=None,
                        help="integer upscale factor (default: largest that fits the desktop)")
    parser.add_argument("--cell", type=int, default=None,
                        help=f"glyph cell size in px, e.g. 16 or 24 (default {DEFAULT_CELL})")
    args = parser.parse_args()

    if args.out:
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    pygame.init()
    # CELL is read by every draw call, so the chosen cell has to land before the
    # font is sliced and the internal surface is made.
    global CELL
    CELL, scale = pick_geometry(args.cell, args.scale)
    size = (COLS * CELL * scale, ROWS * CELL * scale)
    pygame.display.set_mode(size)
    pygame.display.set_caption("Project Nightshift")
    print(f"{size[0]}x{size[1]}  (cell {CELL}, scale {scale}) -- override with --cell N / --scale N")

    seed = args.seed if args.seed is not None else random.randrange(10**5)
    world = new_world(seed)

    font = Font(SHEET)
    internal = pygame.Surface((COLS * CELL, ROWS * CELL)).convert()
    scanlines = make_scanlines(size)
    vignette = make_vignette(size)
    grain = make_grain((COLS * CELL, ROWS * CELL), random.Random(seed))

    if args.out:
        frame = render_frame(font, internal, scanlines, vignette, grain, world, 12)
        pygame.image.save(frame, str(args.out))
        pygame.quit()
        print(f"wrote {args.out} (seed {seed})")
        return

    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()
    tick_count = 0
    debug = False
    show_help = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if show_help:
                    # Any key dismisses, so Escape closes the list before it quits.
                    show_help = False
                elif is_help_key(event):
                    show_help = True
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    debug = not debug
                else:
                    world = handle_key(world, event)
        screen.blit(render_frame(font, internal, scanlines, vignette, grain,
                                 world, tick_count, debug, show_help), (0, 0))
        pygame.display.flip()
        tick_count += 1
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
