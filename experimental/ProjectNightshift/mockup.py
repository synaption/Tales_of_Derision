"""Playable prototype for Project Nightshift.

Draws with the MSX-UnDeadPeopleEdition CP437 sheet on a low-resolution internal
surface, and runs the Phase 1 core underneath it: a seeded room-and-corridor
generator, raycast field of view with tile memory, the energy scheduler, and
enemies that chase down a flow field.

    python mockup.py                 # play
    python mockup.py --seed 18421    # fixed seed
    python mockup.py --out shot.png  # headless screenshot

Keys: WASD/arrows move (bump to attack, bump a door to open it), shift+dir
sprint, space wait, F fire, R reload, Q first aid, N new level, F1 debug,
Escape quit.
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
CELL = 16                # internal-surface cell size
COLS, ROWS = 40, 25      # internal surface in cells -> 640x400
MAX_SCALE = 4            # nearest-neighbour upscale ceiling; actual pick fits the desktop

VIEW_W, VIEW_H = 26, 20  # map viewport, top-left at (0, 1)
MAP_X, MAP_Y = 0, 1
DIVIDER_X = 26
PANEL_X, PANEL_W = 27, 13
LOG_Y = 22

LEVEL_W, LEVEL_H = 56, 38

# --- palette ---------------------------------------------------------------
# Desaturated, slightly sick. Everything else is this multiplied by light.
BLACK = (0, 0, 0)
UI_DIM = (104, 116, 100)
UI_TEXT = (172, 188, 156)
UI_HOT = (198, 78, 56)
UI_WARM = (196, 172, 116)

# --- tiles -----------------------------------------------------------------
# char -> (cp437 glyph, colour, walkable, opaque)
TILES = {
    "#": ("█", (114, 106, 94), False, True),      # wall
    ".": (".", (96, 92, 90), True, False),      # floorboards
    ",": ("░", (88, 78, 62), True, False),      # debris
    '"': ("░", (132, 58, 52), True, False),     # carpet
    "=": ("≡", (150, 110, 62), False, False),   # furniture
    "T": ("Φ", (150, 146, 132), False, True),   # statue
    "+": ("+", (158, 116, 62), False, True),    # closed door
    "'": ("'", (128, 98, 56), True, False),     # open door
    "*": ("☼", (236, 194, 112), True, False),   # candle
    "%": ("%", (124, 28, 30), True, False),     # gore
    "<": ("<", (172, 172, 182), True, False),   # entrance
    ">": (">", (172, 172, 182), True, False),   # stairs down
}

# char -> (cp437 glyph, colour, display name)
ITEMS = {
    "!": ("!", (198, 62, 58), "FIRST AID"),
    "/": ("/", (150, 152, 160), "CROWBAR"),
    "&": ("¶", (190, 180, 150), "NOTE"),
    "k": ("§", (204, 172, 80), "BRASS KEY"),
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
COST_MOVE = 100
COST_SPRINT = 150
COST_DOOR = 75
COST_PICKUP = 50
COST_FIRE = 100
COST_MELEE = 100
COST_RELOAD = 125
COST_HEAL = 150

BAYER = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
]

# Python's cp437 codec maps 0x00-0x1F to C0 controls, so the graphical glyphs
# that live down there need their indices spelled out.
LOW_GLYPHS = {"☼": 15, "¶": 20, "§": 21, "↑": 24, "→": 26, "▲": 30}


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
    sense: int = 0
    energy: int = 0
    aware: bool = False
    target: tuple[int, int] | None = None
    patience: int = 0

    @property
    def alive(self) -> bool:
        return self.hp > 0

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
    return Actor("@", (244, 238, 214), "YOU", x, y, 100, 100, 100, (8, 14))


ENEMY_KINDS = {
    "z": dict(glyph="z", colour=(132, 152, 92), name="SHAMBLER",
              hp=30, energy_gain=60, damage=(6, 12), sense=7),
    "d": dict(glyph="d", colour=(152, 112, 72), name="HOUND",
              hp=18, energy_gain=140, damage=(4, 8), sense=9),
}


def make_enemy(kind: str, x: int, y: int) -> Actor:
    spec = ENEMY_KINDS[kind]
    return Actor(spec["glyph"], spec["colour"], spec["name"], x, y,
                 spec["hp"], spec["hp"], spec["energy_gain"],
                 spec["damage"], spec["sense"])


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


class Level:
    def __init__(self, rng: random.Random, floor: int) -> None:
        self.rng = rng
        self.floor = floor
        self.tiles = [["#"] * LEVEL_W for _ in range(LEVEL_H)]
        self.items: dict[tuple[int, int], str] = {}
        self.rooms: list[Room] = []
        self.enemies: list[Actor] = []
        self.explored: set[tuple[int, int]] = set()
        self.visible: set[tuple[int, int]] = set()
        self.flow: dict[tuple[int, int], int] = {}
        self._carve()
        self._decorate()
        self.player = make_player(*self.rooms[0].centre)
        self._populate()
        self.candles = [(x, y) for y in range(LEVEL_H) for x in range(LEVEL_W)
                        if self.tiles[y][x] == "*"]

    # -- terrain ------------------------------------------------------------
    def _carve(self) -> None:
        rng = self.rng
        for _ in range(400):
            if len(self.rooms) >= 14:
                break
            w, h = rng.randint(5, 11), rng.randint(4, 8)
            x, y = rng.randint(2, LEVEL_W - w - 3), rng.randint(2, LEVEL_H - h - 3)
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
            if self.rng.random() < 0.5:
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
            self.tiles[y][x] = "+" if self.rng.random() < 0.65 else "'"

    def _is_doorway(self, x: int, y: int) -> bool:
        if not (0 < x < LEVEL_W - 1 and 0 < y < LEVEL_H - 1):
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
                for _ in range(rng.randint(2, 5)):
                    x, y = rng.randint(room.x, room.x + room.w - 1), rng.randint(room.y, room.y + room.h - 1)
                    if self.tiles[y][x] == ".":
                        self.tiles[y][x] = "="
            elif role == "storage":
                for _ in range(rng.randint(4, 10)):
                    x, y = rng.randint(room.x, room.x + room.w - 1), rng.randint(room.y, room.y + room.h - 1)
                    if self.tiles[y][x] == ".":
                        self.tiles[y][x] = ","
            elif role == "gallery":
                for x in range(room.x + 1, room.x + room.w - 1, 3):
                    if self.tiles[room.y][x] == ".":
                        self.tiles[room.y][x] = "T"

            if index and rng.random() < 0.4:
                for _ in range(8):
                    x, y = rng.randint(room.x, room.x + room.w - 1), rng.randint(room.y, room.y + room.h - 1)
                    if self.tiles[y][x] == ".":
                        self.tiles[y][x] = "*"
                        break

    def _populate(self) -> None:
        rng = self.rng
        start = self.rooms[0]
        self.tiles[start.centre[1]][start.centre[0]] = "<"

        far = max(self.rooms[1:], key=lambda r: math.dist(r.centre, start.centre))
        self.tiles[far.centre[1]][far.centre[0]] = ">"
        self.exit = far.centre

        pool = "!!/&kaa" if self.floor > 1 else "!!!/&ka"
        for glyph in pool:
            cell = self._free_cell(exclude_room=None)
            if cell:
                self.items[cell] = glyph

        for _ in range(len(self.rooms) // 2 + self.floor):
            cell = self._free_cell(exclude_room=start)
            if cell:
                kind = "d" if rng.random() < 0.35 else "z"
                self.enemies.append(make_enemy(kind, *cell))

        for _ in range(rng.randint(3, 8)):
            cell = self._free_cell(exclude_room=start)
            if cell and self.tiles[cell[1]][cell[0]] == ".":
                self.tiles[cell[1]][cell[0]] = "%"

    def _free_cell(self, exclude_room: Room | None) -> tuple[int, int] | None:
        rooms = [r for r in self.rooms if r is not exclude_room]
        for _ in range(200):
            room = self.rng.choice(rooms)
            x = self.rng.randint(room.x, room.x + room.w - 1)
            y = self.rng.randint(room.y, room.y + room.h - 1)
            if self.tiles[y][x] not in (".", ",", '"'):
                continue
            if (x, y) in self.items or self.actor_at(x, y):
                continue
            return x, y
        return None

    # -- queries ------------------------------------------------------------
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < LEVEL_W and 0 <= y < LEVEL_H

    def walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and TILES[self.tiles[y][x]][2]

    def pathable(self, x: int, y: int) -> bool:
        """Walkable, plus closed doors -- anything that can open one may route through."""
        return self.walkable(x, y) or (self.in_bounds(x, y) and self.tiles[y][x] == "+")

    def opaque(self, x: int, y: int) -> bool:
        return not self.in_bounds(x, y) or TILES[self.tiles[y][x]][3]

    def connected(self) -> bool:
        """Every room centre and the exit must be reachable from the entrance."""
        flow = compute_flow(self, limit=LEVEL_W * LEVEL_H)
        return all(room.centre in flow for room in self.rooms) and self.exit in flow

    def actor_at(self, x: int, y: int) -> Actor | None:
        for enemy in self.enemies:
            if enemy.alive and enemy.x == x and enemy.y == y:
                return enemy
        return None

    def actors(self) -> list[Actor]:
        return [self.player, *[e for e in self.enemies if e.alive]]


# ---------------------------------------------------------------------------
# field of view and flow field
# ---------------------------------------------------------------------------
def compute_fov(level: Level, ox: int, oy: int, radius: int) -> set[tuple[int, int]]:
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
            if not level.in_bounds(tx, ty) or math.dist((tx, ty), (ox, oy)) > radius:
                break
            visible.add((tx, ty))
            if level.opaque(tx, ty):
                break
    return visible


def compute_flow(level: Level, limit: int = 22) -> dict[tuple[int, int], int]:
    """Breadth-first distance field to the player; enemies just walk downhill."""
    origin = (level.player.x, level.player.y)
    flow = {origin: 0}
    queue = deque([origin])
    while queue:
        x, y = queue.popleft()
        distance = flow[(x, y)]
        if distance >= limit:
            continue
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) in flow or not level.pathable(nx, ny):
                continue
            flow[(nx, ny)] = distance + 1
            queue.append((nx, ny))
    return flow


# ---------------------------------------------------------------------------
# world
# ---------------------------------------------------------------------------
@dataclass
class World:
    seed: int
    rng: random.Random
    level: Level
    floor: int = 1
    turn: int = 0
    noise: int = 0
    ammo: int = 6
    ammo_reserve: int = 12
    inventory: dict[str, int] = field(default_factory=lambda: {"FIRST AID": 1})
    messages: list[tuple[str, tuple[int, int, int]]] = field(default_factory=list)
    dead: bool = False
    descended: bool = False

    @property
    def player(self) -> Actor:
        return self.level.player

    def log(self, text: str, colour=UI_TEXT) -> None:
        self.messages.append((text.upper(), colour))
        del self.messages[:-2]

    def carried(self) -> int:
        return sum(self.inventory.values())


def new_world(seed: int, floor: int = 1) -> World:
    rng = random.Random(seed + floor * 977)
    for _ in range(16):
        level = Level(rng, floor)
        if level.connected():
            break
    else:
        raise RuntimeError(f"seed {seed} floor {floor}: no valid layout in 16 attempts")
    world = World(seed=seed, rng=rng, level=level, floor=floor)
    world.log("the door locks behind you.", UI_DIM)
    refresh_senses(world)
    return world


def refresh_senses(world: World) -> None:
    level = world.level
    level.visible = compute_fov(level, level.player.x, level.player.y, SIGHT_RADIUS)
    level.explored |= level.visible
    level.flow = compute_flow(level)


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------
def damage_roll(actor: Actor, rng: random.Random) -> int:
    return rng.randint(*actor.damage)


def alert(world: World, radius: int) -> None:
    """A noise pulls every enemy in radius toward the player's position."""
    player = world.player
    for enemy in world.level.enemies:
        if enemy.alive and math.dist((enemy.x, enemy.y), (player.x, player.y)) <= radius:
            enemy.aware = True
            enemy.target = (player.x, player.y)
            enemy.patience = 12


def tick(world: World) -> None:
    """Hand out energy until somebody can act, then let the enemies spend it."""
    for actor in world.level.actors():
        actor.gain()
    world.turn += 1
    if world.turn % 4 == 0 and world.noise > 0:
        world.noise -= 1
    resolve_enemies(world)


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
    level = world.level
    player = level.player
    tx, ty = player.x + dx, player.y + dy

    target = level.actor_at(tx, ty)
    if target is not None:
        perform(world, COST_MELEE, lambda: melee(world, target))
        return

    if level.in_bounds(tx, ty) and level.tiles[ty][tx] == "+":
        perform(world, COST_DOOR, lambda: open_door(world, tx, ty))
        return

    if not level.walkable(tx, ty):
        return

    steps = 2 if sprint else 1
    cost = COST_SPRINT if sprint else COST_MOVE
    perform(world, cost, lambda: walk(world, dx, dy, steps, sprint))


def walk(world: World, dx: int, dy: int, steps: int, sprint: bool) -> None:
    level = world.level
    player = level.player
    for _ in range(steps):
        tx, ty = player.x + dx, player.y + dy
        if not level.walkable(tx, ty) or level.actor_at(tx, ty):
            break
        player.x, player.y = tx, ty
    if sprint:
        world.noise = min(10, world.noise + 3)
        alert(world, 9)
    pick_up(world)
    if (player.x, player.y) == level.exit:
        world.descended = True


def pick_up(world: World) -> None:
    level = world.level
    cell = (level.player.x, level.player.y)
    glyph = level.items.get(cell)
    if glyph is None:
        return
    name = ITEMS[glyph][2]
    if world.carried() >= 8 and glyph != "a":
        world.log(f"no room for the {name.lower()}.", UI_DIM)
        return

    # Stooping costs on top of the move that carried you here.
    level.player.energy = max(0, level.player.energy - COST_PICKUP)
    del level.items[cell]
    if glyph == "a":
        world.ammo_reserve += 6
        world.log("you pocket six rounds.", UI_WARM)
    else:
        world.inventory[name] = world.inventory.get(name, 0) + 1
        world.log(f"picked up {name.lower()}.", UI_WARM)


def open_door(world: World, x: int, y: int) -> None:
    world.level.tiles[y][x] = "'"
    world.noise = min(10, world.noise + 1)
    alert(world, 5)
    world.log("the hinges shriek.", UI_DIM)


def melee(world: World, target: Actor) -> None:
    weapon = "CROWBAR" if world.inventory.get("CROWBAR") else "fists"
    bonus = 6 if weapon == "CROWBAR" else 0
    target.hp -= damage_roll(world.player, world.rng) + bonus
    world.noise = min(10, world.noise + 2)
    alert(world, 6)
    if target.alive:
        world.log(f"you strike the {target.name.lower()}.", UI_TEXT)
    else:
        world.log(f"the {target.name.lower()} comes apart.", UI_WARM)
        world.level.tiles[target.y][target.x] = "%"


def fire(world: World) -> None:
    level = world.level
    if world.ammo <= 0:
        world.log("the hammer falls on an empty chamber.", UI_HOT)
        return
    visible = [e for e in level.enemies
               if e.alive and (e.x, e.y) in level.visible]
    if not visible:
        world.log("you fire into the dark. nothing.", UI_DIM)
        world.ammo -= 1
        world.noise = 10
        alert(world, 16)
        return
    target = min(visible, key=lambda e: math.dist((e.x, e.y), (level.player.x, level.player.y)))
    world.ammo -= 1
    target.hp -= world.rng.randint(22, 34)
    world.noise = 10
    alert(world, 16)
    if target.alive:
        world.log(f"the {target.name.lower()} staggers.", UI_TEXT)
    else:
        world.log(f"the {target.name.lower()} drops.", UI_WARM)
        level.tiles[target.y][target.x] = "%"


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
    world.inventory["FIRST AID"] -= 1
    if not world.inventory["FIRST AID"]:
        del world.inventory["FIRST AID"]
    player = world.player
    player.hp = min(player.max_hp, player.hp + 32)
    world.log("you bind the wound. it holds.", UI_WARM)


# ---------------------------------------------------------------------------
# enemy turns
# ---------------------------------------------------------------------------
def resolve_enemies(world: World) -> None:
    for enemy in world.level.enemies:
        guard = 0
        while enemy.alive and not world.dead and enemy_turn(world, enemy):
            guard += 1
            if guard > 8:
                break


def enemy_turn(world: World, enemy: Actor) -> bool:
    """Take one action. Returns False when the enemy banks energy instead.

    Banking rather than burning the surplus on a wait is what makes speed mean
    anything: a shambler gaining 60 has to sit out a tick to afford a 100 move.
    """
    level = world.level
    player = level.player
    here = (enemy.x, enemy.y)
    distance = math.dist(here, (player.x, player.y))

    # Player FOV doubles as the enemy's: if you can see it, it can see you.
    if here in level.visible and distance <= enemy.sense:
        enemy.aware = True
        enemy.target = (player.x, player.y)
        enemy.patience = 12

    if distance <= 1.5 and enemy.aware:
        if not enemy.can_afford(COST_MELEE):
            return False
        enemy.spend(COST_MELEE)
        player.hp -= damage_roll(enemy, world.rng)
        world.log(f"the {enemy.name.lower()} tears into you.", UI_HOT)
        if player.hp <= 0:
            world.dead = True
        return True

    step = None
    if enemy.aware:
        # Downhill on the flow field when the player is reachable, otherwise
        # grope toward the last known position.
        current = level.flow.get(here)
        if current is not None:
            options = [(level.flow.get(n, 999), n) for n in neighbours(here)
                       if level.pathable(*n) and not level.actor_at(*n)]
            options = [o for o in options if o[0] < current]
            if options:
                step = min(options)[1]
        if step is None and enemy.target:
            step = grope(level, here, enemy.target)
    elif world.rng.random() < 0.6:
        step = world.rng.choice(list(neighbours(here)))

    if step is None or level.actor_at(*step) or step == (player.x, player.y):
        if not enemy.can_afford(COST_WAIT):
            return False
        enemy.spend(COST_WAIT)
        return True

    if level.tiles[step[1]][step[0]] == "+":
        # Doors are a delay for the enemy, not a wall -- and they stay open.
        if not enemy.can_afford(COST_DOOR):
            return False
        enemy.spend(COST_DOOR)
        level.tiles[step[1]][step[0]] = "'"
        if step in level.visible:
            world.log(f"the {enemy.name.lower()} shoulders a door open.", UI_HOT)
        else:
            world.log("a door swings open somewhere.", UI_DIM)
    elif level.walkable(*step):
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


def neighbours(cell: tuple[int, int]):
    x, y = cell
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def grope(level: Level, here: tuple[int, int], target: tuple[int, int]):
    best = None
    for option in neighbours(here):
        if not level.walkable(*option) or level.actor_at(*option):
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


def light_at(level: Level, x: int, y: int, flicker: float) -> float:
    player = level.player
    total = 1.0 - (math.dist((x, y), (player.x, player.y)) / (LANTERN_RADIUS * flicker)) ** 1.15
    total = max(0.0, total)
    for cx, cy in level.candles:
        distance = math.dist((x, y), (cx, cy))
        if distance < CANDLE_RADIUS:
            total += 0.7 * (1.0 - distance / CANDLE_RADIUS)
    return min(1.0, total)


def dither(value: float, x: int, y: int) -> float:
    """Ordered-dither a brightness into LIGHT_LEVELS bands."""
    threshold = (BAYER[y % 4][x % 4] + 0.5) / 16.0
    return min(1.0, math.floor(value * LIGHT_LEVELS + threshold) / LIGHT_LEVELS)


def camera(level: Level) -> tuple[int, int]:
    cx = max(0, min(level.player.x - VIEW_W // 2, LEVEL_W - VIEW_W))
    cy = max(0, min(level.player.y - VIEW_H // 2, LEVEL_H - VIEW_H))
    return cx, cy


def draw_map(surface, font: Font, world: World, flicker: float, debug: bool) -> None:
    level = world.level
    ox, oy = camera(level)
    for sy in range(VIEW_H):
        for sx in range(VIEW_W):
            x, y = ox + sx, oy + sy
            if (x, y) not in level.explored:
                continue

            seen = (x, y) in level.visible
            lit = max(light_at(level, x, y, flicker), AMBIENT) if seen else 0.0
            level_value = dither(lit if seen else MEMORY_LEVEL, x, y)
            if level_value <= 0.0:
                continue

            glyph, colour, _, _ = TILES[level.tiles[y][x]]
            if (x, y) in level.items and seen:
                glyph, colour, _ = ITEMS[level.items[(x, y)]]

            colour = shade(colour, lit)
            font.blit(surface, glyph, MAP_X + sx, MAP_Y + sy, scale_colour(colour, level_value))

    for actor in level.actors():
        if (actor.x, actor.y) not in level.visible:
            continue
        sx, sy = actor.x - ox, actor.y - oy
        if not (0 <= sx < VIEW_W and 0 <= sy < VIEW_H):
            continue
        lit = light_at(level, actor.x, actor.y, flicker)
        level_value = dither(max(lit, 0.45), actor.x, actor.y)
        colour = actor.colour
        if lit < 0.25 and actor is not level.player:
            # Outside the lantern things read as silhouettes, not sprites.
            colour = (int(colour[0] * 0.85), int(colour[1] * 0.6), int(colour[2] * 0.7))
        font.blit(surface, actor.glyph, MAP_X + sx, MAP_Y + sy, scale_colour(colour, level_value))
        if debug and actor.aware:
            font.blit(surface, "!", MAP_X + sx, MAP_Y + sy - 1, UI_HOT)


def shade(colour, lit: float):
    if lit < 0.05:
        grey = sum(colour) / 3.0
        return (int(grey * 0.7), int(grey * 0.8), int(grey * 1.0))
    warmth = lit * 0.3
    return tuple(int(c * (1.0 - warmth) + tint * warmth)
                 for c, tint in zip(colour, (255, 226, 186)))


def bar(value: int, maximum: int, width: int = 10) -> str:
    filled = max(0, min(width, int(round(width * value / maximum))))
    return "█" * filled + "░" * (width - filled)


def threat_level(world: World) -> tuple[str, tuple[int, int, int]]:
    aware = sum(1 for e in world.level.enemies if e.alive and e.aware)
    if aware >= 3:
        return "HIGH", UI_HOT
    if aware:
        return "CLOSE", UI_WARM
    return "QUIET", UI_DIM


def draw_hud(surface, font: Font, world: World, tick_count: int, debug: bool) -> None:
    player = world.player
    font.text(surface, "NIGHTSHIFT".ljust(14) + f"MANSION {world.floor}F", 0, 0, UI_DIM)
    font.text(surface, f"SEED {world.seed}"[:PANEL_W], PANEL_X, 0, UI_DIM)

    for y in range(0, LOG_Y - 1):
        font.blit(surface, "│", DIVIDER_X, y, UI_DIM)
    font.text(surface, "─" * COLS, 0, LOG_Y - 1, UI_DIM)
    font.blit(surface, "┴", DIVIDER_X, LOG_Y - 1, UI_DIM)

    rule = "─" * PANEL_W
    threat, threat_colour = threat_level(world)
    panel: list[tuple[str, tuple[int, int, int]]] = [
        ("   STATUS", UI_TEXT),
        (rule, UI_DIM),
        (f"HP {bar(player.hp, player.max_hp)}", (176, 62, 58)),
        (f"{player.hp:>7}/{player.max_hp}", UI_TEXT),
        (f"EN {bar(player.energy, player.max_energy)}", (120, 174, 168)),
        (f"{player.energy:>7}/{player.max_energy}", UI_TEXT),
        (rule, UI_DIM),
        (") REVOLVER", UI_DIM),
        (f"  AMMO  {world.ammo}/ 6", UI_WARM if world.ammo else UI_HOT),
        (f"  RESERVE  {world.ammo_reserve:>2}", UI_TEXT),
        (rule, UI_DIM),
        (f"CARRY    {world.carried()}/8", UI_DIM),
    ]
    for name, count in list(world.inventory.items())[:5]:
        # Count goes in front: a trailing "x2" is what gets clipped at 13 columns.
        prefix = f"{count}x" if count > 1 else "-"
        panel.append((f"{prefix} {name}"[:PANEL_W], UI_TEXT))
    panel.append((rule, UI_DIM))
    panel.append((f"NOISE  {bar(world.noise, 10, 5)}", UI_TEXT))
    panel.append((f"THREAT {threat:>6}", threat_colour))
    panel.append((f"TURN {world.turn:>8}", UI_DIM))

    for offset, (line, colour) in enumerate(panel[:LOG_Y - 2]):
        font.text(surface, line[:PANEL_W], PANEL_X, MAP_Y + offset, colour)

    for offset, (line, colour) in enumerate(world.messages[-2:]):
        font.text(surface, f"> {line}"[:COLS], 0, LOG_Y + offset, colour)

    if debug:
        awake = sum(1 for e in world.level.enemies if e.alive and e.aware)
        alive = sum(1 for e in world.level.enemies if e.alive)
        footer = f"ROOMS {len(world.level.rooms)}  LIVE {alive}  AWAKE {awake}  FLOW {len(world.level.flow)}"
    else:
        footer = "MOVE 100  SPRINT 150  FIRE 100  ? KEYS"
    font.text(surface, footer[:COLS], 0, LOG_Y + 2, UI_DIM)

    if world.dead:
        banner(surface, font, "YOU DIED", "N FOR A NEW SEED", tick_count)
    elif world.descended:
        banner(surface, font, "STAIRS DOWN", "SPACE TO DESCEND", tick_count)


HELP_LINES = [
    ("WASD / ARROWS", "MOVE 100"),
    ("  + SHIFT", "SPRINT 150"),
    ("BUMP ENEMY", "ATTACK 100"),
    ("BUMP DOOR", "OPEN 75"),
    ("SPACE", "WAIT 50"),
    ("F  FIRE REVOLVER", "100"),
    ("R  RELOAD", "125"),
    ("Q  FIRST AID", "150"),
    (None, None),
    ("N", "NEW LEVEL"),
    ("F1", "DEBUG OVERLAY"),
    ("?", "THIS LIST"),
    ("ESC", "QUIT"),
]


def draw_help(surface, font: Font) -> None:
    inner = VIEW_W - 2
    body = [f"{left:<{inner - len(right)}}{right}" if left else "─" * inner
            for left, right in HELP_LINES]
    body = ["CONTROLS", "─" * inner, *body, "─" * inner,
            "ENERGY BUYS ACTIONS.", "TIME PASSES TO AFFORD."]

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
    x = (VIEW_W - len(title)) // 2
    font.text(surface, title, x, y, UI_HOT if (tick_count // 20) % 2 == 0 else UI_WARM)
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


def pick_scale(requested: int | None) -> int:
    """Largest integer upscale that fits the desktop, leaving room for chrome.

    Nearest-neighbour only looks right at whole multiples, so this steps down to
    the next whole factor rather than fitting the screen exactly.
    """
    if requested:
        return max(1, requested)
    try:
        desktop_w, desktop_h = pygame.display.get_desktop_sizes()[0]
    except (AttributeError, IndexError, pygame.error):
        info = pygame.display.Info()
        desktop_w, desktop_h = info.current_w, info.current_h
    # Title bar, borders, and a taskbar all eat into what a window can claim.
    fits = min((desktop_w - 32) // (COLS * CELL), (desktop_h - 96) // (ROWS * CELL))
    return max(1, min(MAX_SCALE, fits))


def is_help_key(event) -> bool:
    """'?' is shift+/ on US layouts but its own key elsewhere; accept either."""
    return getattr(event, "unicode", "") == "?" or event.key == pygame.K_QUESTION


def handle_key(world: World, event) -> World:
    if event.key == pygame.K_n:
        return new_world(world.rng.randrange(10**5), 1)

    if world.dead:
        return world

    if world.descended and event.key == pygame.K_SPACE:
        following = new_world(world.seed, world.floor + 1)
        following.ammo, following.ammo_reserve = world.ammo, world.ammo_reserve
        following.inventory = dict(world.inventory)
        following.player.hp = world.player.hp
        following.log("deeper. the air is colder.", UI_DIM)
        return following

    if event.key in DIRECTIONS:
        dx, dy = DIRECTIONS[event.key]
        sprint = bool(event.mod & pygame.KMOD_SHIFT)
        try_move(world, dx, dy, sprint)
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
    args = parser.parse_args()

    if args.out:
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    pygame.init()
    scale = pick_scale(args.scale)
    size = (COLS * CELL * scale, ROWS * CELL * scale)
    pygame.display.set_mode(size)
    pygame.display.set_caption("Project Nightshift")
    print(f"{size[0]}x{size[1]} (scale {scale}) -- override with --scale N")

    seed = args.seed if args.seed is not None else random.randrange(10**5)
    world = new_world(seed)

    font = Font(SHEET)
    internal = pygame.Surface((COLS * CELL, ROWS * CELL)).convert()
    scanlines = make_scanlines(size)
    vignette = make_vignette(size)
    grain = make_grain((COLS * CELL, ROWS * CELL), random.Random(seed))

    if args.out:
        frame = render_frame(font, internal, scanlines, vignette, grain, world, 12, False)
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
