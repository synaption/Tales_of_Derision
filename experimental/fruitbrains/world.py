"""Scenes: the town square outside and the cottage interior.

A scene is a ground tile grid plus a flat list of props.  Props carry their own
solid rect (so villagers walk *behind* the roof of a house but bump into its
walls) and a sort key, so props and fruits go into one depth-sorted draw list.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import pygame

from assets import Assets
from render import TILE, Camera, PixelArt, blit_art


@dataclass
class Prop:
    art: PixelArt
    x: float
    y: float
    solid: pygame.Rect | None = None
    sort_y: float | None = None
    flat: bool = False          # drawn with the ground, never sorted against actors

    @property
    def depth(self) -> float:
        return self.sort_y if self.sort_y is not None else self.y + self.art.h

    def draw(self, surface: pygame.Surface, cam: Camera, now: float) -> None:
        blit_art(surface, cam, self.art, self.x, self.y)


@dataclass
class Door:
    rect: pygame.Rect
    target: str
    spawn: tuple[float, float]
    label: str = "door"


class Bird:
    """Ambient wildlife: hops around, pecks, and startles when you get close."""

    def __init__(self, pos, assets: Assets, index: int) -> None:
        self.pos = pygame.Vector2(pos)
        self.home = pygame.Vector2(pos)
        self.assets = assets
        self.facing = 1
        self.timer = index * 0.7
        self.frame = 0.0
        self.hop = 0.0

    def update(self, dt: float, threat: pygame.Vector2 | None) -> None:
        self.frame += dt * 6
        self.timer -= dt
        if threat is not None and self.pos.distance_to(threat) < 90:
            away = (self.pos - threat)
            if away.length_squared() < 1:
                away = pygame.Vector2(1, 0)
            self.pos += away.normalize() * 130 * dt
            self.hop = 1.0
            self.facing = 1 if away.x > 0 else -1
        elif self.timer <= 0:
            self.timer = random.uniform(1.2, 3.4)
            step = pygame.Vector2(random.uniform(-40, 40), random.uniform(-24, 24))
            self.pos = self.home + (self.pos - self.home + step) * 0.6
            self.facing = 1 if step.x >= 0 else -1
            self.hop = 0.55
        self.hop = max(0.0, self.hop - dt)

    def draw(self, surface: pygame.Surface, cam: Camera, now: float) -> None:
        frames = self.assets.birds if self.facing > 0 else self.assets.birds_left
        art = frames[int(self.frame) % len(frames)]
        lift = math.sin(min(1.0, self.hop) * math.pi) * 10
        blit_art(surface, cam, art, self.pos.x - art.w / 2, self.pos.y - art.h - lift)

    @property
    def depth(self) -> float:
        return self.pos.y


@dataclass
class Scene:
    name: str
    title: str
    ground: list[list[PixelArt]]
    props: list[Prop] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    birds: list[Bird] = field(default_factory=list)
    sky: tuple[int, int, int] = (156, 218, 216)
    indoors: bool = False
    fill: PixelArt | None = None      # tiled beyond the map edge, if any

    @property
    def bounds(self) -> pygame.Rect:
        return pygame.Rect(0, 0, len(self.ground[0]) * TILE, len(self.ground) * TILE)

    @property
    def solids(self) -> list[pygame.Rect]:
        return [p.solid for p in self.props if p.solid is not None]

    def draw_ground(self, surface: pygame.Surface, cam: Camera) -> None:
        view = cam.visible()
        w, h = len(self.ground[0]), len(self.ground)
        inside = self.fill is None
        x0 = max(0, view.left // TILE) if inside else view.left // TILE
        x1 = min(w, view.right // TILE + 1) if inside else view.right // TILE + 1
        y0 = max(0, view.top // TILE) if inside else view.top // TILE
        y1 = min(h, view.bottom // TILE + 1) if inside else view.bottom // TILE + 1
        for ty in range(y0, y1):
            row = self.ground[ty] if 0 <= ty < h else None
            for tx in range(x0, x1):
                art = row[tx] if row is not None and 0 <= tx < w else self.fill
                if art is not None:
                    blit_art(surface, cam, art, tx * TILE, ty * TILE)


def _grass(assets: Assets, w: int, h: int) -> list[list[PixelArt]]:
    rng = random.Random(3)
    return [[rng.choice(assets.grass) for _ in range(w)] for _ in range(h)]


def _paint(ground, assets: Assets, art_name: str, tx: int, ty: int, tw: int, th: int,
           source: str = "town") -> None:
    art = getattr(assets, source)[art_name]
    for y in range(ty, ty + th):
        if 0 <= y < len(ground):
            for x in range(tx, tx + tw):
                if 0 <= x < len(ground[0]):
                    ground[y][x] = art


def build_town(assets: Assets) -> Scene:
    W, H = 40, 26
    ground = _grass(assets, W, H)
    town = assets.town
    rng = random.Random(19)

    # Mossy plaza around the fountain, with a ragged edge so it reads as worn
    # grass rather than a rectangle.
    for ty in range(H):
        for tx in range(W):
            d = ((tx - 17.0) / 5.4) ** 2 + ((ty - 16.0) / 3.4) ** 2
            if d < 0.82 or (d < 1.0 and rng.random() < 0.55):
                ground[ty][tx] = town["tile_moss"]

    # Brick walkway: cottage door -> plaza -> east to the market stall.
    _paint(ground, assets, "tile_path", 16, 7, 2, 7)
    _paint(ground, assets, "tile_path", 21, 15, 7, 2)
    _paint(ground, assets, "tile_path", 27, 11, 2, 5)
    _paint(ground, assets, "tile_wood", 26, 8, 5, 4)       # market stall decking

    props: list[Prop] = []

    # --- cottage -----------------------------------------------------------
    hx, hy = 14 * TILE, 2 * TILE
    house = town["house"]
    door = pygame.Rect(hx + 60, hy + 150, 34, 30)
    props.append(Prop(house, hx, hy, sort_y=hy + house.h - 6))
    # Walls collide, but the doorway between them is left open.
    for rect in (pygame.Rect(hx + 4, hy + 96, 56, 66), pygame.Rect(hx + 94, hy + 96, 78, 66)):
        props.append(Prop(_blank(), rect.x, rect.y, solid=rect, sort_y=-1e9))

    # --- fountain plaza ----------------------------------------------------
    fountain = town["fountain"]
    fx, fy = 17 * TILE - fountain.w / 2, 16 * TILE - fountain.h / 2
    props.append(Prop(fountain, fx, fy, solid=pygame.Rect(fx + 6, fy + 20, fountain.w - 12, 32)))

    # --- market stall ------------------------------------------------------
    sx, sy = 26 * TILE, 8 * TILE
    roof, post = town["stall_roof"], town["stall_post"]
    for px in (sx + 20, sx + 116):
        props.append(Prop(post, px, sy + 44, solid=pygame.Rect(px, sy + 66, post.w, 10)))
    props.append(Prop(roof, sx + 22, sy, sort_y=sy + 118))
    props.append(Prop(town["case"], sx + 8, sy + 76, solid=pygame.Rect(sx + 8, sy + 100, 64, 40)))
    props.append(Prop(town["anvil"], sx + 90, sy + 96, solid=pygame.Rect(sx + 90, sy + 104, 48, 18)))

    # --- fences: the town's south edge and a flower garden ------------------
    fence, fence_end = town["fence"], town["fence_end"]

    def fence_run(tx0: int, tx1: int, ty: int, gap: int | None = None) -> None:
        for tx in range(tx0, tx1):
            if tx == gap:
                continue
            art = fence_end if tx in (tx0, tx1 - 1) else fence
            props.append(Prop(art, tx * TILE, ty * TILE,
                              solid=pygame.Rect(tx * TILE, ty * TILE + 20, TILE, 12)))

    fence_run(2, 14, 22)
    fence_run(24, 37, 22)
    fence_run(3, 10, 4)                                    # garden, north side
    fence_run(3, 10, 9, gap=6)                             # ...south side, with a gate
    props.append(Prop(town["gate"], 6 * TILE - 2, 9 * TILE - 2, flat=True))

    # --- scatter -----------------------------------------------------------
    props.append(Prop(town["rock_big"], 12 * TILE, 12 * TILE,
                      solid=pygame.Rect(12 * TILE, 12 * TILE + 16, 28, 14)))
    props.append(Prop(town["rock_flat"], 31 * TILE, 19 * TILE,
                      solid=pygame.Rect(31 * TILE, 19 * TILE + 8, 32, 16)))
    props.append(Prop(town["grave"], 34 * TILE, 4 * TILE,
                      solid=pygame.Rect(34 * TILE, 4 * TILE + 18, 24, 12)))
    for _ in range(40):                                    # wildflowers everywhere
        props.append(Prop(rng.choice(assets.flowers),
                          rng.uniform(TILE, (W - 2) * TILE), rng.uniform(TILE, (H - 2) * TILE),
                          flat=True))
    for _ in range(26):                                    # ...thick inside the garden
        props.append(Prop(rng.choice(assets.flowers),
                          rng.uniform(3.2 * TILE, 9.6 * TILE), rng.uniform(5 * TILE, 9 * TILE),
                          flat=True))

    birds = [Bird((rng.uniform(3 * TILE, 36 * TILE), rng.uniform(4 * TILE, 21 * TILE)), assets, i)
             for i in range(6)]

    scene = Scene("town", "Fruit Town", ground, props, birds=birds, fill=assets.grass[0])
    scene.doors.append(Door(door, "house", (0, 0), "the cottage"))   # spawn set in build_scenes
    return scene


def build_house(assets: Assets) -> Scene:
    W, H = 16, 10
    interior, town = assets.interior, assets.town
    ground = [[interior["floor" if (tx + ty) % 7 else "floor_alt"] for tx in range(W)]
              for ty in range(H)]
    # Wall art is 16x32 (wallpaper plus its skirting), so it covers the top two rows.
    props: list[Prop] = [Prop(interior["wall" if tx % 5 else "wall_warm"], tx * TILE, 0, flat=True)
                         for tx in range(W)]

    wall_y = 2 * TILE
    props.append(Prop(_blank(), 0, 0, solid=pygame.Rect(0, 0, W * TILE, wall_y - 8), sort_y=-1e9))
    for rect in (pygame.Rect(-8, 0, 8, H * TILE), pygame.Rect(W * TILE, 0, 8, H * TILE),
                 pygame.Rect(0, H * TILE - 4, W * TILE, 12)):
        props.append(Prop(_blank(), rect.x, rect.y, solid=rect, sort_y=-1e9))

    def place(art, tx, ty, solid_h=None, source="interior"):
        art_obj = getattr(assets, source)[art] if isinstance(art, str) else art
        x, y = tx * TILE, ty * TILE
        solid = None
        if solid_h:
            solid = pygame.Rect(x + 4, y + art_obj.h - solid_h, art_obj.w - 8, solid_h)
        props.append(Prop(art_obj, x, y, solid=solid))
        return props[-1]

    # Wall dressing sits flat against the wallpaper.
    for name, tx in (("art_rose", 1), ("picture", 4), ("picture_alt", 6), ("poster_alt", 11)):
        art = interior[name]
        props.append(Prop(art, tx * TILE + 6, TILE - art.h / 2 + 12, flat=True))
    props.append(Prop(town["window"], 8 * TILE + 4, TILE - 6, flat=True))
    props.append(Prop(interior["shelf"], 13 * TILE, TILE + 14, flat=True))

    # Sitting room on the left, kitchen right, bed in the back corner.
    props.append(Prop(interior["rug"], 3 * TILE + 8, 5 * TILE + 8, flat=True))
    props.append(Prop(interior["rug_red"], 6 * TILE, 7 * TILE + 12, flat=True))
    place("sofa", 2, 3, 18)
    place("sofa_small", 6, 3, 14)
    place("chair_left", 2, 6, 14)
    place("chair_right", 5, 6, 14)
    place("chair_front", 4, 4, 12)
    place("stove", 13, 3, 16)
    place("flowers", 11, 4, 10)
    place("plant", 14, 6, 14)
    place(town["bed"], 9, 6, 22)

    # Doormat by the exit.
    props.append(Prop(interior["shelf"], 7 * TILE + 4, (H - 1) * TILE + 4, flat=True))
    exit_rect = pygame.Rect(7 * TILE, (H - 1) * TILE, 2 * TILE + 16, TILE)

    scene = Scene("house", "Cottage", ground, props, sky=(58, 44, 52), indoors=True)
    scene.doors.append(Door(exit_rect, "town", (0, 0), "outside"))
    return scene


_BLANK: PixelArt | None = None


def _blank() -> PixelArt:
    """A 1x1 transparent sprite -- used for pure colliders."""
    global _BLANK
    if _BLANK is None:
        _BLANK = PixelArt(pygame.Surface((1, 1), pygame.SRCALPHA), 1)
    return _BLANK


def build_scenes(assets: Assets) -> dict[str, Scene]:
    town = build_town(assets)
    house = build_house(assets)
    # Each door lands you just clear of the other side's trigger.
    front, mat = town.doors[0], house.doors[0]
    front.spawn = (mat.rect.centerx, mat.rect.top - 46)
    mat.spawn = (front.rect.centerx, front.rect.bottom + 40)
    return {"town": town, "house": house}
