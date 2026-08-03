"""Visual mockup for Project Nightshift.

Renders a single hand-authored mansion scene with the MSX-UnDeadPeopleEdition
CP437 sheet to prove out the intended presentation: a low-resolution internal
surface, dithered light falloff, tile memory, the energy HUD, and a PS1-style
post pass (grain, scanlines, vignette) before a nearest-neighbour upscale.

    python mockup.py                 # interactive window
    python mockup.py --out shot.png  # headless screenshot
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pygame

HERE = Path(__file__).parent
SHEET = HERE / "MSX-UnDeadPeopleEdition.png"

SHEET_CELL = 32          # native glyph size in the sheet
CELL = 16                # internal-surface cell size
COLS, ROWS = 40, 26      # internal surface in cells
SCALE = 2                # nearest-neighbour upscale factor

MAP_W, MAP_H = 26, 21    # map viewport, top-left at (0, 1)
MAP_X, MAP_Y = 0, 1
DIVIDER_X = 26
PANEL_X, PANEL_W = 27, 13
LOG_Y = 23

# --- palette ---------------------------------------------------------------
# Desaturated, slightly sick. Everything else is this multiplied by light.
BLACK = (0, 0, 0)
UI_DIM = (104, 116, 100)
UI_TEXT = (172, 188, 156)
UI_HOT = (198, 78, 56)
UI_WARM = (196, 172, 116)

# --- the scene -------------------------------------------------------------
# 26 x 21. Mansion, first floor, entrance hall and the west servant passage.
LEVEL = [
    "##########################",
    "#......#........#........#",
    "#.=..=.#...TT...#..%%,...#",
    "#......'........+....z...#",
    "#..&...#...\"\"...#........#",
    "#......#...\"\"...#...=....#",
    "####'#####..''..#........#",
    "#....#....##..##+####'####",
    "#.!..'....#....#.........#",
    "#....#....#.@..'....d....#",
    "#....#....#....#.........#",
    "##'###....######.....=...#",
    "#.....*...#........*.....#",
    "#.....,...+..............#",
    "#..k..,...#.......##+#####",
    "#.....,...#.......#......#",
    "####,#####.......##...z..#",
    "#...,.....'.......#...%..#",
    "#.>.,........<....#../...#",
    "#...,.............#......#",
    "##########################",
]

# char -> (cp437 glyph, colour, blocks_light)
TILES = {
    "#": ("█", (96, 88, 78), True),    # wall
    ".": (".", (82, 78, 78), False),          # floorboards
    ",": ("░", (88, 78, 62), False),     # debris
    '"': ("░", (132, 58, 52), False),    # carpet
    "=": ("≡", (150, 110, 62), True),     # furniture
    "T": ("Φ", (150, 146, 132), True),   # statue / portrait
    "+": ("+", (158, 116, 62), True),         # closed door
    "'": ("'", (128, 98, 56), False),         # open door
    "!": ("!", (198, 62, 58), False),         # first aid
    "/": ("/", (150, 152, 160), False),       # weapon
    "&": ("¶", (190, 180, 150), False),  # document
    "k": ("§", (204, 172, 80), False),   # key
    "%": ("%", (124, 28, 30), False),         # gore
    "*": ("☼", (236, 194, 112), False),  # candle
    "<": ("<", (172, 172, 182), False),       # entrance
    ">": (">", (172, 172, 182), False),       # stairs down
    "@": ("@", (244, 238, 214), False),       # player
    "z": ("z", (132, 152, 92), False),        # shambler
    "d": ("d", (152, 112, 72), False),        # hound
}

ENTITIES = set("@zd")
FLOOR_UNDER_ENTITY = "."

PLAYER = (12, 9)
LIGHTS = [(12, 9, 9.5, 1.0)]  # lantern; candles are appended from the map
MEMORY_RADIUS = 14.0
MEMORY_LEVEL = 0.30
LIGHT_LEVELS = 7

BAYER = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
]


# Python's cp437 codec maps 0x00-0x1F to C0 controls, so the graphical glyphs
# that live down there need their indices spelled out.
LOW_GLYPHS = {"☼": 15, "¶": 20, "§": 21, "↑": 24, "→": 26, "▲": 30}


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


def scale_colour(colour, amount: float):
    return tuple(min(255, max(0, int(channel * amount))) for channel in colour)


def light_at(x: float, y: float, flicker: float) -> float:
    total = 0.0
    for lx, ly, radius, strength in LIGHTS:
        radius *= flicker if strength >= 1.0 else (2.0 - flicker)
        distance = math.hypot(x - lx, y - ly)
        if distance >= radius:
            continue
        total += strength * (1.0 - (distance / radius) ** 1.15)
    return min(1.0, total)


def dither(value: float, x: int, y: int) -> float:
    """Ordered-dither a brightness into LIGHT_LEVELS bands."""
    threshold = (BAYER[y % 4][x % 4] + 0.5) / 16.0
    return min(1.0, math.floor(value * LIGHT_LEVELS + threshold) / LIGHT_LEVELS)


def draw_map(surface: pygame.Surface, font: Font, flicker: float) -> None:
    px, py = PLAYER
    for y, row in enumerate(LEVEL):
        for x, char in enumerate(row):
            if math.hypot(x - px, y - py) > MEMORY_RADIUS:
                continue

            lit = light_at(x, y, flicker)
            level = dither(max(lit, MEMORY_LEVEL), x, y)
            if level <= 0.0:
                continue

            if char in ENTITIES:
                glyph, colour, _ = TILES[FLOOR_UNDER_ENTITY]
                font.blit(surface, glyph, MAP_X + x, MAP_Y + y, scale_colour(colour, level))
                glyph, colour, _ = TILES[char]
                # Things outside the lantern read as silhouettes, not sprites.
                if lit < 0.25:
                    colour = scale_colour(colour, 0.85)
                    colour = (colour[0], int(colour[1] * 0.7), int(colour[2] * 0.8))
            else:
                glyph, colour, _ = TILES[char]

            if lit < 0.05:
                # Remembered, unlit: cold and flat.
                grey = sum(colour) / 3.0
                colour = (int(grey * 0.7), int(grey * 0.8), int(grey * 1.0))
            else:
                # Lit: pulled toward lamp-warm in proportion to the light.
                warmth = lit * 0.3
                colour = tuple(
                    int(channel * (1.0 - warmth) + tint * warmth)
                    for channel, tint in zip(colour, (255, 226, 186))
                )

            font.blit(surface, glyph, MAP_X + x, MAP_Y + y, scale_colour(colour, level))


def bar(value: int, maximum: int, width: int = 10) -> str:
    filled = int(round(width * value / maximum))
    return "█" * filled + "░" * (width - filled)


def draw_hud(surface: pygame.Surface, font: Font, tick: int) -> None:
    font.text(surface, "NIGHTSHIFT".ljust(14) + "MANSION 1F", 0, 0, UI_DIM)
    font.text(surface, "SEED 18421", PANEL_X, 0, UI_DIM)

    for y in range(0, LOG_Y - 1):
        font.blit(surface, "│", DIVIDER_X, y, UI_DIM)
    font.text(surface, "─" * COLS, 0, LOG_Y - 1, UI_DIM)
    font.blit(surface, "┴", DIVIDER_X, LOG_Y - 1, UI_DIM)

    rule = "─" * PANEL_W
    panel: list[tuple[str, tuple[int, int, int]]] = [
        ("   STATUS", UI_TEXT),
        (rule, UI_DIM),
        (f"HP {bar(68, 100)}", (176, 62, 58)),
        ("      68/100", UI_TEXT),
        (f"EN {bar(160, 200)}", (120, 174, 168)),
        ("     160/200", UI_TEXT),
        (rule, UI_DIM),
        ("EQUIPPED", UI_DIM),
        (") REVOLVER", UI_TEXT),
        ("  AMMO  4/ 6", UI_WARM),
        (rule, UI_DIM),
        ("CARRY    6/8", UI_DIM),
        ("! FIRST AID", UI_TEXT),
        ("! FIRST AID", UI_TEXT),
        ("¶ NOTE: WING", UI_TEXT),
        ("§ BRASS KEY", UI_WARM),
        ("/ CROWBAR", UI_TEXT),
        ("☼ FLARE", UI_TEXT),
        (rule, UI_DIM),
        (f"NOISE  {bar(3, 10, 5)}", UI_TEXT),
        ("THREAT   HIGH", UI_HOT if (tick // 30) % 2 == 0 else (128, 52, 40)),
    ]
    for offset, (line, colour) in enumerate(panel):
        font.text(surface, line[:PANEL_W], PANEL_X, MAP_Y + offset, colour)

    log = [
        ("> The east door is locked. Ornate lock.", UI_TEXT),
        ("> Something drags itself, west passage.", UI_HOT),
        ("MOVE 100  SPRINT 150  FIRE 100  SRCH 125", UI_DIM),
    ]
    for offset, (line, colour) in enumerate(log):
        font.text(surface, line[:COLS], 0, LOG_Y + offset, colour)


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
            amount = max(0.0, 1.0 - 0.52 * distance**1.8)
            shade = int(255 * amount)
            surface.fill((shade, shade, shade), (x, y, 8, 1))
    return surface


def make_grain(size, rng: random.Random) -> pygame.Surface:
    surface = pygame.Surface(size).convert()
    for y in range(size[1]):
        for x in range(0, size[0], 2):
            value = rng.randint(0, 9)
            surface.fill((value, value, value), (x, y, 2, 1))
    return surface


def build_lights() -> None:
    for y, row in enumerate(LEVEL):
        for x, char in enumerate(row):
            if char == "*":
                LIGHTS.append((x, y, 4.0, 0.7))


def render_frame(font, internal, scanlines, vignette, grain, tick) -> pygame.Surface:
    flicker = 1.0 + 0.045 * math.sin(tick * 0.19) + 0.02 * math.sin(tick * 0.71)

    internal.fill(BLACK)
    draw_map(internal, font, flicker)
    draw_hud(internal, font, tick)

    offset = (tick * 7) % grain.get_height()
    internal.blit(grain, (0, -offset), special_flags=pygame.BLEND_RGB_ADD)
    internal.blit(grain, (0, grain.get_height() - offset), special_flags=pygame.BLEND_RGB_ADD)

    frame = pygame.transform.scale(internal, (COLS * CELL * SCALE, ROWS * CELL * SCALE))
    frame.blit(scanlines, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
    frame.blit(vignette, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="render one frame headless and exit")
    parser.add_argument("--seed", type=int, default=18421)
    args = parser.parse_args()

    if args.out:
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    pygame.init()
    size = (COLS * CELL * SCALE, ROWS * CELL * SCALE)
    pygame.display.set_mode(size)
    pygame.display.set_caption("Project Nightshift - mockup")

    rng = random.Random(args.seed)
    build_lights()
    font = Font(SHEET)
    internal = pygame.Surface((COLS * CELL, ROWS * CELL)).convert()
    scanlines = make_scanlines(size)
    vignette = make_vignette(size)
    grain = make_grain((COLS * CELL, ROWS * CELL), rng)

    if args.out:
        frame = render_frame(font, internal, scanlines, vignette, grain, 12)
        pygame.image.save(frame, str(args.out))
        pygame.quit()
        print(f"wrote {args.out}")
        return

    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()
    tick = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
        screen.blit(render_frame(font, internal, scanlines, vignette, grain, tick), (0, 0))
        pygame.display.flip()
        tick += 1
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
