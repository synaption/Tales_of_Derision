"""Sprite slicing for the "Cozy Town" / "Interior" free tilesets.

Both sheets are 16 px art.  Rects below were measured off the sheets, so the
crops are tight -- a prop's world size comes from its own art plus its
``art_scale`` (2x by default), never from a hardcoded tile assumption.
"""

from __future__ import annotations

import random
from pathlib import Path

import pygame

from render import ART_SCALE, PixelArt


ROOT = Path(__file__).resolve().parent
FRUIT_DIR = ROOT / "Fruits_Separated"
TOWN_SHEET = ROOT / "reference" / "town free" / "town free" / "free version.png"
INTERIOR_SHEET = ROOT / "reference" / "Interior free" / "interior free" / "interior free.png"

TOWN_RECTS = {
    "house":      (3, 7, 88, 89),
    "fountain":   (133, 4, 38, 28),
    "stall_roof": (96, 16, 48, 32),
    "stall_post": (96, 0, 14, 16),
    "case":       (112, 48, 32, 32),
    "anvil":      (117, 82, 24, 14),
    "window":     (146, 49, 13, 14),
    "bed":        (147, 64, 27, 32),
    "grave":      (114, 97, 12, 15),
    "rock_big":   (129, 97, 14, 15),
    "rock_flat":  (144, 100, 16, 12),
    "fence":      (16, 96, 16, 16),
    "fence_end":  (2, 96, 14, 16),
    "fence_post": (80, 96, 6, 32),
    "gate":       (48, 128, 27, 16),
    "tile_path":  (96, 48, 16, 16),
    "tile_wood":  (96, 64, 16, 16),
    "tile_moss":  (96, 80, 16, 16),
    "bird_a":     (113, 118, 13, 10),
    "bird_b":     (128, 119, 14, 9),
    "bird_c":     (144, 120, 15, 8),
}

INTERIOR_RECTS = {
    "sofa":        (1, 11, 30, 21),
    "sofa_small":  (32, 17, 16, 15),
    "chair_back":  (49, 13, 13, 19),
    "chair_right": (65, 13, 12, 19),
    "chair_left":  (83, 13, 12, 19),
    "chair_front": (97, 16, 13, 16),
    "stove":       (1, 32, 14, 16),
    "plant":       (98, 32, 13, 16),
    "rug":         (19, 39, 10, 6),
    "rug_red":     (67, 39, 10, 6),
    "flowers":     (82, 36, 11, 11),
    "art_rose":    (0, 49, 16, 15),
    "picture":     (19, 55, 10, 6),
    "picture_alt": (51, 55, 10, 6),
    "shelf":       (84, 54, 24, 6),
    "poster":      (1, 64, 15, 16),
    "poster_alt":  (19, 64, 13, 16),
    "wall":        (112, 0, 16, 32),
    "wall_warm":   (128, 0, 16, 32),
    "wall_brick":  (144, 16, 16, 16),
    "floor":       (112, 32, 16, 16),
    "floor_alt":   (128, 32, 16, 16),
}


def _slice(sheet: pygame.Surface, rects: dict, scale: int = ART_SCALE) -> dict[str, PixelArt]:
    out = {}
    for name, (x, y, w, h) in rects.items():
        piece = pygame.Surface((w, h), pygame.SRCALPHA)
        piece.blit(sheet, (0, 0), pygame.Rect(x, y, w, h))
        out[name] = PixelArt(piece, scale)
    return out


def _grass_tiles(count: int = 4) -> list[PixelArt]:
    """The free sheets ship no ground tile, so grow one in the same 16 px idiom."""
    rng = random.Random(11)
    base, dark, light = (118, 186, 106), (96, 166, 90), (146, 204, 118)
    tiles = []
    for _ in range(count):
        surf = pygame.Surface((16, 16))
        surf.fill(base)
        for _ in range(26):
            x, y = rng.randrange(16), rng.randrange(16)
            surf.set_at((x, y), dark if rng.random() < .55 else light)
        for _ in range(3):                       # a couple of little blades
            x, y = rng.randrange(1, 15), rng.randrange(2, 15)
            surf.set_at((x, y), dark)
            surf.set_at((x, y - 1), dark)
        tiles.append(PixelArt(surf, ART_SCALE))
    return tiles


def _flower_tiles() -> list[PixelArt]:
    out = []
    for petal in ((246, 172, 179), (255, 232, 119), (214, 174, 240)):
        surf = pygame.Surface((7, 8), pygame.SRCALPHA)
        pygame.draw.line(surf, (76, 151, 80), (3, 7), (3, 3))
        for dx, dy in ((3, 1), (2, 2), (4, 2), (3, 3)):
            surf.set_at((dx, dy), petal)
        surf.set_at((3, 2), (255, 252, 224))
        out.append(PixelArt(surf, ART_SCALE))
    return out


class Assets:
    def __init__(self) -> None:
        town_sheet = pygame.image.load(TOWN_SHEET).convert_alpha()
        interior_sheet = pygame.image.load(INTERIOR_SHEET).convert_alpha()
        self.town = _slice(town_sheet, TOWN_RECTS)
        self.interior = _slice(interior_sheet, INTERIOR_RECTS)
        self.grass = _grass_tiles()
        self.flowers = _flower_tiles()
        self.birds = [self.town[n] for n in ("bird_a", "bird_b", "bird_c")]
        self.birds_left = [art.flipped() for art in self.birds]

        self.fruit_paths = sorted(FRUIT_DIR.glob("*.png"))
        if not self.fruit_paths:
            raise SystemExit(f"No fruit PNGs found in {FRUIT_DIR}")
        # Fruit art is 32 px -- same 2x treatment, so a villager is two tiles tall.
        self.fruits = {p.stem: PixelArt(pygame.image.load(p).convert_alpha(), ART_SCALE)
                       for p in self.fruit_paths}
