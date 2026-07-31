"""Sprites for the juice bench, out of the tileset the project already ships.

The bench started with text glyphs, which are fine for reading a grid and
useless for judging animation: a `k` has no silhouette to squash, no facing to
lunge along and no mass to topple. Sprites make every effect in `juicefx.py`
legible in a way letters never do.

This reads `gfx/tilesets/Hexany/creatures_transparent.png`, which is already in
the repo and already referenced by `gfx/tilesets/pygame_tileset_config.json` --
no new asset and no download. It is a 16x20 grid of 16px tiles; most of them
are near-white silhouettes, which is exactly what you want, because a white
sprite multiplied by a colour is a tinted sprite and one sheet becomes as many
monsters as you have colours.

Two rules the rest of the bench depends on:

* **Nearest-neighbour, always.** Pixel art run through `smoothscale` turns to
  porridge, and the squash-and-stretch effect scales sprites every single
  frame. `pygame.transform.scale` and `.rotate` are both nearest and both stay
  crisp, so a stretched sprite still looks drawn rather than melted.
* **Integer base scale.** 16px sources at a 32px tile is exactly 2x, so the
  resting sprite has no resampling artefacts at all. The squash then deforms
  something already clean.

Everything is cached on the way through: a tint is a surface copy and a fill,
which is not free, and the bench asks for the same handful of sprites sixty
times a second.

If the sheet is missing the loader falls back to lettered glyphs, so the bench
still runs on a checkout without the art.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import pygame

#: Repo-root-relative location of the sheet, resolved from this file.
SHEET_PATH = os.path.join("gfx", "tilesets", "Hexany", "creatures_transparent.png")
SOURCE_TILE = 16


def _repo_root() -> str:
    """experimental/juice/tiles.py -> the repo root, two levels up."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        os.pardir, os.pardir))


class SpriteSheet:
    """A sliced, scaled, tinted, cached view of one tile sheet.

    `sprite(col, row, tint)` is the whole interface. Sprites are keyed by all
    three, so asking for the same monster twice costs a dict lookup.
    """

    def __init__(self, tile: int, path: Optional[str] = None,
                 source_tile: int = SOURCE_TILE):
        self.tile = tile
        self.source_tile = source_tile
        self.path = path or os.path.join(_repo_root(), SHEET_PATH)
        self.sheet: Optional[pygame.Surface] = None
        self.available = False
        self._cache: Dict[Tuple[int, int, Optional[Tuple[int, int, int]]],
                          pygame.Surface] = {}
        self._fallback_font: Optional[pygame.font.Font] = None

        if os.path.exists(self.path):
            surf = pygame.image.load(self.path)
            # convert_alpha needs a display; without one the raw surface still
            # blits correctly, just slower. The headless tests rely on this.
            try:
                surf = surf.convert_alpha()
            except pygame.error:
                surf = surf.convert_alpha() if pygame.display.get_init() else surf
            self.sheet = surf
            self.available = True

    @property
    def scale(self) -> int:
        return max(1, self.tile // self.source_tile)

    def sprite(self, col: int, row: int,
               tint: Optional[Tuple[int, int, int]] = None) -> pygame.Surface:
        """One tile, scaled to `tile` px and optionally tinted."""
        key = (col, row, tint)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        if not self.available:
            surf = self._lettered(col, row, tint)
        else:
            s = self.source_tile
            rect = pygame.Rect(col * s, row * s, s, s)
            surf = pygame.Surface((s, s), pygame.SRCALPHA)
            surf.blit(self.sheet, (0, 0), rect)
            if tint is not None:
                # The art is near-white, so a multiply is a tint. Alpha is left
                # at 255 in the mask so the silhouette is untouched.
                surf.fill((*tint, 255), special_flags=pygame.BLEND_RGBA_MULT)
            size = s * self.scale
            surf = pygame.transform.scale(surf, (size, size))

        self._cache[key] = surf
        return surf

    def _lettered(self, col: int, row: int, tint) -> pygame.Surface:
        """Fallback for a checkout with no art: a letter in a box."""
        if self._fallback_font is None:
            self._fallback_font = pygame.font.Font(None, int(self.tile * 0.9))
        ch = chr(ord("a") + (col + row * 3) % 26)
        return self._fallback_font.render(ch, True, tint or (220, 220, 220))


#: The bench's cast, as (column, row) into the creatures sheet. Picked by eye
#: for silhouettes that survive being squashed, stretched and spun: a shape you
#: can still name at 32px with a 25% deformation on it.
SPRITES = {
    "player": (0, 0),        # the blue adventurer, the only pre-coloured one
    "goblin": (0, 4),
    "lizard": (5, 4),
    "ogre": (3, 4),
    "skeleton": (0, 6),
    "wolf": (1, 9),
    "raven": (3, 10),
    "slime": (7, 14),
    "dummy": (6, 14),        # a crate, for the thing that never moves or dies
}
