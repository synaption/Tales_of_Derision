"""Mixed-scale renderer for Fruit Brains.

Three things share one screen here:

* **pixel art** -- every sprite carries its own ``art_scale`` (the town and
  interior tilesets are 16 px art rendered at 2x, so one tile is 32 world
  units).  Nothing assumes a single global pixel size, so a 1x, 2x or 3x asset
  can sit next to each other in the same scene.
* **HD vector art** -- faces, rubber-hose limbs, shadows and bubbles are drawn
  straight into screen space, so they gain real detail as you zoom in instead
  of turning into fat pixels.
* **HD text** -- fonts are rasterised at the on-screen size they end up being,
  never scaled up from a small render.

World units are the common currency: 1 tile = ``TILE`` world units, and the
camera's ``zoom`` is the only thing converting world units to screen pixels.
"""

from __future__ import annotations

import math

import pygame


TILE = 32               # world units per tile (16 px art at 2x)
ART_SCALE = 2           # default pixel-art scale for the new assets
MIN_ZOOM, MAX_ZOOM = 0.5, 5.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class PixelArt:
    """A pixel-art sprite plus the scale it wants to be drawn at."""

    __slots__ = ("surface", "art_scale", "w", "h")

    def __init__(self, surface: pygame.Surface, art_scale: int = ART_SCALE) -> None:
        self.surface = surface
        self.art_scale = art_scale
        self.w = surface.get_width() * art_scale     # size in world units
        self.h = surface.get_height() * art_scale

    def flipped(self) -> "PixelArt":
        return PixelArt(pygame.transform.flip(self.surface, True, False), self.art_scale)


class Camera:
    """Smoothly follows a focus point and smoothly zooms."""

    def __init__(self, view_size: tuple[int, int]) -> None:
        self.view = pygame.Vector2(view_size)
        self.center = pygame.Vector2()
        self.zoom = 1.0
        self.target_zoom = 1.0
        self.bounds: pygame.Rect | None = None
        # Screen area not covered by HUD chrome; speech balloons stay inside it.
        self.safe = pygame.Rect(0, 0, int(view_size[0]), int(view_size[1]))

    def zoom_by(self, factor: float) -> None:
        self.target_zoom = clamp(self.target_zoom * factor, MIN_ZOOM, MAX_ZOOM)

    def snap_to(self, focus: pygame.Vector2) -> None:
        self.center.update(focus)
        self._clamp()

    def update(self, dt: float, focus: pygame.Vector2) -> None:
        # Exponential smoothing: frame-rate independent and never overshoots.
        self.zoom += (self.target_zoom - self.zoom) * (1 - math.exp(-9 * dt))
        self.center += (focus - self.center) * (1 - math.exp(-7 * dt))
        self._clamp()

    def _clamp(self) -> None:
        if self.bounds is None:
            return
        half_w = self.view.x / (2 * self.zoom)
        half_h = self.view.y / (2 * self.zoom)
        b = self.bounds
        self.center.x = (b.centerx if b.width <= half_w * 2
                         else clamp(self.center.x, b.left + half_w, b.right - half_w))
        self.center.y = (b.centery if b.height <= half_h * 2
                         else clamp(self.center.y, b.top + half_h, b.bottom - half_h))

    def to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        return ((wx - self.center.x) * self.zoom + self.view.x / 2,
                (wy - self.center.y) * self.zoom + self.view.y / 2)

    def to_world(self, sx: float, sy: float) -> pygame.Vector2:
        return pygame.Vector2((sx - self.view.x / 2) / self.zoom + self.center.x,
                              (sy - self.view.y / 2) / self.zoom + self.center.y)

    def visible(self, pad: float = TILE) -> pygame.Rect:
        half_w = self.view.x / (2 * self.zoom) + pad
        half_h = self.view.y / (2 * self.zoom) + pad
        return pygame.Rect(round(self.center.x - half_w), round(self.center.y - half_h),
                           round(half_w * 2), round(half_h * 2))


class ArtCache:
    """Nearest-neighbour rescales, memoised by the exact on-screen size."""

    def __init__(self, limit: int = 512) -> None:
        self._cache: dict[tuple[int, int, int], pygame.Surface] = {}
        self._limit = limit

    def get(self, art: PixelArt, w: int, h: int) -> pygame.Surface:
        key = (id(art.surface), w, h)
        scaled = self._cache.get(key)
        if scaled is None:
            if len(self._cache) > self._limit:
                self._cache.clear()
            scaled = pygame.transform.scale(art.surface, (w, h))
            self._cache[key] = scaled
        return scaled


CACHE = ArtCache()


def blit_art(surface: pygame.Surface, cam: Camera, art: PixelArt, wx: float, wy: float) -> None:
    """Draw pixel art at a world position, seam-free at fractional zoom.

    Both corners are rounded independently so neighbouring tiles always share
    an edge -- rounding only the origin leaves 1 px cracks when zoom is not a
    whole number.
    """
    x0, y0 = cam.to_screen(wx, wy)
    x1, y1 = cam.to_screen(wx + art.w, wy + art.h)
    left, top = round(x0), round(y0)
    w, h = round(x1) - left, round(y1) - top
    if w <= 0 or h <= 0 or left > surface.get_width() or top > surface.get_height():
        return
    if left + w < 0 or top + h < 0:
        return
    surface.blit(CACHE.get(art, w, h), (left, top))


# --- HD layer ---------------------------------------------------------------

_FONTS: dict[tuple[str | None, int], pygame.font.Font] = {}


def hd_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Fonts are rasterised at their final on-screen size, so text stays sharp."""
    size = max(8, min(200, round(size)))
    key = ("bold" if bold else None, size)
    font = _FONTS.get(key)
    if font is None:
        font = pygame.font.Font(None, size)
        font.set_bold(bold)
        _FONTS[key] = font
    return font


def hd_text(surface: pygame.Surface, text: str, pos, size: int, color, *,
            center: bool = False, bold: bool = False, shadow=None) -> pygame.Rect:
    font = hd_font(size, bold)
    image = font.render(text, True, color)          # antialiased
    rect = image.get_rect(center=pos) if center else image.get_rect(topleft=pos)
    if shadow is not None:
        ghost = font.render(text, True, shadow)
        surface.blit(ghost, rect.move(max(1, size // 18), max(1, size // 18)))
    surface.blit(image, rect)
    return rect


def hd_line(surface: pygame.Surface, color, points, width: float) -> None:
    """Round-ended stroke, drawn in screen space so it sharpens as you zoom."""
    w = max(1, round(width))
    pts = [(round(x), round(y)) for x, y in points]
    if len(pts) > 1:
        pygame.draw.lines(surface, color, False, pts, w)
    radius = w // 2
    if radius:
        for point in (pts[0], pts[-1]):
            pygame.draw.circle(surface, color, point, radius)


def hd_arc(surface: pygame.Surface, color, rect, start: float, stop: float, width: float) -> None:
    pygame.draw.arc(surface, color, rect, start, stop, max(1, round(width)))


def hd_circle(surface: pygame.Surface, color, center, radius: float, width: float = 0) -> None:
    pygame.draw.circle(surface, color, (round(center[0]), round(center[1])),
                       max(1, round(radius)), 0 if width == 0 else max(1, round(width)))


def hd_ellipse(surface: pygame.Surface, color, rect, width: float = 0) -> None:
    r = pygame.Rect(round(rect[0]), round(rect[1]), max(1, round(rect[2])), max(1, round(rect[3])))
    pygame.draw.ellipse(surface, color, r, 0 if width == 0 else max(1, round(width)))
