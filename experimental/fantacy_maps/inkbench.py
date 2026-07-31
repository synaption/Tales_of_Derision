"""The ink workbench: an application for looking at the effect in `inkfx`.

Everything here is about driving and inspecting the effect rather than being
it -- the window and its resolution menu, the slider panel, the help card, the
demo page of artwork, the featured creature and the sprite strip, and the key
bindings that tie them together. None of it is needed to use the effect; it is
needed to judge one.

Run it with `python3 inkGL.py`, which is a two-line launcher for this.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import moderngl
import pygame

from inkfx import (
    INK_DRY_SECONDS,
    INK_SUPERSAMPLE,
    PAGE_WET_FLOOR,
    PANEL_EDGE,
    PANEL_KEY,
    PANEL_TEXT,
    PANEL_TITLE,
    PEN_SPEED,
    InkCanvas,
    InkSettings,
    Panel,
    ParchmentInkRenderer,
    SpriteReveal,
    View,
    load_ink_image,
    make_panel_surface,
    open_window,
    restart_on_the_graphics_card,
    sprite_reveal,
)


WINDOW_SIZE = (1000, 700)
# The demo artwork is composed against this page and scaled to whatever window
# the menu is set to, so every resolution shows the same layout.
DESIGN_SIZE = (1000, 700)
# How long an idle page may go without being redrawn. The frame is only drawn
# when something has changed, so this is purely insurance against a repaint
# nobody told us about; once a second is often enough to be invisible and rare
# enough to cost nothing.
IDLE_REDRAW_FRAMES = 60

REPO_ROOT = Path(__file__).resolve().parents[2]
SPRITE_SHEET_PATH = (
    REPO_ROOT / "gfx" / "tilesets" / "Hexany" / "monochrome_32x32_transparent.png"
)
SPRITE_TILE_SIZE = 32

FEATURE_IMAGE_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "inv"
    / "creature_059_smoothed_512.png"
)
FEATURE_IMAGE_HEIGHT = 160

@dataclass
class DisplayOptions:
    """What the menu can change but a running renderer cannot.

    Both of these decide how much texture gets allocated, so changing either
    means tearing the renderer down and building it again. They are kept apart
    from `InkSettings` for that reason: everything there is live, everything
    here costs a rebuild, which is why the options page has an Apply row.
    """

    size: tuple[int, int] = WINDOW_SIZE
    supersample: int = INK_SUPERSAMPLE
    fullscreen: bool = False

    SIZES = (
        (800, 560),
        (1000, 700),
        (1280, 720),
        (1600, 900),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
    )
    SUPERSAMPLES = (1, 2, 3, 4)
    MODES = (False, True)

    # The list is deliberately not filtered against the desktop. What a window
    # manager will grant is not reliably knowable in advance -- fullscreen tends
    # to land on the desktop resolution whatever it was asked for -- so the size
    # actually granted is read back afterwards instead of guessed at here.
    def choices(self, field: str) -> tuple:
        """The values `field` can take, in the order the menu steps through."""
        if field == "size":
            return self.SIZES
        if field == "fullscreen":
            return self.MODES
        return self.SUPERSAMPLES

    def cycle(self, field: str, step: int) -> None:
        choices = self.choices(field)
        current = getattr(self, field)
        if current in choices:
            index = choices.index(current)
        elif field == "size":
            # The granted size need not be one that was offered, so step off
            # the nearest entry by area rather than back to the top of the list.
            index = min(
                range(len(choices)),
                key=lambda i: abs(
                    choices[i][0] * choices[i][1] - current[0] * current[1]
                ),
            )
        else:
            index = 0
        setattr(self, field, choices[(index + step) % len(choices)])

    @staticmethod
    def _format(field: str, value) -> str:
        if field == "size":
            return f"{value[0]} x {value[1]}"
        if field == "fullscreen":
            return "Fullscreen" if value else "Windowed"
        return "off" if value == 1 else f"{value}x"

    def text(self, field: str) -> str:
        return self._format(field, getattr(self, field))

    def widest_text(self, field: str) -> str:
        """The longest reading a field can show, for sizing its column."""
        return max(
            (self._format(field, value) for value in self.choices(field)), key=len
        )


class SpriteSheet:
    """A grid of equally sized tiles cut out of one image.

    The Hexany monochrome sheet is pure white on transparency, so a tile's
    alpha channel drops straight into the ink mask and the sprite is lit as
    though it had been stamped onto the parchment in the same ink as the
    hand-drawn strokes.
    """

    def __init__(self, path: Path, tile_size: int = SPRITE_TILE_SIZE) -> None:
        source = pygame.image.load(str(path))

        # Blitting onto an SRCALPHA surface resolves the palette + colour key
        # of an indexed PNG into real per-pixel alpha, and unlike
        # convert_alpha() it does not need a display surface.
        self.surface = pygame.Surface(source.get_size(), pygame.SRCALPHA)
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(source, (0, 0))

        self.tile_size = tile_size
        sheet_width, sheet_height = self.surface.get_size()
        self.columns = sheet_width // tile_size
        self.rows = sheet_height // tile_size
        self._scaled_cache: dict[tuple[int, int], pygame.Surface] = {}

    @property
    def count(self) -> int:
        return self.columns * self.rows

    def tile_rect(self, index: int) -> pygame.Rect:
        index %= self.count
        column = index % self.columns
        row = index // self.columns
        return pygame.Rect(
            column * self.tile_size,
            row * self.tile_size,
            self.tile_size,
            self.tile_size,
        )

    def tile(self, index: int, scale: int = 1) -> pygame.Surface:
        """Return one tile, optionally point-scaled to keep pixels crisp."""
        index %= self.count
        key = (index, scale)
        cached = self._scaled_cache.get(key)
        if cached is not None:
            return cached

        tile = pygame.Surface(
            (self.tile_size, self.tile_size), pygame.SRCALPHA
        )
        tile.fill((0, 0, 0, 0))
        tile.blit(self.surface, (0, 0), self.tile_rect(index))

        if scale != 1:
            size = (self.tile_size * scale, self.tile_size * scale)
            tile = pygame.transform.scale(tile, size)

        self._scaled_cache[key] = tile
        return tile

    def draw(
        self,
        target: pygame.Surface,
        index: int,
        position: tuple[int, int],
        scale: int = 1,
        opacity: int = 255,
    ) -> pygame.Rect:
        """Stamp one tile onto `target`, treating it as ink of a given density."""
        tile = self.tile(index, scale)
        if opacity != 255:
            tile = tile.copy()
            tile.fill(
                (255, 255, 255, opacity),
                special_flags=pygame.BLEND_RGBA_MULT,
            )
        return target.blit(tile, position)


def load_sprite_sheet(path: Path = SPRITE_SHEET_PATH) -> SpriteSheet | None:
    """Load the sprite sheet, or return None so the demo still runs without it."""
    try:
        return SpriteSheet(path)
    except (pygame.error, FileNotFoundError) as exc:
        print(f"Could not load sprite sheet {path}: {exc}", file=sys.stderr)
        return None


def design_unit(size: tuple[int, int]) -> float:
    """Page pixels per design-space unit for a page of this size.

    The smaller of the two ratios, so a composition laid out against
    `DESIGN_SIZE` is letterboxed into an oddly shaped window rather than
    cropped by it.
    """
    return min(size[0] / DESIGN_SIZE[0], size[1] / DESIGN_SIZE[1])


def feature_placement(size: tuple[int, int]) -> tuple[tuple[float, float], float]:
    """Where the featured creature sits on the page, in design-space units.

    One source of truth for two callers who must agree exactly: the artwork
    that draws it, and the tracer that has to put the pen in the same place.
    The page's own size in design units comes out of `design_unit`, so the
    creature stays the same distance from the corner in any window shape.
    """
    unit = design_unit(size)
    return (
        (size[0] / unit - 178.0, size[1] / unit - 150.0),
        float(FEATURE_IMAGE_HEIGHT),
    )


def feature_rect(size: tuple[int, int], scale: int) -> pygame.Rect | None:
    """Where the featured creature goes on the mask, aligned to the page grid.

    One source of truth for two things that have to agree to the pixel: the
    printed artwork, and the animation that uncovers the same sprite in its
    place. Snapped so a whole number of mask texels falls inside every page
    pixel, because the animation keeps a page-resolution copy of its timings
    beside the mask-resolution one and the two can only line up on a boundary.
    """
    unit = scale * design_unit(size)
    centre, height = feature_placement(size)
    image = load_ink_image(FEATURE_IMAGE_PATH, max(1, round(height * unit)))
    if image is None:
        return None

    rect = image.get_rect()
    rect.center = (round(centre[0] * unit), round(centre[1] * unit))
    rect.left -= rect.left % scale
    rect.top -= rect.top % scale
    return rect


def make_demo_ink(
    size: tuple[int, int],
    sprites: SpriteSheet | None = None,
    sprite_page: int = 0,
    scale: int = 1,
    include_feature: bool = True,
) -> pygame.Surface:
    """Create anti-aliased ink artwork; its alpha channel becomes the ink mask.

    Everything below is measured in the design space named by `DESIGN_SIZE` and
    converted on its way to the drawing call, by `at` for a position and `span`
    for a length. That absorbs two things at once: the window can be any of the
    sizes the menu offers, and the mask is drawn at `scale` times the page. Type
    and line weights go through the same conversion, so the artwork is laid out
    at the final resolution rather than enlarged afterwards and gains real
    detail instead of a smoother version of the same staircase.
    """
    page_width, page_height = size
    width, height = page_width * scale, page_height * scale
    ink_mask = pygame.Surface((width, height), pygame.SRCALPHA)
    ink_mask.fill((0, 0, 0, 0))

    design_width = DESIGN_SIZE[0]
    unit = scale * design_unit(size)

    def at(x: float, y: float) -> tuple[int, int]:
        return (round(x * unit), round(y * unit))

    def span(value: float) -> int:
        return max(1, round(value * unit))

    title_font = pygame.font.SysFont("serif", span(58), bold=True)
    body_font = pygame.font.SysFont("serif", span(28))
    small_font = pygame.font.SysFont("serif", span(20), italic=True)

    def draw_text(
        text: str,
        font: pygame.font.Font,
        position: tuple[int, int],
        opacity: int = 255,
    ) -> None:
        rendered = font.render(text, True, (255, 255, 255))
        rendered.set_alpha(opacity)
        ink_mask.blit(rendered, at(*position))

    draw_text("The Cartographer's Ledger", title_font, (90, 72))
    draw_text(
        "Move the mouse: one light, two shader materials.",
        body_font,
        (94, 152),
    )
    draw_text(
        "Rough parchment scatters light; fresh ink forms a glossy coat.",
        small_font,
        (96, 196),
    )

    white = (255, 255, 255, 255)
    rule_y = at(0, 245)[1]
    pygame.draw.line(
        ink_mask, white, (span(95), rule_y), (width - span(95), rule_y), span(5)
    )
    pygame.draw.circle(ink_mask, white, (width // 2, rule_y), span(13), span(3))

    points: list[tuple[int, int]] = []
    for x in range(120, design_width - 120, 8):
        y = 390 + int(52 * math.sin(x * 0.018)) + int(16 * math.sin(x * 0.053))
        points.append(at(x, y))
    # Stretch the wave across whatever width the page turned out to be.
    stretch = (width - 2 * span(120)) / max(1, points[-1][0] - points[0][0])
    points = [
        (round(span(120) + (x - points[0][0]) * stretch), y) for x, y in points
    ]
    pygame.draw.lines(ink_mask, white, False, points, span(12))

    # The rose sits up beside the title so the lower right stays free for the
    # featured creature and the sprite strip.
    center = (width - span(145), at(0, 320)[1])
    pygame.draw.circle(ink_mask, white, center, span(62), span(5))
    for angle in range(0, 360, 45):
        vector = pygame.Vector2(0, -span(54)).rotate(angle)
        endpoint = (round(center[0] + vector.x), round(center[1] + vector.y))
        pygame.draw.line(ink_mask, white, center, endpoint, span(5))
    pygame.draw.circle(ink_mask, white, center, span(10))

    # Thick pools produce broad dark shapes and very visible coat highlights.
    for position, radius in [((430, 520), 30), ((512, 496), 18), ((594, 528), 24)]:
        pygame.draw.circle(ink_mask, white, at(*position), span(radius))

    # A few pressure-varying pen strokes.
    for index in range(5):
        y = 292 + index * 18
        start = at(112, y)
        end = at(340 + index * 34, y + random.Random(index).randint(-5, 5))
        pygame.draw.aaline(ink_mask, white, start, end)
        pygame.draw.line(ink_mask, white, start, end, span(2 + index // 2))

    # Left out when the creature is about to be drawn by the pen instead: it
    # cannot be animated on to a page that already has it.
    if include_feature:
        placed = feature_rect(size, scale)
        if placed is not None:
            feature = load_ink_image(FEATURE_IMAGE_PATH, placed.height)
            ink_mask.blit(feature, placed, special_flags=pygame.BLEND_RGBA_MAX)

    if sprites is not None:
        per_page = sprites.columns
        pages = max(1, sprites.count // per_page)
        page = sprite_page % pages
        first_tile = page * per_page

        draw_text(
            f"Hexany 32x32 - tiles {first_tile}-{first_tile + per_page - 1}"
            f" of {sprites.count}",
            small_font,
            (100, 538),
        )

        # Point-scaled pixel art, so the tiles keep their hard edges and simply
        # resolve them on more texels. The factor stays whole for that reason.
        tile_scale = max(1, round(2 * unit))
        spacing = sprites.tile_size * tile_scale + span(12)
        for column in range(per_page):
            sprites.draw(
                ink_mask,
                first_tile + column,
                (span(100) + column * spacing, at(0, 566)[1]),
                scale=tile_scale,
            )

    return ink_mask


class HelpOverlay(Panel):
    """A card of controls, shown over the page when the user presses `?`."""

    # None starts a new group of related keys.
    ROWS: tuple[tuple[str, str] | None, ...] = (
        ("Drag", "draw wet ink"),
        ("Right-drag", "erase"),
        ("Shift+Wheel", "brush size"),
        None,
        ("Middle-drag", "pan the page"),
        ("Wheel", "zoom about the cursor"),
        ("0 / Home", "reset the view"),
        None,
        ("Move mouse", "move the lamp"),
        ("- / =", "lamp height"),
        ("W", "re-wet the whole page"),
        ("[ / ]", "hold the wetness, pausing the dry-down"),
        None,
        ("U / Ctrl+Z", "undo a stroke"),
        ("C", "clear every stroke"),
        ("T", "draw the creature, traced from its sprite"),
        ("Arrows", "sprite sheet page"),
        ("R", "regenerate the parchment"),
        None,
        ("Tab", "sliders for drying and the oil slick"),
        ("1 - 5", "final, normals, roughness, ink, wetness"),
        ("?", "hide this card"),
        ("Esc", "menu: display options and quit"),
    )

    TITLE = "Controls"
    PADDING = 24
    LINE_HEIGHT = 24
    GROUP_GAP = 12

    def __init__(self, context: moderngl.Context, window_size: tuple[int, int]) -> None:
        super().__init__(context, window_size, self._render_panel())

    def _render_panel(self) -> pygame.Surface:
        title_font = pygame.font.SysFont("serif", 22, bold=True)
        key_font = pygame.font.SysFont("monospace", 14, bold=True)
        text_font = pygame.font.SysFont("serif", 17)

        rows = [row for row in self.ROWS if row is not None]
        key_column = max(key_font.size(keys)[0] for keys, _ in rows) + 20
        body_width = max(text_font.size(text)[0] for _, text in rows)
        title = title_font.render(self.TITLE, True, PANEL_TITLE)

        width = self.PADDING * 2 + max(key_column + body_width, title.get_width())
        height = (
            self.PADDING * 2
            + title.get_height()
            + 14
            + len(rows) * self.LINE_HEIGHT
            + sum(self.GROUP_GAP for row in self.ROWS if row is None)
        )

        panel = make_panel_surface((width, height))
        panel.blit(title, (self.PADDING, self.PADDING))
        y = self.PADDING + title.get_height() + 14

        for row in self.ROWS:
            if row is None:
                y += self.GROUP_GAP
                continue
            keys, text = row
            panel.blit(key_font.render(keys, True, PANEL_KEY), (self.PADDING, y + 2))
            panel.blit(
                text_font.render(text, True, PANEL_TEXT),
                (self.PADDING + key_column, y),
            )
            y += self.LINE_HEIGHT

        return panel


@dataclass(frozen=True)
class Slider:
    """One tunable value on the slider panel.

    `logarithmic` is for the controls whose useful range is multiplicative -- the
    drying rate and the slick zoom -- so that the middle of the track is the
    default and each half is an equal factor either side of it.
    """

    field: str
    label: str
    minimum: float
    maximum: float
    style: str = "plain"
    logarithmic: bool = False

    def position(self, value: float) -> float:
        """Where a value sits along the track, 0 to 1."""
        value = max(self.minimum, min(self.maximum, value))
        if self.logarithmic:
            return math.log(value / self.minimum) / math.log(
                self.maximum / self.minimum
            )
        return (value - self.minimum) / (self.maximum - self.minimum)

    def value(self, position: float) -> float:
        """The value at a point along the track, 0 to 1."""
        position = max(0.0, min(1.0, position))
        if self.logarithmic:
            return self.minimum * (self.maximum / self.minimum) ** position
        return self.minimum + (self.maximum - self.minimum) * position

    def text(self, value: float) -> str:
        if self.style == "seconds":
            # Read out the time ink takes to dry rather than the bare multiplier.
            return f"{INK_DRY_SECONDS / value:.1f} s"
        if self.style == "percent":
            return f"{value * 100:.0f}%"
        if self.style == "times":
            return f"{value:.2f}x"
        if self.style == "pixels":
            return f"{value:.1f} px"
        if self.style == "speed":
            return f"{value:.0f} px/s"
        if self.style == "rate":
            # Enough decimals to still read as a number four decades down.
            if value < 0.001:
                return f"{value:.4f}"
            if value < 0.1:
                return f"{value:.3f}"
        return f"{value:.2f}"

    def widest_text(self) -> str:
        candidates = (
            self.text(self.minimum),
            self.text(self.maximum),
            self.text(self.value(0.5)),
        )
        return max(candidates, key=len)


class SliderPanel(Panel):
    """Draggable controls for the material parameters, shown on Tab.

    The panel owns its own hit testing: while it is visible it takes any mouse
    event over itself, so tuning a slider never leaves ink on the page and the
    lamp stops following the cursor.
    """

    # The three multiplicative controls span two to four decades, so all of them
    # are logarithmic: a linear track would spend most of its length on values
    # nobody wants and leave the slow end unreachable.
    SLIDERS = (
        Slider(
            "dry_rate",
            "Ink dries in",
            INK_DRY_SECONDS / 60.0,
            INK_DRY_SECONDS / 1.0,
            "seconds",
            logarithmic=True,
        ),
        Slider("slick_swirl", "Swirl speed", 0.0001, 1.5, "rate", logarithmic=True),
        Slider(
            "swirl_dryness", "Slows as it dries", 0.1, 10.0, logarithmic=True
        ),
        Slider("slick_zoom", "Slick size", 0.5, 30.0, "times", logarithmic=True),
        Slider("slick_opacity", "Iridescence", 0.0, 1.5, "percent"),
        Slider("paper_relief", "Weave depth", 0.0, 3.0, "percent"),
        # Below about two and a half pixels a thread the mip chain averages
        # the weave away and only the photograph's broad blotches survive, so
        # the track stops before it gets there.
        Slider("weave_size", "Weave size", 2.5, 40.0, "pixels", logarithmic=True),
        # Log-scaled so that 1.0, the straight line, sits in the middle of the
        # track with an equal factor of slow and fast either side of it.
        Slider("weave_soak", "Weave soaks in", 0.1, 10.0, logarithmic=True),
        # What makes a replay read as a hand rather than a wipe is how the pen
        # races the dry-down, so this belongs beside "Ink dries in": slow enough
        # and the tail is dry before the figure is finished, fast enough and the
        # whole drawing is still glossy when the nib lifts.
        Slider("pen_speed", "Pen speed", 40.0, 4000.0, "speed", logarithmic=True),
    )

    TITLE = "Wet ink"
    PADDING = 18
    ROW_HEIGHT = 42
    TRACK_HEIGHT = 6
    KNOB_RADIUS = 7
    MARGIN = 20

    def __init__(
        self,
        context: moderngl.Context,
        window_size: tuple[int, int],
        settings: InkSettings,
    ) -> None:
        self.title_font = pygame.font.SysFont("serif", 19, bold=True)
        self.label_font = pygame.font.SysFont("serif", 16)
        self.value_font = pygame.font.SysFont("monospace", 14, bold=True)

        label_width = max(
            self.label_font.size(slider.label)[0] for slider in self.SLIDERS
        )
        value_width = max(
            self.value_font.size(slider.widest_text())[0] for slider in self.SLIDERS
        )
        title = self.title_font.render(self.TITLE, True, PANEL_TITLE)

        self._label_width = label_width
        self._value_width = value_width
        self._header = title.get_height() + 12
        # A track long enough to be worth dragging, whatever the labels measure.
        self._track_width = 190
        size = (
            self.PADDING * 2 + label_width + 14 + self._track_width + 14 + value_width,
            self.PADDING * 2 + self._header + len(self.SLIDERS) * self.ROW_HEIGHT,
        )

        # Tucked into the top left, clear of the page's title and the lamp.
        super().__init__(context, window_size, make_panel_surface(size),
                         (self.MARGIN, self.MARGIN))
        self.dragging: Slider | None = None
        self.refresh(settings)

    def _track_rect(self, index: int) -> pygame.Rect:
        """Where a slider's track sits, in window pixels."""
        return pygame.Rect(
            self.rect.left + self.PADDING + self._label_width + 14,
            self.rect.top
            + self.PADDING
            + self._header
            + index * self.ROW_HEIGHT
            + self.ROW_HEIGHT // 2
            - self.TRACK_HEIGHT // 2,
            self._track_width,
            self.TRACK_HEIGHT,
        )

    def refresh(self, settings: InkSettings) -> None:
        """Redraw the panel for the current values and push it to the GPU."""
        panel = make_panel_surface(self.rect.size)
        panel.blit(
            self.title_font.render(self.TITLE, True, PANEL_TITLE),
            (self.PADDING, self.PADDING),
        )

        for index, slider in enumerate(self.SLIDERS):
            value = getattr(settings, slider.field)
            # The track rect is in window pixels; shift it into panel space.
            track = self._track_rect(index).move(-self.rect.left, -self.rect.top)
            row_middle = track.centery

            label = self.label_font.render(slider.label, True, PANEL_TEXT)
            panel.blit(
                label, (self.PADDING, row_middle - label.get_height() // 2)
            )

            filled = round(track.width * slider.position(value))
            pygame.draw.rect(panel, (58, 47, 36, 255), track, border_radius=3)
            if filled:
                pygame.draw.rect(
                    panel,
                    (150, 112, 58, 255),
                    pygame.Rect(track.left, track.top, filled, track.height),
                    border_radius=3,
                )
            pygame.draw.circle(
                panel, PANEL_KEY, (track.left + filled, row_middle), self.KNOB_RADIUS
            )

            reading = self.value_font.render(slider.text(value), True, PANEL_KEY)
            panel.blit(
                reading,
                (
                    self.rect.width - self.PADDING - reading.get_width(),
                    row_middle - reading.get_height() // 2,
                ),
            )

        self.update(panel)

    def _set_from_mouse(self, slider: Slider, index: int, x: int,
                        settings: InkSettings) -> None:
        track = self._track_rect(index)
        position = (x - track.left) / max(1, track.width)
        setattr(settings, slider.field, slider.value(position))
        self.refresh(settings)

    def handle(self, event: pygame.event.Event, settings: InkSettings) -> bool:
        """Consume a mouse event aimed at the panel. Returns True if it was ours."""
        if not self.visible:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if not self.rect.collidepoint(event.pos):
                return False
            if event.button == 1:
                for index, slider in enumerate(self.SLIDERS):
                    # A generous grab area: the whole row, not just the track.
                    row = self._track_rect(index).inflate(
                        self.KNOB_RADIUS * 2, self.ROW_HEIGHT
                    )
                    if row.collidepoint(event.pos):
                        self.dragging = slider
                        self._set_from_mouse(slider, index, event.pos[0], settings)
                        break
            return True

        if event.type == pygame.MOUSEMOTION:
            if self.dragging is not None:
                index = self.SLIDERS.index(self.dragging)
                self._set_from_mouse(self.dragging, index, event.pos[0], settings)
                return True
            return self.rect.collidepoint(event.pos)

        if event.type == pygame.MOUSEBUTTONUP:
            was_dragging = self.dragging is not None
            self.dragging = None
            return was_dragging or self.rect.collidepoint(event.pos)

        return False


@dataclass(frozen=True)
class MenuItem:
    """One row of the menu: a command to run, or an option to step through."""

    label: str
    action: str
    field: str = ""


class MenuPanel(Panel):
    """The Escape menu: resume, display options, quit.

    While it is up it swallows every event, so backing out of it never leaves a
    stray stroke or a moved lamp behind. Keyboard and mouse both drive it, and
    it does its own navigating; the only things it hands back to the caller are
    the two it cannot do itself, quitting and rebuilding the renderer.
    """

    PAGES: dict[str, tuple[MenuItem, ...]] = {
        "main": (
            MenuItem("Resume", "close"),
            MenuItem("Options", "page:options"),
            MenuItem("Quit", "quit"),
        ),
        "options": (
            MenuItem("Display", "cycle", "fullscreen"),
            MenuItem("Resolution", "cycle", "size"),
            MenuItem("Anti-aliasing", "cycle", "supersample"),
            MenuItem("Apply", "apply"),
            MenuItem("Back", "page:main"),
        ),
    }
    TITLES = {"main": "Paused", "options": "Display options"}
    FOOTERS = {
        "main": "Arrows and Enter, or the mouse",
        "options": "Left / Right changes a setting",
    }

    PADDING = 26
    ROW_HEIGHT = 36
    GAP = 40
    HIGHLIGHT = (58, 47, 36, 255)
    DIMMED = (132, 122, 104)

    def __init__(
        self,
        context: moderngl.Context,
        window_size: tuple[int, int],
        display: DisplayOptions,
    ) -> None:
        self.title_font = pygame.font.SysFont("serif", 26, bold=True)
        self.row_font = pygame.font.SysFont("serif", 19)
        self.value_font = pygame.font.SysFont("monospace", 16, bold=True)
        self.footer_font = pygame.font.SysFont("serif", 14, italic=True)

        self.page = "main"
        self.index = 0
        # Set by the caller when a rebuild did not land on what was asked for,
        # so the resolution row snapping back to something else is explained
        # rather than just puzzling.
        self.note = ""

        label_width = max(
            self.row_font.size(item.label)[0]
            for items in self.PAGES.values()
            for item in items
        )
        value_width = max(
            self.value_font.size(display.widest_text(item.field))[0]
            for items in self.PAGES.values()
            for item in items
            if item.field
        )
        title_height = max(
            self.title_font.size(title)[1] for title in self.TITLES.values()
        )
        footer_height = max(
            self.footer_font.size(footer)[1] for footer in self.FOOTERS.values()
        )

        self._header = title_height + 18
        self._footer = footer_height + 16
        # One size fits both pages: the quad behind the card is fixed, and a
        # menu that resized itself as you stepped through it would be worse.
        self._rows = max(len(items) for items in self.PAGES.values())
        size = (
            self.PADDING * 2 + label_width + self.GAP + value_width,
            self.PADDING * 2
            + self._header
            + self._rows * self.ROW_HEIGHT
            + self._footer,
        )
        super().__init__(context, window_size, make_panel_surface(size))
        self.refresh(display, display)

    @property
    def items(self) -> tuple[MenuItem, ...]:
        return self.PAGES[self.page]

    def _row_rect(self, index: int) -> pygame.Rect:
        """Where a row sits, in window pixels."""
        return pygame.Rect(
            self.rect.left + self.PADDING // 2,
            self.rect.top + self.PADDING + self._header + index * self.ROW_HEIGHT,
            self.rect.width - self.PADDING,
            self.ROW_HEIGHT,
        )

    def refresh(self, display: DisplayOptions, active: DisplayOptions) -> None:
        """Redraw for the current page and push it to the GPU.

        `display` is what the menu is editing and `active` is what the renderer
        was actually built with, so Apply can say whether there is anything
        waiting rather than looking like a button that does nothing.
        """
        panel = make_panel_surface(self.rect.size)
        title = self.title_font.render(self.TITLES[self.page], True, PANEL_TITLE)
        panel.blit(title, ((self.rect.width - title.get_width()) // 2, self.PADDING))

        pending = display != active
        for index, item in enumerate(self.items):
            row = self._row_rect(index).move(-self.rect.left, -self.rect.top)
            selected = index == self.index
            waiting = item.action != "apply" or pending

            if selected:
                pygame.draw.rect(panel, self.HIGHLIGHT, row, border_radius=4)

            # Apply is lit while there is something to apply and greyed once
            # there is not, so the row says whether pressing it would do work.
            colour = PANEL_TEXT
            if selected or (item.action == "apply" and pending):
                colour = PANEL_KEY
            if not waiting:
                colour = self.DIMMED
            label = self.row_font.render(item.label, True, colour)
            panel.blit(
                label,
                (row.left + self.PADDING // 2, row.centery - label.get_height() // 2),
            )

            if item.field:
                reading = self.value_font.render(
                    f"< {display.text(item.field)} >", True, colour
                )
                panel.blit(
                    reading,
                    (
                        row.right - self.PADDING // 2 - reading.get_width(),
                        row.centery - reading.get_height() // 2,
                    ),
                )

        if self.note:
            message, colour = self.note, PANEL_KEY
        elif self.page == "options" and pending:
            message, colour = "restart pending", PANEL_KEY
        else:
            message, colour = self.FOOTERS[self.page], self.DIMMED
        footer = self.footer_font.render(message, True, colour)
        panel.blit(
            footer,
            (
                (self.rect.width - footer.get_width()) // 2,
                self.rect.height - self.PADDING - footer.get_height(),
            ),
        )

        self.update(panel)

    def open(self) -> None:
        self.page = "main"
        self.index = 0
        self.note = ""
        self.visible = True

    def _cycle(self, display: DisplayOptions, field: str, step: int) -> None:
        # Whatever the last rebuild had to say about itself is stale the moment
        # the user asks for something different.
        self.note = ""
        display.cycle(field, step)

    def _move(self, step: int) -> None:
        self.index = (self.index + step) % len(self.items)

    def _go(self, page: str) -> None:
        self.page = page
        self.index = 0

    def activate(self, display: DisplayOptions) -> str:
        """Run the selected row. Returns the part the caller has to do."""
        item = self.items[self.index]
        if item.action == "close":
            self.visible = False
        elif item.action.startswith("page:"):
            self._go(item.action.split(":", 1)[1])
        elif item.action == "cycle":
            self._cycle(display, item.field, 1)
        else:
            return item.action
        return ""

    def handle(
        self,
        event: pygame.event.Event,
        display: DisplayOptions,
        active: DisplayOptions,
    ) -> str:
        """Consume one event. Returns "quit", "apply", or nothing."""
        action = ""

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.page == "main":
                    self.visible = False
                else:
                    self._go("main")
            elif event.key in (pygame.K_UP, pygame.K_w, pygame.K_KP8):
                self._move(-1)
            elif event.key in (pygame.K_DOWN, pygame.K_s, pygame.K_KP2):
                self._move(1)
            elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                item = self.items[self.index]
                if item.action == "cycle":
                    self._cycle(
                        display, item.field, 1 if event.key == pygame.K_RIGHT else -1
                    )
            elif event.key in (
                pygame.K_RETURN,
                pygame.K_KP_ENTER,
                pygame.K_SPACE,
            ):
                action = self.activate(display)

        elif event.type == pygame.MOUSEMOTION:
            for index in range(len(self.items)):
                if self._row_rect(index).collidepoint(event.pos):
                    self.index = index
                    break

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index in range(len(self.items)):
                if self._row_rect(index).collidepoint(event.pos):
                    self.index = index
                    action = self.activate(display)
                    break

        elif event.type == pygame.MOUSEWHEEL:
            item = self.items[self.index]
            if item.action == "cycle":
                self._cycle(display, item.field, 1 if event.y > 0 else -1)
            else:
                self._move(-1 if event.y > 0 else 1)

        self.refresh(display, active)
        return action


def print_demo_page(
    renderer: ParchmentInkRenderer,
    sprite_page: int = 0,
    include_feature: bool = True,
    changed: pygame.Rect | None = None,
) -> None:
    """Print the demo artwork on to a bare renderer's page.

    The renderer starts blank -- what is on the page is not its business -- so
    anything that wants the ledger, the sprite strip and the featured creature
    asks for them here. That includes the workbench itself, by way of
    `DemoPage`, and any test that needs something on the page to look at.
    """
    renderer.set_page_art(
        make_demo_ink(
            renderer.size,
            load_sprite_sheet(),
            sprite_page,
            scale=renderer.canvas.scale,
            include_feature=include_feature,
        ),
        changed,
    )


class DemoPage:
    """The workbench's page: the demo artwork, and the panels sitting over it.

    Composed around a `ParchmentInkRenderer` rather than derived from one. The
    renderer knows how to light a page and nothing about what is printed on it;
    this knows what the demo prints -- the ledger, the sprite strip, the
    featured creature -- and the chrome used to inspect it. Keeping the two
    apart is what lets the effect be dropped into the game without dragging a
    slider panel along with it.
    """

    def __init__(
        self,
        display: DisplayOptions,
        settings: InkSettings,
        seed: int = 7,
        headless: bool = False,
    ) -> None:
        # `headless` goes straight through to the renderer, so the whole
        # workbench -- panels and all -- can be exercised with no window.
        self.renderer = ParchmentInkRenderer(
            display.size, seed, display.supersample, headless
        )
        self.sprites = load_sprite_sheet()
        self.sprite_page = 0
        # The featured creature is printed on the page until the pen is asked
        # to draw it, at which point it has to come off first.
        self.show_feature = True

        context, size = self.renderer.context, self.renderer.size
        self.help = HelpOverlay(context, size)
        self.sliders = SliderPanel(context, size, settings)
        self.menu = MenuPanel(context, size, replace(display))

        self.reprint()

    # --- what is printed on the page ------------------------------------

    def reprint(self, changed: pygame.Rect | None = None) -> None:
        """Redraw the demo artwork underneath whatever has been drawn on it."""
        print_demo_page(
            self.renderer, self.sprite_page, self.show_feature, changed
        )

    def change_sprite_page(self, delta: int) -> None:
        if self.sprites is None:
            return
        pages = max(1, self.sprites.count // self.sprites.columns)
        self.sprite_page = (self.sprite_page + delta) % pages
        self.reprint()

    def show_creature(self, visible: bool) -> None:
        """Put the featured creature into the printed artwork, or take it out.

        It has to come out before the pen can draw it, and go back in when the
        pen has finished, so that the drawing settles into the page as printed
        artwork and survives an undo like the rest of it.
        """
        if self.show_feature == visible:
            return
        self.show_feature = visible

        # Nothing outside the creature's own corner differs between the two.
        scale = self.canvas.scale
        placed = feature_rect(self.renderer.size, scale)
        self.reprint(
            None
            if placed is None
            else pygame.Rect(
                placed.left // scale,
                placed.top // scale,
                -(-placed.width // scale),
                -(-placed.height // scale),
            )
        )

    def draw_feature(self, speed: float) -> SpriteReveal | None:
        """Set the pen to draw the featured creature where it is printed."""
        scale = self.canvas.scale
        placed = feature_rect(self.renderer.size, scale)
        if placed is None:
            return None
        return sprite_reveal(
            FEATURE_IMAGE_PATH,
            placed,
            scale,
            scale * design_unit(self.renderer.size),
            speed,
        )

    # --- the seam the loop talks to -------------------------------------

    @property
    def canvas(self) -> InkCanvas:
        return self.renderer.canvas

    def draw(self, light_position, settings: InkSettings, view: View) -> None:
        """The page, then the chrome, in the order they have to appear."""
        self.renderer.render(light_position, settings, view)
        if self.sliders.visible:
            self.sliders.draw()
        if self.help.visible:
            self.help.draw()
        # The menu is modal, so it goes over everything else.
        if self.menu.visible:
            self.menu.draw()

    def release(self) -> None:
        self.help.release()
        self.sliders.release()
        self.menu.release()
        self.renderer.release()


def rebuild_page(
    old: DemoPage,
    display: DisplayOptions,
    settings: InkSettings,
    view: View,
) -> tuple[DemoPage, View]:
    """Reopen the window for new display options, carrying the page across.

    Every option changes how much texture is allocated, so there is nothing to
    do but build everything again. What survives is what the user made: the
    strokes, which are kept as points and can simply be laid down onto the new
    mask, along with the view, the material settings and which panels were up.

    `display.size` is updated to the size the window manager actually granted,
    so the menu reports where you ended up rather than where you aimed.
    """
    strokes = old.canvas.strokes
    sprite_page = old.sprite_page
    show_feature = old.show_feature
    seed = old.renderer.seed
    panels = (old.help.visible, old.sliders.visible, old.menu.visible, old.menu.page)
    old.release()

    display.size = open_window(display.size, display.fullscreen)
    fresh = DemoPage(display, settings, seed)

    # Set both before reprinting, so the artwork is only drawn once however
    # many of them changed.
    fresh.show_feature = show_feature
    fresh.sprite_page = sprite_page
    fresh.reprint()
    if strokes:
        fresh.canvas.strokes = strokes
        fresh.canvas.recomposite()
        fresh.renderer.upload_ink()

    fresh.help.visible, fresh.sliders.visible = panels[0], panels[1]
    fresh.menu.visible, fresh.menu.page = panels[2], panels[3]
    fresh.sliders.refresh(settings)
    fresh.menu.refresh(display, display)

    fresh_view = View(display.size)
    fresh_view.zoom = view.zoom
    fresh_view.center_x, fresh_view.center_y = view.center_x, view.center_y
    return fresh, fresh_view


def update_caption(settings: InkSettings, brush_radius: int, view: View) -> None:
    modes = ("final", "normals", "roughness", "ink mask", "wetness")
    pygame.display.set_caption(
        "ModernGL Parchment + Fresh Ink  |  "
        f"wetness {settings.wetness:.2f} "
        f"({'drying' if settings.drying else 'held'})  "
        f"light height {settings.light_height:.2f}  "
        f"brush {brush_radius}px  "
        f"zoom {view.zoom * 100:.0f}%  "
        f"view {modes[settings.debug_mode]}  |  press ? for controls"
    )


def main(allow_restart: bool = False) -> None:
    pygame.init()
    pygame.font.init()

    display = DisplayOptions()
    # Built before the page, which hands it to the slider panel.
    settings = InkSettings()
    try:
        display.size = open_window(display.size, display.fullscreen)
        if allow_restart:
            # Before anything is built, since this may not come back.
            restart_on_the_graphics_card(
                moderngl.create_context(require=330).info["GL_RENDERER"]
            )
        page = DemoPage(display, settings)
    except Exception as exc:
        pygame.quit()
        raise SystemExit(
            "Could not create the OpenGL 3.3 / ModernGL renderer. "
            "Update the graphics driver or relax the requested context version.\n"
            f"Original error: {exc}"
        ) from exc

    # Worth one line: which of these it is decides whether the demo runs at
    # sixty frames a second or at fifteen, and nothing else on screen says so.
    print(f"Rendering with {page.renderer.context.info['GL_RENDERER']}", file=sys.stderr)

    # What the renderer standing in front of us was actually built with, so the
    # menu can tell a pending change from an applied one.
    active = replace(display)

    clock = pygame.time.Clock()
    view = View(display.size)
    # How wide the pen the user draws with is: an input setting, not a property
    # of the ink, so it lives here rather than in the material.
    brush_radius = 5
    update_caption(settings, brush_radius, view)

    light_position = (page.renderer.size[0] // 2, page.renderer.size[1] // 2)
    drawing = False
    panning = False
    elapsed = 0.0
    shown_wetness = settings.wetness
    player: SpriteReveal | None = None

    def clear_for_drawing() -> None:
        """Take the ink off the page and dry it, ready for the pen to start.

        The drying matters as much as the clearing: a drawing begun a moment
        after the last one would otherwise start on a soaking sheet, and the
        nib's own wetness -- the whole point of drawing the figure rather than
        fading it in -- would be lost in it.
        """
        page.canvas.clear_strokes()
        page.canvas.dry_now()
        page.renderer.upload_ink()
        page.renderer.upload_wetness()
        settings.wetness = PAGE_WET_FLOOR
        settings.drying = True

    running = True
    # The picture is redrawn when something asks for it; these say whether
    # anything has. `idle_frames` forces one through now and then regardless,
    # so a repaint the window manager wanted and did not tell us about cannot
    # leave the page stale indefinitely.
    needs_frame = True
    idle_frames = 0
    while running:
        caption_changed = False
        events = pygame.event.get()
        # Generous on purpose: it is far cheaper to draw a frame that turned
        # out to be identical than to work out which events change the picture
        # and be wrong about one of them.
        handled_event = bool(events)
        for event in events:
            if event.type == pygame.QUIT:
                running = False
                continue

            # The menu is modal: while it is up nothing else sees an event, so
            # arrowing through it cannot also nudge the lamp or the page.
            if page.menu.visible:
                action = page.menu.handle(event, display, active)
                if action == "quit":
                    running = False
                elif action == "apply" and display != active:
                    wanted = display.size
                    page, view = rebuild_page(page, display, settings, view)
                    active = replace(display)
                    if display.size != wanted:
                        page.menu.note = (
                            f"window manager kept {display.text('size')}"
                        )
                    page.menu.refresh(display, active)
                    light_position = (page.renderer.size[0] // 2, page.renderer.size[1] // 2)
                    caption_changed = True
                    # The rebuild took about a second; do not hand that to the
                    # dry-down as if it were one very long frame.
                    elapsed = 0.0
                    clock.tick()
                continue

            # The slider panel gets first refusal on the mouse, so dragging a
            # control neither draws ink nor drags the lamp along with it.
            if page.sliders.handle(event, settings):
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                panning = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                panning = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                # Taking the pen back interrupts the replay rather than fighting
                # it for the page.
                player = None
                drawing = True
                dirty = page.canvas.begin_stroke(
                    view.screen_to_canvas(event.pos),
                    brush_radius,
                    erase=event.button == 3,
                )
                page.renderer.upload_stroke(dirty)
            elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 3):
                drawing = False
                page.canvas.end_stroke()
            elif event.type == pygame.MOUSEMOTION:
                if panning:
                    view.pan_by(event.rel)
                elif drawing:
                    page.renderer.upload_stroke(
                        page.canvas.extend_stroke(view.screen_to_canvas(event.pos))
                    )
                else:
                    # The light stays put while drawing so a fresh stroke can be
                    # judged under steady lighting instead of a moving highlight.
                    light_position = event.pos
            elif event.type == pygame.MOUSEWHEEL:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    brush_radius = max(1, min(48, brush_radius + event.y))
                else:
                    view.zoom_by(event.y, pygame.mouse.get_pos())
                caption_changed = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Escape no longer quits on the spot; it asks.
                    page.menu.open()
                    page.menu.refresh(display, active)
                    drawing = panning = False
                    if player is not None:
                        page.canvas.end_stroke()
                        player = None
                elif getattr(event, "unicode", "") == "?" or event.key in (
                    pygame.K_QUESTION,
                    pygame.K_SLASH,
                    pygame.K_F1,
                ):
                    # The key that produces `?` moves around between layouts, so
                    # trust the character the event carries where there is one.
                    page.help.toggle()
                elif event.key == pygame.K_TAB:
                    page.sliders.toggle()
                    page.sliders.refresh(settings)
                elif event.key == pygame.K_r:
                    page.renderer.regenerate_paper()
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA):
                    # Taking manual control pauses the dry-down, otherwise the
                    # decay would pull the value straight back down again.
                    settings.drying = False
                    settings.wetness = max(0.0, settings.wetness - 0.06)
                    caption_changed = True
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD):
                    settings.drying = False
                    settings.wetness = min(1.0, settings.wetness + 0.06)
                    caption_changed = True
                elif event.key == pygame.K_w:
                    settings.rewet()
                    caption_changed = True
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    settings.light_height = max(0.06, settings.light_height - 0.025)
                    caption_changed = True
                elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    settings.light_height = min(0.70, settings.light_height + 0.025)
                    caption_changed = True
                elif event.key in (pygame.K_1, pygame.K_KP1):
                    settings.debug_mode = 0
                    caption_changed = True
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    settings.debug_mode = 1
                    caption_changed = True
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    settings.debug_mode = 2
                    caption_changed = True
                elif event.key in (pygame.K_4, pygame.K_KP4):
                    settings.debug_mode = 3
                    caption_changed = True
                elif event.key in (pygame.K_5, pygame.K_KP5):
                    settings.debug_mode = 4
                    caption_changed = True
                elif event.key == pygame.K_LEFT:
                    page.change_sprite_page(-1)
                elif event.key == pygame.K_RIGHT:
                    page.change_sprite_page(1)
                elif event.key == pygame.K_u or (
                    event.key == pygame.K_z and event.mod & pygame.KMOD_CTRL
                ):
                    if page.canvas.undo():
                        page.renderer.upload_ink()
                elif event.key == pygame.K_c:
                    player = None
                    cleared = page.canvas.clear_strokes()
                    # Putting the creature back reprints the base art, which
                    # uploads the mask itself; only a bare clear has to.
                    if page.show_feature:
                        if cleared:
                            page.renderer.upload_ink()
                    else:
                        page.show_creature(True)
                elif event.key == pygame.K_t:
                    # Take the printed creature off the page and let the pen
                    # put it back, stroke by stroke.
                    drawing_feature = page.draw_feature(settings.pen_speed)
                    if drawing_feature is not None:
                        page.show_creature(False)
                        clear_for_drawing()
                        player = drawing_feature
                elif event.key in (pygame.K_0, pygame.K_KP0, pygame.K_HOME):
                    view.reset()
                    caption_changed = True

        # The menu pauses the page, so ink is not quietly drying out behind it.
        if page.menu.visible:
            elapsed = 0.0

        # The pen moves before the page dries, so the ink it laid down this
        # frame is at its freshest when the light hits it.
        if player is not None and not page.menu.visible:
            player.speed = settings.pen_speed
            page.renderer.upload_stroke(player.update(elapsed, page.canvas))
            if player.finished:
                # A finished drawing is simply the printed artwork again, so
                # hand it back to the base and it survives an undo like
                # everything else on the page.
                page.show_creature(True)
                player = None
                needs_frame = True

        # The page-wide ink and each individual stroke dry on the same clock.
        if settings.advance(elapsed) and abs(settings.wetness - shown_wetness) >= 0.01:
            caption_changed = True

        drying_region = page.canvas.dry(elapsed, settings.dry_rate)
        if drying_region is not None:
            page.renderer.upload_wetness(drying_region)

        if caption_changed:
            shown_wetness = settings.wetness
            update_caption(settings, brush_radius, view)

        # A page of dry ink under a lamp that is not moving is the same picture
        # frame after frame, and drawing it again costs the whole shader --
        # which is the whole frame, this being one full-screen quad. So the
        # frame is drawn when something asked for it, and while anything is
        # still moving of its own accord: the pen, the dry-down, or the
        # iridescence swirling in ink that has not set yet.
        moving = (
            player is not None
            or page.canvas.wet_bounds is not None
            or (settings.drying and settings.wetness > PAGE_WET_FLOOR + 1e-4)
        )
        if handled_event or moving or needs_frame or idle_frames >= IDLE_REDRAW_FRAMES:
            page.draw(light_position, settings, view)
            pygame.display.flip()
            needs_frame = False
            idle_frames = 0
        else:
            idle_frames += 1

        elapsed = clock.tick(60) / 1000.0

    page.release()
    pygame.quit()


if __name__ == "__main__":
    # Only when run as a program: importing this module must never relaunch
    # anything, or a test harness that imports it would restart itself.
    main(allow_restart=True)