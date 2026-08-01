"""A roguelike drawn in wet ink on parchment: the second `inkfx` workbench.

`inkbench.py` looks at the effect by putting a ledger page under it. This one
looks at it by putting a game under it, which asks harder questions: the page
now changes several times a second, in scattered handfuls of cells rather than
all at once, and every change has to reach the graphics card without redrawing
the sheet.

What that buys is the thing worth seeing. The dungeon is not drawn and then
lit; it is *written*. A cell you have never been to is bare parchment. Step
into sight of it and its glyph goes on as fresh ink -- glossy, iridescent,
catching the lamp -- and dries into the page over the next few seconds. Walk
away and it stays behind as a faint, dry memory of itself, because that is
what a map you drew as you went would look like.

A lamp hangs over the player and the walls stop it. That falls out of the game
rather than being a second system: the sim already shadowcasts to decide what
the player can see, and since the light is where the player is, what it can
see and what the lamp reaches are the same set of cells. It is handed to the
renderer as an occlusion map and costs one texture lookup.

The player is not the only thing carrying a light: a kobold has a lantern and
an ogre a brand. Every light in the place burns the same colour, and they
differ in how far they throw. Each is a lamp of its own with its own reach
and -- because the light map keeps one channel per lamp -- its own shadows, so
a wall between you and the ogre stops your light without touching his.

You do not have to see what is holding a light to see the light. The test is
whether any of what it throws lands where you can see, so a torch around a
corner lights the far wall of your passage well before its owner appears.

None of it is on or off. The map is drawn at three texels to the tile and
filtered on the way to the shader, and each lamp eases out towards the edge of
its reach, so a tile can be half lit -- and past the last of the light the
sheet is left to the ambient term, which is as dark as the page gets.

The pieces:

    roguesim.py  the dungeon: entities, rooms, sight, hunting. No pixels and
                 no pygame; it can be played to the end with no display at all.
    inkfx.py     the parchment and the ink.
    this file    the tileset, the layout, and the loop tying them together.

Run it with `python3 rogue.py`. `python3 rogue.py --headless out.png` plays a
few turns and saves the page with no window involved anywhere.

The map is ink and belongs to the sheet, so it pans and zooms with it. The
reading down the right is not: it is composited over the finished frame in
window pixels and stays put whatever the camera does. The lamp splits the
difference -- it is placed in window pixels, because that is where the shader
hangs it, but it is aimed at wherever on the page the player is standing.

Controls are on the `?` card; the short version is that the arrows, the numeric
keypad or the vi keys move, `.` waits, the middle mouse button drags the page
about, `Shift+N` starts a new dungeon and `R` a new sheet of parchment.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy
import pygame

import roguesim
from inkfx import (
    INK_SUPERSAMPLE,
    LAMP_COLOUR,
    MAX_LIGHTS,
    PAGE_WET_FLOOR,
    PANEL_EDGE,
    PANEL_KEY,
    PANEL_TEXT,
    PANEL_TITLE,
    InkCanvas,
    InkSettings,
    Light,
    Panel,
    ParchmentInkRenderer,
    View,
    make_panel_surface,
    open_window,
    restart_on_the_graphics_card,
)


WINDOW_SIZE = (1100, 760)
IDLE_REDRAW_FRAMES = 60

REPO_ROOT = Path(__file__).resolve().parents[2]
TILESET_DIR = REPO_ROOT / "gfx" / "tilesets" / "dwarf_fortress"
# The depixelised sheet: every glyph traced off the 64px original and rebuilt
# at eight times the size, so an edge that used to be a staircase is a curve.
# That is what makes zooming into the page worth doing.
TILESET_PATH = TILESET_DIR / "hack_square_64x64_x8_hard1.png"
# The pixel-art original, if the big one has not been built yet.
TILESET_FALLBACK = TILESET_DIR / "hack_square_64x64.png"

# Roughly how many dungeon cells fit across the board, whatever size the page
# is. The cell size follows from this, so the game looks the same at 800x560
# as at 4K rather than turning into a differently sized dungeon.
TARGET_COLUMNS = 26

# How dark the ink is. Cells in sight are drawn at full strength; cells you
# have been to but cannot currently see are the same glyph, faint -- a map
# drawn from memory rather than one you are looking at.
INK_SEEN = 255
INK_REMEMBERED = 84

# How much of the player's own lamp still finds a mapped cell they cannot
# currently see. Small on purpose: it is there to keep the remembered map
# faintly legible on unlit parchment and nothing more. Everything the player
# has never been to gets none of it at all, which is what keeps the sheet
# around the map properly dark.
REMEMBERED_GLOW = 16

# How finely the light map is drawn, in texels per dungeon cell. The map is
# filtered linearly on the way to the shader, so this is not the resolution
# the light is seen at -- it is how often the falloff is resampled, and three
# is enough for a gradient with no steps in it. It also sets how sharp a wall
# edge can be, since occlusion is per cell: a shadow boundary is soft over
# one texel, which at three per cell is a third of a tile of penumbra.
LIGHT_TEXELS = 3

# Where a lamp starts to fade, as a fraction of its radius. Kept low, so most
# of a lamp's reach is spent fading: the shader's own falloff is gentle across
# a span this small, and without a long ramp here a pool of light reads as a
# flat disc with an edge on it rather than as light.
LIGHT_FADE_FROM = 0.20

# How much of the window the reading down the right takes, and the range it is
# allowed to take it in. A share alone would give an unreadable sliver on a
# small window and a column of enormous type on a large one.
PANEL_SHARE = 0.26
PANEL_MIN_WIDTH = 230
PANEL_MAX_WIDTH = 420


# --- the tileset ----------------------------------------------------------


class CodePage437:
    """A 16x16 sheet of square glyphs, cut up and handed out as ink.

    Every glyph the game asks for arrives as a surface whose alpha is its
    coverage and whose colour is white, which is exactly what `InkCanvas`
    wants: the ink mask is an alpha channel and nothing else. In a sheet laid
    out this way a glyph's index is its code point, so `ord("@")` is the
    player's tile with no table in between.

    Glyphs are scaled on demand and kept. A dungeon uses about a dozen
    distinct characters, so scaling lazily costs a dozen resizes rather than
    the two hundred and fifty six that resizing the whole sheet would do --
    and the sheet at hand is 8192 square, a quarter of a gigabyte, so that
    difference is worth having.
    """

    GRID = 16

    def __init__(self, path: Path) -> None:
        self.path = path
        sheet = pygame.image.load(str(path))
        if not sheet.get_flags() & pygame.SRCALPHA:
            # An indexed or colour-keyed sheet has no real alpha until
            # something resolves it. Blitting onto SRCALPHA does that, and
            # unlike `convert_alpha` it needs no display surface.
            resolved = pygame.Surface(sheet.get_size(), pygame.SRCALPHA)
            resolved.fill((0, 0, 0, 0))
            resolved.blit(sheet, (0, 0))
            sheet = resolved

        self.sheet = sheet
        width, height = sheet.get_size()
        self.tile_width = width // self.GRID
        self.tile_height = height // self.GRID
        self._cache: dict[tuple[int, int, int], pygame.Surface] = {}

    def glyph(self, code: int, size: int, opacity: int = 255) -> pygame.Surface:
        """One character at `size` pixels square, as ink of a given density."""
        key = (code % 256, size, opacity)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        index = code % 256
        source = pygame.Rect(
            (index % self.GRID) * self.tile_width,
            (index // self.GRID) * self.tile_height,
            self.tile_width,
            self.tile_height,
        )
        tile = pygame.Surface(source.size, pygame.SRCALPHA)
        tile.fill((0, 0, 0, 0))
        tile.blit(self.sheet, (0, 0), source, special_flags=pygame.BLEND_RGBA_MAX)

        if source.width != size or source.height != size:
            # Smooth, not point sampled: these glyphs were rebuilt as curves
            # precisely so that resizing them would not give a staircase back.
            tile = pygame.transform.smoothscale(tile, (size, size))

        # Colour never reaches the shader, but a glyph that is white
        # everywhere cannot fringe grey when it is scaled, and it keeps the
        # mask readable when it is looked at directly in the ink debug view.
        tile.fill((255, 255, 255, 0), special_flags=pygame.BLEND_RGBA_MAX)
        if opacity != 255:
            tile.fill(
                (255, 255, 255, opacity), special_flags=pygame.BLEND_RGBA_MULT
            )

        self._cache[key] = tile
        return tile


def load_tileset() -> CodePage437 | None:
    """The upscaled sheet, or the pixel-art original, or nothing at all."""
    for path in (TILESET_PATH, TILESET_FALLBACK):
        if not path.exists():
            continue
        try:
            return CodePage437(path)
        except pygame.error as exc:
            print(f"Could not load tileset {path}: {exc}", file=sys.stderr)
    print(f"No tileset found in {TILESET_DIR}", file=sys.stderr)
    return None


# --- where things go on the page ------------------------------------------


@dataclass(frozen=True)
class Layout:
    """The window divided into a board on the left and a reading on the right.

    Two different kinds of coordinate meet here, so it is worth being exact
    about which is which. `origin`, `cell` and everything `cell_rect` returns
    are *page* pixels: they are drawn onto the sheet, and they pan and zoom
    with it. `panel` is *window* pixels: it is composited over the finished
    frame and the camera never touches it.

    They happen to coincide at rest, because the page is exactly the size of
    the window. They stop coinciding the moment anything is panned, which is
    the entire reason the reading is a panel and not ink.
    """

    cell: int
    columns: int
    rows: int
    origin: tuple[int, int]
    panel: pygame.Rect

    @property
    def grid_size(self) -> tuple[int, int]:
        return (self.columns, self.rows)

    def cell_rect(self, x: int, y: int) -> pygame.Rect:
        """The page pixels one dungeon cell occupies."""
        return pygame.Rect(
            self.origin[0] + x * self.cell,
            self.origin[1] + y * self.cell,
            self.cell,
            self.cell,
        )


def layout_for(size: tuple[int, int]) -> Layout:
    """Fit a dungeon and a reading onto a window of this size."""
    width, height = size
    margin = max(6, width // 70)

    panel_width = min(
        max(round(width * PANEL_SHARE), PANEL_MIN_WIDTH), PANEL_MAX_WIDTH
    )
    # However the shares work out, the board keeps the larger half.
    panel_width = min(panel_width, width // 2)
    panel = pygame.Rect(
        width - panel_width - margin, margin, panel_width, height - margin * 2
    )

    # The board is what is left, and the cell size comes off that rather than
    # off the window, so widening the reading does not silently make the
    # dungeon a different shape.
    board_width = max(8, panel.left - margin * 2)
    cell = max(8, board_width // TARGET_COLUMNS)

    columns = max(4, board_width // cell)
    rows = max(4, (height - margin * 2) // cell)

    # Centre the grid in whatever is left over, so a board whose width does not
    # divide evenly does not put all its slack down one side.
    left = margin + (board_width - columns * cell) // 2
    top = margin + ((height - margin * 2) - rows * cell) // 2
    return Layout(
        cell=cell,
        columns=columns,
        rows=rows,
        origin=(left, top),
        panel=panel,
    )


# --- the page -------------------------------------------------------------


class RoguePage:
    """The dungeon as artwork on an ink page, kept in step with the game.

    Composed around a `ParchmentInkRenderer` and a `roguesim.Game`, derived
    from neither. It owns one idea: which cells are currently printed on the
    sheet, and what to do when that stops being true.

    Nothing here draws hand strokes -- the mouse moves the lamp -- so the
    canvas's base artwork and the composite that gets uploaded are the same
    picture at all times. That is what makes a repaint cheap; see `_publish`.
    """

    def __init__(
        self,
        size: tuple[int, int],
        settings: InkSettings,
        seed: int = 1,
        supersample: int = INK_SUPERSAMPLE,
        headless: bool = False,
    ) -> None:
        # `headless` goes straight through to the renderer, so the whole game
        # can be played and photographed with no window.
        self.renderer = ParchmentInkRenderer(size, seed, supersample, headless)
        self.settings = settings
        self.layout = layout_for(size)
        self.tileset = load_tileset()

        context = self.renderer.context
        self.status = StatusPanel(context, size, self.layout.panel)
        self.help = _help_card(context, size)
        self.new_dungeon(seed)

    # --- the seam the loop talks to --------------------------------------

    @property
    def canvas(self) -> InkCanvas:
        return self.renderer.canvas

    @property
    def size(self) -> tuple[int, int]:
        return self.renderer.size

    def lamps(self, view: View) -> list[Light]:
        """Every light burning on the page, in the slots the mask expects.

        The shader hangs lamps in front of the *screen*, not on the sheet, so
        following something around the page means asking the view where it has
        ended up. Panning slides every pool of light along with the map instead
        of leaving them behind.

        The order matters and is the game's: slot 0 is the player, and each
        one after is a light-carrier in sight. It has to match the order the
        channels were written in `catch_up`, which is why both read the same
        `game.lights` list rather than each working it out.
        """
        return [
            Light(
                view.canvas_to_screen(self.layout.cell_rect(*source.cell).center),
                LAMP_COLOUR,
                # No ring. The workbench draws one to show where a lamp being
                # dragged about has got to; here the lamps stand for things
                # already on the page, and a lamp whose owner is round a
                # corner would otherwise leave a disc hanging in the dark.
                marker=0.0,
            )
            for source in self.game.lights[:MAX_LIGHTS]
        ]

    def draw(self, lights: Sequence[Light], view: View) -> None:
        """The page, then the chrome, in the order they have to appear."""
        self.renderer.render(lights, self.settings, view)
        if self.status.visible:
            self.status.draw()
        # The help card is modal-ish, so it goes over the reading as well.
        if self.help.visible:
            self.help.draw()

    def release(self) -> None:
        self.status.release()
        self.help.release()
        self.renderer.release()

    # --- keeping the page and the game in step ---------------------------

    def new_dungeon(self, seed: int) -> None:
        """Start a fresh game, and print its opening view on a blank sheet."""
        self.seed = seed
        # The renderer's occlusion map has one channel per lamp, so that is
        # how many the dungeon is allowed to light at once.
        self.game = roguesim.Game(
            self.layout.grid_size, seed=seed, light_slots=MAX_LIGHTS
        )

        art = pygame.Surface(self.canvas.mask_size, pygame.SRCALPHA)
        art.fill((0, 0, 0, 0))
        self.renderer.set_page_art(art)

        # Nothing carries over from the last dungeon, so the wetness should
        # not either: the first room has to appear on a dry sheet, or the
        # fresh ink has nothing to stand out against.
        self.canvas.dry_now()
        self.renderer.upload_wetness()
        self.settings.wetness = PAGE_WET_FLOOR
        self.settings.drying = True

        self.catch_up()
        self.status.refresh(self.game)
        self.game.messages_changed = False

    def catch_up(self) -> None:
        """Repaint whatever the game says has changed, and wet what is new.

        Called after every turn. The work is proportional to what moved -- a
        step lights perhaps thirty cells -- rather than to the size of the
        page, which at 2x supersampling is eight megapixels of surface and the
        best part of a tenth of a second to send to the card.
        """
        dirty, fresh = self.game.take_dirty()
        base = self.canvas.base
        for x, y in dirty:
            region = self.layout.cell_rect(x, y)
            mask_rect = self.canvas.mask_rect(region)
            base.fill((0, 0, 0, 0), mask_rect)

            lit = bool(self.game.visible[x, y])
            code = self._glyph_at(x, y)
            if code is not None and self.tileset is not None:
                base.blit(
                    self.tileset.glyph(
                        code,
                        self.layout.cell * self.canvas.scale,
                        INK_SEEN if lit else INK_REMEMBERED,
                    ),
                    mask_rect,
                    special_flags=pygame.BLEND_RGBA_MAX,
                )
            self._publish(region)

        # The lamps have all moved, so the light map is redrawn whole rather
        # than followed cell by cell -- see `relight`.
        self.relight()

        if fresh:
            self._wet(fresh)

    def relight(self) -> None:
        """Redraw the light map from every lamp currently burning.

        Rebuilt whole rather than patched cell by cell. It is a few thousand
        texels -- one map for the whole board, three per tile -- so working it
        out with array arithmetic and sending all of it costs less than
        deciding which parts of it moved, and every lamp moves every turn
        anyway.

        Two things combine into each texel. Occlusion comes from the sim, per
        cell, and is what makes a wall a wall. Falloff is worked out here, per
        texel, from the texel's own distance to the lamp -- so a pool of light
        is bright in the middle and fades out at its edge, and a cell can be
        half lit rather than only lit or unlit.
        """
        layout = self.layout
        columns, rows = layout.columns, layout.rows
        step = layout.cell / LIGHT_TEXELS

        # A cell of margin all round. The map clamps to its own edge outside
        # itself, so the border has to be dark or the lighting would smear
        # outwards across the rest of the sheet forever.
        across = (columns + 2) * LIGHT_TEXELS
        down = (rows + 2) * LIGHT_TEXELS
        left = layout.origin[0] - layout.cell
        top = layout.origin[1] - layout.cell

        # The middle of each texel, in page pixels, and the cell it falls in.
        columns_index = numpy.arange(across)
        rows_index = numpy.arange(down)
        page_x = left + (columns_index + 0.5) * step
        page_y = top + (rows_index + 0.5) * step
        cell_x = columns_index // LIGHT_TEXELS
        cell_y = rows_index // LIGHT_TEXELS

        channels = numpy.zeros((across, down, 4), dtype=numpy.uint8)
        padded = numpy.zeros((columns + 2, rows + 2), dtype=bool)

        for slot, source in enumerate(self.game.lights[:MAX_LIGHTS]):
            padded[:] = False
            padded[1:-1, 1:-1] = source.reach
            visible = padded[cell_x][:, cell_y]

            centre = layout.cell_rect(*source.cell).center
            span = max(1.0, source.radius * layout.cell)
            away = numpy.hypot(
                page_x[:, None] - centre[0], page_y[None, :] - centre[1]
            ) / span
            strength = numpy.where(visible, 1.0 - _smoothstep(LIGHT_FADE_FROM, 1.0, away), 0.0)

            if slot == 0:
                # The one place light is added rather than shaped: enough of
                # the reader's own lamp to keep a mapped room legible once
                # they have walked out of it. Nowhere they have never been
                # gets any, which is what keeps the rest of the sheet dark.
                mapped = numpy.zeros_like(padded)
                mapped[1:-1, 1:-1] = self.game.explored
                strength = numpy.maximum(
                    strength,
                    mapped[cell_x][:, cell_y] * (REMEMBERED_GLOW / 255.0),
                )

            channels[..., slot] = numpy.round(strength * 255.0)

        surface = pygame.Surface((across, down), pygame.SRCALPHA)
        pygame.surfarray.pixels3d(surface)[:] = channels[..., :3]
        pygame.surfarray.pixels_alpha(surface)[:] = channels[..., 3]

        # Kept, not just sent: it is what the page is actually lit by, so it
        # is the thing worth asking questions of.
        self.light_map = surface
        self.light_rect = pygame.Rect(
            left, top, (columns + 2) * layout.cell, (rows + 2) * layout.cell
        )
        self.renderer.set_light_mask(surface, covers=self.light_rect)

    def light_at(self, cell: tuple[int, int], slot: int = 0) -> int:
        """How much of one lamp lands in the middle of a cell, 0 to 255.

        Reads the map that was built and sent rather than working the answer
        out again, so a caller asks the same question the shader does.
        """
        x, y = cell
        middle = LIGHT_TEXELS // 2
        return self.light_map.get_at(
            ((x + 1) * LIGHT_TEXELS + middle, (y + 1) * LIGHT_TEXELS + middle)
        )[slot]

    def _glyph_at(self, x: int, y: int) -> int | None:
        """What a cell shows: nothing, the floor plan, or whoever is on it."""
        if not self.game.explored[x, y]:
            return None
        if self.game.visible[x, y]:
            # Only what you can actually see; the map remembers rooms, not
            # where a monster was standing when you last looked in.
            standing = self.game.glyph_at(x, y)
            if standing is not None:
                return standing
        tile = self.game.dungeon.tiles[x, y]
        return (
            roguesim.FLOOR_GLYPH if tile == roguesim.FLOOR else roguesim.WALL_GLYPH
        )

    def _wet(self, cells: set[tuple[int, int]]) -> None:
        """Flood the newly discovered cells with fresh ink.

        Wetness is held at page resolution rather than the mask's, and it is
        only ever seen through the coverage mask, so filling whole cells is
        both correct and cheaper than following the glyph outlines: bare
        parchment inside a wet cell has no ink on it to be wet.
        """
        wetness = self.canvas.wetness
        for x, y in cells:
            region = self.layout.cell_rect(x, y).clip(wetness.get_rect())
            if not region.width or not region.height:
                continue
            wetness.fill((255, 255, 255), region)
            self.renderer.upload_wetness(self.canvas.mark_wet(region))

    def _publish(self, region: pygame.Rect) -> None:
        """Push one repainted rectangle of base artwork through to the card.

        `InkCanvas.recomposite` would rebuild the whole composite to move a
        single glyph, because in general the composite is the base with
        hand-drawn strokes stamped over it. This page draws no strokes, so the
        two pictures are identical and copying across the changed rectangle is
        exact as well as several hundred times cheaper.

        The destination is cleared first so the copy replaces rather than
        blends: a glyph that got fainter would otherwise keep its old weight.
        """
        mask_rect = self.canvas.mask_rect(region)
        composite = self.canvas.surface
        composite.fill((0, 0, 0, 0), mask_rect)
        composite.blit(
            self.canvas.base,
            mask_rect,
            mask_rect,
            special_flags=pygame.BLEND_RGBA_MAX,
        )
        self.renderer.upload_ink(region)

    def refresh_status(self) -> None:
        """Redraw the reading after a turn.

        Unconditional in the numbers -- the turn count moves every time --
        but it is only ever called when a turn actually passed, so a keypress
        that bumped into a wall does not re-upload the card.
        """
        self.game.messages_changed = False
        self.status.refresh(self.game)


def _smoothstep(low: float, high: float, values: numpy.ndarray) -> numpy.ndarray:
    """The usual S-curve between two edges, over a whole array at once.

    Flat at both ends and steepest in the middle, which is what a light wants
    at the edge of its reach: a linear ramp gives away where it was cut off.
    """
    eased = numpy.clip((values - low) / (high - low), 0.0, 1.0)
    return eased * eased * (3.0 - 2.0 * eased)


# --- the reading down the right -------------------------------------------


def wrap(text: str, font: pygame.font.Font, width: int) -> list[str]:
    """Break a line into as many as it takes to fit `width`.

    Greedy and measured against the font itself rather than a character
    count, because the panel is narrow enough that guessing would be wrong
    often and visibly.
    """
    words = text.split()
    if not words:
        return [""]

    lines, line = [], words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        if font.size(candidate)[0] <= width:
            line = candidate
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


class StatusPanel(Panel):
    """The reading of the game, over the page rather than written on it.

    Everything here is in window pixels and stays where it is put, whatever
    the camera does -- which is the point of it being a panel. The map is ink
    and belongs to the sheet; the turn count and the message log are the
    player's instruments and belong to the screen.

    The card is redrawn onto a surface of fixed size and re-uploaded, because
    `Panel` fixes its quad at construction; a reading whose box changed shape
    as the numbers in it got longer would be worse than one that does not.
    """

    def __init__(
        self,
        context: moderngl.Context,
        window_size: tuple[int, int],
        rect: pygame.Rect,
    ) -> None:
        super().__init__(
            context, window_size, make_panel_surface(rect.size), rect.topleft
        )
        body = max(12, rect.width // 19)
        self.title_font = pygame.font.SysFont("serif", round(body * 1.5), bold=True)
        self.label_font = pygame.font.SysFont("sans", body, bold=True)
        self.body_font = pygame.font.SysFont("sans", body)
        self.padding = max(10, rect.width // 18)
        self.visible = True

    def refresh(self, game: roguesim.Game) -> None:
        """Redraw the card from the game and send it to the card's texture."""
        surface = make_panel_surface(self.rect.size)
        pad = self.padding
        inner = self.rect.width - pad * 2
        y = pad

        surface.blit(self.title_font.render("Ink & Dungeon", True, PANEL_TITLE), (pad, y))
        y += self.title_font.get_linesize() + pad // 2

        health = game.health
        y = self._reading(surface, "turn", str(game.turn), pad, y)
        y = self._reading(
            surface,
            "health",
            f"{max(0, health.current)} / {health.maximum}",
            pad,
            y,
        )
        y = self._bar(surface, max(0, health.current) / health.maximum, pad, y, inner)
        y = self._reading(
            surface,
            "still moving",
            str(game.monsters_left) if game.alive else "-",
            pad,
            y,
        )

        y += pad // 2
        pygame.draw.line(
            surface, PANEL_EDGE, (pad, y), (self.rect.width - pad, y), 1
        )
        y += pad

        # Newest last, the way a log reads, and clipped rather than scrolled:
        # anything that will not fit is older than the player still cares about.
        for message in game.messages:
            for line in wrap(message, self.body_font, inner):
                if y + self.body_font.get_linesize() > self.rect.height - pad:
                    break
                surface.blit(
                    self.body_font.render(line, True, PANEL_TEXT), (pad, y)
                )
                y += self.body_font.get_linesize()
            y += 3

        hint = self.body_font.render("? for controls", True, PANEL_EDGE)
        surface.blit(hint, (pad, self.rect.height - pad - hint.get_height()))

        self.update(surface)

    def _reading(
        self, surface: pygame.Surface, label: str, value: str, pad: int, y: int
    ) -> int:
        """One label on the left, its value against the right edge."""
        surface.blit(self.body_font.render(label, True, PANEL_TEXT), (pad, y))
        rendered = self.label_font.render(value, True, PANEL_KEY)
        surface.blit(rendered, (self.rect.width - pad - rendered.get_width(), y))
        return y + self.label_font.get_linesize()

    def _bar(
        self, surface: pygame.Surface, fraction: float, pad: int, y: int, width: int
    ) -> int:
        """The health, again, as something readable without arithmetic."""
        gap = max(3, pad // 3)
        height = max(6, self.body_font.get_linesize() // 2)
        track = pygame.Rect(pad, y + gap, width, height)
        pygame.draw.rect(surface, PANEL_EDGE, track, 1)
        filled = round((width - 2) * max(0.0, min(1.0, fraction)))
        if filled:
            pygame.draw.rect(
                surface,
                PANEL_KEY,
                pygame.Rect(track.left + 1, track.top + 1, filled, height - 2),
            )
        return track.bottom + pad // 2


# --- the help card --------------------------------------------------------

HELP_LINES = (
    ("", "Ink & Dungeon"),
    ("arrows / numpad / hjkl yubn", "move, or attack what is there"),
    (". or 5 or space", "wait a turn"),
    ("Shift+N", "a new dungeon on a clean sheet"),
    ("R", "a fresh sheet of parchment"),
    ("W", "wet the whole page again"),
    ("middle drag", "pan the page"),
    ("wheel", "zoom into the ink; 0 resets"),
    ("1 2 3 4 5", "final / normals / roughness / ink / wetness"),
    ("?", "this card"),
    ("Esc", "quit"),
)


def _help_card(context: moderngl.Context, window_size: tuple[int, int]) -> Panel:
    """Lay the controls out on a card and hand it to a `Panel` to composite."""
    key_font = pygame.font.SysFont("monospace", 15)
    text_font = pygame.font.SysFont("sans", 15)
    title_font = pygame.font.SysFont("sans", 18, bold=True)

    padding, gap, line = 18, 22, 22
    key_width = max(key_font.size(key)[0] for key, _ in HELP_LINES)
    text_width = max(
        (text_font if key else title_font).size(text)[0]
        for key, text in HELP_LINES
    )
    surface = make_panel_surface(
        (
            padding * 2 + key_width + gap + text_width,
            padding * 2 + line * len(HELP_LINES),
        )
    )

    for index, (key, text) in enumerate(HELP_LINES):
        y = padding + index * line
        if key:
            surface.blit(key_font.render(key, True, PANEL_KEY), (padding, y))
            surface.blit(
                text_font.render(text, True, PANEL_TEXT),
                (padding + key_width + gap, y),
            )
        else:
            surface.blit(title_font.render(text, True, PANEL_TITLE), (padding, y))

    return Panel(context, window_size, surface)


# --- input ----------------------------------------------------------------

# Every key that means a direction, waiting on the spot included. Three
# layouts at once, because a roguelike that only accepts one is irritating and
# because they cost a dictionary entry each.
MOVES: dict[int, tuple[int, int]] = {
    pygame.K_LEFT: (-1, 0), pygame.K_RIGHT: (1, 0),
    pygame.K_UP: (0, -1), pygame.K_DOWN: (0, 1),

    pygame.K_KP4: (-1, 0), pygame.K_KP6: (1, 0),
    pygame.K_KP8: (0, -1), pygame.K_KP2: (0, 1),
    pygame.K_KP7: (-1, -1), pygame.K_KP9: (1, -1),
    pygame.K_KP1: (-1, 1), pygame.K_KP3: (1, 1),
    pygame.K_KP5: (0, 0),

    pygame.K_h: (-1, 0), pygame.K_l: (1, 0),
    pygame.K_k: (0, -1), pygame.K_j: (0, 1),
    pygame.K_y: (-1, -1), pygame.K_u: (1, -1),
    pygame.K_b: (-1, 1), pygame.K_n: (1, 1),

    pygame.K_PERIOD: (0, 0), pygame.K_SPACE: (0, 0),
}

DEBUG_KEYS = {
    pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3, pygame.K_5: 4,
}

HELP_KEYS = (pygame.K_QUESTION, pygame.K_SLASH, pygame.K_F1)


def update_caption(page: RoguePage, view: View) -> None:
    game = page.game
    state = "dead" if not game.alive else f"{game.monsters_left} still moving"
    pygame.display.set_caption(
        f"Ink & Dungeon  |  turn {game.turn}  "
        f"health {max(0, game.health.current)}/{game.health.maximum}  "
        f"{state}  zoom {view.zoom * 100:.0f}%  |  press ? for controls"
    )


def handle_key(page: RoguePage, event: pygame.event.Event, view: View) -> bool:
    """One keypress. Returns True if the caption needs redoing.

    Split out of the loop so a test can press keys at the game without
    standing up an event queue, and because the order of these branches
    matters: `n` is a direction, so `Shift+N` has to be caught before the
    movement table sees it.
    """
    settings = page.settings

    if getattr(event, "unicode", "") == "?" or event.key in HELP_KEYS:
        # The key that produces `?` moves around between layouts, so trust the
        # character the event carries wherever there is one.
        page.help.toggle()
    elif event.key == pygame.K_n and event.mod & pygame.KMOD_SHIFT:
        page.new_dungeon(page.seed + 1)
        return True
    elif event.key in MOVES:
        if page.game.act(*MOVES[event.key]):
            page.catch_up()
            page.refresh_status()
            return True
    elif event.key == pygame.K_r:
        page.renderer.regenerate_paper()
    elif event.key == pygame.K_w:
        settings.rewet()
    elif event.key in DEBUG_KEYS:
        settings.debug_mode = DEBUG_KEYS[event.key]
    elif event.key in (pygame.K_0, pygame.K_KP0, pygame.K_HOME):
        view.reset()
        return True
    return False


def main(allow_restart: bool = False) -> None:
    pygame.init()
    pygame.font.init()

    settings = InkSettings()
    try:
        size = open_window(WINDOW_SIZE)
        if allow_restart:
            # Before anything is built, since this may not come back.
            restart_on_the_graphics_card(
                moderngl.create_context(require=330).info["GL_RENDERER"]
            )
        page = RoguePage(size, settings)
    except Exception as exc:
        pygame.quit()
        raise SystemExit(
            "Could not create the OpenGL 3.3 / ModernGL renderer. "
            "Update the graphics driver or relax the requested context version.\n"
            f"Original error: {exc}"
        ) from exc

    # Worth one line: which of these it is decides whether this runs at sixty
    # frames a second or at fifteen, and nothing on screen says so.
    print(
        f"Rendering with {page.renderer.context.info['GL_RENDERER']}",
        file=sys.stderr,
    )

    clock = pygame.time.Clock()
    view = View(size)
    elapsed = 0.0
    panning = False
    update_caption(page, view)

    running = True
    # The picture is redrawn when something asks for it; these say whether
    # anything has. `idle_frames` forces one through now and then regardless,
    # so a repaint the window manager wanted and did not mention cannot leave
    # the page stale indefinitely.
    needs_frame = True
    idle_frames = 0
    while running:
        events = pygame.event.get()
        # Generous on purpose: it is far cheaper to draw a frame that turned
        # out identical than to decide which events change the picture and be
        # wrong about one of them.
        handled_event = bool(events)
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                panning = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                panning = False
            elif event.type == pygame.MOUSEMOTION and panning:
                # Drag the sheet under the cursor rather than move a camera
                # over it: the page follows the hand, which is what the
                # gesture reads as.
                view.pan_by(event.rel)
            elif event.type == pygame.MOUSEWHEEL:
                view.zoom_by(event.y, pygame.mouse.get_pos())
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif handle_key(page, event, view):
                    update_caption(page, view)

        # The page-wide wetness and each freshly written cell dry on one clock.
        settings.advance(elapsed)
        drying = page.canvas.dry(elapsed, settings.dry_rate)
        if drying is not None:
            page.renderer.upload_wetness(drying)

        # A turn-based game spends most of its life showing an unchanged
        # picture, and drawing it again costs the entire fragment shader --
        # the page is one full-screen quad. So a frame goes out when something
        # asked for one, and while any ink is still setting.
        moving = page.canvas.wet_bounds is not None or (
            settings.drying and settings.wetness > PAGE_WET_FLOOR + 1e-4
        )
        if handled_event or moving or needs_frame or idle_frames >= IDLE_REDRAW_FRAMES:
            # Worked out per frame rather than stored: the lamp is over the
            # player in page space, but it is placed in window space, so
            # panning and zooming move it just as much as walking does.
            page.draw(page.lamps(view), view)
            pygame.display.flip()
            needs_frame = False
            idle_frames = 0
        else:
            idle_frames += 1

        elapsed = clock.tick(60) / 1000.0

    page.release()
    pygame.quit()


def play_headless(
    destination: Path,
    size: tuple[int, int] = WINDOW_SIZE,
    turns: int = 30,
    seed: int = 1,
) -> Path:
    """Play a few turns with no window at all, and save the page.

    The point of the exercise: the same renderer, the same page and the same
    game as the interactive version, with nothing between them and a file.
    """
    import random

    pygame.init()
    pygame.font.init()

    settings = InkSettings()
    page = RoguePage(size, settings, seed=seed, headless=True)

    rng = random.Random(seed)
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (-1, -1)]
    for _ in range(turns):
        if not page.game.alive:
            break
        if page.game.act(*rng.choice(directions)):
            page.catch_up()
            page.refresh_status()

    view = View(size)
    page.draw(page.lamps(view), view)
    pygame.image.save(page.renderer.snapshot(), str(destination))
    page.release()
    pygame.quit()
    return destination


if __name__ == "__main__":
    if "--headless" in sys.argv:
        index = sys.argv.index("--headless")
        target = (
            Path(sys.argv[index + 1])
            if len(sys.argv) > index + 1
            else Path("rogue_headless.png")
        )
        print(f"wrote {play_headless(target)}")
    else:
        # Only when run as a program: importing this must never relaunch
        # anything, or a test harness that imported it would restart itself.
        main(allow_restart=True)
