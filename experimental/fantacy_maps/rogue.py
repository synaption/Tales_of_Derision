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

The pieces:

    roguesim.py  the dungeon: entities, rooms, sight, hunting. No pixels and
                 no pygame; it can be played to the end with no display at all.
    inkfx.py     the parchment and the ink.
    this file    the tileset, the layout, and the loop tying them together.

Run it with `python3 rogue.py`. `python3 rogue.py --headless out.png` plays a
few turns and saves the page with no window involved anywhere.

Controls are on the `?` card; the short version is that the arrows, the numeric
keypad or the vi keys move, `.` waits, `Shift+N` starts a new dungeon and `R`
a new sheet of parchment.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import moderngl
import pygame

import roguesim
from inkfx import (
    INK_SUPERSAMPLE,
    PAGE_WET_FLOOR,
    PANEL_KEY,
    PANEL_TEXT,
    PANEL_TITLE,
    InkCanvas,
    InkSettings,
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

# Roughly how many dungeon cells fit across the page, whatever size it is. The
# cell size follows from this, so the game looks the same at 800x560 as at 4K
# rather than turning into a differently sized dungeon.
TARGET_COLUMNS = 30

# How dark the ink is. Cells in sight are drawn at full strength; cells you
# have been to but cannot currently see are the same glyph, faint -- a map
# drawn from memory rather than one you are looking at.
INK_SEEN = 255
INK_REMEMBERED = 84

STATUS_LINES = 5


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
    """The page divided into a grid, with a strip of text under it.

    All page pixels. The ink mask is `scale` times finer, and that conversion
    happens at the point of drawing rather than being carried around, so there
    is only ever one kind of coordinate in this file.
    """

    cell: int
    columns: int
    rows: int
    origin: tuple[int, int]
    status_top: int
    text_size: int
    line_height: int

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
    """Fit a dungeon and a message strip onto a page of this size."""
    width, height = size
    cell = max(8, width // TARGET_COLUMNS)
    margin = max(4, cell // 2)
    text_size = max(11, round(cell * 0.46))
    line_height = round(text_size * 1.35)
    status = STATUS_LINES * line_height + margin

    columns = max(4, (width - margin * 2) // cell)
    rows = max(4, (height - status - margin * 2) // cell)

    # Centre the grid in whatever is left over, so a page whose width does not
    # divide evenly does not put all its slack down one side.
    left = margin + ((width - margin * 2) - columns * cell) // 2
    return Layout(
        cell=cell,
        columns=columns,
        rows=rows,
        origin=(left, margin),
        status_top=margin + rows * cell + margin,
        text_size=text_size,
        line_height=line_height,
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

        scale = self.canvas.scale
        self.font = pygame.font.SysFont("serif", self.layout.text_size * scale)
        self.title_font = pygame.font.SysFont(
            "serif", round(self.layout.text_size * 1.15) * scale, bold=True
        )

        self.help = _help_card(self.renderer.context, size)
        self.new_dungeon(seed)

    # --- the seam the loop talks to --------------------------------------

    @property
    def canvas(self) -> InkCanvas:
        return self.renderer.canvas

    @property
    def size(self) -> tuple[int, int]:
        return self.renderer.size

    def draw(self, light_position: tuple[int, int], view: View) -> None:
        """The page, then the chrome, in the order they have to appear."""
        self.renderer.render(light_position, self.settings, view)
        if self.help.visible:
            self.help.draw()

    def release(self) -> None:
        self.help.release()
        self.renderer.release()

    # --- keeping the page and the game in step ---------------------------

    def new_dungeon(self, seed: int) -> None:
        """Start a fresh game, and print its opening view on a blank sheet."""
        self.seed = seed
        self.game = roguesim.Game(self.layout.grid_size, seed=seed)

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
        self._draw_status()

    def catch_up(self) -> None:
        """Repaint whatever the game says has changed, and wet what is new.

        Called after every turn. The work is proportional to what moved -- a
        step lights perhaps thirty cells -- rather than to the size of the
        page, which at 2x supersampling is eight megapixels of surface and the
        best part of a tenth of a second to send to the card.
        """
        dirty, fresh = self.game.take_dirty()
        if not dirty:
            return

        base = self.canvas.base
        for x, y in dirty:
            region = self.layout.cell_rect(x, y)
            mask_rect = self.canvas.mask_rect(region)
            base.fill((0, 0, 0, 0), mask_rect)

            code = self._glyph_at(x, y)
            if code is not None and self.tileset is not None:
                base.blit(
                    self.tileset.glyph(
                        code,
                        self.layout.cell * self.canvas.scale,
                        INK_SEEN if self.game.visible[x, y] else INK_REMEMBERED,
                    ),
                    mask_rect,
                    special_flags=pygame.BLEND_RGBA_MAX,
                )
            self._publish(region)

        if fresh:
            self._wet(fresh)

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
        """Reprint the message strip, if the game has said anything new."""
        if not self.game.messages_changed:
            return
        self.game.messages_changed = False
        self._draw_status()

    def _draw_status(self) -> None:
        """The health, the turn count and the last few messages, in ink.

        Written onto the page rather than composited over it as a panel: it is
        the margin of the map, so it should soak into the parchment and dry
        alongside everything else drawn there.
        """
        layout = self.layout
        scale = self.canvas.scale
        strip = pygame.Rect(
            0,
            layout.status_top,
            self.size[0],
            self.size[1] - layout.status_top,
        )
        self.canvas.base.fill((0, 0, 0, 0), self.canvas.mask_rect(strip))

        health = self.game.health
        left = layout.origin[0] * scale
        top = layout.status_top * scale

        heading = (
            f"turn {self.game.turn}     "
            f"health {max(0, health.current)}/{health.maximum}     "
            f"still moving: {self.game.monsters_left}"
        )
        self.canvas.base.blit(
            self.title_font.render(heading, True, (255, 255, 255)), (left, top)
        )
        for index, message in enumerate(self.game.messages, start=1):
            self.canvas.base.blit(
                self.font.render(message, True, (255, 255, 255)),
                (left, top + index * layout.line_height * scale),
            )

        self._publish(strip)
        # It was written just now, so it is wet just like the map is.
        self.canvas.wetness.fill((255, 255, 255), strip)
        self.renderer.upload_wetness(self.canvas.mark_wet(strip))


# --- the help card --------------------------------------------------------

HELP_LINES = (
    ("", "Ink & Dungeon"),
    ("arrows / numpad / hjkl yubn", "move, or attack what is there"),
    (". or 5 or space", "wait a turn"),
    ("Shift+N", "a new dungeon on a clean sheet"),
    ("R", "a fresh sheet of parchment"),
    ("W", "wet the whole page again"),
    ("mouse", "move the lamp"),
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
    light_position = (size[0] // 2, size[1] // 2)
    elapsed = 0.0
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
            elif event.type == pygame.MOUSEMOTION:
                light_position = event.pos
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
            page.draw(light_position, view)
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

    page.draw((size[0] // 2, size[1] // 3), View(size))
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
