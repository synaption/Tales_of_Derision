"""The juice workbench, rendered on the graphics card.

Same bench, same sim, same panel. The renderer is the only thing that changed,
and the import list is the argument for why the split was worth having:

    from juicefx import ...        # every effect, as maths. Unchanged.
    from rogue_juice import World  # the sim, the toggles, the sliders.
    from audiofx import SoundBank  # unchanged.
    from terrain import build_hills  # the hills, as data. No pygame, no GL.

The sim has exactly one thing in it that this file put there, and it is worth
naming because the rest of the split depends on nobody adding a second: hills
need the sim to have an opinion about whether a step is legal, which is not a
rendering question and cannot be answered from out here. So `World` grew
`terrain` (a slot, empty by default) and `step_allowed` (a method that returns
`True` when the slot is empty). The software bench never fills it and is the
arena it always was; this one fills it with a `terrain.Terrain` and everything
else about verticality -- how high a level looks, which sprites are lifted,
what the cliff face is made of, whether corpses roll off it -- stays here.

Not one line of `juicefx.py` knows this file exists. `Body` already carried
`ox, oy, sx, sy, angle, alpha, flash` and a lag vector, which turns out to be a
per-instance attribute block; the springs that drove ten CPU-drawn bands now
drive a continuous displacement field, and nothing about the springs changed.

What is different to look at, in the order it is worth looking at it:

* **the slime** deforms continuously instead of in ten strips, so it can smear
  as far as you like without coming apart -- there is no `link` constraint here
  because there is nothing to hold together;
* **creatures are lit**, with normals generated from their own silhouettes, so
  a torch rakes across them and a hit flash shades the room;
* **sprites move sub-pixel and stay crisp**, which the software bench has to
  offer as a three-way compromise;
* **the floor ripples continuously**, as a displacement of the texture rather
  than a repaint of the tiles it passes;
* **the dead come off the grid**, into a real-time circle-and-velocity solver:
  bodies slide, bounce off walls and each other, are shoved out of the way by
  anything that walks through them, and cartwheel away from an explosion.
  `RagdollField` is the only thing here that is not a renderer, and it is here
  because a corpse takes no turn and blocks no tile, so nothing is left that
  needs it on a grid at all. Press **B** to throw a bomb -- which arcs, bounces
  off walls through the corpses' own wall pass, and goes off on a fuse;
* **the floor has hills in it**, which is the one thing here that is not
  purely a picture. Ground comes in whole levels: same level is one floor, the
  boundary between two is a cliff nobody walks up, and a stair is the single
  tile that ramps between them -- so the sim has a new answer to "can I go
  that way", and it is one method (`World.step_allowed`) with a flat default.
  Everything else about it *is* a picture: a level is a number of pixels a
  thing is drawn further up the screen, its shadow is not moved with it, and
  the floor shader works out which terrace is visible at each fragment and
  draws the cliff face hanging under it. A creature behind a hill is hidden by
  it without a depth buffer or a sort -- its fragments ask the terrain the same
  question the floor does and stand down -- which is what keeps the batch to
  one draw call in painter's order and the sparks additive. `terrain.py` holds
  the model and has no pygame and no GL in it, so which tiles are walkable is a
  question with a headless answer;
* **one draw call** for every sprite, spark, shadow, decal, ring, arc and cut.

The panel is drawn by the software renderer into an offscreen surface and
uploaded as a texture, which keeps every slider, blurb and scroll behaviour
identical and costs one upload a frame. Text layout is the one thing pygame
does better than a weekend of shader work, and the escape menu is drawn into
the same surface for the same reason.

**Escape** opens that menu: resume, display options, save the current settings
as your defaults, quit. The window's size, fullscreen and frame cap are live;
the settings land in `juice_settings.json` next to the checkout, under merge
rules (`juicesettings.py`) that let the file and the build disagree in either
direction, because this bench grows a slider most times anybody opens it.

    python3 rogue_juice_gl.py
    python3 rogue_juice_gl.py --headless out.png
"""

from __future__ import annotations

import math
import os
import signal
import sys
import time
from dataclasses import dataclass

if "--headless" in sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import moderngl
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glfx  # noqa: E402
import juicesettings  # noqa: E402
import rogue_juice as rj  # noqa: E402
import terrain as terrainfx  # noqa: E402
import tiles  # noqa: E402
from juicefx import ambient_offset, clamp, hash01, shadow_of  # noqa: E402

try:
    import audiofx
    HAVE_AUDIO = True
except Exception:                                    # pragma: no cover
    audiofx = None                                   # type: ignore
    HAVE_AUDIO = False


TILE = rj.TILE
VIEW_W, VIEW_H = rj.VIEW_W, rj.VIEW_H
WIN_W, WIN_H = rj.WIN_W, rj.WIN_H
GRID_W, GRID_H = rj.GRID_W, rj.GRID_H
PX = rj.PX
FPS = rj.FPS


# ---------------------------------------------------------------------------
# The extra controls the card makes possible
# ---------------------------------------------------------------------------
#
# Added to the existing registry rather than replacing it, so the panel, the
# blurbs, the scrolling and the drag handling all carry over untouched. Two
# software-only controls are dropped because the hardware makes the question
# they answer disappear.

GL_TOGGLES = [
    ("normals", "normal-mapped sprites", "screen",
     "Light the creatures with normals generated from their own silhouettes, "
     "so a torch rakes across a body instead of uniformly brightening it. No "
     "new art: a blurred alpha mask's gradient is the normal of an inflated "
     "shape."),
    ("softbody", "continuous soft body", "movement",
     "Deform the slime as a field rather than as ten strips. Off, it falls "
     "back to a rigid drag -- which is what a blit can do, and the reason the "
     "CPU bench needs bands and a surface-tension clamp at all."),
    ("sharp", "sharp bilinear", "screen",
     "Flat inside a texel, blended across the boundary over one screen pixel. "
     "Crisp at rest and smooth at any sub-pixel speed. Off is plain bilinear, "
     "which is the mush that made everyone snap to whole pixels in the first "
     "place."),
    ("grade", "impact grade", "screen",
     "Push the whole frame warm for a few frames on contact. A colour shift is "
     "a hit confirmation the eye reads without noticing it."),
    ("scanlines", "scanlines", "screen",
     "A CRT line pattern. Free here, and impossible on the CPU without walking "
     "every pixel.", False),
    ("refract", "heat refraction", "screen",
     "Bend the frame around each torch. The software bench re-blits ninety rows "
     "per torch to fake this; here every fragment simply asks where to sample."),
    ("hills", "hills and stairs", "world",
     "Give the floor a third dimension. Ground comes in whole levels: tiles at "
     "the same level are one continuous floor, the boundary between two is a "
     "cliff you cannot walk up or down, and a stair is the one tile that ramps "
     "between them. Everything standing on high ground is drawn lifted -- and "
     "its shadow is not, which is the entire height cue -- while the lights, "
     "the wall shadows and the sim's own idea of where anybody is stay on the "
     "flat plane, because a grid with a genuine third axis in it is a "
     "different game. Off, the same arena is flat and every step is legal, "
     "which is the comparison: watch how much of 'terrain' is one number and a "
     "shadow that refuses to move."),
    ("ragdoll", "corpse ragdolls", "attack",
     "Take the dead off the grid. A corpse has no turn and blocks nothing, so "
     "nothing is left that needs it on a tile -- it becomes a circle with a "
     "velocity that slides, bounces off walls, is shoved aside in real time by "
     "anything that walks through it, and goes end over end when something "
     "detonates next to it. Press B to throw a bomb and watch. Off, bodies lie "
     "exactly where they fell, which is what `leave_corpse` does on its own."),
]

GL_PARAMS = [
    ("soft_power", "soft-body falloff", "movement", 1.7, 0.4, 5.0,
     "Shape of the displacement gradient across the body. 1 is a linear smear, "
     "high numbers keep the front rigid and let only the tail string out.",
     "{:.2f}"),
    ("sharpness", "filter sharpness", "screen", 1.0, 0.0, 1.0,
     "0 is plain bilinear, 1 is sharp bilinear. Drag it down while something "
     "is moving slowly to see exactly what the filter is buying.", "{:.2f}"),
    ("normal_depth", "normal depth", "screen", 1.0, 0.0, 2.0,
     "How far the generated normals turn away at a sprite's rim. Past about "
     "1.5 the creatures start to look like foil balloons.", "{:.2f}"),
    ("light_height", "light height", "screen", 82.0, 2.0, 240.0,
     "How far the lights float above the floor plane. This is the number that "
     "decides whether a torch lights a room or lights a coin: low is a raking, "
     "dramatic light that dies within a tile, high flattens everything out.",
     "{:.0f}"),
    ("light_contrast", "light / dark contrast", "screen", 1.0, 0.0, 10.0,
     "Difference between lit and shadowed areas. Zero flattens the lighting; "
     "one is natural; ten removes ambient light completely, making areas "
     "outside direct light pitch black.", "{:.2f}"),
    ("wall_shadow_amt", "wall shadow amount", "screen", 1.0, 0.0, 1.0,
     "Opacity of shadows cast by walls. One makes walls fully block direct "
     "light; zero lets light pass through them.", "{:.2f}"),
    ("npc_shadow_amt", "NPC shadow amount", "screen", 0.55, 0.0, 1.0,
     "Opacity of the soft shadows cast by monsters and the training dummy, "
     "independent of wall shadows.", "{:.2f}"),
    ("bloom_knee", "bloom knee", "screen", 0.45, 0.01, 2.0,
     "How softly the threshold lets a pixel in. A hard knee makes bloom "
     "flicker on moving highlights.", "{:.2f}"),
    ("grade_amt", "grade strength", "screen", 0.5, 0.0, 1.0,
     "How far the frame is pushed warm on impact.", "{:.2f}"),
    ("scanline_amt", "scanline depth", "screen", 0.14, 0.0, 0.6,
     "", "{:.2f}"),

    ("hill_rise", "level height", "world", 14.0, 0.0, 34.0,
     "How far one level of ground lifts what is standing on it, in pixels. "
     "This is the only number that decides whether the terrain reads as a kerb "
     "or as a plateau, and it is worth dragging to both ends: at two pixels "
     "the cliff faces are a drawn line and the hill is a floor pattern, and "
     "somewhere around ten the same geometry starts being a place you climb "
     "onto. Nothing about the sim changes as it moves -- the stairs are in the "
     "same tiles at either end.", "{:.0f}"),
    ("hill_shade", "cliff shade", "world", 0.42, 0.05, 1.0,
     "How dark the exposed side of the ground is against its top. One is no "
     "difference at all, which makes a plateau look like a floor that has slid "
     "upwards; around a half reads as rock in its own shadow.", "{:.2f}"),
    ("hill_slide", "downhill slide", "world", 34.0, 0.0, 120.0,
     "How hard gravity pulls a corpse along sloping ground. This is what a "
     "cliff is *for*: bodies do not settle on a ledge, they go over it, fall "
     "the height of the level and pile up at the bottom -- and a bomb thrown "
     "onto a hill rolls off it before the fuse runs out. Zero leaves the dead "
     "lying wherever the blast put them, on a slope or not.", "{:.0f}"),

    ("rag_size", "corpse radius", "attack", 0.34, 0.12, 0.70,
     "Collision radius of a body, in tiles. This is the one number that "
     "decides whether a pile of corpses reads as a heap or as a stack of "
     "coins, and it is also what keeps them out of walls.", "{:.2f}"),
    ("rag_bounce", "corpse bounce", "attack", 0.36, 0.0, 0.95,
     "How much speed survives hitting a wall. Zero is a sack of wet sand, "
     "which is honest; a third of it back is what reads as a body with bones "
     "in. Past about 0.7 they pinball and stop looking dead.", "{:.2f}"),
    ("rag_drag", "floor drag", "attack", 5.0, 0.2, 14.0,
     "How fast the floor takes the speed back, per second. A body coasts "
     "roughly its launch speed divided by this, in pixels, before it stops -- "
     "so low numbers slide corpses around the room like curling stones.",
     "{:.1f}"),
    ("rag_shove", "shove strength", "attack", 1.8, 0.0, 6.0,
     "How hard a walking body pushes a dead one out of the way. This is the "
     "whole point of taking corpses off the grid: on it, the only two answers "
     "to walking into a body are 'blocked' and 'nothing there'.", "{:.2f}"),
    ("rag_blast", "blast force", "attack", 1100.0, 0.0, 6000.0,
     "Speed, in pixels a second, given to a body at the dead centre of a "
     "bomb, falling off to nothing at the edge of the wave and divided by the "
     "body's weight. Smaller rings -- the one an ordinary killing blow raises "
     "-- get a steeply smaller share of it.", "{:.0f}"),
    ("rag_launch", "death launch", "attack", 170.0, 0.0, 900.0,
     "The shove a fresh corpse gets away from whatever killed it, on top of "
     "whatever speed the death animation left it carrying. Zero drops it "
     "straight down where it stood.", "{:.0f}"),
    ("rag_spin", "tumble", "attack", 1.3, 0.0, 5.0,
     "How much of a body's sideways speed turns into rotation on a bounce. "
     "Zero slides them around flat; high numbers make everything cartwheel.",
     "{:.2f}"),
    ("bomb_throw", "bomb throw speed", "attack", 460.0, 60.0, 1400.0,
     "How hard B lobs a bomb, in pixels a second. It leaves the hand on an "
     "arc, bounces off walls, rolls, and goes off where it stops -- so this "
     "slider is really 'how far away can you put an explosion'.", "{:.0f}"),
    ("bomb_fuse", "bomb fuse", "attack", 0.95, 0.05, 4.0,
     "Seconds between the throw and the bang. Short enough and it goes off in "
     "the air, which is a different weapon; long enough and you have time to "
     "walk into your own blast.", "{:.2f}s"),
]

#: The GL bench's own key list, shown in the panel footer. The software
#: bench's `HELP_LINES` is left alone: `p` does nothing here, `esc` opens a
#: menu instead of quitting, and `b` throws a bomb, none of which is true over
#: there. `rj.Renderer.help_lines` is the seam.
GL_HELP_LINES = [
    "click a row to toggle   drag a slider",
    "right-click a slider to reset it",
    "wheel scrolls this panel",
    "",
    "wasd / arrows  move, walk into a thing",
    "cliffs bump -- stairs are the way up",
    "space swing   x get hit   K kill   r reset",
    "b throw a bomb   Tab A/B   F1/F2 on/off",
    "F3 defaults   [ ] easing   m music   - = amount",
    "esc  menu, options, save and quit",
]

#: Software-only controls that the hardware makes meaningless.
#: `pixel_mode` existed because rounding a blit destination was the only
#: control there was; `light warmth` existed because a multiply cannot brighten.
DROP_CHOICES = ["pixel_mode"]
DROP_PARAMS = ["light_warm"]


def gl_juice() -> rj.Juice:
    """The software registry plus the hardware's own controls."""
    juice = rj.Juice()
    for spec in GL_TOGGLES:
        key, label, group, blurb = spec[:4]
        on = spec[4] if len(spec) > 4 else True
        juice.toggles[key] = rj.Toggle(key, label, group, blurb, on)
    for spec in GL_PARAMS:
        juice.params.add(*spec)
    for key in DROP_CHOICES:
        juice.choices.pop(key, None)
    for key in DROP_PARAMS:
        juice.params.params.pop(key, None)
    # Lighting is cheap enough to leave on now: on the CPU it was 3.8ms and
    # defaulted off, here it is part of the same pass that was going to run
    # anyway.
    juice.toggles["light"].on = True
    # Bigger than the software default, which was sized around a light pass
    # that cost four milliseconds. This one is part of a pass that was going to
    # run anyway, so the radius can be what looks right.
    juice.params["light_radius"].value = 7.5
    juice.params["light_radius"].default = 7.5
    return juice


# ---------------------------------------------------------------------------
# Display settings and the escape menu
# ---------------------------------------------------------------------------
#
# Two things that only exist once a bench is something you *open* rather than
# something you run: a window whose size is your decision, and somewhere to
# say so. Both are deliberately outside the panel -- the panel is the subject
# of the experiment and every row on it is an effect being argued about, while
# a resolution is a fact about your monitor.


#: Offered in the options list, smallest first. The desktop's own size is
#: appended at runtime if it is not already here, because that is the one
#: everybody actually wants and it is the one that cannot be hard-coded.
RESOLUTIONS = [
    (1180, 600),      # the original: an 800x600 view and a 380px panel
    (1280, 720),
    (1440, 810),
    (1600, 900),
    (1920, 1080),
]

#: 0 means "as fast as it will go", which is the honest way to read the ms
#: counter -- a capped frame time tells you what the cap is, not what the
#: frame costs.
FRAME_CAPS = [30, 60, 120, 144, 240, 0]

#: The view cannot get so small that the panel is the window. Below this the
#: camera clamp starts fighting the arena and the HUD runs off the bottom.
MIN_VIEW = (480, 360)


@dataclass
class Display:
    """Everything about the window, in one place that can be saved.

    Separate from `Juice` on purpose. A juice parameter is a claim about what
    looks good and belongs in the file the bench argues about; a resolution is
    a fact about the machine, and mixing the two means copying a settings file
    to another computer breaks the window.
    """

    width: int = rj.WIN_W
    height: int = rj.WIN_H
    fullscreen: bool = False
    frame_cap: int = FPS
    #: Applied when the window is created, so a change here is a note for the
    #: next launch. Honest labelling in the menu beats pretending otherwise.
    vsync: bool = True

    @property
    def size(self):
        return self.width, self.height

    def clamped(self):
        """The size actually usable, with the panel and the minimum view."""
        w = max(MIN_VIEW[0] + rj.PANEL_W, int(self.width))
        h = max(MIN_VIEW[1], int(self.height))
        return w, h

    def load(self, settings: juicesettings.Settings):
        self.width = settings.get("display", "width", self.width)
        self.height = settings.get("display", "height", self.height)
        self.fullscreen = settings.get("display", "fullscreen", self.fullscreen)
        self.frame_cap = settings.get("display", "frame_cap", self.frame_cap)
        self.vsync = settings.get("display", "vsync", self.vsync)
        return self

    def store(self, settings: juicesettings.Settings):
        settings.set("display", "width", int(self.width))
        settings.set("display", "height", int(self.height))
        settings.set("display", "fullscreen", bool(self.fullscreen))
        settings.set("display", "frame_cap", int(self.frame_cap))
        settings.set("display", "vsync", bool(self.vsync))


def _set_layout(width: int, height: int):
    """Point both modules' layout globals at a new window size.

    The only mutable global state the bench has, and it is here rather than
    spread over two files because getting one of the four out of step is a
    frame that draws the panel over the arena. `rogue_juice.py` computes its
    own geometry from `VIEW_W`/`WIN_W` at call time, and so does everything in
    this file, so this is the whole of a resolution change.
    """
    global VIEW_W, VIEW_H, WIN_W, WIN_H
    rj.WIN_W, rj.WIN_H = int(width), int(height)
    rj.VIEW_W, rj.VIEW_H = int(width) - rj.PANEL_W, int(height)
    VIEW_W, VIEW_H = rj.VIEW_W, rj.VIEW_H
    WIN_W, WIN_H = rj.WIN_W, rj.WIN_H


def apply_window(display: Display) -> bool:
    """Resize (or fullscreen) the real window, keeping the GL context alive.

    `pygame.display.set_mode` would build a new window and take the context
    with it, which means rebuilding every texture, shader and buffer in the
    bench. SDL can simply resize the window it already has, and moderngl never
    notices -- so this goes through `pygame._sdl2` and falls back to reporting
    failure rather than doing anything drastic.
    """
    try:
        from pygame._sdl2.video import Window
        window = Window.from_display_module()
        if display.fullscreen:
            window.set_fullscreen(desktop=True)
        else:
            window.set_windowed()
            window.size = display.clamped()
        return True
    except Exception:                                # pragma: no cover
        return False


def resolution_choices():
    """The offered sizes, with the desktop's own folded in and duplicates gone."""
    sizes = list(RESOLUTIONS)
    try:
        for size in pygame.display.get_desktop_sizes():
            if tuple(size) not in sizes:
                sizes.append(tuple(size))
    except Exception:                                # pragma: no cover
        pass                                         # no display module yet
    return sorted(set(sizes))


class Menu:
    """The escape menu: resume, options, save, quit.

    Drawn with pygame into the same offscreen surface the panel uses and
    uploaded as part of the same texture, which is why it costs nothing extra
    and why every font here is the panel's font. A menu drawn in GL would be a
    text layout engine, and the panel already settled that argument.

    Holds no state that is not the menu's own: which page is showing, which row
    is under the cursor, and nothing else. What a row *does* is the bench's
    business -- `activate` returns a string and the caller decides.
    """

    WIDTH = 380
    ROW_H = 30

    #: (key, label) for the plain pages. Value rows are built per frame,
    #: because their right-hand side is a live reading of `Display`.
    MAIN = [("resume", "Resume"),
            ("options", "Options"),
            ("save", "Save current settings as defaults"),
            ("quit", "Quit")]

    def __init__(self):
        self.open = False
        self.page = "main"
        self.index = 0
        #: (key, rect) for everything drawn this frame, registered from the
        #: same loop that draws it -- so a row can never be clickable where it
        #: is not shown, which is the bug every hand-rolled menu has.
        self.hits: list = []
        self.status = ""
        self.status_age = 0.0

    # -- state --------------------------------------------------------------
    def show(self, page: str = "main"):
        self.open = True
        self.page = page
        self.index = 0

    def close(self):
        self.open = False
        self.page = "main"

    def back(self) -> bool:
        """Escape: out of a sub-page, or out of the menu. True if still open."""
        if self.page != "main":
            self.page = "main"
            self.index = 1                            # back onto "Options"
            return True
        self.close()
        return False

    def say(self, text: str):
        self.status = text
        self.status_age = 0.0

    def rows(self, display: Display):
        """The current page as (key, label, value) triples.

        Built fresh every frame rather than cached, so a value row cannot show
        a stale reading -- which is the only kind of bug an options screen
        really has.
        """
        if self.page == "main":
            return [(key, label, None) for key, label in self.MAIN]
        w, h = display.clamped()
        cap = display.frame_cap
        return [
            ("resolution", "Resolution", f"{w} x {h}"),
            ("fullscreen", "Fullscreen", "on" if display.fullscreen else "off"),
            ("frame_cap", "Frame cap", f"{cap} fps" if cap else "uncapped"),
            ("vsync", "VSync", ("on" if display.vsync else "off") + "  (on restart)"),
            ("back", "Back", None),
        ]

    # -- input --------------------------------------------------------------
    def key(self, event, display: Display):
        """One key press. Returns an action for the bench, or None.

        Arrow keys move and adjust, enter activates, escape steps back. The
        left/right split matters: on a value row they cycle the value and on an
        action row they do nothing, so there is never a keypress that both
        changes a setting and leaves the menu.
        """
        rows = self.rows(display)
        if event.key in (pygame.K_ESCAPE,):
            return "back"
        if event.key in (pygame.K_UP, pygame.K_w, pygame.K_k):
            self.index = (self.index - 1) % len(rows)
            return None
        if event.key in (pygame.K_DOWN, pygame.K_s, pygame.K_j):
            self.index = (self.index + 1) % len(rows)
            return None
        key = rows[self.index][0]
        if event.key in (pygame.K_LEFT, pygame.K_a, pygame.K_h):
            return f"{key}:-1" if rows[self.index][2] is not None else None
        if event.key in (pygame.K_RIGHT, pygame.K_d, pygame.K_l):
            return f"{key}:+1" if rows[self.index][2] is not None else None
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            return f"{key}:+1" if rows[self.index][2] is not None else key
        return None

    def point(self, pos, display: Display):
        """Hover. Returns True if the cursor is over the card at all."""
        for i, (_key, rect) in enumerate(self.hits):
            if rect.collidepoint(pos):
                self.index = i
                return True
        return False

    def click(self, pos, button: int, display: Display):
        """A click on a row. Right-click steps a value backwards."""
        for i, (key, rect) in enumerate(self.hits):
            if not rect.collidepoint(pos):
                continue
            self.index = i
            rows = self.rows(display)
            if i < len(rows) and rows[i][2] is not None:
                return f"{key}:{'-1' if button == 3 else '+1'}"
            return key
        return None

    # -- drawing ------------------------------------------------------------
    def draw(self, surface, ui: rj.Renderer, display: Display, dt: float):
        """Dim the frame, then a card in the middle of it."""
        self.status_age += dt
        rows = self.rows(display)
        self.index = min(self.index, len(rows) - 1)
        self.hits.clear()

        win_w, win_h = surface.get_size()
        veil = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
        veil.fill((6, 7, 10, 208))
        surface.blit(veil, (0, 0))

        title = "OPTIONS -- DISPLAY" if self.page != "main" else "JUICE WORKBENCH"
        body_h = len(rows) * self.ROW_H
        card = pygame.Rect(0, 0, self.WIDTH, body_h + 104)
        card.center = (win_w // 2, win_h // 2)
        pygame.draw.rect(surface, rj.PANEL_BG, card)
        pygame.draw.rect(surface, rj.PANEL_LINE, card, 1)

        surface.blit(ui.font_big.render(title, True, rj.INK),
                     (card.left + 18, card.top + 14))
        pygame.draw.line(surface, rj.PANEL_LINE,
                         (card.left + 14, card.top + 40),
                         (card.right - 14, card.top + 40))

        y = card.top + 52
        for i, (key, label, value) in enumerate(rows):
            rect = pygame.Rect(card.left + 8, y, card.width - 16, self.ROW_H - 4)
            self.hits.append((key, rect))
            picked = i == self.index
            if picked:
                pygame.draw.rect(surface, rj.TRACK_FILL, rect)
            colour = rj.INK if picked else rj.DIM
            surface.blit(ui.font_ui_b.render(label, True, colour),
                         (rect.left + 10, rect.top + 5))
            if value is not None:
                text = f"< {value} >" if picked else value
                surf = ui.font_ui.render(text, True, rj.GOLD if picked else rj.DIM)
                surface.blit(surf, (rect.right - 10 - surf.get_width(),
                                    rect.top + 6))
            y += self.ROW_H

        hint = ("arrows move and change   enter picks   esc backs out"
                if self.page != "main" else
                "arrows move   enter picks   esc resumes")
        surface.blit(ui.font_ui.render(hint, True, rj.DIM),
                     (card.left + 18, card.bottom - 40))
        if self.status and self.status_age < 6.0:
            surface.blit(ui.font_ui.render(self.status, True, rj.ACCENT),
                         (card.left + 18, card.bottom - 24))


# ---------------------------------------------------------------------------
# Corpse physics
# ---------------------------------------------------------------------------
#
# Everything alive in this bench is on the grid, and it is on the grid because
# the *sim* needs it there: a living thing takes a turn, occupies a tile, and
# has to be something you can walk into. A corpse has none of that. It takes no
# turn, it blocks nothing, and the only thing left that cares where it is, is
# the eye.
#
# So the moment `leave_corpse` drops one on the floor it is adopted by this
# field, and from there it stops being a tile and becomes a circle with a
# velocity: it slides, it bounces off walls and off other bodies, it is shoved
# aside in real time by anything that walks through it, and it goes end over
# end when something detonates next to it. None of that is turn-based, because
# none of it needs to be -- there is no question the sim can ask that this can
# answer wrongly.
#
# Two deliberate non-features:
#
# * it is a *circle*, not a jointed skeleton. Sixteen sprites' worth of art is
#   one quad each; there are no limbs to articulate, so a per-limb solver would
#   be solving for a picture nobody is drawing. One circle plus a rotation is
#   the whole of what a 40x40 body can show.
# * it never touches `rogue_juice.py`. It reads the wall grid through
#   `world.blocked`, reads entity positions through `world.entities`, watches
#   `world.fx.shockwaves` for explosions, and writes `x`, `y` and `angle` back
#   onto the `Corpse` objects the renderer was already drawing. Nothing in here
#   imports moderngl and nothing in here needs a frame, which is what lets
#   `gltest.py` drive the whole system on a machine with no GL at all.


@dataclass
class Ragdoll:
    """One dead body, off the grid, in world pixels.

    `corpse` is the `juicefx.Corpse` this drives; the field owns a strong
    reference to it for exactly as long as it is tracked, which is also what
    makes it safe to key the lookup table on `id()`.
    """

    corpse: object
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    angle: float = 90.0            # degrees, the Corpse convention: +-90 is flat
    spin: float = 0.0              # degrees a second
    radius: float = TILE * 0.34
    mass: float = 1.0
    #: Height of the ground under it, in pixels, and its own height above that
    #: ground. Both are zero on a flat arena and neither is a third axis: `z`
    #: exists so that sliding off a ledge is a fall rather than a teleport, and
    #: `ground` so the body knows the moment the floor went away. A `Bomb`
    #: carries the same two under the same names, which is what lets one wall
    #: solver serve both.
    z: float = 0.0
    vz: float = 0.0
    ground: float = 0.0
    #: An impact wobble, as a signed fraction: +ve is squat and wide. Sprung
    #: rather than decayed, so a body that lands hard overshoots once.
    squash: float = 0.0
    squash_vel: float = 0.0
    resting: bool = False

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def scale(self):
        """(sx, sy) for the quad. Along the sprite's own axes, so a body that
        slams into a wall flattens across whichever way it happens to be
        lying -- which is the right answer often enough at this size."""
        q = clamp(self.squash, -0.4, 0.4)
        return 1.0 + q, 1.0 - q


@dataclass
class Bomb:
    """A thrown bomb, in the air and then on the floor.

    Carries the same `x, y, vx, vy, radius, spin, squash_vel` block a `Ragdoll`
    does, on purpose: `_walls` never asks what it is holding, so a bomb bounces
    off masonry through exactly the code the corpses use and there is only one
    place where a circle can be wrong about a wall.

    `z` is the only thing it has that a body does not -- height above the
    floor, which a corpse never needed because a corpse is always on it.
    """

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    z: float = 0.0                 # pixels above the floor
    vz: float = 0.0
    ground: float = 0.0            # height of that floor, if the arena has hills
    radius: float = TILE * 0.22
    spin: float = 0.0
    angle: float = 0.0
    mass: float = 1.0
    fuse: float = 1.0
    max_fuse: float = 1.0
    squash: float = 0.0            # unused, and here so `_walls` can write it
    squash_vel: float = 0.0

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def t(self) -> float:
        return 1.0 - clamp(self.fuse / self.max_fuse) if self.max_fuse else 1.0


class RagdollField:
    """The bodies on the floor, and the physics that moves them.

    Stepped by the game loop, not by the renderer: `render()` must be free to
    run twice on the same world (the headless path does exactly that) without
    time passing, so nothing in here is called from a draw.

    Owned by `GLRenderer` only because the bench has nowhere else to hang
    per-world state without editing the sim -- see `GLRenderer.ragdolls`.
    """

    #: The physics runs at a fixed rate and the frame is chopped into as many
    #: of these as it needs. A body leaving a blast at 3000px/s covers two
    #: tiles in a 60Hz frame, and a single step that long walks straight
    #: through a wall without ever overlapping it.
    MAX_SUBSTEPS = 8

    #: Below this, in pixels a second, a body is simply stopped. Without a
    #: floor on the speed, drag is asymptotic and every corpse in the room
    #: creeps forever at a hundredth of a pixel.
    SLEEP_SPEED = 5.0 * PX
    SLEEP_SPIN = 8.0               # degrees a second

    #: A bomb's ring, and the yardstick every other shockwave in the bench is
    #: measured against -- see `_wave_force`.
    BOMB_RADIUS = TILE * 4.2
    #: Force reaches this much further than the ring is drawn, so a body just
    #: outside the visible edge still gets shifted.
    BLAST_REACH = 1.6

    #: Gravity for anything with a height, in pixels a second squared.
    GRAVITY = 1500.0 * PX

    def __init__(self, seed: int = 0x1D0D):
        self.bodies: list[Ragdoll] = []
        #: Bombs in flight. Kept beside the corpses rather than in with them
        #: because they are the one thing here that is not dead: they have a
        #: height, a fuse, and an ending.
        self.thrown: list[Bomb] = []
        self._by_corpse: dict[int, Ragdoll] = {}
        #: id(entity) -> [entity, x, y, vx, vy], last frame. The strong
        #: reference is the point: it keeps `id()` from being recycled under
        #: us, and it keeps the corpse's killer alive for the one frame
        #: between an entity being removed and its body being adopted.
        self._tracked: dict[int, list] = {}
        #: id(entity) -> (vx, vy, spin) owed to a body that was blown up while
        #: it was still playing its death animation. Spent when it lands.
        self._owed: dict[int, tuple] = {}
        #: id(shockwave) -> shockwave, for the ones already turned into force.
        self._seen_waves: dict[int, object] = {}
        self.seed = seed
        self._n = 0

    # -- deterministic noise ----------------------------------------------
    def _rand(self) -> float:
        """0..1, from the same integer sequence every run. Same reason the
        particle system has one: a bench that cannot be replayed cannot be
        compared against itself."""
        self._n += 1
        return (hash01(self._n, self.seed) + 1.0) * 0.5

    # -- lookup -------------------------------------------------------------
    def of(self, corpse):
        """The physics state for a corpse, or None if it is not tracked."""
        return self._by_corpse.get(id(corpse))

    def clear(self):
        self.bodies.clear()
        self.thrown.clear()
        self._by_corpse.clear()
        self._tracked.clear()
        self._owed.clear()
        self._seen_waves.clear()

    # -- the frame ----------------------------------------------------------
    def update(self, world: rj.World, dt: float):
        """Adopt, then simulate. Call once per frame, after `world.update`."""
        self._retire(world)
        gone = self._departed(world)
        self._adopt(world, gone)
        self._track(world, dt)
        self._watch_explosions(world)

        j = world.juice
        if dt <= 0.0:
            return
        # A bomb in the air is not a corpse and does not stop being a bomb
        # because the ragdoll toggle is off -- the explosion is still worth
        # having with the bodies nailed down, which is exactly the comparison
        # the toggle is for.
        self._fly(world, dt)
        if not j.on("ragdoll"):
            # Bodies left exactly where they are, including any velocity they
            # had when the toggle went off, so switching it back on resumes
            # rather than teleporting.
            return

        # Sub-stepping is priced off the fastest body in the room, so the
        # common case -- everything lying still -- is one step.
        fastest = max((b.speed for b in self.bodies), default=0.0)
        steps = 1
        if fastest * dt > TILE * 0.3:
            steps = min(self.MAX_SUBSTEPS, int(fastest * dt / (TILE * 0.3)) + 1)
        h = dt / steps
        for _ in range(steps):
            for b in self.bodies:
                self._integrate(world, b, h)

        # Body-on-body and body-under-foot run once a frame at the full `dt`.
        # Neither is a fast collision: two corpses settling into each other
        # move at walking pace, and the thing shoving them is a walking pace by
        # definition. Sub-stepping them would triple the cost of the frame to
        # resolve contacts that were never going to tunnel.
        self._separate(world)
        self._shove(world, dt)
        # Both contact passes move bodies by hand, and either can put one
        # inside the masonry: a heap settling against a wall, or someone
        # standing on a corpse that is already up against one. The wall pass
        # gets the last word, so nothing is ever *drawn* embedded.
        for b in self.bodies:
            self._walls(world, b)
            b.corpse.x, b.corpse.y, b.corpse.angle = b.x, b.y, b.angle

    # -- bookkeeping --------------------------------------------------------
    def _retire(self, world: rj.World):
        """Drop the bodies whose corpses have faded out of the world.

        Rebuilt rather than diffed, because a frame in which one corpse fades
        and another appears leaves the counts equal and the contents different
        -- and a `Ragdoll` holding the only reference to a dead `Corpse` is
        exactly how `id()` gets recycled underneath the lookup table.
        """
        if not self.bodies:
            return
        live = {id(c) for c in world.corpses}
        self.bodies = [b for b in self.bodies if id(b.corpse) in live]
        self._by_corpse = {id(b.corpse): b for b in self.bodies}

    def _departed(self, world: rj.World) -> list:
        """Entities that were here last frame and are not here now.

        `leave_corpse` runs in the same `world.update` that removes the entity,
        so the thing that just died and the body that just appeared are always
        one frame apart -- which is how a corpse gets told how fast it was
        moving and how heavy it was, without the sim having to record either.
        """
        here = {id(e) for e in world.entities}
        gone = [rec for key, rec in self._tracked.items() if key not in here]
        for rec in gone:
            self._tracked.pop(id(rec[0]), None)
        return gone

    def _track(self, world: rj.World, dt: float):
        """Remember where everything alive is, and how fast it got there.

        The velocity is a plain frame difference, which is noisy and exactly
        right: what the shove wants to know is how fast this creature moved
        *this* frame, not the average of a walk it finished half a second ago.
        """
        inv = 1.0 / max(dt, 1e-4)
        for e in world.entities:
            x, y = e.world_pos()
            rec = self._tracked.get(id(e))
            if rec is None:
                self._tracked[id(e)] = [e, x, y, 0.0, 0.0]
            else:
                rec[3], rec[4] = (x - rec[1]) * inv, (y - rec[2]) * inv
                rec[1], rec[2] = x, y
        # Anything owed a blast that never became a corpse -- the toggle is
        # off, or the sim was reset -- has missed its chance. Dropped here,
        # after `_adopt` has had the frame it needs to claim it.
        if self._owed:
            here = {id(e) for e in world.entities}
            self._owed = {k: v for k, v in self._owed.items() if k in here}

    def _adopt(self, world: rj.World, gone: list):
        """Give every untracked corpse a body, and throw it."""
        j = world.juice
        for c in world.corpses:
            if id(c) in self._by_corpse:
                continue
            rec = self._nearest_departure(c, gone)
            mass = 1.0
            vx = vy = spin = 0.0
            if rec is not None:
                e = rec[0]
                mass = max(0.35, getattr(e, "weight", 1.0))
                # Whatever the death animation was doing when it ran out. A
                # burst throws the body; a slump barely moves it.
                vx, vy = rec[3], rec[4]
                owed = self._owed.pop(id(e), None)
                if owed is not None:
                    # Blown up mid-death: the blast was spent on a thing that
                    # was not a ragdoll yet, so it was kept until now.
                    vx += owed[1]
                    vy += owed[2]
                    spin += owed[3]
            # Away from the player, who is standing where the blow came from.
            # `leave_corpse` already picks which way up the body lands on the
            # same reasoning, so this only makes the two agree.
            px, py = world.player.world_pos()
            dx, dy = c.x - px, c.y - py
            d = math.hypot(dx, dy) or 1.0
            launch = j.p("rag_launch") * PX * j.intensity
            vx += dx / d * launch
            vy += dy / d * launch
            spin += (self._rand() - 0.5) * 2.0 * launch * j.p("rag_spin")
            b = Ragdoll(corpse=c, x=c.x, y=c.y, vx=vx, vy=vy, angle=c.angle,
                        spin=spin, radius=TILE * j.p("rag_size"), mass=mass)
            # Seated on whatever it died standing on, so its first step is not
            # a body discovering that the hill it has been on all along is a
            # drop it has yet to fall down.
            rise = self._rise(world)
            b.ground = world.terrain.lift_at(b.x, b.y) * rise if rise > 0.0 else 0.0
            self.bodies.append(b)
            self._by_corpse[id(c)] = b

    @staticmethod
    def _nearest_departure(corpse, gone: list):
        """Match a fresh corpse to the entity it came from, by position.

        Identity would be nicer, but a `Corpse` does not carry one and giving
        it one is an edit to the sim. Two things cannot die on the same tile in
        the same frame, so 'the departure that was closest' is exact whenever
        it matters and harmless when it does not.
        """
        best, best_d = None, TILE * 2.5
        for rec in gone:
            d = math.hypot(rec[1] - corpse.x, rec[2] - corpse.y)
            if d < best_d:
                best, best_d = rec, d
        return best

    # -- the ground ---------------------------------------------------------
    #
    # Three small methods, and between them they are all the corpse physics
    # knows about hills. Everything else in this class was written for a flat
    # room and did not have to change: a body still slides, still bounces off
    # masonry and still gets shoved by whoever walks through it, and the ground
    # under it is a number it consults rather than a case it handles.

    def _rise(self, world: rj.World) -> float:
        """Pixels per level of ground, or zero if the arena is flat.

        Reads the switch off the terrain rather than off the juice registry, so
        a field driven with a world that has no terrain in it -- the software
        bench's, or a test's -- never asks for a parameter that only the GL
        build defines.
        """
        t = world.terrain
        if t is None or not t.enabled:
            return 0.0
        return world.juice.p("hill_rise") * PX

    def _fall(self, world: rj.World, b, h: float) -> float:
        """Keep a thing on the ground, and drop it when the ground goes away.

        The whole of falling off a cliff is the second line: when the floor
        under a body is lower than it was, the difference is not a new position
        but a new *height*, and gravity spends it over the next few frames.
        Written that way, a body that slides over a ledge falls off it, a body
        that is shoved over one falls off it, and a body blown over one falls
        off it, without any of those three knowing that ledges exist.

        Returns the speed it landed at, or zero if it did not land this step.
        """
        rise = self._rise(world)
        ground = world.terrain.lift_at(b.x, b.y) * rise if rise > 0.0 else 0.0
        b.z += b.ground - ground
        b.ground = ground
        if b.z < 0.0:
            # Walked, slid or was shoved *up* a ramp. The ground came to meet
            # it, which is not a bounce.
            b.z, b.vz = 0.0, max(b.vz, 0.0)
        if b.z <= 0.0 and b.vz <= 0.0:
            b.z = 0.0
            return 0.0
        b.vz -= self.GRAVITY * h
        b.z += b.vz * h
        if b.z > 0.0:
            return 0.0
        landed = -b.vz
        b.z, b.vz = 0.0, 0.0
        return landed

    def _slope(self, world: rj.World, b, h: float):
        """Gravity along the ground, which is what makes a slope a slope.

        Only while the thing is actually touching it: something in the air is
        already being pulled the other way by `_fall`, and pushing it sideways
        as well would make a body arc as it fell.
        """
        rise = self._rise(world)
        if rise <= 0.0 or b.z > 1.0:
            return
        gx, gy = world.terrain.downhill(b.x, b.y)
        if gx == 0.0 and gy == 0.0:
            return
        push = world.juice.p("hill_slide") * rise * h
        b.vx += gx * push
        b.vy += gy * push

    # -- integration --------------------------------------------------------
    def _integrate(self, world: rj.World, b: Ragdoll, h: float):
        j = world.juice
        drag = j.p("rag_drag")
        # Exponential rather than linear, so the same slider means the same
        # thing whatever the frame rate and a body never drags itself backwards
        # through zero on a long frame.
        keep = math.exp(-drag * h)
        b.vx *= keep
        b.vy *= keep
        b.spin *= math.exp(-drag * 0.85 * h)
        # After the drag and before the sleep test, so a body on a slope is
        # re-accelerated every step and creeps down it rather than being
        # stopped dead halfway by the speed floor.
        self._slope(world, b, h)

        if b.speed < self.SLEEP_SPEED:
            b.vx = b.vy = 0.0
        else:
            b.x += b.vx * h
            b.y += b.vy * h

        b.angle += b.spin * h
        self._walls(world, b)

        landed = self._fall(world, b, h)
        if landed > 90.0 * PX:
            b.squash_vel += landed * 0.006
            self._thud(world, b, landed)

        # The squash is a spring, not a decay: a body that lands hard flattens,
        # overshoots on the way back and settles. A decay only ever flattens.
        b.squash_vel += (-b.squash * 340.0 - b.squash_vel * 13.0) * h
        b.squash += b.squash_vel * h

        # In the air is not at rest, however still the body looks from above --
        # without this a corpse halfway down a cliff face settles flat and
        # stops taking the tumble the fall was going to give it.
        b.resting = (b.speed <= 0.0 and abs(b.spin) < self.SLEEP_SPIN
                     and b.z <= 0.0)
        if b.resting:
            b.spin = 0.0
            # Settle flat. `leave_corpse`'s convention is that +-90 is a body
            # lying down, so a ragdoll that has stopped rolls the last few
            # degrees into the nearest of them rather than freezing mid-tumble
            # -- a corpse stopped at 20 degrees reads as one still falling.
            flat = round((b.angle - 90.0) / 180.0) * 180.0 + 90.0
            b.angle += (flat - b.angle) * min(1.0, 5.0 * h)

    def _solid(self, world: rj.World, b, tx: int, ty: int) -> bool:
        """Whether this tile stops the thing at `b`, at the height it is at.

        Masonry always. A step *up* in the ground only while the thing is low
        enough to hit it -- which is one condition doing two jobs: a corpse
        cannot slide up a cliff it could not have walked up, and a bomb thrown
        over one sails across instead of bouncing off thin air. A step *down*
        is never solid, because the difference between a cliff and a wall is
        precisely that you can go over the edge of it.
        """
        if world.blocked(tx, ty):
            return True
        rise = self._rise(world)
        if rise <= 0.0:
            return False
        step = world.terrain.step_up(b.x, b.y, tx, ty) * rise
        return step > 2.0 and b.z < step

    def _walls(self, world: rj.World, b: Ragdoll):
        """Push the circle out of every solid tile it overlaps, and bounce it.

        Circle against the tile *rectangle*, not against the tile centre: the
        nearest point on a box is what gives a body sliding along a wall a
        clean tangent, and what stops it catching on the seam between two
        tiles of the same wall.

        "Solid" is `_solid` rather than `world.blocked`, so the same code
        bounces a body off the side of a hill -- masonry and a cliff face are
        the same problem and there is still only one place a circle can be
        wrong about one.
        """
        j = world.juice
        bounce = j.p("rag_bounce")
        spin_gain = j.p("rag_spin")
        r = b.radius
        hardest = 0.0
        tx0, tx1 = int((b.x - r) // TILE), int((b.x + r) // TILE)
        ty0, ty1 = int((b.y - r) // TILE), int((b.y + r) // TILE)
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                if not self._solid(world, b, tx, ty):
                    continue
                left, top = tx * TILE, ty * TILE
                nearest_x = clamp(b.x, left, left + TILE)
                nearest_y = clamp(b.y, top, top + TILE)
                dx, dy = b.x - nearest_x, b.y - nearest_y
                d2 = dx * dx + dy * dy
                if d2 >= r * r:
                    continue
                if d2 > 1e-9:
                    d = math.sqrt(d2)
                    nx, ny, pen = dx / d, dy / d, r - d
                else:
                    # Dead centre inside the tile -- a body that was launched
                    # through a wall by a big enough blast. There is no nearest
                    # point to push away from, so it leaves by the shallowest
                    # face it is behind.
                    nx, ny, pen = self._escape(b, left, top, r)
                b.x += nx * pen
                b.y += ny * pen
                vn = b.vx * nx + b.vy * ny
                if vn >= 0.0:
                    continue                      # already leaving; do not grab it
                tvx, tvy = b.vx - vn * nx, b.vy - vn * ny
                # The tangent keeps most of itself (a body scrapes along a
                # wall), the normal comes back scaled by the restitution, and
                # the part of the tangent the wall ate turns into rotation.
                b.vx = tvx * 0.82 - vn * nx * bounce
                b.vy = tvy * 0.82 - vn * ny * bounce
                tangent = tvx * -ny + tvy * nx
                b.spin += tangent * spin_gain * 0.55
                # Deliberately not scaled by the tumble slider: a body still
                # deforms on a wall with the spin turned all the way down.
                b.squash_vel -= vn * 0.006
                # One noise per step, however many tiles of the same wall the
                # body happens to be touching -- an inside corner is two
                # contacts and one thud.
                hardest = max(hardest, -vn)
        if hardest > 120.0 * PX:
            self._thud(world, b, hardest)

    @staticmethod
    def _escape(b: Ragdoll, left: float, top: float, r: float):
        """Shallowest way out of a tile a body is entirely inside of."""
        outs = ((-1.0, 0.0, b.x - left + r),
                (1.0, 0.0, left + TILE - b.x + r),
                (0.0, -1.0, b.y - top + r),
                (0.0, 1.0, top + TILE - b.y + r))
        return min(outs, key=lambda o: o[2])

    def _thud(self, world: rj.World, b: Ragdoll, speed: float):
        """A hard landing is worth a noise and a puff, at most a few a second."""
        j = world.juice
        if j.on("particles"):
            world.fx.particles.burst(
                b.x, b.y, count=3, speed=(40 * PX, 150 * PX), life=(0.15, 0.32),
                size=3.0 * PX, gravity=700.0 * PX, colors=((90, 96, 116),))
        if speed > 420.0 * PX:
            world.sfx("bump", gain=0.5, wx=b.x)

    # -- contacts -----------------------------------------------------------
    def _separate(self, world: rj.World):
        """Corpse against corpse, so a pile is a pile and not one sprite.

        Naive pairs, which is the right algorithm at forty bodies: a grid or a
        sweep costs more to build each frame than the 800 distance checks it
        saves, and the corpse list is capped at 40 by `leave_corpse` anyway.
        The x-gap test in front of the square root is what actually pays.
        """
        n = len(self.bodies)
        if n < 2:
            return
        bounce = world.juice.p("rag_bounce")
        for i in range(n - 1):
            a = self.bodies[i]
            for k in range(i + 1, n):
                c = self.bodies[k]
                lim = a.radius + c.radius
                dx = c.x - a.x
                if dx > lim or dx < -lim:
                    continue
                dy = c.y - a.y
                if dy > lim or dy < -lim:
                    continue
                d2 = dx * dx + dy * dy
                if d2 >= lim * lim:
                    continue
                if d2 > 1e-9:
                    d = math.sqrt(d2)
                    nx, ny = dx / d, dy / d
                else:
                    # Perfectly stacked: any direction will do, as long as it
                    # is the *same* one every frame or the pair will buzz --
                    # and it must not come from `id()`, which is a different
                    # number every run and would take determinism with it.
                    ang = i * 2.39996                     # the golden angle
                    nx, ny, d = math.cos(ang), math.sin(ang), 0.0
                pen = lim - d
                total = a.mass + c.mass
                a.x -= nx * pen * (c.mass / total)
                a.y -= ny * pen * (c.mass / total)
                c.x += nx * pen * (a.mass / total)
                c.y += ny * pen * (a.mass / total)
                # Approach speed along the contact, shared out by mass. Bodies
                # are soft, so most of it is absorbed rather than returned.
                vn = (c.vx - a.vx) * nx + (c.vy - a.vy) * ny
                if vn >= 0.0:
                    continue
                imp = -vn * (1.0 + bounce * 0.5) / total
                a.vx -= nx * imp * c.mass
                a.vy -= ny * imp * c.mass
                c.vx += nx * imp * a.mass
                c.vy += ny * imp * a.mass
                spin = world.juice.p("rag_spin") * 0.25
                a.spin -= vn * spin
                c.spin += vn * spin

    def _shove(self, world: rj.World, dt: float):
        """The living walk through the dead, and the dead give way.

        The whole argument for taking corpses off the grid is in this method.
        On the grid there are exactly two things walking into a body can mean
        -- blocked, or nothing there -- and both are wrong. Here the body moves
        out of the way at the speed of the thing pushing it, this frame, in the
        middle of a step, with no turn taken and no tile changing hands.
        """
        j = world.juice
        shove = j.p("rag_shove")
        if shove <= 0.0 or not self.bodies:
            return
        for e in world.entities:
            rec = self._tracked.get(id(e))
            if rec is None:
                continue
            ex, ey, evx, evy = rec[1], rec[2], rec[3], rec[4]
            reach = TILE * 0.38
            for b in self.bodies:
                lim = reach + b.radius
                dx, dy = b.x - ex, b.y - ey
                if dx > lim or dx < -lim or dy > lim or dy < -lim:
                    continue
                d2 = dx * dx + dy * dy
                if d2 >= lim * lim:
                    continue
                if d2 > 1e-9:
                    d = math.sqrt(d2)
                    nx, ny = dx / d, dy / d
                else:
                    nx, ny, d = 0.0, 1.0, 0.0
                pen = lim - d
                # The walker is immovable -- it is still a grid creature and
                # its tile is not negotiable -- so the body takes all of the
                # separation.
                b.x += nx * pen
                b.y += ny * pen
                # Speed comes from two places: the walker's own motion along
                # the contact, which is what makes a body skid ahead of a
                # charging ogre, and the overlap itself, which is what makes
                # one squeeze out from under someone standing still on it.
                #
                # Written as a target speed the contact drives the body up to,
                # not as an impulse added every frame: an impulse applied for
                # as long as two things overlap accumulates, and a corpse you
                # lean on quietly builds up enough speed to cross the arena.
                closing = max(0.0, evx * nx + evy * ny)
                target = (closing + pen * 6.0) * shove / b.mass
                along = b.vx * nx + b.vy * ny
                if target > along:
                    gain = (target - along) * min(1.0, 16.0 * dt)
                    b.vx += nx * gain
                    b.vy += ny * gain
                    b.spin += (nx * evy - ny * evx) * j.p("rag_spin") * 0.05
                    b.squash_vel += gain * 0.004

    # -- explosions ---------------------------------------------------------
    def _watch_explosions(self, world: rj.World):
        """Turn every shockwave the sim raises into force on the bodies.

        Watching the effect rather than the cause means every explosion in the
        bench -- a killing blow, a bomb, anything added later -- throws corpses
        without knowing that corpses can be thrown. The ring on screen and the
        force in the physics are then the same event by construction, which is
        the part that would otherwise drift.
        """
        live = world.fx.shockwaves
        if not live:
            self._seen_waves.clear()
            return
        for s in live:
            if id(s) in self._seen_waves:
                continue
            self._seen_waves[id(s)] = s
            self.blast(world, s.x, s.y, s.max_radius * self.BLAST_REACH,
                       self._wave_force(world, s.max_radius))
        if len(self._seen_waves) > len(live):
            ids = {id(s) for s in live}
            self._seen_waves = {k: v for k, v in self._seen_waves.items()
                                if k in ids}

    # -- thrown bombs ---------------------------------------------------------
    def throw(self, world: rj.World, thrower=None):
        """Lob a bomb out of an entity's hand along the way it is facing.

        Thrown rather than placed, because where an explosion happens is the
        interesting decision and dropping one at your feet takes that decision
        away. It leaves on an arc, so it clears a body standing in front of
        you; it bounces off walls, so a corridor can be banked; and it goes off
        on a fuse wherever it has ended up, so a bad throw is a bad throw.
        """
        j = world.juice
        e = thrower if thrower is not None else world.player
        x, y = e.world_pos()
        dx, dy = e.body.facing
        d = math.hypot(dx, dy) or 1.0
        speed = j.p("bomb_throw") * PX
        fuse = j.p("bomb_fuse")
        self.thrown.append(Bomb(
            x=x + dx / d * TILE * 0.35, y=y + dy / d * TILE * 0.35,
            vx=dx / d * speed, vy=dy / d * speed,
            z=TILE * 0.45, vz=speed * 0.42,
            spin=(self._rand() - 0.5) * 900.0,
            fuse=fuse, max_fuse=fuse))
        world.sfx("swipe", gain=0.6, wx=x, pitch=4.0)
        del self.thrown[:-12]

    def _fly(self, world: rj.World, dt: float):
        """Move the bombs, and set off the ones whose fuse has run out.

        Height is integrated separately from the ground plane and drawn as an
        offset, which is the same trick the hop uses: there is no third axis
        here, only a number that lifts a sprite off its own shadow.
        """
        if not self.thrown:
            return
        drag = world.juice.p("rag_drag")
        for bomb in list(self.thrown):
            bomb.fuse -= dt
            # The ground first, so a bomb that has just rolled over a ledge is
            # airborne again by the time the arc below looks at it -- and then
            # bounces when it arrives, which is the same landing it gets off a
            # throw. A bomb thrown *onto* a hill is handled by the same line
            # from the other side: the ground rises, the height is spent.
            rise = self._rise(world)
            ground = world.terrain.lift_at(bomb.x, bomb.y) * rise if rise > 0.0 else 0.0
            bomb.z = max(0.0, bomb.z + bomb.ground - ground)
            bomb.ground = ground
            if bomb.z > 0.0 or bomb.vz > 0.0:
                bomb.vz -= self.GRAVITY * dt
                bomb.z += bomb.vz * dt
                if bomb.z <= 0.0:
                    # Lands, keeps a little of the drop as a hop, and loses a
                    # chunk of its ground speed to the impact.
                    bomb.z = 0.0
                    if bomb.vz < -60.0 * PX:
                        bomb.vz = -bomb.vz * 0.34
                        # Most of the ground speed goes into the floor on the
                        # first touch. A bomb that keeps it skitters away like
                        # a hockey puck and you can never put one where you
                        # meant to -- which is the only thing a throw is for.
                        bomb.vx *= 0.55
                        bomb.vy *= 0.55
                        world.sfx("bump", gain=0.35, wx=bomb.x, pitch=6.0)
                    else:
                        bomb.vz = 0.0
            else:
                # Rolling. Only touches the floor's drag once it is on it, and
                # more of it than a body gets: this thing is a cast-iron ball.
                keep = math.exp(-drag * 1.4 * dt)
                bomb.vx *= keep
                bomb.vy *= keep
                # Which is also why it is the thing on this floor least likely
                # to stay where you put it on a slope.
                self._slope(world, bomb, dt)
            bomb.x += bomb.vx * dt
            bomb.y += bomb.vy * dt
            bomb.angle += bomb.spin * dt
            bomb.spin *= math.exp(-1.6 * dt)
            self._walls(world, bomb)
            if bomb.fuse <= 0.0:
                self.thrown.remove(bomb)
                self.detonate(world, bomb.x, bomb.y)

    def _wave_force(self, world: rj.World, ring_radius: float) -> float:
        """How hard a ring of that size throws things, in pixels a second.

        Measured against a bomb's ring and raised to the fourth, which is
        steeper than any physics would give you and is the point. An ordinary
        killing blow raises a shockwave too, at a little over half a bomb's
        radius, and on any gentler curve every sword stroke fires the body
        across the arena -- leaving the actual explosion with nothing left to
        be. A kill should tumble a corpse a couple of tiles; a bomb should
        clear the room.
        """
        scale = (max(ring_radius, 1.0) / self.BOMB_RADIUS) ** 4
        return world.juice.p("rag_blast") * PX * scale

    def blast(self, world: rj.World, x: float, y: float, radius: float,
              force: float):
        """Throw everything dead within `radius` away from (x, y).

        Anything still playing its death animation is not a ragdoll yet, so its
        share is written down and handed over the moment its body lands --
        otherwise the one case a player is most likely to try, blowing up
        something as it dies, is the one case that does nothing.
        """
        j = world.juice
        spin_gain = j.p("rag_spin")
        for b in self.bodies:
            dx, dy = b.x - x, b.y - y
            d = math.hypot(dx, dy)
            if d > radius:
                continue
            if d < 1e-3:
                a = self._rand() * math.tau
                dx, dy, d = math.cos(a), math.sin(a), 1.0
            fall = (1.0 - d / radius) ** 1.4
            imp = force * fall * j.intensity / b.mass
            b.vx += dx / d * imp
            b.vy += dy / d * imp
            b.spin += (self._rand() - 0.5) * 2.2 * imp * spin_gain
            b.squash_vel += imp * 0.004

        for e in world.entities:
            if not e.dying:
                continue
            ex, ey = e.world_pos()
            dx, dy = ex - x, ey - y
            d = math.hypot(dx, dy)
            if d > radius:
                continue
            fall = (1.0 - d / radius) ** 1.4
            imp = force * fall * j.intensity / max(0.35, getattr(e, "weight", 1.0))
            d = d or 1.0
            owed = self._owed.get(id(e), (e, 0.0, 0.0, 0.0))
            self._owed[id(e)] = (e,
                                 owed[1] + dx / d * imp,
                                 owed[2] + dy / d * imp,
                                 owed[3] + (self._rand() - 0.5) * 2.2 * imp
                                 * spin_gain)

    def detonate(self, world: rj.World, x: float, y: float,
                 radius: float = BOMB_RADIUS):
        """A bomb: the reason the bench has an explosion to test against.

        Nothing here is new -- it is the same shockwave, the same particles,
        the same trauma and the same light that a killing blow already raises,
        fired at a point instead of at a victim. Anything caught in it dies,
        which is what puts bodies in the air rather than merely nudging the
        ones already on the floor.
        """
        j = world.juice
        world.sfx("death", gain=1.0, wx=x, pitch=-7.0)
        world.sfx("pop", gain=0.9, wx=x, pitch=-4.0)
        world.say("BOOM", rj.GOLD)

        for e in list(world.entities):
            if e is world.player or e.invincible or e.dying:
                continue
            ex, ey = e.world_pos()
            if math.hypot(ex - x, ey - y) <= radius * 0.8:
                world.kill(e)

        if j.on("particles"):
            world.fx.particles.burst(
                x, y, count=int(j.p("part_count") * 3.5),
                speed=(180 * PX, j.p("part_speed") * 2.0 * PX),
                life=(0.3, 0.85), size=5.0 * PX, gravity=760.0 * PX,
                colors=((255, 238, 190), (255, 170, 70), (190, 90, 40)))
        if j.on("decals"):
            world.decals.splat(x, y, (70, 58, 46), count=10, spread=34.0 * PX,
                               radius=9.0 * PX, life=j.p("decal_life"))
        if j.on("shockwave"):
            world.fx.shockwave(x, y, max_radius=radius, life=0.45,
                               width=8.0 * PX, color=(255, 226, 170))
            # Claimed before `_watch_explosions` sees it. The force below is
            # the same number the watcher would have worked out from the ring,
            # but it has to be applied whether or not the ring was drawn.
            self._seen_waves[id(world.fx.shockwaves[-1])] = world.fx.shockwaves[-1]
        if j.on("ripple"):
            world.fx.impact(x, y, strength=j.amt(j.p("ripple_str") * 2.0 * PX),
                            life=0.55, speed=j.p("ripple_speed") * PX,
                            wavelength=54.0 * PX)
        if j.on("light"):
            world.lights.append([x, y, 2.6, 0.3, 0.3])
        if j.on("shake"):
            world.trauma.add(j.amt(0.85))
        if j.on("scrflash"):
            world.screen_flash = max(world.screen_flash, j.amt(0.55))

        self.blast(world, x, y, radius * self.BLAST_REACH,
                   self._wave_force(world, radius))


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------


class GLRenderer:
    """Floor, one instanced batch, a bloom pyramid, one composite, one UI blit."""

    #: Sprites packed into the atlas, plus the digits for damage numbers --
    #: which are drawn as instanced quads in the world so they shake with it,
    #: rather than being pasted on flat afterwards.
    DIGITS = "0123456789"

    #: How much the baked floor is brightened to serve as an albedo. See
    #: `_build_floor`: a palette chosen to look right unlit is not one.
    ALBEDO_GAIN = 2.6

    #: Panel redraws per second. See `draw_ui` -- text layout is the most
    #: expensive thing left in the frame and the least in need of 60Hz.
    UI_HZ = 30.0

    def __init__(self, ctx, target, world: rj.World, headless: bool = False):
        self.ctx = ctx
        self.target = target
        self.headless = headless
        self.frame_ms = 0.0
        self.upload_ms = 0.0
        self._ui_drawn = 0.0
        self._ui_mouse = None
        # Set by anything that can change what the panel says -- which is any
        # input at all. Input is rare and a stale panel is the one thing a
        # throttle must never cause, so the flag is cheaper than being clever
        # about which events matter.
        self.ui_dirty = True

        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        # Premultiplied alpha: the fragment shader outputs `rgb * a`, so a
        # bright particle adds light instead of merely covering what is behind
        # it, and a stack of sparks reads as a glow rather than as paint.
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

        # Not rendering concerns, and none of them know anything about GL --
        # but the bench has nowhere else to keep per-window state without
        # editing the sim, and everything that builds a renderer wants all
        # three. The field is stepped by the loop (`run`, `run_headless`) and
        # never from `render`, because a draw must not advance time.
        self.ragdolls = RagdollField()
        self.settings = juicesettings.Settings()
        self.display = Display()
        self.menu = Menu()

        # The hills, composed into the world rather than built into it. The sim
        # asks it one question (`World.step_allowed`) and otherwise does not
        # know it is there; this file asks it how high the ground is under
        # every sprite; the corpse physics asks it which way is downhill. The
        # software bench never gets one, which is why that bench is flat.
        self.terrain = terrainfx.build_hills(world.grid, tile=TILE)
        world.terrain = self.terrain
        self.sync_terrain(world)
        #: The tallest a level may be drawn -- see `rise`. Worked out once,
        #: because `max_level` walks the whole grid and `rise` is asked per
        #: quad: leaving it as a property call cost a millisecond a frame, all
        #: of it on the processor, for an answer that cannot change.
        self.rise_ceiling = glfx.TERRAIN_ROWS * TILE / max(1, self.terrain.max_level)

        self.sheet = tiles.SpriteSheet(TILE)
        self.atlas = self._build_atlas()
        self.digit_uv = {ch: self.atlas.uv[f"digit_{ch}"] for ch in self.DIGITS
                         if f"digit_{ch}" in self.atlas.uv}

        self.quad_prog = ctx.program(vertex_shader=glfx._inject(glfx.QUAD_VS),
                                     fragment_shader=glfx._inject(glfx.QUAD_FS))
        self.floor_prog = ctx.program(vertex_shader=glfx._inject(glfx.FLOOR_VS),
                                      fragment_shader=glfx._inject(glfx.FLOOR_FS))
        self.batch = glfx.QuadBatch(ctx, self.quad_prog)

        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]],
                           dtype="f4")
        self.floor_vbo = ctx.buffer(corners.tobytes())
        self.floor_vao = ctx.vertex_array(
            self.floor_prog, [(self.floor_vbo, "2f", "in_corner")])

        self.post = glfx.PostChain(ctx, (VIEW_W, VIEW_H))

        # The panel and the log, drawn by the software renderer. Constructing it
        # costs a few surfaces we never use; what it buys is that every row,
        # blurb, slider and scroll behaviour is literally the same code.
        self.ui = rj.Renderer()
        self.ui.help_lines = GL_HELP_LINES
        self.ui_surface = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        self.ui_tex = ctx.texture((WIN_W, WIN_H), 4)
        self.ui_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)  # already 1:1
        self.ui_tex.repeat_x = self.ui_tex.repeat_y = False

        self._build_floor(world)
        # `u1` textures sampled through sampler2D are normalized: byte 255 is
        # 1.0. The logical grid uses 0/1, so uploading it directly turns a wall
        # into 1/255 and the shader's > 0.5 occupancy test never sees it.
        wall_bytes = (np.asarray(world.grid, dtype=np.uint8) * 255).tobytes()
        # `f1` is an 8-bit normalized texture, matching the shader's sampler2D.
        # `u1` would create an integer texture and requires usampler2D; pairing
        # it with sampler2D silently yielded zero occupancy on Mesa.
        self.wall_grid = ctx.texture((GRID_W, GRID_H), 1, wall_bytes, dtype="f1")
        self.wall_grid.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.wall_grid.repeat_x = self.wall_grid.repeat_y = False

        # One texel a tile, two channels: the level and the stair code. Nearest
        # filtering because a level is a whole number and an interpolated one
        # is a height nothing stands at -- the ramp across a stair is computed
        # from the code, not blended out of the neighbours.
        self.terrain_tex = ctx.texture((GRID_W, GRID_H), 2, self.terrain.pack(),
                                       dtype="f1")
        self.terrain_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.terrain_tex.repeat_x = self.terrain_tex.repeat_y = False

    # -- setup ------------------------------------------------------------
    def _build_atlas(self):
        """Sprites and digits in one page.

        The digits are here rather than in the UI overlay because damage
        numbers have to live in the *world*: drawn flat on top afterwards they
        would sit perfectly still while the screen shook underneath them, which
        reads worse than having no numbers at all. Putting them in the same
        atlas keeps the whole frame at one draw call.
        """
        font = pygame.font.Font(pygame.font.match_font(
            "dejavusansmono,couriernew,consolas,monospace"), 40)
        font.set_bold(True)
        digits = {f"digit_{ch}": font.render(ch, True, (255, 255, 255))
                  for ch in self.DIGITS}
        return glfx.Atlas(self.ctx, self.sheet, list(tiles.SPRITES),
                          tiles.SPRITES, extra=digits)

    def _build_floor(self, world: rj.World):
        """Bake the arena once, with its tufts, and derive a normal map from it.

        On the CPU the bake exists to avoid repainting five hundred rounded
        rects a frame, and the ripple has to punch holes in it. Here it is
        simply a texture: the ripple and the ambient sway are displacements of
        the coordinate that samples it, so nothing is ever repainted at all --
        and the sway can cover the whole floor instead of a few dozen props,
        because it costs the same either way.
        """
        surf = self.ui._bake_floor(world)
        # Tufts baked in, since the whole floor now sways as one.
        for kind, wx, wy, phase in world.props:
            if kind != "tuft":
                continue
            for i in (-3, 0, 3):
                pygame.draw.line(surf, (44, 58, 50), (wx + i, wy),
                                 (wx + i, wy - 6 - abs(i) * 0.4), 1)

        rgb = np.frombuffer(pygame.image.tobytes(surf, "RGB", False),
                            dtype=np.uint8).reshape(surf.get_height(),
                                                    surf.get_width(), 3)
        # The floor palette was picked to look right with no lighting on it,
        # which makes it far too dark to be an *albedo*: at 28/255 a torch can
        # multiply it by two and it is still black. Under a lighting model the
        # texture stops being the finished pixels and becomes what the surface
        # reflects, so it is scaled up here and the light is what brings it
        # back down. `u_unlit_scale` puts it back when lighting is switched
        # off, so the toggle still compares like with like.
        bright = np.clip(rgb.astype(np.float32) * self.ALBEDO_GAIN, 0, 255)
        self.floor_tex = self.ctx.texture(surf.get_size(), 3,
                                          bright.astype(np.uint8).tobytes())
        self.floor_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.floor_tex.repeat_x = self.floor_tex.repeat_y = False
        normals = glfx.normal_from_luminance(rgb, strength=2.2, blur=1)
        self.floor_normal = self.ctx.texture(surf.get_size(), 3, normals.tobytes())
        self.floor_normal.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.floor_size = surf.get_size()

    # -- elevation ----------------------------------------------------------
    #
    # Three lines of arithmetic, and the reason they are worth naming is that
    # *everything* about verticality here is one of them. There is no third
    # axis: a level of ground is a number of pixels a thing is drawn further up
    # the screen, and the whole of the effect is which things get the offset
    # (bodies, corpses, bombs, the numbers over their heads) and which pointedly
    # do not (shadows, the lights, the sim).

    def sync_terrain(self, world: rj.World):
        """Point the terrain at whatever the panel currently says.

        Called from the step *and* from the draw. `render` must be safe to run
        twice on the same world with no time passing -- the headless path does
        exactly that -- and copying a flag out of the toggle registry is not
        time passing; it is the draw reading the same switch as everything
        else. Leaving it to the step alone would mean a frame rendered without
        one (a test, a screenshot) drawing hills the sim had switched off.
        """
        self.terrain.enabled = world.juice.on("hills")

    def rise(self, world: rj.World) -> float:
        """Pixels of lift per level of ground. Zero when hills are switched off.

        Capped so the tallest ground in the room stays inside the handful of
        tile rows the floor shader searches for a visible surface. Past that
        cap a plateau would be lifted clean over its own cliff face and the
        room would show holes -- and it is a cap rather than an assertion
        because the alternative is a slider with a range that silently depends
        on how many levels somebody drew into `terrain.ARENA`.
        """
        if not self.terrain.enabled:
            return 0.0
        return min(world.juice.p("hill_rise") * PX, self.rise_ceiling)

    def lift(self, world: rj.World, wx: float, wy: float) -> float:
        """How far above the flat plane the ground under a world point is.

        Convenient rather than fast, and used where a handful of calls happen.
        The batch fill asks `lifter` for a closure instead, because it asks
        this question a few hundred times a frame and the two lookups behind
        `rise` are then a few hundred lookups that never change their answer.
        """
        rise = self.rise(world)
        return self.terrain.lift_at(wx, wy) * rise if rise > 0.0 else 0.0

    def lifter(self, world: rj.World):
        """A `lift(x, y)` for this frame, with the constants already read."""
        rise = self.rise(world)
        if rise <= 0.0:
            return lambda wx, wy: 0.0
        lift_at = self.terrain.lift_at
        return lambda wx, wy: lift_at(wx, wy) * rise

    # -- per frame --------------------------------------------------------
    def camera_uniforms(self, world: rj.World):
        """Shake, kick, roll and zoom, applied to the geometry.

        The software bench has to `rotozoom` the finished frame, which resamples
        every pixel and drags the frame's own edges into view. Here they are
        four numbers in the vertex shader: the world is simply drawn from a
        different place, so there is no resampling and no edge to hide.
        """
        j = world.juice
        dx = dy = roll = 0.0
        if j.on("shake"):
            sx, sy, sr = world.trauma.offset(
                max_px=j.p("shake_px") * PX * j.intensity,
                max_deg=2.4 * j.intensity)
            dx, dy, roll = sx, sy, sr
        if j.on("kick"):
            dx += world.kick[0].value
            dy += world.kick[1].value
        if j.on("tilt"):
            roll += world.camera.tilt.value
        zoom = clamp(world.camera.zoom.value if j.on("zoom") else 1.0, 0.85, 1.3)
        return {
            "u_camera": (world.camera.x, world.camera.y),
            "u_half": (VIEW_W * 0.5, VIEW_H * 0.5),
            "u_roll": math.radians(roll),
            "u_zoom": zoom,
            "u_shake": (dx, dy),
        }

    def lights_of(self, world: rj.World):
        """(x, y, radius, strength) for everything giving off light."""
        j = world.juice
        if not j.on("light"):
            return []
        radius = j.p("light_radius") * TILE
        out = [(*world.player.world_pos(), radius, 1.0)]
        flicker = world.elapsed
        for kind, wx, wy, phase in world.props:
            if kind == "torch":
                # A torch that is perfectly steady is a lamp. Two sines at
                # unrelated rates keep it from looking like a pulse.
                f = 0.86 + 0.09 * math.sin(flicker * 7.3 + phase) \
                    + 0.05 * math.sin(flicker * 17.1 + phase * 2.0)
                out.append((wx, wy, radius * 0.85, f))
        for x, y, s, life, maxl in world.lights:
            out.append((x, y, radius * 1.1, s * (life / maxl)))
        return out[:glfx.MAX_LIGHTS]

    def _set_lighting(self, program, world: rj.World, lights):
        j = world.juice
        n = glfx.write_vec4_array(program, "u_lights", lights, glfx.MAX_LIGHTS)
        if "u_light_count" in program:
            program["u_light_count"] = n
        if "u_light_color" in program:
            program["u_light_color"] = (1.0, 0.86, 0.66)
        if "u_ambient" in program:
            a = j.p("light_ambient") if j.on("light") else 1.0
            program["u_ambient"] = (a, a * 1.01, a * 1.08)   # a cool shadow
        if "u_light_height" in program:
            program["u_light_height"] = j.p("light_height")
        if "u_lit" in program:
            program["u_lit"] = 1.0 if j.on("light") else 0.0
        if "u_sharpness" in program:
            program["u_sharpness"] = j.p("sharpness") if j.on("sharp") else 0.0
        if "u_light_contrast" in program:
            program["u_light_contrast"] = j.p("light_contrast")
            program["u_wall_shadow_amount"] = j.p("wall_shadow_amt")
            program["u_npc_shadow_amount"] = j.p("npc_shadow_amt")
        if "u_grid_size" in program:
            program["u_grid_size"] = (GRID_W, GRID_H)
            program["u_grid_extent"] = (GRID_W * TILE, GRID_H * TILE)
            casters = []
            # The player carries the principal light, so only NPC silhouettes
            # cast the deliberately soft dynamic shadows requested here.
            for e in world.entities[1:]:
                if not e.dying:
                    x, y = e.world_pos()
                    casters.append((x, y, TILE * 0.32, TILE * 0.14))
            count = glfx.write_vec4_array(program, "u_shadow_casters", casters,
                                           glfx.MAX_SHADOW_CASTERS)
            program["u_shadow_caster_count"] = count
            self.wall_grid.use(2)
            program["u_wall_grid"] = 2

    def draw_floor(self, world: rj.World, lights):
        j = world.juice
        p = self.floor_prog
        for key, value in self.camera_uniforms(world).items():
            p[key] = value
        p["u_extent"] = (GRID_W * TILE, GRID_H * TILE)
        p["u_floor_size"] = self.floor_size
        p["u_time"] = world.elapsed
        p["u_ambient_sway"] = (j.p("ambient_amt") * j.intensity
                               if j.on("ambient") else 0.0)
        p["u_unlit_scale"] = 1.0 / self.ALBEDO_GAIN

        waves, lens = [], []
        if j.on("ripple"):
            for imp in world.fx.impacts[:glfx.MAX_WAVES]:
                waves.append((imp.x, imp.y, imp.speed * imp.t,
                              imp.strength * (1.0 - imp.t) ** 2))
                lens.append(imp.wavelength)
        n = glfx.write_vec4_array(p, "u_waves", waves, glfx.MAX_WAVES)
        glfx.write_float_array(p, "u_wave_len", lens, glfx.MAX_WAVES)
        p["u_wave_count"] = n

        self._set_lighting(p, world, lights)
        self.floor_tex.use(0)
        self.floor_normal.use(1)
        p["u_floor"] = 0
        p["u_floor_normal"] = 1
        p["u_cliff_shade"] = j.p("hill_shade")
        self._set_terrain(p, world)
        self.floor_vao.render(mode=moderngl.TRIANGLE_STRIP)

    def _set_terrain(self, program, world: rj.World):
        """The elevation uniforms, for either program that was injected with it.

        Both the floor shader and the quad shader carry a copy of the terrain
        code -- one to draw the ground, one to ask whether the ground is in the
        way -- so both want the same three uniforms and there is one place they
        are written.
        """
        if "u_terrain" not in program:
            # A driver is free to strip a uniform whose result never reaches an
            # output, so a program that stops *using* the terrain stops
            # declaring it. Guarded the same way `_set_lighting` guards its
            # own, because the alternative is a KeyError from a shader edit
            # rather than a picture that has quietly lost a feature.
            return
        self.terrain_tex.use(3)
        program["u_terrain"] = 3
        program["u_tile"] = float(TILE)
        # The one uniform that turns the terrain on. At zero the floor shader
        # takes the flat path in a single branch and the quad shader skips its
        # occlusion test outright, so switching hills off costs nothing rather
        # than costing a disabled feature.
        program["u_rise"] = self.rise(world)

    # -- the batch --------------------------------------------------------
    def fill_batch(self, world: rj.World):
        """Everything else, in painter's order, into one instance buffer.

        The `lift=` on a good half of these calls is the terrain, and what is
        interesting about it is the calls that do *not* have one. A shadow
        stays on the ground the body is standing on, so it is lifted with the
        ground and not with the body -- which is the entire reason a creature
        on a plateau reads as being up there rather than as being drawn wrong.
        Particles are left flat on purpose: there are thousands of them, they
        are in the air anyway, and half a level of error on a spark that lives
        a third of a second is not worth a terrain lookup per particle per
        frame. Rings and ripples are flat because they are events on the floor
        plane, and lifting one would mean deciding which of the four terraces
        it crosses it belongs to.
        """
        b = self.batch
        j = world.juice
        lift = self.lifter(world)
        b.clear()
        white = self.atlas.white

        if j.on("decals"):
            for d in world.decals.decals:
                b.add(d.x, d.y, d.radius * 2.4, d.radius * 2.4 * d.squash,
                      uv=white, color=[c / 255.0 for c in d.color],
                      alpha=0.85 * d.alpha, shape=glfx.SHAPE_DISC,
                      params=(0.55, 0, 0, 0), lift=lift(d.x, d.y))

        # Torch flames. The tufts are baked into the floor because the floor
        # itself sways now; a flame has to be drawn because it also flickers.
        for kind, wx, wy, phase in world.props:
            if kind != "torch":
                continue
            t = world.elapsed * 9.0 + phase
            wob = math.sin(t) * 1.2
            if j.on("ambient"):
                ox, oy = ambient_offset(wx, wy, world.elapsed + phase,
                                        j.p("ambient_amt") * j.intensity)
                wob += ox
            flame = lift(wx, wy)
            b.add(wx + wob, wy - 8.0, 9.0, 15.0, uv=white,
                  color=(1.6, 0.8, 0.25), alpha=0.9, shape=glfx.SHAPE_DISC,
                  params=(0.85, 0, 0, 0), lift=flame)
            b.add(wx + wob * 0.6, wy - 9.0, 4.5, 9.0, uv=white,
                  color=(2.4, 2.0, 1.2), alpha=1.0, shape=glfx.SHAPE_DISC,
                  params=(0.7, 0, 0, 0), lift=flame)

        if j.on("shadow"):
            for e in world.entities:
                if e.dying:
                    continue
                ox, size, alpha = shadow_of(e.body, base=j.p("shadow_size"))
                alpha *= j.p("shadow_alpha") * j.intensity
                if alpha <= 0.02:
                    continue
                sx = (e.body.tx + 0.5 + ox) * TILE
                sy = (e.body.ty + 0.5) * TILE + TILE * 0.36
                # Lifted by the *tile* rather than by the body: a shadow
                # belongs to the ground, and the ground does not hop.
                b.add(sx, sy, TILE * size * 1.3, TILE * size * 0.55, uv=white,
                      color=(0.0, 0.0, 0.0), alpha=alpha,
                      shape=glfx.SHAPE_DISC, params=(0.7, 0, 0, 0),
                      lift=lift(*e.tile_center()))
            # A corpse that slides needs a shadow for the same reason a hop
            # does: without one it reads as a decal painted on the floor
            # rather than as an object being moved across it.
            for c in world.corpses:
                a = c.alpha * j.p("shadow_alpha") * j.intensity * 1.1
                if a <= 0.02:
                    continue
                rag = self.ragdolls.of(c)
                b.add(c.x, c.y + TILE * 0.16, TILE * 0.92, TILE * 0.34,
                      uv=white, color=(0.0, 0.0, 0.0), alpha=a,
                      shape=glfx.SHAPE_DISC, params=(0.62, 0, 0, 0),
                      lift=rag.ground if rag is not None
                      else lift(c.x, c.y))

        for c in world.corpses:
            uv = self.atlas.uv.get(c.sprite)
            if uv is None:
                continue
            rag = self.ragdolls.of(c)
            sx, sy = rag.scale if rag is not None else (1.0, 1.0)
            # A tracked body carries its own height, because it may be halfway
            # down a cliff and the ground under it says nothing about that.
            high = (rag.ground + rag.z) if rag is not None else lift(c.x, c.y)
            b.add(c.x, c.y, TILE * sx, TILE * sy, rot=math.radians(c.angle),
                  uv=uv, color=self._tint(c), alpha=c.alpha, lift=high)

        for e in world.entities:
            self._add_trail(b, lift, e)
        for e in world.entities:
            self._add_tail(b, world, lift, e)
            self._add_body(b, world, lift, e)

        self._add_bombs(b, world)

        if j.on("shockwave"):
            for s in world.fx.shockwaves:
                r = s.radius
                if r < 2:
                    continue
                b.add(s.x, s.y, r * 2, r * 2, uv=white,
                      color=[c / 255.0 * 1.6 for c in s.color],
                      alpha=0.85 * s.alpha, shape=glfx.SHAPE_RING,
                      params=(max(0.04, s.thickness / max(r, 1.0)), 0, 0, 0))

        for s in world.fx.slashes:
            reach = s.radius * rj.lerp(0.55, 1.15, s.progress)
            b.add(s.x, s.y, (reach + 8) * 2, (reach + 8) * 2,
                  rot=-math.radians(s.angle), uv=white,
                  color=(1.7, 1.7, 1.7), alpha=0.95 * s.alpha,
                  shape=glfx.SHAPE_ARC,
                  params=(reach / (reach + 8), math.radians(s.sweep),
                          0.10 + 0.25 * (1.0 - s.t), 0.0),
                  lift=lift(s.x, s.y))

        for c in world.fx.cuts:
            b.add(c.x, c.y, c.length, c.thickness * 2.4,
                  rot=-math.radians(c.angle), uv=white,
                  color=[v / 255.0 for v in c.color], alpha=c.alpha,
                  shape=glfx.SHAPE_CUT, params=(c.progress, 0, 0, 0),
                  lift=lift(c.x, c.y))

        for p in world.fx.particles.particles:
            size = p.size * (1.0 - p.t) * 2.2
            if size < 0.6:
                continue
            b.add(p.x, p.y, size, size, uv=white,
                  color=[v / 255.0 * 1.5 for v in p.color],
                  alpha=(1.0 - p.t) ** 0.6, shape=glfx.SHAPE_DISC,
                  params=(0.85, 0, 0, 0))

        if j.on("numbers"):
            self._add_numbers(b, world, lift)

    def _tint(self, e):
        return [c / 255.0 for c in e.color] if e.tint else (1.0, 1.0, 1.0)

    def _add_bombs(self, b, world):
        """A bomb in the air, its shadow on the floor, and a fuse that hurries.

        The shadow is the whole reason a thrown thing reads as thrown: without
        one, a sprite rising up the screen is indistinguishable from a sprite
        moving away, and the throw looks like a slide. It also has to stay on
        the ground while the bomb does not, which is why the height is an
        offset on the draw rather than anything the physics knows about.
        """
        white = self.atlas.white
        for bomb in self.ragdolls.thrown:
            # `air` is how high the bomb is above its own ground and shrinks its
            # shadow; `bomb.ground` is how high that ground is and moves both,
            # which is what keeps a bomb rolling along a plateau looking like it
            # is on the plateau rather than hovering over the room.
            air = clamp(bomb.z / (TILE * 1.2), 0.0, 1.0)
            b.add(bomb.x, bomb.y + TILE * 0.12,
                  TILE * (0.56 - 0.18 * air), TILE * (0.23 - 0.08 * air),
                  uv=white, color=(0.0, 0.0, 0.0),
                  alpha=(0.5 - 0.24 * air) * world.juice.p("shadow_alpha") * 2.0,
                  shape=glfx.SHAPE_DISC, params=(0.62, 0, 0, 0),
                  lift=bomb.ground)
            # The fuse spends its last third flashing, faster as it goes: a
            # constant blink says "a bomb", an accelerating one says "now".
            urgency = bomb.t ** 3
            blink = 0.5 + 0.5 * math.sin(world.elapsed * (18.0 + 60.0 * urgency))
            hot = 0.25 + 0.75 * urgency * blink
            # A bomb in the air is drawn well above the spot it is over, so the
            # foot of its quad is no guide to whether a hill is in front of it.
            # It stands where its shadow is.
            stands = bomb.y + TILE * 0.26
            b.add(bomb.x, bomb.y - bomb.z, TILE * 0.52, TILE * 0.52,
                  rot=math.radians(bomb.angle), uv=white,
                  color=(0.16 + hot * 1.9, 0.15 + hot * 0.7, 0.17 + hot * 0.2),
                  alpha=1.0, shape=glfx.SHAPE_DISC, params=(0.8, 0, 0, 0),
                  lift=bomb.ground, depth=stands)
            b.add(bomb.x + math.cos(math.radians(bomb.angle)) * TILE * 0.2,
                  bomb.y - bomb.z - math.sin(math.radians(bomb.angle)) * TILE * 0.2,
                  TILE * 0.16, TILE * 0.16, uv=white,
                  color=(2.6 * (0.4 + hot), 1.9 * (0.3 + hot), 0.7),
                  alpha=0.95, shape=glfx.SHAPE_DISC, params=(0.5, 0, 0, 0),
                  lift=bomb.ground, depth=stands)

    def _add_trail(self, b, lift, e):
        uv = self.atlas.uv.get(e.sprite)
        if uv is None:
            return
        for g in e.trail.ghosts:
            # Each ghost takes the height of the ground it was left on, so an
            # afterimage of a run up a staircase climbs the staircase.
            b.add(g.x, g.y, TILE * g.sx, TILE * g.sy, rot=-math.radians(g.angle),
                  uv=uv, color=self._tint(e), alpha=0.42 * g.alpha,
                  params=(1.0, 0, 0, 0),       # unlit: an echo is not a body
                  lift=lift(g.x, g.y))

    def _add_tail(self, b, world, lift, e):
        if e.tail is None or not world.juice.on("tail"):
            return
        pts = e.tail.points(e.body)
        white = self.atlas.white
        col = [c / 255.0 for c in e.tail_color]
        # One sample for the whole chain, at the body it hangs off. A cape is
        # attached to a creature and not to the floor, so a tail trailing back
        # over a ledge should stay level with its owner rather than pour down
        # the cliff behind them.
        high = lift(*e.world_pos())
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            x0, y0, x1, y1 = x0 * TILE, y0 * TILE, x1 * TILE, y1 * TILE
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length < 0.5:
                continue
            t = i / max(1, len(pts) - 2)
            # Overlapping hard-edged capsules rather than soft blobs: a soft
            # ellipse per link draws as a row of dots, which reads as a string
            # of beads instead of a tail.
            b.add((x0 + x1) * 0.5, (y0 + y1) * 0.5, length * 2.0,
                  max(2.0, (1.0 - t) ** 0.8 * TILE * 0.42),
                  rot=math.atan2(dy, dx), uv=white, color=col,
                  alpha=1.0, shape=glfx.SHAPE_DISC, params=(0.35, 0, 0, 0),
                  lift=high)

    def _add_body(self, b, world, lift, e):
        uv = self.atlas.uv.get(e.sprite)
        if uv is None:
            return
        body = e.body
        j = world.juice
        wx, wy = e.world_pos()

        # The soft body. `Jelly` hands the CPU renderer ten band offsets; the
        # same numbers describe a field, so what the shader wants is the *worst*
        # of them -- how far the trailing edge is behind -- and the axis it lies
        # along. Everything between the two edges is then interpolated by the
        # gradient rather than quantised into strips.
        lag = (0.0, 0.0)
        if body.shear and j.on("softbody") and j.on("jelly"):
            head, tail = body.shear[-1], body.shear[0]
            fx, fy = body.facing
            if abs(fx) >= abs(fy) and fx < 0:
                head, tail = tail, head
            elif abs(fy) > abs(fx) and fy < 0:
                head, tail = tail, head
            span = (tail - head) * TILE
            if body.shear_axis == 0:
                lag = (span, 0.0)
            else:
                lag = (0.0, span)
        elif body.shear and j.on("jelly"):
            # The honest fallback: a rigid drag, which is all a blit can do
            # without cutting the sprite up.
            off = sum(body.shear) / len(body.shear) * TILE
            lag = (off, 0.0) if body.shear_axis == 0 else (0.0, off)
            wx += lag[0] * 0.5
            wy += lag[1] * 0.5
            lag = (0.0, 0.0)

        # Mirroring is a swap of the *u* bounds only. Swapping the whole pair
        # flips both axes and stands the creature on its head, which is the one
        # bug this line will ever have.
        (u0, v0), (u1, v1) = uv
        cell = ((u1, v0), (u0, v1)) if e.flip else uv
        # Sampled at the drawn position rather than at the logical tile, so the
        # rise happens across the step animation: walking up a stair, the body
        # is lifted a fraction of a level per frame by the ramp it is standing
        # on, and the climb is the tween the sim already had.
        b.add(wx, wy, TILE * body.sx, TILE * body.sy,
              rot=-math.radians(body.angle), uv=cell,
              color=self._tint(e), alpha=body.alpha, flash=body.flash,
              lag=lag, params=(0.0, 0.0 if j.on("normals") else 1.0, 0, 0),
              lift=lift(wx, wy))

    def _add_numbers(self, b, world, lift):
        """Damage numbers as instanced digit quads, so they live in the world.

        Drawn into the UI overlay instead, they would hold perfectly still
        while the screen shook underneath them, which reads worse than having
        no numbers at all.
        """
        if not self.digit_uv:
            return
        #: Damage numbers are the one thing in the batch that is not *in* the
        #: room -- they are a reading, drawn in world space only so they shake
        #: with it. A hill hiding the number that says how hard you just hit
        #: something would be the terrain eating a piece of the interface, so
        #: they stand further forward than any ground in the arena can.
        in_front = (GRID_H + 1) * TILE
        for f in world.fx.floaters.floaters:
            size = TILE * 0.8 * f.scale
            text = f.text
            x = f.x - (len(text) - 1) * size * 0.28
            high = lift(f.x, f.y)
            for ch in text:
                uv = self.digit_uv.get(ch)
                if uv is not None:
                    b.add(x, f.y, size, size, uv=uv,
                          color=[c / 255.0 * 1.35 for c in f.color],
                          alpha=clamp(f.alpha), params=(1.0, 0, 0, 0),
                          lift=high, depth=in_front)
                x += size * 0.56

    # -- resolution ---------------------------------------------------------
    def resize(self, world: rj.World, width: int, height: int):
        """Re-lay the bench out at a new window size, on the same GL context.

        Everything that is sized in pixels is rebuilt: the scene and bloom
        buffers, the panel surface and its texture, and the software renderer
        whose own scratch surfaces are view-sized. The context, the atlas and
        the baked floor survive, because none of them ever knew how big the
        window was.

        `VIEW_W` and friends are module globals read at call time in both
        `rogue_juice.py` and here, which is what makes this possible at all: no
        function captured the old size, so setting the globals is the layout
        change and the rebuild below is only the buffers catching up.
        """
        width = max(MIN_VIEW[0] + rj.PANEL_W, int(width))
        height = max(MIN_VIEW[1], int(height))
        if (width, height) == (WIN_W, WIN_H):
            return False
        _set_layout(width, height)

        self.post.release()
        self.post = glfx.PostChain(self.ctx, (VIEW_W, VIEW_H))
        self.ui_tex.release()
        self.ui_surface = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        self.ui_tex = self.ctx.texture((WIN_W, WIN_H), 4)
        self.ui_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.ui_tex.repeat_x = self.ui_tex.repeat_y = False
        # Its buffers, not the whole renderer: the GL bench only ever asks it
        # for the panel and the HUD, neither of which touches a view-sized
        # surface, but leaving stale ones behind is a trap for whoever next
        # asks it for something else -- and rebuilding it outright would reload
        # five fonts on every frame of a window drag.
        self.ui.resize()
        self.ui_dirty = True
        world.clamp_camera()
        return True

    # -- UI ---------------------------------------------------------------
    def draw_ui(self, world: rj.World, mouse, force: bool = False):
        """Redraw the panel and the log, and upload them.

        Throttled, because this is the most expensive thing left in the frame
        by a wide margin: laying out ninety rows of text in pygame and encoding
        a 1180x600 surface costs about 1.2ms, against roughly 1ms for
        *everything else in the renderer put together*. A panel does not need
        sixty updates a second -- nothing on it moves faster than the eye can
        read -- so it gets thirty, and the mouse forces one whenever it moves so
        hovering and dragging stay immediate.
        """
        now = time.perf_counter()
        moved = mouse != self._ui_mouse
        # The menu is a paused state: nothing is competing for the frame, and a
        # throttled highlight under a moving cursor is the one place the 30Hz
        # panel would be felt.
        if not (force or moved or self.ui_dirty or self.menu.open
                or now - self._ui_drawn >= 1.0 / self.UI_HZ):
            return
        elapsed = min(0.25, now - self._ui_drawn)
        self._ui_drawn = now
        self.ui_dirty = False
        self._ui_mouse = mouse
        self.ui_surface.fill((0, 0, 0, 0))
        self.ui.frame_ms = self.frame_ms
        self.ui.draw_hud(self.ui_surface, world)
        self.ui.draw_panel(self.ui_surface, world, mouse)
        if self.menu.open:
            self.menu.draw(self.ui_surface, self.ui, self.display, elapsed)
        self.ui_tex.write(pygame.image.tobytes(self.ui_surface, "RGBA", True))
        self.upload_ms += ((time.perf_counter() - now) * 1000.0 - self.upload_ms) * 0.08

    # -- the frame --------------------------------------------------------
    def step(self, world: rj.World, dt: float):
        """Advance everything the renderer owns that is not drawing.

        Called after `world.update` and before `render`, which is the order the
        two things here need: the corpse field adopts the bodies the sim has
        just laid down, and the terrain switch has to agree with the panel
        before either the physics or the picture reads it.
        """
        self.sync_terrain(world)
        self.ragdolls.update(world, dt)

    def render(self, world: rj.World, mouse=(0, 0)):
        j = world.juice
        self.sync_terrain(world)
        lights = self.lights_of(world)

        self.post.scene.use()
        self.ctx.clear(*[c / 255.0 for c in rj.BG], 1.0)
        self.draw_floor(world, lights)

        p = self.quad_prog
        for key, value in self.camera_uniforms(world).items():
            p[key] = value
        p["u_atlas_size"] = self.atlas.size
        p["u_soft_power"] = j.p("soft_power")
        p["u_soft_pinch"] = j.p("jelly_stretch") * j.intensity
        self._set_lighting(p, world, lights)
        self._set_terrain(p, world)
        self.atlas.albedo.use(0)
        self.atlas.normal.use(1)
        p["u_albedo"] = 0
        p["u_normal"] = 1
        self.fill_batch(world)
        self.batch.render()

        bloom = j.p("bloom_amt") * j.intensity if j.on("bloom") else 0.0
        if bloom > 0.001:
            self.post.build_bloom(j.p("bloom_thresh") / 255.0, j.p("bloom_knee"))

        self.draw_ui(world, mouse)

        haze = []
        if j.on("refract") or j.on("haze"):
            cam = self.camera_uniforms(world)
            for kind, wx, wy, phase in world.props:
                if kind != "torch":
                    continue
                sx = (wx - cam["u_camera"][0]) * cam["u_zoom"] + VIEW_W * 0.5
                sy = (wy - cam["u_camera"][1]) * cam["u_zoom"] + VIEW_H * 0.5
                if -200 < sx < VIEW_W + 200 and -200 < sy < VIEW_H + 200:
                    haze.append((sx, sy, TILE * 5.0, 1.0))
        n = glfx.write_vec4_array(self.post.comp_prog, "u_haze", haze,
                                  glfx.MAX_LIGHTS)

        self.post.composite(self.target, self.ui_tex, {
            "u_bloom_amount": bloom,
            "u_rgb_split": world.rgb_split if j.on("rgbsplit") else 0.0,
            "u_texel": (1.0 / VIEW_W, 1.0 / VIEW_H),
            "u_flash": clamp(world.screen_flash) if j.on("scrflash") else 0.0,
            "u_flash_color": (1.0, 0.96, 0.88),
            "u_vignette": j.p("vig_base") / 200.0 if j.on("vignette") else 0.0,
            "u_vignette_pulse": (clamp(world.vignette_pulse) * 0.7
                                 if j.on("vignette") else 0.0),
            "u_scanlines": j.p("scanline_amt") if j.on("scanlines") else 0.0,
            "u_grade": (clamp(world.screen_flash * 4.0) * j.p("grade_amt")
                        if j.on("grade") else 0.0),
            "u_time": world.elapsed,
            "u_haze_count": n,
            "u_haze_amount": (j.p("haze_amt") * j.intensity
                              if (j.on("refract") or j.on("haze")) else 0.0),
            "u_view": (float(VIEW_W), float(VIEW_H)),
        })

    def release(self):
        self.batch.release()
        self.post.release()
        self.atlas.release()
        self.floor_tex.release()
        self.floor_normal.release()
        self.wall_grid.release()
        self.terrain_tex.release()
        self.floor_vao.release()
        self.floor_vbo.release()
        self.ui_tex.release()
        self.quad_prog.release()
        self.floor_prog.release()


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def menu_action(action: str, world: rj.World, renderer: GLRenderer) -> bool:
    """Carry out one thing the menu asked for. False means quit.

    Split out from the event handler because the actions are the interesting
    part and the key mapping is not: everything here can be driven by name
    from a test, with no window, no click and no keyboard.
    """
    menu, display, settings = renderer.menu, renderer.display, renderer.settings
    key, _, step = action.partition(":")
    delta = -1 if step == "-1" else 1

    if key == "resume":
        menu.close()
    elif key == "back":
        menu.back()
    elif key == "options":
        menu.show("options")
    elif key == "quit":
        return False
    elif key == "save":
        settings.capture_juice(world.juice)
        display.store(settings)
        settings.save()
        menu.say(settings.note)
        world.say(settings.note, rj.ACCENT)
    elif key == "resolution":
        sizes = resolution_choices()
        now = display.clamped()
        try:
            i = sizes.index(now)
        except ValueError:
            i = 0
        width, height = sizes[(i + delta) % len(sizes)]
        display.width, display.height = width, height
        _apply_display(world, renderer, resized=True)
    elif key == "fullscreen":
        display.fullscreen = not display.fullscreen
        _apply_display(world, renderer, resized=True)
    elif key == "frame_cap":
        caps = FRAME_CAPS
        i = caps.index(display.frame_cap) if display.frame_cap in caps else 1
        display.frame_cap = caps[(i + delta) % len(caps)]
    elif key == "vsync":
        display.vsync = not display.vsync
        menu.say("vsync applies on the next restart")
    renderer.ui_dirty = True
    return True


def _apply_display(world: rj.World, renderer: GLRenderer, resized: bool):
    """Push a `Display` at the real window, then re-lay the bench out.

    The window is asked first and *measured* afterwards rather than assumed: a
    window manager is free to refuse a size, and going fullscreen gives you the
    desktop's size, not the one in the settings.
    """
    if not resized:
        return
    if not apply_window(renderer.display):
        renderer.menu.say("this build cannot resize the window -- saved anyway")
        return
    try:
        width, height = pygame.display.get_window_size()
    except Exception:                                # pragma: no cover
        width, height = renderer.display.clamped()
    renderer.resize(world, width, height)


def handle_event(event, world: rj.World, renderer: GLRenderer) -> bool:
    """The software handler, with the panel pointed at the UI renderer.

    The panel is the software renderer's, so its own hit testing, scrolling and
    drag handling apply unchanged -- the only difference is that the surface it
    drew onto is now a texture.

    The menu, when it is up, gets everything: a modal that lets keys through to
    the thing behind it is how you end up walking into a wall while picking a
    resolution.
    """
    renderer.ui_dirty = True                     # any input can change the panel
    if event.type == pygame.QUIT:
        return False

    if renderer.menu.open:
        if event.type == pygame.KEYDOWN:
            action = renderer.menu.key(event, renderer.display)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
            action = renderer.menu.click(event.pos, event.button,
                                         renderer.display)
        elif event.type == pygame.MOUSEMOTION:
            renderer.menu.point(event.pos, renderer.display)
            return True
        else:
            return True
        return menu_action(action, world, renderer) if action else True

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
            renderer.menu.show()                 # escape opens it, not quits
            return True
        if event.key == pygame.K_p:
            return True                          # pixel mode has no meaning here
        if event.key == pygame.K_b:
            renderer.ragdolls.throw(world)
            return True
        if event.key == pygame.K_F3:
            # "Back to defaults" has to mean the saved ones once there are
            # saved ones, or the menu's save button is a button that undoes
            # itself the next time anybody presses F3.
            renderer.settings.restore_defaults(world.juice)
            world.say("sliders back to defaults", rj.ACCENT)
            return True
        if event.key == pygame.K_r:
            # The sim's reset empties `world.corpses`; the field has to let go
            # of the bodies that were pointing at them in the same breath.
            renderer.ragdolls.clear()
    if event.type == pygame.VIDEORESIZE:         # pragma: no cover -- user drag
        renderer.resize(world, event.w, event.h)
        renderer.display.width, renderer.display.height = event.w, event.h
        return True
    return rj.handle_event(event, world, renderer.ui)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def build(headless: bool = False, audio=True):
    """Context, world and renderer, in the order that has to happen."""
    bank = None
    if audio and HAVE_AUDIO and not headless:
        bank = audiofx.SoundBank()
        bank.init_mixer()
        bank.load_all()

    pygame.init()
    pygame.font.init()
    if headless:
        pygame.display.set_mode((WIN_W, WIN_H))
        ctx = glfx.create_context(headless=True)
        target = ctx.simple_framebuffer((WIN_W, WIN_H), components=4)
    else:
        glfx.open_gl_window((VIEW_W, VIEW_H))
        ctx = glfx.create_context()
        target = ctx.screen
        target.viewport = (0, 0, VIEW_W, VIEW_H)

    world = rj.World(gl_juice(), bank)
    renderer = GLRenderer(ctx, target, world, headless=headless)
    return ctx, target, world, renderer


def run():
    # The window is the *view*, and the panel is drawn into it as an overlay,
    # so unlike the software bench there is no second region to size: the GL
    # surface is the whole window.
    bank = None
    renderer = None
    running = True

    # SIGINT normally becomes KeyboardInterrupt, but treating both common
    # termination signals as a request to leave the loop gives terminals,
    # launchers and IDE stop buttons the same orderly shutdown as Escape.
    def request_exit(_signum, _frame):
        nonlocal running
        running = False

    old_sigint = signal.signal(signal.SIGINT, request_exit)
    old_sigterm = signal.signal(signal.SIGTERM, request_exit)
    try:
        if HAVE_AUDIO:
            bank = audiofx.SoundBank()
            bank.init_mixer()
            bank.load_all()

        pygame.init()
        pygame.font.init()
        pygame.joystick.init()

        # Read before the window exists, because two of the things in it --
        # the size and the swap interval -- can only be chosen while it is
        # being made. The juice half is applied further down, once there is a
        # registry to apply it to.
        settings = juicesettings.Settings()
        settings.load()
        display = Display().load(settings)
        _set_layout(*display.clamped())
        glfx.open_gl_window((WIN_W, WIN_H), vsync=display.vsync, resizable=True)
        ctx = glfx.create_context()
        glfx.restart_on_the_card(ctx.info["GL_RENDERER"])
        # A window manager may have had opinions. Believe the window.
        _set_layout(*pygame.display.get_window_size())

        target = ctx.screen
        target.viewport = (0, 0, WIN_W, WIN_H)
        juice = gl_juice()
        settings.apply_juice(juice)
        world = rj.World(juice, bank)
        if pygame.joystick.get_count():           # pragma: no cover
            world.pad = pygame.joystick.Joystick(0)
            world.pad.init()
        renderer = GLRenderer(ctx, target, world)
        renderer.settings = settings
        renderer.display = display
        if display.fullscreen:
            _apply_display(world, renderer, resized=True)
        print(f"rendering on {ctx.info['GL_RENDERER']}")
        world.say(settings.note, rj.DIM)
        world.say("K kills, B throws a bomb, esc opens the menu", rj.ACCENT)

        pygame.key.set_repeat(180, 70)
        clock = pygame.time.Clock()
        while running:
            # A frame cap of zero means "do not wait", which is what makes the
            # ms readout worth reading.
            raw = clock.tick(display.frame_cap) if display.frame_cap else clock.tick()
            dt = min(raw / 1000.0, 1.0 / 20.0)
            for event in pygame.event.get():
                if not handle_event(event, world, renderer):
                    running = False
                    break
            # Do not submit another expensive frame after quit/window-close.
            if not running:
                break

            t0 = time.perf_counter()
            if not renderer.menu.open:
                # The menu pauses. A settings screen that lets the fight carry
                # on behind it is a settings screen you cannot use.
                world.update(dt)
                # After the sim, because it adopts the corpses the sim just
                # laid down; before the render, which reads what it wrote.
                renderer.step(world, dt)
            ctx.screen.viewport = (0, 0, WIN_W, WIN_H)
            renderer.render(world, pygame.mouse.get_pos())
            ms = (time.perf_counter() - t0) * 1000.0
            renderer.frame_ms += (ms - renderer.frame_ms) * 0.08
            pygame.display.flip()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if renderer is not None:
            renderer.release()
        if bank is not None:
            bank.stop_all()
        pygame.quit()


def run_headless(path: str):
    """The scripted swing, rendered with no window and read back to a PNG.

    Same regression value as the software bench's headless path: it drives the
    real shaders through the real batch, so a broken uniform shows up here
    rather than the first time somebody opens the window.
    """
    pygame.init()
    pygame.font.init()
    # `rj.Renderer` calls `Surface.convert()`, which needs a display to convert
    # *to*. Under the dummy video driver this opens nothing and costs nothing;
    # the GL context below is a standalone EGL one and never touches it.
    pygame.display.set_mode((WIN_W, WIN_H))
    ctx = glfx.create_context(headless=True)
    target = ctx.simple_framebuffer((WIN_W, WIN_H), components=4)
    juice = gl_juice()
    juice.set_all(True)
    world = rj.World(juice, None)
    renderer = GLRenderer(ctx, target, world, headless=True)
    dt = 1.0 / FPS

    def step(n):
        for _ in range(n):
            world.update(dt)
            renderer.step(world, dt)

    px, py = world.player.tile
    tx, ty = rj.DUMMY_POS
    while (px, py) != (tx + 1, ty):
        dx = (tx + 1 > px) - (tx + 1 < px)
        dy = 0 if dx else (ty > py) - (ty < py)
        if not dx and not dy:
            break
        world.try_move(world.player, dx, dy)
        step(14)
        if world.player.tile == (px, py):
            break
        px, py = world.player.tile
    world.try_move(world.player, -1, 0)
    step(int((juice.p("windup_time") + juice.p("attack_time") * 0.32) / dt) + 5)

    target.viewport = (0, 0, WIN_W, WIN_H)
    renderer.render(world, (WIN_W - 200, 300))
    image = pygame.image.frombytes(
        bytes(target.read(components=3)), (WIN_W, WIN_H), "RGB", True)
    pygame.image.save(image, path)
    print(f"wrote {path} on {ctx.info['GL_RENDERER']} "
          f"({renderer.batch.count} quads in one draw call, "
          f"{len(world.fx)} live effects)")
    renderer.release()
    pygame.quit()


def main():
    if "--headless" in sys.argv:
        i = sys.argv.index("--headless")
        path = sys.argv[i + 1] if len(sys.argv) > i + 1 else "juice_gl_headless.png"
        run_headless(path)
    else:
        run()


if __name__ == "__main__":
    main()
