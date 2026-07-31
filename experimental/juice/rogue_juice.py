"""A juice workbench: every trick that makes a tile-based roguelike feel good,
one toggle at a time.

A grid roguelike is the hardest kind of game to make feel physical. Everything
happens on integer coordinates, one discrete step at a time, with no momentum
and no sub-tile positions -- the sim is a spreadsheet that occasionally changes
a cell. "Juice" is the layer of lies you tell on top of that spreadsheet so it
reads as bodies moving through a room.

This is a bench for looking at those lies individually. There is a small room,
a player and four training dummies, and a panel down the right listing every
effect the bench knows. Click any of them to turn it off and on. The point of
the bench is not the demo -- it is the *comparison*: turn hit-stop off and hit
a dummy, turn it back on and hit it again, and the difference is much larger
than reading about it suggests. Tab is a master switch for exactly that, so
you can flip the whole thing between "raw sim" and "juiced" mid-swing.

The effects, grouped as they are in the panel:

    movement   hop arcs, squash and stretch, idle breathing, afterimage
               trails, landing dust, and a bounce when you walk into a wall
    attack     anticipation, the lunge, hit-stop, hit flash, knockback,
               a shiver on the victim, a weapon arc, and a death animation
    camera     trauma-based screen shake, a directional kick, a zoom punch,
               a roll tilt, and a follow camera that lags
    world      particles, floating damage numbers, shockwave rings, a wave
               that runs through the floor tiles, a screen flash, a vignette
               pulse and an RGB split

None of the maths lives here. It is all in `juicefx.py`, which has no pygame in
it at all and can be tested with no display -- `juicetest.py` does exactly
that. This file is only the tileset, the input, the panel and the compositing,
which is the split that matters: an effect you cannot test without opening a
window is an effect you will never tune.

Run it with `python3 rogue_juice.py`. `python3 rogue_juice.py --headless
out.png` plays a scripted swing and saves the frame at the moment of impact,
with no window opened anywhere.
"""

from __future__ import annotations

import math
import os
import sys
import time

# The video driver has to be chosen before pygame brings the display up, so
# the headless flag is read here rather than in main(). Without this a
# headless run still tries to open a window, which on WSL means either a stray
# window on the desktop or a hang against a stale WSLg session.
if "--headless" in sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from juicefx import (  # noqa: E402
    ATTACK_TIME, ATTACK_WINDUP, EASINGS, FLASH_TIME, HITSTOP_HEAVY,
    HITSTOP_LIGHT, MOVE_TIME, TRAUMA_HEAVY, TRAUMA_LIGHT,
    Animator, Anticipate, Body, Breathe, Callback, Camera, DeathSpin,
    EffectField, Flash, GhostTrail, HitStop, Hop, Knockback, Lunge, Parallel,
    Sequence, Shiver, Slide, Spring, Trauma, Wait, clamp, lerp,
)

# ---------------------------------------------------------------------------
# Layout and palette
# ---------------------------------------------------------------------------

TILE = 34
# The map is deliberately a little larger than the viewport in both axes, so
# the camera has somewhere to move -- a camera that cannot pan cannot lag, and
# camera lag is one of the effects on the list.
GRID_W, GRID_H = 24, 18
VIEW_W, VIEW_H = 800, 600
# The toggles run in two columns, which is what keeps the window short enough
# to fit under a taskbar -- a single column of 28 rows needs 640px of panel on
# its own, before the header and the blurb.
PANEL_W = 380
WIN_W, WIN_H = VIEW_W + PANEL_W, VIEW_H     # 1180x600
# Panel space reserved for the hover blurb / control list. Sized to swallow
# the slack under the easing gallery rather than leave it as a dead band, so
# the blurb gets room to wrap and the controls get room to be spelled out.
FOOTER_H = 164

#: The control list, shown whenever the mouse is not over a toggle.
HELP_LINES = [
    "click a row to toggle it",
    "",
    "wasd / arrows   move, and attack by",
    "                walking into a dummy",
    "space  swing at air     x  get hit",
    "K      kill the nearest r  reset",
    "",
    "Tab  A/B the whole lot   F1/F2  all on/off",
    "[ ]  move easing         - =  intensity",
    "esc  quit",
]
FPS = 60

#: Pixel-space scale factor. Distances on a Body are in tiles and so survive a
#: change of tile size on their own, but particle speeds, shake amplitudes and
#: shockwave radii are in pixels and do not. Everything in pixels is written as
#: a multiple of this, so the bench feels identical whatever TILE is set to.
PX = TILE / 40.0

BG = (14, 15, 20)
FLOOR = (38, 42, 55)
FLOOR_ALT = (30, 33, 44)
WALL = (58, 62, 80)
WALL_TOP = (78, 84, 108)
PANEL_BG = (20, 21, 28)
PANEL_LINE = (44, 47, 60)
INK = (208, 214, 230)
DIM = (110, 118, 140)
ACCENT = (120, 220, 190)
WARN = (240, 130, 110)
GOLD = (250, 205, 120)

PLAYER_COLOR = (225, 235, 250)
DUMMY_COLORS = [(210, 130, 120), (150, 190, 130), (170, 150, 220), (220, 190, 120)]


# ---------------------------------------------------------------------------
# The toggle registry
# ---------------------------------------------------------------------------


class Toggle:
    """One switchable effect: a flag, a label, and a note on what it buys."""

    __slots__ = ("key", "label", "group", "blurb", "on")

    def __init__(self, key: str, label: str, group: str, blurb: str, on: bool = True):
        self.key = key
        self.label = label
        self.group = group
        self.blurb = blurb
        self.on = on


#: Order here is the order they appear in the panel.
TOGGLE_SPECS = [
    # key          label               group       blurb
    ("tween", "move tween", "movement",
     "Animate the step at all. Off, the sprite teleports the instant the sim "
     "moves it -- which is what the game actually does underneath."),
    ("hop", "hop arc", "movement",
     "Lift the step off the floor. Needs the move tween. A slide reads as a "
     "chess piece, an arc reads as a body."),
    ("squash", "squash & stretch", "movement",
     "Stretch along the direction of travel, squash on landing. Conserve volume or it looks broken."),
    ("bob", "idle breathing", "movement",
     "A slow scale wobble so a standing figure is not a dead pixel. Nobody notices it until it is gone."),
    ("ghost", "afterimage trail", "movement",
     "Faded echoes of the last few frames. Sells speed on a fast move, noise on a slow one."),
    ("dust", "landing dust", "movement",
     "A puff of floor at the end of a hop. Tells you the feet touched something."),
    ("wallbump", "wall bump", "movement",
     "A failed move still needs an answer. Bounce off and shiver, no damage."),

    ("windup", "anticipation", "attack",
     "Lean away before striking. Costs 90ms and does more for weight than any particle."),
    ("lunge", "lunge", "attack",
     "Drive at the target and ease back. Fast out, slow home -- the asymmetry is the effect."),
    ("hitstop", "hit-stop", "attack",
     "Freeze everything for 3-7 frames on contact. The strongest effect here per line of code."),
    ("hitflash", "hit flash", "attack",
     "Whiten the victim. Often the only hit confirmation that is legible at tile size."),
    ("knockback", "knockback", "attack",
     "Shove the victim and let it ease home. Without it the attacker punches a poster."),
    ("shiver", "victim shiver", "attack",
     "Jitter the body it landed on, decaying fast. Reads as the blow ringing through."),
    ("slash", "weapon arc", "attack",
     "A six-frame sweep across the target tile. The eye fills in a sword that is never drawn."),
    ("death", "death animation", "attack",
     "Spin, shrink, fall, fade. A thing that vanishes instantly leaves you unsure you hit it."),

    ("shake", "screen shake", "camera",
     "Trauma budget, displacement = trauma squared. Everything adds trauma, nothing sets offset."),
    ("kick", "directional kick", "camera",
     "Punch the whole frame along the blow and spring back. Gives the hit a direction."),
    ("zoom", "zoom punch", "camera",
     "A damped spring on scale. Tiny numbers -- three percent is plenty at this size."),
    ("tilt", "roll tilt", "camera",
     "A degree or two of roll on impact. Use sparingly, it makes people seasick."),
    ("lag", "camera lag", "camera",
     "Follow with an exponential delay so the player can pull away from centre and drift back."),

    ("particles", "particles", "world",
     "A cone of sparks with varied speed and drag. One speed for all of them reads as a cartwheel."),
    ("numbers", "damage numbers", "world",
     "Pop in on ease-out-back, rise, fall away. Reads the amount without a log line."),
    ("shockwave", "shockwave ring", "world",
     "An expanding ring, most of it in the first three frames. Force leaving a point."),
    ("ripple", "floor ripple", "world",
     "A wave through the tiles themselves, so the grid is a surface and not wallpaper."),
    ("scrflash", "screen flash", "world",
     "A frame or two of tint over everything. Save it for the big ones."),
    ("vignette", "vignette pulse", "world",
     "Darkened corners that clamp down on impact. Focuses the eye at the centre."),
    ("rgbsplit", "RGB split", "world",
     "Separate the colour channels for a few frames. Watch the ms readout -- "
     "on the CPU this one effect is most of a 60Hz frame."),
    ("logpop", "message pop", "world",
     "Scale each log line in as it lands and fade it out after. Even the text "
     "is animated -- turn it off and the log goes back to a printout."),
]


class Juice:
    """All the toggles, the master A/B switch, and the global intensity dial.

    `on(name)` is the only thing the rest of the file asks. Routing every check
    through here is what makes the master switch and the intensity dial one-line
    features instead of an edit at thirty call sites.
    """

    def __init__(self) -> None:
        self.toggles = {k: Toggle(k, lbl, grp, blurb) for k, lbl, grp, blurb in TOGGLE_SPECS}
        self.master = True          # False = the raw, unjuiced sim
        self.intensity = 1.0        # scales every amplitude, not the timings
        self.move_ease = "out_quad"

    def on(self, name: str) -> bool:
        return self.master and self.toggles[name].on

    def amt(self, value: float) -> float:
        """Scale an amplitude by the intensity dial."""
        return value * self.intensity

    def set_all(self, state: bool) -> None:
        for t in self.toggles.values():
            t.on = state

    @property
    def groups(self):
        seen = []
        for t in self.toggles.values():
            if t.group not in seen:
                seen.append(t.group)
        return seen


# ---------------------------------------------------------------------------
# The tiny sim
# ---------------------------------------------------------------------------


class Entity:
    """A glyph with a body, an animator and some hit points.

    Composition rather than a class tree: a monster is not a subclass of
    anything, it is an Entity that happens to hold different values. All the
    behaviour lives in the Motions attached to its Animator, which is what lets
    the juice layer be swapped out wholesale by the master toggle.
    """

    def __init__(self, glyph: str, x: int, y: int, color, name: str, hp: int = 5):
        self.glyph = glyph
        self.color = color
        self.name = name
        self.body = Body(tx=float(x), ty=float(y))
        self.anim = Animator()
        self.trail = GhostTrail()
        self.hp = hp
        self.max_hp = hp
        self.dying = False
        self.death_timer = 0.0
        # Seeded off the position so a row of dummies does not breathe in step.
        # Persistent, so it is gated by `anim.persistent_enabled` rather than by
        # declining to play it -- see World.update.
        self.anim.add_persistent(Breathe(phase=(x * 3 + y * 7) % 7))

    @property
    def tile(self):
        return int(self.body.tx), int(self.body.ty)

    def world_pos(self):
        """Pixel centre including every juice offset applied this frame."""
        b = self.body
        return ((b.tx + 0.5 + b.ox) * TILE, (b.ty + 0.5 + b.oy) * TILE)

    def tile_center(self):
        """Pixel centre of the logical tile, ignoring the juice offsets."""
        return ((self.body.tx + 0.5) * TILE, (self.body.ty + 0.5) * TILE)


def build_map():
    """A room with a wall border and a few pillars to bump into."""
    grid = [[0] * GRID_W for _ in range(GRID_H)]
    for x in range(GRID_W):
        grid[0][x] = grid[GRID_H - 1][x] = 1
    for y in range(GRID_H):
        grid[y][0] = grid[y][GRID_W - 1] = 1
    for (px, py) in [(7, 6), (7, 14), (17, 6), (17, 14), (12, 3), (12, 17)]:
        grid[py][px] = 1
    return grid


class World:
    """Map, entities, and the shared effect pools. No drawing happens here.

    Everything in this class can be stepped with no display up, which is what
    `juicetest.py` leans on -- the sim and the effect state are testable, and
    only the compositing needs a surface.
    """

    def __init__(self, juice: Juice):
        self.juice = juice
        self.grid = build_map()
        self.fx = EffectField()
        self.trauma = Trauma()
        self.hitstop = HitStop()
        self.camera = Camera()
        self.kick = [Spring(0.0, 0.0, stiffness=210.0, damping=16.0),
                     Spring(0.0, 0.0, stiffness=210.0, damping=16.0)]
        self.screen_flash = 0.0
        self.vignette_pulse = 0.0
        self.rgb_split = 0.0
        self.log: list[list] = []          # [text, age, colour]
        self.reset()

    # -- setup ------------------------------------------------------------
    def reset(self):
        self.player = Entity("@", 12, 10, PLAYER_COLOR, "you", hp=99)
        self.entities = [self.player]
        for i, (x, y) in enumerate([(10, 8), (14, 8), (10, 12), (14, 12)]):
            self.entities.append(
                Entity("k", x, y, DUMMY_COLORS[i], f"dummy {i + 1}", hp=4))
        self.fx.clear()
        self.log.clear()
        cx, cy = self.player.tile_center()
        self.camera.snap(cx, cy)
        self.clamp_camera()
        self.say("bump a dummy to swing at it", ACCENT)

    def clamp_camera(self):
        """Keep the viewport inside the map, centring on any axis too small.

        Without this the camera happily drifts off the edge of the room and
        half the screen is background, which makes every camera effect harder
        to read against nothing.
        """
        for axis, extent, view in (("x", GRID_W * TILE, VIEW_W),
                                   ("y", GRID_H * TILE, VIEW_H)):
            if extent <= view:
                setattr(self.camera, axis, extent * 0.5)
            else:
                v = getattr(self.camera, axis)
                setattr(self.camera, axis, min(max(v, view * 0.5), extent - view * 0.5))

    def say(self, text, color=INK):
        self.log.append([text, 0.0, color])
        del self.log[:-6]

    # -- queries ----------------------------------------------------------
    def blocked(self, x: int, y: int) -> bool:
        if not (0 <= x < GRID_W and 0 <= y < GRID_H):
            return True
        return self.grid[y][x] == 1

    def entity_at(self, x: int, y: int):
        for e in self.entities:
            if not e.dying and e.tile == (x, y):
                return e
        return None

    def nearest_dummy(self):
        others = [e for e in self.entities if e is not self.player and not e.dying]
        if not others:
            return None
        px, py = self.player.tile
        return min(others, key=lambda e: (e.tile[0] - px) ** 2 + (e.tile[1] - py) ** 2)

    # -- actions ----------------------------------------------------------
    def try_move(self, e: Entity, dx: int, dy: int):
        """One turn's worth of movement, which may turn into an attack.

        Note the order: the *logical* move happens immediately and completely,
        and only then is an animation attached to drag the picture back to
        where the entity used to be. The sim never waits for a tween.
        """
        if e.anim.busy or self.hitstop.frozen:
            return
        e.body.facing = (dx, dy)
        nx, ny = e.tile[0] + dx, e.tile[1] + dy

        target = self.entity_at(nx, ny)
        if target is not None and target is not e:
            self.attack(e, target, dx, dy)
            return

        if self.blocked(nx, ny):
            self.bump_wall(e, dx, dy)
            return

        e.body.tx, e.body.ty = float(nx), float(ny)   # instant, authoritative
        j = self.juice
        if not j.on("tween"):
            # The raw sim: the sprite is simply somewhere else now. This is the
            # baseline every other movement effect is an argument against.
            return
        if j.on("hop"):
            e.anim.play(Hop(duration=MOVE_TIME, dx=dx, dy=dy,
                            height=j.amt(0.34),
                            squash=j.amt(0.26) if j.on("squash") else 0.0,
                            ease=EASINGS[j.move_ease]))
            if j.on("dust"):
                # Fired at the end of the hop rather than the start, so it is
                # the landing that kicks up floor and not the take-off.
                e.anim.play(Sequence([Wait(duration=MOVE_TIME * 0.86),
                                      Callback(lambda e=e: self.landing_dust(e))]))
        else:
            e.anim.play(Slide(duration=MOVE_TIME, dx=dx, dy=dy,
                              ease=EASINGS[j.move_ease]))

    def landing_dust(self, e: Entity):
        x, y = e.tile_center()
        self.fx.particles.burst(
            x, y + TILE * 0.34, count=6, direction=(0, -1), spread=2.4,
            speed=(30 * PX, 110 * PX), life=(0.16, 0.34), size=3.0 * PX,
            colors=((90, 96, 118), (70, 76, 96)), gravity=260.0 * PX, drag=3.0)

    def bump_wall(self, e: Entity, dx: int, dy: int):
        """A move that failed. It still gets an animation -- silence reads as a
        dropped input, and the player will just press the key again."""
        if not self.juice.on("wallbump"):
            return
        e.anim.play(Sequence([
            Lunge(duration=0.16, dx=dx, dy=dy, reach=self.juice.amt(0.22),
                  out_frac=0.35, stretch=0.1),
            Shiver(duration=0.16, amplitude=self.juice.amt(0.05), seed=int(e.body.tx)),
        ]))
        if self.juice.on("shake"):
            self.trauma.add(0.12)
        wx = (e.body.tx + 0.5 + dx * 0.55) * TILE
        wy = (e.body.ty + 0.5 + dy * 0.55) * TILE
        if self.juice.on("particles"):
            self.fx.particles.burst(
                wx, wy, count=5, direction=(-dx, -dy), spread=1.6,
                speed=(60 * PX, 190 * PX), life=(0.15, 0.3), size=3.0 * PX,
                colors=((120, 126, 150),), gravity=500.0 * PX)

    def attack(self, attacker: Entity, target: Entity, dx: int, dy: int):
        """Schedule a swing.

        The whole attack is one nested structure, and the reason it is worth
        writing this way is the `Callback` buried in the middle: the damage,
        the shake, the sparks and the freeze all fire on the frame the fist
        *arrives*, not the frame the key was pressed. Getting that one moment
        right is most of what makes an attack feel connected.
        """
        j = self.juice
        steps = []
        if j.on("windup"):
            steps.append(Anticipate(duration=ATTACK_WINDUP, dx=dx, dy=dy,
                                    amount=j.amt(0.20), squash=j.amt(0.18)))

        impact = Callback(lambda: self.land_blow(attacker, target, dx, dy))
        if j.on("lunge"):
            lunge = Lunge(duration=ATTACK_TIME, dx=dx, dy=dy,
                          reach=j.amt(0.55), out_frac=0.32, stretch=j.amt(0.22))
            # Contact is the moment the lunge is fully extended.
            steps.append(Parallel([lunge,
                                   Sequence([Wait(duration=ATTACK_TIME * 0.32),
                                             impact])]))
        elif j.on("tween"):
            steps.append(impact)
            steps.append(Wait(duration=ATTACK_TIME))
        else:
            # Nothing to watch, so nothing to wait for -- the blow simply lands.
            steps.append(impact)
        attacker.anim.play(Sequence(steps))

    def land_blow(self, attacker: Entity, target: Entity, dx: int, dy: int):
        """The moment of contact: the damage, plus everything that confirms it."""
        if target.dying:
            return
        j = self.juice
        damage = 1 + (abs(hash((target.name, target.hp))) % 3)
        target.hp -= damage
        killing = target.hp <= 0

        tx, ty = target.tile_center()
        # Sparks want to appear on the contact face, not in the middle of the
        # victim -- so pull them back towards the attacker by half a tile.
        cx, cy = tx - dx * TILE * 0.4, ty - dy * TILE * 0.4
        heavy = 1.6 if killing else 1.0

        # -- what the victim does -----------------------------------------
        if j.on("hitflash"):
            target.anim.play(Flash(duration=FLASH_TIME, strength=1.0))
        if j.on("knockback"):
            target.anim.play(Knockback(duration=0.3, dx=dx, dy=dy,
                                       distance=j.amt(0.3 * heavy)))
        if j.on("shiver"):
            target.anim.play(Shiver(duration=0.26, amplitude=j.amt(0.07 * heavy),
                                    seed=int(target.body.tx * 13 + target.body.ty)))

        # -- what the frame does ------------------------------------------
        if j.on("hitstop"):
            self.hitstop.hit(HITSTOP_HEAVY if killing else HITSTOP_LIGHT)
        if j.on("shake"):
            self.trauma.add(j.amt(TRAUMA_HEAVY if killing else TRAUMA_LIGHT))
        if j.on("kick"):
            self.kick[0].kick(-dx * j.amt(900.0 * PX * heavy))
            self.kick[1].kick(-dy * j.amt(900.0 * PX * heavy))
        if j.on("zoom"):
            self.camera.punch(zoom=j.amt(1.1 * heavy))
        if j.on("tilt"):
            # Sideways blows roll the frame; vertical ones get a fixed nudge,
            # since a straight up-down hit has no roll axis of its own.
            self.camera.punch(tilt=j.amt((-dx * 26.0 + (14.0 if dx == 0 else 0.0)) * heavy))
        if j.on("scrflash"):
            self.screen_flash = max(self.screen_flash, j.amt(0.16 * heavy))
        if j.on("vignette"):
            self.vignette_pulse = max(self.vignette_pulse, j.amt(0.55 * heavy))
        if j.on("rgbsplit"):
            self.rgb_split = max(self.rgb_split, j.amt(5.0 * heavy))

        # -- what the world does ------------------------------------------
        if j.on("particles"):
            self.fx.particles.burst(
                cx, cy, count=int(12 * heavy), direction=(dx, dy), spread=1.9,
                speed=(140 * PX, 430 * PX * heavy), life=(0.22, 0.5),
                size=4.0 * PX, gravity=900.0 * PX,
                colors=(target.color, (255, 240, 200), (255, 190, 120)))
        if j.on("numbers"):
            self.fx.floaters.add(str(damage), tx, ty - TILE * 0.3,
                                 vx=lerp(-40, 40, ((damage * 37) % 10) / 10.0) * PX,
                                 vy=-150.0 * PX * heavy, gravity=150.0 * PX,
                                 color=GOLD if killing else (255, 236, 170),
                                 life=0.85, max_life=0.85)
        if j.on("shockwave"):
            self.fx.shockwave(cx, cy, max_radius=j.amt(TILE * 1.75 * heavy),
                              life=0.35, width=5.0 * PX * heavy,
                              color=(255, 240, 210))
        if j.on("ripple"):
            self.fx.impact(cx, cy, strength=j.amt(7.0 * PX * heavy), life=0.5,
                           speed=520.0 * PX, wavelength=60.0 * PX)
        if j.on("slash"):
            self.fx.slash(tx - dx * TILE * 0.25, ty - dy * TILE * 0.25,
                          angle=math.degrees(math.atan2(-dy, dx)),
                          radius=TILE * 0.62, life=0.16)

        if killing:
            self.kill(target)
        else:
            self.say(f"{target.name} takes {damage}", target.color)

    def kill(self, target: Entity):
        target.dying = True
        target.anim.clear()
        j = self.juice
        if j.on("death"):
            target.anim.play(DeathSpin(duration=0.55, spin=430.0, drop=0.45))
            target.death_timer = 0.55
        else:
            target.death_timer = 0.0     # gone on the next frame, no ceremony
        if j.on("particles"):
            x, y = target.tile_center()
            self.fx.particles.burst(x, y, count=22, speed=(90 * PX, 380 * PX),
                                    life=(0.3, 0.75), size=4.5 * PX,
                                    gravity=900.0 * PX,
                                    colors=(target.color, (255, 220, 180)))
        self.say(f"{target.name} dies", WARN)

    def strike_player(self):
        """Have the nearest dummy hit the player -- the same code, reversed, so
        the receiving end of every effect can be looked at too."""
        d = self.nearest_dummy()
        if d is None or d.anim.busy:
            return
        px, py = self.player.tile
        ddx, ddy = px - d.tile[0], py - d.tile[1]
        # Snapped to the dominant axis; the bench is not a pathfinder.
        if abs(ddx) >= abs(ddy):
            dx, dy = (1 if ddx >= 0 else -1), 0
        else:
            dx, dy = 0, (1 if ddy > 0 else -1)
        self.attack(d, self.player, dx, dy)

    def swing(self):
        """Attack whatever is in front, or whiff at empty air."""
        p = self.player
        if p.anim.busy or self.hitstop.frozen:
            return
        dx, dy = p.body.facing
        target = self.entity_at(p.tile[0] + dx, p.tile[1] + dy)
        if target is not None and target is not p:
            self.attack(p, target, dx, dy)
            return

        j = self.juice
        steps = []
        if j.on("windup"):
            steps.append(Anticipate(duration=ATTACK_WINDUP, dx=dx, dy=dy,
                                    amount=j.amt(0.18)))
        if j.on("lunge"):
            steps.append(Lunge(duration=ATTACK_TIME, dx=dx, dy=dy,
                               reach=j.amt(0.45), out_frac=0.32))
        if steps:
            p.anim.play(Sequence(steps))
        if j.on("slash"):
            cx, cy = p.tile_center()
            self.fx.slash(cx + dx * TILE * 0.55, cy + dy * TILE * 0.55,
                          angle=math.degrees(math.atan2(-dy, dx)),
                          radius=TILE * 0.55, life=0.16)

    # -- per-frame --------------------------------------------------------
    def update(self, dt: float):
        """One frame.

        Two clocks, and the split between them *is* hit-stop: `anim_dt` is what
        the bodies and the world effects run on and it is zero while frozen,
        while `dt` keeps running for the shake and the springs. A frame in which
        literally nothing changes reads as the game hanging; a frame in which
        everything but the shake is still reads as a blow landing.
        """
        anim_dt = self.hitstop.consume(dt)

        breathing = self.juice.on("bob")
        for e in list(self.entities):
            was_busy = e.anim.busy
            e.anim.persistent_enabled = breathing
            e.anim.update(e.body, anim_dt)
            if self.juice.on("ghost"):
                x, y = e.world_pos()
                b = e.body
                e.trail.sample(x, y, b.sx, b.sy, b.angle, anim_dt, moving=was_busy)
            else:
                e.trail.clear()
            e.trail.update(dt)

            if e.dying:
                e.death_timer -= anim_dt
                if e.death_timer <= 0.0:
                    self.entities.remove(e)

        self.fx.update(anim_dt)
        self.trauma.update(dt)
        for k in self.kick:
            k.update(dt)
        self.camera.update(dt)

        cx, cy = self.player.world_pos()
        self.camera.follow(cx, cy, dt, smooth=self.juice.on("lag"))
        self.clamp_camera()

        self.screen_flash = max(0.0, self.screen_flash - dt * 1.6)
        self.vignette_pulse = max(0.0, self.vignette_pulse - dt * 2.2)
        self.rgb_split = max(0.0, self.rgb_split - dt * 26.0)

        for entry in self.log:
            entry[1] += dt


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class Renderer:
    """Everything that touches a pixel.

    Kept as its own object holding no sim state, so the World above can be
    stepped by a test with this class never constructed at all.
    """

    def __init__(self):
        mono = pygame.font.match_font(
            "dejavusansmono,couriernew,consolas,freemono,monospace")

        def load(size, bold=False):
            f = pygame.font.Font(mono, size) if mono else pygame.font.Font(None, int(size * 1.3))
            f.set_bold(bold)
            return f

        self.font_tile = load(int(TILE * 0.82), bold=True)
        self.font_ui = load(13)
        self.font_ui_b = load(13, bold=True)
        self.font_big = load(18, bold=True)
        self.font_num = load(18, bold=True)

        self.view = pygame.Surface((VIEW_W, VIEW_H)).convert()
        self.overlay = pygame.Surface((VIEW_W, VIEW_H), pygame.SRCALPHA)
        self.scratch = pygame.Surface((VIEW_W, VIEW_H), pygame.SRCALPHA)
        # Preallocated channel buffers for the RGB split. Copying a 900x760
        # surface is 2.7MB of allocation, and doing that twice a frame while
        # the effect is running is most of what it used to cost.
        self.split_r = pygame.Surface((VIEW_W, VIEW_H)).convert()
        self.split_b = pygame.Surface((VIEW_W, VIEW_H)).convert()
        self.vignette = self._make_vignette()
        self._glyphs: dict = {}
        self.rows: list = []       # (rect, toggle key), rebuilt every panel draw
        self.hover: str | None = None
        # Smoothed cost of a whole frame, shown in the panel. Several of these
        # effects are not free -- the RGB split alone is most of a frame's
        # budget -- and a bench that hid that would be teaching the wrong
        # lesson. Toggle one and watch the number move.
        self.frame_ms = 0.0

    # -- helpers ----------------------------------------------------------
    def _make_vignette(self):
        """Nested rectangle outlines, darkest at the edge. Built once.

        A true per-pixel radial falloff would mean either numpy or a few hundred
        thousand Python-level pixel writes; at this size, behind 40% alpha,
        nobody can tell the two apart.
        """
        surf = pygame.Surface((VIEW_W, VIEW_H), pygame.SRCALPHA)
        steps = 80
        reach = min(VIEW_W, VIEW_H) * 0.6
        band = int(reach / steps) + 2
        for i in range(steps):
            t = i / steps
            a = int(255 * (t ** 2.4))
            inset = int((1.0 - t) * reach)
            rect = pygame.Rect(inset, inset, VIEW_W - inset * 2, VIEW_H - inset * 2)
            if rect.w <= 2 or rect.h <= 2:
                continue
            pygame.draw.rect(surf, (0, 0, 0, a), rect, width=band)
        return surf

    def glyph(self, ch: str, color):
        key = (ch, color)
        surf = self._glyphs.get(key)
        if surf is None:
            surf = self.font_tile.render(ch, True, color).convert_alpha()
            self._glyphs[key] = surf
        return surf

    def to_view(self, world: World, wx: float, wy: float):
        """World pixels -> view-surface pixels."""
        return (wx - world.camera.x + VIEW_W * 0.5,
                wy - world.camera.y + VIEW_H * 0.5)

    # -- the world --------------------------------------------------------
    def draw_world(self, world: World):
        self.view.fill(BG)
        self.draw_floor(world)
        self.draw_shockwaves(world)
        self.draw_entities(world)
        self.draw_slashes(world)
        self.draw_particles(world)
        self.draw_floaters(world)

    def draw_floor(self, world: World):
        """Tiles, each displaced by whatever impact waves are passing through.

        This is the effect people forget. Shaking the camera moves everything
        together, which the eye reads as the *camera* being hit. Moving tiles
        against each other is the only way the floor itself takes part.
        """
        v = self.view
        rippling = world.juice.on("ripple") and world.fx.impacts
        # Only walk the tiles that can be on screen.
        x0 = max(0, int((world.camera.x - VIEW_W * 0.5) // TILE) - 1)
        x1 = min(GRID_W, int((world.camera.x + VIEW_W * 0.5) // TILE) + 2)
        y0 = max(0, int((world.camera.y - VIEW_H * 0.5) // TILE) - 1)
        y1 = min(GRID_H, int((world.camera.y + VIEW_H * 0.5) // TILE) + 2)

        for ty in range(y0, y1):
            for tx in range(x0, x1):
                wx, wy = (tx + 0.5) * TILE, (ty + 0.5) * TILE
                ox = oy = 0.0
                if rippling:
                    ox, oy = world.fx.tile_offset(wx, wy)
                sx, sy = self.to_view(world, wx + ox, wy + oy)
                rect = pygame.Rect(0, 0, TILE - 2, TILE - 2)
                rect.center = (int(sx), int(sy))
                if world.grid[ty][tx]:
                    pygame.draw.rect(v, WALL, rect, border_radius=3)
                    pygame.draw.rect(v, WALL_TOP, rect.inflate(-8, -8), border_radius=2)
                else:
                    shade = FLOOR if (tx + ty) % 2 == 0 else FLOOR_ALT
                    pygame.draw.rect(v, shade, rect, border_radius=3)

    def draw_entities(self, world: World):
        for e in world.entities:
            self.draw_trail(world, e)
            self.draw_body(world, e)
            if not e.dying and e is not world.player:
                self.draw_hp(world, e)

    def _transform(self, base, sx: float, sy: float, angle: float):
        """Squash, stretch and spin one glyph.

        The non-uniform scale has to happen before the rotation, or a squashed
        sprite would end up squashed along the *screen* axes instead of its own.
        """
        img = base
        if abs(sx - 1.0) > 0.005 or abs(sy - 1.0) > 0.005:
            w = max(1, int(base.get_width() * sx))
            h = max(1, int(base.get_height() * sy))
            img = pygame.transform.smoothscale(base, (w, h))
        if abs(angle) > 0.1:
            img = pygame.transform.rotozoom(img, angle, 1.0)
        return img

    def draw_trail(self, world: World, e: Entity):
        base = self.glyph(e.glyph, e.color)
        for g in e.trail.ghosts:
            img = self._transform(base, g.sx, g.sy, g.angle).copy()
            img.set_alpha(int(110 * g.alpha))
            sx, sy = self.to_view(world, g.x, g.y)
            self.view.blit(img, img.get_rect(center=(int(sx), int(sy))))

    def draw_body(self, world: World, e: Entity):
        b = e.body
        img = self._transform(self.glyph(e.glyph, e.color), b.sx, b.sy, b.angle)

        if b.flash > 0.0 or b.alpha < 1.0:
            img = img.copy()
            if b.flash > 0.0:
                # Adding to RGB but not to alpha whitens the glyph while
                # leaving its silhouette exactly as it was.
                white = img.copy()
                white.fill((255, 255, 255, 0), special_flags=pygame.BLEND_RGBA_ADD)
                white.set_alpha(int(255 * clamp(b.flash)))
                img.blit(white, (0, 0))
            if b.alpha < 1.0:
                img.set_alpha(int(255 * clamp(b.alpha)))

        wx, wy = e.world_pos()
        sx, sy = self.to_view(world, wx, wy)
        self.view.blit(img, img.get_rect(center=(int(sx), int(sy))))

    def draw_hp(self, world: World, e: Entity):
        wx, wy = e.world_pos()
        sx, sy = self.to_view(world, wx, wy - TILE * 0.55)
        w = TILE - 12
        rect = pygame.Rect(0, 0, w, 3)
        rect.center = (int(sx), int(sy))
        pygame.draw.rect(self.view, (60, 40, 44), rect)
        frac = clamp(e.hp / e.max_hp)
        if frac > 0:
            pygame.draw.rect(self.view, e.color,
                             pygame.Rect(rect.left, rect.top, int(w * frac), 3))

    def draw_particles(self, world: World):
        for p in world.fx.particles.particles:
            size = max(1, int(p.size * (1.0 - p.t)))
            a = int(255 * (1.0 - p.t) ** 0.6)
            if a <= 2:
                continue
            sx, sy = self.to_view(world, p.x, p.y)
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            surf.fill((*p.color, a))
            self.view.blit(surf, (int(sx - size), int(sy - size)))

    def draw_floaters(self, world: World):
        for f in world.fx.floaters.floaters:
            txt = self.font_num.render(f.text, True, f.color)
            if abs(f.scale - 1.0) > 0.01:
                w = max(1, int(txt.get_width() * f.scale))
                h = max(1, int(txt.get_height() * f.scale))
                txt = pygame.transform.smoothscale(txt, (w, h))
            txt = txt.copy()
            txt.set_alpha(int(255 * clamp(f.alpha)))
            sx, sy = self.to_view(world, f.x, f.y)
            self.view.blit(txt, txt.get_rect(center=(int(sx), int(sy))))

    def draw_shockwaves(self, world: World):
        for s in world.fx.shockwaves:
            r = int(s.radius)
            a = int(220 * s.alpha)
            if r < 2 or a <= 2:
                continue
            surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*s.color, a), (r + 2, r + 2), r, s.thickness)
            sx, sy = self.to_view(world, s.x, s.y)
            self.view.blit(surf, (int(sx - r - 2), int(sy - r - 2)))

    def draw_slashes(self, world: World):
        """The weapon sweep, as a lens-shaped polygon that opens over its life.

        Cheaper than an arc primitive with a varying width, and at six frames
        nobody is going to tell the difference.
        """
        for s in world.fx.slashes:
            a = int(240 * s.alpha)
            if a <= 3:
                continue
            sx, sy = self.to_view(world, s.x, s.y)
            span = math.radians(s.sweep)
            start = math.radians(s.angle) - span * 0.5
            reach = s.radius * lerp(0.55, 1.15, s.progress)
            outer, inner = [], []
            for i in range(11):
                t = i / 10.0
                ang = start + span * t
                thick = math.sin(t * math.pi) * 5.0 * PX * (1.0 - s.t)
                ca, sa = math.cos(ang), -math.sin(ang)
                outer.append((sx + ca * (reach + thick), sy + sa * (reach + thick)))
                inner.append((sx + ca * (reach - thick), sy + sa * (reach - thick)))
            self.scratch.fill((0, 0, 0, 0))
            pygame.draw.polygon(self.scratch, (*s.color, a), outer + inner[::-1])
            self.view.blit(self.scratch, (0, 0))

    # -- compositing ------------------------------------------------------
    def composite(self, screen, world: World):
        """Put the finished world on the screen, via every whole-frame effect.

        Order matters. Shake, kick, zoom and tilt move the picture; the flash
        and the vignette are stuck to the *screen* and must not move with it,
        or the vignette would slide off its own corners every time something
        explodes.
        """
        j = world.juice
        # Done first, on the fixed-size view rather than on the rotozoomed
        # result, so it can use preallocated buffers. A lens would aberrate
        # after the camera transform; at five pixels for eleven frames, the
        # difference is not something anyone is going to catch.
        if j.on("rgbsplit") and world.rgb_split > 0.4:
            self._apply_rgb_split(world.rgb_split)

        dx = dy = roll = 0.0
        if j.on("shake"):
            sx, sy, sr = world.trauma.offset(max_px=16.0 * PX * j.intensity,
                                             max_deg=2.4 * j.intensity)
            dx, dy, roll = dx + sx, dy + sy, roll + sr
        if j.on("kick"):
            dx += world.kick[0].value
            dy += world.kick[1].value
        if j.on("tilt"):
            roll += world.camera.tilt.value
        zoom = clamp(world.camera.zoom.value if j.on("zoom") else 1.0, 0.85, 1.3)

        if abs(roll) > 0.05 or abs(zoom - 1.0) > 0.002:
            # rotozoom allocates a whole new surface, so it is kept off the
            # path entirely on frames where neither is actually in play.
            shown = pygame.transform.rotozoom(self.view, roll, zoom)
        else:
            shown = self.view
        pos = shown.get_rect(center=(int(VIEW_W * 0.5 + dx), int(VIEW_H * 0.5 + dy)))
        screen.fill(BG, (0, 0, VIEW_W, VIEW_H))
        screen.blit(shown, pos)

        if j.on("scrflash") and world.screen_flash > 0.001:
            self.overlay.fill((255, 245, 225, int(200 * clamp(world.screen_flash))))
            screen.blit(self.overlay, (0, 0))

        if j.on("vignette"):
            self.vignette.set_alpha(int(90 + 130 * clamp(world.vignette_pulse)))
            screen.blit(self.vignette, (0, 0))

    def _apply_rgb_split(self, amount: float):
        """Chromatic aberration, in place on the view.

        Pull red and blue into scratch buffers, mask the view down to green,
        then add the two back at opposite offsets. Five full-frame operations
        and no allocation -- still the most expensive thing in this file, which
        is why it is a toggle and why it only runs for the handful of frames
        after a heavy hit.
        """
        d = int(clamp(amount, 0.0, 12.0))
        if d < 1:
            return
        self.split_r.blit(self.view, (0, 0))
        self.split_r.fill((255, 0, 0), special_flags=pygame.BLEND_RGB_MULT)
        self.split_b.blit(self.view, (0, 0))
        self.split_b.fill((0, 0, 255), special_flags=pygame.BLEND_RGB_MULT)
        self.view.fill((0, 255, 0), special_flags=pygame.BLEND_RGB_MULT)
        self.view.blit(self.split_r, (-d, 0), special_flags=pygame.BLEND_RGB_ADD)
        self.view.blit(self.split_b, (d, 0), special_flags=pygame.BLEND_RGB_ADD)

    # -- the panel --------------------------------------------------------
    def draw_panel(self, screen, world: World, mouse):
        j = world.juice
        x0 = VIEW_W
        screen.fill(PANEL_BG, pygame.Rect(x0, 0, PANEL_W, WIN_H))
        pygame.draw.line(screen, PANEL_LINE, (x0, 0), (x0, WIN_H))

        screen.blit(self.font_big.render("JUICE", True, INK), (x0 + 14, 8))
        screen.blit(self.font_ui_b.render("ON" if j.master else "OFF (Tab)", True,
                                          ACCENT if j.master else WARN), (x0 + 86, 13))
        cost = f"{self.frame_ms:4.1f} ms"
        # Red once a frame costs more than a 60Hz slot.
        cost_col = WARN if self.frame_ms > 16.7 else DIM
        screen.blit(self.font_ui.render(cost, True, cost_col),
                    (x0 + PANEL_W - 14 - self.font_ui.size(cost)[0], 13))

        # Two columns, split so the taller pair of groups sits on the left.
        # Anything that changes the toggle list will change the balance; the
        # column bottoms are measured rather than assumed, so it stays tidy.
        self.rows.clear()
        self.hover = None
        mx, my = mouse
        col_w = (PANEL_W - 34) // 2
        bottom = 0
        for ci, groups in enumerate((("movement", "attack"), ("camera", "world"))):
            cx = x0 + 14 + ci * (col_w + 6)
            y = 34
            for gi, group in enumerate(groups):
                if gi:
                    y += 10
                screen.blit(self.font_ui_b.render(group.upper(), True, ACCENT), (cx, y))
                pygame.draw.line(screen, PANEL_LINE, (cx, y + 15), (cx + col_w, y + 15))
                y += 19
                for t in j.toggles.values():
                    if t.group != group:
                        continue
                    rect = pygame.Rect(cx - 4, y - 2, col_w + 8, 18)
                    self.rows.append((rect, t.key))
                    if rect.collidepoint(mx, my):
                        self.hover = t.key
                        pygame.draw.rect(screen, (32, 34, 44), rect, border_radius=3)
                    label_col = INK if t.on else (78, 84, 100)
                    if not j.master:
                        label_col = (58, 61, 72)
                    screen.blit(self.font_ui.render("[x]" if t.on else "[ ]", True,
                                                    ACCENT if t.on else (70, 74, 90)),
                                (cx + 2, y))
                    screen.blit(self.font_ui.render(t.label, True, label_col), (cx + 32, y))
                    y += 18
            bottom = max(bottom, y)

        # The easing gallery, under the columns.
        y = bottom + 12
        screen.blit(self.font_ui.render(f"move ease  {j.move_ease}", True, DIM),
                    (x0 + 14, y))
        screen.blit(self.font_ui.render(f"x{j.intensity:.2f}", True, GOLD),
                    (x0 + PANEL_W - 52, y))
        self._draw_curve(screen, x0 + 14, y + 17, EASINGS[j.move_ease])

        # The blurb for whatever the mouse is over, wrapped into the footer.
        footer = WIN_H - FOOTER_H
        pygame.draw.line(screen, PANEL_LINE, (x0 + 10, footer), (x0 + PANEL_W - 10, footer))
        ty = footer + 7
        if self.hover:
            t = j.toggles[self.hover]
            screen.blit(self.font_ui_b.render(t.label, True, GOLD), (x0 + 14, ty))
            ty += 16
            for line in self._wrap(t.blurb, PANEL_W - 30, self.font_ui):
                screen.blit(self.font_ui.render(line, True, DIM), (x0 + 14, ty))
                ty += 14
        else:
            for line in HELP_LINES:
                if line:
                    screen.blit(self.font_ui.render(line, True, DIM), (x0 + 14, ty))
                ty += 14

    def _draw_curve(self, screen, x, y, fn):
        """A little plot of the active easing curve. Seeing the overshoot in
        `out_back` as a line makes the on-screen snap far easier to reason about
        than staring at the sprite does."""
        w, h = PANEL_W - 28, 46
        pygame.draw.rect(screen, (26, 28, 36), (x, y, w, h), border_radius=3)
        pygame.draw.line(screen, PANEL_LINE, (x, y + h - 8), (x + w, y + h - 8))
        pts = [(x + i, y + h - 8 - fn(i / (w - 1)) * (h - 16)) for i in range(w)]
        pygame.draw.lines(screen, ACCENT, False, pts, 2)

    @staticmethod
    def _wrap(text: str, width: int, font):
        lines, cur = [], ""
        for word in text.split():
            trial = f"{cur} {word}".strip()
            if font.size(trial)[0] <= width:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def draw_hud(self, screen, world: World):
        """The message log, each line popping in as it arrives.

        Held well clear of the bottom edge: a window that ends up a few pixels
        under a taskbar should lose empty floor, not the newest message.
        """
        popping = world.juice.on("logpop")
        y = VIEW_H - 44
        for text, age, color in reversed(world.log):
            a = int(255 * clamp((2.6 - age) / 0.8)) if popping else 255
            if a > 4:
                surf = self.font_ui_b.render(text, True, color)
                if popping and age < 0.18:   # a scale-up, so new text is noticed
                    s = 1.0 + 0.35 * (1.0 - age / 0.18)
                    surf = pygame.transform.smoothscale(
                        surf, (int(surf.get_width() * s), int(surf.get_height() * s)))
                surf = surf.copy()
                surf.set_alpha(a)
                screen.blit(surf, (16, y))
            y -= 17

        if not world.juice.master:
            screen.blit(self.font_big.render("RAW SIM  (juice off)", True, WARN), (16, 14))

    def panel_click(self, world: World, pos) -> bool:
        for rect, key in self.rows:
            if rect.collidepoint(pos):
                t = world.juice.toggles[key]
                t.on = not t.on
                world.say(f"{t.label}: {'on' if t.on else 'off'}",
                          ACCENT if t.on else DIM)
                return True
        return False


# ---------------------------------------------------------------------------
# Input and the loop
# ---------------------------------------------------------------------------

MOVE_KEYS = {
    pygame.K_LEFT: (-1, 0), pygame.K_RIGHT: (1, 0),
    pygame.K_UP: (0, -1), pygame.K_DOWN: (0, 1),
    pygame.K_a: (-1, 0), pygame.K_d: (1, 0),
    pygame.K_w: (0, -1), pygame.K_s: (0, 1),
    pygame.K_h: (-1, 0), pygame.K_l: (1, 0),
    pygame.K_k: (0, -1), pygame.K_j: (0, 1),
    pygame.K_KP4: (-1, 0), pygame.K_KP6: (1, 0),
    pygame.K_KP8: (0, -1), pygame.K_KP2: (0, 1),
}
# `k` is vi-up, so the kill shortcut is shift-K -- movement keys win, because
# that is what a roguelike player's hands expect.

EASE_ORDER = ["out_quad", "out_cubic", "out_quint", "out_back", "out_elastic",
              "out_bounce", "in_out_cubic", "linear"]


def handle_event(event, world: World, renderer: Renderer) -> bool:
    """Handle one event. Returns False to quit."""
    j = world.juice
    if event.type == pygame.QUIT:
        return False
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        renderer.panel_click(world, event.pos)
        return True
    if event.type != pygame.KEYDOWN:
        return True

    shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
    if event.key == pygame.K_ESCAPE or (event.key == pygame.K_q and not shift):
        return False

    if event.key == pygame.K_k and shift:
        d = world.nearest_dummy()
        if d:
            d.hp = 1
            world.land_blow(world.player, d, *world.player.body.facing)
    elif event.key in MOVE_KEYS:
        world.try_move(world.player, *MOVE_KEYS[event.key])
    elif event.key == pygame.K_SPACE:
        world.swing()
    elif event.key == pygame.K_x:
        world.strike_player()
    elif event.key == pygame.K_r:
        world.reset()
    elif event.key == pygame.K_TAB:
        j.master = not j.master
        world.say("juice ON" if j.master else "juice OFF -- raw sim",
                  ACCENT if j.master else WARN)
    elif event.key == pygame.K_F1:
        j.set_all(True)
        world.say("all effects on", ACCENT)
    elif event.key == pygame.K_F2:
        j.set_all(False)
        world.say("all effects off", DIM)
    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        j.intensity = max(0.0, j.intensity - 0.1)
    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
        j.intensity = min(2.0, j.intensity + 0.1)
    elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
        step = 1 if event.key == pygame.K_RIGHTBRACKET else -1
        j.move_ease = EASE_ORDER[(EASE_ORDER.index(j.move_ease) + step) % len(EASE_ORDER)]
        world.say(f"move ease: {j.move_ease}", ACCENT)
    return True


def draw_frame(screen, renderer: Renderer, world: World, mouse=(0, 0)):
    """One complete frame, factored out so the headless path uses it too."""
    renderer.draw_world(world)
    renderer.composite(screen, world)
    renderer.draw_hud(screen, world)
    renderer.draw_panel(screen, world, mouse)


def run():
    pygame.init()
    pygame.display.set_caption("juice workbench")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.key.set_repeat(220, 90)
    clock = pygame.time.Clock()

    world = World(Juice())
    renderer = Renderer()

    running = True
    while running:
        # Clamped, so a stall -- dragging the window, a breakpoint -- cannot
        # teleport every animation to its end state in one enormous step.
        dt = min(clock.tick(FPS) / 1000.0, 1.0 / 20.0)
        for event in pygame.event.get():
            if not handle_event(event, world, renderer):
                running = False

        t0 = time.perf_counter()
        world.update(dt)
        draw_frame(screen, renderer, world, pygame.mouse.get_pos())
        # Measured around the work, not around clock.tick, so the readout is
        # the cost of the frame and not the time spent waiting for the next one.
        ms = (time.perf_counter() - t0) * 1000.0
        renderer.frame_ms += (ms - renderer.frame_ms) * 0.08
        pygame.display.flip()

    pygame.quit()


def run_headless(path: str):
    """Play a scripted swing with no window and save the frame at impact.

    This is the regression test for the render path: it drives the same draw
    code the interactive loop does, so a crash in compositing shows up here
    rather than the first time somebody opens the bench.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    world = World(Juice())
    renderer = Renderer()
    dt = 1.0 / FPS

    def step(n):
        for _ in range(n):
            world.update(dt)

    # The player starts at (12,10) and dummy 1 stands at (10,8): walk up and
    # across to (10,9), then step north into it.
    for move in [(0, -1), (-1, 0), (-1, 0)]:
        world.try_move(world.player, *move)
        step(14)
    world.try_move(world.player, 0, -1)       # walks into dummy 1 -> attack
    assert world.player.tile == (10, 9), "the scripted walk drifted off course"
    # Stop a few frames past contact, with the sparks still in the air.
    step(int((ATTACK_WINDUP + ATTACK_TIME * 0.32) / dt) + 5)

    draw_frame(screen, renderer, world, (WIN_W - 200, 300))
    pygame.image.save(screen, path)
    pygame.quit()
    print(f"wrote {path} ({len(world.fx)} live effects at the moment of impact)")


def main():
    if "--headless" in sys.argv:
        i = sys.argv.index("--headless")
        path = sys.argv[i + 1] if len(sys.argv) > i + 1 else "juice_headless.png"
        run_headless(path)
    else:
        run()


if __name__ == "__main__":
    main()
