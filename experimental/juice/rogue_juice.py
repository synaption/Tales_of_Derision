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
from collections import deque

# The video driver has to be chosen before pygame brings the display up, so
# the headless flag is read here rather than in main(). Without this a
# headless run still tries to open a window, which on WSL means either a stray
# window on the desktop or a hang against a stale WSLg session.
if "--headless" in sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tiles  # noqa: E402
from juicefx import (  # noqa: E402
    ATTACK_TIME, ATTACK_WINDUP, DEATHS, EASINGS, FLASH_TIME, HITSTOP_HEAVY,
    HITSTOP_LIGHT, MOVE_TIME, TRAUMA_HEAVY, TRAUMA_LIGHT,
    Animator, Anticipate, Body, Breathe, Callback, Camera,
    EffectField, Flash, GhostTrail, HitStop, Hop, Jelly, Knockback, Lunge, Parallel,
    Sequence, Shiver, Slide, Spring, Trauma, Wait, clamp, lerp,
)

# ---------------------------------------------------------------------------
# Layout and palette
# ---------------------------------------------------------------------------

# 32 is exactly twice the 16px source art, so a resting sprite is drawn with no
# resampling at all and the squash effect deforms something already clean.
TILE = 32
# A 40x20 arena, which is far larger than the viewport in both axes -- the
# camera has to pan, which is what makes the lag and the shake worth looking
# at, and there is room for the enemies to actually path around cover.
GRID_W, GRID_H = 40, 20
VIEW_W, VIEW_H = 800, 600
# The toggles run in two columns, which is what keeps the window short enough
# to fit under a taskbar -- a single column of 28 rows needs 640px of panel on
# its own, before the header and the blurb.
PANEL_W = 380
WIN_W, WIN_H = VIEW_W + PANEL_W, VIEW_H     # 1180x600
# Panel space reserved for the hover blurb / control list. Sized to swallow
# the slack under the easing gallery rather than leave it as a dead band, so
# the blurb gets room to wrap and the controls get room to be spelled out.
FOOTER_H = 156

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

BG = (10, 11, 15)
# The floor is deliberately dim and low-contrast. It was competing with the
# sprites, and the floor's job here is to be a surface you can see the ripple
# travel through -- not to be looked at.
FLOOR = (28, 31, 41)
FLOOR_ALT = (24, 27, 36)
WALL = (62, 68, 88)
WALL_TOP = (96, 104, 132)
PANEL_BG = (20, 21, 28)
PANEL_LINE = (44, 47, 60)
INK = (208, 214, 230)
DIM = (110, 118, 140)
ACCENT = (120, 220, 190)
WARN = (240, 130, 110)
GOLD = (250, 205, 120)

PLAYER_COLOR = (225, 235, 250)

#: The roster: (sprite key, name, tint, hp, spawn x, spawn y).
#: Health is deliberately generous -- at four points a monster died to the
#: first or second blow, which meant you spent the whole session watching death
#: animations and almost never saw a hit *land* on something still standing.
MONSTERS = [
    ("goblin", "goblin", (150, 200, 130), 14, 13, 5),
    ("goblin", "goblin", (150, 200, 130), 14, 27, 15),
    ("lizard", "lizard", (120, 200, 190), 16, 24, 4),
    ("skeleton", "skeleton", (225, 225, 210), 12, 9, 14),
    ("wolf", "wolf", (200, 160, 130), 12, 33, 9),
    ("wolf", "wolf", (200, 160, 130), 12, 6, 9),
    ("raven", "raven", (170, 150, 220), 10, 22, 16),
    ("slime", "slime", (140, 220, 160), 18, 35, 13),
    ("ogre", "ogre", (215, 140, 130), 24, 30, 4),
]
DUMMY_POS = (17, 10)

#: Death animations are handed out round-robin at spawn rather than at random,
#: so a single fight is guaranteed to show several different ones.
DEATH_ORDER = ["spin", "topple", "launch", "melt", "burst"]

#: How long an enemy waits between acting, and how long the player is locked
#: out after acting. One shared pace keeps the raw-sim mode honest -- juice off
#: changes how a turn *looks*, never how fast the game runs.
TURN_TIME = MOVE_TIME


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
     "Flat on take-off and landing, drawn out at the apex. Get the sign "
     "backwards and the two phases cancel into a jitter you cannot see."),
    ("jelly", "slime drag", "movement",
     "Secondary motion: a loose spring chases the body, so it pours into a "
     "move, stretches on the way and wobbles to a stop by itself."),
    ("bob", "idle breathing", "movement",
     "A slow scale wobble so a standing figure is not a dead pixel. Nobody notices it until it is gone."),
    ("ghost", "afterimage trail", "movement",
     "Faded echoes of the last few frames. Sells speed on a fast move, noise on a slow one."),
    ("dust", "landing dust", "movement",
     "A puff of floor kicked sideways at the end of a hop. Tells you the feet "
     "touched something, and gives the landing squash a reason."),
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
     "A six-frame sweep around the target tile. The eye fills in a sword that is never drawn."),
    ("cut", "slash cut", "attack",
     "The edge going through rather than around: a straight line that wipes on "
     "fast and holds. The arc says a swing happened, the cut says it landed."),
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

    def __init__(self, sprite: str, x: int, y: int, color, name: str, hp: int = 5,
                 *, tint: bool = True, invincible: bool = False,
                 stationary: bool = False, death: str = "spin"):
        self.sprite = sprite            # key into tiles.SPRITES
        self.color = color              # tint, and the colour of its sparks
        self.tint = tint                # False for art that is already coloured
        self.name = name
        self.body = Body(tx=float(x), ty=float(y))
        self.anim = Animator()
        self.trail = GhostTrail()
        self.hp = hp
        self.max_hp = hp
        # Invincible things still take the full hit *presentation* -- flash,
        # knockback, sparks, screen shake -- and simply never run out of health.
        # That is the whole point of a bench: you want to feel being hit.
        self.invincible = invincible
        self.stationary = stationary    # true for the training dummy
        self.death = death              # which exit animation it plays
        self.dying = False
        self.death_timer = 0.0
        # Seeded off the position so a row of dummies does not breathe in step.
        # Persistent, so it is gated by `anim.persistent_enabled` rather than by
        # declining to play it -- see World.update.
        self.anim.add_persistent(Breathe(phase=(x * 3 + y * 7) % 7))
        # Secondary motion, so it runs after the hop rather than alongside it.
        # A slime is looser and heavier than everything else on the board.
        loose = sprite == "slime"
        # Only the spring differs by creature: the slime is softer and less
        # damped, so it lags further and rings longer, and the deformation
        # follows from that rather than being dialled up separately.
        self.jelly = Jelly(stiffness=78.0 if loose else 120.0,
                           damping=6.0 if loose else 9.0,
                           drag=0.7 if loose else 0.55)
        # Kept so the intensity dial can scale the amplitudes every frame. The
        # stiffness and damping are *not* scaled -- those set the character of
        # the wobble, and the dial is for how big everything is, not how it
        # behaves.
        self.jelly_base = (self.jelly.drag, self.jelly.stretch)
        self.anim.add_post(self.jelly)

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
    """A walled 40x20 arena with pillars and a few short walls.

    The cover is not decoration: without something to walk around, the enemy
    pathfinding is a straight line and there is nothing to see. The blocks are
    placed so most approaches to the middle have a corner in them.
    """
    grid = [[0] * GRID_W for _ in range(GRID_H)]
    for x in range(GRID_W):
        grid[0][x] = grid[GRID_H - 1][x] = 1
    for y in range(GRID_H):
        grid[y][0] = grid[y][GRID_W - 1] = 1

    for (px, py) in [(6, 4), (6, 15), (12, 7), (12, 12), (28, 7), (28, 12),
                     (33, 4), (33, 15), (24, 3), (24, 16)]:
        grid[py][px] = 1
    # Short walls, each with an open end, so there is somewhere to path around.
    for x in range(9, 14):
        grid[3][x] = 1
    for x in range(26, 31):
        grid[16][x] = 1
    for y in range(7, 13):
        grid[y][14] = 1
    for y in range(6, 11):
        grid[y][30] = 1
    # The middle is deliberately left clear: the player spawns there and every
    # neighbouring tile has to be walkable, or the first step of a fresh bench
    # is a wall bump.
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
        self.turn = 0
        self.turn_cooldown = 0.0           # seconds until the player may act
        self.goal_map: list[list] = []     # distance-to-player, rebuilt per turn
        self.player = None
        self.reset()

    # -- setup ------------------------------------------------------------
    def reset(self):
        """Respawn the monsters, leaving the player exactly where they stand.

        Resetting used to teleport you back to the middle, which is wrong for a
        bench: you pick a spot with a good view of an effect, hit reset for a
        fresh set of targets, and you want to still be looking at it.
        """
        first_run = getattr(self, "player", None) is None
        if first_run:
            self.player = Entity("player", GRID_W // 2, GRID_H // 2, PLAYER_COLOR,
                                 "you", hp=99, tint=False, invincible=True)
        else:
            self.player.anim.clear()
            self.player.trail.clear()
            self.player.body.reset_juice()

        px, py = self.player.tile
        self.entities = [self.player]
        for kind, name, color, hp, x, y in MONSTERS:
            if (x, y) == (px, py) or self.blocked(x, y):
                continue
            self.entities.append(Entity(kind, x, y, color, name, hp=hp,
                                        death=DEATH_ORDER[len(self.entities) %
                                                          len(DEATH_ORDER)]))
        # The punching bag: never moves, never dies, always tells you the number.
        if (DUMMY_POS) != (px, py) and not self.blocked(*DUMMY_POS):
            self.entities.append(
                Entity("dummy", *DUMMY_POS, (190, 150, 110), "training dummy",
                       hp=999, invincible=True, stationary=True))

        self.fx.clear()
        self.log.clear()
        self.turn_cooldown = 0.0
        self.rebuild_goal_map()
        if first_run:
            cx, cy = self.player.tile_center()
            self.camera.snap(cx, cy)
            self.clamp_camera()
        self.say("walk into something to hit it", ACCENT)

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

    def rebuild_goal_map(self):
        """Breadth-first flood outward from the player: a Dijkstra goal map.

        Every enemy needs a route to the same place, so pathing them one at a
        time with A* would solve the same problem over and over. One flood fill
        costs a single pass over 800 cells and then *every* enemy's move is a
        look at four neighbours and a comparison -- and it is naturally
        cooperative, since walking downhill from different starts spreads them
        out around corners instead of stacking them into a queue.

        Unreachable cells keep `None`, which is how an enemy behind a sealed
        wall knows to stand still rather than jitter against it.
        """
        self.goal_map = [[None] * GRID_W for _ in range(GRID_H)]
        sx, sy = self.player.tile
        if self.blocked(sx, sy):
            return
        self.goal_map[sy][sx] = 0
        frontier = deque([(sx, sy)])
        while frontier:
            x, y = frontier.popleft()
            d = self.goal_map[y][x] + 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if self.blocked(nx, ny) or self.goal_map[ny][nx] is not None:
                    continue
                self.goal_map[ny][nx] = d
                frontier.append((nx, ny))

    def step_toward_player(self, e: Entity):
        """One downhill step on the goal map. True if it moved or attacked."""
        x, y = e.tile
        here = self.goal_map[y][x]
        if here is None:
            return False

        best = None
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            d = None if self.blocked(nx, ny) else self.goal_map[ny][nx]
            if d is None or d >= here:
                continue
            occupant = self.entity_at(nx, ny)
            if occupant is self.player:
                e.body.facing = (dx, dy)
                self.attack(e, self.player, dx, dy)
                return True
            if occupant is not None:
                continue          # another monster has the tile; wait a turn
            if best is None or d < best[0]:
                best = (d, dx, dy)

        if best is None:
            return False
        self.move_entity(e, best[1], best[2])
        return True

    def take_enemy_turns(self):
        """Everything that is not the player gets one action."""
        for e in list(self.entities):
            if e is self.player or e.dying or e.stationary:
                continue
            self.step_toward_player(e)

    def nearest_enemy(self, *, killable: bool = False, mobile: bool = False):
        """The closest other entity, optionally filtered.

        The filters matter now that not everything on the board is the same
        kind of thing: the kill shortcut must skip the invincible training
        dummy (or it silently does nothing), and the "hit me" key must skip it
        too, since a punching bag that punches back is not a punching bag.
        """
        others = [e for e in self.entities
                  if e is not self.player and not e.dying
                  and not (killable and e.invincible)
                  and not (mobile and e.stationary)]
        if not others:
            return None
        px, py = self.player.tile
        return min(others, key=lambda e: (e.tile[0] - px) ** 2 + (e.tile[1] - py) ** 2)

    # -- actions ----------------------------------------------------------
    def move_entity(self, e: Entity, dx: int, dy: int):
        """Commit a move and attach whatever animation the juice asks for.

        Note the order: the *logical* move happens immediately and completely,
        and only then is an animation attached to drag the picture back to
        where the entity used to be. The sim never waits for a tween -- which
        is also why the enemies can path with no idea that animation exists.
        """
        e.body.facing = (dx, dy)
        e.body.tx += dx                              # instant, authoritative
        e.body.ty += dy

        j = self.juice
        if not j.on("tween"):
            # The raw sim: the sprite is simply somewhere else now. This is the
            # baseline every other movement effect is an argument against.
            return
        e.anim.clear()
        if j.on("hop"):
            e.anim.play(Hop(duration=MOVE_TIME, dx=dx, dy=dy,
                            height=j.amt(0.34),
                            squash=j.amt(0.30) if j.on("squash") else 0.0,
                            ease=EASINGS[j.move_ease]))
            if j.on("dust"):
                # Fired at the end of the hop rather than the start, so it is
                # the landing that kicks up floor and not the take-off.
                e.anim.play(Sequence([Wait(duration=MOVE_TIME * 0.86),
                                      Callback(lambda e=e: self.landing_dust(e))]))
        else:
            e.anim.play(Slide(duration=MOVE_TIME, dx=dx, dy=dy,
                              ease=EASINGS[j.move_ease]))

    def try_move(self, e: Entity, dx: int, dy: int):
        """The player's turn: walk, or attack whatever is in the way."""
        if e.anim.busy or self.hitstop.frozen or self.turn_cooldown > 0.0:
            return
        e.body.facing = (dx, dy)
        nx, ny = e.tile[0] + dx, e.tile[1] + dy

        target = self.entity_at(nx, ny)
        if target is not None and target is not e:
            self.attack(e, target, dx, dy)
        elif self.blocked(nx, ny):
            self.bump_wall(e, dx, dy)
            return                       # a move into a wall costs no turn
        else:
            self.move_entity(e, dx, dy)
        self.end_player_turn()

    def end_player_turn(self):
        """Everything else gets to act, and the player is locked out briefly."""
        self.turn += 1
        self.turn_cooldown = TURN_TIME
        self.rebuild_goal_map()
        self.take_enemy_turns()

    def landing_dust(self, e: Entity):
        """A puff kicked out sideways from under the feet.

        Thrown left and right along the floor rather than upwards, because dust
        that rises reads as smoke. Slow, low gravity and high drag, so it
        billows and stalls instead of arcing away like a spark.
        """
        x, y = e.tile_center()
        for direction in ((-1.0, -0.25), (1.0, -0.25)):
            self.fx.particles.burst(
                x, y + TILE * 0.42, count=5, direction=direction, spread=1.1,
                speed=(70 * PX, 210 * PX), life=(0.25, 0.5), size=5.0 * PX,
                colors=((132, 140, 166), (108, 116, 142), (92, 99, 124)),
                gravity=120.0 * PX, drag=4.2)

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
        damage = 2 + (abs(hash((target.name, target.hp, self.turn))) % 4)
        if not target.invincible:
            target.hp -= damage
        killing = target.hp <= 0 and not target.invincible

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
            # Deliberately short-ranged: the wave front covers about three
            # tiles before it dies. A ripple that crosses the whole arena reads
            # as an earthquake rather than as a blow landing here, and it makes
            # the renderer repaint the entire floor for a displacement of
            # nothing.
            self.fx.impact(cx, cy, strength=j.amt(7.5 * PX * heavy), life=0.42,
                           speed=270.0 * PX, wavelength=46.0 * PX)
        if j.on("slash"):
            self.fx.slash(tx - dx * TILE * 0.25, ty - dy * TILE * 0.25,
                          angle=math.degrees(math.atan2(-dy, dx)),
                          radius=TILE * 0.62, life=0.16)
        if j.on("cut"):
            # Across the blow rather than along it, and canted, because a cut
            # square to the attack direction reads as a wall rather than a swing.
            self.fx.cut(tx, ty, angle=math.degrees(math.atan2(-dy, dx)) + 118.0,
                        length=TILE * 1.5, thickness=6.0 * PX * heavy, life=0.15)

        if killing:
            self.kill(target)
        elif target is self.player:
            self.say(f"you shrug off {damage}", ACCENT)
        else:
            self.say(f"{target.name} takes {damage}", target.color)

    def kill(self, target: Entity):
        if target.invincible:
            return                       # the training dummy outlives everyone
        target.dying = True
        target.anim.clear()
        j = self.juice
        if j.on("death"):
            # Which exit it plays was decided at spawn, so a fight shows a
            # spread of them rather than the same one five times.
            facing = 1.0 if target.body.tx >= self.player.body.tx else -1.0
            motion = DEATHS[target.death](facing)
            target.anim.play(motion)
            target.death_timer = motion.duration
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
        d = self.nearest_enemy(mobile=True)
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
        if p.anim.busy or self.hitstop.frozen or self.turn_cooldown > 0.0:
            return
        dx, dy = p.body.facing
        target = self.entity_at(p.tile[0] + dx, p.tile[1] + dy)
        if target is not None and target is not p:
            self.attack(p, target, dx, dy)
            self.end_player_turn()
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
        self.end_player_turn()

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
        wobbling = self.juice.on("jelly")
        for e in list(self.entities):
            was_busy = e.anim.busy
            e.anim.persistent_enabled = breathing
            # The jelly spring keeps integrating either way -- see Jelly -- so
            # switching it on mid-stride does not snap the body across a lag it
            # accumulated while nobody was looking.
            e.jelly.enabled = wobbling and not e.dying
            e.jelly.drag, e.jelly.stretch = (v * self.juice.intensity
                                             for v in e.jelly_base)
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
        # Ticks on the real clock, not the frozen one, so hit-stop does not
        # silently lengthen the turn.
        self.turn_cooldown = max(0.0, self.turn_cooldown - dt)

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
        self.sheet = tiles.SpriteSheet(TILE)
        self._floor = None          # baked on first draw; the arena is static
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

    def sprite_for(self, e: Entity):
        """The resting sprite for an entity, tinted and cached by the sheet."""
        col, row = tiles.SPRITES.get(e.sprite, tiles.SPRITES["goblin"])
        return self.sheet.sprite(col, row, e.color if e.tint else None)

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
        self.draw_cuts(world)
        self.draw_particles(world)
        self.draw_floaters(world)

    def _tile_rect(self, cx: float, cy: float) -> pygame.Rect:
        rect = pygame.Rect(0, 0, TILE - 2, TILE - 2)
        rect.center = (int(cx), int(cy))
        return rect

    def _paint_tile(self, surf, world: World, tx: int, ty: int, cx: float, cy: float):
        rect = self._tile_rect(cx, cy)
        if world.grid[ty][tx]:
            pygame.draw.rect(surf, WALL, rect, border_radius=3)
            pygame.draw.rect(surf, WALL_TOP, rect.inflate(-8, -8), border_radius=2)
        else:
            shade = FLOOR if (tx + ty) % 2 == 0 else FLOOR_ALT
            pygame.draw.rect(surf, shade, rect, border_radius=3)

    def _bake_floor(self, world: World):
        """Draw the whole 40x20 arena once, into one surface.

        Repainting 500-odd rounded rects every frame cost more than every
        sprite, particle and slash in the bench put together. The arena never
        changes, so it is baked once and the frame becomes a single blit; only
        the handful of tiles a ripple is passing through get redrawn by hand.
        """
        surf = pygame.Surface((GRID_W * TILE, GRID_H * TILE)).convert()
        surf.fill(BG)
        for ty in range(GRID_H):
            for tx in range(GRID_W):
                self._paint_tile(surf, world, tx, ty,
                                 (tx + 0.5) * TILE, (ty + 0.5) * TILE)
        return surf

    def draw_floor(self, world: World):
        """The baked arena, plus any tiles an impact wave is shoving about.

        The ripple is the effect people forget. Shaking the camera moves
        everything together, which the eye reads as the *camera* being hit.
        Moving tiles against each other is the only way the floor itself takes
        part -- so where a wave is passing, the baked copy is painted out and
        those tiles are redrawn at their displaced positions.
        """
        v = self.view
        if self._floor is None:
            self._floor = self._bake_floor(world)
        v.blit(self._floor, (int(VIEW_W * 0.5 - world.camera.x),
                             int(VIEW_H * 0.5 - world.camera.y)))

        rippling = world.juice.on("ripple") and world.fx.impacts
        if not rippling:
            return
        # A ripple only touches the tiles its wave front has reached, so the
        # union of the live impacts' reach is computed once here and used to
        # skip the per-tile displacement call for everything outside it. On a
        # 40x20 arena that is the difference between 475 calls a frame and a
        # few dozen.
        # A wave only touches the tiles its front has reached, so the union of
        # the live impacts bounds the repaint to a few dozen tiles.
        box = None
        for imp in world.fx.impacts:
            reach = imp.speed * imp.t + imp.wavelength * 1.5 + TILE
            r = pygame.Rect(imp.x - reach, imp.y - reach, reach * 2, reach * 2)
            box = r if box is None else box.union(r)

        x0 = max(0, int(box.left // TILE))
        x1 = min(GRID_W, int(box.right // TILE) + 1)
        y0 = max(0, int(box.top // TILE))
        y1 = min(GRID_H, int(box.bottom // TILE) + 1)
        if x1 <= x0 or y1 <= y0:
            return

        for ty in range(y0, y1):
            for tx in range(x0, x1):
                wx, wy = (tx + 0.5) * TILE, (ty + 0.5) * TILE
                ox, oy = world.fx.tile_offset(wx, wy)
                if abs(ox) < 0.4 and abs(oy) < 0.4:
                    continue          # the baked copy is already correct here
                # Paint out the baked tile first, or the displaced one would be
                # drawn on top of a stationary twin of itself.
                sx, sy = self.to_view(world, wx, wy)
                v.fill(BG, self._tile_rect(sx, sy).inflate(4, 4))
                sx, sy = self.to_view(world, wx + ox, wy + oy)
                self._paint_tile(v, world, tx, ty, sx, sy)

    def draw_entities(self, world: World):
        for e in world.entities:
            self.draw_trail(world, e)
            self.draw_body(world, e)
            # No bar on things that cannot lose the fight: the player and the
            # training dummy are both invincible, and a full bar that never
            # moves is just clutter.
            if not e.dying and not e.invincible:
                self.draw_hp(world, e)

    def _transform(self, base, sx: float, sy: float, angle: float):
        """Squash, stretch and spin one sprite.

        Nearest-neighbour throughout -- `scale` and `rotate`, never
        `smoothscale` or `rotozoom`. Pixel art put through a smooth filter goes
        to mush, and squash-and-stretch resamples the sprite on every single
        frame, so a smooth path would leave a monster blurry for the entire
        length of every hop.

        The non-uniform scale has to happen before the rotation, or a squashed
        sprite would end up squashed along the *screen* axes instead of its own.
        """
        img = base
        if abs(sx - 1.0) > 0.005 or abs(sy - 1.0) > 0.005:
            w = max(1, int(base.get_width() * sx))
            h = max(1, int(base.get_height() * sy))
            img = pygame.transform.scale(base, (w, h))
        if abs(angle) > 0.1:
            img = pygame.transform.rotate(img, angle)
        return img

    def draw_trail(self, world: World, e: Entity):
        base = self.sprite_for(e)
        for g in e.trail.ghosts:
            img = self._transform(base, g.sx, g.sy, g.angle).copy()
            img.set_alpha(int(110 * g.alpha))
            sx, sy = self.to_view(world, g.x, g.y)
            self.view.blit(img, img.get_rect(center=(int(sx), int(sy))))

    def _blit_deformed(self, img, center, shear, axis: int):
        """Blit a sprite in bands, each slid by its own amount.

        This is the only place the renderer draws something that is not a rigid
        sprite, and the whole difficulty is that bands which have slid apart
        leave gaps of background *through* the creature -- which reads as the
        sprite falling to pieces rather than stretching.

        So each band is drawn wide enough to reach its neighbour: its own share
        plus however far the two have slid apart. The extra pixels are the
        neighbour's, redrawn at this band's offset, so a stretched body smears
        along its own edge instead of tearing. Bands are drawn back to front,
        and each one covers the overspill of the one before it.
        """
        w, h = img.get_size()
        n = len(shear)
        span = w if axis == 0 else h
        left = center[0] - w * 0.5
        top = center[1] - h * 0.5
        for i, off in enumerate(shear):
            lo = (i * span) // n
            size = max(1, ((i + 1) * span) // n - lo)
            if i < n - 1:
                size += 1 + int(abs(shear[i + 1] - off) * TILE)
            slide = off * TILE
            if axis == 0:
                self.view.blit(img, (int(left + lo + slide), int(top)),
                               pygame.Rect(lo, 0, size, h))
            else:
                self.view.blit(img, (int(left), int(top + lo + slide)),
                               pygame.Rect(0, lo, w, size))

    def draw_body(self, world: World, e: Entity):
        b = e.body
        img = self._transform(self.sprite_for(e), b.sx, b.sy, b.angle)

        if b.flash > 0.0 or b.alpha < 1.0:
            img = img.copy()
            if b.flash > 0.0:
                # Adding to RGB but not to alpha whitens the sprite while
                # leaving its silhouette exactly as it was.
                white = img.copy()
                white.fill((255, 255, 255, 0), special_flags=pygame.BLEND_RGBA_ADD)
                white.set_alpha(int(255 * clamp(b.flash)))
                img.blit(white, (0, 0))
            if b.alpha < 1.0:
                img.set_alpha(int(255 * clamp(b.alpha)))

        wx, wy = e.world_pos()
        sx, sy = self.to_view(world, wx, wy)
        if b.shear:
            self._blit_deformed(img, (sx, sy), b.shear, b.shear_axis)
        else:
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

    def _stamp(self, shapes):
        """Draw translucent polygons onto the view via the scratch layer.

        The scratch surface exists because pygame cannot draw a *translucent*
        polygon straight onto an opaque surface. The trap is that clearing the
        whole 800x600 scratch for every slash costs more than everything else
        in the frame put together -- so only the shape's own bounding box is
        cleared and blitted, which is typically a couple of tiles' worth.
        """
        bounds = None
        for pts, _ in shapes:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            r = pygame.Rect(int(min(xs)) - 2, int(min(ys)) - 2,
                            int(max(xs) - min(xs)) + 5, int(max(ys) - min(ys)) + 5)
            bounds = r if bounds is None else bounds.union(r)
        if bounds is None:
            return
        bounds = bounds.clip(self.scratch.get_rect())
        if bounds.w <= 0 or bounds.h <= 0:
            return
        self.scratch.fill((0, 0, 0, 0), bounds)
        for pts, color in shapes:
            pygame.draw.polygon(self.scratch, color, pts)
        self.view.blit(self.scratch, bounds.topleft, bounds)

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
            self._stamp([(outer + inner[::-1], (*s.color, a))])

    def draw_cuts(self, world: World):
        """The straight cut: a lens tapering to a point at both ends.

        Drawn twice -- a wide coloured body and a thin white core -- which is
        the cheapest way to make a flat polygon look like it is glowing.
        """
        for c in world.fx.cuts:
            a = int(255 * c.alpha)
            if a <= 3:
                continue
            (tx, ty), (hx, hy) = c.endpoints()
            tail = self.to_view(world, tx, ty)
            head = self.to_view(world, hx, hy)
            dx, dy = head[0] - tail[0], head[1] - tail[1]
            span = math.hypot(dx, dy)
            if span < 2.0:
                continue
            # Unit normal, to give the lens its width.
            nx, ny = -dy / span, dx / span
            shapes = []
            for width, color in ((c.thickness, c.color),
                                 (c.thickness * 0.35, (255, 255, 255))):
                pts = []
                steps = 9
                for i in range(steps + 1):          # one edge out...
                    t = i / steps
                    w = math.sin(t * math.pi) ** 0.6 * width
                    pts.append((tail[0] + dx * t + nx * w, tail[1] + dy * t + ny * w))
                for i in range(steps, -1, -1):      # ...and the other back
                    t = i / steps
                    w = math.sin(t * math.pi) ** 0.6 * width
                    pts.append((tail[0] + dx * t - nx * w, tail[1] + dy * t - ny * w))
                shapes.append((pts, (*color, a)))
            self._stamp(shapes)

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

        if abs(roll) > 0.25 or abs(zoom - 1.0) > 0.004:
            # rotozoom allocates and resamples a whole new surface, so it is
            # kept off the path entirely on frames where neither is actually in
            # play. The thresholds are set at the point where the roll and the
            # zoom stop being visible rather than at zero, which skips the long
            # decaying tail of every shake -- most of the frames, in a fight.
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
                    rect = pygame.Rect(cx - 4, y - 2, col_w + 8, 16)
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
                    y += 16
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
        d = world.nearest_enemy(killable=True)
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

    # The player starts mid-arena and the training dummy stands at DUMMY_POS;
    # walk over and hit it, which is reproducible because the dummy is the one
    # thing on the board that never moves.
    px, py = world.player.tile
    tx, ty = DUMMY_POS
    while (px, py) != (tx + 1, ty):
        dx = (tx + 1 > px) - (tx + 1 < px)
        dy = 0 if dx else (ty > py) - (ty < py)
        if not dx and not dy:
            break
        world.try_move(world.player, dx, dy)
        step(12)
        if world.player.tile == (px, py):
            break                             # blocked; near enough for a frame
        px, py = world.player.tile
    world.try_move(world.player, -1, 0)        # walks into the dummy -> attack
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
