"""A juice workbench: every trick that makes a tile-based roguelike feel good,
one toggle at a time -- and now one slider at a time as well.

A grid roguelike is the hardest kind of game to make feel physical. Everything
happens on integer coordinates, one discrete step at a time, with no momentum
and no sub-tile positions -- the sim is a spreadsheet that occasionally changes
a cell. "Juice" is the layer of lies you tell on top of that spreadsheet so it
reads as bodies moving through a room.

This is a bench for looking at those lies individually. There is a room, a
player, a cast of monsters and a training dummy, and a scrolling panel down the
right listing every effect the bench knows. Click a row to switch it off and
on; drag a slider to find the amplitude at which it stops helping and starts
hurting. That second question is the more useful one -- almost every effect
here is good at some size and ridiculous at twice it, and the only way to find
the line is to drag past it.

The point of the bench is not the demo, it is the *comparison*: turn hit-stop
off and hit a dummy, turn it back on and hit it again. Tab is a master switch
for exactly that, so you can flip the whole thing between "raw sim" and
"juiced" mid-swing.

The effects, grouped as they are in the panel:

    movement   hop arcs, squash and stretch, slime drag, banking, idle
               breathing and fidgets, turn-to-face, afterimages, landing dust,
               blob shadows, trailing tails, and a bounce off walls
    attack     anticipation, the lunge, hit-stop, hit flash, knockback,
               a shiver, a weapon arc, a cut, crits, weight, deaths, corpses
    camera     trauma shake, directional kick, zoom punch, roll tilt, follow
               lag, look-ahead and gamepad rumble
    world      particles, damage numbers, shockwaves, a floor ripple, blood
               decals and ambient sway
    screen     screen flash, vignette, RGB split, bloom, lighting with cast
               shadows, lit sprites, outlines, heat haze, and the
               pixel-snapping mode
    feel       input buffering and turn pacing -- juice with no pixels in it,
               and the two that matter most
    sound      layered impacts with pre-rendered pitch variation, and ducking

None of the maths lives here. It is in `juicefx.py`, which has no pygame in it
at all, and `audiofx.py`, whose synthesis and pitch machinery are numpy and are
tested with no audio device. `juicetest.py` exercises both with nothing open.
This file is the tileset, the input, the panel and the compositing, which is
the split that matters: an effect you cannot test without opening a window is
an effect you will never tune.

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
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tiles  # noqa: E402
from juicefx import (  # noqa: E402
    ATTACK_TIME, ATTACK_WINDUP, DEATHS, EASINGS, FLASH_TIME,
    Animator, Anticipate, Body, Breathe, Callback, Camera, ChipBar, Corpse,
    DecalField, EffectField, FaceFlip, Fidget, Flash, GhostTrail, HitStop, Hop,
    InputBuffer, Jelly, Knockback, Lean, Lunge, MOVE_TIME, Parallel, Params,
    RumbleMap, Sequence, Shiver, Slide, Spring, Tail, Trauma, TurnPacer, Wait,
    ambient_offset, clamp, hash01, lerp, shadow_of, tier_for,
)

try:
    import audiofx
    HAVE_AUDIO = True
except Exception:                                    # pragma: no cover
    audiofx = None                                   # type: ignore
    HAVE_AUDIO = False

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
# The panel is a single scrolling column now. Two columns were what kept the
# window short when there were thirty rows; there are over ninety, so no amount
# of column juggling fits them and the list has to scroll instead. One column
# is then simply easier to read, and leaves room for a slider's label, track
# and value on one line.
PANEL_W = 380
WIN_W, WIN_H = VIEW_W + PANEL_W, VIEW_H     # 1180x600
# Panel space reserved for the hover blurb / control list.
FOOTER_H = 132
HEADER_H = 30

#: The control list, shown whenever the mouse is not over a row.
HELP_LINES = [
    "click a row to toggle   drag a slider",
    "right-click a slider to reset it",
    "wheel scrolls this panel",
    "",
    "wasd / arrows  move, walk into a thing",
    "space swing   x get hit   K kill   r reset",
    "Tab A/B all   F1/F2 on/off   F3 defaults",
    "[ ] move easing   p pixel mode   m music",
    "- = intensity     esc quit",
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
TRACK_BG = (30, 32, 42)
TRACK_FILL = (44, 78, 74)
INK = (208, 214, 230)
DIM = (110, 118, 140)
ACCENT = (120, 220, 190)
WARN = (240, 130, 110)
GOLD = (250, 205, 120)
BLOOD = (150, 40, 48)

PLAYER_COLOR = (225, 235, 250)

#: The roster: (sprite key, name, tint, hp, weight, spawn x, spawn y).
#: Health is deliberately generous -- at four points a monster died to the
#: first or second blow, which meant you spent the whole session watching death
#: animations and almost never saw a hit *land* on something still standing.
#: Weight divides the knockback: a raven is thrown across the room by a blow
#: that barely rocks the ogre, from one number and no special cases.
MONSTERS = [
    ("goblin", "goblin", (150, 200, 130), 14, 1.0, 13, 5),
    ("goblin", "goblin", (150, 200, 130), 14, 1.0, 27, 15),
    ("lizard", "lizard", (120, 200, 190), 16, 0.9, 24, 4),
    ("skeleton", "skeleton", (225, 225, 210), 12, 0.7, 9, 14),
    ("wolf", "wolf", (200, 160, 130), 12, 1.1, 33, 9),
    ("wolf", "wolf", (200, 160, 130), 12, 1.1, 6, 9),
    ("raven", "raven", (170, 150, 220), 10, 0.4, 22, 16),
    ("slime", "slime", (140, 220, 160), 18, 1.4, 35, 13),
    ("ogre", "ogre", (215, 140, 130), 24, 2.6, 30, 4),
]
DUMMY_POS = (17, 10)

#: Which creatures drag something behind them. A tail or a cape is the most
#: legible follow-through there is, because it is a part that is visibly late.
TAILED = {
    "player": (5, 0.13, (150, 90, 110)),      # a cape
    "wolf": (4, 0.16, (150, 120, 96)),
    "lizard": (5, 0.15, (90, 150, 142)),
    "raven": (3, 0.14, (120, 106, 158)),
    "slime": (3, 0.13, (104, 165, 120)),
}

#: Death animations are handed out round-robin at spawn rather than at random,
#: so a single fight is guaranteed to show several different ones.
DEATH_ORDER = ["spin", "topple", "launch", "melt", "burst"]


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


class Choice:
    """A setting with three or four named states rather than two.

    Pixel snapping is the reason this type exists: "smooth or not" is not the
    question, because there are three defensible answers and the interesting
    one is in the middle.
    """

    __slots__ = ("key", "label", "group", "blurb", "options", "index")

    def __init__(self, key, label, group, blurb, options, index=0):
        self.key = key
        self.label = label
        self.group = group
        self.blurb = blurb
        self.options = options
        self.index = index

    @property
    def value(self) -> str:
        return self.options[self.index]

    def cycle(self, step: int = 1) -> str:
        self.index = (self.index + step) % len(self.options)
        return self.value


#: Order here is the order they appear in the panel.
#: (key, label, group, blurb) or (..., default_on)
TOGGLE_SPECS = [
    # -- movement ---------------------------------------------------------
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
     "Ten springs across the sprite with a stiffness gradient, so the leading "
     "edge outruns the trailing one and the body strings out as it moves."),
    ("lean", "lean into moves", "movement",
     "Two or three degrees of bank in the travel direction, with a counter-"
     "swing on the way out. Invisible still, unmistakable in motion."),
    ("bob", "idle breathing", "movement",
     "A slow scale wobble so a standing figure is not a dead pixel. Nobody "
     "notices it until it is gone."),
    ("fidget", "idle fidgets", "movement",
     "An occasional twitch on a jittered timer. Continuous motion becomes "
     "wallpaper; an event does not."),
    ("faceflip", "turn to face", "movement",
     "Squash through zero width to change facing, and swap the sprite at the "
     "pinch. A one-frame flip reads as a glitch."),
    ("shadow", "blob shadow", "movement",
     "An ellipse that shrinks and fades as the body rises. The cheapest height "
     "cue there is -- without it a hop is ambiguous with getting bigger."),
    ("tail", "tails & capes", "movement",
     "A spring chain dragged behind the body. Follow-through on a *part*, "
     "which is what animators reach for before deforming the whole shape."),
    ("ghost", "afterimage trail", "movement",
     "Faded echoes of the last few frames. Sells speed on a fast move, noise "
     "on a slow one."),
    ("dust", "landing dust", "movement",
     "A puff of floor kicked sideways at the end of a hop. Tells you the feet "
     "touched something, and gives the landing squash a reason."),
    ("wallbump", "wall bump", "movement",
     "A failed move still needs an answer. Bounce off and shiver, no damage."),

    # -- attack -----------------------------------------------------------
    ("windup", "anticipation", "attack",
     "Lean away before striking. Costs 90ms and does more for weight than any "
     "particle."),
    ("lunge", "lunge", "attack",
     "Drive at the target and ease back. Fast out, slow home -- the asymmetry "
     "is the effect."),
    ("hitstop", "hit-stop", "attack",
     "Freeze everything for 3-7 frames on contact. The strongest effect here "
     "per line of code."),
    ("hitflash", "hit flash", "attack",
     "Whiten the victim. Often the only hit confirmation that is legible at "
     "tile size."),
    ("knockback", "knockback", "attack",
     "Shove the victim and let it ease home. Without it the attacker punches "
     "a poster."),
    ("weight", "weight matters", "attack",
     "Divide the reaction by the target's mass. One number turns one knockback "
     "into nine: the raven flies, the ogre barely rocks."),
    ("crit", "critical hits", "attack",
     "Tier the feedback. A crit is the same hit with more of everything -- "
     "louder, brighter, longer freeze -- not a different one."),
    ("shiver", "victim shiver", "attack",
     "Jitter the body it landed on, decaying fast. Reads as the blow ringing "
     "through."),
    ("slash", "weapon arc", "attack",
     "A six-frame sweep around the target tile. The eye fills in a sword that "
     "is never drawn."),
    ("cut", "slash cut", "attack",
     "The edge going through rather than around: a straight line that wipes on "
     "fast and holds. The arc says a swing happened, the cut says it landed."),
    ("chip", "chip damage bar", "attack",
     "A pale ghost bar left where the health was, held and then drained. Shows "
     "*how much* was taken as a width, instead of a bar that is merely shorter "
     "than the last time you looked."),
    ("death", "death animation", "attack",
     "Spin, shrink, fall, fade. A thing that vanishes instantly leaves you "
     "unsure you hit it."),
    ("corpse", "corpses remain", "attack",
     "Leave the body on the floor afterwards. A beautiful death that then "
     "blinks out has undone its own work."),

    # -- camera -----------------------------------------------------------
    ("shake", "screen shake", "camera",
     "Trauma budget, displacement = trauma squared. Everything adds trauma, "
     "nothing sets offset."),
    ("kick", "directional kick", "camera",
     "Punch the whole frame along the blow and spring back. Gives the hit a "
     "direction."),
    ("zoom", "zoom punch", "camera",
     "A damped spring on scale. Tiny numbers -- three percent is plenty at "
     "this size."),
    ("tilt", "roll tilt", "camera",
     "A degree or two of roll on impact. Use sparingly, it makes people "
     "seasick."),
    ("lag", "camera lag", "camera",
     "Follow with an exponential delay so the player can pull away from centre "
     "and drift back."),
    ("lead", "look-ahead", "camera",
     "Bias the camera along the way you are facing, so you see more of where "
     "you are going than of where you have been."),
    ("rumble", "gamepad rumble", "camera",
     "The same trauma number, sent to the motors. Available in pygame since "
     "2.0.2 and almost always forgotten. No pad, no effect."),

    # -- world ------------------------------------------------------------
    ("particles", "particles", "world",
     "A cone of sparks with varied speed and drag. One speed for all of them "
     "reads as a cartwheel."),
    ("numbers", "damage numbers", "world",
     "Pop in on ease-out-back, rise, fall away. Reads the amount without a log "
     "line."),
    ("shockwave", "shockwave ring", "world",
     "An expanding ring, most of it in the first three frames. Force leaving a "
     "point."),
    ("ripple", "floor ripple", "world",
     "A wave through the tiles themselves, so the grid is a surface and not "
     "wallpaper."),
    ("decals", "blood decals", "world",
     "Marks that stay. Everything else here is an event; a decal is evidence, "
     "and a room you have fought through should look like one."),
    ("ambient", "ambient sway", "world",
     "Tufts and torches drifting on a slow wave. Not an effect on anything -- "
     "just proof the room is not a screenshot."),

    # -- screen -----------------------------------------------------------
    ("scrflash", "screen flash", "screen",
     "A frame or two of tint over everything. Save it for the big ones."),
    ("vignette", "vignette pulse", "screen",
     "Darkened corners that clamp down on impact. Focuses the eye at the "
     "centre."),
    ("rgbsplit", "RGB split", "screen",
     "Separate the colour channels for a few frames. Watch the ms readout -- "
     "on the CPU this one effect is most of a 60Hz frame."),
    ("bloom", "bloom", "screen",
     "Downscale, threshold, upscale, add. Pygame has no shaders, but a box "
     "blur out of two smoothscales is a perfectly good glow."),
    ("light", "lighting", "screen",
     "A light map multiplied over the frame, with a light at every torch and "
     "one on each hit. The flash lights the room instead of only the victim. "
     "Walls block it and bodies cast into it -- see the four sliders under "
     "this row. Off by default: it is a look, not a piece of feel.", False),
    ("normals", "lit sprites", "screen",
     "Rake the light across each creature instead of only dimming it: a warm "
     "band on the side facing the nearest light and a cool one opposite. The "
     "card generates real normals from the sprite's own silhouette; this is "
     "the two-blit software reading of the same idea, and at 32px it is most "
     "of what those normals were buying. Needs the lighting on."),
    ("outline", "sprite outlines", "screen",
     "The silhouette blitted four times behind the sprite. Lifts creatures off "
     "a busy floor for four extra blits each."),
    ("haze", "heat haze", "screen",
     "Rows re-blitted at sine offsets above each torch -- the same trick the "
     "slime's bands use, pointed at the background. Off by default; it only "
     "makes sense with the lighting on.", False),

    # -- feel -------------------------------------------------------------
    ("buffer", "input buffering", "feel",
     "Hold a press that arrives mid-animation and spend it the moment the turn "
     "opens. No pixels, and the biggest single win on this list."),
    ("pacing", "turn pacing", "feel",
     "Shorten the turn while the player keeps walking, and give it back in "
     "full the moment they stop. Corridors sprint, fights stay heavy."),

    # -- sound ------------------------------------------------------------
    ("sound", "sound", "sound",
     "Layered impact + material + tier, each drawn from a pre-rendered pitch "
     "ladder. Half of juice is audible and the bench had none of it."),
    ("music", "music", "sound",
     "The background track, ducked under every hit. Ducking is hit-stop for "
     "the ears: same instant, same job.", False),
    ("footsteps", "footsteps", "sound",
     "A quiet scuff per step. Only noticed when a character crosses a room in "
     "silence."),
    ("logpop", "message pop", "sound",
     "Scale each log line in as it lands and fade it out after. Even the text "
     "is animated -- turn it off and the log goes back to a printout."),
]

#: (key, label, group, value, lo, hi, blurb, fmt)
#: Anything with a number in it that a person might reasonably want different.
PARAM_SPECS = [
    ("intensity", "master intensity", "movement", 1.0, 0.0, 2.0,
     "Scales every amplitude in the bench at once, and none of the timings.", "{:.2f}"),
    ("move_time", "step time", "movement", MOVE_TIME, 0.04, 0.40,
     "How long a step takes to draw. Past about 0.22s the game starts to feel "
     "sticky no matter how pretty the step is.", "{:.3f}s"),
    ("hop_height", "hop height", "movement", 0.34, 0.0, 1.0,
     "Peak lift of a step, in tiles.", "{:.2f}"),
    ("squash_amt", "squash amount", "movement", 0.30, 0.0, 0.8,
     "Deformation at take-off and apex. Past ~0.45 it stops being a body.", "{:.2f}"),
    ("jelly_stiff", "slime stiffness", "movement", 120.0, 20.0, 320.0,
     "Spring rate of the leading band. Sets the pitch of the wobble.", "{:.0f}"),
    ("jelly_damp", "slime damping", "movement", 9.0, 2.0, 26.0,
     "Below ~6 it is a water balloon, above ~18 it is rubber. Critical for the "
     "default stiffness is about 22.", "{:.1f}"),
    ("jelly_drag", "slime drag", "movement", 0.58, 0.0, 1.0,
     "How much of each band's lag is actually applied.", "{:.2f}"),
    ("jelly_stretch", "slime pinch", "movement", 0.35, 0.0, 1.0,
     "Cross-axis squeeze per tile of string-out. Compounds with the hop's own "
     "squash, which is why it is kept modest.", "{:.2f}"),
    ("lean_deg", "lean angle", "movement", 7.0, 0.0, 30.0,
     "Degrees of bank at full lean.", "{:.1f}d"),
    ("bob_amt", "breath depth", "movement", 0.035, 0.0, 0.15,
     "Idle scale wobble. Large enough to see is already too large.", "{:.3f}"),
    ("fidget_period", "fidget interval", "movement", 3.4, 0.5, 12.0,
     "Average seconds between twitches, jittered per entity.", "{:.1f}s"),
    ("shadow_size", "shadow size", "movement", 0.60, 0.0, 1.2,
     "Ellipse width as a fraction of a tile, at ground level.", "{:.2f}"),
    ("shadow_alpha", "shadow darkness", "movement", 0.45, 0.0, 1.0,
     "How dark the blob is when the body is on the floor.", "{:.2f}"),
    ("tail_stiff", "tail stiffness", "movement", 220.0, 30.0, 600.0,
     "Spring rate along the chain. Soft chains whip, stiff ones follow.", "{:.0f}"),
    ("tail_damp", "tail damping", "movement", 12.0, 2.0, 40.0,
     "How fast the whip settles.", "{:.1f}"),
    ("ghost_life", "afterimage life", "movement", 0.20, 0.05, 0.8,
     "Seconds an echo survives. Long trails read as motion blur, short ones as "
     "speed.", "{:.2f}s"),
    ("dust_count", "dust per landing", "movement", 5.0, 0.0, 20.0,
     "Particles per foot on touchdown.", "{:.0f}"),

    ("windup_time", "wind-up time", "attack", ATTACK_WINDUP, 0.0, 0.4,
     "How long the wound-up pose is held before the strike.", "{:.3f}s"),
    ("attack_time", "attack time", "attack", ATTACK_TIME, 0.06, 0.6,
     "Length of the whole swing. Contact is at a third of it.", "{:.3f}s"),
    ("lunge_reach", "lunge reach", "attack", 0.55, 0.0, 1.2,
     "How far into the target's tile the attacker drives, in tiles.", "{:.2f}"),
    ("hitstop_light", "hit-stop (light)", "attack", 0.055, 0.0, 0.25,
     "Freeze on an ordinary hit. Three frames at 60Hz.", "{:.3f}s"),
    ("hitstop_heavy", "hit-stop (kill)", "attack", 0.11, 0.0, 0.4,
     "Freeze on a killing blow. Past ~0.15s it reads as a stutter rather than "
     "an impact.", "{:.3f}s"),
    ("flash_time", "hit flash time", "attack", FLASH_TIME, 0.02, 0.5,
     "How long the victim stays white.", "{:.3f}s"),
    ("knock_dist", "knockback", "attack", 0.30, 0.0, 1.2,
     "Tiles the victim is shoved, before weight divides it.", "{:.2f}"),
    ("shiver_amt", "shiver amount", "attack", 0.07, 0.0, 0.3,
     "Amplitude of the victim's jitter, in tiles.", "{:.3f}"),
    ("crit_chance", "crit chance", "attack", 0.22, 0.0, 1.0,
     "Share of blows that get the escalated tier.", "{:.2f}"),
    ("corpse_life", "corpse life", "attack", 26.0, 1.0, 90.0,
     "Seconds a body lies there before fading.", "{:.0f}s"),

    ("shake_px", "shake distance", "camera", 16.0, 0.0, 60.0,
     "Peak displacement in pixels at full trauma.", "{:.0f}px"),
    ("shake_decay", "trauma decay", "camera", 1.6, 0.2, 6.0,
     "Trauma drained per second. This is what stops shakes stacking.", "{:.2f}"),
    ("shake_freq", "shake frequency", "camera", 22.0, 4.0, 60.0,
     "Noise samples per second. Low is a lurch, high is a buzz.", "{:.0f}"),
    ("trauma_light", "trauma per hit", "camera", 0.28, 0.0, 1.0,
     "Added by an ordinary blow.", "{:.2f}"),
    ("trauma_heavy", "trauma per kill", "camera", 0.55, 0.0, 1.0,
     "Added by a killing one.", "{:.2f}"),
    ("kick_force", "kick force", "camera", 900.0, 0.0, 3000.0,
     "Impulse into the directional kick spring.", "{:.0f}"),
    ("zoom_punch", "zoom punch", "camera", 1.1, 0.0, 6.0,
     "Impulse into the zoom spring. The result is clamped to +/-30%.", "{:.2f}"),
    ("tilt_punch", "tilt punch", "camera", 26.0, 0.0, 120.0,
     "Impulse into the roll spring, in degrees per second.", "{:.0f}"),
    ("cam_smooth", "camera smoothing", "camera", 0.0001, 0.000001, 0.05,
     "Fraction of the follow error still left after one second. Smaller is "
     "snappier; this is the frame-rate-independent form.", "{:.5f}"),
    ("cam_lead", "look-ahead", "camera", 1.4, 0.0, 5.0,
     "Tiles the camera biases ahead of the facing direction.", "{:.2f}"),
    ("rumble_str", "rumble strength", "camera", 1.0, 0.0, 1.0,
     "Scales both motors. Needs a pad plugged in.", "{:.2f}"),

    ("spawn_count", "enemies spawned", "world", float(len(MONSTERS)), 0.0, 48.0,
     "How many monsters a reset puts on the floor. The first nine are the "
     "hand-placed roster -- one of each, at spots picked so most approaches "
     "have a corner in them -- and past that the cast repeats onto scattered "
     "free tiles. Takes effect on the next reset (r), because respawning the "
     "room under a fight in progress is not a thing a slider should do.",
     "{:.0f}"),
    ("part_count", "particles per hit", "world", 12.0, 0.0, 60.0,
     "Sparks in the contact burst, before the tier multiplies it.", "{:.0f}"),
    ("part_speed", "particle speed", "world", 430.0, 40.0, 1200.0,
     "Top of the speed range. The bottom is fixed, and the *spread* is what "
     "stops a burst reading as a wheel.", "{:.0f}"),
    ("wave_radius", "shockwave radius", "world", 1.75, 0.2, 5.0,
     "Peak ring radius, in tiles.", "{:.2f}"),
    ("ripple_str", "ripple strength", "world", 7.5, 0.0, 30.0,
     "Peak floor displacement in pixels.", "{:.1f}"),
    ("ripple_speed", "ripple speed", "world", 270.0, 60.0, 900.0,
     "How fast the wave front travels. Fast and far means repainting most of "
     "the floor -- watch the ms readout.", "{:.0f}"),
    ("decal_life", "decal life", "world", 22.0, 1.0, 120.0,
     "Seconds a blood mark stays before fading out.", "{:.0f}s"),
    ("ambient_amt", "ambient sway", "world", 1.6, 0.0, 8.0,
     "Peak drift of the tufts and torches, in pixels.", "{:.2f}"),

    ("flash_amt", "screen flash", "screen", 0.16, 0.0, 0.8,
     "Peak whiteness over the whole frame.", "{:.2f}"),
    ("vig_amt", "vignette pulse", "screen", 0.55, 0.0, 1.0,
     "How far the corners clamp down on impact.", "{:.2f}"),
    ("vig_base", "vignette base", "screen", 90.0, 0.0, 200.0,
     "Resting darkness at the edges, before any pulse.", "{:.0f}"),
    ("rgb_amt", "RGB split", "screen", 5.0, 0.0, 20.0,
     "Channel separation in pixels at the moment of impact.", "{:.1f}"),
    ("bloom_amt", "bloom strength", "screen", 0.60, 0.0, 1.5,
     "How much of the blurred copy is added back.", "{:.2f}"),
    ("bloom_thresh", "bloom threshold", "screen", 140.0, 0.0, 255.0,
     "Brightness a pixel must beat to glow. At zero the whole frame hazes "
     "over, which is the classic way to overdo it.", "{:.0f}"),
    ("light_radius", "light radius", "screen", 5.5, 1.0, 14.0,
     "Reach of each light, in tiles.", "{:.2f}"),
    ("light_ambient", "ambient light", "screen", 0.34, 0.0, 1.0,
     "How visible an unlit corner is. Zero is a torch-only dungeon.", "{:.2f}"),
    ("light_warm", "light warmth", "screen", 0.30, 0.0, 1.0,
     "How much of the light map is *added* as well as multiplied. Multiply "
     "alone can only darken, so on a near-black floor a torch would light "
     "nothing; this is the half that puts brightness back.", "{:.2f}"),
    ("light_height", "light height", "screen", 82.0, 2.0, 240.0,
     "How far the lights float above the floor plane, in pixels. Low is a "
     "torch held at the floor: a hot spot a tile wide that falls away hard, "
     "and a dimmer room around it. High is a lamp on the ceiling, which "
     "flattens the pool out and lifts everything in it. It is the shape of the "
     "falloff and not its reach -- the radius slider still says how far the "
     "light goes.", "{:.0f}"),
    ("light_contrast", "light / dark contrast", "screen", 1.0, 0.0, 10.0,
     "Difference between lit and shadowed areas. Zero flattens the lighting; "
     "one is natural; ten removes ambient light completely, making areas "
     "outside direct light pitch black. Below one it also fades the cast "
     "shadows out, because a flat room should not have hard ones in it.",
     "{:.2f}"),
    ("wall_shadow_amt", "wall shadow amount", "screen", 1.0, 0.0, 1.0,
     "Opacity of the shadows walls throw. One makes masonry a real occluder -- "
     "a pillar between you and a torch puts a wedge of dark across the floor "
     "and the wedge swings as you walk. Zero lets light pass through walls, "
     "which is what every light map without a visibility pass does.", "{:.2f}"),
    ("npc_shadow_amt", "NPC shadow amount", "screen", 0.55, 0.0, 1.0,
     "Opacity of the soft shadows monsters and the dummy throw, independent of "
     "the wall shadows. Deliberately soft and deliberately weak: a hard body "
     "shadow at this sprite size reads as a second creature.", "{:.2f}"),
    ("normal_depth", "sprite light depth", "screen", 1.0, 0.0, 2.0,
     "How hard the raking light is pushed across a creature. Past about 1.5 "
     "they stop being lit and start being two-tone.", "{:.2f}"),
    ("outline_alpha", "outline strength", "screen", 0.75, 0.0, 1.0,
     "Opacity of the silhouette behind each sprite.", "{:.2f}"),
    ("haze_amt", "haze amount", "screen", 2.2, 0.0, 10.0,
     "Peak row displacement above a torch, in pixels.", "{:.2f}"),

    ("buffer_win", "buffer window", "feel", 0.16, 0.0, 0.5,
     "How long an early press stays live. Longer than about a turn and the "
     "character walks on after you stop.", "{:.3f}s"),
    ("turn_fast", "fastest turn", "feel", 0.06, 0.02, 0.30,
     "Turn length once the player has been walking for a few steps.", "{:.3f}s"),
    ("pace_gain", "pace ramp", "feel", 0.34, 0.05, 1.0,
     "Streak gained per consecutive step. 0.34 is about three steps to full "
     "speed.", "{:.2f}"),
    ("pace_decay", "pace decay", "feel", 3.2, 0.2, 12.0,
     "Streak lost per second standing still. Fast, because the streak should "
     "mean 'still walking', not 'walked recently'.", "{:.1f}"),

    ("snd_master", "master volume", "sound", 0.8, 0.0, 1.0, "", "{:.2f}"),
    ("snd_sfx", "sfx volume", "sound", 0.85, 0.0, 1.0, "", "{:.2f}"),
    ("snd_music", "music volume", "sound", 0.40, 0.0, 1.0, "", "{:.2f}"),
    ("snd_pitch", "pitch variation", "sound", 1.0, 0.0, 1.0,
     "Spread of the pre-rendered pitch ladder actually used. At zero every hit "
     "plays the identical sample, which is the stapler.", "{:.2f}"),
    ("snd_duck", "music ducking", "sound", 0.45, 0.0, 1.0,
     "How far the music is pushed down under a hit.", "{:.2f}"),
]

EASE_ORDER = ["out_quad", "out_cubic", "out_quint", "out_back", "out_elastic",
              "out_bounce", "in_out_cubic", "linear"]

CHOICE_SPECS = [
    ("move_ease", "move easing", "movement",
     "The curve a step follows. Swapping this one function is often the whole "
     "difference between a spreadsheet and a body.", EASE_ORDER, 0),
    ("pixel_mode", "pixel mode", "screen",
     "snapped: camera and sprites on whole pixels -- crispest, and slow motion "
     "visibly stair-steps. rounded: smooth camera, sprites still snapped. "
     "subpixel: sprites blended across two pixels, so motion is smooth and "
     "everything is very slightly soft. There is no free option.",
     ["rounded", "snapped", "subpixel"], 0),
]

#: Panel order. Groups are drawn top to bottom in this order.
GROUPS = ["movement", "attack", "camera", "world", "screen", "feel", "sound"]


class Juice:
    """All the toggles, the sliders, the master A/B switch and the intensity.

    `on(name)` and `p(name)` are the only things the rest of the file asks.
    Routing every check and every number through here is what makes the master
    switch, the intensity dial and the whole slider panel one-line features
    instead of an edit at a hundred call sites.
    """

    def __init__(self) -> None:
        self.toggles = {}
        for spec in TOGGLE_SPECS:
            key, lbl, grp, blurb = spec[:4]
            on = spec[4] if len(spec) > 4 else True
            self.toggles[key] = Toggle(key, lbl, grp, blurb, on)
        self.params = Params(PARAM_SPECS)
        self.choices = {c[0]: Choice(*c) for c in CHOICE_SPECS}
        self.master = True          # False = the raw, unjuiced sim

    # -- the three questions the rest of the file asks --------------------
    def on(self, name: str) -> bool:
        return self.master and self.toggles[name].on

    def p(self, name: str) -> float:
        return self.params(name)

    def choice(self, name: str) -> str:
        return self.choices[name].value

    @property
    def intensity(self) -> float:
        return self.params("intensity")

    @intensity.setter
    def intensity(self, v: float) -> None:
        self.params["intensity"].value = clamp(v, 0.0, 2.0)

    def amt(self, value: float) -> float:
        """Scale an amplitude by the intensity dial."""
        return value * self.intensity

    def set_all(self, state: bool) -> None:
        for t in self.toggles.values():
            t.on = state

    def defaults(self) -> None:
        self.params.reset()
        for c in self.choices.values():
            c.index = 0


# ---------------------------------------------------------------------------
# The tiny sim
# ---------------------------------------------------------------------------


class Entity:
    """A sprite with a body, an animator and some hit points.

    Composition rather than a class tree: a monster is not a subclass of
    anything, it is an Entity that happens to hold different values. All the
    behaviour lives in the Motions attached to its Animator, which is what lets
    the juice layer be swapped out wholesale by the master toggle.
    """

    def __init__(self, sprite: str, x: int, y: int, color, name: str, hp: int = 5,
                 *, tint: bool = True, invincible: bool = False,
                 stationary: bool = False, death: str = "spin",
                 weight: float = 1.0):
        self.sprite = sprite            # key into tiles.SPRITES
        self.color = color              # tint, and the colour of its sparks
        self.tint = tint                # False for art that is already coloured
        self.name = name
        self.body = Body(tx=float(x), ty=float(y))
        self.anim = Animator()
        self.trail = GhostTrail()
        self.hp = hp
        self.max_hp = hp
        self.weight = weight            # divides every reaction it receives
        self.material = audiofx.MATERIALS.get(sprite, "flesh") if HAVE_AUDIO else "flesh"
        # Invincible things still take the full hit *presentation* -- flash,
        # knockback, sparks, screen shake -- and simply never run out of health.
        # That is the whole point of a bench: you want to feel being hit.
        self.invincible = invincible
        self.stationary = stationary    # true for the training dummy
        self.death = death              # which exit animation it plays
        self.dying = False
        self.death_timer = 0.0
        self.flip = False               # drawn mirrored, swapped at the pinch
        # A bar that is late catching up, so a drop is legible as a width
        # rather than as "shorter than last time I looked".
        self.chip = ChipBar()

        # Seeded off the position so a row of dummies does not act in step.
        # Persistent, so they are gated by `anim.persistent_enabled` rather than
        # by declining to play them -- see World.update.
        self.breathe = Breathe(phase=(x * 3 + y * 7) % 7)
        self.fidget = Fidget(seed=x * 31 + y * 17)
        self.anim.add_persistent(self.breathe)
        self.anim.add_persistent(self.fidget)

        # Secondary motion, so it runs after the hop rather than alongside it.
        # A slime is looser and heavier than everything else on the board.
        loose = sprite == "slime"
        # Only the spring differs by creature: the slime is softer and less
        # damped, so it lags further and rings longer, and the deformation
        # follows from that rather than being dialled up separately.
        self.jelly = Jelly(stiffness=78.0 if loose else 120.0,
                           damping=6.0 if loose else 9.0,
                           drag=0.7 if loose else 0.55)
        self.jelly_scale = (self.jelly.stiffness / 120.0, self.jelly.damping / 9.0,
                            self.jelly.drag / 0.58)
        self.anim.add_post(self.jelly)

        spec = TAILED.get(sprite)
        self.tail = None
        if spec is not None:
            nodes, spacing, tint_col = spec
            self.tail = Tail(nodes=nodes, spacing=spacing,
                             anchor=(0.0, 0.06 if sprite != "player" else -0.02))
            self.tail_color = tint_col
            self.anim.add_post(self.tail)

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


def build_props(grid):
    """Scatter tufts on the floor and torches on the walls.

    Ambience needs something to *be* ambient. Wobbling the floor tiles
    themselves would mean repainting the whole baked arena every frame -- the
    exact cost the bake was there to remove -- so the sway is given to a few
    dozen small props instead. That is also how it is done for real: the ground
    is static and the grass moves.

    Returns a list of (kind, world x, world y, phase).
    """
    props = []
    n = 0
    for ty in range(1, GRID_H - 1):
        for tx in range(1, GRID_W - 1):
            if grid[ty][tx]:
                continue
            n += 1
            r = hash01(tx * 73 + ty * 131, 9)
            if r > 0.80:
                props.append(("tuft", (tx + 0.5) * TILE + hash01(n, 3) * 8,
                              (ty + 0.62) * TILE + hash01(n, 4) * 5,
                              (hash01(n, 5) + 1.0) * 3.1))
    # Torches on a handful of the standalone pillars.
    for (px, py) in [(6, 4), (12, 12), (28, 7), (33, 15)]:
        props.append(("torch", (px + 0.5) * TILE, (py + 0.18) * TILE,
                      (px * 0.7 + py * 1.3) % 6.28))
    return props


class World:
    """Map, entities, and the shared effect pools. No drawing happens here.

    Everything in this class can be stepped with no display up, which is what
    `juicetest.py` leans on -- the sim and the effect state are testable, and
    only the compositing needs a surface.
    """

    def __init__(self, juice: Juice, audio=None):
        self.juice = juice
        self.audio = audio
        self.grid = build_map()
        self.props = build_props(self.grid)
        self.fx = EffectField()
        self.decals = DecalField()
        self.corpses: list[Corpse] = []
        self.trauma = Trauma()
        self.hitstop = HitStop()
        self.camera = Camera()
        self.kick = [Spring(0.0, 0.0, stiffness=210.0, damping=16.0),
                     Spring(0.0, 0.0, stiffness=210.0, damping=16.0)]
        self.buffer = InputBuffer()
        self.pacer = TurnPacer()
        self.rumble = RumbleMap()
        self.pad = None                    # a joystick, if one turns up
        self.screen_flash = 0.0
        self.vignette_pulse = 0.0
        self.rgb_split = 0.0
        self.lights: list[list] = []       # [x, y, strength, life, max_life]
        self.log: list[list] = []          # [text, age, colour]
        self.turn = 0
        self.turn_cooldown = 0.0           # seconds until the player may act
        self.goal_map: list[list] = []     # distance-to-player, rebuilt per turn
        self.elapsed = 0.0
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
                                 "you", hp=99, tint=False, invincible=True,
                                 weight=1.0)
        else:
            self.player.anim.clear()
            self.player.trail.clear()
            self.player.body.reset_juice()

        px, py = self.player.tile
        self.entities = [self.player]
        for kind, name, color, hp, weight, x, y in self.spawn_list():
            if (x, y) == (px, py) or self.blocked(x, y):
                continue
            self.entities.append(Entity(kind, x, y, color, name, hp=hp,
                                        weight=weight,
                                        death=DEATH_ORDER[len(self.entities) %
                                                          len(DEATH_ORDER)]))
        # The punching bag: never moves, never dies, always tells you the number.
        if (DUMMY_POS) != (px, py) and not self.blocked(*DUMMY_POS):
            self.entities.append(
                Entity("dummy", *DUMMY_POS, (190, 150, 110), "training dummy",
                       hp=999, invincible=True, stationary=True, weight=3.0))

        self.fx.clear()
        self.decals.clear()
        self.corpses.clear()
        self.lights.clear()
        self.log.clear()
        self.buffer.clear()
        self.pacer.streak = 0.0
        self.turn_cooldown = 0.0
        self.rebuild_goal_map()
        if first_run:
            cx, cy = self.player.tile_center()
            self.camera.snap(cx, cy)
            self.clamp_camera()
        self.say("walk into something to hit it", ACCENT)

    def spawn_list(self):
        """The roster a reset should use, at whatever length the slider asks.

        Up to nine it is the hand-placed roster, unchanged and in order, which
        is what keeps the default arena exactly the arena it always was: one of
        each creature, at spots chosen so most approaches to the middle have a
        corner in them. Past nine the cast repeats onto free tiles, so the
        slider can put forty bodies in the room to watch a system under load
        without anybody having to place them.

        The extra tiles are picked by a seeded hash rather than `random`, for
        the same reason nothing else in the bench uses one: the same slider
        position has to give the same arena every time, or two runs cannot be
        compared. Free tiles are sorted by that hash, which is a shuffle that
        happens to be reproducible -- and taking them in order means every
        extra spawn lands somewhere distinct with no rejection loop to run dry.
        """
        want = max(0, int(round(self.juice.p("spawn_count"))))
        if want <= len(MONSTERS):
            return MONSTERS[:want]

        px, py = self.player.tile
        taken = {(px, py), DUMMY_POS} | {(m[5], m[6]) for m in MONSTERS}
        free = [(x, y)
                for y in range(1, GRID_H - 1) for x in range(1, GRID_W - 1)
                if not self.blocked(x, y) and (x, y) not in taken]
        free.sort(key=lambda t: hash01(t[0] * GRID_H + t[1], 977))

        roster = list(MONSTERS)
        for i in range(min(want - len(MONSTERS), len(free))):
            kind, name, color, hp, weight, _x, _y = MONSTERS[i % len(MONSTERS)]
            x, y = free[i]
            roster.append((kind, name, color, hp, weight, x, y))
        return roster

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

    # -- sound ------------------------------------------------------------
    def pan_of(self, wx: float) -> float:
        """Where a world x sits across the viewport, as -1..1.

        Panning a hit to where it happened is nearly free and does the same job
        for the ear that a directional camera kick does for the eye.
        """
        return clamp((wx - self.camera.x) / (VIEW_W * 0.5), -1.0, 1.0)

    def sfx(self, name: str, *, gain: float = 1.0, wx: float = None,
            pitch: float = 0.0):
        if self.audio is None or not self.juice.on("sound"):
            return
        self.audio.play(name, gain=gain, pitch=pitch,
                        pan=0.0 if wx is None else self.pan_of(wx))

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
                self.face(e, dx, dy)
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
    def face(self, e: Entity, dx: int, dy: int):
        """Point an entity, animating the turn if it is a left-right reversal.

        Only horizontal reversals get the pinch, because those are the ones
        where the *sprite* actually changes -- turning to face up or down does
        not mirror anything, so there is nothing to hide at the pinch and the
        squash would read as a hiccup. Walking in a straight line therefore
        never plays it, which is the point: it should mark the moment the
        character changes their mind.
        """
        e.body.facing = (dx, dy)
        if dx == 0:
            return
        want = dx < 0
        if want == e.flip:
            return
        if not self.juice.on("faceflip"):
            e.flip = want
            return
        e.anim.play(FaceFlip(duration=0.11,
                             swap=lambda e=e, w=want: setattr(e, "flip", w)))

    def move_entity(self, e: Entity, dx: int, dy: int):
        """Commit a move and attach whatever animation the juice asks for.

        Note the order: the *logical* move happens immediately and completely,
        and only then is an animation attached to drag the picture back to
        where the entity used to be. The sim never waits for a tween -- which
        is also why the enemies can path with no idea that animation exists.
        """
        j = self.juice
        e.body.facing = (dx, dy)
        e.body.tx += dx                              # instant, authoritative
        e.body.ty += dy

        if j.on("footsteps"):
            self.sfx("step", gain=0.9 if e is self.player else 0.45,
                     wx=e.tile_center()[0])

        if not j.on("tween"):
            # The raw sim: the sprite is simply somewhere else now. This is the
            # baseline every other movement effect is an argument against.
            self.face(e, dx, dy)
            return
        move_time = self.turn_time
        e.anim.clear()
        # After the clear, or the turn-to-face would be wiped by the very move
        # that asked for it.
        self.face(e, dx, dy)
        if j.on("hop"):
            e.anim.play(Hop(duration=move_time, dx=dx, dy=dy,
                            height=j.amt(j.p("hop_height")),
                            squash=j.amt(j.p("squash_amt")) if j.on("squash") else 0.0,
                            ease=EASINGS[j.choice("move_ease")]))
            if j.on("dust") and j.p("dust_count") >= 1.0:
                # Fired at the end of the hop rather than the start, so it is
                # the landing that kicks up floor and not the take-off.
                e.anim.play(Sequence([Wait(duration=move_time * 0.86),
                                      Callback(lambda e=e: self.landing_dust(e))]))
        else:
            e.anim.play(Slide(duration=move_time, dx=dx, dy=dy,
                              ease=EASINGS[j.choice("move_ease")]))
        if j.on("lean"):
            e.anim.play(Lean(duration=move_time * 1.4, dx=dx, dy=dy,
                             amount=j.amt(j.p("lean_deg"))))

    @property
    def turn_time(self) -> float:
        """How long this turn is allowed to take. Not a constant any more."""
        self.pacer.base = self.juice.p("move_time")
        self.pacer.fastest = min(self.juice.p("turn_fast"), self.pacer.base)
        self.pacer.gain = self.juice.p("pace_gain")
        self.pacer.decay = self.juice.p("pace_decay")
        self.pacer.enabled = self.juice.on("pacing")
        return self.pacer.time

    def try_move(self, e: Entity, dx: int, dy: int):
        """The player's turn: walk, or attack whatever is in the way."""
        if not self.can_act():
            # The press is not thrown away -- it is held, and spent the moment
            # the turn opens. This one branch is most of what "responsive"
            # means, and it has no pixels in it at all.
            if self.juice.on("buffer"):
                self.buffer.window = self.juice.p("buffer_win")
                self.buffer.press(("move", dx, dy))
            return
        nx, ny = e.tile[0] + dx, e.tile[1] + dy

        target = self.entity_at(nx, ny)
        if target is not None and target is not e:
            self.face(e, dx, dy)
            self.attack(e, target, dx, dy)
        elif self.blocked(nx, ny):
            e.body.facing = (dx, dy)
            self.bump_wall(e, dx, dy)
            return                       # a move into a wall costs no turn
        else:
            self.move_entity(e, dx, dy)
        self.end_player_turn()

    def can_act(self) -> bool:
        return not (self.player.anim.busy or self.hitstop.frozen
                    or self.turn_cooldown > 0.0)

    def end_player_turn(self):
        """Everything else gets to act, and the player is locked out briefly."""
        self.turn += 1
        # Read before the streak is wound on, so this turn's lockout is the
        # same length as the animation `move_entity` already scheduled -- and
        # so the *first* step after a pause is at full weight. The speed-up is
        # earned by the steps that follow it, not granted to the one that
        # starts the run.
        self.turn_cooldown = self.turn_time
        self.pacer.stepped()
        self.rebuild_goal_map()
        self.take_enemy_turns()

    def landing_dust(self, e: Entity):
        """A puff kicked out sideways from under the feet.

        Thrown left and right along the floor rather than upwards, because dust
        that rises reads as smoke. Slow, low gravity and high drag, so it
        billows and stalls instead of arcing away like a spark.
        """
        x, y = e.tile_center()
        count = int(self.juice.p("dust_count"))
        for direction in ((-1.0, -0.25), (1.0, -0.25)):
            self.fx.particles.burst(
                x, y + TILE * 0.42, count=count, direction=direction, spread=1.1,
                speed=(70 * PX, 210 * PX), life=(0.25, 0.5), size=5.0 * PX,
                colors=((132, 140, 166), (108, 116, 142), (92, 99, 124)),
                gravity=120.0 * PX, drag=4.2)

    def bump_wall(self, e: Entity, dx: int, dy: int):
        """A move that failed. It still gets an animation -- silence reads as a
        dropped input, and the player will just press the key again."""
        j = self.juice
        self.sfx("bump", wx=e.tile_center()[0])
        if not j.on("wallbump"):
            return
        e.anim.play(Sequence([
            Lunge(duration=0.16, dx=dx, dy=dy, reach=j.amt(0.22),
                  out_frac=0.35, stretch=0.1),
            Shiver(duration=0.16, amplitude=j.amt(0.05), seed=int(e.body.tx)),
        ]))
        if j.on("shake"):
            self.trauma.add(0.12)
        wx = (e.body.tx + 0.5 + dx * 0.55) * TILE
        wy = (e.body.ty + 0.5 + dy * 0.55) * TILE
        if j.on("particles"):
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

        The swing *sound*, by contrast, fires now -- on the wind-up. Audio that
        travels ahead of the picture is what makes the picture land on time,
        and a whoosh that starts at contact is always heard as late.
        """
        j = self.juice
        steps = []
        attack_time = j.p("attack_time")
        self.sfx("swing", gain=0.9, wx=attacker.tile_center()[0])
        if j.on("windup"):
            steps.append(Anticipate(duration=j.p("windup_time"), dx=dx, dy=dy,
                                    amount=j.amt(0.20), squash=j.amt(0.18)))

        impact = Callback(lambda: self.land_blow(attacker, target, dx, dy))
        if j.on("lunge"):
            lunge = Lunge(duration=attack_time, dx=dx, dy=dy,
                          reach=j.amt(j.p("lunge_reach")), out_frac=0.32,
                          stretch=j.amt(0.22))
            # Contact is the moment the lunge is fully extended.
            steps.append(Parallel([lunge,
                                   Sequence([Wait(duration=attack_time * 0.32),
                                             impact])]))
        elif j.on("tween"):
            steps.append(impact)
            steps.append(Wait(duration=attack_time))
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
        crit = j.on("crit") and (hash01(self.turn * 7 + int(target.body.tx), 17)
                                 + 1.0) * 0.5 < j.p("crit_chance")
        if crit:
            damage = int(damage * 2.2) + 1
        if not target.invincible:
            target.hp -= damage
        killing = target.hp <= 0 and not target.invincible

        # One tier, applied to every channel at once. A crit is not a different
        # effect -- it is more of the one the player already knows, which is why
        # it reads as an escalation instead of as a surprise.
        tier = tier_for(crit, killing)
        heavy = tier.scale
        mass = target.weight if j.on("weight") else 1.0

        tx, ty = target.tile_center()
        # Sparks want to appear on the contact face, not in the middle of the
        # victim -- so pull them back towards the attacker by half a tile.
        cx, cy = tx - dx * TILE * 0.4, ty - dy * TILE * 0.4

        # -- what the ear does --------------------------------------------
        if self.audio is not None and j.on("sound"):
            self.audio.play_hit(target.material, gain=0.6 + 0.4 * heavy,
                                pan=self.pan_of(cx),
                                tier="kill" if killing else "crit" if crit else "normal")
            self.audio.duck(j.p("snd_duck") * heavy, 0.05 + 0.05 * heavy)
        if target is self.player:
            self.sfx("hurt", wx=cx)

        # -- what the victim does -----------------------------------------
        if j.on("hitflash"):
            target.anim.play(Flash(duration=j.p("flash_time"), strength=1.0))
        if j.on("knockback"):
            target.anim.play(Knockback(duration=0.3, dx=dx, dy=dy,
                                       distance=j.amt(j.p("knock_dist") * heavy) / mass))
        if j.on("shiver"):
            target.anim.play(Shiver(duration=0.26,
                                    amplitude=j.amt(j.p("shiver_amt") * heavy) / mass,
                                    seed=int(target.body.tx * 13 + target.body.ty)))

        # -- what the frame does ------------------------------------------
        if j.on("hitstop"):
            base = j.p("hitstop_heavy") if killing else j.p("hitstop_light")
            self.hitstop.hit(base * tier.hitstop)
        if j.on("shake"):
            self.trauma.decay = j.p("shake_decay")
            self.trauma.frequency = j.p("shake_freq")
            self.trauma.add(j.amt((j.p("trauma_heavy") if killing
                                   else j.p("trauma_light")) * tier.scale))
        if j.on("kick"):
            f = j.amt(j.p("kick_force") * PX * heavy)
            self.kick[0].kick(-dx * f)
            self.kick[1].kick(-dy * f)
        if j.on("zoom"):
            self.camera.punch(zoom=j.amt(j.p("zoom_punch") * heavy))
        if j.on("tilt"):
            # Sideways blows roll the frame; vertical ones get a fixed nudge,
            # since a straight up-down hit has no roll axis of its own.
            t = j.p("tilt_punch")
            self.camera.punch(tilt=j.amt((-dx * t + (t * 0.54 if dx == 0 else 0.0)) * heavy))
        if j.on("scrflash"):
            self.screen_flash = max(self.screen_flash, j.amt(j.p("flash_amt") * heavy))
        if j.on("vignette"):
            self.vignette_pulse = max(self.vignette_pulse, j.amt(j.p("vig_amt") * heavy))
        if j.on("rgbsplit"):
            self.rgb_split = max(self.rgb_split, j.amt(j.p("rgb_amt") * heavy))
        if j.on("light"):
            # The blow lights the room, not only the victim. Cheap, and it is
            # the difference between a flash on a sprite and a flash *in a
            # place*.
            self.lights.append([cx, cy, 1.4 * heavy, 0.22, 0.22])

        # -- what the world does ------------------------------------------
        if j.on("particles"):
            self.fx.particles.burst(
                cx, cy, count=int(j.p("part_count") * heavy),
                direction=(dx, dy), spread=1.9,
                speed=(140 * PX, j.p("part_speed") * PX * heavy), life=(0.22, 0.5),
                size=4.0 * PX, gravity=900.0 * PX,
                colors=(target.color, (255, 240, 200), (255, 190, 120)))
        if j.on("decals"):
            self.decals.splat(cx, cy + TILE * 0.28,
                              tuple(int(c * 0.45) for c in target.color),
                              count=3 + int(2 * heavy), spread=16.0 * heavy,
                              radius=5.0 * PX * heavy, life=j.p("decal_life"))
        if j.on("numbers"):
            self.fx.floaters.add(str(damage), tx, ty - TILE * 0.3,
                                 vx=lerp(-40, 40, ((damage * 37) % 10) / 10.0) * PX,
                                 vy=-150.0 * PX * heavy, gravity=150.0 * PX,
                                 color=GOLD if (killing or crit) else (255, 236, 170),
                                 life=0.85, max_life=0.85,
                                 scale_pop=0.9 * heavy)
        if j.on("shockwave"):
            self.fx.shockwave(cx, cy, max_radius=j.amt(TILE * j.p("wave_radius") * heavy),
                              life=0.35, width=5.0 * PX * heavy,
                              color=(255, 240, 210))
        if j.on("ripple"):
            # Deliberately short-ranged: the wave front covers about three
            # tiles before it dies. A ripple that crosses the whole arena reads
            # as an earthquake rather than as a blow landing here, and it makes
            # the renderer repaint the entire floor for a displacement of
            # nothing.
            self.fx.impact(cx, cy, strength=j.amt(j.p("ripple_str") * PX * heavy),
                           life=0.42, speed=j.p("ripple_speed") * PX,
                           wavelength=46.0 * PX)
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
        elif crit:
            self.say(f"CRIT! {target.name} takes {damage}", GOLD)
        else:
            self.say(f"{target.name} takes {damage}", target.color)

    def kill(self, target: Entity):
        if target.invincible:
            return                       # the training dummy outlives everyone
        target.dying = True
        target.anim.clear()
        j = self.juice
        self.sfx("death", wx=target.tile_center()[0])
        if j.on("death"):
            # Which exit it plays was decided at spawn, so a fight shows a
            # spread of them rather than the same one five times.
            facing = 1.0 if target.body.tx >= self.player.body.tx else -1.0
            motion = DEATHS[target.death](facing)
            target.anim.play(motion)
            target.death_timer = motion.duration
            if target.death == "burst":
                self.sfx("pop", wx=target.tile_center()[0])
        else:
            target.death_timer = 0.0     # gone on the next frame, no ceremony
        x, y = target.tile_center()
        if j.on("particles"):
            self.fx.particles.burst(x, y, count=22, speed=(90 * PX, 380 * PX),
                                    life=(0.3, 0.75), size=4.5 * PX,
                                    gravity=900.0 * PX,
                                    colors=(target.color, (255, 220, 180)))
        if j.on("decals"):
            self.decals.splat(x, y + TILE * 0.25, BLOOD, count=8, spread=30.0,
                              radius=7.0 * PX, life=j.p("decal_life"))
        self.say(f"{target.name} dies", WARN)

    def leave_corpse(self, target: Entity):
        """Called when the death animation runs out, before the entity goes."""
        if not self.juice.on("corpse"):
            return
        x, y = target.tile_center()
        life = self.juice.p("corpse_life")
        self.corpses.append(Corpse(
            sprite=target.sprite, color=target.color, tint=target.tint,
            x=x, y=y + TILE * 0.22,
            angle=90.0 if target.body.tx >= self.player.body.tx else -90.0,
            life=life, max_life=life))
        del self.corpses[:-40]

    def strike_player(self):
        """Have the nearest monster hit the player -- the same code, reversed,
        so the receiving end of every effect can be looked at too."""
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
        if not self.can_act():
            if self.juice.on("buffer"):
                self.buffer.window = self.juice.p("buffer_win")
                self.buffer.press(("swing",))
            return
        dx, dy = p.body.facing
        target = self.entity_at(p.tile[0] + dx, p.tile[1] + dy)
        if target is not None and target is not p:
            self.attack(p, target, dx, dy)
            self.end_player_turn()
            return

        j = self.juice
        self.sfx("swipe", gain=0.7, wx=p.tile_center()[0])
        steps = []
        if j.on("windup"):
            steps.append(Anticipate(duration=j.p("windup_time"), dx=dx, dy=dy,
                                    amount=j.amt(0.18)))
        if j.on("lunge"):
            steps.append(Lunge(duration=j.p("attack_time"), dx=dx, dy=dy,
                               reach=j.amt(j.p("lunge_reach") * 0.8), out_frac=0.32))
        if steps:
            p.anim.play(Sequence(steps))
        if j.on("slash"):
            cx, cy = p.tile_center()
            self.fx.slash(cx + dx * TILE * 0.55, cy + dy * TILE * 0.55,
                          angle=math.degrees(math.atan2(-dy, dx)),
                          radius=TILE * 0.55, life=0.16)
        self.end_player_turn()

    def spend_buffered(self):
        """Take a held press the moment the turn opens.

        Deliberately after everything else in the frame, and deliberately only
        one: the buffer exists to stop an early press being *lost*, not to let
        the player queue a route.
        """
        if not self.juice.on("buffer") or not self.can_act():
            return
        action = self.buffer.take()
        if action is None:
            return
        if action[0] == "move":
            self.try_move(self.player, action[1], action[2])
        elif action[0] == "swing":
            self.swing()

    # -- per-frame --------------------------------------------------------
    def update(self, dt: float):
        """One frame.

        Two clocks, and the split between them *is* hit-stop: `anim_dt` is what
        the bodies and the world effects run on and it is zero while frozen,
        while `dt` keeps running for the shake and the springs. A frame in which
        literally nothing changes reads as the game hanging; a frame in which
        everything but the shake is still reads as a blow landing.
        """
        j = self.juice
        anim_dt = self.hitstop.consume(dt)
        self.elapsed += dt

        breathing = j.on("bob") or j.on("fidget")
        wobbling = j.on("jelly")
        stiff, damp, drag = j.p("jelly_stiff"), j.p("jelly_damp"), j.p("jelly_drag")
        for e in list(self.entities):
            was_busy = e.anim.busy
            # Persistent motions have no "played" moment to decline, so the
            # only way to switch one off is a gate -- and the two idles share
            # one gate, so each also carries its own amplitude.
            e.anim.persistent_enabled = breathing
            e.breathe.amplitude = j.p("bob_amt") * j.intensity if j.on("bob") else 0.0
            e.fidget.period = j.p("fidget_period")
            e.fidget.amplitude = 0.05 * j.intensity if j.on("fidget") else 0.0
            # The jelly spring keeps integrating either way -- see Jelly -- so
            # switching it on mid-stride does not snap the body across a lag it
            # accumulated while nobody was looking.
            e.jelly.enabled = wobbling and not e.dying
            e.jelly.stiffness = stiff * e.jelly_scale[0]
            e.jelly.damping = damp * e.jelly_scale[1]
            e.jelly.drag = drag * e.jelly_scale[2] * j.intensity
            e.jelly.stretch = j.p("jelly_stretch") * j.intensity
            if e.tail is not None:
                e.tail.stiffness = j.p("tail_stiff")
                e.tail.damping = j.p("tail_damp")
            e.anim.update(e.body, anim_dt)

            e.chip.delay = 0.22
            e.chip.set(e.hp / e.max_hp if e.max_hp else 1.0)
            e.chip.update(dt)

            if j.on("ghost"):
                x, y = e.world_pos()
                b = e.body
                e.trail.life = j.p("ghost_life")
                e.trail.sample(x, y, b.sx, b.sy, b.angle, anim_dt, moving=was_busy)
            else:
                e.trail.clear()
            e.trail.update(dt)

            if e.dying:
                e.death_timer -= anim_dt
                if e.death_timer <= 0.0:
                    self.leave_corpse(e)
                    self.entities.remove(e)

        self.fx.update(anim_dt)
        self.decals.update(dt)
        for c in self.corpses:
            c.life -= dt
        self.corpses = [c for c in self.corpses if c.life > 0.0]
        for l in self.lights:
            l[3] -= dt
        self.lights = [l for l in self.lights if l[3] > 0.0]

        self.trauma.decay = j.p("shake_decay")
        self.trauma.frequency = j.p("shake_freq")
        self.trauma.update(dt)
        for k in self.kick:
            k.update(dt)
        self.camera.update(dt)
        self.camera.smoothing = j.p("cam_smooth")

        cx, cy = self.player.world_pos()
        if j.on("lead"):
            fx, fy = self.player.body.facing
            lead = j.p("cam_lead") * TILE
            cx += fx * lead
            cy += fy * lead
        self.camera.follow(cx, cy, dt, smooth=j.on("lag"))
        self.clamp_camera()

        self.pump_rumble(dt)

        self.screen_flash = max(0.0, self.screen_flash - dt * 1.6)
        self.vignette_pulse = max(0.0, self.vignette_pulse - dt * 2.2)
        self.rgb_split = max(0.0, self.rgb_split - dt * 26.0)
        # Ticks on the real clock, not the frozen one, so hit-stop does not
        # silently lengthen the turn.
        self.turn_cooldown = max(0.0, self.turn_cooldown - dt)
        if self.turn_cooldown <= 0.0 and not self.player.anim.busy:
            self.pacer.idle(dt)

        self.buffer.update(dt)
        self.spend_buffered()

        if self.audio is not None:
            self.audio.enabled = j.on("sound")
            self.audio.master = j.p("snd_master")
            self.audio.buses["sfx"].volume = j.p("snd_sfx")
            self.audio.buses["music"].volume = j.p("snd_music") if j.on("music") else 0.0
            self.audio.pitch_jitter = j.p("snd_pitch")
            self.audio.update(dt)

        for entry in self.log:
            entry[1] += dt

    def pump_rumble(self, dt: float):
        """Send the shake budget to the pad, if there is one.

        The same number that displaces the frame drives the motors, so the two
        can never disagree about how hard something hit -- which is exactly why
        rumble is worth wiring to trauma rather than to individual events.
        """
        if self.pad is None or not self.juice.on("rumble"):
            return
        self.rumble.strength = self.juice.p("rumble_str")
        out = self.rumble.update(self.trauma.shake, dt)
        if out is None:
            return
        try:
            self.pad.rumble(out[0], out[1], out[2])
        except Exception:                            # pragma: no cover
            self.pad = None


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

        #: What the footer lists when the mouse is not over a row. An attribute
        #: rather than the constant directly, because the GL bench has keys
        #: this one does not (a menu on escape, a bomb on B) and a shared
        #: module constant edited at import would put them in both.
        self.help_lines = HELP_LINES

        self.font_tile = load(int(TILE * 0.82), bold=True)
        self.font_ui = load(12)
        self.font_ui_b = load(12, bold=True)
        self.font_big = load(18, bold=True)
        self.font_num = load(18, bold=True)

        self.resize()
        #: (size, height, gain) -> the radial falloff for one light. Three
        #: numbers rather than one because the *shape* of a light is a slider
        #: now and not a constant, so a blob cannot simply be scaled.
        self._light_cache: dict = {}
        #: (size, x, y, amount) -> the wall shadows for one light. Torches do
        #: not move, so theirs are built once and blitted for the rest of the
        #: session; only the light the player carries is rebuilt as it walks.
        self._wall_cache: dict = {}
        #: size -> a scratch surface to assemble one light's visibility in,
        #: and a second for the body shadows that get multiplied into it.
        self._mask: dict = {}
        self._shade: dict = {}
        #: The rake gradient, one per quantised light direction.
        self._ramp_cache: dict = {}
        #: (sprite surface, screen x, screen y, world x, world y) for every
        #: body drawn this frame, so the sprite lighting can run *after* the
        #: light map has been multiplied over the frame. Lighting a sprite
        #: before the multiply would only get it darkened again.
        self._lit: list = []
        self.sheet = tiles.SpriteSheet(TILE)
        self._floor = None          # baked on first draw; the arena is static
        self._shadow = self._make_shadow()
        self._decal_cache: dict = {}
        self._silhouettes: dict = {}
        self.rows: list = []       # (rect, kind, key), rebuilt every panel draw
        self.hover: tuple | None = None
        self.scroll = 0.0
        self.content_h = 0
        self.drag: str | None = None
        # Smoothed cost of a whole frame, shown in the panel. Several of these
        # effects are not free -- the RGB split alone is most of a frame's
        # budget -- and a bench that hid that would be teaching the wrong
        # lesson. Toggle one and watch the number move.
        self.frame_ms = 0.0

    def resize(self):
        """(Re)allocate every buffer whose size is the *view's* size.

        Its own method because the GL bench can change the window at runtime,
        and rebuilding the whole renderer to do it would reload five fonts and
        re-bake the vignette for a resize the user is still dragging. The fonts
        and the baked art do not care how big the window is; these do.
        """
        self.view = pygame.Surface((VIEW_W, VIEW_H)).convert()
        self.overlay = pygame.Surface((VIEW_W, VIEW_H), pygame.SRCALPHA)
        self.scratch = pygame.Surface((VIEW_W, VIEW_H), pygame.SRCALPHA)
        # Preallocated channel buffers for the RGB split. Copying a full view
        # is over a megabyte of allocation, and doing that twice a frame while
        # the effect is running is most of what it used to cost.
        self.split_r = pygame.Surface((VIEW_W, VIEW_H)).convert()
        self.split_b = pygame.Surface((VIEW_W, VIEW_H)).convert()
        # Bloom works at an eighth resolution: the downscale *is* the blur, and
        # at 100x75 the two smoothscales are cheap enough to leave on.
        self.bloom_small = pygame.Surface((max(1, VIEW_W // 8),
                                           max(1, VIEW_H // 8))).convert()
        self.bloom_big = pygame.Surface((VIEW_W, VIEW_H)).convert()
        # The light map is built at half resolution and scaled up. Light is the
        # lowest-frequency thing in the frame -- a gradient over five tiles has
        # nothing in it that a 400x300 buffer cannot hold -- so building it at
        # full size was paying four times over for detail that does not exist.
        # Halving it took the pass from 4.9ms to under 2.
        self.light_small = pygame.Surface((max(1, VIEW_W // 2),
                                           max(1, VIEW_H // 2))).convert()
        self.lightmap = pygame.Surface((VIEW_W, VIEW_H)).convert()
        self.vignette = self._make_vignette()

    # -- prebuilt surfaces -------------------------------------------------
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

    def _make_shadow(self):
        """One soft ellipse, drawn once and scaled per body.

        Concentric ellipses rather than a gradient, for the same reason as the
        vignette: at 32px behind 45% alpha the difference is theoretical.
        """
        w, h = TILE * 2, int(TILE * 1.1)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        steps = 8
        for i in range(steps, 0, -1):
            t = i / steps
            rect = pygame.Rect(0, 0, int(w * t), int(h * t))
            rect.center = (w // 2, h // 2)
            pygame.draw.ellipse(surf, (0, 0, 0, int(26 * (1.0 - t) + 12)), rect)
        return surf

    @staticmethod
    def _cached(store: dict, key):
        """A cache read that also marks the entry as the freshest.

        Least-recently-used and not "clear it when it gets big", because what
        gets big is the light the *player* is carrying -- a new key every time
        they take a step -- and a cache that throws everything out to make room
        for it would rebuild all four static torches every few frames to store
        one walking light that is never asked for twice.
        """
        hit = store.pop(key, None)
        if hit is not None:
            store[key] = hit                 # back to the young end
        return hit

    @staticmethod
    def _keep(store: dict, key, value, limit: int):
        store[key] = value
        while len(store) > limit:
            del store[next(iter(store))]     # the oldest untouched entry
        return value

    def _light_blob(self, size: int, height: float, gain: float):
        """A warm radial falloff as an *additive* RGB surface.

        Additive rather than alpha-blended so overlapping lights brighten each
        other, which is what light does and what a stack of alpha circles
        conspicuously fails to do.

        The falloff is two terms, and the second one is what `light height`
        buys. A light is a point floating above the floor plane, so the floor
        under it is hit square on and the floor a few tiles away is hit at a
        glance: the cosine of that angle is `h / hypot(distance, h)`, which is
        a term the card gets for free out of a dot product and this has to bake
        into the gradient instead. Drop the light to the floor and the hot spot
        collapses to a tile across with the rest of the room dimmer behind it;
        lift it to the ceiling and it flattens out into the plain squared
        falloff the bench used to have, with everything under it brighter.
        Same reach either way -- the radius slider owns that.

        Cached on all three numbers because two of them are sliders. The blob
        is a few dozen concentric circles, which is cheap enough to rebuild
        while one is being dragged and far too expensive to rebuild per frame.
        """
        key = (size, int(height / 4.0), int(gain * 20.0))
        hit = self._cached(self._light_cache, key)
        if hit is not None:
            return hit
        surf = pygame.Surface((size, size)).convert()
        surf.fill((0, 0, 0))
        r = size // 2
        h = max(1.0, height)
        for i in range(r, 0, -2):
            t = i / r                       # 1 at the rim, 0 in the middle
            # `size` is the light's radius in *world* pixels: the map is built
            # at half resolution, so one blob pixel is two of them.
            atten = (1.0 - t) ** 2
            rake = h / math.hypot(t * size, h)
            v = clamp(atten * (0.45 + 0.55 * rake) * gain)
            pygame.draw.circle(surf, (int(232 * v), int(206 * v), int(160 * v)),
                               (r, r), i)
        surf.set_colorkey(None)
        return self._keep(self._light_cache, key, surf, 10)

    def _scratch(self, store: dict, size: int):
        """A reusable surface one light's visibility is assembled in.

        Kept one per size rather than one in total, because the three kinds of
        light in the room are three different radii and a single scratch would
        be reallocated three times a frame -- which is the allocation the
        half-resolution map exists to avoid.
        """
        hit = self._cached(store, size)
        if hit is None:
            hit = self._keep(store, size,
                             pygame.Surface((size, size)).convert(), 4)
        return hit

    def _wall_shadows(self, world: World, wx: float, wy: float, size: int,
                      amount: float):
        """White where this light reaches, dark where masonry is in the way.

        The card asks the question per fragment: march sixty-four steps towards
        the light and see if a wall cell was crossed. There is no marching a
        pixel here, so the question is turned round and answered per *wall*
        instead -- the shadow a box throws is the quad between its two
        silhouette corners and those corners projected away from the light,
        which is one polygon fill for a shape a ray march would have paid four
        hundred thousand samples for.

        Two details are worth the lines:

        * the light's own tile never occludes, or a torch mounted on a pillar
          would put the pillar between itself and the room;
        * wall tiles are painted back in afterwards, because a wall the light
          can see is lit whatever is behind it. Without that, the far side of a
          pillar goes to a flat silhouette and the room reads as a hole.

        Built in the light's own space -- offsets from the light, halved -- so
        it does not depend on where the camera happens to be, which is what
        makes it cacheable for the torches at all.
        """
        key = (size, round(wx), round(wy), int(amount * 32))
        hit = self._cached(self._wall_cache, key)
        if hit is not None:
            return hit

        surf = pygame.Surface((size, size)).convert()
        surf.fill((255, 255, 255))
        reach = float(size)                  # the light's radius, world pixels
        half = size * 0.5
        level = (int(255 * (1.0 - clamp(amount))),) * 3
        far = reach * 2.2                    # past the rim, so no shadow ends
        ltx, lty = int(wx // TILE), int(wy // TILE)

        def local(px, py):
            return ((px - wx) * 0.5 + half, (py - wy) * 0.5 + half)

        lit_walls = []
        x0 = max(0, int((wx - reach) // TILE))
        x1 = min(GRID_W - 1, int((wx + reach) // TILE))
        y0 = max(0, int((wy - reach) // TILE))
        y1 = min(GRID_H - 1, int((wy + reach) // TILE))
        for ty in range(y0, y1 + 1):
            row = world.grid[ty]
            for tx in range(x0, x1 + 1):
                if not row[tx] or (tx, ty) == (ltx, lty):
                    continue
                left, top = tx * TILE, ty * TILE
                cx, cy = left + TILE * 0.5, top + TILE * 0.5
                lx, ly = local(left, top)
                lit_walls.append(pygame.Rect(math.floor(lx), math.floor(ly),
                                             TILE // 2 + 1, TILE // 2 + 1))
                if math.hypot(cx - wx, cy - wy) > reach + TILE:
                    continue
                # The two corners furthest apart in angle *as seen from the
                # light* are the silhouette. Angles are taken relative to the
                # tile's own bearing so the comparison cannot straddle the
                # wrap at pi and pick the wrong pair.
                base = math.atan2(cy - wy, cx - wx)
                lo = hi = None
                for px, py in ((left, top), (left + TILE, top),
                               (left + TILE, top + TILE), (left, top + TILE)):
                    a = (math.atan2(py - wy, px - wx) - base + math.pi) \
                        % (2.0 * math.pi) - math.pi
                    if lo is None or a < lo[0]:
                        lo = (a, px, py)
                    if hi is None or a > hi[0]:
                        hi = (a, px, py)
                quad = []
                for _a, px, py in (lo, hi):
                    away = math.hypot(px - wx, py - wy) or 1.0
                    quad.append((px, py))                       # the corner
                    quad.append((px + (px - wx) / away * far,   # and its ray
                                 py + (py - wy) / away * far))
                near_a, far_a, near_b, far_b = quad
                pygame.draw.polygon(surf, level, [local(*near_a), local(*far_a),
                                                  local(*far_b), local(*near_b)])
        for rect in lit_walls:
            surf.fill((255, 255, 255), rect)
        return self._keep(self._wall_cache, key, surf, 10)

    def _cast_shadows(self, wx: float, wy: float, size: int, casters,
                      amount: float):
        """The soft shadows the bodies in the room throw away from one light.

        Returned as its own layer rather than drawn straight into the light's
        visibility, and that is not tidiness -- a `draw` call *replaces* the
        pixels it covers, so a body standing in front of a wall used to stamp
        its 55%-dark trapezoid over the wall's fully-dark wedge and cut a
        lighter body-shaped hole through it. Visibility terms multiply: the
        shader says `wall_shadow * npc_shadow` and this layer is the second
        half of that product, blended in with `BLEND_RGB_MULT`.

        Three nested trapezoids rather than a blur: a body shadow at this size
        is a smudge a couple of tiles long, and the eye reads three steps of
        penumbra as a soft edge quite happily -- especially after the whole
        light map is scaled up from half resolution, which softens it again for
        nothing. A real blur would cost more than every other light in the room
        put together. The layers are drawn widest first across *every* caster
        before the next one starts, for the same reason the layer exists at
        all: one body's soft outer edge must never land on another body's core.

        Returns None when nothing is in reach, so the caller can skip a blit.
        """
        reach = float(size)
        half = size * 0.5
        near = [(cx, cy, rad, dx / d, dy / d, reach - d + TILE)
                for cx, cy, rad in casters
                for dx, dy, d in ((cx - wx, cy - wy,
                                   math.hypot(cx - wx, cy - wy)),)
                if 1e-3 < d <= reach]
        if not near:
            return None
        surf = self._scratch(self._shade, size)
        surf.fill((255, 255, 255))
        for spread, weight in ((2.1, 0.34), (1.5, 0.66), (1.0, 1.0)):
            level = (int(255 * (1.0 - clamp(amount * weight))),) * 3
            for cx, cy, rad, ux, uy, length in near:
                px, py = -uy, ux
                # `length` only reaches as far as the light still carries:
                # past the rim there is nothing left to take away.
                edge, far_edge = rad * spread, rad * spread * 1.8
                pts = ((cx + px * edge, cy + py * edge),
                       (cx - px * edge, cy - py * edge),
                       (cx - px * far_edge + ux * length,
                        cy - py * far_edge + uy * length),
                       (cx + px * far_edge + ux * length,
                        cy + py * far_edge + uy * length))
                pygame.draw.polygon(
                    surf, level,
                    [((x - wx) * 0.5 + half, (y - wy) * 0.5 + half)
                     for x, y in pts])
        return surf

    # -- helpers ----------------------------------------------------------
    def sprite_for(self, e: Entity, flip: bool = None):
        """The resting sprite for an entity, tinted and cached by the sheet."""
        col, row = tiles.SPRITES.get(e.sprite, tiles.SPRITES["goblin"])
        img = self.sheet.sprite(col, row, e.color if e.tint else None)
        if (e.flip if flip is None else flip):
            key = ("flip", e.sprite, e.color if e.tint else None)
            hit = self._silhouettes.get(key)
            if hit is None:
                hit = pygame.transform.flip(img, True, False)
                self._silhouettes[key] = hit
            return hit
        return img

    def silhouette(self, e: Entity):
        """The sprite as a flat black shape, for the outline pass."""
        key = ("sil", e.sprite, e.flip)
        hit = self._silhouettes.get(key)
        if hit is None:
            base = self.sprite_for(e).copy()
            # Multiplying the colour to zero leaves the alpha, and the alpha is
            # the silhouette. No per-pixel work anywhere.
            base.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
            self._silhouettes[key] = hit = base
        return hit

    def to_view(self, world: World, wx: float, wy: float):
        """World pixels -> view-surface pixels."""
        cx, cy = world.camera.x, world.camera.y
        if world.juice.choice("pixel_mode") == "snapped":
            # Snapping the *camera* is what actually stops the shimmer: a
            # fractional camera re-rounds every sprite differently each frame,
            # so a stationary row of pillars crawls.
            cx, cy = round(cx), round(cy)
        return (wx - cx + VIEW_W * 0.5, wy - cy + VIEW_H * 0.5)

    def _blit_at(self, img, sx: float, sy: float, mode: str, alpha: float = 1.0):
        """Blit centred at a possibly-fractional position.

        In `subpixel` the sprite is drawn twice, weighted by the fractional
        part -- which is genuine sub-pixel positioning, at the cost of one
        extra blit and a slight softness while it is between pixels. It is the
        honest trade: pixel art can be crisp or it can move smoothly at low
        speeds, and this is the dial between the two.
        """
        r = img.get_rect()
        x, y = sx - r.w * 0.5, sy - r.h * 0.5
        if mode != "subpixel":
            if alpha < 1.0:
                img = img.copy()
                img.set_alpha(int(255 * clamp(alpha)))
            self.view.blit(img, (round(x), round(y)))
            return
        ix, iy = math.floor(x), math.floor(y)
        fx, fy = x - ix, y - iy
        for ox, oy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            a = int(255 * clamp(w * alpha))
            if a <= 3:
                continue
            tmp = img.copy()
            tmp.set_alpha(a)
            self.view.blit(tmp, (ix + ox, iy + oy))

    # -- the world --------------------------------------------------------
    def draw_world(self, world: World):
        j = world.juice
        self.view.fill(BG)
        self._lit.clear()
        self.draw_floor(world)
        if j.on("decals"):
            self.draw_decals(world)
        self.draw_props(world)
        self.draw_shockwaves(world)
        self.draw_corpses(world)
        self.draw_entities(world)
        if j.on("light"):
            self.apply_light(world)
            if j.on("normals"):
                self.apply_sprite_light(world)
        if j.on("haze"):
            self.draw_haze(world)
        self.draw_slashes(world)
        self.draw_cuts(world)
        self.draw_particles(world)
        self.draw_bars(world)
        self.draw_floaters(world)
        if j.on("bloom"):
            self.apply_bloom(world)

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
        ox, oy = self.to_view(world, 0.0, 0.0)
        v.blit(self._floor, (int(ox), int(oy)))

        rippling = world.juice.on("ripple") and world.fx.impacts
        if not rippling:
            return
        # A wave only touches the tiles its front has reached, so the union of
        # the live impacts bounds the repaint to a few dozen tiles. On a 40x20
        # arena that is the difference between 475 displacement calls a frame
        # and a few dozen.
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
                dx, dy = world.fx.tile_offset(wx, wy)
                if abs(dx) < 0.4 and abs(dy) < 0.4:
                    continue          # the baked copy is already correct here
                # Paint out the baked tile first, or the displaced one would be
                # drawn on top of a stationary twin of itself.
                sx, sy = self.to_view(world, wx, wy)
                v.fill(BG, self._tile_rect(sx, sy).inflate(4, 4))
                sx, sy = self.to_view(world, wx + dx, wy + dy)
                self._paint_tile(v, world, tx, ty, sx, sy)

    def _decal_blob(self, radius: int, color, squash: float):
        """One pre-rendered ellipse per (size, colour). Decals do not move, so
        every one of them can come out of a cache with no drawing at all."""
        key = (radius, color)
        hit = self._decal_cache.get(key)
        if hit is None:
            w, h = radius * 2, max(2, int(radius * 2 * squash))
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, (*color, 190), surf.get_rect())
            pygame.draw.ellipse(surf, (*color, 255), surf.get_rect().inflate(-3, -2))
            self._decal_cache[key] = hit = surf
        return hit

    def draw_decals(self, world: World):
        left, top = self.to_view(world, 0.0, 0.0)
        for d in world.decals.decals:
            sx, sy = d.x + left, d.y + top
            if not (-40 < sx < VIEW_W + 40 and -40 < sy < VIEW_H + 40):
                continue
            blob = self._decal_blob(max(2, int(d.radius)), d.color, d.squash)
            a = d.alpha
            if a < 0.99:
                blob = blob.copy()
                blob.set_alpha(int(255 * a))
            self.view.blit(blob, blob.get_rect(center=(int(sx), int(sy))))

    def draw_props(self, world: World):
        """Tufts and torches, swaying if the ambient toggle is on.

        Cheap on purpose: this is the effect whose entire job is to stop the
        room looking like a screenshot, and it should not cost more than the
        things that are actually happening in it.
        """
        j = world.juice
        amp = j.p("ambient_amt") * j.intensity if j.on("ambient") else 0.0
        t = world.elapsed
        left, top = self.to_view(world, 0.0, 0.0)
        for kind, wx, wy, phase in world.props:
            sx, sy = wx + left, wy + top
            if not (-20 < sx < VIEW_W + 20 and -20 < sy < VIEW_H + 20):
                continue
            ox, oy = ambient_offset(wx, wy, t + phase, amp) if amp else (0.0, 0.0)
            if kind == "tuft":
                # Three blades, the tops leaning and the roots pinned -- which
                # is the whole difference between grass moving and a sprite
                # sliding about on the floor.
                for i in (-3, 0, 3):
                    bx = sx + i
                    pygame.draw.line(self.view, (44, 58, 50),
                                     (bx, sy), (bx + ox * (1.0 + abs(i) * 0.1),
                                                sy - 6 - abs(i) * 0.4), 1)
            else:
                flame = pygame.Rect(0, 0, 6, 11)
                flame.center = (int(sx + ox * 1.6), int(sy + oy - 4))
                pygame.draw.rect(self.view, (60, 52, 44),
                                 pygame.Rect(int(sx) - 2, int(sy) - 1, 4, 8))
                pygame.draw.ellipse(self.view, (236, 170, 78), flame)
                pygame.draw.ellipse(self.view, (255, 228, 150), flame.inflate(-3, -5))

    def torch_lights(self, world: World):
        """(x, y, radius, strength) in world pixels for everything that lights.

        A radius per light rather than one for all of them: a torch is a
        smaller light than the one the player carries and the flare a blow
        raises is a bigger one, and the ratios are the same three the GL bench
        uses so the two rooms light the same way.
        """
        j = world.juice
        radius = j.p("light_radius") * TILE
        px, py = world.player.world_pos()
        out = [(px, py, radius, 1.0)]
        t = world.elapsed
        for kind, wx, wy, phase in world.props:
            if kind != "torch":
                continue
            # A torch that is perfectly steady is a lamp. Two sines at
            # unrelated rates keep it from looking like a pulse.
            f = (0.86 + 0.09 * math.sin(t * 7.3 + phase)
                 + 0.05 * math.sin(t * 17.1 + phase * 2.0))
            out.append((wx, wy, radius * 0.85, f))
        for x, y, s, life, maxl in world.lights:
            out.append((x, y, radius * 1.1, s * (life / maxl)))
        return out

    def apply_light(self, world: World):
        """Multiply the frame by a light map built from additive blobs.

        No shaders, so this is the software cousin of the GL bench's lighting:
        an ambient grey, a warm radial gradient added at every light, and one
        multiply over the view. What the card does per fragment is done here
        per *occluder* instead -- one polygon per wall and three per body, into
        the light's own little surface, before the gradient is multiplied
        through it. The arithmetic that decides how the room ends up looking is
        the same in both:

        * `light height` shapes the falloff (`_light_blob`),
        * `contrast` takes the ambient down and the direct light up, and fades
          the shadows out below one,
        * `wall shadow` and `NPC shadow` are the two visibility terms.

        The bill is a handful of small polygon fills plus the three full-frame
        operations the pass always cost. Everything static -- which is every
        torch in the room -- comes out of a cache and costs one blit.
        """
        j = world.juice
        # From the authored ambient at contrast 1 to no ambient at all at 10,
        # linearly, so the last part of the slider stays useful instead of
        # approaching black without ever arriving.
        contrast = max(j.p("light_contrast"), 0.01)
        ambient_scale = 1.0 - clamp((contrast - 1.0) / 9.0)
        gain = lerp(0.65, 1.35, clamp(contrast * 0.5))
        height = max(2.0, j.p("light_height"))
        # Under one, the shadows fade out with everything else: a room with no
        # contrast in it should not have hard-edged shadows lying around.
        shadow_mix = min(contrast, 1.0)
        wall_amt = clamp(j.p("wall_shadow_amt")) * shadow_mix
        npc_amt = clamp(j.p("npc_shadow_amt")) * shadow_mix

        amb = int(255 * clamp(j.p("light_ambient") * ambient_scale))
        self.light_small.fill((amb, amb, amb))

        casters = []
        if npc_amt > 0.01:
            # The player carries the principal light, so only the other bodies
            # cast: a creature standing on its own light source would spend the
            # whole game inside its own shadow.
            for e in world.entities[1:]:
                if e.dying:
                    continue
                x, y = e.world_pos()
                casters.append((x, y, TILE * 0.32))

        for wx, wy, radius, strength in self.torch_lights(world):
            # Quantised to eight pixels, because the radius comes off a slider
            # and a gradient rebuilt every frame costs more than the whole rest
            # of the light pass.
            size = int(radius) // 8 * 8                  # half res, so *1
            if size < 8 or strength <= 0.01:
                continue
            sx, sy = self.to_view(world, wx, wy)
            sx, sy = sx * 0.5, sy * 0.5
            half = size * 0.5
            if not (-size < sx < VIEW_W * 0.5 + size and -size < sy < VIEW_H * 0.5 + size):
                continue
            blob = self._light_blob(size, height, gain)
            img = blob
            shadowed = wall_amt > 0.01 or (casters and npc_amt > 0.01)
            if shadowed:
                img = self._scratch(self._mask, size)
                if wall_amt > 0.01:
                    img.blit(self._wall_shadows(world, wx, wy, size, wall_amt),
                             (0, 0))
                else:
                    img.fill((255, 255, 255))
                if casters and npc_amt > 0.01:
                    # Multiplied in, not drawn in: two things in the way of the
                    # same light are darker than either, never lighter.
                    shade = self._cast_shadows(wx, wy, size, casters, npc_amt)
                    if shade is not None:
                        img.blit(shade, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
                img.blit(blob, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
            if strength < 0.97:
                if not shadowed:
                    img = blob.copy()
                img.fill((int(255 * clamp(strength)),) * 3,
                         special_flags=pygame.BLEND_RGB_MULT)
            self.light_small.blit(img, (int(sx - half), int(sy - half)),
                                  special_flags=pygame.BLEND_RGB_ADD)
        pygame.transform.smoothscale(self.light_small, (VIEW_W, VIEW_H), self.lightmap)
        self.view.blit(self.lightmap, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

        # A multiply on its own is not enough here, and the reason is worth
        # knowing: this floor is already almost black, so scaling it by a light
        # map moves it from 28 to 12 -- both of which read as "black". Multiply
        # can only ever take brightness *away*, and a torch is supposed to put
        # some back. So the same map is added on top at a low weight, which
        # gives a lit floor a visible warm pool instead of merely a less dark
        # one. Two full-frame ops rather than one, and worth every microsecond.
        warm = clamp(world.juice.p("light_warm"))
        if warm > 0.01:
            k = int(255 * warm)
            self.lightmap.fill((k, k, k), special_flags=pygame.BLEND_RGB_MULT)
            self.view.blit(self.lightmap, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    def _rake_ramp(self, bucket: int):
        """A one-sided gradient pointing at a light, as an alpha mask.

        Sixteen of these, built once each and stretched to whatever size a
        sprite happens to be this frame. The card generates a normal per texel
        out of the sprite's blurred alpha and takes a dot product with the
        light; at 32 pixels what that actually produces is a bright band down
        the side facing the light, falling off to nothing across the middle,
        and this is that band with the derivation left out.
        """
        hit = self._ramp_cache.get(bucket)
        if hit is not None:
            return hit
        n = 64
        surf = pygame.Surface((n, n), pygame.SRCALPHA)
        surf.fill((255, 255, 255, 0))
        ang = bucket * (2.0 * math.pi / 16.0)
        ux, uy = math.cos(ang), math.sin(ang)
        px, py = -uy, ux
        c = n * 0.5
        bands = 14
        for i in range(bands):
            t0, t1 = i / bands, (i + 1) / bands
            a = int(255 * ((t0 + t1) * 0.5) ** 1.35)
            d0, d1 = t0 * c * 1.5, t1 * c * 1.5
            quad = []
            for d, side in ((d0, 1), (d0, -1), (d1, -1), (d1, 1)):
                quad.append((c + ux * d + px * side * n,
                             c + uy * d + py * side * n))
            pygame.draw.polygon(surf, (255, 255, 255, a), quad)
        self._ramp_cache[bucket] = surf
        return surf

    @staticmethod
    def _rake_at(lights, wx: float, wy: float, height: float):
        """Which way the light comes from at a point, and how much of it.

        The sum of every light that reaches, weighted by its falloff and by how
        far it is *sideways*: a torch directly overhead has no direction to
        rake from, which is the same `light height` term the floor gradient
        uses and the reason raising the lights flattens the creatures too.
        """
        ax = ay = total = 0.0
        for lx, ly, radius, strength in lights:
            dx, dy = lx - wx, ly - wy
            d = math.hypot(dx, dy)
            if d < 1e-3 or d > radius:
                continue
            weight = (1.0 - d / radius) ** 2 * strength * (d / math.hypot(d, height))
            ax += dx / d * weight
            ay += dy / d * weight
            total += weight
        m = math.hypot(ax, ay)
        if total <= 1e-4 or m <= 1e-6:
            return 0.0, 0.0, 0.0
        return ax / m, ay / m, clamp(total)

    def apply_sprite_light(self, world: World):
        """Rake the light across every body drawn this frame.

        Runs *after* the light map has been multiplied over the view, and that
        ordering is the whole trick: the multiply is what puts a creature in a
        dark corner into the dark, and this is what says which side of it the
        torch is on. Done in the other order the highlight would simply be
        dimmed along with everything else.

        Two blits a body -- a warm one on the lit side, a cool one on the
        other -- both shaped by the sprite's own alpha, so nothing lands off
        the silhouette and no per-pixel work happens anywhere.
        """
        j = world.juice
        depth = clamp(j.p("normal_depth") * j.intensity, 0.0, 2.0)
        if depth <= 0.02 or not self._lit:
            return
        lights = self.torch_lights(world)
        height = max(2.0, j.p("light_height"))
        for img, sx, sy, wx, wy in self._lit:
            ux, uy, amt = self._rake_at(lights, wx, wy, height)
            if amt <= 0.02:
                continue
            k = clamp(amt * depth)
            bucket = int(round(math.atan2(uy, ux) / (2.0 * math.pi) * 16.0)) % 16
            size = img.get_size()
            rect = img.get_rect(center=(int(sx), int(sy)))

            lit = img.copy()
            lit.fill((int(104 * k), int(88 * k), int(62 * k)),
                     special_flags=pygame.BLEND_RGB_ADD)
            lit.blit(pygame.transform.scale(self._rake_ramp(bucket), size),
                     (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            self.view.blit(lit, rect)

            # The far side, cooled rather than blackened: a silhouette that
            # goes to black at this size reads as a hole in the floor, which is
            # the same reason the card's shading is half-lambert.
            shade = 1.0 - clamp(k * 0.45)
            dark = img.copy()
            dark.fill((int(255 * shade), int(255 * min(1.0, shade * 1.04)),
                       int(255 * min(1.0, shade * 1.14)), 255),
                      special_flags=pygame.BLEND_RGBA_MULT)
            dark.blit(pygame.transform.scale(self._rake_ramp((bucket + 8) % 16),
                                             size),
                      (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            self.view.blit(dark, rect)

    def apply_bloom(self, world: World):
        """Downscale, threshold, upscale, add. Pygame's answer to a glow shader.

        The downscale *is* the blur -- a box filter over eight pixels -- which
        is why this costs two smoothscales and no per-pixel work. The threshold
        matters more than the strength: without one, the whole frame hazes over
        and the effect reads as a dirty screen rather than as light.
        """
        j = world.juice
        thresh = int(clamp(j.p("bloom_thresh"), 0, 255))
        amount = clamp(j.p("bloom_amt") * j.intensity, 0.0, 1.5)
        if amount <= 0.01:
            return
        pygame.transform.smoothscale(self.view, self.bloom_small.get_size(),
                                     self.bloom_small)
        if thresh:
            # Subtract the threshold and scale what is left back up, so a pixel
            # just over the line still glows faintly instead of jumping in.
            self.bloom_small.fill((thresh, thresh, thresh),
                                  special_flags=pygame.BLEND_RGB_SUB)
        k = int(255 * min(amount, 1.0))
        self.bloom_small.fill((k, k, k), special_flags=pygame.BLEND_RGB_MULT)
        pygame.transform.smoothscale(self.bloom_small, (VIEW_W, VIEW_H),
                                     self.bloom_big)
        self.view.blit(self.bloom_big, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        if amount > 1.0:                       # a second pass, for the overdone end
            self.view.blit(self.bloom_big, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    def draw_haze(self, world: World):
        """Rows re-blitted at sine offsets, in a band above each torch.

        The same trick the slime's bands use, pointed at the background instead
        of at a sprite: pygame cannot displace pixels, but it can blit a row of
        them somewhere else. Kept local to the torches because a full-screen
        version costs six hundred blits and looks like a fault.
        """
        j = world.juice
        amp = j.p("haze_amt") * j.intensity
        if amp <= 0.05:
            return
        t = world.elapsed
        for kind, wx, wy, phase in world.props:
            if kind != "torch":
                continue
            sx, sy = self.to_view(world, wx, wy)
            band = pygame.Rect(int(sx) - TILE, int(sy) - TILE * 3, TILE * 2, TILE * 3)
            band = band.clip(self.view.get_rect())
            if band.w <= 2 or band.h <= 2:
                continue
            src = self.view.subsurface(band).copy()
            for row in range(band.h):
                # Rising as well as wobbling: the offset is sampled further
                # along the wave the higher up the band you go, which reads as
                # heat going up rather than as the wall shivering.
                up = 1.0 - row / band.h
                off = math.sin(t * 5.0 + phase + row * 0.16) * amp * up
                self.view.blit(src, (band.x + off, band.y + row),
                               pygame.Rect(0, row, band.w, 1))

    def draw_corpses(self, world: World):
        for c in world.corpses:
            col, row = tiles.SPRITES.get(c.sprite, tiles.SPRITES["goblin"])
            img = self.sheet.sprite(col, row, c.color if c.tint else None)
            img = pygame.transform.rotate(img, c.angle).copy()
            img.set_alpha(int(255 * clamp(c.alpha)))
            sx, sy = self.to_view(world, c.x, c.y)
            self.view.blit(img, img.get_rect(center=(int(sx), int(sy))))

    def draw_entities(self, world: World):
        j = world.juice
        mode = j.choice("pixel_mode")
        for e in world.entities:
            if j.on("shadow"):
                self.draw_shadow(world, e)
            if e.tail is not None and j.on("tail"):
                self.draw_tail(world, e)
            self.draw_trail(world, e)
            if j.on("outline"):
                self.draw_outline(world, e)
            self.draw_body(world, e, mode)

    def draw_shadow(self, world: World, e: Entity):
        """A blob under the body that shrinks and fades as it rises.

        Stays on the tile the entity logically occupies: it takes the sideways
        offset but never the lift, because a shadow that rises with the body is
        a second sprite rather than a shadow.
        """
        if e.dying:
            return
        j = world.juice
        ox, size, alpha = shadow_of(e.body, base=j.p("shadow_size"))
        alpha *= j.p("shadow_alpha") * j.intensity
        if alpha <= 0.02 or size <= 0.02:
            return
        w = max(2, int(TILE * size * 1.25))
        h = max(2, int(TILE * size * 0.52))
        img = pygame.transform.scale(self._shadow, (w, h)).copy()
        img.set_alpha(int(255 * clamp(alpha)))
        wx = (e.body.tx + 0.5 + ox) * TILE
        wy = (e.body.ty + 0.5) * TILE + TILE * 0.36
        sx, sy = self.to_view(world, wx, wy)
        self.view.blit(img, img.get_rect(center=(int(sx), int(sy))))

    def draw_tail(self, world: World, e: Entity):
        """The spring chain, as a polygon tapering from root to tip.

        Drawn as one shape rather than as segments so it reads as a continuous
        thing; the taper is what makes the tip look light enough to whip.
        """
        pts = e.tail.points(e.body)
        if len(pts) < 2:
            return
        screen = [self.to_view(world, x * TILE, y * TILE) for x, y in pts]
        left, right = [], []
        n = len(screen)
        for i, (sx, sy) in enumerate(screen):
            t = i / (n - 1)
            w = max(1.0, (1.0 - t) ** 0.8 * TILE * 0.22)
            # Normal to the local direction of the chain.
            j0 = max(0, i - 1)
            j1 = min(n - 1, i + 1)
            dx = screen[j1][0] - screen[j0][0]
            dy = screen[j1][1] - screen[j0][1]
            d = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / d * w, dx / d * w
            left.append((sx + nx, sy + ny))
            right.append((sx - nx, sy - ny))
        pygame.draw.polygon(self.view, e.tail_color, left + right[::-1])

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

    def draw_outline(self, world: World, e: Entity):
        """The silhouette, four times, one pixel out in each direction.

        The cheap software substitute for a rim-light shader, and the reason to
        want one: at 32px against a busy floor a creature and the tile behind it
        can be the same value, and a dark edge fixes it for four blits.
        """
        b = e.body
        img = self._transform(self.silhouette(e), b.sx, b.sy, b.angle).copy()
        img.set_alpha(int(255 * clamp(world.juice.p("outline_alpha") * b.alpha)))
        wx, wy = e.world_pos()
        sx, sy = self.to_view(world, wx, wy)
        r = img.get_rect(center=(int(sx), int(sy)))
        for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            self.view.blit(img, (r.x + ox, r.y + oy))

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

    def draw_body(self, world: World, e: Entity, mode: str = "rounded"):
        b = e.body
        img = self._transform(self.sprite_for(e), b.sx, b.sy, b.angle)

        if b.flash > 0.0 or (b.alpha < 1.0 and mode != "subpixel"):
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
            self._blit_at(img, sx, sy, mode,
                          b.alpha if mode == "subpixel" else 1.0)
        # Kept for the sprite lighting, which cannot run until the light map
        # has been multiplied over the frame. Solid bodies only: a corpse on
        # its way out is already fading, and a rim light on something
        # disappearing reads as it coming back.
        if b.alpha > 0.99 and world.juice.on("light") and world.juice.on("normals"):
            self._lit.append((img, sx, sy, wx, wy))

    def draw_bars(self, world: World):
        """Health bars, after the light pass so a dark corner does not hide them.

        The ghost is the effect: a pale bar left where the health used to be,
        held for a moment and then drained, so the *amount* taken is legible as
        a width instead of being inferred from a bar that is now shorter than
        it was the last time anyone looked.
        """
        chipping = world.juice.on("chip")
        for e in world.entities:
            # No bar on things that cannot lose the fight: the player and the
            # training dummy are both invincible, and a full bar that never
            # moves is just clutter.
            if e.dying or e.invincible:
                continue
            wx, wy = e.world_pos()
            sx, sy = self.to_view(world, wx, wy - TILE * 0.55)
            w = TILE - 12
            rect = pygame.Rect(0, 0, w, 3)
            rect.center = (int(sx), int(sy))
            pygame.draw.rect(self.view, (60, 40, 44), rect)
            if chipping and e.chip.ghost > e.chip.value:
                pygame.draw.rect(self.view, (232, 214, 210),
                                 pygame.Rect(rect.left, rect.top,
                                             int(w * clamp(e.chip.ghost)), 3))
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
        whole view-sized scratch for every slash costs more than everything
        else in the frame put together -- so only the shape's own bounding box
        is cleared and blitted, which is typically a couple of tiles' worth.
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
            sx, sy, sr = world.trauma.offset(max_px=j.p("shake_px") * PX * j.intensity,
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
            self.vignette.set_alpha(int(clamp(j.p("vig_base"), 0, 200)
                                        + 130 * clamp(world.vignette_pulse)))
            screen.blit(self.vignette, (0, 0))

    def _apply_rgb_split(self, amount: float):
        """Chromatic aberration, in place on the view.

        Pull red and blue into scratch buffers, mask the view down to green,
        then add the two back at opposite offsets. Five full-frame operations
        and no allocation -- still the most expensive thing in this file, which
        is why it is a toggle and why it only runs for the handful of frames
        after a heavy hit.
        """
        d = int(clamp(amount, 0.0, 20.0))
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
    def panel_rect(self) -> pygame.Rect:
        """The scrolling region, between the fixed header and the fixed footer."""
        return pygame.Rect(VIEW_W, HEADER_H, PANEL_W, WIN_H - FOOTER_H - HEADER_H)

    def draw_panel(self, screen, world: World, mouse):
        """Header, a scrolling list of rows, and a footer blurb.

        The list is drawn under a clip rectangle and offset by `scroll`, and hit
        testing is registered from the same loop that draws -- so a row can
        never be clickable at a position it is not drawn at, which is the one
        bug this kind of panel always has.
        """
        j = world.juice
        x0 = VIEW_W
        screen.fill(PANEL_BG, pygame.Rect(x0, 0, PANEL_W, WIN_H))
        pygame.draw.line(screen, PANEL_LINE, (x0, 0), (x0, WIN_H))

        # -- header -------------------------------------------------------
        screen.blit(self.font_big.render("JUICE", True, INK), (x0 + 14, 4))
        screen.blit(self.font_ui_b.render("ON" if j.master else "OFF (Tab)", True,
                                          ACCENT if j.master else WARN), (x0 + 78, 9))
        cost = f"{self.frame_ms:4.1f} ms"
        # Red once a frame costs more than a 60Hz slot.
        cost_col = WARN if self.frame_ms > 16.7 else DIM
        screen.blit(self.font_ui.render(cost, True, cost_col),
                    (x0 + PANEL_W - 14 - self.font_ui.size(cost)[0], 9))
        pygame.draw.line(screen, PANEL_LINE, (x0 + 10, HEADER_H - 2),
                         (x0 + PANEL_W - 10, HEADER_H - 2))

        # -- the scrolling list -------------------------------------------
        view = self.panel_rect()
        self.rows.clear()
        self.hover = None
        mx, my = mouse
        inside = view.collidepoint(mx, my)
        screen.set_clip(view)

        left = x0 + 12
        width = PANEL_W - 26
        y = view.top - int(self.scroll)
        for group in GROUPS:
            y = self._draw_group_header(screen, group, left, width, y)
            for t in j.toggles.values():
                if t.group != group:
                    continue
                y = self._draw_toggle(screen, j, t, left, width, y, view,
                                      (mx, my) if inside else None)
            for c in j.choices.values():
                if c.group != group:
                    continue
                y = self._draw_choice(screen, c, left, width, y, view,
                                      (mx, my) if inside else None)
                if c.key == "move_ease":
                    y = self._draw_curve(screen, left, y, EASINGS[c.value], width)
            for p in j.params.group(group):
                y = self._draw_slider(screen, p, left, width, y, view,
                                      (mx, my) if inside else None)
            y += 8

        screen.set_clip(None)
        self.content_h = y + int(self.scroll) - view.top
        self._draw_scrollbar(screen, view)

        # -- footer -------------------------------------------------------
        footer = WIN_H - FOOTER_H
        pygame.draw.line(screen, PANEL_LINE, (x0 + 10, footer), (x0 + PANEL_W - 10, footer))
        ty = footer + 6
        if self.hover:
            kind, key = self.hover
            item = (j.toggles[key] if kind == "t" else
                    j.choices[key] if kind == "c" else j.params[key])
            head = item.label
            if kind == "p":
                head = f"{item.label}   {item.text}   ({item.lo:g} - {item.hi:g})"
            elif kind == "c":
                head = f"{item.label}   {item.value}"
            screen.blit(self.font_ui_b.render(head, True, GOLD), (x0 + 14, ty))
            ty += 15
            for line in self._wrap(item.blurb, PANEL_W - 30, self.font_ui):
                screen.blit(self.font_ui.render(line, True, DIM), (x0 + 14, ty))
                ty += 13
        else:
            for line in self.help_lines:
                if line:
                    screen.blit(self.font_ui.render(line, True, DIM), (x0 + 14, ty))
                ty += 13

    def _draw_group_header(self, screen, group, x, w, y):
        if y > -20 and y < WIN_H:
            screen.blit(self.font_ui_b.render(group.upper(), True, ACCENT), (x, y + 2))
            pygame.draw.line(screen, PANEL_LINE, (x, y + 16), (x + w, y + 16))
        return y + 20

    def _register(self, rect, kind, key, view, mouse):
        """Record a row for hit testing, clipped to what is actually on screen.

        The clip matters. A row half-scrolled under the footer is *drawn* half
        a row tall, and registering its full height would leave a strip of the
        panel that responds to clicks with no visible control on it -- the
        classic scrolling-list bug. Clipping keeps the two in step; the width is
        untouched, so a half-visible slider still drags correctly.
        """
        hit = rect.clip(view)
        if hit.w <= 0 or hit.h <= 0:
            return False
        self.rows.append((hit, kind, key))
        if mouse and hit.collidepoint(mouse):
            self.hover = (kind, key)
            return True
        return False

    def _draw_toggle(self, screen, j, t, x, w, y, view, mouse):
        rect = pygame.Rect(x - 3, y, w + 6, 15)
        hot = self._register(rect, "t", t.key, view, mouse)
        if rect.bottom >= view.top and rect.top <= view.bottom:
            if hot:
                pygame.draw.rect(screen, (32, 34, 44), rect, border_radius=3)
            label_col = INK if t.on else (78, 84, 100)
            if not j.master:
                label_col = (58, 61, 72)
            screen.blit(self.font_ui.render("[x]" if t.on else "[ ]", True,
                                            ACCENT if t.on else (70, 74, 90)), (x + 2, y + 1))
            screen.blit(self.font_ui.render(t.label, True, label_col), (x + 30, y + 1))
        return y + 15

    def _draw_choice(self, screen, c, x, w, y, view, mouse):
        rect = pygame.Rect(x - 3, y, w + 6, 15)
        hot = self._register(rect, "c", c.key, view, mouse)
        if rect.bottom >= view.top and rect.top <= view.bottom:
            if hot:
                pygame.draw.rect(screen, (32, 34, 44), rect, border_radius=3)
            screen.blit(self.font_ui.render("<->", True, GOLD), (x + 2, y + 1))
            screen.blit(self.font_ui.render(c.label, True, INK), (x + 30, y + 1))
            val = self.font_ui.render(c.value, True, ACCENT)
            screen.blit(val, (x + w - val.get_width() - 2, y + 1))
        return y + 15

    def _draw_slider(self, screen, p, x, w, y, view, mouse):
        """Label, track and value on one 16px line.

        A slider that needs three lines does not fit ninety of them in a panel,
        so the track *is* the row: the fill runs behind the label and the value
        sits at the right-hand end. Dragging anywhere on the row sets it.
        """
        rect = pygame.Rect(x - 3, y, w + 6, 16)
        hot = self._register(rect, "p", p.key, view, mouse)
        if rect.bottom < view.top or rect.top > view.bottom:
            return y + 16
        track = pygame.Rect(x, y + 1, w, 13)
        pygame.draw.rect(screen, TRACK_BG, track, border_radius=2)
        fill = track.copy()
        fill.w = max(1, int(w * p.norm))
        pygame.draw.rect(screen, TRACK_FILL if not hot else (58, 100, 94), fill,
                         border_radius=2)
        # The handle: a bright rule at the value, so a slider near either end is
        # still readable when the fill has nothing to show.
        hx = track.x + fill.w
        pygame.draw.line(screen, ACCENT if hot else (90, 160, 148),
                         (hx, track.top), (hx, track.bottom - 1), 2)
        screen.blit(self.font_ui.render(p.label, True, INK if hot else (176, 184, 202)),
                    (x + 5, y + 2))
        val = self.font_ui.render(p.text, True, GOLD)
        screen.blit(val, (x + w - val.get_width() - 5, y + 2))
        return y + 16

    def _draw_curve(self, screen, x, y, fn, w):
        """A little plot of the active easing curve. Seeing the overshoot in
        `out_back` as a line makes the on-screen snap far easier to reason about
        than staring at the sprite does."""
        h = 40
        if y + h >= HEADER_H and y <= WIN_H:
            pygame.draw.rect(screen, (26, 28, 36), (x, y, w, h), border_radius=3)
            pygame.draw.line(screen, PANEL_LINE, (x, y + h - 7), (x + w, y + h - 7))
            pts = [(x + i, y + h - 7 - fn(i / (w - 1)) * (h - 14)) for i in range(w)]
            pygame.draw.lines(screen, ACCENT, False, pts, 2)
        return y + h + 4

    def _draw_scrollbar(self, screen, view):
        if self.content_h <= view.h:
            return
        frac = view.h / self.content_h
        h = max(24, int(view.h * frac))
        travel = view.h - h
        t = clamp(self.scroll / max(1.0, self.content_h - view.h))
        bar = pygame.Rect(view.right - 5, view.top + int(travel * t), 3, h)
        pygame.draw.rect(screen, (60, 64, 82), bar, border_radius=2)

    def scroll_by(self, dy: float, world: World):
        view = self.panel_rect()
        self.scroll = max(0.0, min(self.scroll + dy,
                                   max(0.0, self.content_h - view.h)))

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
        # The pacing streak, as a bar: the one piece of state in the bench that
        # changes how the game plays and has nothing on screen to show for it.
        if world.juice.on("pacing") and world.pacer.streak > 0.01:
            w = int(90 * world.pacer.streak)
            pygame.draw.rect(screen, (40, 44, 56), (16, VIEW_H - 22, 90, 4))
            pygame.draw.rect(screen, ACCENT, (16, VIEW_H - 22, w, 4))

    # -- panel input ------------------------------------------------------
    def panel_hit(self, pos):
        for rect, kind, key in self.rows:
            if rect.collidepoint(pos):
                return kind, key, rect
        return None

    def panel_click(self, world: World, pos, button: int = 1) -> bool:
        hit = self.panel_hit(pos)
        if hit is None:
            return False
        kind, key, rect = hit
        j = world.juice
        if kind == "t":
            t = j.toggles[key]
            t.on = not t.on
            world.say(f"{t.label}: {'on' if t.on else 'off'}",
                      ACCENT if t.on else DIM)
            world.sfx("ui_on" if t.on else "ui_off")
        elif kind == "c":
            c = j.choices[key]
            c.cycle(-1 if button == 3 else 1)
            world.say(f"{c.label}: {c.value}", ACCENT)
            world.sfx("ui_on")
        elif kind == "p":
            p = j.params[key]
            if button == 3:
                p.reset()
                world.say(f"{p.label}: {p.text} (default)", DIM)
            else:
                self.drag = key
                self._drag_to(j, key, pos, rect)
        return True

    def _drag_to(self, juice, key, pos, rect=None):
        if rect is None:
            for r, kind, k in self.rows:
                if kind == "p" and k == key:
                    rect = r
                    break
        if rect is None:
            return
        # The track is inset by three pixels from the registered row.
        juice.params[key].set_norm((pos[0] - (rect.x + 3)) / max(1, rect.w - 6))

    def panel_drag(self, world: World, pos):
        if self.drag is not None:
            self._drag_to(world.juice, self.drag, pos)


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


def handle_event(event, world: World, renderer: Renderer) -> bool:
    """Handle one event. Returns False to quit."""
    j = world.juice
    if event.type == pygame.QUIT:
        return False
    if event.type == pygame.MOUSEWHEEL:
        renderer.scroll_by(-event.y * 38, world)
        return True
    if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
        renderer.panel_click(world, event.pos, event.button)
        return True
    if event.type == pygame.MOUSEBUTTONUP:
        renderer.drag = None
        return True
    if event.type == pygame.MOUSEMOTION:
        renderer.panel_drag(world, event.pos)
        return True
    if event.type == pygame.JOYDEVICEADDED:          # pragma: no cover
        world.pad = pygame.joystick.Joystick(event.device_index)
        world.pad.init()
        world.say("gamepad connected -- rumble live", ACCENT)
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
    elif event.key == pygame.K_p:
        c = j.choices["pixel_mode"]
        world.say(f"pixel mode: {c.cycle()}", ACCENT)
    elif event.key == pygame.K_m:
        t = j.toggles["music"]
        t.on = not t.on
        if t.on and world.audio is not None and not world.audio.music_ready:
            world.audio.load_music()
        world.say("music on" if t.on else "music off", ACCENT if t.on else DIM)
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
    elif event.key == pygame.K_F3:
        j.defaults()
        world.say("sliders back to defaults", ACCENT)
    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        j.intensity = j.intensity - 0.1
    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
        j.intensity = j.intensity + 0.1
    elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
        c = j.choices["move_ease"]
        c.cycle(1 if event.key == pygame.K_RIGHTBRACKET else -1)
        world.say(f"move ease: {c.value}", ACCENT)
    return True


def draw_frame(screen, renderer: Renderer, world: World, mouse=(0, 0)):
    """One complete frame, factored out so the headless path uses it too."""
    renderer.draw_world(world)
    renderer.composite(screen, world)
    renderer.draw_hud(screen, world)
    renderer.draw_panel(screen, world, mouse)


def make_audio(enabled: bool = True):
    """Bring the sound bank up, or return None and let the bench run silent."""
    if not HAVE_AUDIO or not enabled:
        return None
    bank = audiofx.SoundBank()
    bank.init_mixer()
    bank.load_all()
    return bank


def run():
    # The mixer has to be configured before pygame.init() brings it up with the
    # default 4096-sample buffer, which is 93ms of latency -- nearly six frames,
    # and enough to make a perfectly timed hit feel mushy.
    audio = make_audio()
    pygame.init()
    pygame.joystick.init()
    pygame.display.set_caption("juice workbench")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.key.set_repeat(180, 70)
    clock = pygame.time.Clock()

    world = World(Juice(), audio)
    if pygame.joystick.get_count():                  # pragma: no cover
        world.pad = pygame.joystick.Joystick(0)
        world.pad.init()
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

    if audio is not None:
        audio.stop_all()
    pygame.quit()


def run_headless(path: str):
    """Play a scripted swing with no window and save the frame at impact.

    This is the regression test for the render path: it drives the same draw
    code the interactive loop does, so a crash in compositing shows up here
    rather than the first time somebody opens the bench.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    juice = Juice()
    # Everything on, including the ones that default off, so the headless run
    # exercises every draw path rather than the cheap subset.
    juice.set_all(True)
    world = World(juice, None)
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
        step(14)
        if world.player.tile == (px, py):
            break                             # blocked; near enough for a frame
        px, py = world.player.tile
    world.try_move(world.player, -1, 0)        # walks into the dummy -> attack
    # Stop a few frames past contact, with the sparks still in the air.
    step(int((juice.p("windup_time") + juice.p("attack_time") * 0.32) / dt) + 5)

    draw_frame(screen, renderer, world, (WIN_W - 200, 300))
    pygame.image.save(screen, path)
    pygame.quit()
    print(f"wrote {path} ({len(world.fx)} live effects, "
          f"{len(world.decals)} decals at the moment of impact)")


def main():
    if "--headless" in sys.argv:
        i = sys.argv.index("--headless")
        path = sys.argv[i + 1] if len(sys.argv) > i + 1 else "juice_headless.png"
        run_headless(path)
    else:
        run()


if __name__ == "__main__":
    main()
