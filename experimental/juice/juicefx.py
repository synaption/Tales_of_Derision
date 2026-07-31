"""The juice: every trick this workbench knows, as maths and nothing else.

There is not a line of pygame in here. That is deliberate -- the whole point
of the exercise is that "juice" is a presentation layer that sits *on top of*
a turn-based sim without being tangled into it, and the cleanest way to prove
that is to make the effects run, and be tested, with no display anywhere.
`juicetest.py` exercises the lot of it headless.

The governing idea is one sentence long: **the sim is instant, the picture is
late.** When the player walks north the logical position changes on that frame,
finally and completely; the sim is free to path, spot and swing on the new
tile immediately. What the eye sees is a body still standing on the old tile
and hurrying to catch up, and every effect in this file is a variation on how
it hurries.

That gives the arrangement its shape:

    Body      where a thing is drawn. Logical tile coords, plus a set of
              offsets the juice layer scribbles on: shift, scale, spin, fade,
              whiten. Reset to identity every frame.
    Motion    something that scribbles on a Body for a while and then reports
              that it is finished. `Hop`, `Lunge`, `Knockback`, `DeathSpin`...
    Animator  holds the Motions running on one Body, wipes the Body's juice
              fields at the top of each frame and lets every Motion contribute.

Because the Body is wiped and then *accumulated* into -- offsets add, scales
multiply, whiteness takes the max -- Motions compose without knowing about each
other. A hop and a hit-flash and an idle breath all land on the same body in
the same frame and none of them has to check what the others did. That is the
composition-over-inheritance rule doing real work rather than being a slogan:
there is no `HoppingFlashingBreathingSprite` class, there are three small
Motions in a list.

`Sequence` and `Parallel` are Motions too, so an attack is spelled out as a
plain nested structure -- wind up, then lunge out, then *at the moment of
contact* fire a callback that hurts the target, then settle back.

The rest is the ambient furniture that is not attached to any one body:
`Trauma` (screen shake with a decaying budget), `Spring` (damped springs for
camera punch), and the small self-updating pools -- particles, floating damage
numbers, expanding shockwaves, tile ripples.

Every duration is in seconds, every distance on a Body is in *tiles* (so the
effects survive a change of tile size), and every random-looking thing is
driven by a seeded hash rather than `random`, so two runs with the same input
draw the same picture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence as Seq, Tuple


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------
#
# An easing function maps a progress value in 0..1 to an eased 0..1. This is
# the smallest and highest-leverage piece of juice there is: the difference
# between a move that feels like a spreadsheet and one that feels alive is
# very often nothing more than swapping `linear` for `ease_out_back`.
#
# The families, and what they read as on screen:
#
#   in     starts slow, ends fast    -- weight, wind-up, things falling
#   out    starts fast, ends slow    -- arrival, impact, most UI motion
#   inout  slow at both ends         -- deliberate, considered movement
#   back   overshoots and returns    -- snap, confidence, cartoon anticipation
#   elastic wobbles into place       -- rubbery, comedic
#   bounce  lands, bounces, settles  -- heavy, physical


def linear(t: float) -> float:
    return t


def ease_in_quad(t: float) -> float:
    return t * t


def ease_out_quad(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


def ease_in_out_quad(t: float) -> float:
    return 2 * t * t if t < 0.5 else 1.0 - (-2 * t + 2) ** 2 / 2


def ease_in_cubic(t: float) -> float:
    return t * t * t


def ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    return 4 * t * t * t if t < 0.5 else 1.0 - (-2 * t + 2) ** 3 / 2


def ease_out_quint(t: float) -> float:
    return 1.0 - (1.0 - t) ** 5


def ease_in_back(t: float, overshoot: float = 1.70158) -> float:
    c = overshoot + 1.0
    return c * t * t * t - overshoot * t * t


def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    """Overshoots the target and comes back. The single most useful easing."""
    c = overshoot + 1.0
    return 1.0 + c * (t - 1.0) ** 3 + overshoot * (t - 1.0) ** 2


def ease_out_elastic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    p = 2 * math.pi / 3
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * p) + 1.0


def ease_out_bounce(t: float) -> float:
    n, d = 7.5625, 2.75
    if t < 1 / d:
        return n * t * t
    if t < 2 / d:
        t -= 1.5 / d
        return n * t * t + 0.75
    if t < 2.5 / d:
        t -= 2.25 / d
        return n * t * t + 0.9375
    t -= 2.625 / d
    return n * t * t + 0.984375


def ease_in_bounce(t: float) -> float:
    return 1.0 - ease_out_bounce(1.0 - t)


#: Name -> function, for the workbench's easing gallery and for saved presets.
EASINGS: dict[str, Callable[[float], float]] = {
    "linear": linear,
    "in_quad": ease_in_quad,
    "out_quad": ease_out_quad,
    "in_out_quad": ease_in_out_quad,
    "in_cubic": ease_in_cubic,
    "out_cubic": ease_out_cubic,
    "in_out_cubic": ease_in_out_cubic,
    "out_quint": ease_out_quint,
    "in_back": ease_in_back,
    "out_back": ease_out_back,
    "out_elastic": ease_out_elastic,
    "out_bounce": ease_out_bounce,
    "in_bounce": ease_in_bounce,
}


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def damp(current: float, target: float, smoothing: float, dt: float) -> float:
    """Frame-rate independent exponential approach.

    The naive `x += (target - x) * 0.1` every frame is the most common bug in
    juice code: it converges faster on a fast machine than a slow one, so the
    game *feels different* depending on the frame rate. `smoothing` here is the
    fraction of the remaining distance still left after one full second, which
    makes the result independent of how the second was chopped up.
    """
    return target + (current - target) * (smoothing ** dt)


# ---------------------------------------------------------------------------
# Deterministic noise
# ---------------------------------------------------------------------------


def hash01(i: int, seed: int = 0) -> float:
    """Integer -> pseudo-random float in -1..1. No global RNG state touched."""
    h = (i * 374761393 + seed * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    return (h / 0xFFFFFFFF) * 2.0 - 1.0


def value_noise(t: float, seed: int = 0) -> float:
    """Smooth 1D noise in -1..1, sampled at integer steps of `t`.

    Screen shake wants *noise*, not `random()` per frame. Independent random
    samples give a buzzing, high-frequency fuzz that reads as a broken cable;
    smoothly interpolated noise gives a swing with a direction to it, which
    reads as a blow landing. It is the same difference as between static and a
    struck bell.
    """
    i = math.floor(t)
    f = t - i
    u = f * f * (3.0 - 2.0 * f)  # smoothstep
    return lerp(hash01(i, seed), hash01(i + 1, seed), u)


# ---------------------------------------------------------------------------
# Body: the thing the juice is applied to
# ---------------------------------------------------------------------------


@dataclass
class Body:
    """Where and how one entity is drawn this frame.

    `tx`/`ty` are the logical tile the sim believes the entity occupies, and
    they jump instantly. Everything below them is presentation, is wiped to
    identity at the top of every frame by `Animator.update`, and is rebuilt by
    whatever Motions are running.

    The units matter: `ox`/`oy` are in **tiles**, not pixels, so an effect
    tuned at 32px still looks right at 64px.
    """

    tx: float = 0.0
    ty: float = 0.0

    ox: float = 0.0      # visual offset from the tile centre, in tiles
    oy: float = 0.0
    sx: float = 1.0      # scale, 1.0 = natural size
    sy: float = 1.0
    angle: float = 0.0   # degrees, counter-clockwise
    alpha: float = 1.0   # 0 = invisible
    flash: float = 0.0   # 0..1, how far towards solid white the glyph is tinted

    facing: Tuple[int, int] = (0, 1)

    def reset_juice(self) -> None:
        """Return every presentation field to "no effect applied"."""
        self.ox = self.oy = 0.0
        self.sx = self.sy = 1.0
        self.angle = 0.0
        self.alpha = 1.0
        self.flash = 0.0

    # The four accumulate operations. Motions only ever touch a Body through
    # these, which is what lets any number of them run at once without one
    # stamping on another's work.
    def shift(self, dx: float, dy: float) -> None:
        self.ox += dx
        self.oy += dy

    def scale(self, sx: float, sy: float) -> None:
        self.sx *= sx
        self.sy *= sy

    def rotate(self, deg: float) -> None:
        self.angle += deg

    def fade(self, a: float) -> None:
        self.alpha = min(self.alpha, a)

    def whiten(self, f: float) -> None:
        self.flash = max(self.flash, f)


# ---------------------------------------------------------------------------
# Motions
# ---------------------------------------------------------------------------


class Motion(Protocol):
    """Anything that scribbles on a Body over time.

    `update` is handed the Body and the elapsed seconds and returns True when
    it is finished and should be dropped. That is the entire contract -- a
    Motion needs no base class, so new ones can be a six-line dataclass.
    """

    def update(self, body: Body, dt: float) -> bool: ...


@dataclass
class _Timed:
    """Shared clock for the fixed-length Motions. Not part of the protocol."""

    duration: float = 0.25
    elapsed: float = 0.0

    def tick(self, dt: float) -> Tuple[float, bool]:
        """Advance and return (progress 0..1, finished)."""
        self.elapsed += dt
        if self.duration <= 0.0:
            return 1.0, True
        p = self.elapsed / self.duration
        return (1.0, True) if p >= 1.0 else (p, False)


@dataclass
class Wait(_Timed):
    """Does nothing for a while. Only useful inside a `Sequence`, where it is
    the anticipation beat -- the held breath before the swing that makes the
    swing read as a decision rather than a twitch."""

    def update(self, body: Body, dt: float) -> bool:
        _, done = self.tick(dt)
        return done


@dataclass
class Callback:
    """Fires a function the instant it is reached and finishes immediately.

    This is how the sim gets spliced into the middle of an animation: the
    damage lands on the frame the fist arrives, not on the frame the button
    was pressed, and the whole schedule stays readable as one `Sequence`.
    """

    fn: Callable[[], None]

    def update(self, body: Body, dt: float) -> bool:
        self.fn()
        return True


@dataclass
class Sequence:
    """Runs child Motions one after another."""

    children: List[Motion] = field(default_factory=list)
    index: int = 0

    def update(self, body: Body, dt: float) -> bool:
        # A zero-length child (Callback, Wait(0)) must not eat a whole frame,
        # so keep consuming children until one of them wants time.
        while self.index < len(self.children):
            if self.children[self.index].update(body, dt):
                self.index += 1
                dt = 0.0  # the rest of this frame's time is already spent
                continue
            return False
        return True

    @property
    def done(self) -> bool:
        return self.index >= len(self.children)


@dataclass
class Parallel:
    """Runs child Motions together; finishes when the last one does."""

    children: List[Motion] = field(default_factory=list)

    def update(self, body: Body, dt: float) -> bool:
        self.children = [c for c in self.children if not c.update(body, dt)]
        return not self.children


@dataclass
class Hop(_Timed):
    """A move between two tiles, arced and with squash and stretch.

    The logical move has *already happened* -- `dx`/`dy` say which way the
    entity came from, in tiles, and the Motion's job is to drag the picture
    back there and let it catch up. So the offset starts at the old tile and
    eases to zero.

    Three things are happening at once and each is doing separate work:

    * the horizontal ease (`out_quad`) is the travel;
    * `height`, a half sine, lifts it off the floor, which is what makes a
      step read as a *hop* rather than a slide;
    * the squash term stretches the body along its direction of travel at the
      peak and squashes it flat at both ends. Volume is roughly conserved --
      stretched thin is also taller -- because that is the rule the eye knows
      from every cartoon it has ever seen, and breaking it looks like a bug.
    """

    dx: float = 0.0           # tiles from the *old* position to the new one
    dy: float = 0.0
    height: float = 0.35      # peak lift, in tiles
    squash: float = 0.25      # 0 disables the deformation
    ease: Callable[[float], float] = ease_out_quad

    def update(self, body: Body, dt: float) -> bool:
        p, done = self.tick(dt)
        e = self.ease(p)
        # Trail back towards where we came from, shrinking to nothing.
        body.shift(-self.dx * (1.0 - e), -self.dy * (1.0 - e))
        body.shift(0.0, -self.height * math.sin(math.pi * p))

        if self.squash:
            # +1 at the apex (stretched), -1 at take-off and landing (squashed).
            s = -math.cos(2.0 * math.pi * p)
            k = self.squash * s
            horizontal = abs(self.dx) > abs(self.dy)
            if horizontal:
                body.scale(1.0 + k, 1.0 - k * 0.6)
            else:
                body.scale(1.0 - k * 0.6, 1.0 + k)
            # Squashing about the centre would make a landing look like it
            # sank into the floor; nudge down so the feet stay put instead.
            body.shift(0.0, (1.0 - (1.0 + k)) * 0.5 if not horizontal else 0.0)
        return done


@dataclass
class Slide(_Timed):
    """A move with no arc -- just an eased glide. The plain alternative to
    `Hop`, and the honest comparison for judging whether the hop is too much."""

    dx: float = 0.0
    dy: float = 0.0
    ease: Callable[[float], float] = ease_out_cubic

    def update(self, body: Body, dt: float) -> bool:
        p, done = self.tick(dt)
        e = self.ease(p)
        body.shift(-self.dx * (1.0 - e), -self.dy * (1.0 - e))
        return done


@dataclass
class Anticipate(_Timed):
    """The wind-up: lean *away* from where you are about to go.

    Almost free, and it does more for the weight of an attack than any amount
    of particles. The eye reads the backswing as intent, so the strike that
    follows is something the character chose to do.
    """

    dx: float = 0.0
    dy: float = 0.0
    amount: float = 0.18
    duration: float = 0.10
    squash: float = 0.18

    def update(self, body: Body, dt: float) -> bool:
        p, done = self.tick(dt)
        e = math.sin(p * math.pi * 0.5)  # ease out into the held pose
        body.shift(-self.dx * self.amount * e, -self.dy * self.amount * e)
        if self.squash:
            k = self.squash * e
            body.scale(1.0 + k * 0.5, 1.0 - k)  # crouch
            body.shift(0.0, k * 0.5)
        return done


@dataclass
class Lunge(_Timed):
    """Drive towards a neighbouring tile and come back. The bump attack.

    Out fast on `ease_out_quint` and back slow: the asymmetry is the whole
    effect. Equal speeds both ways read as a nervous jiggle.
    """

    dx: float = 0.0
    dy: float = 0.0
    reach: float = 0.55       # tiles, as a fraction of the way to the target
    out_frac: float = 0.3     # share of the duration spent travelling out
    duration: float = 0.22
    stretch: float = 0.2

    def update(self, body: Body, dt: float) -> bool:
        p, done = self.tick(dt)
        if p < self.out_frac:
            e = ease_out_quint(p / self.out_frac)
        else:
            e = 1.0 - ease_out_cubic((p - self.out_frac) / (1.0 - self.out_frac))
        body.shift(self.dx * self.reach * e, self.dy * self.reach * e)
        if self.stretch:
            k = self.stretch * e
            if abs(self.dx) > abs(self.dy):
                body.scale(1.0 + k, 1.0 - k * 0.5)
            else:
                body.scale(1.0 - k * 0.5, 1.0 + k)
        return done


@dataclass
class Knockback(_Timed):
    """Shoved away from a blow and easing home. What the *receiver* does.

    Every hit needs a reaction on both sides or it reads as the attacker
    walking through a wall poster.
    """

    dx: float = 0.0
    dy: float = 0.0
    distance: float = 0.35
    duration: float = 0.28
    ease: Callable[[float], float] = ease_out_cubic

    def update(self, body: Body, dt: float) -> bool:
        p, done = self.tick(dt)
        r = (1.0 - self.ease(p)) * self.distance
        body.shift(self.dx * r, self.dy * r)
        return done


@dataclass
class Flash(_Timed):
    """Whiten and fall back to normal. The cheapest possible hit confirmation,
    and on a tile grid it is often the only one that is legible at all."""

    duration: float = 0.12
    strength: float = 1.0

    def update(self, body: Body, dt: float) -> bool:
        p, done = self.tick(dt)
        body.whiten(self.strength * (1.0 - p))
        return done


@dataclass
class Pop(_Timed):
    """Scale overshoot -- something arriving, levelling up, being noticed."""

    duration: float = 0.3
    amount: float = 0.4
    ease: Callable[[float], float] = ease_out_back

    def update(self, body: Body, dt: float) -> bool:
        p, done = self.tick(dt)
        s = 1.0 + self.amount * (1.0 - self.ease(p))
        body.scale(s, s)
        return done


@dataclass
class Shiver(_Timed):
    """Per-entity jitter. A body shake rather than a camera shake, for the
    thing that took the hit -- or for a held pose that should look strained."""

    duration: float = 0.25
    amplitude: float = 0.09
    frequency: float = 34.0
    seed: int = 0

    def update(self, body: Body, dt: float) -> bool:
        p, done = self.tick(dt)
        a = self.amplitude * (1.0 - p) ** 2
        t = self.elapsed * self.frequency
        body.shift(value_noise(t, self.seed) * a, value_noise(t, self.seed + 91) * a)
        return done


@dataclass
class Breathe:
    """Never finishes. A slow scale wobble so an idle body is not a dead one.

    Small enough that nobody consciously notices it and large enough that
    turning it off makes the screen look like a screenshot. Seeded off the
    entity so a row of monsters does not pulse in unison like a chorus line.
    """

    amplitude: float = 0.035
    frequency: float = 1.6
    phase: float = 0.0
    elapsed: float = 0.0

    def update(self, body: Body, dt: float) -> bool:
        self.elapsed += dt
        s = math.sin(self.elapsed * self.frequency * math.tau + self.phase)
        body.scale(1.0 - self.amplitude * s * 0.5, 1.0 + self.amplitude * s)
        body.shift(0.0, -self.amplitude * s * 0.5)  # keep the feet on the floor
        return False


@dataclass
class DeathSpin(_Timed):
    """Spin, shrink, fall and fade. A removal that the eye can follow.

    A corpse that vanishes on the frame it dies leaves the player unsure they
    hit anything; half a second of exit animation answers the question.
    """

    duration: float = 0.55
    spin: float = 420.0       # total degrees
    drop: float = 0.5         # tiles fallen

    def update(self, body: Body, dt: float) -> bool:
        p, done = self.tick(dt)
        body.rotate(self.spin * ease_in_quad(p))
        s = 1.0 - ease_in_quad(p)
        body.scale(max(s, 0.01), max(s, 0.01))
        body.shift(0.0, self.drop * ease_in_quad(p))
        body.fade(1.0 - ease_in_quad(p))
        return done


class Animator:
    """The Motions currently running on one Body.

    Wipes the Body to identity, then lets every Motion add its contribution.
    Order does not matter, which is the property worth protecting.
    """

    def __init__(self) -> None:
        self.motions: List[Motion] = []
        self.persistent: List[Motion] = []
        # Persistent motions are ambient rather than triggered, so the only way
        # to switch one off is a gate here. Without it an idle-breathing toggle
        # has nothing to hold onto: there is no moment at which the breath is
        # "played" that a caller could decline to reach.
        self.persistent_enabled = True

    def play(self, motion: Optional[Motion]) -> None:
        """Add a one-shot Motion. `None` is accepted and ignored, so callers
        can write `anim.play(Hop(...) if juiced else Slide(...))` without a
        branch for the un-juiced case."""
        if motion is not None:
            self.motions.append(motion)

    def add_persistent(self, motion: Motion) -> None:
        """Add a Motion that is never dropped (idle breathing, a fear tremble)."""
        self.persistent.append(motion)

    def clear(self) -> None:
        self.motions.clear()

    @property
    def busy(self) -> bool:
        """True while a one-shot is running -- ask before starting another turn."""
        return bool(self.motions)

    def update(self, body: Body, dt: float) -> None:
        body.reset_juice()
        if self.persistent_enabled:
            for m in self.persistent:
                m.update(body, dt)
        if self.motions:
            self.motions = [m for m in self.motions if not m.update(body, dt)]


# ---------------------------------------------------------------------------
# Camera-scale effects
# ---------------------------------------------------------------------------


@dataclass
class Trauma:
    """Screen shake as a budget that drains, after Squirrel Eiserloh's talk.

    Effects that each add their own shake fight each other and stack into a
    seizure. Instead everything contributes *trauma*, one number in 0..1 that
    decays on its own, and the actual displacement is `trauma ** power`. The
    exponent is what makes it usable: a small hit at 0.3 trauma displaces by
    0.09 of the maximum -- a tap -- while a big one at 0.9 gives 0.81, and the
    tail end of every shake dies away quickly instead of buzzing.

    Sampling smooth noise at three different seeds gives x, y and roll that
    are unrelated to each other, so the frame tumbles rather than sliding on a
    diagonal.
    """

    amount: float = 0.0
    decay: float = 1.6        # trauma per second
    frequency: float = 22.0
    power: float = 2.0
    time: float = 0.0
    seed: int = 1234

    def add(self, amount: float) -> None:
        self.amount = clamp(self.amount + amount)

    def update(self, dt: float) -> None:
        self.time += dt
        self.amount = max(0.0, self.amount - self.decay * dt)

    @property
    def shake(self) -> float:
        return self.amount ** self.power

    def offset(self, max_px: float = 14.0, max_deg: float = 2.5
               ) -> Tuple[float, float, float]:
        s = self.shake
        if s <= 0.0:
            return 0.0, 0.0, 0.0
        t = self.time * self.frequency
        return (value_noise(t, self.seed) * max_px * s,
                value_noise(t, self.seed + 17) * max_px * s,
                value_noise(t, self.seed + 41) * max_deg * s)


@dataclass
class Spring:
    """A damped spring. Camera zoom punch, UI pops, anything that should
    settle rather than stop.

    Kick the velocity and let it ring down. Stiffness sets the pitch, damping
    how many wobbles you get -- and `dt` is subdivided because a stiff spring
    integrated in one big Euler step at a low frame rate will happily explode.
    """

    value: float = 0.0
    target: float = 0.0
    velocity: float = 0.0
    stiffness: float = 260.0
    damping: float = 18.0

    def kick(self, impulse: float) -> None:
        self.velocity += impulse

    def update(self, dt: float, max_step: float = 1.0 / 240.0) -> None:
        steps = max(1, int(math.ceil(dt / max_step)))
        h = dt / steps
        for _ in range(steps):
            a = -self.stiffness * (self.value - self.target) - self.damping * self.velocity
            self.velocity += a * h
            self.value += self.velocity * h


@dataclass
class Camera:
    """A camera that is always slightly behind where it should be.

    The lag is the effect. Snapping to the player each frame makes the world
    feel like it is sliding under a fixed sprite; a lagging camera lets the
    player pull away from the centre when they move and drift back when they
    stop, which is most of what "responsive" means.
    """

    x: float = 0.0
    y: float = 0.0
    smoothing: float = 0.0001   # fraction of error surviving after 1s
    zoom: Spring = field(default_factory=lambda: Spring(1.0, 1.0, stiffness=190.0, damping=14.0))
    tilt: Spring = field(default_factory=lambda: Spring(0.0, 0.0, stiffness=150.0, damping=11.0))
    lead: float = 0.0           # tiles of look-ahead in the facing direction

    def snap(self, x: float, y: float) -> None:
        self.x, self.y = x, y

    def follow(self, x: float, y: float, dt: float, smooth: bool = True) -> None:
        if smooth:
            self.x = damp(self.x, x, self.smoothing, dt)
            self.y = damp(self.y, y, self.smoothing, dt)
        else:
            self.snap(x, y)

    def punch(self, zoom: float = 0.0, tilt: float = 0.0) -> None:
        self.zoom.kick(zoom)
        self.tilt.kick(tilt)

    def update(self, dt: float) -> None:
        self.zoom.update(dt)
        self.tilt.update(dt)


@dataclass
class HitStop:
    """Freeze everything for a handful of frames on impact.

    The most powerful effect in this file per line of code and the easiest to
    overdo. Stopping the world for 60-120ms at the moment of contact makes the
    blow land, because the eye is given a still frame to read. Note that the
    *shake* deliberately keeps running while the animation is frozen -- a
    completely dead frame reads as a dropped frame rather than an impact.
    """

    remaining: float = 0.0

    def hit(self, seconds: float) -> None:
        self.remaining = max(self.remaining, seconds)

    def consume(self, dt: float) -> float:
        """Tick the freeze and return the dt the animation layer may use."""
        if self.remaining <= 0.0:
            return dt
        self.remaining -= dt
        return 0.0

    @property
    def frozen(self) -> bool:
        return self.remaining > 0.0


# ---------------------------------------------------------------------------
# Self-updating pools of small things
# ---------------------------------------------------------------------------


@dataclass
class Particle:
    x: float          # world pixels
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: Tuple[int, int, int]
    gravity: float = 900.0
    drag: float = 1.4
    spin: float = 0.0
    angle: float = 0.0

    @property
    def t(self) -> float:
        """0 at birth, 1 at death -- what the renderer fades and shrinks on."""
        return 1.0 - clamp(self.life / self.max_life) if self.max_life else 1.0


class ParticleField:
    """A flat list of particles with no pooling cleverness.

    A burst is a handful of sparks thrown along a cone, and the two details
    that matter are that speed and lifetime are *varied* per particle -- a
    burst where everything moves at the same speed reads as a wheel -- and
    that drag is applied, so they leap out and stall rather than sailing off.
    """

    def __init__(self, seed: int = 7) -> None:
        self.particles: List[Particle] = []
        self._n = 0
        self.seed = seed

    def _rand(self) -> float:
        """Deterministic 0..1. Same seed, same burst, every run."""
        self._n += 1
        return (hash01(self._n, self.seed) + 1.0) * 0.5

    def burst(self, x: float, y: float, count: int = 14, *,
              direction: Optional[Tuple[float, float]] = None,
              spread: float = math.pi, speed: Tuple[float, float] = (120.0, 420.0),
              life: Tuple[float, float] = (0.25, 0.6), size: float = 4.0,
              colors: Seq[Tuple[int, int, int]] = ((255, 240, 190),),
              gravity: float = 900.0, drag: float = 1.4) -> None:
        base = math.atan2(direction[1], direction[0]) if direction else 0.0
        arc = spread if direction else math.tau
        for _ in range(count):
            a = base + (self._rand() - 0.5) * arc
            sp = lerp(speed[0], speed[1], self._rand() ** 1.6)
            lf = lerp(life[0], life[1], self._rand())
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(a) * sp, vy=math.sin(a) * sp,
                life=lf, max_life=lf,
                size=size * lerp(0.6, 1.3, self._rand()),
                color=colors[int(self._rand() * len(colors)) % len(colors)],
                gravity=gravity, drag=drag,
                spin=(self._rand() - 0.5) * 720.0,
            ))

    def update(self, dt: float) -> None:
        alive: List[Particle] = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0.0:
                continue
            p.vx -= p.vx * p.drag * dt
            p.vy -= p.vy * p.drag * dt
            p.vy += p.gravity * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.angle += p.spin * dt
            alive.append(p)
        self.particles = alive

    def clear(self) -> None:
        self.particles.clear()

    def __len__(self) -> int:
        return len(self.particles)


@dataclass
class Floater:
    """A damage number. Rises, drifts, pops in and fades out."""

    text: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = -110.0
    life: float = 0.9
    max_life: float = 0.9
    color: Tuple[int, int, int] = (255, 235, 120)
    gravity: float = 150.0
    scale_pop: float = 0.9

    @property
    def t(self) -> float:
        return 1.0 - clamp(self.life / self.max_life) if self.max_life else 1.0

    @property
    def scale(self) -> float:
        """Punches in over the first fifth of its life, then holds."""
        return 1.0 + self.scale_pop * (1.0 - ease_out_back(clamp(self.t / 0.2)))

    @property
    def alpha(self) -> float:
        """Fully opaque until the last third, then out."""
        return clamp((1.0 - self.t) / 0.35)


class FloaterField:
    def __init__(self) -> None:
        self.floaters: List[Floater] = []

    def add(self, text: str, x: float, y: float, **kw) -> Floater:
        f = Floater(text=text, x=x, y=y, **kw)
        self.floaters.append(f)
        return f

    def update(self, dt: float) -> None:
        for f in self.floaters:
            f.life -= dt
            f.vy += f.gravity * dt
            f.x += f.vx * dt
            f.y += f.vy * dt
        self.floaters = [f for f in self.floaters if f.life > 0.0]

    def clear(self) -> None:
        self.floaters.clear()

    def __len__(self) -> int:
        return len(self.floaters)


@dataclass
class Shockwave:
    """An expanding ring. Reads as force leaving a point.

    Grows on `ease_out_quint` -- almost all of the expansion happens in the
    first few frames, which is what makes it a blast rather than a bubble --
    and thins as it goes so it does not end life as a fat donut.
    """

    x: float
    y: float
    max_radius: float = 90.0
    life: float = 0.4
    max_life: float = 0.4
    color: Tuple[int, int, int] = (255, 240, 200)
    width: float = 5.0

    @property
    def t(self) -> float:
        return 1.0 - clamp(self.life / self.max_life) if self.max_life else 1.0

    @property
    def radius(self) -> float:
        return self.max_radius * ease_out_quint(self.t)

    @property
    def alpha(self) -> float:
        return (1.0 - self.t) ** 1.5

    @property
    def thickness(self) -> int:
        return max(1, int(self.width * (1.0 - self.t)))


@dataclass
class SlashArc:
    """A sweep of the weapon: an arc drawn across the target tile.

    A bump attack with no weapon in it is two rectangles touching. An arc that
    exists for six frames is enough for the eye to fill in a sword.
    """

    x: float
    y: float
    angle: float             # degrees, the middle of the sweep
    radius: float = 34.0
    sweep: float = 150.0     # degrees covered by the arc at full extension
    life: float = 0.18
    max_life: float = 0.18
    color: Tuple[int, int, int] = (255, 255, 255)

    @property
    def t(self) -> float:
        return 1.0 - clamp(self.life / self.max_life) if self.max_life else 1.0

    @property
    def progress(self) -> float:
        return ease_out_quad(self.t)

    @property
    def alpha(self) -> float:
        return (1.0 - self.t) ** 0.8


@dataclass
class TileImpact:
    """A dent in the floor that spreads outward and dies.

    The trick that makes a grid feel like a surface rather than a backdrop:
    the tiles near a blow are pushed by a radial wave, so the world itself
    reacts instead of only the sprites on it. Costs one sine per visible tile.
    """

    x: float                  # world pixels, the centre of the blow
    y: float
    life: float = 0.5
    max_life: float = 0.5
    strength: float = 7.0     # peak displacement, pixels
    speed: float = 520.0      # how fast the ring travels outward, px/sec
    wavelength: float = 60.0

    @property
    def t(self) -> float:
        return 1.0 - clamp(self.life / self.max_life) if self.max_life else 1.0

    def displacement(self, px: float, py: float) -> Tuple[float, float]:
        """Offset (in pixels) this impact applies to a tile drawn at px,py."""
        dx, dy = px - self.x, py - self.y
        dist = math.hypot(dx, dy)
        if dist < 0.001:
            return 0.0, 0.0
        front = self.speed * self.t
        # Only tiles the wave front has reached, and only just behind it.
        lag = dist - front
        if lag > 0.0 or lag < -self.wavelength * 1.5:
            return 0.0, 0.0
        wave = math.sin(lag / self.wavelength * math.pi)
        amp = self.strength * (1.0 - self.t) ** 2 * wave
        # Falls off with distance so the blow stays local to where it landed.
        amp *= 1.0 / (1.0 + dist / 120.0)
        return dx / dist * amp, dy / dist * amp


class EffectField:
    """One home for the ambient pools, so the workbench ticks it in one line."""

    def __init__(self, seed: int = 7) -> None:
        self.particles = ParticleField(seed)
        self.floaters = FloaterField()
        self.shockwaves: List[Shockwave] = []
        self.slashes: List[SlashArc] = []
        self.impacts: List[TileImpact] = []

    def shockwave(self, x: float, y: float, **kw) -> None:
        self.shockwaves.append(Shockwave(x=x, y=y, **kw))

    def slash(self, x: float, y: float, angle: float, **kw) -> None:
        self.slashes.append(SlashArc(x=x, y=y, angle=angle, **kw))

    def impact(self, x: float, y: float, **kw) -> None:
        self.impacts.append(TileImpact(x=x, y=y, **kw))

    def tile_offset(self, px: float, py: float) -> Tuple[float, float]:
        """Summed displacement of every live impact at one tile position."""
        ox = oy = 0.0
        for imp in self.impacts:
            dx, dy = imp.displacement(px, py)
            ox += dx
            oy += dy
        return ox, oy

    def update(self, dt: float) -> None:
        self.particles.update(dt)
        self.floaters.update(dt)
        for pool in (self.shockwaves, self.slashes, self.impacts):
            for item in pool:
                item.life -= dt
        self.shockwaves = [s for s in self.shockwaves if s.life > 0.0]
        self.slashes = [s for s in self.slashes if s.life > 0.0]
        self.impacts = [i for i in self.impacts if i.life > 0.0]

    def clear(self) -> None:
        self.particles.clear()
        self.floaters.clear()
        self.shockwaves.clear()
        self.slashes.clear()
        self.impacts.clear()

    def __len__(self) -> int:
        return (len(self.particles) + len(self.floaters) + len(self.shockwaves)
                + len(self.slashes) + len(self.impacts))


# ---------------------------------------------------------------------------
# Afterimages
# ---------------------------------------------------------------------------


@dataclass
class Ghost:
    """One frozen frame of a body, kept around to be drawn faded behind it."""

    x: float          # world pixels
    y: float
    sx: float
    sy: float
    angle: float
    life: float
    max_life: float

    @property
    def alpha(self) -> float:
        return clamp(self.life / self.max_life) ** 2 if self.max_life else 0.0


class GhostTrail:
    """A short history of where a body was, sampled on a fixed interval.

    Sampling on a *timer* rather than every frame is the point -- every frame
    gives a solid smear that changes with the frame rate, while four samples
    spaced 25ms apart give distinct echoes that look the same on any machine.
    """

    def __init__(self, interval: float = 0.028, life: float = 0.2, limit: int = 8) -> None:
        self.interval = interval
        self.life = life
        self.limit = limit
        self.ghosts: List[Ghost] = []
        self._accum = 0.0

    def sample(self, x: float, y: float, sx: float, sy: float, angle: float,
               dt: float, moving: bool) -> None:
        if not moving:
            self._accum = self.interval  # so the next move records immediately
            return
        self._accum += dt
        if self._accum < self.interval:
            return
        self._accum = 0.0
        self.ghosts.append(Ghost(x, y, sx, sy, angle, self.life, self.life))
        if len(self.ghosts) > self.limit:
            del self.ghosts[0]

    def update(self, dt: float) -> None:
        for g in self.ghosts:
            g.life -= dt
        self.ghosts = [g for g in self.ghosts if g.life > 0.0]

    def clear(self) -> None:
        self.ghosts.clear()


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
#
# Durations that have been tuned by eye in the workbench, kept in one place so
# the numbers are not scattered through the render code as magic constants.

MOVE_TIME = 0.16          # a step. Above ~0.22 the game starts to feel sticky.
ATTACK_WINDUP = 0.09      # the wound-up pose, held
ATTACK_TIME = 0.20
HITSTOP_LIGHT = 0.055     # ~3 frames at 60Hz
HITSTOP_HEAVY = 0.11      # ~7 frames. Any longer and it reads as a stutter.
TRAUMA_LIGHT = 0.28
TRAUMA_HEAVY = 0.55
FLASH_TIME = 0.13


if __name__ == "__main__":  # pragma: no cover - convenience only
    import juicetest

    juicetest.main()
