"""Sound for the juice bench: synthesis, pitch variation, layering and ducking.

Roughly half of what "juice" means is audible, and the bench had none of it.
This module is the other half.

The three things that actually matter, in order:

**1. Pitch variation.** The single most important sound trick in an action
game, and the one pygame does not give you. `pygame.mixer.Sound` has a volume
and a per-channel left/right balance and nothing else -- there is no way to ask
it to play a buffer faster. Five hits in a row through one unvaried sample
sound like a stapler, and no amount of visual polish rescues that.

The fix is to spend the CPU at load time instead of at play time: every voice
is rendered once and then *resampled* into a ladder of pitch variants, which
are ordinary Sounds. Playing one at random is then free. `pitch_variants` is
the whole trick and it is eleven lines.

**2. Layering.** One "hit" is not one sound. It is an impact (the low thud
that carries the force), a material (what the thing is made of), and often a
voice or a highlight on top. Three short samples on three channels, each with
its own pitch jitter, give a combinatorial spread from a handful of clips --
and let a crit be *the same hit plus one more layer* rather than a different
sound, which is why crits feel like an escalation instead of a substitution.

**3. Timing.** A swing sound belongs on the *wind-up*, not the contact. Sound
travelling ahead of the picture is what makes the picture land on time. The
bench fires audio from the same `Callback` structure the visuals use, so this
is a scheduling property rather than something bolted on.

Everything above the mixer boundary is numpy and is tested with no audio device
at all: `synth_*` return plain float arrays, `pitch_variants` and `resample`
are array maths, and `SoundBank` keeps a `log` of what it was asked to play so
a test can assert that a killing blow layered a crit on top of a hit without
ever opening a sound card. If numpy or the mixer is missing the bank goes
silent and the bench runs exactly as before.

The two clips already in `audio/sfx` are loaded and given the same variant
treatment; the rest is synthesised, which keeps the bench a single checkout
with no downloads and makes every sound tweakable by changing a number rather
than by finding a new file.
"""

from __future__ import annotations

import math
import os
import wave
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:                                  # pragma: no cover
    np = None                                        # type: ignore
    HAVE_NUMPY = False


#: The mixer runs at this rate and everything is synthesised at it, so nothing
#: is resampled on the way to the card except deliberately, for pitch.
RATE = 44100

#: Mixer buffer, in samples. 512 at 44.1kHz is about 12ms of latency, which is
#: under a frame. The pygame default of 4096 is 93ms -- nearly six frames late,
#: which is enough to make a perfectly timed hit feel mushy. This is the one
#: mixer setting that changes how the game *feels* rather than how it sounds.
BUFFER = 512

#: How many channels to keep for effects. Layered hits use three at once and
#: several can overlap, so the default eight runs out audibly.
CHANNELS = 24


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        os.pardir, os.pardir))


SFX_DIR = os.path.join(_repo_root(), "audio", "sfx")
MUSIC_DIR = os.path.join(_repo_root(), "audio", "music")


# ---------------------------------------------------------------------------
# Array helpers -- pure numpy, no mixer, no pygame
# ---------------------------------------------------------------------------


def semitone_ratio(semitones: float) -> float:
    """Playback speed for a pitch shift. An octave is a factor of two."""
    return 2.0 ** (semitones / 12.0)


def resample(samples, ratio: float):
    """Linear-interpolated resample. `ratio` > 1 is higher and shorter.

    This is a *speed* change, so it moves formants along with the pitch -- the
    chipmunk effect. That is wrong for dialogue and exactly right for impacts,
    where "shorter and brighter" is what a harder hit sounds like anyway.
    """
    n = len(samples)
    if n == 0 or abs(ratio - 1.0) < 1e-6:
        return samples
    out_n = max(1, int(n / ratio))
    # Sample positions in the source, one per output sample.
    pos = np.arange(out_n, dtype=np.float32) * ratio
    i = np.clip(pos.astype(np.int32), 0, n - 2)
    frac = pos - i
    return (samples[i] * (1.0 - frac) + samples[i + 1] * frac).astype(np.float32)


def normalize(samples, peak: float = 0.92):
    """Scale to a fixed peak. Keeps hand-tuned voices from drowning each other."""
    m = float(np.max(np.abs(samples))) if len(samples) else 0.0
    return samples * (peak / m) if m > 1e-9 else samples


def env(n: int, attack: float, decay: float, rate: int = RATE,
        curve: float = 2.2):
    """Attack-decay envelope, `attack`/`decay` in seconds.

    Percussive shape: the attack is linear because a hit has no ramp worth
    hearing, and the decay is a power curve because a linear fade-out sounds
    like someone turning a knob rather than like energy leaving a body.
    """
    a = max(1, int(attack * rate))
    d = max(1, n - a)
    out = np.empty(n, dtype=np.float32)
    out[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    tail = np.linspace(0.0, 1.0, d, dtype=np.float32)
    out[a:] = (1.0 - tail) ** curve
    # `decay` shortens the tail by scaling the curve's reach, so a voice can be
    # made snappier without changing its length or its pitch.
    if decay > 0.0 and decay < 1.0:
        out[a:] *= (1.0 - tail) ** (1.0 / max(decay, 0.05) - 1.0)
    return out


def tone(n: int, f0: float, f1: Optional[float] = None, rate: int = RATE,
         shape: str = "sine"):
    """A tone that may sweep from `f0` to `f1`, exponentially.

    Sweeps are exponential rather than linear because pitch is perceived
    logarithmically: a linear sweep from 200Hz to 50Hz spends most of its time
    in the top octave and reads as a click with a thud after it.
    """
    t = np.arange(n, dtype=np.float32) / rate
    if f1 is None or abs(f1 - f0) < 1e-6:
        phase = 2.0 * math.pi * f0 * t
    else:
        k = math.log(max(f1, 1.0) / max(f0, 1.0))
        # Integral of f0 * e^(k t / T) dt, so the phase is continuous.
        T = n / rate
        phase = 2.0 * math.pi * f0 * T / k * (np.exp(k * t / T) - 1.0)
    if shape == "sine":
        return np.sin(phase).astype(np.float32)
    if shape == "square":
        return np.sign(np.sin(phase)).astype(np.float32)
    if shape == "saw":
        return (2.0 * ((phase / (2.0 * math.pi)) % 1.0) - 1.0).astype(np.float32)
    raise ValueError(shape)


def noise(n: int, seed: int = 1):
    """White noise from a seeded generator, so a build sounds like the last one."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, n).astype(np.float32)


def svf(x, cutoff, rate: int = RATE, q: float = 1.0, mode: str = "low"):
    """Chamberlin state-variable filter with a per-sample cutoff.

    A plain Python loop, because the filter is recursive and cannot be
    vectorised -- but it only ever runs at *load* time, on clips a few thousand
    samples long, so the cost is a few milliseconds once rather than anything
    the frame budget ever sees.

    `cutoff` may be a scalar or an array as long as `x`, which is what makes
    the swept-filter voices (a swing whoosh, a slime gloop) possible: the
    character of those sounds is entirely in the cutoff moving.
    """
    n = len(x)
    fc = np.full(n, float(cutoff), dtype=np.float32) if np.isscalar(cutoff) \
        else np.asarray(cutoff, dtype=np.float32)
    # The SVF goes unstable as the cutoff approaches a quarter of the rate.
    fc = np.clip(fc, 20.0, rate * 0.24)
    f = (2.0 * np.sin(np.pi * fc / rate)).astype(np.float32)
    damp = float(np.clip(1.0 / max(q, 0.5), 0.0, 2.0))

    low = band = 0.0
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        high = x[i] - low - damp * band
        band += f[i] * high
        low += f[i] * band
        out[i] = low if mode == "low" else band if mode == "band" else high
    return out


def mix(*layers):
    """Sum arrays of differing lengths, padding to the longest."""
    layers = [l for l in layers if l is not None and len(l)]
    if not layers:
        return np.zeros(1, dtype=np.float32)
    n = max(len(l) for l in layers)
    out = np.zeros(n, dtype=np.float32)
    for l in layers:
        out[:len(l)] += l
    return out


def to_int16(samples, gain: float = 1.0):
    """Float -1..1 -> the int16 the mixer wants, hard-clipped."""
    return (np.clip(samples * gain, -1.0, 1.0) * 32767.0).astype(np.int16)


# ---------------------------------------------------------------------------
# The voices
# ---------------------------------------------------------------------------
#
# Each returns mono float32 in roughly -1..1. They are deliberately short: an
# impact sound that outlasts its animation smears into the next one, and in a
# roguelike the next one is 160ms away.


def synth_impact(rate: int = RATE, dur: float = 0.20, f0: float = 180.0,
                 f1: float = 46.0, click: float = 0.35, seed: int = 3):
    """The low thud that carries the force. Under every hit in the bench.

    A pitch sweep downwards is what a struck body sounds like -- the impact
    excites the whole thing at once and the high partials die first -- and the
    noise click on the front is the contact itself. Take the click away and the
    sound is a drum; take the sweep away and it is a click.
    """
    n = int(dur * rate)
    body = tone(n, f0, f1, rate) * env(n, 0.001, 0.7, rate, curve=2.6)
    snap = svf(noise(n, seed), 2600.0, rate, q=0.8) * env(n, 0.0005, 0.12, rate, curve=6.0)
    return normalize(mix(body, snap * click))


def synth_material(kind: str, rate: int = RATE, seed: int = 11):
    """What the thing is made of. The layer that tells flesh from bone.

    All five are the same two ingredients -- filtered noise and a few partials
    -- in different proportions. That is the point: a material library does not
    need five recording sessions, it needs five sets of numbers.
    """
    if kind == "flesh":
        n = int(0.16 * rate)
        wet = svf(noise(n, seed), np.linspace(1400, 300, n), rate, q=1.4)
        return normalize(wet * env(n, 0.001, 0.35, rate, curve=2.4), 0.7)
    if kind == "bone":
        n = int(0.13 * rate)
        crack = svf(noise(n, seed), 3200.0, rate, q=2.6, mode="band")
        ring = (tone(n, 900, 820, rate) * 0.3 + tone(n, 1470, 1380, rate) * 0.2)
        return normalize(mix(crack, ring * env(n, 0.001, 0.2, rate))
                         * env(n, 0.0005, 0.15, rate, curve=4.0), 0.7)
    if kind == "stone":
        n = int(0.22 * rate)
        grit = svf(noise(n, seed), 700.0, rate, q=0.7)
        return normalize(grit * env(n, 0.002, 0.5, rate, curve=2.0), 0.7)
    if kind == "metal":
        # Inharmonic partials, which is the whole difference between a bell and
        # a note: the ratios are deliberately not whole numbers.
        n = int(0.5 * rate)
        parts = [(1.0, 0.5), (2.76, 0.3), (5.40, 0.18), (8.93, 0.1)]
        rung = mix(*[tone(n, 620 * r, 615 * r, rate) * a for r, a in parts])
        return normalize(rung * env(n, 0.001, 0.9, rate, curve=1.4), 0.6)
    if kind == "slime":
        n = int(0.26 * rate)
        # A resonant filter swept downwards is the "gloop": the resonance is
        # heard as a pitch falling through the noise, which reads as wet.
        gloop = svf(noise(n, seed), np.linspace(1800, 180, n), rate, q=6.0, mode="band")
        return normalize(gloop * env(n, 0.004, 0.5, rate, curve=1.8), 0.75)
    raise ValueError(kind)


def synth_swing(rate: int = RATE, dur: float = 0.20, seed: int = 21):
    """A whoosh: noise through a bandpass that sweeps up and back down.

    The sweep going *up then down* is what makes it read as something passing
    the ear rather than as a hiss. Fired on the wind-up, not on contact.
    """
    n = int(dur * rate)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    # Clamped at zero because sin(pi) lands a hair below it in float32, and a
    # negative base under a fractional power is a NaN that poisons the clip.
    arch = np.clip(np.sin(t * math.pi), 0.0, 1.0)
    cut = 400.0 + 2600.0 * arch ** 1.4
    air = svf(noise(n, seed), cut, rate, q=3.2, mode="band")
    return normalize(air * (arch ** 1.6).astype(np.float32), 0.55)


def synth_step(rate: int = RATE, seed: int = 31):
    """A footfall. Very short, very quiet, and the thing you only miss when the
    character crosses a room in silence."""
    n = int(0.075 * rate)
    thud = tone(n, 130, 70, rate) * 0.6
    scuff = svf(noise(n, seed), 1100.0, rate, q=0.9) * 0.5
    return normalize(mix(thud, scuff) * env(n, 0.001, 0.25, rate, curve=3.2), 0.35)


def synth_blip(rate: int = RATE, freq: float = 880.0, dur: float = 0.06,
               shape: str = "square"):
    """A UI click. Square, because a sine blip is inaudible over a fight."""
    n = int(dur * rate)
    return normalize(tone(n, freq, freq * 0.92, rate, shape)
                     * env(n, 0.001, 0.2, rate, curve=3.0), 0.4)


def synth_crit(rate: int = RATE, seed: int = 41):
    """The extra layer a critical hit adds on top of an ordinary one.

    Bright, short and pitched *up*, so it cuts through the thud it is stacked
    on instead of muddying it. Escalation, not substitution.
    """
    n = int(0.28 * rate)
    ping = mix(tone(n, 1760, 1900, rate) * 0.5, tone(n, 2640, 2850, rate) * 0.25)
    return normalize(ping * env(n, 0.0008, 0.35, rate, curve=2.2), 0.55)


def synth_death(rate: int = RATE, seed: int = 51):
    """A descending tone with a noise tail: something has stopped working."""
    n = int(0.55 * rate)
    fall = tone(n, 420, 62, rate, "saw") * 0.45
    dust = svf(noise(n, seed), np.linspace(2000, 200, n), rate, q=1.1) * 0.5
    return normalize(mix(fall, dust) * env(n, 0.004, 0.8, rate, curve=1.8), 0.7)


def synth_pop(rate: int = RATE, seed: int = 61):
    """A burst. For the deaths that pop rather than fall over."""
    n = int(0.18 * rate)
    body = svf(noise(n, seed), np.linspace(3000, 400, n), rate, q=1.6)
    return normalize(body * env(n, 0.0008, 0.18, rate, curve=3.4), 0.7)


def synth_bump(rate: int = RATE, seed: int = 71):
    """Walking into a wall. Dull, dead, no ring -- a refusal, not an event."""
    n = int(0.12 * rate)
    dull = tone(n, 150, 90, rate) * 0.5
    tap = svf(noise(n, seed), 900.0, rate, q=0.8) * 0.4
    return normalize(mix(dull, tap) * env(n, 0.001, 0.15, rate, curve=4.0), 0.45)


def synth_hurt(rate: int = RATE, seed: int = 81):
    """The player taking it. Lower and longer than the hits they deal, so the
    ear can tell incoming from outgoing without looking."""
    n = int(0.3 * rate)
    grunt = svf(mix(tone(n, 240, 150, rate, "saw") * 0.5, noise(n, seed) * 0.35),
                np.linspace(1200, 400, n), rate, q=2.0)
    return normalize(grunt * env(n, 0.006, 0.5, rate, curve=2.0), 0.6)


# ---------------------------------------------------------------------------
# Loading the clips the repo already ships
# ---------------------------------------------------------------------------


def load_wav(path: str, rate: int = RATE):
    """Read a wav to mono float32 at `rate`, with no pygame anywhere.

    Uses the stdlib `wave` module rather than `pygame.mixer.Sound` so clips can
    be loaded, resampled and inspected in a test with no audio device -- the
    same rule the rest of the bench follows for the display.
    """
    with wave.open(path, "rb") as w:
        n_chan = w.getnchannels()
        width = w.getsampwidth()
        src_rate = w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:                                            # pragma: no cover
        raise ValueError(f"unsupported sample width {width}")

    if n_chan > 1:
        data = data.reshape(-1, n_chan).mean(axis=1)
    if src_rate != rate:
        data = resample(data, src_rate / rate)
    return np.ascontiguousarray(data, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pitch variants
# ---------------------------------------------------------------------------


def pitch_variants(samples, spread: float = 2.0, count: int = 5):
    """A ladder of pitch-shifted copies, evenly spaced across +/- `spread`.

    This is the workaround for the mixer's one real gap. It costs a few
    milliseconds and a few hundred kilobytes at load time and buys unlimited
    free variation afterwards, which is the right trade for a sound that plays
    five times in two seconds.

    An odd `count` is worth keeping so the middle rung is the unshifted
    original: the ear anchors on it and hears the others as variation around a
    sound rather than as five different sounds.
    """
    if count <= 1 or spread <= 0.0:
        return [samples]
    return [resample(samples, semitone_ratio(-spread + 2.0 * spread * i / (count - 1)))
            for i in range(count)]


# ---------------------------------------------------------------------------
# Buses
# ---------------------------------------------------------------------------


@dataclass
class Bus:
    """A named volume group with a duck.

    Ducking is the cheapest mix trick there is: pull the music down a few dB
    for the length of a hit and the hit sounds twice as big without being any
    louder. It is also the audio half of hit-stop -- the same instant, doing
    the same job to a different sense.
    """

    name: str
    volume: float = 1.0
    duck_amount: float = 0.0        # 0..1, how far down it is being pushed
    duck_hold: float = 0.0          # seconds left at full duck
    recover: float = 3.0            # duck units per second once the hold ends

    def duck(self, amount: float, hold: float = 0.06) -> None:
        self.duck_amount = max(self.duck_amount, min(1.0, amount))
        self.duck_hold = max(self.duck_hold, hold)

    def update(self, dt: float) -> None:
        if self.duck_hold > 0.0:
            self.duck_hold = max(0.0, self.duck_hold - dt)
            return
        self.duck_amount = max(0.0, self.duck_amount - self.recover * dt)

    @property
    def gain(self) -> float:
        return self.volume * (1.0 - self.duck_amount)


# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------


@dataclass
class Voice:
    """One sound, as a set of pitch variants plus how it should be played."""

    name: str
    bus: str = "sfx"
    gain: float = 1.0
    spread: float = 2.0              # semitones of pitch jitter
    variants: List = field(default_factory=list)     # numpy arrays
    sounds: List = field(default_factory=list)       # pygame Sounds, if any


#: A hit is a stack, not a sample. Each entry is (voice, gain, pitch bias in
#: semitones) and the tier system adds to the stack rather than replacing it.
HIT_LAYERS: Dict[str, Sequence[Tuple[str, float, float]]] = {
    "flesh": (("impact", 1.0, 0.0), ("mat_flesh", 0.8, 0.0)),
    "bone": (("impact", 0.9, 2.0), ("mat_bone", 0.9, 0.0)),
    "stone": (("impact", 1.0, -3.0), ("mat_stone", 0.85, 0.0)),
    "metal": (("impact", 0.8, 1.0), ("mat_metal", 0.5, 0.0)),
    "slime": (("impact", 0.7, -4.0), ("mat_slime", 1.0, 0.0)),
}

#: Which material each sprite in the bench is made of. A skeleton that sounds
#: like a goblin is a missed opportunity for one line of table.
MATERIALS = {
    "player": "flesh", "goblin": "flesh", "lizard": "flesh", "ogre": "flesh",
    "wolf": "flesh", "raven": "flesh", "skeleton": "bone", "slime": "slime",
    "dummy": "stone",
}


class SoundBank:
    """Every voice, the buses, and the only place the mixer is touched.

    Degrades to silence in three separate ways -- no numpy, no mixer, or the
    caller switching sound off -- and in all three the rest of the bench is
    unchanged. `log` records every requested play whether or not anything came
    out, which is what lets the timing and layering be tested headless.
    """

    def __init__(self, rate: int = RATE, enabled: bool = True):
        self.rate = rate
        self.enabled = enabled
        self.voices: Dict[str, Voice] = {}
        self.buses = {"sfx": Bus("sfx", 0.85), "music": Bus("music", 0.4),
                      "ui": Bus("ui", 0.5)}
        self.master = 0.8
        self.pitch_jitter = 1.0      # multiplies every voice's own spread
        self.log: List[Tuple[str, float, float]] = []   # (name, gain, pan)
        self.log_limit = 200
        self.available = False       # a real mixer is up
        self.music_ready = False
        self._n = 0                  # play counter, drives variant choice
        self._channels: List = []

    # -- setup ------------------------------------------------------------
    def init_mixer(self) -> bool:
        """Bring the mixer up with a small buffer. Safe to call twice."""
        if not HAVE_NUMPY:
            return False
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.pre_init(self.rate, -16, 2, BUFFER)
                pygame.mixer.init(self.rate, -16, 2, BUFFER)
            pygame.mixer.set_num_channels(CHANNELS)
            self.available = True
        except Exception:                            # pragma: no cover
            self.available = False
        return self.available

    def _make_sounds(self, variants) -> List:
        """numpy mono -> a list of pygame Sounds, stereo-duplicated."""
        if not self.available:
            return []
        import pygame
        out = []
        for v in variants:
            mono = to_int16(v)
            stereo = np.ascontiguousarray(np.repeat(mono[:, None], 2, axis=1))
            out.append(pygame.sndarray.make_sound(stereo))
        return out

    def register(self, name: str, samples, *, bus: str = "sfx",
                 gain: float = 1.0, spread: float = 2.0, count: int = 5) -> Voice:
        """Render a voice's pitch ladder and hand it to the mixer."""
        variants = pitch_variants(samples, spread, count) if HAVE_NUMPY else []
        v = Voice(name=name, bus=bus, gain=gain, spread=spread,
                  variants=variants, sounds=self._make_sounds(variants))
        self.voices[name] = v
        return v

    def load_all(self) -> None:
        """Build the whole cast. A few tens of milliseconds, once."""
        if not HAVE_NUMPY:
            return
        r = self.rate
        self.register("impact", synth_impact(r), gain=0.9, spread=2.5)
        for kind in ("flesh", "bone", "stone", "metal", "slime"):
            self.register(f"mat_{kind}", synth_material(kind, r), gain=0.8, spread=2.0)
        self.register("swing", synth_swing(r), gain=0.55, spread=2.5)
        self.register("step", synth_step(r), gain=0.30, spread=3.0, count=7)
        self.register("bump", synth_bump(r), gain=0.5, spread=1.5)
        self.register("crit", synth_crit(r), gain=0.5, spread=1.0)
        self.register("death", synth_death(r), gain=0.7, spread=2.0)
        self.register("pop", synth_pop(r), gain=0.6, spread=2.5)
        self.register("hurt", synth_hurt(r), gain=0.6, spread=1.5)
        self.register("ui_on", synth_blip(r, 1180.0), bus="ui", gain=0.35, spread=0.6)
        self.register("ui_off", synth_blip(r, 620.0), bus="ui", gain=0.35, spread=0.6)

        # The two clips the repo already had. They get exactly the same variant
        # treatment as the synthesised ones -- a wav is not a special case.
        for fname, name, gain in (("swipe.wav", "swipe", 0.5),
                                  ("splat_quick.wav", "splat", 0.7)):
            path = os.path.join(SFX_DIR, fname)
            if os.path.exists(path):
                try:
                    self.register(name, load_wav(path, r), gain=gain, spread=2.5)
                except Exception:                    # pragma: no cover
                    pass

    def load_music(self, name: str = "204 People Without Hope.ogg") -> bool:
        """Queue the background track. Ducking it is half of why it is here."""
        if not self.available:
            return False
        path = os.path.join(MUSIC_DIR, name)
        if not os.path.exists(path):
            return False
        try:
            import pygame
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(self.master * self.buses["music"].gain)
            pygame.mixer.music.play(-1)
            self.music_ready = True
        except Exception:                            # pragma: no cover
            self.music_ready = False
        return self.music_ready

    # -- playing ----------------------------------------------------------
    def _rand(self) -> float:
        """Deterministic 0..1 that advances on every play. Same run, same mix."""
        self._n += 1
        h = (self._n * 2654435761) & 0xFFFFFFFF
        h ^= h >> 15
        return ((h * 2246822519) & 0xFFFFFFFF) / 0xFFFFFFFF

    def play(self, name: str, *, gain: float = 1.0, pan: float = 0.0,
             pitch: float = 0.0) -> None:
        """Play one voice. `pan` is -1 left to +1 right, `pitch` in semitones.

        The pitch bias picks a *rung* of the pre-rendered ladder rather than
        resampling now: the whole design exists so that nothing expensive
        happens on the frame a sword lands.
        """
        v = self.voices.get(name)
        if v is None:
            return
        self.log.append((name, gain, pan))
        del self.log[:-self.log_limit]
        if not (self.enabled and self.available and v.sounds):
            return

        bus = self.buses.get(v.bus)
        vol = self.master * (bus.gain if bus else 1.0) * v.gain * gain
        if vol <= 0.003:
            return

        # Jitter is in ladder rungs. The bias shifts which part of the ladder
        # is drawn from, so a crit's layers sit higher without a second clip.
        n = len(v.sounds)
        centre = (n - 1) * 0.5
        span = (n - 1) * 0.5 * self.pitch_jitter
        idx = centre + pitch / max(v.spread, 0.01) * centre + (self._rand() - 0.5) * 2.0 * span
        sound = v.sounds[int(max(0, min(n - 1, round(idx))))]

        import pygame
        ch = pygame.mixer.find_channel(True)
        if ch is None:                               # pragma: no cover
            return
        # Constant-power panning: a hard pan keeps its loudness instead of
        # dipping through the middle, which matters when the camera pans.
        p = max(-1.0, min(1.0, pan))
        left = math.cos((p + 1.0) * math.pi / 4.0)
        right = math.sin((p + 1.0) * math.pi / 4.0)
        ch.set_volume(min(1.0, vol * left * 1.414), min(1.0, vol * right * 1.414))
        ch.play(sound)

    def play_hit(self, material: str, *, gain: float = 1.0, pan: float = 0.0,
                 tier: str = "normal") -> None:
        """A layered hit: impact + material, plus a tier layer on top.

        `tier` is the whole reason this is a method and not three `play` calls
        at the call site -- a crit and a kill are *the same sound with more on
        top*, so the ear hears an escalation of something it already knows.
        """
        for voice, g, bias in HIT_LAYERS.get(material, HIT_LAYERS["flesh"]):
            self.play(voice, gain=g * gain, pan=pan, pitch=bias)
        if tier == "crit":
            self.play("crit", gain=0.9 * gain, pan=pan)
        elif tier == "kill":
            self.play("crit", gain=0.6 * gain, pan=pan, pitch=-2.0)
            self.play("splat", gain=0.8 * gain, pan=pan)

    def duck(self, amount: float = 0.45, hold: float = 0.07,
             buses: Sequence[str] = ("music",)) -> None:
        for name in buses:
            b = self.buses.get(name)
            if b:
                b.duck(amount, hold)

    def update(self, dt: float) -> None:
        for b in self.buses.values():
            b.update(dt)
        if self.music_ready:
            try:
                import pygame
                pygame.mixer.music.set_volume(
                    self.master * self.buses["music"].gain if self.enabled else 0.0)
            except Exception:                        # pragma: no cover
                pass

    def stop_all(self) -> None:                      # pragma: no cover
        if self.available:
            import pygame
            pygame.mixer.stop()
