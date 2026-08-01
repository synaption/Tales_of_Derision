#!/usr/bin/env python3
"""
procedural_babble.py

Dependency-free procedural speech-babble synthesizer.

The module converts ordinary text into deterministic fictional syllables and
synthesizes the result entirely in code. It uses no recordings, speech models,
or external assets.

Import examples
---------------
from procedural_babble import synthesize, save_wav, wav_bytes, babble_text

save_wav("hello.wav", "Hello there!", voice="chirpy", emotion="happy")
samples = synthesize("Where did you find that?", voice="mellow")
data = wav_bytes("Good morning!", voice="tiny")

Command line
------------
python procedural_babble.py "Hello there!" -o hello.wav
python procedural_babble.py "What is that?" -o question.wav --voice tiny --emotion curious
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import random
import re
import struct
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TAU = math.tau
DEFAULT_SAMPLE_RATE = 22_050

# Fictional phoneme inventory. These are labels used by the synthesizer rather
# than an attempt to reproduce any real language.
ONSETS: tuple[str, ...] = (
    "", "b", "d", "g", "k", "m", "n", "p", "t", "v", "z",
    "l", "r", "w", "y", "sh", "ch", "f", "s", "h", "j",
)
CODAS: tuple[str, ...] = ("", "", "", "", "m", "n", "p", "t", "k", "s", "l")

# Approximate formant centers in hertz. They create distinct synthetic vowel
# colors without using recorded speech.
VOWELS: Mapping[str, tuple[float, float, float]] = {
    "a":  (800.0, 1150.0, 2900.0),
    "e":  (500.0, 1700.0, 2500.0),
    "i":  (300.0, 2200.0, 3000.0),
    "o":  (500.0,  900.0, 2500.0),
    "u":  (350.0,  750.0, 2200.0),
    "ai": (650.0, 1600.0, 2800.0),
    "oo": (300.0,  650.0, 2100.0),
}


@dataclass(frozen=True)
class Voice:
    """Reusable procedural voice profile."""

    name: str = "chirpy"
    base_pitch: float = 230.0
    rate: float = 1.0
    pitch_jitter: float = 0.035
    brightness: float = 1.0
    breathiness: float = 0.025
    formant_scale: float = 1.0
    syllable_gap: float = 0.018
    seed: int = 1


VOICES: Mapping[str, Voice] = {
    "chirpy": Voice(
        name="chirpy",
        base_pitch=245.0,
        rate=1.08,
        pitch_jitter=0.045,
        brightness=1.12,
        breathiness=0.025,
        formant_scale=1.05,
        syllable_gap=0.014,
        seed=11,
    ),
    "mellow": Voice(
        name="mellow",
        base_pitch=165.0,
        rate=0.90,
        pitch_jitter=0.020,
        brightness=0.82,
        breathiness=0.018,
        formant_scale=0.92,
        syllable_gap=0.026,
        seed=23,
    ),
    "tiny": Voice(
        name="tiny",
        base_pitch=330.0,
        rate=1.20,
        pitch_jitter=0.060,
        brightness=1.24,
        breathiness=0.030,
        formant_scale=1.13,
        syllable_gap=0.010,
        seed=37,
    ),
    "gruff": Voice(
        name="gruff",
        base_pitch=120.0,
        rate=0.84,
        pitch_jitter=0.028,
        brightness=0.72,
        breathiness=0.055,
        formant_scale=0.86,
        syllable_gap=0.030,
        seed=53,
    ),
}

# pitch, duration, expressiveness, breathiness
EMOTIONS: Mapping[str, tuple[float, float, float, float]] = {
    "neutral": (1.00, 1.00, 1.00, 1.00),
    "happy":   (1.10, 0.90, 1.25, 0.90),
    "curious": (1.06, 1.00, 1.35, 1.00),
    "sad":     (0.90, 1.20, 0.55, 1.25),
    "angry":   (0.94, 0.80, 1.15, 1.35),
    "sleepy":  (0.84, 1.35, 0.35, 1.15),
    "excited": (1.18, 0.74, 1.55, 0.85),
}

_WORD_OR_MARK = re.compile(r"[A-Za-z0-9']+|[.!?,;:…—-]|\s+|.", re.UNICODE)
_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _voice_from(value: str | Voice) -> Voice:
    if isinstance(value, Voice):
        return value
    try:
        return VOICES[value]
    except KeyError as exc:
        choices = ", ".join(sorted(VOICES))
        raise ValueError(f"Unknown voice {value!r}. Choose from: {choices}") from exc


def _emotion_from(value: str) -> tuple[float, float, float, float]:
    try:
        return EMOTIONS[value]
    except KeyError as exc:
        choices = ", ".join(sorted(EMOTIONS))
        raise ValueError(f"Unknown emotion {value!r}. Choose from: {choices}") from exc


def _estimated_syllables(word: str) -> int:
    groups = len(_VOWEL_GROUP.findall(word))
    if groups == 0:
        groups = max(1, round(len(word) / 3.5))
    return max(1, min(4, groups))


def _choose_syllables(word: str, voice: Voice) -> list[tuple[str, str, str]]:
    rng = random.Random(_stable_int(word.lower(), voice.seed, "syllables"))
    count = _estimated_syllables(word)

    result: list[tuple[str, str, str]] = []
    for index in range(count):
        onset = rng.choice(ONSETS)
        vowel = rng.choice(tuple(VOWELS))
        coda = rng.choice(CODAS) if index == count - 1 else ""
        result.append((onset, vowel, coda))
    return result


def babble_text(text: str, voice: str | Voice = "chirpy") -> str:
    """
    Return the deterministic fictional syllable spelling used for ``text``.

    This is mainly useful for debugging, subtitles, lip-sync markers, or tests.
    The audible output is produced by :func:`synthesize`.
    """
    profile = _voice_from(voice)
    output: list[str] = []

    for token in _WORD_OR_MARK.findall(text):
        if re.fullmatch(r"[A-Za-z0-9']+", token):
            syllables = _choose_syllables(token, profile)
            rendered = "".join(onset + vowel + coda for onset, vowel, coda in syllables)
            if token[:1].isupper():
                rendered = rendered.capitalize()
            output.append(rendered)
        else:
            output.append(token)

    return "".join(output)


def _append_silence(target: list[float], seconds: float, sample_rate: int) -> None:
    target.extend([0.0] * max(0, int(seconds * sample_rate)))


def _fade_envelope(index: int, total: int, attack: float = 0.10, release: float = 0.18) -> float:
    if total <= 1:
        return 0.0
    x = index / (total - 1)
    attack_gain = min(1.0, x / max(attack, 1e-6))
    release_gain = min(1.0, (1.0 - x) / max(release, 1e-6))
    return max(0.0, min(attack_gain, release_gain))


def _noise_segment(
    label: str,
    duration: float,
    pitch: float,
    voice: Voice,
    rng: random.Random,
    sample_rate: int,
    coda: bool = False,
) -> list[float]:
    total = max(1, int(duration * sample_rate))
    result: list[float] = []
    previous = 0.0
    low = 0.0

    plosives = {"p", "t", "k", "b", "d", "g"}
    fricatives = {"s", "sh", "f", "h", "v", "z", "ch", "j"}
    sonorants = {"m", "n", "l", "r", "w", "y"}

    for i in range(total):
        env = _fade_envelope(i, total, attack=0.03, release=0.30 if not coda else 0.52)
        white = rng.uniform(-1.0, 1.0)
        high = white - previous
        previous = white
        low = 0.82 * low + 0.18 * white

        if label in {"s", "sh", "ch"}:
            noise = high
        elif label in {"f", "h"}:
            noise = 0.55 * high + 0.45 * low
        else:
            noise = low

        t = i / sample_rate
        voiced = math.sin(TAU * pitch * t)

        if label in plosives:
            burst = math.exp(-10.0 * i / total)
            value = (0.42 * noise + (0.10 if label in {"b", "d", "g"} else 0.0) * voiced) * burst
        elif label in fricatives:
            voiced_amount = 0.11 if label in {"v", "z", "j"} else 0.0
            value = 0.25 * noise + voiced_amount * voiced
        elif label in sonorants:
            value = 0.19 * voiced + 0.045 * low
        else:
            value = 0.0

        result.append(value * env * voice.brightness)

    return result


def _harmonic_weights(
    pitch: float,
    formants: Sequence[float],
    brightness: float,
    sample_rate: int,
) -> list[float]:
    max_harmonic = max(1, min(18, int((sample_rate * 0.45) / max(pitch, 1.0))))
    weights: list[float] = []

    for harmonic in range(1, max_harmonic + 1):
        frequency = harmonic * pitch
        base = 1.0 / (harmonic ** (1.18 / max(brightness, 0.25)))
        formant_boost = 0.16

        for formant_index, center in enumerate(formants):
            bandwidth = (120.0, 180.0, 260.0)[formant_index]
            distance = (frequency - center) / bandwidth
            formant_boost += math.exp(-0.5 * distance * distance)

        weights.append(base * formant_boost)

    normalization = sum(weights) or 1.0
    return [weight / normalization for weight in weights]


def _vowel_segment(
    vowel: str,
    duration: float,
    start_pitch: float,
    end_pitch: float,
    voice: Voice,
    expressiveness: float,
    breath_multiplier: float,
    rng: random.Random,
    sample_rate: int,
) -> list[float]:
    total = max(1, int(duration * sample_rate))
    formants = tuple(value * voice.formant_scale for value in VOWELS[vowel])
    midpoint_pitch = (start_pitch + end_pitch) * 0.5
    weights = _harmonic_weights(midpoint_pitch, formants, voice.brightness, sample_rate)

    phases = [rng.random() * TAU for _ in weights]
    result: list[float] = []
    low_noise = 0.0

    vibrato_rate = 4.8 + rng.random() * 1.8
    vibrato_depth = voice.pitch_jitter * expressiveness
    tremolo_rate = 2.0 + rng.random() * 1.4

    for i in range(total):
        x = i / max(1, total - 1)
        pitch = start_pitch + (end_pitch - start_pitch) * x
        pitch *= 1.0 + vibrato_depth * math.sin(TAU * vibrato_rate * (i / sample_rate))
        env = _fade_envelope(i, total, attack=0.07, release=0.15)

        sample = 0.0
        for harmonic_index, weight in enumerate(weights, start=1):
            phases[harmonic_index - 1] += TAU * pitch * harmonic_index / sample_rate
            sample += weight * math.sin(phases[harmonic_index - 1])

        white = rng.uniform(-1.0, 1.0)
        low_noise = 0.90 * low_noise + 0.10 * white
        breath = (white - 0.55 * low_noise) * voice.breathiness * breath_multiplier
        tremolo = 0.97 + 0.03 * math.sin(TAU * tremolo_rate * (i / sample_rate))

        result.append((sample * 1.55 + breath) * env * tremolo)

    return result


def _pitch_shape(
    base_pitch: float,
    syllable_index: int,
    syllable_count: int,
    punctuation: str,
    expressiveness: float,
    rng: random.Random,
) -> tuple[float, float]:
    position = syllable_index / max(1, syllable_count - 1)
    random_offset = rng.uniform(-0.045, 0.045) * expressiveness

    start = base_pitch * (1.0 + random_offset)
    end = start

    if punctuation == "?":
        end *= 1.0 + (0.18 * expressiveness * (0.35 + 0.65 * position))
    elif punctuation == "!":
        start *= 1.05
        end *= 1.10
    elif punctuation in {".", "…"}:
        end *= 1.0 - (0.12 * expressiveness * position)
    else:
        end *= 1.0 + rng.uniform(-0.035, 0.035) * expressiveness

    return start, end


def _next_punctuation(tokens: Sequence[str], start: int) -> str:
    for token in tokens[start + 1 :]:
        if token.isspace():
            continue
        if token in ".!?,;:…":
            return token
        if re.fullmatch(r"[A-Za-z0-9']+", token):
            return ""
    return ""


def synthesize(
    text: str,
    voice: str | Voice = "chirpy",
    emotion: str = "neutral",
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    volume: float = 0.92,
) -> list[float]:
    """
    Synthesize ``text`` into mono floating-point samples in the range [-1, 1].

    The same text, voice, and emotion always produce the same result.

    Parameters
    ----------
    text:
        Ordinary display dialogue.
    voice:
        A built-in voice name or a custom :class:`Voice`.
    emotion:
        One of ``neutral``, ``happy``, ``curious``, ``sad``, ``angry``,
        ``sleepy``, or ``excited``.
    sample_rate:
        Output sample rate. 22,050 Hz is a good game-friendly default.
    volume:
        Peak normalization target from 0.0 to 1.0.
    """
    if sample_rate < 8_000:
        raise ValueError("sample_rate must be at least 8000")
    if not 0.0 <= volume <= 1.0:
        raise ValueError("volume must be between 0.0 and 1.0")
    if not text:
        return []

    profile = _voice_from(voice)
    pitch_multiplier, duration_multiplier, expressiveness, breath_multiplier = _emotion_from(emotion)
    tokens = _WORD_OR_MARK.findall(text)
    output: list[float] = []

    # A text-level seed makes the result reproducible while allowing repeated
    # words in different sentences to receive subtly different prosody.
    master_rng = random.Random(_stable_int(text, profile.seed, emotion, sample_rate))

    for token_index, token in enumerate(tokens):
        if token.isspace():
            _append_silence(output, 0.018 / profile.rate, sample_rate)
            continue

        if token in {".", "!", "?"}:
            _append_silence(output, 0.18 * duration_multiplier / profile.rate, sample_rate)
            continue
        if token in {",", ";", ":"}:
            _append_silence(output, 0.09 * duration_multiplier / profile.rate, sample_rate)
            continue
        if token == "…":
            _append_silence(output, 0.32 * duration_multiplier / profile.rate, sample_rate)
            continue
        if token in {"—", "-"}:
            _append_silence(output, 0.06 * duration_multiplier / profile.rate, sample_rate)
            continue
        if not re.fullmatch(r"[A-Za-z0-9']+", token):
            continue

        syllables = _choose_syllables(token, profile)
        punctuation = _next_punctuation(tokens, token_index)
        word_rng = random.Random(_stable_int(token.lower(), token_index, text, profile.seed))
        word_pitch = profile.base_pitch * pitch_multiplier
        word_pitch *= 1.0 + master_rng.uniform(-0.035, 0.035) * expressiveness

        for syllable_index, (onset, vowel, coda) in enumerate(syllables):
            syllable_rng = random.Random(
                _stable_int(text, token_index, syllable_index, profile.seed, emotion)
            )
            start_pitch, end_pitch = _pitch_shape(
                word_pitch,
                syllable_index,
                len(syllables),
                punctuation,
                expressiveness,
                word_rng,
            )

            if onset:
                onset_duration = (0.018 if onset in {"p", "t", "k", "b", "d", "g"} else 0.032)
                onset_duration *= duration_multiplier / profile.rate
                output.extend(
                    _noise_segment(
                        onset,
                        onset_duration,
                        start_pitch,
                        profile,
                        syllable_rng,
                        sample_rate,
                    )
                )

            vowel_duration = word_rng.uniform(0.070, 0.115)
            vowel_duration *= duration_multiplier / profile.rate
            if syllable_index == len(syllables) - 1 and punctuation in {"?", "!", "…"}:
                vowel_duration *= 1.12

            output.extend(
                _vowel_segment(
                    vowel,
                    vowel_duration,
                    start_pitch,
                    end_pitch,
                    profile,
                    expressiveness,
                    breath_multiplier,
                    syllable_rng,
                    sample_rate,
                )
            )

            if coda:
                output.extend(
                    _noise_segment(
                        coda,
                        0.020 * duration_multiplier / profile.rate,
                        end_pitch,
                        profile,
                        syllable_rng,
                        sample_rate,
                        coda=True,
                    )
                )

            if syllable_index < len(syllables) - 1:
                _append_silence(output, profile.syllable_gap, sample_rate)

        _append_silence(output, 0.022 * duration_multiplier / profile.rate, sample_rate)

    peak = max((abs(sample) for sample in output), default=0.0)
    if peak > 0.0:
        gain = volume / peak
        output = [max(-1.0, min(1.0, sample * gain)) for sample in output]

    return output


def pcm16(samples: Iterable[float]) -> bytes:
    """Convert floating-point samples to little-endian signed 16-bit PCM."""
    chunks = bytearray()
    for sample in samples:
        clamped = max(-1.0, min(1.0, float(sample)))
        chunks.extend(struct.pack("<h", round(clamped * 32767.0)))
    return bytes(chunks)


def wav_bytes(
    text: str,
    voice: str | Voice = "chirpy",
    emotion: str = "neutral",
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    volume: float = 0.92,
) -> bytes:
    """Return a complete mono 16-bit WAV file as bytes."""
    samples = synthesize(
        text,
        voice=voice,
        emotion=emotion,
        sample_rate=sample_rate,
        volume=volume,
    )

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16(samples))
    return buffer.getvalue()


def save_wav(
    path: str | Path,
    text: str,
    voice: str | Voice = "chirpy",
    emotion: str = "neutral",
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    volume: float = 0.92,
) -> Path:
    """Synthesize dialogue and write it to ``path``. Returns the written path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        wav_bytes(
            text,
            voice=voice,
            emotion=emotion,
            sample_rate=sample_rate,
            volume=volume,
        )
    )
    return destination


def custom_voice(
    base: str | Voice = "chirpy",
    **changes: object,
) -> Voice:
    """
    Return a modified copy of a built-in or custom voice.

    Example:
        robot = custom_voice(
            "tiny",
            name="robot",
            base_pitch=290.0,
            pitch_jitter=0.005,
            breathiness=0.0,
        )
    """
    return replace(_voice_from(base), **changes)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic fictional speech-babble as a WAV file."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="Hello there! What did you find?",
        help="Display dialogue to synthesize.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="babble.wav",
        help="Output WAV path. Default: babble.wav",
    )
    parser.add_argument(
        "--voice",
        choices=sorted(VOICES),
        default="chirpy",
        help="Procedural voice profile.",
    )
    parser.add_argument(
        "--emotion",
        choices=sorted(EMOTIONS),
        default="neutral",
        help="Prosody preset.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Output sample rate. Default: {DEFAULT_SAMPLE_RATE}",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=0.92,
        help="Peak volume from 0.0 to 1.0. Default: 0.92",
    )
    parser.add_argument(
        "--print-babble",
        action="store_true",
        help="Print the generated fictional syllable spelling.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        path = save_wav(
            args.output,
            args.text,
            voice=args.voice,
            emotion=args.emotion,
            sample_rate=args.sample_rate,
            volume=args.volume,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.print_babble:
        print(babble_text(args.text, args.voice))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())