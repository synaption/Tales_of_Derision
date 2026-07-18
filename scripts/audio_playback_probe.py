#!/usr/bin/env python3
"""Standalone audio playback probe.

Use this to test music playback quality without running the game loop.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tempfile
import wave
import math
from pathlib import Path


def pick_track(music_dir: Path) -> Path | None:
    supported_suffixes = {".ogg", ".wav", ".mp3", ".flac", ".m4a"}
    if not music_dir.exists():
        return None

    candidates = sorted(
        path
        for path in music_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe WSL/Linux audio playback independent of game loop")
    parser.add_argument("--seconds", type=int, default=30, help="How long to play the track")
    parser.add_argument("--buffer", type=int, default=65536, help="Pygame mixer buffer size")
    parser.add_argument("--frequency", type=int, default=44100, help="Mixer sample rate")
    parser.add_argument(
        "--driver",
        default="pulseaudio",
        help="SDL_AUDIODRIVER to force (pulseaudio/pipewire/alsa/dsp/auto)",
    )
    parser.add_argument(
        "--track",
        type=Path,
        default=None,
        help="Optional explicit track path. Defaults to first file under audio/music.",
    )
    parser.add_argument(
        "--mode",
        choices=["music", "tone"],
        default="music",
        help="music: play a file from audio/music, tone: play generated sine-wave WAV",
    )
    parser.add_argument("--tone-hz", type=int, default=440, help="Tone frequency when --mode tone")
    return parser.parse_args()


def _write_tone_wav(path: Path, seconds: int, hz: int, sample_rate: int) -> None:
    frame_count = max(1, seconds * sample_rate)
    amplitude = 0.2
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for index in range(frame_count):
            sample = int(32767 * amplitude * math.sin(2 * math.pi * hz * (index / sample_rate)))
            frames += sample.to_bytes(2, byteorder="little", signed=True)
            frames += sample.to_bytes(2, byteorder="little", signed=True)

        wav_file.writeframes(bytes(frames))


def main() -> int:
    args = parse_args()

    if args.driver and args.driver.lower() != "auto":
        os.environ["SDL_AUDIODRIVER"] = args.driver

    project_root = Path(__file__).resolve().parent.parent
    generated_track: Path | None = None
    if args.mode == "tone":
        temp_file = tempfile.NamedTemporaryFile(prefix="audio_probe_", suffix=".wav", delete=False)
        temp_file.close()
        generated_track = Path(temp_file.name)
        _write_tone_wav(generated_track, seconds=max(2, args.seconds), hz=args.tone_hz, sample_rate=args.frequency)
        track = generated_track
    else:
        track = args.track or pick_track(project_root / "audio" / "music")
        if track is None:
            print("No playable track found in audio/music.", file=sys.stderr)
            return 2

    try:
        import pygame
    except ModuleNotFoundError:
        print("pygame is not installed. Run: python3 -m pip install --user pygame", file=sys.stderr)
        return 3

    print(f"Track: {track}")
    print(f"SDL_AUDIODRIVER={os.environ.get('SDL_AUDIODRIVER', '<default>')}")
    print(f"PULSE_SERVER={os.environ.get('PULSE_SERVER', '<unset>')}")
    print(
        f"Mixer config: frequency={args.frequency} size=-16 channels=2 buffer={args.buffer} seconds={args.seconds}"
    )

    try:
        pygame.mixer.init(
            frequency=args.frequency,
            size=-16,
            channels=2,
            buffer=args.buffer,
            allowedchanges=0,
        )
        pygame.mixer.music.load(str(track))
        pygame.mixer.music.play(-1)

        start = time.time()
        while time.time() - start < args.seconds:
            elapsed = int(time.time() - start)
            print(f"[{elapsed:02d}s] busy={pygame.mixer.music.get_busy()}")
            time.sleep(1)

        pygame.mixer.music.stop()
        pygame.mixer.quit()
        print("Probe complete.")
        return 0
    except Exception as exc:
        print(f"Audio probe failed: {exc}", file=sys.stderr)
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        return 1
    finally:
        if generated_track is not None:
            try:
                generated_track.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
