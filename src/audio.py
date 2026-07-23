"""Audio: background music + combat sound effects.

Isolated from the turn loop so ``main`` stays about game flow. Everything degrades
gracefully -- if pygame's mixer can't open a device (headless CI, no audio hardware)
the game continues silently. pygame is imported lazily so importing this module never
forces an audio backend.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

AUDIO_SAMPLE_RATE = 44100
AUDIO_SAMPLE_SIZE = -16
AUDIO_CHANNELS = 2
AUDIO_BUFFER_SIZES = (16384, 8192, 4096, 2048)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _audio_driver_order() -> list[str | None]:
    if os.environ.get("PULSE_SERVER"):
        return ["pulseaudio", "pipewire", "alsa", None, "dsp"]
    return [None, "pipewire", "pulseaudio", "alsa", "dsp"]


def _audio_buffer_order(options: dict | None) -> list[int]:
    configured = None
    if isinstance(options, dict):
        configured = options.get("audio_buffer")

    if isinstance(configured, int) and configured > 0:
        sizes = [configured, *AUDIO_BUFFER_SIZES]
        seen: set[int] = set()
        ordered: list[int] = []
        for size in sizes:
            if size not in seen:
                ordered.append(size)
                seen.add(size)
        return ordered

    return list(AUDIO_BUFFER_SIZES)


def _pick_music_track(music_dir: Path) -> Path | None:
    if not music_dir.exists() or not music_dir.is_dir():
        return None

    supported_suffixes = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}
    candidates = sorted(
        path
        for path in music_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )
    if not candidates:
        return None

    return candidates[0]


def _init_pygame_mixer(options: dict | None = None) -> Any | None:
    try:
        pygame = __import__("pygame")
    except ModuleNotFoundError:
        return None

    if pygame.mixer.get_init() is not None:
        return pygame

    original_driver = os.environ.get("SDL_AUDIODRIVER")
    last_error: Exception | None = None
    for driver in _audio_driver_order():
        for buffer_size in _audio_buffer_order(options):
            try:
                if driver is None:
                    if original_driver is None:
                        os.environ.pop("SDL_AUDIODRIVER", None)
                    else:
                        os.environ["SDL_AUDIODRIVER"] = original_driver
                else:
                    os.environ["SDL_AUDIODRIVER"] = driver

                pygame.mixer.quit()
                pygame.mixer.init(
                    frequency=AUDIO_SAMPLE_RATE,
                    size=AUDIO_SAMPLE_SIZE,
                    channels=AUDIO_CHANNELS,
                    buffer=buffer_size,
                    allowedchanges=0,
                )
                if original_driver is None:
                    os.environ.pop("SDL_AUDIODRIVER", None)
                else:
                    os.environ["SDL_AUDIODRIVER"] = original_driver
                return pygame
            except Exception as exc:
                last_error = exc
                try:
                    pygame.mixer.quit()
                except Exception:
                    pass

    if original_driver is None:
        os.environ.pop("SDL_AUDIODRIVER", None)
    else:
        os.environ["SDL_AUDIODRIVER"] = original_driver

    if last_error is not None:
        print(f"Audio disabled: {last_error}", file=sys.stderr)
    return None


def start_background_music(options: dict | None = None) -> Any | None:
    """Init the mixer and loop the first supported track in ``audio/music/``.
    Returns the pygame module (for later stop), or ``None`` if audio is unavailable."""
    pygame = _init_pygame_mixer(options)
    if pygame is None:
        return None

    track = _pick_music_track(_PROJECT_ROOT / "audio" / "music")
    if track is None:
        return pygame

    try:
        pygame.mixer.music.load(str(track))
        pygame.mixer.music.play(-1)
    except Exception as exc:
        print(f"Music disabled: {exc}", file=sys.stderr)

    return pygame


def stop_background_music(pygame_module: Any | None) -> None:
    if pygame_module is None:
        return
    try:
        pygame_module.mixer.music.stop()
    except Exception:
        pass
    try:
        pygame_module.mixer.quit()
    except Exception:
        pass


class CombatSfxPlayer:
    """Plays melee/death one-shots on a dedicated channel, queuing death after a
    melee hit so both are heard. Silent if audio is off or unavailable."""

    def __init__(self, pygame_module: Any | None, options: dict | None = None):
        self._pygame = pygame_module
        self._channel: Any | None = None
        self._enabled = True
        if isinstance(options, dict):
            self._enabled = bool(options.get("combat_sfx", True))

        self._melee_sound: Any | None = None
        self._death_sound: Any | None = None
        if not self._enabled or self._pygame is None:
            return

        try:
            self._channel = self._pygame.mixer.find_channel()
        except Exception:
            self._channel = None

        self._melee_sound = self._load_sound(options, "melee_attack_sfx", "audio/sfx/swipe.wav")
        self._death_sound = self._load_sound(options, "death_sfx", "audio/sfx/splat_quick.wav")

    @staticmethod
    def _resolve_sound_path(path_value: str) -> Path:
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate
        return _PROJECT_ROOT / candidate

    def _load_sound(self, options: dict | None, key: str, fallback: str) -> Any | None:
        configured = fallback
        if isinstance(options, dict) and isinstance(options.get(key), str):
            configured = options[key]

        try:
            sound_path = self._resolve_sound_path(configured)
            if not sound_path.exists():
                return None
            return self._pygame.mixer.Sound(str(sound_path))
        except Exception:
            return None

    def _play(self, sound: Any | None, queue_if_busy: bool = False) -> None:
        if sound is None:
            return
        try:
            if self._channel is not None:
                if queue_if_busy and self._channel.get_busy():
                    self._channel.queue(sound)
                else:
                    self._channel.play(sound)
                return
            sound.play()
        except Exception:
            return

    def play_melee_attack(self) -> None:
        self._play(self._melee_sound)

    def play_death(self) -> None:
        # When called immediately after melee, queue death so it plays next.
        self._play(self._death_sound, queue_if_busy=True)
