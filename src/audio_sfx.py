"""Small combat SFX helper with safe fallbacks.

Uses pygame mixer if available and audio init succeeds. If anything fails,
methods become no-ops so gameplay is unaffected.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

DEFAULT_ATTACK_SFX = Path("400 Sounds Pack/Combat and Gore/swipe.wav")
DEFAULT_DEATH_SFX = Path("400 Sounds Pack/Combat and Gore/splat_quick.wav")


class CombatSfxPlayer:
    def __init__(self, options: dict | None = None) -> None:
        self._enabled = bool((options or {}).get("combat_sfx", True))
        self._pygame: Any | None = None
        self._attack_sound: Any | None = None
        self._death_sound: Any | None = None
        self._load_default_sounds()

    @staticmethod
    def _sfx_base_dir() -> Path:
        return Path(__file__).resolve().parent.parent / "audio" / "music" / "sfx"

    def _load_default_sounds(self) -> None:
        if not self._enabled:
            return

        attack_path = self._sfx_base_dir() / DEFAULT_ATTACK_SFX
        death_path = self._sfx_base_dir() / DEFAULT_DEATH_SFX
        if not attack_path.exists() or not death_path.exists():
            self._enabled = False
            return

        try:
            import pygame
        except ModuleNotFoundError:
            self._enabled = False
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._pygame = pygame
            self._attack_sound = pygame.mixer.Sound(str(attack_path))
            self._death_sound = pygame.mixer.Sound(str(death_path))
        except Exception:
            self._enabled = False
            self._attack_sound = None
            self._death_sound = None

    def play_attack(self) -> None:
        if not self._enabled or self._attack_sound is None:
            return
        try:
            self._attack_sound.play()
        except Exception:
            pass

    def play_death(self) -> None:
        if not self._enabled or self._death_sound is None:
            return
        try:
            self._death_sound.play()
        except Exception:
            pass
