"""pyRL2 entry point.

Turn loop: show title/menu, then block for an action, run the systems, repeat.
The game logic stays renderer-agnostic while the default runtime uses pygame.
"""
import argparse
import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import esper

from components import Asleep, Bed, BerryBush, BlocksMovement, Chest, Corpse, Deer, Dialogue, Diet, Enemy, Equipment, Family, Fish, Friendly, Gender, Home, Inventory, Meat, NPC, Name, Needs, Personality, Player, Position, Relationships, Renderable, Resident, Seaweed, Stove, Tree, Vision, Well, WorldClock
from onymancer import make_onymancer
from game_map import GameMap
from items import (
    BERRIES,
    WOOD,
    WOOD_DOOR,
    WOOD_WALL,
    WOOD_WINDOW,
    cook_meat,
    craft_cost,
    hunger_restored,
    is_placeable,
    is_raw_meat,
    placed_tile,
    thirst_restored,
)
from persistence import (
    DEFAULT_SAVE_FILE,
    bootstrap_files,
    first_player_position,
    load_game,
    load_options,
    save_game,
    save_options,
)
from renderer.base import Renderer
from renderer.pygame_renderer import PygameRenderer
from systems import FishAiProcessor, HousingProcessor, MovementProcessor, NeedsProcessor, NpcAiProcessor, RenderProcessor, TimeProcessor, TreeGrowthProcessor, WAIT_ACTION, active_statuses, bed_owner, bubbles_active, friendship, furnish_house, go_to_sleep, interact, pick_berries, player_is_animated, queue_message, set_bed_owner, slay_entity, spawn_speech_bubble, status_label, wake_up, world_clock

# The world is a 3x3 grid of 120x60 sections: the habitable island sits in the
# centre section, ringed by a coastline, and the surrounding eight sections are
# open ocean full of fish and seaweed. The land itself is the old 120x60 world.
LAND_WIDTH = 120
LAND_HEIGHT = 60
MAP_WIDTH = LAND_WIDTH * 3
MAP_HEIGHT = LAND_HEIGHT * 3

_CARDINAL_ACTION_DELTAS = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}

_VECTOR_TO_ACTION = {
    (-1, -1): "move_up_left",
    (0, -1): "move_up",
    (1, -1): "move_up_right",
    (-1, 0): "move_left",
    (1, 0): "move_right",
    (-1, 1): "move_down_left",
    (0, 1): "move_down",
    (1, 1): "move_down_right",
}

_RELEASE_TO_DIRECTION = {
    "release_up": "move_up",
    "release_down": "move_down",
    "release_left": "move_left",
    "release_right": "move_right",
}

_SCALE_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

AUDIO_SAMPLE_RATE = 44100
AUDIO_SAMPLE_SIZE = -16
AUDIO_CHANNELS = 2
AUDIO_BUFFER_SIZES = (16384, 8192, 4096, 2048)

# Pygbag runs the whole program on the browser's single main thread. Any loop
# that would block (spin on input) must hand control back to the browser with
# ``await asyncio.sleep(...)`` or the tab freezes, and web audio/display need
# their own handling. On desktop these paths keep the original behavior.
IS_WEB = sys.platform == "emscripten"

# Idle input polling cadence on web. A zero-delay spin (asyncio.sleep(0)) pegs
# the browser's single main thread and starves the audio mixer -> clicks/pops.
# ~60Hz frees the thread for audio while keeping input latency imperceptible.
_WEB_IDLE_POLL_SECONDS = 1 / 60

# Web audio buffer (frames). On the single web thread the per-turn render blocks
# audio mixing; a large buffer keeps enough pre-mixed audio queued to play
# through that hitch without underrunning. Latency is irrelevant for looping
# background music, so we bias big. Override per-run with "web_audio_buffer" in
# options.json to A/B test without editing code.
WEB_AUDIO_BUFFER = 16384


def _audio_driver_order() -> list[str | None]:
    if os.environ.get("PULSE_SERVER"):
        return ["pulseaudio", "pipewire", "alsa", None, "dsp"]
    return [None, "pipewire", "pulseaudio", "alsa", "dsp"]


def _coerce_scale(value: object) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return 1.0
    if scale < 0.5:
        return 0.5
    if scale > 3.0:
        return 3.0
    return scale


def _next_scale(current: float, direction: int = 1) -> float:
    current = _coerce_scale(current)
    if current not in _SCALE_STEPS:
        nearest = min(range(len(_SCALE_STEPS)), key=lambda idx: abs(_SCALE_STEPS[idx] - current))
        return _SCALE_STEPS[nearest]
    idx = _SCALE_STEPS.index(current)
    return _SCALE_STEPS[(idx + direction) % len(_SCALE_STEPS)]


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


def _pick_music_track(
    music_dir: Path,
    preferred_suffixes: tuple[str, ...] | None = None,
) -> Path | None:
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

    if preferred_suffixes:
        # Prefer formats the target runtime can actually decode (e.g. the web
        # build can't rely on mp3), then fall back to name order.
        def _rank(path: Path) -> int:
            suffix = path.suffix.lower()
            if suffix in preferred_suffixes:
                return preferred_suffixes.index(suffix)
            return len(preferred_suffixes)

        candidates.sort(key=lambda path: (_rank(path), path.name))

    return candidates[0]


def _web_audio_buffer_order(options: dict | None) -> list[int]:
    configured = None
    if isinstance(options, dict):
        configured = options.get("web_audio_buffer")

    first = configured if isinstance(configured, int) and configured > 0 else WEB_AUDIO_BUFFER
    ordered: list[int] = []
    for size in (first, 16384, 8192, 4096, 2048):
        if size > 0 and size not in ordered:
            ordered.append(size)
    return ordered


def _init_web_mixer(pygame: Any, options: dict | None) -> Any | None:
    # Let SDL open the device at the browser AudioContext's native rate
    # (AUDIO_ALLOW_FREQUENCY_CHANGE) instead of forcing 44100Hz. Forcing a rate
    # the browser doesn't use makes SDL resample every audio callback on the busy
    # main thread, which crackles; matching the device rate resamples each Sound
    # once at load instead. Paired with a moderate buffer (not the desktop ~3s).
    allowed = getattr(pygame, "AUDIO_ALLOW_FREQUENCY_CHANGE", 1)
    last_error: Exception | None = None
    for buffer_size in _web_audio_buffer_order(options):
        try:
            pygame.mixer.quit()
            pygame.mixer.init(
                frequency=AUDIO_SAMPLE_RATE,
                size=AUDIO_SAMPLE_SIZE,
                channels=AUDIO_CHANNELS,
                buffer=buffer_size,
                allowedchanges=allowed,
            )
            init = pygame.mixer.get_init()
            print(f"Web audio: buffer={buffer_size}, init={init}", file=sys.stderr)
            return pygame
        except Exception as exc:
            last_error = exc
            try:
                pygame.mixer.quit()
            except Exception:
                pass
    if last_error is not None:
        print(f"Audio disabled: {last_error}", file=sys.stderr)
    return None


def _init_pygame_mixer(options: dict | None = None) -> Any | None:
    try:
        pygame = __import__("pygame")
    except ModuleNotFoundError:
        return None

    if pygame.mixer.get_init() is not None:
        return pygame

    if IS_WEB:
        # The desktop driver-probing dance below is meaningless in the browser
        # (one audio backend) and its huge buffer is the wrong default for web.
        return _init_web_mixer(pygame, options)

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


# Keeps the web music Sound + channel alive (Sound is garbage-collected
# otherwise) and lets teardown stop the loop.
_web_music_state: dict[str, Any] = {}


def _start_background_music(options: dict | None = None) -> Any | None:
    pygame = _init_pygame_mixer(options)
    if pygame is None:
        return None

    music_dir = Path(__file__).resolve().parent.parent / "audio" / "music"

    if IS_WEB:
        # pygbag's SDL_mixer streaming (mixer.music) is unreliable and mp3 often
        # won't decode, so play music as an in-memory Sound loop -- the same path
        # the working SFX use. Reserve channel 0 for it so SFX (find_channel)
        # never steals the music channel.
        track = _pick_music_track(music_dir, preferred_suffixes=(".ogg", ".wav", ".mp3"))
        if track is None:
            return pygame
        try:
            pygame.mixer.set_reserved(1)
            channel = pygame.mixer.Channel(0)
            sound = pygame.mixer.Sound(str(track))
            channel.play(sound, loops=-1)
            _web_music_state["sound"] = sound
            _web_music_state["channel"] = channel
        except Exception as exc:
            print(f"Music disabled: {exc}", file=sys.stderr)
        return pygame

    track = _pick_music_track(music_dir)
    if track is None:
        return pygame

    try:
        pygame.mixer.music.load(str(track))
        pygame.mixer.music.play(-1)
    except Exception as exc:
        print(f"Music disabled: {exc}", file=sys.stderr)

    return pygame


class _CombatSfxPlayer:
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
        return Path(__file__).resolve().parent.parent / candidate

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


def _stop_background_music(pygame_module: Any | None) -> None:
    if pygame_module is None:
        return
    channel = _web_music_state.get("channel")
    if channel is not None:
        try:
            channel.stop()
        except Exception:
            pass
        _web_music_state.clear()
    try:
        pygame_module.mixer.music.stop()
    except Exception:
        pass
    try:
        pygame_module.mixer.quit()
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pyRL2")
    parser.add_argument(
        "--save_file",
        type=Path,
        help="load/save this file and bypass title screen + main menu",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="render a single gameplay frame, save it to this path, and exit",
    )
    parser.add_argument(
        "--rat-flood",
        action="store_true",
        help="spawn cave rats on every walkable tile for stress testing",
    )
    return parser.parse_args()


_MENU_TITLE_COLOR = (236, 236, 236)
_MENU_TEXT_COLOR = (210, 210, 210)
_MENU_MUTED_COLOR = (156, 156, 156)
_MENU_SELECTED_BG = (44, 44, 44)
_MENU_SELECTED_TEXT = (252, 252, 252)
_TITLE_SPLASH_SECONDS = 3.0
_GOBLIN_GREEN = (82, 166, 74)
_RAT_BROWN = (128, 92, 60)
_TREE_GREEN = (58, 138, 66)
_WELL_STONE = (150, 168, 190)
_STOVE_IRON = (120, 120, 128)
_DEER_TAN = (176, 132, 92)
_BED_WOOD = (156, 116, 72)
_BERRY_RED = (200, 74, 96)
_SEAWEED_GREEN = (54, 148, 116)
_FISH_SILVER = (210, 224, 236)
# Ocean creatures/plants render with a water-blue cell background so they blend
# into the sea instead of punching a black hole through the water tile (the
# fallback glyph sheet fills a solid background). Matches systems._WATER_BLUE.
_OCEAN_WATER_BG = (64, 118, 190)
_HUMAN_SKIN_TONES: list[tuple[int, int, int]] = [
    (255, 238, 220),
    (248, 227, 208),
    (241, 216, 196),
    (234, 205, 184),
    (227, 194, 172),
    (220, 183, 160),
    (213, 172, 148),
    (206, 161, 136),
    (194, 149, 123),
    (182, 137, 110),
    (170, 125, 98),
    (158, 113, 86),
    (146, 101, 74),
    (134, 89, 62),
    (122, 77, 50),
    (110, 65, 38),
]


def _capture_frame_screenshot(renderer: Renderer, output_path: Path) -> None:
    save_fn = getattr(renderer, "save_screenshot", None)
    if not callable(save_fn):
        print("Screenshot capture not supported by active renderer.", file=sys.stderr)
        return

    try:
        save_fn(output_path)
        print(f"Screenshot saved to {output_path}")
    except Exception as exc:
        print(f"Failed to save screenshot: {exc}", file=sys.stderr)


def _ui_grid_size(renderer: Renderer) -> tuple[int, int]:
    get_ui_grid = getattr(renderer, "get_ui_grid_size", None)
    if callable(get_ui_grid):
        cols, rows = get_ui_grid()
        if isinstance(cols, int) and isinstance(rows, int):
            return (max(20, cols), max(12, rows))

    get_grid = getattr(renderer, "get_grid_size", None)
    if callable(get_grid):
        cols, rows = get_grid()
        if isinstance(cols, int) and isinstance(rows, int):
            return (max(20, cols), max(12, rows))

    return (100, 40)


def _draw_ui_text(
    renderer: Renderer,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int] | None = None,
    max_cells: int | None = None,
) -> None:
    clipped_text = text
    if max_cells is not None:
        clipped_text = clipped_text[: max(0, max_cells)]

    draw_text_tinted = getattr(renderer, "draw_text_tinted", None)
    if color is not None and callable(draw_text_tinted):
        draw_text_tinted(x, y, clipped_text, color)
        return

    draw_text_clipped = getattr(renderer, "draw_text_clipped", None)
    if max_cells is not None and callable(draw_text_clipped):
        draw_text_clipped(x, y, text, max_cells)
        return

    renderer.draw_text(x, y, clipped_text)


def _fill_ui_cells(
    renderer: Renderer,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    fill_cells = getattr(renderer, "fill_cells", None)
    if callable(fill_cells):
        fill_cells(x, y, width, height, color)


def _draw_menu_shell(
    renderer: Renderer,
    title: str,
    subtitle: str | None = None,
    footer: str | None = None,
    width: int = 76,
    height: int = 24,
    overlay_game: bool = False,
) -> tuple[int, int, int, int]:
    drew_game_background = False
    if overlay_game and esper.get_processor(RenderProcessor) is not None:
        # The game behind a menu is static, so render it once and reuse a
        # snapshot for subsequent menu frames. Re-running esper.process() every
        # keypress is the main cost that makes menu scrolling stutter audio on
        # web. The snapshot is invalidated by the game loop when play resumes.
        blit_backdrop = getattr(renderer, "blit_backdrop", None)
        has_backdrop = getattr(renderer, "has_backdrop", None)
        capture_backdrop = getattr(renderer, "capture_backdrop", None)
        reused = False
        if callable(blit_backdrop) and callable(has_backdrop) and has_backdrop():
            reused = blit_backdrop()
        if not reused:
            esper.process(None)
            if callable(capture_backdrop):
                capture_backdrop()
        drew_game_background = True

    if drew_game_background:
        draw_overlay = getattr(renderer, "draw_overlay", None)
        if callable(draw_overlay):
            draw_overlay((0, 0, 0), 144)
    else:
        draw_menu_backdrop = getattr(renderer, "draw_menu_backdrop", None)
        if callable(draw_menu_backdrop):
            draw_menu_backdrop()
        else:
            renderer.clear()

    cols, rows = _ui_grid_size(renderer)
    width = max(30, min(width, cols - 2))
    height = max(12, min(height, rows - 2))
    x = max(1, (cols - width) // 2)
    y = max(1, (rows - height) // 2)

    draw_panel = getattr(renderer, "draw_panel", None)
    if callable(draw_panel):
        draw_panel(x, y, width, height, title=title)
    else:
        top = "+" + ("-" * max(0, width - 2)) + "+"
        renderer.draw_text(x, y, top)
        for row in range(1, max(1, height - 1)):
            renderer.draw_text(x, y + row, "|" + (" " * max(0, width - 2)) + "|")
        if height > 1:
            renderer.draw_text(x, y + height - 1, top)
        renderer.draw_text(x + 2, y, title)

    if subtitle:
        _draw_ui_text(renderer, x + 2, y + 2, subtitle, _MENU_TEXT_COLOR, width - 4)
    if footer:
        _draw_ui_text(renderer, x + 2, y + height - 2, footer, _MENU_MUTED_COLOR, width - 4)

    return (x, y, width, height)


def _draw_menu_options(
    renderer: Renderer,
    x: int,
    y: int,
    width: int,
    options: list[str],
    selected: int,
) -> None:
    for idx, option_text in enumerate(options):
        row_y = y + idx
        is_selected = idx == selected
        if is_selected:
            _fill_ui_cells(renderer, x, row_y, width, 1, _MENU_SELECTED_BG)

        prefix = ">" if is_selected else " "
        color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
        _draw_ui_text(
            renderer,
            x + 1,
            row_y,
            f"{prefix} {option_text}",
            color,
            max_cells=max(1, width - 2),
        )


def _scroll_start(selected_index: int, total_items: int, visible_rows: int) -> int:
    if total_items <= visible_rows:
        return 0

    candidate = selected_index - (visible_rows // 2)
    candidate = max(0, candidate)
    return min(candidate, total_items - visible_rows)


async def _await_action(renderer: Renderer) -> str | None:
    """Return the next input action, yielding to the browser event loop on web.

    Desktop keeps the renderer's blocking poll (unchanged idle behavior). Under
    pygbag we poll without blocking and sleep a real frame on every empty poll so
    the browser can pump events, repaint, and service audio between frames.
    """
    if not IS_WEB:
        return renderer.poll_action()

    poll_nonblocking = getattr(renderer, "poll_action_nonblocking", None)
    while True:
        if callable(poll_nonblocking):
            action = poll_nonblocking()
        else:
            action = renderer.poll_action()
        if action is not None:
            # Sleep a real frame before returning to the heavy, synchronous render
            # this action triggers. During sustained/held movement actions stream
            # out back-to-back; asyncio.sleep(0) returns to Python almost
            # immediately without giving the browser wall-clock time to run the
            # audio callback, so the buffer drains faster the longer you move and
            # the music stutters. A real ~16ms yield guarantees audio refill time
            # between renders (movement is already gated by key-repeat, so this
            # doesn't slow it) and is imperceptible latency for a single input.
            await asyncio.sleep(_WEB_IDLE_POLL_SECONDS)
            return action
        await asyncio.sleep(_WEB_IDLE_POLL_SECONDS)


# How often to re-render while the player has an active status animation, so the
# sub-second identifiers (e.g. the 0.5s "~") are visible without waiting for
# input. Shorter than the shortest status frame; only used while animating.
_STATUS_ANIM_POLL_SECONDS = 0.2


async def _await_action_or_idle(renderer: Renderer, idle_timeout: float) -> str | None:
    """Like ``_await_action`` but returns ``None`` after ``idle_timeout`` seconds
    with no input -- an animation tick. Non-blocking on both desktop and web so
    idle status animations keep playing while we wait."""
    poll_nonblocking = getattr(renderer, "poll_action_nonblocking", None)
    poll = poll_nonblocking if callable(poll_nonblocking) else renderer.poll_action
    deadline = time.monotonic() + idle_timeout
    while True:
        action = poll()
        if action is not None:
            if IS_WEB:
                await asyncio.sleep(_WEB_IDLE_POLL_SECONDS)
            return action
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(_WEB_IDLE_POLL_SECONDS)


async def _draw_title_screen(renderer: Renderer) -> bool:
    poll_action_nonblocking = getattr(renderer, "poll_action_nonblocking", None)
    splash_seconds = max(0.5, _TITLE_SPLASH_SECONDS)
    end_time = time.monotonic() + splash_seconds

    while True:
        draw_menu_backdrop = getattr(renderer, "draw_menu_backdrop", None)
        if callable(draw_menu_backdrop):
            draw_menu_backdrop()
        else:
            renderer.clear()

        cols, rows = _ui_grid_size(renderer)
        title = "TALES OF DERISION"
        subtitle = "A pygame ECS roguelike prototype"

        text_block_height = 3  # title row + spacer row + subtitle row
        title_y = max(1, (rows - text_block_height) // 2)
        subtitle_y = title_y + 2

        _draw_ui_text(renderer, max(0, (cols - len(title)) // 2), title_y, title, _MENU_TITLE_COLOR)
        _draw_ui_text(renderer, max(0, (cols - len(subtitle)) // 2), subtitle_y, subtitle, _MENU_TEXT_COLOR)
        renderer.present()

        if callable(poll_action_nonblocking):
            action = poll_action_nonblocking()
        else:
            action = await _await_action(renderer)

        if action == "quit":
            return False

        if action in {"menu_select", "confirm_action", "open_pause_menu"}:
            return True

        if time.monotonic() >= end_time:
            return True

        if callable(poll_action_nonblocking):
            # ~60fps splash tick. asyncio.sleep yields to the browser on web and
            # keeps CPU low on desktop (unlike time.sleep, which would block the
            # WASM thread and freeze the page).
            await asyncio.sleep(0.016)


async def _draw_main_menu(renderer: Renderer) -> str:
    options: list[tuple[str, str]] = [
        ("continue", "Continue"),
        ("new_game", "New Game"),
        ("quit", "Quit"),
    ]
    selected = 0

    while True:
        x, y, width, _height = _draw_menu_shell(
            renderer,
            title="MAIN MENU",
            subtitle="Choose your path",
            footer="[W/S] move   [Enter/Space] select   [Esc] quit",
            width=72,
            height=22,
        )
        labels = [label for _value, label in options]
        _draw_menu_options(renderer, x + 4, y + 7, width - 8, labels, selected)
        renderer.present()

        action = await _await_action(renderer)
        if action in {"quit", "open_pause_menu"}:
            return "quit"
        if action == "move_up":
            selected = (selected - 1) % len(options)
        elif action == "move_down":
            selected = (selected + 1) % len(options)
        elif action in {"menu_select", "confirm_action"}:
            return options[selected][0]


async def _draw_options_menu(renderer: Renderer, options: dict) -> str:
    def apply_renderer_options() -> None:
        apply_fn = getattr(renderer, "apply_options", None)
        if callable(apply_fn):
            apply_fn(options)

    selected = 0

    while True:
        fullscreen = bool(options.get("fullscreen", False))
        show_fps = bool(options.get("show_fps", False))
        tile_scale = _coerce_scale(options.get("tile_scale", 1.0))
        ui_scale = _coerce_scale(options.get("ui_scale", 1.0))
        items = [
            f"Fullscreen: {'ON' if fullscreen else 'OFF'}",
            f"Show FPS: {'ON' if show_fps else 'OFF'}",
            f"Tile Scale: < {tile_scale:.2f}x >",
            f"UI Scale: < {ui_scale:.2f}x >",
            "Back",
        ]

        x, y, width, _height = _draw_menu_shell(
            renderer,
            title="OPTIONS",
            subtitle="Display and interface",
            footer="[W/S] move   [A/D] adjust   [Enter/Space] toggle/select   [Esc] back",
            width=78,
            height=24,
            overlay_game=True,
        )
        _draw_menu_options(renderer, x + 4, y + 7, width - 8, items, selected)
        renderer.present()

        action = await _await_action(renderer)
        if action in {"open_pause_menu", "quit"}:
            return "back"
        if action == "move_up":
            selected = (selected - 1) % len(items)
        elif action == "move_down":
            selected = (selected + 1) % len(items)
        elif action == "move_left":
            if selected == 2:
                options["tile_scale"] = _next_scale(tile_scale, direction=-1)
                save_options(options)
                apply_renderer_options()
            elif selected == 3:
                options["ui_scale"] = _next_scale(ui_scale, direction=-1)
                save_options(options)
                apply_renderer_options()
        elif action == "move_right":
            if selected == 2:
                options["tile_scale"] = _next_scale(tile_scale, direction=1)
                save_options(options)
                apply_renderer_options()
            elif selected == 3:
                options["ui_scale"] = _next_scale(ui_scale, direction=1)
                save_options(options)
                apply_renderer_options()
        elif action in {"menu_select", "confirm_action"}:
            if selected == 0:
                options["fullscreen"] = not fullscreen
                save_options(options)
                apply_renderer_options()
            elif selected == 1:
                options["show_fps"] = not show_fps
                save_options(options)
            elif selected == 2:
                options["tile_scale"] = _next_scale(tile_scale, direction=1)
                save_options(options)
                apply_renderer_options()
            elif selected == 3:
                options["ui_scale"] = _next_scale(ui_scale, direction=1)
                save_options(options)
                apply_renderer_options()
            else:
                return "back"


async def _draw_pause_menu(renderer: Renderer, options: dict) -> str:
    menu_items = ["Save Game", "Options", "Quit"]
    selected = 0

    while True:
        x, y, width, _height = _draw_menu_shell(
            renderer,
            title="PAUSE",
            subtitle="Session controls",
            footer="[W/S] move   [Enter/Space] select   [Esc] resume",
            width=68,
            height=22,
            overlay_game=True,
        )
        _draw_menu_options(renderer, x + 4, y + 7, width - 8, menu_items, selected)
        renderer.present()

        action = await _await_action(renderer)
        if action == "open_pause_menu":
            return "resume"
        if action == "quit":
            return "quit"
        if action == "move_up":
            selected = (selected - 1) % len(menu_items)
        elif action == "move_down":
            selected = (selected + 1) % len(menu_items)
        elif action in {"menu_select", "confirm_action"}:
            chosen = menu_items[selected].lower().replace(" ", "_")
            if chosen == "options":
                await _draw_options_menu(renderer, options)
                continue
            return chosen


async def _run_startup_flow(
    renderer: Renderer,
    requested_save_file: Path | None,
    game_map: GameMap,
    player_position: Position,
) -> tuple[bool, GameMap, Position, Path]:
    selected_save_file = requested_save_file or DEFAULT_SAVE_FILE
    if requested_save_file is not None:
        loaded_map, loaded_player_position = load_game(
            selected_save_file,
            MAP_WIDTH,
            MAP_HEIGHT,
        )
        return (True, loaded_map, loaded_player_position, selected_save_file)

    if not await _draw_title_screen(renderer):
        return (False, game_map, player_position, selected_save_file)

    menu_choice = await _draw_main_menu(renderer)
    if menu_choice == "quit":
        return (False, game_map, player_position, selected_save_file)
    if menu_choice == "continue":
        loaded_map, loaded_player_position = load_game(
            DEFAULT_SAVE_FILE,
            MAP_WIDTH,
            MAP_HEIGHT,
        )
        return (True, loaded_map, loaded_player_position, selected_save_file)
    if menu_choice == "new_game":
        save_game(game_map, DEFAULT_SAVE_FILE, player_position)
        return (True, game_map, player_position, selected_save_file)

    return (False, game_map, player_position, selected_save_file)


def _default_equipment_slots() -> dict[str, str | None]:
    return {
        "head": None,
        "chest": None,
        "hands": None,
        "legs": None,
        "feet": None,
        "main hand": None,
        "off hand": None,
        "ring": None,
    }


def _infer_slot_for_item(item_name: str) -> str | None:
    lowered = item_name.lower()
    if any(token in lowered for token in ("helm", "hood", "hat", "crown")):
        return "head"
    if any(token in lowered for token in ("chest", "tunic", "armor", "robe", "coat")):
        return "chest"
    if any(token in lowered for token in ("glove", "gauntlet")):
        return "hands"
    if any(token in lowered for token in ("pants", "greave", "leggings", "trousers")):
        return "legs"
    if any(token in lowered for token in ("boot", "shoe", "sandal")):
        return "feet"
    if any(token in lowered for token in ("shield", "buckler", "offhand", "off-hand")):
        return "off hand"
    if any(token in lowered for token in ("ring",)):
        return "ring"
    if any(token in lowered for token in ("sword", "dagger", "axe", "mace", "club", "staff", "spear", "bow")):
        return "main hand"
    return None


def _equip_inventory_item(inventory: Inventory, equipment: Equipment, item_index: int) -> str:
    if item_index < 0 or item_index >= len(inventory.items):
        return "No item selected."

    item_name = inventory.items[item_index]
    slot_name = _infer_slot_for_item(item_name)
    if slot_name is None:
        return f"Cannot equip {item_name}."

    if slot_name not in equipment.slots:
        equipment.slots[slot_name] = None

    replaced = equipment.slots.get(slot_name)
    equipment.slots[slot_name] = item_name
    inventory.items.pop(item_index)
    if replaced:
        inventory.items.append(replaced)
        return f"Equipped {item_name} to {slot_name}; unequipped {replaced}."
    return f"Equipped {item_name} to {slot_name}."


def _unequip_slot(inventory: Inventory, equipment: Equipment, slot_name: str) -> str:
    current = equipment.slots.get(slot_name)
    if not current:
        return f"Nothing equipped in {slot_name}."

    equipment.slots[slot_name] = None
    inventory.items.append(current)
    return f"Unequipped {current} from {slot_name}."


def _item_visual(item_name: str) -> tuple[str, str, tuple[int, int, int] | None, tuple[int, int, int] | None]:
    lowered = item_name.lower()

    if any(token in lowered for token in ("sword", "dagger", "knife", "rapier", "spear", "javelin", "halberd", "axe", "mace", "club", "staff", "bow", "crossbow")):
        return (")", "valuable", (220, 220, 220), None)

    if any(token in lowered for token in ("helm", "hood", "hat", "crown", "chest", "tunic", "armor", "robe", "coat", "shield", "buckler", "pants", "greave", "leggings", "trousers", "boot", "shoe", "sandal", "glove", "gauntlet")):
        return ("[", "valuable", (176, 188, 214), None)

    if any(token in lowered for token in ("potion", "flask", "waterskin")):
        return ("!", "valuable", (120, 196, 255), None)

    if any(token in lowered for token in ("bandage", "scroll", "map", "book", "torch")):
        return ("?", "valuable", (236, 210, 150), None)

    if any(token in lowered for token in ("wood", "log", "branch", "kindling")):
        return ("=", "valuable", (150, 111, 70), None)

    if "cooked" in lowered and "meat" in lowered:
        return ("%", "valuable", (196, 118, 74), None)

    if any(token in lowered for token in ("apple", "bread", "meat", "food")):
        return ("%", "valuable", (222, 142, 106), None)

    if any(token in lowered for token in ("coin", "gem", "charm", "ring")):
        return ("$", "valuable", (245, 216, 118), None)

    return ("*", "valuable", None, None)


def _human_skin_tone(seed_text: str, offset: int = 0) -> tuple[int, int, int]:
    seed = 0
    for index, char in enumerate(seed_text):
        seed += (index + 1) * ord(char)
    return _HUMAN_SKIN_TONES[(seed + offset) % len(_HUMAN_SKIN_TONES)]


def _first_player_entity() -> int | None:
    for ent, (_pos, _player) in esper.get_components(Position, Player):
        return ent
    return None


def _entity_name(entity_id: int, fallback: str = "Unknown") -> str:
    if esper.has_component(entity_id, Name):
        return esper.component_for_entity(entity_id, Name).value
    return fallback


def _direction_target_xy(direction_action: str | None, origin: Position) -> tuple[int, int] | None:
    if direction_action in _CARDINAL_ACTION_DELTAS:
        dx, dy = _CARDINAL_ACTION_DELTAS[direction_action]
        return (origin.x + dx, origin.y + dy)
    if direction_action in _VECTOR_TO_ACTION.values():
        for (dx, dy), action in _VECTOR_TO_ACTION.items():
            if action == direction_action:
                return (origin.x + dx, origin.y + dy)
        return None


def _interaction_target_xy(direction_action: str | None, origin: Position) -> tuple[int, int]:
    target_xy = _direction_target_xy(direction_action, origin)
    if target_xy is not None:
        return target_xy
    return (origin.x, origin.y)


def _action_from_held_keys(
    held_directions: set[str],
    pressed_order: dict[str, int] | None = None,
) -> str | None:
    if pressed_order is None:
        pressed_order = {}

    use_recent_preference = bool(pressed_order)

    dx = 0
    dy = 0

    left_held = "move_left" in held_directions
    right_held = "move_right" in held_directions
    up_held = "move_up" in held_directions
    down_held = "move_down" in held_directions

    if left_held and right_held and use_recent_preference:
        left_order = pressed_order.get("move_left", -1)
        right_order = pressed_order.get("move_right", -1)
        dx = -1 if left_order >= right_order else 1
    elif left_held and not right_held:
        dx = -1
    elif right_held and not left_held:
        dx = 1

    if up_held and down_held and use_recent_preference:
        up_order = pressed_order.get("move_up", -1)
        down_order = pressed_order.get("move_down", -1)
        dy = -1 if up_order >= down_order else 1
    elif up_held and not down_held:
        dy = -1
    elif down_held and not up_held:
        dy = 1

    if dx == 0 and dy == 0:
        return None
    return _VECTOR_TO_ACTION.get((dx, dy))


def _find_interaction_npc(direction_action: str | None) -> int | None:
    player_ent = _first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return None

    player_pos = esper.component_for_entity(player_ent, Position)
    position_to_npc: dict[tuple[int, int], int] = {}
    for ent, (pos, _npc) in esper.get_components(Position, NPC):
        if esper.has_component(ent, Enemy):
            continue
        position_to_npc[(pos.x, pos.y)] = ent

    target_xy = _interaction_target_xy(direction_action, player_pos)
    return position_to_npc.get(target_xy)


def _find_interaction_creature(direction_action: str | None) -> int | None:
    """Any creature (friendly, wild, or hostile) on the faced tile. Used so the
    player can examine/interact with anything adjacent, not just friendlies."""
    player_ent = _first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return None

    player_pos = esper.component_for_entity(player_ent, Position)
    target_xy = _interaction_target_xy(direction_action, player_pos)
    for ent, (pos, _npc) in esper.get_components(Position, NPC):
        if (pos.x, pos.y) == target_xy:
            return ent
    return None


def _disposition_of(ent: int) -> str:
    if esper.has_component(ent, Player):
        return "You"
    if esper.has_component(ent, Enemy):
        return "Hostile"
    if esper.has_component(ent, Friendly):
        return "Friendly"
    if esper.has_component(ent, Deer):
        return "Wild animal"
    return "Neutral"


def _friendship_label(score: float) -> str:
    """A word for a friendship score, for status/dialogue screens."""
    if score >= 60:
        return "Close friend"
    if score >= 25:
        return "Friend"
    if score >= 5:
        return "Friendly"
    if score <= -60:
        return "Enemy"
    if score <= -25:
        return "Disliked"
    if score <= -5:
        return "Cold"
    return "Stranger"


def _player_talk(player_ent: int, npc_ent: int) -> str:
    """The player chats with a villager: nudge friendship both ways (via the same
    ``systems.interact`` the NPC social AI uses), pop a speech bubble above them,
    and return a one-line outcome for the dialogue footer."""
    clock = world_clock()
    turn = clock.turn if clock is not None else 0
    _delta_player, delta_npc = interact(player_ent, npc_ent, turn)
    score = 0.0
    if esper.has_component(player_ent, Relationships):
        score = friendship(esper.component_for_entity(player_ent, Relationships), npc_ent)
    indicator = "++" if delta_npc > 0 else ("--" if delta_npc < 0 else "~")
    return f"{indicator} friendship now {int(score)} ({_friendship_label(score)})"


def _creature_status_lines(game_map: GameMap, ent: int) -> list[str]:
    """Human-readable status block for any creature (the player, an NPC, a mob):
    name, disposition, needs, and active statuses."""
    lines = [
        f"Name: {_entity_name(ent, fallback='Unknown')}",
        f"Disposition: {_disposition_of(ent)}",
    ]
    if esper.has_component(ent, Personality):
        traits = esper.component_for_entity(ent, Personality).traits
        lines.append("Traits: " + (", ".join(traits) if traits else "None"))
        player_ent = _first_player_entity()
        if player_ent is not None and esper.has_component(ent, Relationships):
            rel = esper.component_for_entity(ent, Relationships)
            score = friendship(rel, player_ent)
            lines.append(f"Friendship: {int(score)} ({_friendship_label(score)})")
    if esper.has_component(ent, Needs):
        needs = esper.component_for_entity(ent, Needs)
        lines.append(f"Hunger: {int(needs.hunger)}%")
        lines.append(f"Thirst: {int(needs.thirst)}%")
        lines.append(f"Tiredness: {int(needs.tiredness)}%")

    statuses: list[str] = []
    if esper.has_component(ent, Position):
        pos = esper.component_for_entity(ent, Position)
        statuses = [status_label(name) for name in active_statuses(game_map, ent, pos)]
    lines.append("Status: " + (", ".join(statuses) if statuses else "Normal"))
    return lines


def _find_interaction_corpse(direction_action: str | None) -> int | None:
    player_ent = _first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return None

    player_pos = esper.component_for_entity(player_ent, Position)
    position_to_corpse: dict[tuple[int, int], int] = {
        (pos.x, pos.y): ent
        for ent, (pos, _corpse) in esper.get_components(Position, Corpse)
        if _entity_has_tradeable_items(ent)
    }

    target_xy = _interaction_target_xy(direction_action, player_pos)
    return position_to_corpse.get(target_xy)


def _npc_info_lines(game_map: GameMap, npc_ent: int) -> list[str]:
    dialogue_line = "..."
    if esper.has_component(npc_ent, Dialogue):
        dialogue_line = esper.component_for_entity(npc_ent, Dialogue).line

    inventory_count = 0
    if esper.has_component(npc_ent, Inventory):
        inventory_count = len(esper.component_for_entity(npc_ent, Inventory).items)

    equipped_count = 0
    if esper.has_component(npc_ent, Equipment):
        slots = esper.component_for_entity(npc_ent, Equipment).slots
        equipped_count = sum(1 for item in slots.values() if item)

    # Name / Disposition / Needs / Status, then dialogue-specific detail.
    return _creature_status_lines(game_map, npc_ent) + [
        f"Says: {dialogue_line}",
        f"Stock: {inventory_count} carried, {equipped_count} equipped",
    ]


@dataclass
class _TradeEntry:
    kind: str
    item_name: str
    slot_name: str | None = None
    item_index: int | None = None


def _list_trade_entries(actor_ent: int) -> list[_TradeEntry]:
    entries: list[_TradeEntry] = []

    if esper.has_component(actor_ent, Inventory):
        items = esper.component_for_entity(actor_ent, Inventory).items
        for idx, item_name in enumerate(items):
            entries.append(_TradeEntry(kind="inventory", item_name=item_name, item_index=idx))

    if esper.has_component(actor_ent, Equipment):
        slots = esper.component_for_entity(actor_ent, Equipment).slots
        for slot_name in sorted(slots.keys()):
            equipped_item = slots.get(slot_name)
            if equipped_item:
                entries.append(_TradeEntry(kind="equipment", item_name=equipped_item, slot_name=slot_name))

    return entries


def _remove_trade_entry(actor_ent: int, entry: _TradeEntry) -> str | None:
    if entry.kind == "inventory":
        if not esper.has_component(actor_ent, Inventory):
            return None
        inventory = esper.component_for_entity(actor_ent, Inventory)
        if entry.item_index is None:
            return None
        if entry.item_index < 0 or entry.item_index >= len(inventory.items):
            return None
        return inventory.items.pop(entry.item_index)

    if entry.kind == "equipment":
        if not esper.has_component(actor_ent, Equipment) or not entry.slot_name:
            return None
        equipment = esper.component_for_entity(actor_ent, Equipment)
        item = equipment.slots.get(entry.slot_name)
        if not item:
            return None
        equipment.slots[entry.slot_name] = None
        return item

    return None


def _ensure_inventory(actor_ent: int) -> Inventory:
    if esper.has_component(actor_ent, Inventory):
        return esper.component_for_entity(actor_ent, Inventory)

    inventory = Inventory(items=[])
    esper.add_component(actor_ent, inventory)
    return inventory


def _trade_item(source_ent: int, target_ent: int, entry: _TradeEntry) -> str:
    source_name = _entity_name(source_ent)
    target_name = _entity_name(target_ent)

    item_name = _remove_trade_entry(source_ent, entry)
    if item_name is None:
        return "Trade failed. Item was no longer available."

    target_inventory = _ensure_inventory(target_ent)
    target_inventory.items.append(item_name)
    return f"{source_name} traded {item_name} to {target_name}."


def _entity_has_tradeable_items(actor_ent: int) -> bool:
    if esper.has_component(actor_ent, Inventory):
        inventory = esper.component_for_entity(actor_ent, Inventory)
        if inventory.items:
            return True

    if esper.has_component(actor_ent, Equipment):
        slots = esper.component_for_entity(actor_ent, Equipment).slots
        if any(item for item in slots.values()):
            return True

    return False


def _loot_item_from_corpse(corpse_ent: int, player_ent: int, entry: _TradeEntry) -> str:
    corpse_name = _entity_name(corpse_ent, fallback="Corpse")

    item_name = _remove_trade_entry(corpse_ent, entry)
    if item_name is None:
        return "Loot failed. Item was no longer available."

    player_inventory = _ensure_inventory(player_ent)
    player_inventory.items.append(item_name)

    if not _entity_has_tradeable_items(corpse_ent):
        return f"You looted {item_name} from {corpse_name}. Nothing else of value remains."

    return f"You looted {item_name} from {corpse_name}."


def _find_adjacent_feature(direction_action: str | None, component: type) -> int | None:
    """Return the entity carrying ``component`` on the tile the player is facing
    (the aimed direction), or ``None``. Used to interact with wells, stoves, and
    trees the same way corpses/NPCs are targeted."""
    player_ent = _first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return None

    player_pos = esper.component_for_entity(player_ent, Position)
    target_xy = _interaction_target_xy(direction_action, player_pos)
    for ent, (pos, _feature) in esper.get_components(Position, component):
        if (pos.x, pos.y) == target_xy:
            return ent
    return None


def _chop_tree(tree_ent: int, player_ent: int) -> str:
    """Chop one load of wood off a tree. When the last load is taken the tree
    falls (its entity is removed, freeing the tile)."""
    tree = esper.component_for_entity(tree_ent, Tree)
    inventory = _ensure_inventory(player_ent)
    inventory.items.append(WOOD)
    tree.wood -= 1
    if tree.wood <= 0:
        esper.delete_entity(tree_ent, immediate=True)
        return "You fell the tree, gathering a last piece of wood."
    return "You chop a piece of wood from the tree."


def _harvest_bush(bush_ent: int, player_ent: int) -> str:
    """Pick a ripe berry bush into the player's pack. The bush goes bare and its
    berries regrow a week later (handled by the growth processor)."""
    if pick_berries(bush_ent, world_clock()):
        _ensure_inventory(player_ent).items.append(BERRIES)
        return "You pick a handful of ripe berries."
    return "The bush has no ripe berries yet."


def _drink_from_well(well_ent: int, player_ent: int) -> str:
    """Drink from a well, quenching thirst. Wells are a renewable source."""
    if not esper.has_component(player_ent, Needs):
        return "The water is cool and clear."

    needs = esper.component_for_entity(player_ent, Needs)
    if needs.thirst <= 0:
        return "You drink from the well, but you were not thirsty."

    needs.thirst = 0.0
    return "You drink deeply from the well. Your thirst is quenched."


def _cook_at_stove(stove_ent: int, player_ent: int) -> str:
    """Turn one Wood + one Raw Meat into a Cooked Meat at the stove. Needs both:
    wood fuels the fire, raw meat is the ingredient."""
    if not esper.has_component(player_ent, Inventory):
        return "You have nothing to cook."

    inventory = esper.component_for_entity(player_ent, Inventory)
    raw_meat = next((item for item in inventory.items if is_raw_meat(item)), None)

    if raw_meat is None:
        return "You have no raw meat to cook. Butcher it from a corpse first."
    if WOOD not in inventory.items:
        return "The stove is cold. You need wood to make a fire."

    inventory.items.remove(WOOD)
    inventory.items.remove(raw_meat)
    inventory.items.append(cook_meat(raw_meat))
    return f"You light the stove and cook the {raw_meat.lower()} into a hot meal."


def _apply_consumable(player_ent: int, item_index: int) -> str | None:
    """Eat/drink the inventory item at ``item_index`` if it is consumable,
    removing it and reducing the matching need. Returns a message, or ``None``
    when the item is not food/drink (so the caller can fall back to equipping)."""
    if not esper.has_component(player_ent, Inventory):
        return None

    inventory = esper.component_for_entity(player_ent, Inventory)
    if item_index < 0 or item_index >= len(inventory.items):
        return None

    item_name = inventory.items[item_index]
    hunger_value = hunger_restored(item_name)
    thirst_value = thirst_restored(item_name)
    if hunger_value is None and thirst_value is None:
        return None

    if not esper.has_component(player_ent, Needs):
        esper.add_component(player_ent, Needs())
    needs = esper.component_for_entity(player_ent, Needs)

    inventory.items.pop(item_index)
    if hunger_value is not None:
        needs.hunger = max(0.0, needs.hunger - hunger_value)
        return f"You eat the {item_name}."
    needs.thirst = max(0.0, needs.thirst - thirst_value)
    return f"You drink the {item_name}."


# Below this tiredness the player gets a "not tired enough" nudge instead of
# sleeping; the cap bounds the fast-forward loop so a stuck sleeper always wakes.
_MIN_SLEEP_TIREDNESS = 1.0
_SLEEP_MAX_TURNS = 400


def _bed_near_player() -> int | None:
    """The nearest Bed on or beside the player's tile (so the sleep action can bed
    them down rather than pitching a camp), or ``None`` if none is adjacent."""
    player_ent = _first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return None
    ppos = esper.component_for_entity(player_ent, Position)
    best: int | None = None
    best_dist = 2
    for ent, (pos, _bed) in esper.get_components(Position, Bed):
        dist = max(abs(pos.x - ppos.x), abs(pos.y - ppos.y))
        if dist <= 1 and dist < best_dist:
            best, best_dist = ent, dist
    return best


async def _confirm_if_owned_by_other(renderer: Renderer, ent: int, noun: str, verb: str) -> bool:
    """When ``ent`` belongs to someone other than the player, warn whose it is and
    ask for confirmation. Returns True if the player may go ahead (their own/
    unowned property proceeds silently; a decline returns False)."""
    player_ent = _first_player_entity()
    owner = bed_owner(ent)
    if owner is None or owner == player_ent:
        return True
    owner_name = _entity_name(owner, fallback="someone else")
    return await _confirm(
        renderer,
        title="Not Yours",
        lines=[
            f"This {noun} belongs to {owner_name}.",
            f"Are you sure you want to {verb}?",
        ],
    )


async def _confirm(renderer: Renderer, title: str, lines: list[str]) -> bool:
    """A yes/no prompt over the game. Defaults to No; Esc cancels. Returns True
    only if the player explicitly chooses Yes."""
    options = ["No", "Yes"]
    selected = 0
    while True:
        x, y, width, height = _draw_menu_shell(
            renderer,
            title=title,
            subtitle=None,
            footer="[W/S] choose   [Enter] confirm   [Esc] cancel",
            width=60,
            height=12,
            overlay_game=True,
        )
        for idx, line in enumerate(lines):
            _draw_ui_text(renderer, x + 3, y + 3 + idx, line, _MENU_TEXT_COLOR, width - 6)
        _draw_menu_options(renderer, x + 4, y + 3 + len(lines) + 1, width - 8, options, selected)
        renderer.present()

        action = await _await_action(renderer)
        if action in {"quit", "open_pause_menu"}:
            return False
        if action == "move_up":
            selected = (selected - 1) % len(options)
            continue
        if action == "move_down":
            selected = (selected + 1) % len(options)
            continue
        if action in {"menu_select", "confirm_action"}:
            return options[selected] == "Yes"


async def _sleep_player(renderer: Renderer, in_camp: bool) -> None:
    """Send the player to sleep and fast-forward turns until they wake rested.

    The whole world keeps simulating during the rest (NPCs act, needs shift, the
    clock advances), so a night's sleep really passes the night. Each turn yields
    to the browser so the web build's single thread stays responsive."""
    player_ent = _first_player_entity()
    if player_ent is None:
        return
    if not esper.has_component(player_ent, Needs):
        esper.add_component(player_ent, Needs())
    needs = esper.component_for_entity(player_ent, Needs)
    if needs.tiredness < _MIN_SLEEP_TIREDNESS:
        queue_message("You are not tired enough to sleep.")
        esper.process(None)
        return

    queue_message(
        "You set up camp and settle in to rest." if in_camp else "You climb into bed and close your eyes."
    )
    go_to_sleep(player_ent, in_camp=in_camp)

    turns = 0
    while esper.has_component(player_ent, Asleep) and turns < _SLEEP_MAX_TURNS:
        esper.process(WAIT_ACTION)
        turns += 1
        await asyncio.sleep(0)

    # Safety net: never leave the player stuck asleep past the cap.
    if esper.has_component(player_ent, Asleep):
        wake_up(player_ent)
    esper.process(None)


# Order of recipes shown in the Crafting tab.
_CRAFT_MENU = [WOOD_WALL, WOOD_WINDOW, WOOD_DOOR]


def _craft_item(player_ent: int, item_name: str) -> str:
    """Craft one ``item_name`` from Wood in the player's pack, or explain why it
    can't be made. The crafted piece lands in the inventory to be placed later."""
    cost = craft_cost(item_name)
    if cost is None:
        return f"You don't know how to craft {item_name}."
    inventory = _ensure_inventory(player_ent)
    wood_count = inventory.items.count(WOOD)
    if wood_count < cost:
        return f"You need {cost} Wood to craft a {item_name} (you have {wood_count})."
    for _ in range(cost):
        inventory.items.remove(WOOD)
    inventory.items.append(item_name)
    return f"You craft a {item_name} from {cost} Wood."


def _place_buildable_at(
    player_ent: int, game_map: GameMap, item_name: str, target_xy: tuple[int, int]
) -> str:
    """Place a buildable piece onto ``target_xy``, turning it into the matching
    map tile and consuming the item. Returns a log message."""
    tile = placed_tile(item_name)
    if tile is None:
        return f"You can't place the {item_name}."
    tx, ty = target_xy
    if not game_map.in_bounds(tx, ty):
        return "You can't build there."
    if tx == 0 or ty == 0 or tx == game_map.width - 1 or ty == game_map.height - 1:
        return "You can't build on the edge of the world."
    if game_map.tile_at(tx, ty) != game_map.FLOOR:
        return "You can only build on open ground."
    for _ent, (pos,) in esper.get_components(Position):
        if (pos.x, pos.y) == (tx, ty):
            return "Something is in the way."

    inventory = _ensure_inventory(player_ent)
    if item_name not in inventory.items:
        return f"You have no {item_name} to place."
    if not game_map.set_tile(tx, ty, tile):
        return "You can't build there."
    inventory.items.remove(item_name)
    return f"You build a {item_name}."


async def _place_from_inventory(renderer: Renderer, game_map: GameMap, item_name: str) -> None:
    """Prompt for a direction and build the selected piece on that adjacent tile.
    Any non-direction key cancels and keeps the item."""
    player_ent = _first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return
    queue_message(f"Place the {item_name}: press a direction (Esc/other to cancel).")
    esper.process(None)

    action = await _await_action(renderer)
    player_pos = esper.component_for_entity(player_ent, Position)
    target = _direction_target_xy(action, player_pos)
    if target is None:
        queue_message(f"You put the {item_name} away.")
    else:
        queue_message(_place_buildable_at(player_ent, game_map, item_name, target))
    esper.process(None)


async def _draw_trade_menu(renderer: Renderer, npc_ent: int) -> str:
    player_ent = _first_player_entity()
    if player_ent is None:
        return "close"

    selected_panel = "left"
    selected_npc_idx = 0
    selected_player_idx = 0
    message = "Enter: trade selected item  A/D: switch side  W/S: move"

    while True:
        npc_entries = _list_trade_entries(npc_ent)
        player_entries = _list_trade_entries(player_ent)

        if npc_entries:
            selected_npc_idx = max(0, min(selected_npc_idx, len(npc_entries) - 1))
        else:
            selected_npc_idx = 0

        if player_entries:
            selected_player_idx = max(0, min(selected_player_idx, len(player_entries) - 1))
        else:
            selected_player_idx = 0

        npc_name = _entity_name(npc_ent, fallback="NPC")
        player_name = _entity_name(player_ent, fallback="You")

        x, y, width, height = _draw_menu_shell(
            renderer,
            title=f"TRADE - {npc_name}",
            subtitle="Esc/I closes   A/D switches side",
            footer=message,
            width=98,
            height=30,
            overlay_game=True,
        )

        content_y = y + 4
        content_h = max(10, height - 7)
        left_w = max(18, (width - 6) // 2)
        right_w = max(18, width - left_w - 6)
        left_x = x + 2
        right_x = left_x + left_w + 2

        draw_panel = getattr(renderer, "draw_panel", None)
        if callable(draw_panel):
            draw_panel(left_x, content_y, left_w, content_h, title=f"{npc_name} STOCK")
            draw_panel(right_x, content_y, right_w, content_h, title=f"{player_name} STOCK")

        list_start_y = content_y + 3
        visible_rows = max(1, content_h - 5)

        if npc_entries:
            npc_start = _scroll_start(selected_npc_idx, len(npc_entries), visible_rows)
            npc_end = min(len(npc_entries), npc_start + visible_rows)
            for row_offset, entry_idx in enumerate(range(npc_start, npc_end)):
                entry = npc_entries[entry_idx]
                row_y = list_start_y + row_offset
                is_selected = selected_panel == "left" and entry_idx == selected_npc_idx
                if is_selected:
                    _fill_ui_cells(renderer, left_x + 1, row_y, left_w - 2, 1, _MENU_SELECTED_BG)

                if entry.kind == "equipment":
                    label = f"[E] {entry.slot_name}: {entry.item_name}"
                else:
                    label = entry.item_name
                prefix = ">" if is_selected else " "
                color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
                _draw_ui_text(renderer, left_x + 2, row_y, f"{prefix} {label}", color, left_w - 4)
        else:
            is_selected = selected_panel == "left"
            if is_selected:
                _fill_ui_cells(renderer, left_x + 1, list_start_y, left_w - 2, 1, _MENU_SELECTED_BG)
            color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
            prefix = ">" if is_selected else " "
            _draw_ui_text(renderer, left_x + 2, list_start_y, f"{prefix} (empty)", color, left_w - 4)

        if player_entries:
            player_start = _scroll_start(selected_player_idx, len(player_entries), visible_rows)
            player_end = min(len(player_entries), player_start + visible_rows)
            for row_offset, entry_idx in enumerate(range(player_start, player_end)):
                entry = player_entries[entry_idx]
                row_y = list_start_y + row_offset
                is_selected = selected_panel == "right" and entry_idx == selected_player_idx
                if is_selected:
                    _fill_ui_cells(renderer, right_x + 1, row_y, right_w - 2, 1, _MENU_SELECTED_BG)

                if entry.kind == "equipment":
                    label = f"[E] {entry.slot_name}: {entry.item_name}"
                else:
                    label = entry.item_name
                prefix = ">" if is_selected else " "
                color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
                _draw_ui_text(renderer, right_x + 2, row_y, f"{prefix} {label}", color, right_w - 4)
        else:
            is_selected = selected_panel == "right"
            if is_selected:
                _fill_ui_cells(renderer, right_x + 1, list_start_y, right_w - 2, 1, _MENU_SELECTED_BG)
            color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
            prefix = ">" if is_selected else " "
            _draw_ui_text(renderer, right_x + 2, list_start_y, f"{prefix} (empty)", color, right_w - 4)

        renderer.present()

        action = await _await_action(renderer)
        if action in {"open_pause_menu", "open_inventory"}:
            return "close"
        if action == "quit":
            return "quit"
        if action == "move_left":
            selected_panel = "left"
            continue
        if action == "move_right":
            selected_panel = "right"
            continue
        if action == "move_up":
            if selected_panel == "left" and npc_entries:
                selected_npc_idx = (selected_npc_idx - 1) % len(npc_entries)
            elif selected_panel == "right" and player_entries:
                selected_player_idx = (selected_player_idx - 1) % len(player_entries)
            continue
        if action == "move_down":
            if selected_panel == "left" and npc_entries:
                selected_npc_idx = (selected_npc_idx + 1) % len(npc_entries)
            elif selected_panel == "right" and player_entries:
                selected_player_idx = (selected_player_idx + 1) % len(player_entries)
            continue

        if action in {"menu_select", "confirm_action"}:
            if selected_panel == "left":
                if not npc_entries:
                    message = f"{npc_name} has nothing to trade."
                    continue
                entry = npc_entries[selected_npc_idx]
                message = _trade_item(npc_ent, player_ent, entry)
            else:
                if not player_entries:
                    message = "You have nothing to trade."
                    continue
                entry = player_entries[selected_player_idx]
                message = _trade_item(player_ent, npc_ent, entry)


async def _draw_loot_menu(renderer: Renderer, corpse_ent: int) -> str:
    player_ent = _first_player_entity()
    if player_ent is None:
        return "close"

    selected_panel = "left"
    selected_corpse_idx = 0
    selected_player_idx = 0
    message = "Enter: loot selected item  A/D: switch side  W/S: move"

    while True:
        if not esper.entity_exists(corpse_ent):
            return "close"

        corpse_entries = _list_trade_entries(corpse_ent)
        player_entries = _list_trade_entries(player_ent)

        if corpse_entries:
            selected_corpse_idx = max(0, min(selected_corpse_idx, len(corpse_entries) - 1))
        else:
            selected_corpse_idx = 0

        if player_entries:
            selected_player_idx = max(0, min(selected_player_idx, len(player_entries) - 1))
        else:
            selected_player_idx = 0

        corpse_name = _entity_name(corpse_ent, fallback="Corpse")
        player_name = _entity_name(player_ent, fallback="You")

        x, y, width, height = _draw_menu_shell(
            renderer,
            title=f"LOOT - {corpse_name}",
            subtitle="Esc/I closes   A/D switches side",
            footer=message,
            width=98,
            height=30,
            overlay_game=True,
        )

        content_y = y + 4
        content_h = max(10, height - 7)
        left_w = max(18, (width - 6) // 2)
        right_w = max(18, width - left_w - 6)
        left_x = x + 2
        right_x = left_x + left_w + 2

        draw_panel = getattr(renderer, "draw_panel", None)
        if callable(draw_panel):
            draw_panel(left_x, content_y, left_w, content_h, title=f"{corpse_name} ITEMS")
            draw_panel(right_x, content_y, right_w, content_h, title=f"{player_name} STOCK")

        list_start_y = content_y + 3
        visible_rows = max(1, content_h - 5)
        left_line_width = max(1, left_w - 4)
        right_line_width = max(1, right_w - 4)

        if corpse_entries:
            start = _scroll_start(selected_corpse_idx, len(corpse_entries), visible_rows)
            end = min(len(corpse_entries), start + visible_rows)
            for row_offset, entry_idx in enumerate(range(start, end)):
                entry = corpse_entries[entry_idx]
                row_y = list_start_y + row_offset
                is_selected = selected_panel == "left" and entry_idx == selected_corpse_idx
                if is_selected:
                    _fill_ui_cells(renderer, left_x + 1, row_y, left_w - 2, 1, _MENU_SELECTED_BG)

                if entry.kind == "equipment":
                    label = f"[E] {entry.slot_name}: {entry.item_name}"
                else:
                    label = entry.item_name

                prefix = ">" if is_selected else " "
                color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
                _draw_ui_text(renderer, left_x + 2, row_y, f"{prefix} {label}", color, left_line_width)
        else:
            is_selected = selected_panel == "left"
            if is_selected:
                _fill_ui_cells(renderer, left_x + 1, list_start_y, left_w - 2, 1, _MENU_SELECTED_BG)
            prefix = ">" if is_selected else " "
            color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
            _draw_ui_text(
                renderer,
                left_x + 2,
                list_start_y,
                f"{prefix} (empty)",
                color,
                left_line_width,
            )

        if player_entries:
            start = _scroll_start(selected_player_idx, len(player_entries), visible_rows)
            end = min(len(player_entries), start + visible_rows)
            for row_offset, entry_idx in enumerate(range(start, end)):
                entry = player_entries[entry_idx]
                row_y = list_start_y + row_offset
                is_selected = selected_panel == "right" and entry_idx == selected_player_idx
                if is_selected:
                    _fill_ui_cells(renderer, right_x + 1, row_y, right_w - 2, 1, _MENU_SELECTED_BG)

                if entry.kind == "equipment":
                    label = f"[E] {entry.slot_name}: {entry.item_name}"
                else:
                    label = entry.item_name

                prefix = ">" if is_selected else " "
                color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
                _draw_ui_text(renderer, right_x + 2, row_y, f"{prefix} {label}", color, right_line_width)
        else:
            is_selected = selected_panel == "right"
            if is_selected:
                _fill_ui_cells(renderer, right_x + 1, list_start_y, right_w - 2, 1, _MENU_SELECTED_BG)
            prefix = ">" if is_selected else " "
            color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
            _draw_ui_text(
                renderer,
                right_x + 2,
                list_start_y,
                f"{prefix} (empty)",
                color,
                right_line_width,
            )

        renderer.present()

        action = await _await_action(renderer)
        if action in {"open_pause_menu", "open_inventory"}:
            return "close"
        if action == "quit":
            return "quit"
        if action == "move_left":
            selected_panel = "left"
            continue
        if action == "move_right":
            selected_panel = "right"
            continue
        if action == "move_up":
            if selected_panel == "left" and corpse_entries:
                selected_corpse_idx = (selected_corpse_idx - 1) % len(corpse_entries)
            elif selected_panel == "right" and player_entries:
                selected_player_idx = (selected_player_idx - 1) % len(player_entries)
            continue
        if action == "move_down":
            if selected_panel == "left" and corpse_entries:
                selected_corpse_idx = (selected_corpse_idx + 1) % len(corpse_entries)
            elif selected_panel == "right" and player_entries:
                selected_player_idx = (selected_player_idx + 1) % len(player_entries)
            continue

        if action in {"menu_select", "confirm_action"}:
            if selected_panel == "left":
                if not corpse_entries:
                    message = f"{corpse_name} has nothing to loot."
                    continue
                entry = corpse_entries[selected_corpse_idx]
                message = _loot_item_from_corpse(corpse_ent, player_ent, entry)
            else:
                message = "Loot is one-way: you can only take from the corpse."

            if not _entity_has_tradeable_items(corpse_ent):
                return "close"

            if not esper.entity_exists(corpse_ent):
                continue


async def _draw_info_screen(
    renderer: Renderer,
    title: str,
    lines: list[str],
    subtitle: str = "",
) -> str:
    """A read-only panel of text lines; any menu/confirm key closes it. Used for
    the player's status screen and for examining NPCs/mobs."""
    while True:
        x, y, width, _height = _draw_menu_shell(
            renderer,
            title=title,
            subtitle=subtitle,
            footer="[Esc] close",
            width=64,
            height=22,
            overlay_game=True,
        )
        for idx, line in enumerate(lines):
            _draw_ui_text(renderer, x + 3, y + 5 + idx, line, _MENU_TEXT_COLOR, width - 6)
        renderer.present()

        action = await _await_action(renderer)
        if action == "quit":
            return "quit"
        if action in {"open_pause_menu", "open_inventory", "open_status", "menu_select", "confirm_action"}:
            return "close"


# How far the "look" cursor's interactions reach. Trade and melee need the two
# characters to be next to each other (Chebyshev distance 1); talking carries a
# few tiles; viewing someone's status can be done from anywhere on screen.
_LOOK_TALK_RANGE = 5
_LOOK_TRADE_RANGE = 1
_LOOK_MELEE_RANGE = 1


def _chebyshev_from_player(player_pos: Position, x: int, y: int) -> int:
    return max(abs(x - player_pos.x), abs(y - player_pos.y))


def _creature_at_xy(x: int, y: int) -> int | None:
    """The character (NPC or the player) standing on world cell (x, y)."""
    for ent, (pos, _npc) in esper.get_components(Position, NPC):
        if (pos.x, pos.y) == (x, y):
            return ent
    for ent, (pos, _player) in esper.get_components(Position, Player):
        if (pos.x, pos.y) == (x, y):
            return ent
    return None


def _renderable_at_xy(x: int, y: int, skip: int | None = None) -> int | None:
    """Any drawable entity on (x, y) other than ``skip`` (a tree, corpse, item,
    well, ...). Used to name whatever the look cursor rests on."""
    for ent, (pos, _rend) in esper.get_components(Position, Renderable):
        if ent == skip:
            continue
        if (pos.x, pos.y) == (x, y):
            return ent
    return None


def _terrain_name(game_map: GameMap, x: int, y: int) -> str:
    tile = game_map.tile_at(x, y)
    return {
        game_map.WALL: "a wall",
        game_map.WATER: "water",
        game_map.DOOR: "a door",
        game_map.WINDOW: "a window",
        game_map.FLOOR: "open ground",
    }.get(tile, "the ground")


def _look_available_actions(target_ent: int, dist: int) -> list[str]:
    """The interaction verbs the look cursor offers for ``target_ent`` at the
    given distance. Status is always offered; talking reaches a few tiles; trade
    and melee only work when standing right next to the target."""
    if esper.has_component(target_ent, Player):
        return ["Status"]

    options: list[str] = []
    if dist <= _LOOK_TALK_RANGE and esper.has_component(target_ent, Dialogue):
        options.append("Talk")
    if dist <= _LOOK_TRADE_RANGE and esper.has_component(target_ent, Friendly):
        options.append("Trade")
    if dist <= _LOOK_MELEE_RANGE and (
        esper.has_component(target_ent, Enemy) or esper.has_component(target_ent, Deer)
    ):
        options.append("Attack")
    options.append("Status")
    return options


def _look_info_line(
    game_map: GameMap,
    player_pos: Position,
    x: int,
    y: int,
    visible: bool,
    creature_ent: int | None,
) -> str:
    footer = "[dir] move   [Enter] interact   [Esc/L] done"
    if not visible:
        return f"Look: you can't make out that spot.   {footer}"

    dist = _chebyshev_from_player(player_pos, x, y)
    if creature_ent is not None:
        name = _entity_name(creature_ent, fallback="someone")
        if esper.has_component(creature_ent, Player):
            return f"Look: {name} (you).   [Enter] status   [Esc/L] done"
        verbs = ", ".join(_look_available_actions(creature_ent, dist)).lower()
        return f"Look: {name} - {dist} away.   [Enter] {verbs}   [Esc/L] done"

    feature_ent = _renderable_at_xy(x, y)
    if feature_ent is not None:
        name = _entity_name(feature_ent, fallback="something")
        return f"Look: {name} - {dist} away.   [Enter] examine   [Esc/L] done"

    return f"Look: {_terrain_name(game_map, x, y)}.   {footer}"


def _perform_look_attack(target_ent: int) -> None:
    """Strike an adjacent huntable creature from look mode, reusing the movement
    processor's melee/death sfx so it feels like a normal attack, then let a turn
    pass so the world reacts."""
    name = _entity_name(target_ent, fallback="the creature")
    movement = esper.get_processor(MovementProcessor)
    queue_message(f"You attack {name}.")
    if movement is not None and movement.on_melee_attack is not None:
        movement.on_melee_attack()
    slay_entity(target_ent)
    if movement is not None and movement.on_enemy_death is not None:
        movement.on_enemy_death()
    esper.process(WAIT_ACTION)


async def _draw_look_action_menu(
    renderer: Renderer,
    game_map: GameMap,
    target_ent: int,
    options: list[str],
) -> str:
    """A compact action menu for the creature under the look cursor, offering only
    the verbs its distance allows. Mirrors the dialogue menu's controls."""
    selected = 0
    talk_line = ""
    while True:
        if not esper.entity_exists(target_ent):
            return "close"
        name = _entity_name(target_ent, fallback="Creature")
        menu_options = options + ["Leave"]
        footer = talk_line if talk_line else "[W/S] move   [Enter] select   [Esc] close"
        x, y, width, _height = _draw_menu_shell(
            renderer,
            title=f"LOOK - {name}",
            subtitle="Interaction",
            footer=footer,
            width=72,
            height=20,
            overlay_game=True,
        )
        info_lines = _creature_status_lines(game_map, target_ent)
        for idx, line in enumerate(info_lines):
            _draw_ui_text(renderer, x + 3, y + 5 + idx, line, _MENU_TEXT_COLOR, width - 6)
        option_y = y + 5 + len(info_lines) + 2
        _draw_menu_options(renderer, x + 3, option_y, width - 6, menu_options, selected)
        renderer.present()

        action = await _await_action(renderer)
        if action == "quit":
            return "quit"
        if action in {"open_pause_menu", "look"}:
            return "close"
        if action == "move_up":
            selected = (selected - 1) % len(menu_options)
            continue
        if action == "move_down":
            selected = (selected + 1) % len(menu_options)
            continue
        if action in {"menu_select", "confirm_action"}:
            choice = menu_options[selected]
            if choice == "Leave":
                return "close"
            if choice == "Talk":
                if esper.has_component(target_ent, Dialogue):
                    line = esper.component_for_entity(target_ent, Dialogue).line
                    talk_line = f"{name}: \"{line}\""
                else:
                    talk_line = f"{name} has nothing to say."
                continue
            if choice == "Trade":
                if await _draw_trade_menu(renderer, target_ent) == "quit":
                    return "quit"
                talk_line = ""
                continue
            if choice == "Status":
                if await _draw_info_screen(
                    renderer,
                    title=f"STATUS - {name}",
                    lines=_creature_status_lines(game_map, target_ent),
                    subtitle="What you can tell about them",
                ) == "quit":
                    return "quit"
                talk_line = ""
                continue
            if choice == "Attack":
                _perform_look_attack(target_ent)
                return "close"


async def _look_interact(renderer: Renderer, game_map: GameMap, x: int, y: int) -> str:
    """Resolve an Enter press on the look cursor: interact with a creature there,
    loot an adjacent corpse, or just describe whatever the cursor rests on."""
    player_ent = _first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return "close"
    player_pos = esper.component_for_entity(player_ent, Position)
    dist = _chebyshev_from_player(player_pos, x, y)

    creature_ent = _creature_at_xy(x, y)
    if creature_ent is not None:
        if creature_ent == player_ent:
            return await _draw_info_screen(
                renderer,
                title="STATUS - You",
                lines=_creature_status_lines(game_map, player_ent),
                subtitle="How you're holding up",
            )
        options = _look_available_actions(creature_ent, dist)
        return await _draw_look_action_menu(renderer, game_map, creature_ent, options)

    feature_ent = _renderable_at_xy(x, y)
    if feature_ent is not None:
        if (
            dist <= 1
            and esper.has_component(feature_ent, Corpse)
            and _entity_has_tradeable_items(feature_ent)
        ):
            return await _draw_loot_menu(renderer, feature_ent)
        name = _entity_name(feature_ent, fallback="something")
        return await _draw_info_screen(
            renderer,
            title="LOOK",
            lines=[f"You see {name}.", f"Distance: {dist}."],
            subtitle="",
        )

    return await _draw_info_screen(
        renderer,
        title="LOOK",
        lines=[f"You see {_terrain_name(game_map, x, y)}."],
        subtitle="",
    )


def _clear_look_overlay(render: RenderProcessor | None) -> None:
    if render is not None:
        render.look_cursor = None
        render.look_info = None


async def _look_mode(renderer: Renderer, game_map: GameMap) -> str:
    """Drive the look cursor: it starts on the player, moves with the direction
    keys, and Enter interacts with whatever it rests on. Returns "quit" if the
    player quit the game while looking, otherwise "close"."""
    player_ent = _first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return "close"
    player_pos = esper.component_for_entity(player_ent, Position)
    cursor_x, cursor_y = player_pos.x, player_pos.y

    render = esper.get_processor(RenderProcessor)

    while True:
        # Keep the cursor inside the drawn viewport so it's always visible; on the
        # first pass (before any look render) fall back to the whole map.
        if render is not None:
            ox, oy, vw, vh = render.view_bounds()
            cursor_x = min(max(cursor_x, ox), ox + vw - 1)
            cursor_y = min(max(cursor_y, oy), oy + vh - 1)
        else:
            cursor_x = min(max(cursor_x, 0), game_map.width - 1)
            cursor_y = min(max(cursor_y, 0), game_map.height - 1)

        on_player = (cursor_x, cursor_y) == (player_pos.x, player_pos.y)
        visible = on_player or render is None or render.tile_is_visible(cursor_x, cursor_y)
        creature_ent = _creature_at_xy(cursor_x, cursor_y) if visible else None

        if render is not None:
            render.look_cursor = (cursor_x, cursor_y)
            render.look_info = _look_info_line(
                game_map, player_pos, cursor_x, cursor_y, visible, creature_ent
            )
        esper.process(None)

        action = await _await_action(renderer)
        if action == "quit":
            _clear_look_overlay(render)
            return "quit"
        if action in {"open_pause_menu", "look"}:
            break
        if action in _CARDINAL_ACTION_DELTAS:
            dx, dy = _CARDINAL_ACTION_DELTAS[action]
            cursor_x += dx
            cursor_y += dy
            continue
        if action in {"menu_select", "confirm_action"}:
            if not visible:
                continue
            # The interaction menu draws its own frames, so drop the cursor
            # overlay while it's open, then restore it on return.
            _clear_look_overlay(render)
            result = await _look_interact(renderer, game_map, cursor_x, cursor_y)
            if result == "quit":
                _clear_look_overlay(render)
                return "quit"
            continue

    _clear_look_overlay(render)
    esper.process(None)
    return "close"


async def _draw_dialogue_menu(renderer: Renderer, game_map: GameMap, npc_ent: int) -> str:
    selected = 0
    options = ["Talk", "Trade", "Status", "Leave"]
    info_lines = _npc_info_lines(game_map, npc_ent)
    talk_line = ""

    while True:
        npc_name = _entity_name(npc_ent, fallback="Unknown NPC")

        footer = talk_line if talk_line else "[W/S] move   [Enter/Space] select   [Esc] close"
        x, y, width, _height = _draw_menu_shell(
            renderer,
            title=f"DIALOGUE - {npc_name}",
            subtitle="Interaction menu",
            footer=footer,
            width=82,
            height=24,
            overlay_game=True,
        )

        info_y = y + 5
        for idx, line in enumerate(info_lines):
            _draw_ui_text(renderer, x + 3, info_y + idx, line, _MENU_TEXT_COLOR, width - 6)

        option_y = info_y + len(info_lines) + 2
        _draw_ui_text(renderer, x + 3, option_y - 1, "OPTIONS", _MENU_MUTED_COLOR, width - 6)
        _draw_menu_options(renderer, x + 3, option_y, width - 6, options, selected)

        renderer.present()
        action = await _await_action(renderer)

        if action in {"open_pause_menu", "open_inventory"}:
            return "close"
        if action == "quit":
            return "quit"
        if action == "move_up":
            selected = (selected - 1) % len(options)
            continue
        if action == "move_down":
            selected = (selected + 1) % len(options)
            continue

        if action in {"menu_select", "confirm_action"}:
            choice = options[selected]
            if choice == "Leave":
                return "close"
            if choice == "Talk":
                if esper.has_component(npc_ent, Dialogue):
                    line = esper.component_for_entity(npc_ent, Dialogue).line
                    talk_line = f"{npc_name}: \"{line}\""
                else:
                    talk_line = f"{npc_name} has nothing to say."
                # Chatting with a sentient villager builds friendship (Sims-like);
                # show the outcome and refresh the info block's Friendship line.
                player_ent = _first_player_entity()
                if player_ent is not None and esper.has_component(npc_ent, Personality):
                    outcome = _player_talk(player_ent, npc_ent)
                    talk_line = f"{talk_line}  [{outcome}]"
                    info_lines = _npc_info_lines(game_map, npc_ent)
                continue
            if choice == "Trade":
                trade_choice = await _draw_trade_menu(renderer, npc_ent)
                if trade_choice == "quit":
                    return "quit"
                info_lines = _npc_info_lines(game_map, npc_ent)
                talk_line = ""
            if choice == "Status":
                status_choice = await _draw_info_screen(
                    renderer,
                    title=f"STATUS - {npc_name}",
                    lines=_creature_status_lines(game_map, npc_ent),
                    subtitle="What you can tell about them",
                )
                if status_choice == "quit":
                    return "quit"
                info_lines = _npc_info_lines(game_map, npc_ent)
                talk_line = ""


@dataclass
class _InventoryState:
    """Cursor/message state for the Inventory tab, held across frames by the
    player menu so the tab can be re-entered without losing the selection."""
    selected_panel: str = "left"
    selected_slot_idx: int = 0
    selected_item_idx: int = 0
    message: str = "Enter: equip/use   A/D: switch side   W/S: move"


def _render_inventory_body(
    renderer: Renderer,
    x: int,
    content_y: int,
    width: int,
    content_h: int,
    state: _InventoryState,
) -> None:
    """Draw the Inventory tab body (a message line + EQUIPPED and ITEMS panels)
    inside the given content region."""
    player_ent = _first_player_entity()
    inventory_items: list[str] = []
    equipment_slots = _default_equipment_slots()
    if player_ent is not None:
        if esper.has_component(player_ent, Inventory):
            inventory_items = list(esper.component_for_entity(player_ent, Inventory).items)
        if esper.has_component(player_ent, Equipment):
            configured = esper.component_for_entity(player_ent, Equipment).slots
            for slot_name in equipment_slots:
                equipment_slots[slot_name] = configured.get(slot_name)

    _draw_ui_text(renderer, x + 3, content_y, state.message, _MENU_MUTED_COLOR, width - 6)
    panels_y = content_y + 1
    panels_h = max(6, content_h - 1)

    left_w = max(20, (width - 6) // 2)
    right_w = max(20, width - left_w - 6)
    left_x = x + 2
    right_x = left_x + left_w + 2

    draw_panel = getattr(renderer, "draw_panel", None)
    draw_ui_glyph = getattr(renderer, "draw_ui_glyph", None)
    icon_cells = 2 if callable(draw_ui_glyph) else 0
    text_offset = icon_cells + 1 if icon_cells > 0 else 0
    if callable(draw_panel):
        draw_panel(left_x, panels_y, left_w, panels_h, title="EQUIPPED")
        draw_panel(right_x, panels_y, right_w, panels_h, title="ITEMS")

    slot_names = list(equipment_slots.keys())
    list_start_y = panels_y + 3
    visible_rows = max(1, panels_h - 5)

    if slot_names:
        slot_start = _scroll_start(state.selected_slot_idx, len(slot_names), visible_rows)
        slot_end = min(len(slot_names), slot_start + visible_rows)
        for row_offset, slot_idx in enumerate(range(slot_start, slot_end)):
            slot_name = slot_names[slot_idx]
            equipped_item = equipment_slots[slot_name]
            display = equipped_item if equipped_item else "(empty)"
            row_y = list_start_y + row_offset
            is_selected = state.selected_panel == "left" and slot_idx == state.selected_slot_idx
            if is_selected:
                _fill_ui_cells(renderer, left_x + 1, row_y, left_w - 2, 1, _MENU_SELECTED_BG)

            prefix = ">" if is_selected else " "
            color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
            text_x = left_x + 2 + text_offset
            text_width = max(1, left_w - 4 - text_offset)
            glyph_prefix = ""

            if equipped_item:
                glyph, classification, fg, bg = _item_visual(equipped_item)
                icon_drawn = False
                if callable(draw_ui_glyph):
                    icon_drawn = bool(
                        draw_ui_glyph(left_x + 2, row_y, glyph, classification=classification, fg=fg, bg=bg, cell_span=icon_cells)
                    )
                if not icon_drawn:
                    glyph_prefix = f"{glyph} "

            _draw_ui_text(renderer, text_x, row_y, f"{prefix} {slot_name:10}: {glyph_prefix}{display}", color, text_width)

    if inventory_items:
        item_start = _scroll_start(state.selected_item_idx, len(inventory_items), visible_rows)
        item_end = min(len(inventory_items), item_start + visible_rows)
        for row_offset, item_idx in enumerate(range(item_start, item_end)):
            item_name = inventory_items[item_idx]
            row_y = list_start_y + row_offset
            label = chr(ord("a") + item_idx) if item_idx < 26 else "*"
            is_selected = state.selected_panel == "right" and item_idx == state.selected_item_idx
            if is_selected:
                _fill_ui_cells(renderer, right_x + 1, row_y, right_w - 2, 1, _MENU_SELECTED_BG)

            prefix = ">" if is_selected else " "
            color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
            text_x = right_x + 2 + text_offset
            text_width = max(1, right_w - 4 - text_offset)
            glyph, classification, fg, bg = _item_visual(item_name)
            icon_drawn = False
            if callable(draw_ui_glyph):
                icon_drawn = bool(
                    draw_ui_glyph(right_x + 2, row_y, glyph, classification=classification, fg=fg, bg=bg, cell_span=icon_cells)
                )

            label_text = f"{prefix} {label}) {item_name}"
            if not icon_drawn:
                label_text = f"{prefix} {label}) {glyph} {item_name}"
            _draw_ui_text(renderer, text_x, row_y, label_text, color, text_width)
    else:
        is_selected = state.selected_panel == "right"
        if is_selected:
            _fill_ui_cells(renderer, right_x + 1, list_start_y, right_w - 2, 1, _MENU_SELECTED_BG)
        color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
        prefix = ">" if is_selected else " "
        _draw_ui_text(renderer, right_x + 2 + text_offset, list_start_y, f"{prefix} (empty)", color, max(1, right_w - 4 - text_offset))


def _handle_inventory_input(action: str, state: _InventoryState) -> str | None:
    """Apply one input action to the Inventory tab (navigation + equip/use). Tab
    switching and closing are handled by the enclosing player menu. Returns the
    name of a buildable item when the player selects one to place (the menu then
    hands off to placement); otherwise ``None``."""
    player_ent = _first_player_entity()

    slot_count = 0
    item_count = 0
    if player_ent is not None:
        if esper.has_component(player_ent, Equipment):
            slot_count = len(_default_equipment_slots())
        if esper.has_component(player_ent, Inventory):
            item_count = len(esper.component_for_entity(player_ent, Inventory).items)

    if action == "move_left":
        state.selected_panel = "left"
        return
    if action == "move_right":
        state.selected_panel = "right"
        return
    if action == "move_up":
        if state.selected_panel == "left" and slot_count:
            state.selected_slot_idx = (state.selected_slot_idx - 1) % slot_count
        elif state.selected_panel == "right" and item_count:
            state.selected_item_idx = (state.selected_item_idx - 1) % item_count
        return
    if action == "move_down":
        if state.selected_panel == "left" and slot_count:
            state.selected_slot_idx = (state.selected_slot_idx + 1) % slot_count
        elif state.selected_panel == "right" and item_count:
            state.selected_item_idx = (state.selected_item_idx + 1) % item_count
        return

    if action in {"menu_select", "confirm_action"} and player_ent is not None:
        if not esper.has_component(player_ent, Inventory) or not esper.has_component(player_ent, Equipment):
            return None
        inventory = esper.component_for_entity(player_ent, Inventory)
        equipment = esper.component_for_entity(player_ent, Equipment)

        if state.selected_panel == "right":
            if not inventory.items:
                state.message = "No items to use."
            else:
                clamped_idx = max(0, min(state.selected_item_idx, len(inventory.items) - 1))
                item_name = inventory.items[clamped_idx]
                # Buildable pieces hand off to placement; food/drink is consumed;
                # everything else equips.
                if is_placeable(item_name):
                    return item_name
                consume_message = _apply_consumable(player_ent, clamped_idx)
                if consume_message is not None:
                    state.message = consume_message
                else:
                    state.message = _equip_inventory_item(inventory, equipment, clamped_idx)
                state.selected_item_idx = min(state.selected_item_idx, len(inventory.items) - 1) if inventory.items else 0
        else:
            slot_names = list(_default_equipment_slots().keys())
            if slot_names:
                slot_name = slot_names[max(0, min(state.selected_slot_idx, len(slot_names) - 1))]
                state.message = _unequip_slot(inventory, equipment, slot_name)

    return None


@dataclass
class _CraftingState:
    """Cursor/message state for the Crafting tab, held across frames like the
    Inventory tab so a re-entered tab keeps its selection."""
    selected_idx: int = 0
    message: str = "Enter: craft   W/S: move"


def _render_crafting_body(
    renderer: Renderer,
    x: int,
    content_y: int,
    width: int,
    content_h: int,
    state: _CraftingState,
) -> None:
    """Draw the Crafting tab: your Wood on hand and the buildable recipes."""
    player_ent = _first_player_entity()
    wood_count = 0
    if player_ent is not None and esper.has_component(player_ent, Inventory):
        wood_count = esper.component_for_entity(player_ent, Inventory).items.count(WOOD)

    _draw_ui_text(renderer, x + 3, content_y, state.message, _MENU_MUTED_COLOR, width - 6)
    _draw_ui_text(renderer, x + 3, content_y + 1, f"Wood in pack: {wood_count}", _MENU_TEXT_COLOR, width - 6)

    list_y = content_y + 3
    for idx, item_name in enumerate(_CRAFT_MENU):
        cost = craft_cost(item_name) or 0
        affordable = wood_count >= cost
        is_selected = idx == state.selected_idx
        row_y = list_y + idx
        if is_selected:
            _fill_ui_cells(renderer, x + 2, row_y, width - 4, 1, _MENU_SELECTED_BG)
        prefix = ">" if is_selected else " "
        if is_selected:
            color = _MENU_SELECTED_TEXT
        elif affordable:
            color = _MENU_TEXT_COLOR
        else:
            color = _MENU_MUTED_COLOR
        _draw_ui_text(renderer, x + 3, row_y, f"{prefix} {item_name:8} - {cost} Wood", color, width - 6)

    hint_y = list_y + len(_CRAFT_MENU) + 1
    _draw_ui_text(
        renderer,
        x + 3,
        hint_y,
        "Craft pieces here, then place them from the Inventory tab.",
        _MENU_MUTED_COLOR,
        width - 6,
    )


def _handle_crafting_input(action: str, state: _CraftingState) -> None:
    """Navigate the recipe list and craft the selected piece."""
    if action == "move_up":
        state.selected_idx = (state.selected_idx - 1) % len(_CRAFT_MENU)
        return
    if action == "move_down":
        state.selected_idx = (state.selected_idx + 1) % len(_CRAFT_MENU)
        return
    if action in {"menu_select", "confirm_action"}:
        player_ent = _first_player_entity()
        if player_ent is not None:
            state.message = _craft_item(player_ent, _CRAFT_MENU[state.selected_idx])


# The player menu tabs. Inventory, Craft, and Status are live; the rest are
# placeholders for planned screens so the tab framework is already in place.
_PLAYER_MENU_TABS: list[tuple[str, str]] = [
    ("inventory", "Inventory"),
    ("craft", "Craft"),
    ("status", "Status"),
    ("map", "Map"),
    ("journal", "Journal"),
    ("skills", "Skills"),
]
_PLAYER_MENU_KEYS = [key for key, _label in _PLAYER_MENU_TABS]


def _draw_tab_bar(renderer: Renderer, x: int, y: int, width: int, active_index: int) -> None:
    cursor = x
    for idx, (_key, label) in enumerate(_PLAYER_MENU_TABS):
        text = f" {label} "
        if idx == active_index:
            _fill_ui_cells(renderer, cursor, y, len(text), 1, _MENU_SELECTED_BG)
        color = _MENU_SELECTED_TEXT if idx == active_index else _MENU_MUTED_COLOR
        _draw_ui_text(renderer, cursor, y, text, color, max(1, x + width - cursor))
        cursor += len(text) + 1


async def _draw_player_menu(renderer: Renderer, game_map: GameMap, start_tab: str = "inventory") -> tuple[str, str]:
    """Tabbed player menu (Tab cycles tabs; I/C jump to Inventory/Status and
    toggle-close if already there; Esc closes). Returns (result, last_tab) so the
    caller can reopen on the tab you left on."""
    tab = _PLAYER_MENU_KEYS.index(start_tab) if start_tab in _PLAYER_MENU_KEYS else 0
    inv_state = _InventoryState()
    craft_state = _CraftingState()

    while True:
        current_key = _PLAYER_MENU_KEYS[tab]
        x, y, width, height = _draw_menu_shell(
            renderer,
            title="MENU",
            subtitle=None,
            footer="[Tab] switch tab   [I] items   [C] status   [Esc] close",
            width=100,
            height=34,
            overlay_game=True,
        )
        _draw_tab_bar(renderer, x + 2, y + 2, width - 4, tab)
        content_y = y + 4
        content_h = max(8, height - 7)

        if current_key == "inventory":
            _render_inventory_body(renderer, x, content_y, width, content_h, inv_state)
        elif current_key == "craft":
            _render_crafting_body(renderer, x, content_y, width, content_h, craft_state)
        elif current_key == "status":
            player_ent = _first_player_entity()
            lines = _creature_status_lines(game_map, player_ent) if player_ent is not None else ["No player."]
            for idx, line in enumerate(lines):
                _draw_ui_text(renderer, x + 3, content_y + 1 + idx, line, _MENU_TEXT_COLOR, width - 6)
        else:
            label = _PLAYER_MENU_TABS[tab][1]
            _draw_ui_text(renderer, x + 3, content_y + 1, f"{label} - coming soon.", _MENU_MUTED_COLOR, width - 6)

        renderer.present()

        action = await _await_action(renderer)
        if action == "quit":
            return "quit", current_key
        if action == "open_pause_menu":
            return "close", current_key
        if action == "open_menu":  # Tab cycles to the next tab
            tab = (tab + 1) % len(_PLAYER_MENU_KEYS)
            continue
        if action == "open_inventory":
            if current_key == "inventory":
                return "close", current_key
            tab = _PLAYER_MENU_KEYS.index("inventory")
            continue
        if action == "open_status":
            if current_key == "status":
                return "close", current_key
            tab = _PLAYER_MENU_KEYS.index("status")
            continue

        if current_key == "inventory":
            place_item = _handle_inventory_input(action, inv_state)
            if place_item is not None:
                # Hand off to placement; the caller drives the direction prompt.
                return f"place:{place_item}", current_key
        elif current_key == "craft":
            _handle_crafting_input(action, craft_state)


def _spawn_cave_rat(
    x: int,
    y: int,
    *,
    include_loot: bool,
) -> None:
    components: list[object] = [
        Position(x, y),
        Renderable("r", fg=_RAT_BROWN),
        Name("Cave Rat"),
        NPC(),
        Enemy(),
        Vision(6),
        BlocksMovement(),
        Meat("Rat Meat"),
        Diet("carnivore"),
        Needs(),
    ]
    if include_loot:
        components.extend(
            [
                Inventory(items=["String", "Pebble"]),
                Equipment(slots=_default_equipment_slots()),
            ]
        )

    esper.create_entity(*components)


def _setup_world(game_map: GameMap, player_position: Position, rat_flood: bool = False) -> int:
    # Singleton world clock driving the day/night cycle and the night-time
    # tiredness ramp. Created before any creature so the first turn has a time.
    # The game opens mid-morning so the player starts a fresh day in daylight.
    start_clock = WorldClock()
    start_clock.turn = int(start_clock.day_length * 0.2)
    esper.create_entity(start_clock)

    # The onymancer names every villager procedurally. A fixed seed keeps the
    # starting village identical run to run, like the rest of the world layout.
    onymancer = make_onymancer(0x0A_15_15)

    player_name = "You"
    player_skin = _human_skin_tone(player_name)

    player_equipment = _default_equipment_slots()
    player_equipment["main hand"] = "Rusty Sword"
    player_equipment["chest"] = "Traveler Tunic"
    esper.create_entity(
        player_position,
        Renderable("@", fg=player_skin),
        Name(player_name),
        # Placeholder until character creation lets the player choose; the future
        # reproduction system reads this.
        Gender("male"),
        Player(),
        Vision(10),
        BlocksMovement(),
        # Some starting wood so the player can craft walls/doors/windows right
        # away (Crafting tab in the Tab menu, then place from the inventory).
        Inventory(items=["Bandage", "Torch", "Apple", WOOD, WOOD, WOOD, WOOD, WOOD]),
        Equipment(slots=player_equipment),
        Needs(),
    )

    if rat_flood:
        rats_spawned = 0
        player_xy = (player_position.x, player_position.y)
        for y in range(game_map.height):
            for x in range(game_map.width):
                if not game_map.is_walkable(x, y):
                    continue
                if (x, y) == player_xy:
                    continue
                _spawn_cave_rat(x, y, include_loot=False)
                rats_spawned += 1
        return rats_spawned

    guard_pos = Position(max(2, player_position.x - 5), player_position.y)
    rat_pos = Position(min(game_map.width - 3, player_position.x + 6), max(2, player_position.y - 2))

    def spawn_villager(pos: Position, gender: str, traits: list[str], surname: str) -> int:
        # The onymancer coins the given name (flavoured by gender) and joins it to
        # the family surname. Skin tone still seeds off the final name so it stays
        # stable across runs.
        _given, _surname, full = onymancer.full_name(gender, surname)
        return esper.create_entity(
            pos,
            Renderable("v", fg=_human_skin_tone(full, offset=7)),
            Name(full),
            Gender(gender),
            # Links (spouse/parents/children) are wired reciprocally after the
            # whole household is spawned.
            Family(surname=surname),
            NPC(),
            Friendly(),
            Dialogue("##!/$*~# GH01^@"),
            BlocksMovement(),
            Inventory(items=["Bread", "Waterskin"]),
            Equipment(slots=_default_equipment_slots()),
            # Villagers cook: they gather meat + wood and cook at a stove before
            # eating (unlike predator monsters, which eat raw on the spot).
            Diet("cook"),
            Needs(),
            # Residents seek out an unowned house to live in (claiming it as their
            # home), and build a cabin of their own if none is free.
            Resident(),
            # A named personality drives who they befriend and how they socialise.
            Personality(traits=list(traits)),
            Relationships(),
        )

    def _place(offset: tuple[int, int]) -> Position:
        dx, dy = offset
        vx = min(game_map.width - 3, max(2, player_position.x + dx))
        vy = min(game_map.height - 3, max(2, player_position.y + dy))
        return Position(vx, vy)

    # The starting cast forms two households (a full family of four and a couple),
    # so relationships include spouses, parents/children, and siblings from the
    # outset. Traits are the original contrasting personalities -- so the social AI
    # still produces both warming (++) and souring (--) interactions -- while names,
    # genders, and family ties are new. Each entry is (gender, traits, offset);
    # placement stays a deterministic loose cluster near the player.
    households: list[tuple[
        tuple[str, list[str], tuple[int, int]],       # father
        tuple[str, list[str], tuple[int, int]],       # mother
        list[tuple[str, list[str], tuple[int, int]]],  # children
    ]] = [
        (
            ("male", ["Cheerful", "Outgoing"], (-2, 1)),
            ("female", ["Kind", "Shy"], (-3, 2)),
            [
                ("male", ["Grumpy"], (-4, 1)),
                ("female", ["Playful", "Kind"], (-2, 3)),
            ],
        ),
        (
            ("male", ["Aloof"], (-5, 2)),
            ("female", ["Outgoing", "Playful"], (-3, 3)),
            [],
        ),
    ]

    for father_spec, mother_spec, child_specs in households:
        surname = onymancer.surname()
        fg, ftraits, foff = father_spec
        mg, mtraits, moff = mother_spec
        father = spawn_villager(_place(foff), fg, ftraits, surname)
        mother = spawn_villager(_place(moff), mg, mtraits, surname)
        father_fam = esper.component_for_entity(father, Family)
        mother_fam = esper.component_for_entity(mother, Family)
        father_fam.spouse = mother
        mother_fam.spouse = father
        for cg, ctraits, coff in child_specs:
            child = spawn_villager(_place(coff), cg, ctraits, surname)
            child_fam = esper.component_for_entity(child, Family)
            child_fam.parents = [father, mother]
            father_fam.children.append(child)
            mother_fam.children.append(child)

    goblin_equipment = _default_equipment_slots()
    goblin_equipment["main hand"] = "Jagged Dagger"
    esper.create_entity(
        guard_pos,
        Renderable("g", fg=_GOBLIN_GREEN),
        Name("Goblin Scout"),
        NPC(),
        Enemy(),
        Vision(8),
        BlocksMovement(),
        Inventory(items=["Copper Coin", "Bone Charm"]),
        Equipment(slots=goblin_equipment),
        Meat("Goblin Meat"),
        Diet("carnivore"),
        Needs(),
    )
    esper.create_entity(
        rat_pos,
        Renderable("r", fg=_RAT_BROWN),
        Name("Cave Rat"),
        NPC(),
        Enemy(),
        Vision(6),
        BlocksMovement(),
        Inventory(items=["String", "Pebble"]),
        Equipment(slots=_default_equipment_slots()),
        Meat("Rat Meat"),
        Diet("carnivore"),
        Needs(),
    )

    _spawn_environment_features(game_map, player_position)
    if getattr(game_map, "has_ocean", False):
        _spawn_ocean_life(game_map)

    # Furnish the pre-built houses (bed, oven, chest, table, wardrobe,
    # bookshelf). Residents claim these unowned houses as homes at runtime.
    for interior in game_map.find_enclosed_rooms():
        furnish_house(game_map, interior)

    return 1


def _spawn_ocean_life(game_map: GameMap) -> None:
    """Scatter seaweed and fish across the open sea that surrounds the island.
    Fish graze the seaweed (see ``FishAiProcessor``). Placement uses a fixed seed
    so the ocean looks the same every run, matching the RNG-free land layout."""
    rng = random.Random(0x0C_EA_11)
    occupied = {(pos.x, pos.y) for _ent, (pos,) in esper.get_components(Position)}

    for y in range(1, game_map.height - 1):
        for x in range(1, game_map.width - 1):
            if not game_map.is_ocean(x, y) or (x, y) in occupied:
                continue
            roll = rng.random()
            if roll < 0.028:
                esper.create_entity(
                    Position(x, y),
                    Renderable('"', fg=_SEAWEED_GREEN, bg=_OCEAN_WATER_BG),
                    Name("Seaweed"),
                    Seaweed(),
                )
                occupied.add((x, y))
            elif roll < 0.028 + 0.001:
                esper.create_entity(
                    Position(x, y),
                    Renderable("f", fg=_FISH_SILVER, bg=_OCEAN_WATER_BG),
                    Name("Fish"),
                    Fish(),
                    # Fish never go thirsty (they live in water); their only drive
                    # is hunger, which sends them grazing seaweed.
                    Needs(hunger=25.0, thirst=0.0, thirst_rate=0.0, tiredness_rate=0.0),
                )
                occupied.add((x, y))


def _spawn_deer(game_map: GameMap, x: int, y: int) -> None:
    """Create a wild deer: prey that grazes trees and drinks water, and yields
    Deer Meat when hunted."""
    if not game_map.is_walkable(x, y):
        return
    esper.create_entity(
        Position(x, y),
        Renderable("d", fg=_DEER_TAN),
        Name("Deer"),
        NPC(),
        Deer(),
        Diet("herbivore"),
        Vision(8),
        BlocksMovement(),
        Meat("Deer Meat"),
        Needs(hunger=30.0, thirst=30.0),
    )


def _spawn_environment_features(game_map: GameMap, player_position: Position) -> None:
    """Populate the world: a well and stove by the player for the survival loop,
    tree stands and roaming deer scattered across the map, with the wildlife
    biased toward the water so grazing/drinking is nearby."""
    # Safety: never leave the player standing in (or walled by) a lake/river.
    game_map.clear_water_around(player_position.x, player_position.y, radius=2)

    occupied = {
        (pos.x, pos.y)
        for _ent, (pos, _blocks) in esper.get_components(Position, BlocksMovement)
    }

    def place_at(x: int, y: int, *components: object) -> bool:
        if not game_map.is_walkable(x, y):
            return False
        if (x, y) in occupied or (x, y) == (player_position.x, player_position.y):
            return False
        occupied.add((x, y))
        esper.create_entity(Position(x, y), *components)
        return True

    def place(dx: int, dy: int, *components: object) -> bool:
        return place_at(player_position.x + dx, player_position.y + dy, *components)

    place(2, 2, Renderable("O", fg=_WELL_STONE), Name("Stone Well"), Well(), BlocksMovement())
    place(4, 2, Renderable("#", fg=_STOVE_IRON), Name("Iron Stove"), Stove(), BlocksMovement())
    # The player's bed: sleep beside it to rest at home instead of camping. It
    # belongs to the player, so villagers never claim it (even if you wall it in).
    if place(3, 2, Renderable("=", fg=_BED_WOOD), Name("Bed"), Bed()):
        player_ent = _first_player_entity()
        bed_xy = (player_position.x + 3, player_position.y + 2)
        if player_ent is not None:
            for bed_ent, (bpos, _bed) in esper.get_components(Position, Bed):
                if (bpos.x, bpos.y) == bed_xy:
                    set_bed_owner(bed_ent, player_ent)
                    break

    def plant_tree(x: int, y: int) -> None:
        place_at(x, y, Renderable("T", fg=_TREE_GREEN), Name("Tree"), Tree(), BlocksMovement())

    def plant_bush(x: int, y: int) -> None:
        place_at(x, y, Renderable("%", fg=_BERRY_RED), Name("Berry Bush"), BerryBush(), BlocksMovement())

    # A few starter trees within reach, plus deterministic forest stands spread
    # across the map so both the player and grazing deer have wood/food. Twice
    # the trees of the old world -- ten per stand, ten starters.
    for tree_dx, tree_dy in (
        (-3, -2), (-4, -2), (-3, 3), (5, 3), (6, 3),
        (-5, -2), (-4, 3), (6, 4), (-5, 2), (5, 4),
    ):
        plant_tree(player_position.x + tree_dx, player_position.y + tree_dy)

    # A couple of ripe berry bushes within reach of the player to start.
    for bush_dx, bush_dy in ((-2, 3), (4, -2)):
        plant_bush(player_position.x + bush_dx, player_position.y + bush_dy)

    stand_centers = [
        (int(game_map.width * fx), int(game_map.height * fy))
        for fx, fy in ((0.15, 0.25), (0.4, 0.8), (0.7, 0.6), (0.85, 0.75), (0.55, 0.2))
    ]
    for cx, cy in stand_centers:
        for tx, ty in (
            (0, 0), (1, 0), (0, 1), (2, 1), (1, 2),
            (2, 2), (3, 0), (0, 3), (3, 2), (2, 3),
        ):
            plant_tree(cx + tx, cy + ty)
        # Each stand also carries a berry bush for foragers.
        plant_bush(cx + 1, cy + 1)

    # Deer near the tree stands / water so they can graze and drink.
    for cx, cy in stand_centers:
        for dxy in ((-2, 0), (3, 2)):
            _spawn_deer(game_map, cx + dxy[0], cy + dxy[1])


async def main() -> None:
    args = _parse_args()
    bootstrap_files(MAP_WIDTH, MAP_HEIGHT)
    options = load_options()
    if args.screenshot is not None:
        # Run screenshot capture off-screen.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        # Force windowed capture for deterministic screenshot dimensions.
        options["fullscreen"] = False

    pygame_module = None
    combat_sfx = _CombatSfxPlayer(None, options)

    game_map = GameMap(MAP_WIDTH, MAP_HEIGHT)
    player_position = Position(MAP_WIDTH // 2, MAP_HEIGHT // 2)
    startup_save_file = args.save_file
    if args.screenshot is not None and startup_save_file is None:
        startup_save_file = DEFAULT_SAVE_FILE

    try:
        with PygameRenderer(options=options) as renderer:
            startup_ok, game_map, player_position, selected_save_file = await _run_startup_flow(
                renderer,
                startup_save_file,
                game_map,
                player_position,
            )
            if not startup_ok:
                return

            if args.screenshot is None:
                pygame_module = _start_background_music(options)
                combat_sfx = _CombatSfxPlayer(pygame_module, options)

            rat_count = _setup_world(game_map, player_position, rat_flood=args.rat_flood)
            if args.rat_flood:
                print(f"Rat flood mode enabled: spawned {rat_count} cave rats.", file=sys.stderr)
            esper.add_processor(
                MovementProcessor(
                    game_map,
                    on_melee_attack=combat_sfx.play_melee_attack,
                    on_enemy_death=combat_sfx.play_death,
                ),
                priority=1,
            )
            # TimeProcessor runs first (priority above movement) so the clock is
            # current before needs/AI read the time of day this turn.
            esper.add_processor(TimeProcessor(), priority=2)
            # Housing runs before the AI so a villager that just claimed a home
            # can start heading there this turn.
            esper.add_processor(HousingProcessor(game_map), priority=0)
            esper.add_processor(NpcAiProcessor(game_map), priority=0)
            esper.add_processor(FishAiProcessor(game_map), priority=0)
            esper.add_processor(NeedsProcessor(), priority=0)
            esper.add_processor(TreeGrowthProcessor(game_map), priority=0)
            esper.add_processor(RenderProcessor(renderer, game_map), priority=0)

            esper.process()  # initial frame
            if args.screenshot is not None:
                _capture_frame_screenshot(renderer, args.screenshot)
                return

            held_directions: set[str] = set()
            direction_pressed_order = {
                "move_up": -1,
                "move_down": -1,
                "move_left": -1,
                "move_right": -1,
            }
            press_order_counter = 0
            # The player menu (Tab) reopens on whatever tab you left it on.
            last_menu_tab = "inventory"
            invalidate_backdrop = getattr(renderer, "invalidate_backdrop", None)
            while True:
                # Live play has (re)rendered the game, so any menu backdrop
                # snapshot is stale; the next menu to open will re-capture.
                if callable(invalidate_backdrop):
                    invalidate_backdrop()
                # When the player has an active status animation (swimming, on
                # fire, ...), poll on a short timeout and re-render on each idle
                # tick so the identifiers cycle without input; otherwise keep the
                # efficient blocking wait (desktop truly idles).
                if player_is_animated(game_map) or bubbles_active():
                    action = await _await_action_or_idle(renderer, _STATUS_ANIM_POLL_SECONDS)
                    if action is None:
                        esper.process(None)
                        continue
                else:
                    action = await _await_action(renderer)
                if action == "quit":
                    break
                if action == "tile_scale_up":
                    options["tile_scale"] = _next_scale(_coerce_scale(options.get("tile_scale", 1.0)), direction=1)
                    save_options(options)
                    apply_fn = getattr(renderer, "apply_options", None)
                    if callable(apply_fn):
                        apply_fn(options)
                    esper.process(None)
                    continue
                if action == "tile_scale_down":
                    options["tile_scale"] = _next_scale(_coerce_scale(options.get("tile_scale", 1.0)), direction=-1)
                    save_options(options)
                    apply_fn = getattr(renderer, "apply_options", None)
                    if callable(apply_fn):
                        apply_fn(options)
                    esper.process(None)
                    continue
                if action == "ui_layout_changed":
                    save_options(options)
                    esper.process(None)
                    continue
                if action == "sleep":
                    held_directions.clear()
                    # Sleep in a bed if one's at hand (warning first if it's not
                    # yours); otherwise pitch a camp.
                    nearby_bed = _bed_near_player()
                    if nearby_bed is not None:
                        if await _confirm_if_owned_by_other(renderer, nearby_bed, "bed", "sleep here"):
                            await _sleep_player(renderer, in_camp=False)
                        else:
                            esper.process(None)
                    else:
                        await _sleep_player(renderer, in_camp=True)
                    continue
                if action == "look":
                    held_directions.clear()
                    look_choice = await _look_mode(renderer, game_map)
                    if look_choice == "quit":
                        break
                    esper.process(None)
                    continue
                if action in _CARDINAL_ACTION_DELTAS:
                    held_directions.add(action)
                    press_order_counter += 1
                    direction_pressed_order[action] = press_order_counter
                    esper.process(None)
                    continue
                if action in _RELEASE_TO_DIRECTION:
                    held_directions.discard(_RELEASE_TO_DIRECTION[action])
                    esper.process(None)
                    continue
                if action == "confirm_action":
                    live_action = _action_from_held_keys(held_directions, direction_pressed_order)
                    if live_action is not None:
                        esper.process(live_action)
                    else:
                        # No direction held: wait in place, passing a turn (needs
                        # rise, NPCs act) instead of a no-op refresh.
                        esper.process(WAIT_ACTION)
                    continue
                if action == "menu_select":
                    interact_action = _action_from_held_keys(held_directions, direction_pressed_order)
                    interact_creature = _find_interaction_creature(interact_action)
                    if interact_creature is not None:
                        held_directions.clear()
                        if esper.has_component(interact_creature, Friendly):
                            # Friendlies: full dialogue (which shows their status).
                            choice = await _draw_dialogue_menu(renderer, game_map, interact_creature)
                        else:
                            # Wild/hostile creatures: read-only examine of status.
                            name = _entity_name(interact_creature, fallback="Creature")
                            choice = await _draw_info_screen(
                                renderer,
                                title=f"EXAMINE - {name}",
                                lines=_creature_status_lines(game_map, interact_creature),
                                subtitle="What you can tell at a glance",
                            )
                        if choice == "quit":
                            break
                        esper.process(None)
                        continue

                    interact_corpse = _find_interaction_corpse(interact_action)
                    if interact_corpse is not None:
                        loot_choice = await _draw_loot_menu(renderer, interact_corpse)
                        held_directions.clear()
                        if loot_choice == "quit":
                            break
                        esper.process(None)
                        continue

                    interact_chest = _find_adjacent_feature(interact_action, Chest)
                    if interact_chest is not None:
                        held_directions.clear()
                        if await _confirm_if_owned_by_other(renderer, interact_chest, "chest", "open it"):
                            loot_choice = await _draw_loot_menu(renderer, interact_chest)
                            if loot_choice == "quit":
                                break
                        esper.process(None)
                        continue

                    # Environment features: chop a faced tree, drink from a faced
                    # well, or cook at a faced stove. Each queues a log line and
                    # refreshes the frame (a free action, like looting).
                    player_ent = _first_player_entity()
                    if player_ent is not None:
                        interact_tree = _find_adjacent_feature(interact_action, Tree)
                        if interact_tree is not None:
                            queue_message(_chop_tree(interact_tree, player_ent))
                            esper.process(None)
                            continue

                        interact_bush = _find_adjacent_feature(interact_action, BerryBush)
                        if interact_bush is not None:
                            queue_message(_harvest_bush(interact_bush, player_ent))
                            esper.process(None)
                            continue

                        interact_well = _find_adjacent_feature(interact_action, Well)
                        if interact_well is not None:
                            queue_message(_drink_from_well(interact_well, player_ent))
                            esper.process(None)
                            continue

                        interact_stove = _find_adjacent_feature(interact_action, Stove)
                        if interact_stove is not None:
                            queue_message(_cook_at_stove(interact_stove, player_ent))
                            esper.process(None)
                            continue

                        interact_bed = _find_adjacent_feature(interact_action, Bed)
                        if interact_bed is not None:
                            held_directions.clear()
                            if await _confirm_if_owned_by_other(renderer, interact_bed, "bed", "sleep here"):
                                await _sleep_player(renderer, in_camp=False)
                            else:
                                esper.process(None)
                            continue

                if action in {"open_menu", "open_inventory", "open_status"}:
                    # Tab reopens on the last tab; I/C jump straight to a tab.
                    if action == "open_inventory":
                        start_tab = "inventory"
                    elif action == "open_status":
                        start_tab = "status"
                    else:
                        start_tab = last_menu_tab
                    menu_choice, last_menu_tab = await _draw_player_menu(renderer, game_map, start_tab)
                    held_directions.clear()
                    if menu_choice == "quit":
                        break
                    if menu_choice.startswith("place:"):
                        # The player chose a buildable in the inventory; ask for a
                        # direction and build it on that tile.
                        await _place_from_inventory(renderer, game_map, menu_choice[len("place:"):])
                        continue
                    esper.process(None)
                    continue
                if action == "open_pause_menu":
                    pause_choice = await _draw_pause_menu(renderer, options)
                    held_directions.clear()
                    if pause_choice == "save_game":
                        player_pos = first_player_position() or player_position
                        save_game(game_map, selected_save_file, player_pos)
                    elif pause_choice == "quit":
                        break
                    esper.process(None)
                    continue
                esper.process(action)
    finally:
        _stop_background_music(pygame_module)


def run() -> None:
    """Synchronous entry point for desktop runs (`python3 src/main.py`)."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
