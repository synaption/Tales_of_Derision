"""pyRL2 entry point.

Turn loop: show title/menu, then block for an action, run the systems, repeat.
The game logic stays renderer-agnostic while the default runtime uses pygame.
"""
import argparse
import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any

import esper

from components import BlocksMovement, Corpse, Dialogue, Enemy, Equipment, Friendly, Inventory, NPC, Name, Player, Position, Renderable, Vision
from game_map import GameMap
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
from systems import MovementProcessor, NpcAiProcessor, RenderProcessor

MAP_WIDTH = 40
MAP_HEIGHT = 20

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


def _npc_info_lines(npc_ent: int) -> list[str]:
    npc_name = _entity_name(npc_ent, fallback="Unknown NPC")
    disposition = "Friendly"
    if esper.has_component(npc_ent, Enemy):
        disposition = "Hostile"

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

    return [
        f"Name: {npc_name}",
        f"Disposition: {disposition}",
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


async def _draw_dialogue_menu(renderer: Renderer, npc_ent: int) -> str:
    selected = 0
    options = ["Talk", "Trade", "Leave"]
    info_lines = _npc_info_lines(npc_ent)
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
                continue
            if choice == "Trade":
                trade_choice = await _draw_trade_menu(renderer, npc_ent)
                if trade_choice == "quit":
                    return "quit"
                info_lines = _npc_info_lines(npc_ent)
                talk_line = ""


async def _draw_inventory_menu(renderer: Renderer) -> str:
    selected_panel = "left"
    selected_slot_idx = 0
    selected_item_idx = 0
    message = "Enter: equip/unequip  A/D: switch side  W/S: move"

    while True:
        player_ent = _first_player_entity()

        inventory_items: list[str] = []
        equipment_slots = _default_equipment_slots()
        player_name = "You"

        if player_ent is not None:
            if esper.has_component(player_ent, Name):
                player_name = esper.component_for_entity(player_ent, Name).value

            if esper.has_component(player_ent, Inventory):
                inventory_items = list(esper.component_for_entity(player_ent, Inventory).items)

            if esper.has_component(player_ent, Equipment):
                configured = esper.component_for_entity(player_ent, Equipment).slots
                for slot_name in equipment_slots:
                    equipment_slots[slot_name] = configured.get(slot_name)

        x, y, width, height = _draw_menu_shell(
            renderer,
            title=f"INVENTORY - {player_name}",
            subtitle="Esc/I closes   A/D switches side",
            footer=message,
            width=98,
            height=32,
            overlay_game=True,
        )

        content_y = y + 4
        content_h = max(10, height - 7)
        left_w = max(20, (width - 6) // 2)
        right_w = max(20, width - left_w - 6)
        left_x = x + 2
        right_x = left_x + left_w + 2

        draw_panel = getattr(renderer, "draw_panel", None)
        draw_ui_glyph = getattr(renderer, "draw_ui_glyph", None)
        icon_cells = 2 if callable(draw_ui_glyph) else 0
        text_offset = icon_cells + 1 if icon_cells > 0 else 0
        if callable(draw_panel):
            draw_panel(left_x, content_y, left_w, content_h, title="EQUIPPED")
            draw_panel(right_x, content_y, right_w, content_h, title="ITEMS")

        slot_names = list(equipment_slots.keys())
        list_start_y = content_y + 3
        visible_rows = max(1, content_h - 5)

        if slot_names:
            slot_start = _scroll_start(selected_slot_idx, len(slot_names), visible_rows)
            slot_end = min(len(slot_names), slot_start + visible_rows)
            for row_offset, slot_idx in enumerate(range(slot_start, slot_end)):
                slot_name = slot_names[slot_idx]
                equipped_item = equipment_slots[slot_name]
                display = equipped_item if equipped_item else "(empty)"
                row_y = list_start_y + row_offset
                is_selected = selected_panel == "left" and slot_idx == selected_slot_idx
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
                            draw_ui_glyph(
                                left_x + 2,
                                row_y,
                                glyph,
                                classification=classification,
                                fg=fg,
                                bg=bg,
                                cell_span=icon_cells,
                            )
                        )
                    if not icon_drawn:
                        glyph_prefix = f"{glyph} "

                _draw_ui_text(
                    renderer,
                    text_x,
                    row_y,
                    f"{prefix} {slot_name:10}: {glyph_prefix}{display}",
                    color,
                    text_width,
                )

        if inventory_items:
            item_start = _scroll_start(selected_item_idx, len(inventory_items), visible_rows)
            item_end = min(len(inventory_items), item_start + visible_rows)
            for row_offset, item_idx in enumerate(range(item_start, item_end)):
                item_name = inventory_items[item_idx]
                row_y = list_start_y + row_offset
                label = chr(ord("a") + item_idx) if item_idx < 26 else "*"
                is_selected = selected_panel == "right" and item_idx == selected_item_idx
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
                        draw_ui_glyph(
                            right_x + 2,
                            row_y,
                            glyph,
                            classification=classification,
                            fg=fg,
                            bg=bg,
                            cell_span=icon_cells,
                        )
                    )

                label_text = f"{prefix} {label}) {item_name}"
                if not icon_drawn:
                    label_text = f"{prefix} {label}) {glyph} {item_name}"

                _draw_ui_text(
                    renderer,
                    text_x,
                    row_y,
                    label_text,
                    color,
                    text_width,
                )
        else:
            is_selected = selected_panel == "right"
            if is_selected:
                _fill_ui_cells(renderer, right_x + 1, list_start_y, right_w - 2, 1, _MENU_SELECTED_BG)
            color = _MENU_SELECTED_TEXT if is_selected else _MENU_TEXT_COLOR
            prefix = ">" if is_selected else " "
            _draw_ui_text(
                renderer,
                right_x + 2 + text_offset,
                list_start_y,
                f"{prefix} (empty)",
                color,
                max(1, right_w - 4 - text_offset),
            )

        renderer.present()

        action = await _await_action(renderer)
        if action in {"open_pause_menu", "open_inventory"}:
            return "close"
        if action == "quit":
            return "quit"

        if action in {"move_left"}:
            selected_panel = "left"
            continue
        if action in {"move_right"}:
            selected_panel = "right"
            continue

        if action == "move_up":
            if selected_panel == "left" and slot_names:
                selected_slot_idx = (selected_slot_idx - 1) % len(slot_names)
            elif selected_panel == "right" and inventory_items:
                selected_item_idx = (selected_item_idx - 1) % len(inventory_items)
            continue

        if action == "move_down":
            if selected_panel == "left" and slot_names:
                selected_slot_idx = (selected_slot_idx + 1) % len(slot_names)
            elif selected_panel == "right" and inventory_items:
                selected_item_idx = (selected_item_idx + 1) % len(inventory_items)
            continue

        if action in {"menu_select", "confirm_action"} and player_ent is not None:
            if not esper.has_component(player_ent, Inventory):
                continue
            if not esper.has_component(player_ent, Equipment):
                continue

            inventory = esper.component_for_entity(player_ent, Inventory)
            equipment = esper.component_for_entity(player_ent, Equipment)

            if selected_panel == "right":
                if not inventory.items:
                    message = "No items to equip."
                else:
                    clamped_idx = max(0, min(selected_item_idx, len(inventory.items) - 1))
                    message = _equip_inventory_item(inventory, equipment, clamped_idx)
                    if inventory.items:
                        selected_item_idx = min(selected_item_idx, len(inventory.items) - 1)
                    else:
                        selected_item_idx = 0
            else:
                if slot_names:
                    slot_name = slot_names[max(0, min(selected_slot_idx, len(slot_names) - 1))]
                    message = _unequip_slot(inventory, equipment, slot_name)


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
    player_name = "You"
    villager_name = "Friendly Villager"
    player_skin = _human_skin_tone(player_name)
    villager_skin = _human_skin_tone(villager_name, offset=7)

    player_equipment = _default_equipment_slots()
    player_equipment["main hand"] = "Rusty Sword"
    player_equipment["chest"] = "Traveler Tunic"
    esper.create_entity(
        player_position,
        Renderable("@", fg=player_skin),
        Name(player_name),
        Player(),
        Vision(10),
        BlocksMovement(),
        Inventory(items=["Bandage", "Torch", "Apple"]),
        Equipment(slots=player_equipment),
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

    villager_pos = Position(max(2, player_position.x - 2), player_position.y + 1)
    guard_pos = Position(max(2, player_position.x - 5), player_position.y)
    rat_pos = Position(min(game_map.width - 3, player_position.x + 6), max(2, player_position.y - 2))

    esper.create_entity(
        villager_pos,
        Renderable("v", fg=villager_skin),
        Name(villager_name),
        NPC(),
        Friendly(),
        Dialogue("##!/$*~# GH01^@"),
        BlocksMovement(),
        Inventory(items=["Bread", "Waterskin"]),
        Equipment(slots=_default_equipment_slots()),
    )
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
    )
    return 1


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
            esper.add_processor(NpcAiProcessor(game_map), priority=0)
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
            invalidate_backdrop = getattr(renderer, "invalidate_backdrop", None)
            while True:
                # Live play has (re)rendered the game, so any menu backdrop
                # snapshot is stale; the next menu to open will re-capture.
                if callable(invalidate_backdrop):
                    invalidate_backdrop()
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
                        esper.process(None)
                    continue
                if action == "menu_select":
                    interact_action = _action_from_held_keys(held_directions, direction_pressed_order)
                    interact_npc = _find_interaction_npc(interact_action)
                    if interact_npc is not None:
                        dialogue_choice = await _draw_dialogue_menu(renderer, interact_npc)
                        held_directions.clear()
                        if dialogue_choice == "quit":
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

                if action == "open_inventory":
                    inventory_choice = await _draw_inventory_menu(renderer)
                    held_directions.clear()
                    if inventory_choice == "quit":
                        break
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
