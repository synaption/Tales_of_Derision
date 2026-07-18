"""pyRL2 entry point.

Turn loop: show title/menu, then block for an action, run the systems, repeat.
The game logic never touches curses -- swap TerminalRenderer for a
tcod/pygame/raylib renderer and nothing else changes.
"""
import argparse
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Any

import esper

from audio_sfx import CombatSfxPlayer
from components import BlocksMovement, Dialogue, Enemy, Friendly, NPC, Name, Player, Position, Renderable, Vision
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
from renderer.terminal import TerminalRenderer
from systems import MovementProcessor, NpcAiProcessor, RenderProcessor

MAP_WIDTH = 40
MAP_HEIGHT = 20

AUDIO_SAMPLE_RATE = 44100
AUDIO_SAMPLE_SIZE = -16
AUDIO_CHANNELS = 2
AUDIO_BUFFER_SIZES = (16384, 8192, 4096, 2048)


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


def _start_background_music(options: dict | None = None) -> Any | None:
    music_dir = Path(__file__).resolve().parent.parent / "audio" / "music"
    track = _pick_music_track(music_dir)
    if track is None:
        return None

    try:
        import pygame
    except ModuleNotFoundError:
        return None

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
                pygame.mixer.music.load(str(track))
                pygame.mixer.music.play(-1)
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


def _stop_background_music(pygame_module: Any | None) -> None:
    if pygame_module is None:
        return
    try:
        pygame_module.mixer.music.stop()
    except Exception:
        pass


def _music_worker(stop_event: Any, options: dict | None) -> None:
    pygame_module = _start_background_music(options)
    if pygame_module is None:
        return
    try:
        while not stop_event.wait(0.25):
            pass
    finally:
        _stop_background_music(pygame_module)


def _start_background_music_process(options: dict | None) -> tuple[multiprocessing.Process, Any] | None:
    try:
        stop_event = multiprocessing.Event()
        process = multiprocessing.Process(
            target=_music_worker,
            args=(stop_event, options),
            daemon=True,
            name="music-worker",
        )
        process.start()
        return process, stop_event
    except Exception as exc:
        print(f"Audio process disabled: {exc}", file=sys.stderr)
        return None


def _stop_background_music_process(handle: tuple[multiprocessing.Process, Any] | None) -> None:
    if handle is None:
        return
    process, stop_event = handle
    try:
        stop_event.set()
    except Exception:
        pass
    process.join(timeout=2.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
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
    return parser.parse_args()


def _draw_title_screen(renderer: TerminalRenderer) -> bool:
    while True:
        renderer.clear()
        renderer.draw_text(12, 6, "PYRL2")
        renderer.draw_text(7, 8, "A tiny ECS roguelike prototype")
        renderer.draw_text(6, 11, "Press Enter to continue")
        renderer.draw_text(8, 12, "Press Esc to quit")
        renderer.present()

        action = renderer.poll_action()
        if action in {"quit", "open_pause_menu"}:
            return False
        if action == "menu_select":
            return True


def _draw_main_menu(renderer: TerminalRenderer) -> str:
    options = ["Continue", "New Game", "Quit"]
    selected = 0

    while True:
        renderer.clear()
        renderer.draw_text(12, 5, "MAIN MENU")
        for idx, item in enumerate(options):
            prefix = "> " if idx == selected else "  "
            renderer.draw_text(10, 8 + idx, f"{prefix}{item}")
        renderer.draw_text(3, 14, "Use arrows/WASD, Enter to select")
        renderer.present()

        action = renderer.poll_action()
        if action in {"quit", "open_pause_menu"}:
            return "quit"
        if action == "move_up":
            selected = (selected - 1) % len(options)
        elif action == "move_down":
            selected = (selected + 1) % len(options)
        elif action == "menu_select":
            lowered = options[selected].lower().replace(" ", "_")
            return lowered


def _draw_options_menu(renderer: TerminalRenderer, options: dict) -> str:
    selected = 0

    while True:
        fullscreen = bool(options.get("fullscreen", False))
        show_fps = bool(options.get("show_fps", False))
        items = [
            f"Fullscreen: {'ON' if fullscreen else 'OFF'}",
            f"Show FPS: {'ON' if show_fps else 'OFF'}",
            "Back",
        ]

        renderer.clear()
        renderer.draw_text(12, 5, "OPTIONS")
        for idx, item in enumerate(items):
            prefix = "> " if idx == selected else "  "
            renderer.draw_text(8, 8 + idx, f"{prefix}{item}")
        renderer.draw_text(2, 14, "Use arrows/WASD, Enter to toggle/select")
        renderer.draw_text(2, 15, "Esc to return")
        renderer.present()

        action = renderer.poll_action()
        if action in {"open_pause_menu", "quit"}:
            return "back"
        if action == "move_up":
            selected = (selected - 1) % len(items)
        elif action == "move_down":
            selected = (selected + 1) % len(items)
        elif action == "menu_select":
            if selected == 0:
                options["fullscreen"] = not fullscreen
                save_options(options)
            elif selected == 1:
                options["show_fps"] = not show_fps
                save_options(options)
            else:
                return "back"


def _draw_pause_menu(renderer: TerminalRenderer, options: dict) -> str:
    menu_items = ["Save Game", "Options", "Quit"]
    selected = 0

    while True:
        renderer.clear()
        renderer.draw_text(12, 5, "PAUSE MENU")
        for idx, item in enumerate(menu_items):
            prefix = "> " if idx == selected else "  "
            renderer.draw_text(10, 8 + idx, f"{prefix}{item}")
        renderer.draw_text(3, 14, "Use arrows/WASD, Enter to select")
        renderer.draw_text(3, 15, "Esc to resume")
        renderer.present()

        action = renderer.poll_action()
        if action == "open_pause_menu":
            return "resume"
        if action == "quit":
            return "quit"
        if action == "move_up":
            selected = (selected - 1) % len(menu_items)
        elif action == "move_down":
            selected = (selected + 1) % len(menu_items)
        elif action == "menu_select":
            chosen = menu_items[selected].lower().replace(" ", "_")
            if chosen == "options":
                _draw_options_menu(renderer, options)
                continue
            return chosen


def _setup_world(game_map: GameMap, player_position: Position, options: dict) -> None:
    esper.add_processor(MovementProcessor(game_map, combat_sfx=CombatSfxPlayer(options)), priority=1)
    esper.add_processor(NpcAiProcessor(game_map), priority=0)
    esper.create_entity(player_position, Renderable("@"), Name("You"), Player(), Vision(10), BlocksMovement())

    villager_pos = Position(max(2, player_position.x - 2), player_position.y + 1)
    guard_pos = Position(max(2, player_position.x - 5), player_position.y)
    rat_pos = Position(min(game_map.width - 3, player_position.x + 6), max(2, player_position.y - 2))

    esper.create_entity(
        villager_pos,
        Renderable("v"),
        Name("Friendly Villager"),
        NPC(),
        Friendly(),
        Dialogue("##!/$*~# GH01^@"),
        BlocksMovement(),
    )
    esper.create_entity(guard_pos, Renderable("g"), Name("Goblin Scout"), NPC(), Enemy(), Vision(8), BlocksMovement())
    esper.create_entity(rat_pos, Renderable("r"), Name("Cave Rat"), NPC(), Enemy(), Vision(6), BlocksMovement())


def main() -> None:
    args = _parse_args()
    bootstrap_files(MAP_WIDTH, MAP_HEIGHT)
    options = load_options()
    use_audio_process = bool(options.get("audio_separate_process", True))
    audio_process_handle: tuple[multiprocessing.Process, Any] | None = None
    pygame_module: Any | None = None
    if use_audio_process:
        audio_process_handle = _start_background_music_process(options)
        if audio_process_handle is None:
            pygame_module = _start_background_music(options)
    else:
        pygame_module = _start_background_music(options)

    selected_save_file = args.save_file

    if selected_save_file is None:
        selected_save_file = DEFAULT_SAVE_FILE

    game_map = GameMap(MAP_WIDTH, MAP_HEIGHT)
    player_position = Position(MAP_WIDTH // 2, MAP_HEIGHT // 2)

    try:
        with TerminalRenderer() as renderer:
            if args.save_file is None:
                if not _draw_title_screen(renderer):
                    return

                menu_choice = _draw_main_menu(renderer)
                if menu_choice == "quit":
                    return
                if menu_choice == "continue":
                    game_map, player_position = load_game(
                        DEFAULT_SAVE_FILE,
                        MAP_WIDTH,
                        MAP_HEIGHT,
                    )
                elif menu_choice == "new_game":
                    save_game(game_map, DEFAULT_SAVE_FILE, player_position)
            else:
                game_map, player_position = load_game(
                    selected_save_file,
                    MAP_WIDTH,
                    MAP_HEIGHT,
                )

            _setup_world(game_map, player_position, options)
            esper.add_processor(RenderProcessor(renderer, game_map), priority=0)

            esper.process()  # initial frame
            while True:
                action = renderer.poll_action()
                if action == "open_pause_menu":
                    pause_choice = _draw_pause_menu(renderer, options)
                    if pause_choice == "save_game":
                        player_pos = first_player_position() or player_position
                        save_game(game_map, selected_save_file, player_pos)
                    elif pause_choice == "quit":
                        break
                    esper.process(None)
                    continue
                esper.process(action)
    finally:
        _stop_background_music_process(audio_process_handle)
        _stop_background_music(pygame_module)


if __name__ == "__main__":
    main()
