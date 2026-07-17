"""pyRL2 entry point.

Turn loop: show title/menu, then block for an action, run the systems, repeat.
The game logic never touches curses -- swap TerminalRenderer for a
tcod/pygame/raylib renderer and nothing else changes.
"""
import argparse
from pathlib import Path

import esper

from components import Player, Position, Renderable
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
from systems import MovementProcessor, RenderProcessor

MAP_WIDTH = 40
MAP_HEIGHT = 20


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


def _setup_world(game_map: GameMap, player_position: Position) -> None:
    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.create_entity(player_position, Renderable("@"), Player())


def main() -> None:
    args = _parse_args()
    bootstrap_files(MAP_WIDTH, MAP_HEIGHT)

    selected_save_file = args.save_file

    if selected_save_file is None:
        selected_save_file = DEFAULT_SAVE_FILE

    options = load_options()

    game_map = GameMap(MAP_WIDTH, MAP_HEIGHT)
    player_position = Position(MAP_WIDTH // 2, MAP_HEIGHT // 2)

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

        _setup_world(game_map, player_position)
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


if __name__ == "__main__":
    main()
