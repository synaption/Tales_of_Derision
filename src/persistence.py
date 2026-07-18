"""Save/options persistence and bootstrap helpers."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import esper

from components import Player, Position
from game_map import GameMap

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAVES_DIR = DATA_DIR / "saves"
CONFIG_DIR = DATA_DIR / "config"

DEFAULT_SAVE_FILE = SAVES_DIR / "default_save.json"
DEFAULT_OPTIONS_FILE = CONFIG_DIR / "default_options.json"
WORKING_OPTIONS_FILE = CONFIG_DIR / "options.json"

_DEFAULT_OPTIONS = {
    "fullscreen": False,
    "show_fps": False,
    "audio_buffer": 16384,
    "combat_sfx": True,
    "melee_attack_sfx": "audio/sfx/swipe.wav",
    "death_sfx": "audio/sfx/splat_quick.wav",
    "keybinds": {
        "up": ["w", "k", "up"],
        "down": ["s", "j", "down"],
        "left": ["a", "h", "left"],
        "right": ["d", "l", "right"],
        "save": ["p"],
        "quit": ["q", "esc"],
    },
}


def _mkdirs() -> None:
    SAVES_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def create_default_save_file(map_width: int, map_height: int) -> None:
    if DEFAULT_SAVE_FILE.exists():
        return
    save_data = {
        "map": {"width": map_width, "height": map_height},
        "player": {"x": map_width // 2, "y": map_height // 2},
    }
    _write_json(DEFAULT_SAVE_FILE, save_data)


def ensure_options_files() -> None:
    _mkdirs()
    if not DEFAULT_OPTIONS_FILE.exists():
        _write_json(DEFAULT_OPTIONS_FILE, _DEFAULT_OPTIONS)
    if not WORKING_OPTIONS_FILE.exists():
        shutil.copyfile(DEFAULT_OPTIONS_FILE, WORKING_OPTIONS_FILE)


def bootstrap_files(map_width: int, map_height: int) -> None:
    _mkdirs()
    ensure_options_files()
    create_default_save_file(map_width, map_height)


def load_options() -> dict:
    ensure_options_files()
    with WORKING_OPTIONS_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        return dict(_DEFAULT_OPTIONS)
    return payload


def save_options(options: dict) -> None:
    _write_json(WORKING_OPTIONS_FILE, options)


def save_game(game_map: GameMap, save_file: Path, player_pos: Position) -> None:
    save_data = {
        "map": {"width": game_map.width, "height": game_map.height},
        "player": {"x": player_pos.x, "y": player_pos.y},
    }
    _write_json(save_file, save_data)


def load_game(save_file: Path, fallback_width: int, fallback_height: int) -> tuple[GameMap, Position]:
    if not save_file.exists():
        game_map = GameMap(fallback_width, fallback_height)
        return game_map, Position(fallback_width // 2, fallback_height // 2)

    with save_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    map_data = payload.get("map", {})
    player_data = payload.get("player", {})

    width = int(map_data.get("width", fallback_width))
    height = int(map_data.get("height", fallback_height))
    game_map = GameMap(width, height)

    px = int(player_data.get("x", width // 2))
    py = int(player_data.get("y", height // 2))
    if not game_map.in_bounds(px, py):
        px, py = width // 2, height // 2

    return game_map, Position(px, py)


def first_player_position() -> Position | None:
    for _ent, (pos, _player) in esper.get_components(Position, Player):
        return pos
    return None
