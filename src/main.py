"""pyRL2 entry point.

Turn loop: show title/menu, then block for an action, run the systems, repeat.
The game logic stays renderer-agnostic while the default runtime uses pygame.
"""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import esper

from components import BlocksMovement, Dialogue, Enemy, Equipment, Friendly, Inventory, NPC, Name, Player, Position, Renderable, Vision
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
        import pygame
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


def _start_background_music(options: dict | None = None) -> Any | None:
    pygame = _init_pygame_mixer(options)
    if pygame is None:
        return None

    music_dir = Path(__file__).resolve().parent.parent / "audio" / "music"
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
    return parser.parse_args()


def _draw_title_screen(renderer: Renderer) -> bool:
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


def _draw_main_menu(renderer: Renderer) -> str:
    options = ["Continue", "New Game", "Quit"]
    selected = 0

    while True:
        renderer.clear()
        renderer.draw_text(12, 5, "MAIN MENU")
        for idx, item in enumerate(options):
            prefix = "> " if idx == selected else "  "
            renderer.draw_text(10, 8 + idx, f"{prefix}{item}")
        renderer.draw_text(3, 14, "Use W/S to move, Enter to select")
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


def _draw_options_menu(renderer: Renderer, options: dict) -> str:
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
            f"Tile Scale: {tile_scale:.2f}x",
            f"UI Scale: {ui_scale:.2f}x",
            "Back",
        ]

        renderer.clear()
        renderer.draw_text(12, 5, "OPTIONS")
        for idx, item in enumerate(items):
            prefix = "> " if idx == selected else "  "
            renderer.draw_text(8, 8 + idx, f"{prefix}{item}")
        renderer.draw_text(2, 14, "Use W/S to move, A/D to change scale, Enter to toggle/select")
        renderer.draw_text(2, 15, "Esc to return")
        renderer.present()

        action = renderer.poll_action()
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
        elif action == "menu_select":
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


def _draw_pause_menu(renderer: Renderer, options: dict) -> str:
    menu_items = ["Save Game", "Options", "Quit"]
    selected = 0

    while True:
        renderer.clear()
        renderer.draw_text(12, 5, "PAUSE MENU")
        for idx, item in enumerate(menu_items):
            prefix = "> " if idx == selected else "  "
            renderer.draw_text(10, 8 + idx, f"{prefix}{item}")
        renderer.draw_text(3, 14, "Use W/S to move, Enter to select")
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

    preferred_xy = _direction_target_xy(direction_action, player_pos)
    if preferred_xy is not None and preferred_xy in position_to_npc:
        return position_to_npc[preferred_xy]

    for action_name in ("move_up", "move_right", "move_down", "move_left"):
        adjacent_xy = _direction_target_xy(action_name, player_pos)
        if adjacent_xy is not None and adjacent_xy in position_to_npc:
            return position_to_npc[adjacent_xy]
    return None


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


def _draw_trade_menu(renderer: Renderer, npc_ent: int) -> str:
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

        renderer.clear()
        renderer.draw_text(2, 1, f"TRADE - {npc_name}")
        renderer.draw_text(2, 2, "Esc or i to close")

        left_x = 2
        right_x = 40
        top_y = 4

        renderer.draw_text(left_x, top_y, f"== {npc_name} (LEFT) ==")
        y = top_y + 2
        if not npc_entries:
            prefix = "> " if selected_panel == "left" else "  "
            renderer.draw_text(left_x, y, f"{prefix}(empty)")
        else:
            for idx, entry in enumerate(npc_entries):
                prefix = "> " if selected_panel == "left" and idx == selected_npc_idx else "  "
                if entry.kind == "equipment":
                    label = f"[E] {entry.slot_name}: {entry.item_name}"
                else:
                    label = entry.item_name
                renderer.draw_text(left_x, y, f"{prefix}{label}"[:36])
                y += 1

        renderer.draw_text(right_x, top_y, f"== {player_name} (RIGHT) ==")
        y = top_y + 2
        if not player_entries:
            prefix = "> " if selected_panel == "right" else "  "
            renderer.draw_text(right_x, y, f"{prefix}(empty)")
        else:
            for idx, entry in enumerate(player_entries):
                prefix = "> " if selected_panel == "right" and idx == selected_player_idx else "  "
                if entry.kind == "equipment":
                    label = f"[E] {entry.slot_name}: {entry.item_name}"
                else:
                    label = entry.item_name
                renderer.draw_text(right_x, y, f"{prefix}{label}"[:36])
                y += 1

        renderer.draw_text(2, top_y + 14, message[:76])
        renderer.present()

        action = renderer.poll_action()
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

        if action == "menu_select":
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


def _draw_dialogue_menu(renderer: Renderer, npc_ent: int) -> str:
    selected = 0
    options = ["Talk", "Trade", "Leave"]
    info_lines = _npc_info_lines(npc_ent)
    talk_line = ""

    while True:
        npc_name = _entity_name(npc_ent, fallback="Unknown NPC")

        renderer.clear()
        renderer.draw_text(2, 1, f"DIALOGUE - {npc_name}")
        renderer.draw_text(2, 2, "Esc to close")

        y = 4
        for line in info_lines:
            renderer.draw_text(2, y, line[:76])
            y += 1

        y += 1
        renderer.draw_text(2, y, "== OPTIONS ==")
        y += 1
        for idx, option in enumerate(options):
            prefix = "> " if idx == selected else "  "
            renderer.draw_text(2, y + idx, f"{prefix}{option}")

        if talk_line:
            renderer.draw_text(2, y + 5, talk_line[:76])

        renderer.present()
        action = renderer.poll_action()

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

        if action == "menu_select":
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
                trade_choice = _draw_trade_menu(renderer, npc_ent)
                if trade_choice == "quit":
                    return "quit"
                info_lines = _npc_info_lines(npc_ent)
                talk_line = ""


def _draw_inventory_menu(renderer: Renderer) -> str:
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

        renderer.clear()
        renderer.draw_text(2, 1, f"INVENTORY - {player_name}")
        renderer.draw_text(2, 2, "Esc or i to close")

        left_x = 2
        right_x = 40
        top_y = 4

        renderer.draw_text(left_x, top_y, "== EQUIPPED ==")
        slot_names = list(equipment_slots.keys())
        slot_y = top_y + 2
        for idx, slot_name in enumerate(slot_names):
            equipped_item = equipment_slots[slot_name]
            display = equipped_item if equipped_item else "(empty)"
            prefix = "> " if selected_panel == "left" and idx == selected_slot_idx else "  "
            renderer.draw_text(left_x, slot_y, f"{prefix}{slot_name:10}: {display}")
            slot_y += 1

        renderer.draw_text(right_x, top_y, "== ITEMS ==")
        item_y = top_y + 2
        if not inventory_items:
            prefix = "> " if selected_panel == "right" else "  "
            renderer.draw_text(right_x, item_y, f"{prefix}(empty)")
        else:
            for idx, item in enumerate(inventory_items):
                label = chr(ord("a") + idx) if idx < 26 else "*"
                prefix = "> " if selected_panel == "right" and idx == selected_item_idx else "  "
                renderer.draw_text(right_x, item_y, f"{prefix}{label}) {item}")
                item_y += 1

        renderer.draw_text(2, top_y + 13, message[:76])

        renderer.present()

        action = renderer.poll_action()
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

        if action == "menu_select" and player_ent is not None:
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


def _setup_world(game_map: GameMap, player_position: Position) -> None:
    player_equipment = _default_equipment_slots()
    player_equipment["main hand"] = "Rusty Sword"
    player_equipment["chest"] = "Traveler Tunic"
    esper.create_entity(
        player_position,
        Renderable("@"),
        Name("You"),
        Player(),
        Vision(10),
        BlocksMovement(),
        Inventory(items=["Bandage", "Torch", "Apple"]),
        Equipment(slots=player_equipment),
    )

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
        Inventory(items=["Bread", "Waterskin"]),
        Equipment(slots=_default_equipment_slots()),
    )
    goblin_equipment = _default_equipment_slots()
    goblin_equipment["main hand"] = "Jagged Dagger"
    esper.create_entity(
        guard_pos,
        Renderable("g"),
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
        Renderable("r"),
        Name("Cave Rat"),
        NPC(),
        Enemy(),
        Vision(6),
        BlocksMovement(),
        Inventory(items=["String", "Pebble"]),
        Equipment(slots=_default_equipment_slots()),
    )


def main() -> None:
    args = _parse_args()
    bootstrap_files(MAP_WIDTH, MAP_HEIGHT)
    options = load_options()
    pygame_module = _start_background_music(options)
    combat_sfx = _CombatSfxPlayer(pygame_module, options)

    selected_save_file = args.save_file

    if selected_save_file is None:
        selected_save_file = DEFAULT_SAVE_FILE

    game_map = GameMap(MAP_WIDTH, MAP_HEIGHT)
    player_position = Position(MAP_WIDTH // 2, MAP_HEIGHT // 2)

    try:
        with PygameRenderer(options=options) as renderer:
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
            held_directions: set[str] = set()
            direction_pressed_order = {
                "move_up": -1,
                "move_down": -1,
                "move_left": -1,
                "move_right": -1,
            }
            press_order_counter = 0
            while True:
                action = renderer.poll_action()
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
                        dialogue_choice = _draw_dialogue_menu(renderer, interact_npc)
                        held_directions.clear()
                        if dialogue_choice == "quit":
                            break
                        esper.process(None)
                        continue

                if action == "open_inventory":
                    inventory_choice = _draw_inventory_menu(renderer)
                    held_directions.clear()
                    if inventory_choice == "quit":
                        break
                    esper.process(None)
                    continue
                if action == "open_pause_menu":
                    pause_choice = _draw_pause_menu(renderer, options)
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


if __name__ == "__main__":
    main()
