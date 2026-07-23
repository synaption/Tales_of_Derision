"""Presentation layer: title/main/pause/options screens, the tabbed player menu,
dialogue/trade/loot/look panels, low-level menu widgets, and the blocking-input +
scale helpers those screens use.

Everything here talks to a ``Renderer`` and drives the renderer-free
``interactions`` logic; the turn loop in ``main`` calls these screens. Split out of
``main`` so game flow and presentation stay separate. ``ui`` may import
``interactions`` (screens call logic) but never the reverse, and never ``main``.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import esper

from action import BASE_ACTION_COST
from components import (
    Asleep, Corpse, Dialogue, Equipment, Inventory, Needs, Personality, Player,
    Position,
)
from config import MAP_HEIGHT, MAP_WIDTH, WORLD_LAYOUT
from game_map import GameMap
from items import WOOD, craft_cost, default_equipment_slots, is_placeable
from persistence import DEFAULT_SAVE_FILE, load_game, save_game, save_options
from queries import entity_name, first_player_entity
from renderer.base import Renderer
from rng import world_rng
from systems import (
    FishAiProcessor, NpcAiProcessor, RenderProcessor, WAIT_ACTION, bed_owner,
    go_to_sleep, queue_message, wake_up, world_clock,
)
from interactions import (
    _CARDINAL_ACTION_DELTAS, _CRAFT_MENU, _apply_consumable, _chebyshev_from_player,
    _craft_item, _creature_at_xy, _creature_status_lines, _direction_target_xy,
    _entity_has_tradeable_items, _equip_inventory_item, _item_visual,
    _list_trade_entries, _look_available_actions, _loot_item_from_corpse,
    _npc_info_lines, _perform_look_attack, _place_buildable_at, _player_talk,
    _renderable_at_xy, _terrain_name, _trade_item, _unequip_slot,
)


_SCALE_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


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


_MENU_TITLE_COLOR = (236, 236, 236)


_MENU_TEXT_COLOR = (210, 210, 210)


_MENU_MUTED_COLOR = (156, 156, 156)


_MENU_SELECTED_BG = (44, 44, 44)


_MENU_SELECTED_TEXT = (252, 252, 252)


_TITLE_SPLASH_SECONDS = 3.0


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


def _await_action(renderer: Renderer) -> str | None:
    """Block for the next input action and return it."""
    return renderer.poll_action()


def _draw_title_screen(renderer: Renderer) -> bool:
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
            action = _await_action(renderer)

        if action == "quit":
            return False

        if action in {"menu_select", "confirm_action", "open_pause_menu"}:
            return True

        if time.monotonic() >= end_time:
            return True

        if callable(poll_action_nonblocking):
            time.sleep(0.016)  # ~60fps splash tick; keeps the wait from busy-spinning


def _draw_main_menu(renderer: Renderer) -> str:
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

        action = _await_action(renderer)
        if action in {"quit", "open_pause_menu"}:
            return "quit"
        if action == "move_up":
            selected = (selected - 1) % len(options)
        elif action == "move_down":
            selected = (selected + 1) % len(options)
        elif action in {"menu_select", "confirm_action"}:
            return options[selected][0]


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

        action = _await_action(renderer)
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


def _draw_pause_menu(renderer: Renderer, options: dict) -> str:
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

        action = _await_action(renderer)
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
                _draw_options_menu(renderer, options)
                continue
            return chosen


def _run_startup_flow(
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
            WORLD_LAYOUT,
        )
        return (True, loaded_map, loaded_player_position, selected_save_file)

    if not _draw_title_screen(renderer):
        return (False, game_map, player_position, selected_save_file)

    menu_choice = _draw_main_menu(renderer)
    if menu_choice == "quit":
        return (False, game_map, player_position, selected_save_file)
    if menu_choice == "continue":
        loaded_map, loaded_player_position = load_game(
            DEFAULT_SAVE_FILE,
            MAP_WIDTH,
            MAP_HEIGHT,
            WORLD_LAYOUT,
        )
        return (True, loaded_map, loaded_player_position, selected_save_file)
    if menu_choice == "new_game":
        save_game(game_map, DEFAULT_SAVE_FILE, player_position, seed=world_rng().seed)
        return (True, game_map, player_position, selected_save_file)

    return (False, game_map, player_position, selected_save_file)


_MIN_SLEEP_TIREDNESS = 1.0


_SLEEP_MAX_TURNS = 400


def _confirm_if_owned_by_other(renderer: Renderer, ent: int, noun: str, verb: str) -> bool:
    """When ``ent`` belongs to someone other than the player, warn whose it is and
    ask for confirmation. Returns True if the player may go ahead (their own/
    unowned property proceeds silently; a decline returns False)."""
    player_ent = first_player_entity()
    owner = bed_owner(ent)
    if owner is None or owner == player_ent:
        return True
    owner_name = entity_name(owner, fallback="someone else")
    return _confirm(
        renderer,
        title="Not Yours",
        lines=[
            f"This {noun} belongs to {owner_name}.",
            f"Are you sure you want to {verb}?",
        ],
    )


def _confirm(renderer: Renderer, title: str, lines: list[str]) -> bool:
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

        action = _await_action(renderer)
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


def _sleep_player(renderer: Renderer, in_camp: bool) -> None:
    """Send the player to sleep and fast-forward turns until they wake rested.

    The whole world keeps simulating during the rest (NPCs act, needs shift, the
    clock advances), so a night's sleep really passes the night. Each turn yields
    to the browser so the web build's single thread stays responsive."""
    player_ent = first_player_entity()
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

    # Safety net: never leave the player stuck asleep past the cap.
    if esper.has_component(player_ent, Asleep):
        wake_up(player_ent)

    # A night's sleep resolves the *whole* world, not just the region the bed
    # sits in -- region-aware processors otherwise only keep the player's
    # immediate region fully live turn by turn, leaving everywhere else to
    # catch up gradually in the background.
    clock = world_clock()
    if clock is not None:
        target_region_turn = clock.turn // BASE_ACTION_COST
        for processor in (esper.get_processor(NpcAiProcessor), esper.get_processor(FishAiProcessor)):
            if processor is not None:
                processor.scheduler.catch_up_all(target_region_turn)

    esper.process(None)


def _place_from_inventory(renderer: Renderer, game_map: GameMap, item_name: str) -> None:
    """Prompt for a direction and build the selected piece on that adjacent tile.
    Any non-direction key cancels and keeps the item."""
    player_ent = first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return
    queue_message(f"Place the {item_name}: press a direction (Esc/other to cancel).")
    esper.process(None)

    action = _await_action(renderer)
    player_pos = esper.component_for_entity(player_ent, Position)
    target = _direction_target_xy(action, player_pos)
    if target is None:
        queue_message(f"You put the {item_name} away.")
    else:
        queue_message(_place_buildable_at(player_ent, game_map, item_name, target))
    esper.process(None)


def _draw_trade_menu(renderer: Renderer, npc_ent: int) -> str:
    player_ent = first_player_entity()
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

        npc_name = entity_name(npc_ent, fallback="NPC")
        player_name = entity_name(player_ent, fallback="You")

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

        action = _await_action(renderer)
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


def _draw_loot_menu(renderer: Renderer, corpse_ent: int) -> str:
    player_ent = first_player_entity()
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

        corpse_name = entity_name(corpse_ent, fallback="Corpse")
        player_name = entity_name(player_ent, fallback="You")

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

        action = _await_action(renderer)
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


def _draw_info_screen(
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

        action = _await_action(renderer)
        if action == "quit":
            return "quit"
        if action in {"open_pause_menu", "open_inventory", "open_status", "menu_select", "confirm_action"}:
            return "close"


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
        name = entity_name(creature_ent, fallback="someone")
        if esper.has_component(creature_ent, Player):
            return f"Look: {name} (you).   [Enter] status   [Esc/L] done"
        verbs = ", ".join(_look_available_actions(creature_ent, dist)).lower()
        return f"Look: {name} - {dist} away.   [Enter] {verbs}   [Esc/L] done"

    feature_ent = _renderable_at_xy(x, y)
    if feature_ent is not None:
        name = entity_name(feature_ent, fallback="something")
        return f"Look: {name} - {dist} away.   [Enter] examine   [Esc/L] done"

    return f"Look: {_terrain_name(game_map, x, y)}.   {footer}"


def _draw_look_action_menu(
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
        name = entity_name(target_ent, fallback="Creature")
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

        action = _await_action(renderer)
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
                if _draw_trade_menu(renderer, target_ent) == "quit":
                    return "quit"
                talk_line = ""
                continue
            if choice == "Status":
                if _draw_info_screen(
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


def _look_interact(renderer: Renderer, game_map: GameMap, x: int, y: int) -> str:
    """Resolve an Enter press on the look cursor: interact with a creature there,
    loot an adjacent corpse, or just describe whatever the cursor rests on."""
    player_ent = first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return "close"
    player_pos = esper.component_for_entity(player_ent, Position)
    dist = _chebyshev_from_player(player_pos, x, y)

    creature_ent = _creature_at_xy(x, y)
    if creature_ent is not None:
        if creature_ent == player_ent:
            return _draw_info_screen(
                renderer,
                title="STATUS - You",
                lines=_creature_status_lines(game_map, player_ent),
                subtitle="How you're holding up",
            )
        options = _look_available_actions(creature_ent, dist)
        return _draw_look_action_menu(renderer, game_map, creature_ent, options)

    feature_ent = _renderable_at_xy(x, y)
    if feature_ent is not None:
        if (
            dist <= 1
            and esper.has_component(feature_ent, Corpse)
            and _entity_has_tradeable_items(feature_ent)
        ):
            return _draw_loot_menu(renderer, feature_ent)
        name = entity_name(feature_ent, fallback="something")
        return _draw_info_screen(
            renderer,
            title="LOOK",
            lines=[f"You see {name}.", f"Distance: {dist}."],
            subtitle="",
        )

    return _draw_info_screen(
        renderer,
        title="LOOK",
        lines=[f"You see {_terrain_name(game_map, x, y)}."],
        subtitle="",
    )


def _clear_look_overlay(render: RenderProcessor | None) -> None:
    if render is not None:
        render.look_cursor = None
        render.look_info = None


def _look_mode(renderer: Renderer, game_map: GameMap) -> str:
    """Drive the look cursor: it starts on the player, moves with the direction
    keys, and Enter interacts with whatever it rests on. Returns "quit" if the
    player quit the game while looking, otherwise "close"."""
    player_ent = first_player_entity()
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

        action = _await_action(renderer)
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
            result = _look_interact(renderer, game_map, cursor_x, cursor_y)
            if result == "quit":
                _clear_look_overlay(render)
                return "quit"
            continue

    _clear_look_overlay(render)
    esper.process(None)
    return "close"


def _draw_dialogue_menu(renderer: Renderer, game_map: GameMap, npc_ent: int) -> str:
    selected = 0
    options = ["Talk", "Trade", "Status", "Leave"]
    info_lines = _npc_info_lines(game_map, npc_ent)
    talk_line = ""

    while True:
        npc_name = entity_name(npc_ent, fallback="Unknown NPC")

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
        action = _await_action(renderer)

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
                player_ent = first_player_entity()
                if player_ent is not None and esper.has_component(npc_ent, Personality):
                    outcome = _player_talk(player_ent, npc_ent)
                    talk_line = f"{talk_line}  [{outcome}]"
                    info_lines = _npc_info_lines(game_map, npc_ent)
                continue
            if choice == "Trade":
                trade_choice = _draw_trade_menu(renderer, npc_ent)
                if trade_choice == "quit":
                    return "quit"
                info_lines = _npc_info_lines(game_map, npc_ent)
                talk_line = ""
            if choice == "Status":
                status_choice = _draw_info_screen(
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
    player_ent = first_player_entity()
    inventory_items: list[str] = []
    equipment_slots = default_equipment_slots()
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
    player_ent = first_player_entity()

    slot_count = 0
    item_count = 0
    if player_ent is not None:
        if esper.has_component(player_ent, Equipment):
            slot_count = len(default_equipment_slots())
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
            slot_names = list(default_equipment_slots().keys())
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
    player_ent = first_player_entity()
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
        player_ent = first_player_entity()
        if player_ent is not None:
            state.message = _craft_item(player_ent, _CRAFT_MENU[state.selected_idx])


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


def _draw_player_menu(renderer: Renderer, game_map: GameMap, start_tab: str = "inventory") -> tuple[str, str]:
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
            player_ent = first_player_entity()
            lines = _creature_status_lines(game_map, player_ent) if player_ent is not None else ["No player."]
            for idx, line in enumerate(lines):
                _draw_ui_text(renderer, x + 3, content_y + 1 + idx, line, _MENU_TEXT_COLOR, width - 6)
        else:
            label = _PLAYER_MENU_TABS[tab][1]
            _draw_ui_text(renderer, x + 3, content_y + 1, f"{label} - coming soon.", _MENU_MUTED_COLOR, width - 6)

        renderer.present()

        action = _await_action(renderer)
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


