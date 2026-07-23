"""pyRL2 entry point.

Turn loop: show title/menu, then block for an action, run the systems, repeat.
The game logic stays renderer-agnostic while the default runtime uses pygame.
"""
import argparse
import os
from pathlib import Path
import sys
import time

import esper

from action import BASE_ACTION_COST
from components import Bed, BerryBush, Blueprint, Chest, Friendly, Player, Position, Stove, Tree, Well
from game_map import GameMap
from config import DEFAULT_WORLD_SEED, MAP_HEIGHT, MAP_WIDTH
from queries import entity_name, first_player_entity
from worldgen import _setup_world
# Used by the turn loop below. Tests import these helpers from ``interactions`` directly.
from interactions import (
    _CARDINAL_ACTION_DELTAS,
    _action_from_held_keys,
    _bed_near_player,
    _chop_tree,
    _cook_at_stove,
    _creature_status_lines,
    _drink_from_well,
    _find_adjacent_feature,
    _find_interaction_corpse,
    _find_interaction_creature,
    _harvest_bush,
    _work_blueprint,
)
from persistence import (
    DEFAULT_SAVE_FILE,
    bootstrap_files,
    first_player_position,
    load_options,
    save_game,
    save_options,
)
from rng import set_world_rng, world_rng
from ui import (
    _capture_frame_screenshot,
    _coerce_scale,
    _confirm_if_owned_by_other,
    _draw_dialogue_menu,
    _draw_info_screen,
    _draw_loot_menu,
    _draw_pause_menu,
    _draw_player_menu,
    _look_mode,
    _next_scale,
    _place_from_inventory,
    _run_startup_flow,
    _sleep_player,
)
from audio import CombatSfxPlayer, start_background_music, stop_background_music
from content.effects import EffectsProcessor
from renderer.base import Renderer
from renderer.pygame_renderer import PygameRenderer
from systems import FishAiProcessor, HousingProcessor, MovementProcessor, NeedsProcessor, NpcAiProcessor, RenderProcessor, ReproductionProcessor, TimeProcessor, TreeGrowthProcessor, WAIT_ACTION, bubbles_active, player_is_animated, queue_message, world_clock

_RELEASE_TO_DIRECTION = {
    "release_up": "move_up",
    "release_down": "move_down",
    "release_left": "move_left",
    "release_right": "move_right",
}


# How often the idle/animation poll loop re-checks for input while waiting, so a
# genuinely idle wait (status animations, background catch-up) doesn't busy-spin
# the CPU. ~60Hz keeps input latency imperceptible.
_IDLE_POLL_INTERVAL = 1 / 60


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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="world seed for a new game (same seed -> identical world); a loaded "
        "save uses its stored seed instead",
    )
    return parser.parse_args()


# How often to re-render while the player has an active status animation, so the
# sub-second identifiers (e.g. the 0.5s "~") are visible without waiting for
# input. Shorter than the shortest status frame; only used while animating.
_STATUS_ANIM_POLL_SECONDS = 0.2


def _await_action_or_idle(renderer: Renderer, idle_timeout: float) -> str | None:
    """Like ``_await_action`` but returns ``None`` after ``idle_timeout`` seconds
    with no input -- an animation tick. Polls without blocking so idle status
    animations keep playing while we wait, sleeping a frame between polls so the
    wait doesn't busy-spin the CPU."""
    poll_nonblocking = getattr(renderer, "poll_action_nonblocking", None)
    poll = poll_nonblocking if callable(poll_nonblocking) else renderer.poll_action
    deadline = time.monotonic() + idle_timeout
    while True:
        action = poll()
        if action is not None:
            return action
        if time.monotonic() >= deadline:
            return None
        time.sleep(_IDLE_POLL_INTERVAL)


# How often the main loop wakes up with no input at all, so genuinely idle
# time (the player thinking, or away from the keyboard) can be spent paying
# down region simulation debt in the background -- desktop used to block here
# with zero idle CPU use; this trades a little of that for background progress.
_IDLE_POLL_SECONDS = 0.05
# Background pump budget (seconds) for a single idle tick: starts tiny (barely
# more than the per-turn budget already spent inside NpcAiProcessor/
# FishAiProcessor) and ramps up the longer the player goes without acting, up
# to a ceiling that still can't turn into a perceptible stall. Resets to the
# base the instant a real action arrives, so a long idle spell can never dump
# one big burst onto the player's next turn.
_IDLE_PUMP_BASE_BUDGET = 0.004
_IDLE_PUMP_MAX_BUDGET = 0.05
_IDLE_PUMP_RAMP = 1.4


def _idle_pump_budget(idle_ticks: int) -> float:
    return min(_IDLE_PUMP_MAX_BUDGET, _IDLE_PUMP_BASE_BUDGET * (_IDLE_PUMP_RAMP**idle_ticks))


def _pump_background_regions(budget_seconds: float) -> None:
    """Spend up to ``budget_seconds`` advancing the nearest lagging region for
    each region-aware processor, closest to the player first. Safe to call
    often -- with nothing lagging it's just a couple of empty dict scans."""
    clock = world_clock()
    if clock is None:
        return
    player_xy: tuple[int, int] | None = None
    for _ent, (pos, _p) in esper.get_components(Position, Player):
        player_xy = (pos.x, pos.y)
        break
    for processor_type in (NpcAiProcessor, FishAiProcessor):
        processor = esper.get_processor(processor_type)
        if processor is None:
            continue
        player_region = (
            processor.scheduler.region_at(player_xy[0], player_xy[1])
            if player_xy is not None
            else None
        )
        # The scheduler counts in whole region-turns; the clock is in TU.
        target_region_turn = clock.turn // BASE_ACTION_COST
        processor.scheduler.pump_background(
            budget_seconds, player_region, target_region_turn, time.monotonic
        )


def main() -> None:
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
    combat_sfx = CombatSfxPlayer(None, options)

    game_map = GameMap(MAP_WIDTH, MAP_HEIGHT)
    player_position = Position(MAP_WIDTH // 2, MAP_HEIGHT // 2)
    startup_save_file = args.save_file
    if args.screenshot is not None and startup_save_file is None:
        startup_save_file = DEFAULT_SAVE_FILE

    # Install the world seed before any worldgen. For now the whole game uses one
    # fixed seed -- every game (new or loaded) regenerates the same world -- unless
    # --seed overrides it. Saves still record the seed for when we later restore
    # per-save worlds; that stored seed is intentionally ignored for now.
    set_world_rng(args.seed if args.seed is not None else DEFAULT_WORLD_SEED)

    try:
        with PygameRenderer(options=options) as renderer:
            startup_ok, game_map, player_position, selected_save_file = _run_startup_flow(
                renderer,
                startup_save_file,
                game_map,
                player_position,
            )
            if not startup_ok:
                return

            if args.screenshot is None:
                pygame_module = start_background_music(options)
                combat_sfx = CombatSfxPlayer(pygame_module, options)

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
            # Ticks registered status effects (fire, poison, ...). A no-op until an
            # effect declares behaviour; the seam lives in content.effects.
            esper.add_processor(EffectsProcessor(), priority=0)
            esper.add_processor(TreeGrowthProcessor(game_map), priority=0)
            esper.add_processor(ReproductionProcessor(), priority=0)
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
            idle_ticks = 0
            while True:
                # Live play has (re)rendered the game, so any menu backdrop
                # snapshot is stale; the next menu to open will re-capture.
                if callable(invalidate_backdrop):
                    invalidate_backdrop()
                # When the player has an active status animation (swimming, on
                # fire, ...), poll on a short timeout and re-render on each idle
                # tick so the identifiers cycle without input.
                if player_is_animated(game_map) or bubbles_active():
                    action = _await_action_or_idle(renderer, _STATUS_ANIM_POLL_SECONDS)
                    if action is None:
                        _pump_background_regions(_IDLE_PUMP_BASE_BUDGET)
                        esper.process(None)
                        continue
                else:
                    # Otherwise poll on a short timeout too, rather than desktop's
                    # old fully-blocking wait: true idle time (the player thinking,
                    # or away from the keyboard) is exactly when there's the most
                    # spare time to pay down region simulation debt in the
                    # background, ramping up the longer nothing happens.
                    action = _await_action_or_idle(renderer, _IDLE_POLL_SECONDS)
                    if action is None:
                        idle_ticks += 1
                        _pump_background_regions(_idle_pump_budget(idle_ticks))
                        continue
                idle_ticks = 0
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
                        if _confirm_if_owned_by_other(renderer, nearby_bed, "bed", "sleep here"):
                            _sleep_player(renderer, in_camp=False)
                        else:
                            esper.process(None)
                    else:
                        _sleep_player(renderer, in_camp=True)
                    continue
                if action == "look":
                    held_directions.clear()
                    look_choice = _look_mode(renderer, game_map)
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
                            choice = _draw_dialogue_menu(renderer, game_map, interact_creature)
                        else:
                            # Wild/hostile creatures: read-only examine of status.
                            name = entity_name(interact_creature, fallback="Creature")
                            choice = _draw_info_screen(
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
                        loot_choice = _draw_loot_menu(renderer, interact_corpse)
                        held_directions.clear()
                        if loot_choice == "quit":
                            break
                        esper.process(None)
                        continue

                    interact_chest = _find_adjacent_feature(interact_action, Chest)
                    if interact_chest is not None:
                        held_directions.clear()
                        if _confirm_if_owned_by_other(renderer, interact_chest, "chest", "open it"):
                            loot_choice = _draw_loot_menu(renderer, interact_chest)
                            if loot_choice == "quit":
                                break
                        esper.process(None)
                        continue

                    # Environment features: chop a faced tree, drink from a faced
                    # well, or cook at a faced stove. Each queues a log line and
                    # refreshes the frame (a free action, like looting).
                    player_ent = first_player_entity()
                    if player_ent is not None:
                        # A faced blueprint ghost: haul wood into it, or raise it.
                        # Building is labour -- a successful haul/raise spends a
                        # turn (the world simulates a step); a no-op stays free.
                        interact_ghost = _find_adjacent_feature(interact_action, Blueprint)
                        if interact_ghost is not None:
                            message, took_turn = _work_blueprint(interact_ghost, player_ent, game_map)
                            queue_message(message)
                            esper.process(WAIT_ACTION if took_turn else None)
                            continue

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
                            if _confirm_if_owned_by_other(renderer, interact_bed, "bed", "sleep here"):
                                _sleep_player(renderer, in_camp=False)
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
                    menu_choice, last_menu_tab = _draw_player_menu(renderer, game_map, start_tab)
                    held_directions.clear()
                    if menu_choice == "quit":
                        break
                    if menu_choice.startswith("place:"):
                        # The player chose a buildable in the inventory; ask for a
                        # direction and build it on that tile.
                        _place_from_inventory(renderer, game_map, menu_choice[len("place:"):])
                        continue
                    esper.process(None)
                    continue
                if action == "open_pause_menu":
                    pause_choice = _draw_pause_menu(renderer, options)
                    held_directions.clear()
                    if pause_choice == "save_game":
                        player_pos = first_player_position() or player_position
                        save_game(game_map, selected_save_file, player_pos, seed=world_rng().seed)
                    elif pause_choice == "quit":
                        break
                    esper.process(None)
                    continue
                esper.process(action)
    finally:
        stop_background_music(pygame_module)


def run() -> None:
    """Entry point for desktop runs (`python3 src/main.py`)."""
    main()


if __name__ == "__main__":
    run()
