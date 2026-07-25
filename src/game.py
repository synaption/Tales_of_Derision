"""Tales of Derision runtime.

Startup, and the pieces of one turn: wait for a command, interpret it (menus and
world interactions resolve here), then advance the world if it cost a turn. The
loop that strings those together is ``main.main`` -- this module deliberately
does not hide it. The game logic stays renderer-agnostic while the default
runtime uses pygame.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator
import os
from pathlib import Path
import sys
import time

import esper

from action import BASE_ACTION_COST
from components import Bed, BerryBush, Blueprint, Chest, Friendly, Player, Position, Stove, Tree, Well
from game_map import GameMap
from config import (
    ACTIVE_REGION_CATCHUP_STEPS_PER_INPUT, DEFAULT_WORLD_SEED, MAP_HEIGHT, MAP_WIDTH,
    WORLD_LAYOUT, WORLD_SETTLE_TURNS,
)
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
    run_world_generation,
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


def read_command_line() -> argparse.Namespace:
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


def _player_region_for_processors(player_xy: Position | tuple[int, int] | None) -> tuple[int, int] | None:
    if player_xy is None:
        return None
    processor = esper.get_processor(NpcAiProcessor) or esper.get_processor(FishAiProcessor)
    if processor is None:
        return None
    if isinstance(player_xy, Position):
        x, y = player_xy.x, player_xy.y
    else:
        x, y = player_xy
    return processor.scheduler.region_at(x, y)


def _current_target_region_turn() -> int | None:
    clock = world_clock()
    if clock is None:
        return None
    return clock.turn // BASE_ACTION_COST


def _catch_up_entered_region_cooperatively(renderer: Renderer, region_id: tuple[int, int] | None) -> None:
    """Bring an entered region current without one long blocking frame.

    Region entry is one of the few moments where the off-screen simulation must
    become authoritative before the next player command. Do that as a sequence of
    small deterministic scheduler advances, rendering between chunks so the UI
    keeps responding instead of freezing for the whole backlog.
    """
    if region_id is None:
        return
    target_turn = _current_target_region_turn()
    if target_turn is None:
        return
    processors = [
        processor
        for processor in (esper.get_processor(NpcAiProcessor), esper.get_processor(FishAiProcessor))
        if processor is not None
    ]
    if not processors:
        return
    while True:
        any_lagging = False
        for processor in processors:
            if processor.scheduler.region_turn.get(region_id, target_turn) < target_turn:
                any_lagging = True
                processor.scheduler.catch_up_region(
                    region_id,
                    target_turn,
                    max_advances=ACTIVE_REGION_CATCHUP_STEPS_PER_INPUT,
                )
        if not any_lagging:
            return
        esper.process(None)


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
    # Flora growth lags the same way and is paid down in the same spare cycles.
    flora = esper.get_processor(TreeGrowthProcessor)
    if flora is not None:
        flora.pump_flora(budget_seconds, player_xy, time.monotonic)


@dataclass(frozen=True)
class TurnIntent:
    """What the turn loop should do with the command the player just gave.

    ``GameSession.interpret`` resolves the command itself -- menus, dialogue,
    looting, chopping a tree -- so all that's left for the loop is the world:
    does time pass (``world_action``), does the frame need a redraw, or are we
    done playing.
    """

    quit: bool = False
    world_action: str | None = None
    redraw: bool = False


# Stop playing (quit key, or a menu that quit out).
QUIT_GAME = TurnIntent(quit=True)
# Fully handled, and the screen is already current: whatever ran (sleeping,
# placing a building) drew its own final frame.
HANDLED = TurnIntent()
# A free action -- opening a menu, turning to face a tile, chopping a tree.
# Nothing in the world moves; the frame just needs redrawing.
REDRAW = TurnIntent(redraw=True)


def _spend_turn(action: str) -> TurnIntent:
    """The command costs a turn: the systems run with ``action`` and the clock
    advances."""
    return TurnIntent(world_action=action)


class GameSession:
    """One live game: the renderer, the world, and the input state that has to
    survive between commands (held directions, which menu tab was last open,
    which region the player is standing in).

    Every method here is one step of the turn loop in ``main.main``; the loop
    stays there so the shape of a turn is readable in one screen.
    """

    def __init__(
        self,
        renderer: Renderer,
        game_map: GameMap,
        options: dict,
        save_file: Path,
        fallback_position: Position,
    ) -> None:
        self.renderer = renderer
        self.game_map = game_map
        self.options = options
        self.save_file = save_file
        self._fallback_position = fallback_position
        self._held_directions: set[str] = set()
        self._direction_pressed_order = {
            "move_up": -1,
            "move_down": -1,
            "move_left": -1,
            "move_right": -1,
        }
        self._press_order_counter = 0
        # The player menu (Tab) reopens on whatever tab you left it on.
        self._last_menu_tab = "inventory"
        self._player_region = _player_region_for_processors(first_player_position())
        self._idle_ticks = 0
        self._idle_is_animated = False

    # --- 1. input ---------------------------------------------------------

    def next_action(self) -> str | None:
        """Wait briefly for a command. ``None`` means the player didn't act --
        an idle tick, not a turn.

        The wait is a short poll rather than desktop's old fully-blocking one:
        true idle time (the player thinking, or away from the keyboard) is
        exactly when there's the most spare time to simulate off-screen regions.
        """
        # Live play has (re)rendered the game, so any menu backdrop snapshot is
        # stale; the next menu to open will re-capture.
        invalidate_backdrop = getattr(self.renderer, "invalidate_backdrop", None)
        if callable(invalidate_backdrop):
            invalidate_backdrop()
        # With an active status animation (swimming, on fire, ...) idle ticks
        # come on a shorter timeout so the identifiers cycle without input.
        self._idle_is_animated = player_is_animated(self.game_map) or bubbles_active()
        timeout = _STATUS_ANIM_POLL_SECONDS if self._idle_is_animated else _IDLE_POLL_SECONDS
        action = _await_action_or_idle(self.renderer, timeout)
        if action is not None:
            # A long idle spell must never dump one big simulation burst onto
            # the player's next turn, so the ramp resets the moment they act.
            self._idle_ticks = 0
        return action

    # --- 2. idle ----------------------------------------------------------

    def simulate_idle(self) -> None:
        """Spend an idle tick paying down off-screen simulation debt, ramping up
        the budget the longer the player goes without acting."""
        if self._idle_is_animated:
            _pump_background_regions(_IDLE_PUMP_BASE_BUDGET)
            esper.process(None)  # keep the status animation cycling
            return
        self._idle_ticks += 1
        _pump_background_regions(_idle_pump_budget(self._idle_ticks))

    # --- 3. interpret -----------------------------------------------------

    def interpret(self, action: str) -> TurnIntent:
        """Resolve one command, running any menu or interaction it opens, and
        report what the world should do about it."""
        if action == "quit":
            return QUIT_GAME
        if action in {"tile_scale_up", "tile_scale_down", "ui_layout_changed"}:
            return self._change_display(action)
        if action == "sleep":
            return self._sleep()
        if action == "look":
            self._held_directions.clear()
            if _look_mode(self.renderer, self.game_map) == "quit":
                return QUIT_GAME
            return REDRAW
        if action in _CARDINAL_ACTION_DELTAS:
            # Facing is free: pressing a direction turns the player, and the
            # held keys decide where "confirm" walks or swings.
            self._held_directions.add(action)
            self._press_order_counter += 1
            self._direction_pressed_order[action] = self._press_order_counter
            return REDRAW
        if action in _RELEASE_TO_DIRECTION:
            self._held_directions.discard(_RELEASE_TO_DIRECTION[action])
            return REDRAW
        if action == "confirm_action":
            # No direction held: wait in place, passing a turn (needs rise, NPCs
            # act) instead of a no-op refresh.
            return _spend_turn(self._faced_action() or WAIT_ACTION)
        if action == "menu_select":
            intent = self._interact_with_faced_tile()
            if intent is not None:
                return intent
            # Nothing to interact with: fall through and let the systems handle it.
        if action in {"open_menu", "open_inventory", "open_status"}:
            return self._open_player_menu(action)
        if action == "open_pause_menu":
            return self._open_pause_menu()
        return _spend_turn(action)

    # --- 4. apply ---------------------------------------------------------

    def redraw(self) -> None:
        """Refresh the frame without advancing the world."""
        esper.process(None)

    def take_turn(self, action: str) -> None:
        """Run the systems for one player action, then cooperatively settle a
        newly entered region before the next command."""
        esper.process(action)
        current_region = _player_region_for_processors(first_player_position())
        if current_region != self._player_region:
            _catch_up_entered_region_cooperatively(self.renderer, current_region)
        self._player_region = current_region

    # --- command handlers -------------------------------------------------

    def _faced_action(self) -> str | None:
        """The direction the player is facing, from the keys they're holding."""
        return _action_from_held_keys(self._held_directions, self._direction_pressed_order)

    def _change_display(self, action: str) -> TurnIntent:
        if action in {"tile_scale_up", "tile_scale_down"}:
            self.options["tile_scale"] = _next_scale(
                _coerce_scale(self.options.get("tile_scale", 1.0)),
                direction=1 if action == "tile_scale_up" else -1,
            )
            save_options(self.options)
            apply_fn = getattr(self.renderer, "apply_options", None)
            if callable(apply_fn):
                apply_fn(self.options)
        else:  # ui_layout_changed: the renderer already moved the panels.
            save_options(self.options)
        return REDRAW

    def _sleep(self) -> TurnIntent:
        """Sleep in a bed if one's at hand (warning first if it's not yours);
        otherwise pitch a camp."""
        self._held_directions.clear()
        nearby_bed = _bed_near_player()
        if nearby_bed is None:
            _sleep_player(self.renderer, in_camp=True)
            return HANDLED
        if not _confirm_if_owned_by_other(self.renderer, nearby_bed, "bed", "sleep here"):
            return REDRAW
        _sleep_player(self.renderer, in_camp=False)
        return HANDLED

    def _interact_with_faced_tile(self) -> TurnIntent | None:
        """Interact with whatever the player is facing. ``None`` means there was
        nothing there to interact with."""
        faced = self._faced_action()

        creature = _find_interaction_creature(faced)
        if creature is not None:
            self._held_directions.clear()
            if esper.has_component(creature, Friendly):
                # Friendlies: full dialogue (which shows their status).
                choice = _draw_dialogue_menu(self.renderer, self.game_map, creature)
            else:
                # Wild/hostile creatures: read-only examine of status.
                name = entity_name(creature, fallback="Creature")
                choice = _draw_info_screen(
                    self.renderer,
                    title=f"EXAMINE - {name}",
                    lines=_creature_status_lines(self.game_map, creature),
                    subtitle="What you can tell at a glance",
                )
            return QUIT_GAME if choice == "quit" else REDRAW

        corpse = _find_interaction_corpse(faced)
        if corpse is not None:
            loot_choice = _draw_loot_menu(self.renderer, corpse)
            self._held_directions.clear()
            return QUIT_GAME if loot_choice == "quit" else REDRAW

        chest = _find_adjacent_feature(faced, Chest)
        if chest is not None:
            self._held_directions.clear()
            if _confirm_if_owned_by_other(self.renderer, chest, "chest", "open it"):
                if _draw_loot_menu(self.renderer, chest) == "quit":
                    return QUIT_GAME
            return REDRAW

        # Environment features: chop a faced tree, drink from a faced well, or
        # cook at a faced stove. Each queues a log line and refreshes the frame
        # (a free action, like looting).
        player_ent = first_player_entity()
        if player_ent is None:
            return None

        # A faced blueprint ghost: haul wood into it, or raise it. Building is
        # labour -- a successful haul/raise spends a turn (the world simulates a
        # step); a no-op stays free.
        ghost = _find_adjacent_feature(faced, Blueprint)
        if ghost is not None:
            message, took_turn = _work_blueprint(ghost, player_ent, self.game_map)
            queue_message(message)
            return _spend_turn(WAIT_ACTION) if took_turn else REDRAW

        for component, interact in (
            (Tree, _chop_tree),
            (BerryBush, _harvest_bush),
            (Well, _drink_from_well),
            (Stove, _cook_at_stove),
        ):
            feature = _find_adjacent_feature(faced, component)
            if feature is not None:
                queue_message(interact(feature, player_ent))
                return REDRAW

        bed = _find_adjacent_feature(faced, Bed)
        if bed is not None:
            self._held_directions.clear()
            if not _confirm_if_owned_by_other(self.renderer, bed, "bed", "sleep here"):
                return REDRAW
            _sleep_player(self.renderer, in_camp=False)
            return HANDLED

        return None

    def _open_player_menu(self, action: str) -> TurnIntent:
        # Tab reopens on the last tab; I/C jump straight to a tab.
        if action == "open_inventory":
            start_tab = "inventory"
        elif action == "open_status":
            start_tab = "status"
        else:
            start_tab = self._last_menu_tab
        menu_choice, self._last_menu_tab = _draw_player_menu(self.renderer, self.game_map, start_tab)
        self._held_directions.clear()
        if menu_choice == "quit":
            return QUIT_GAME
        if menu_choice.startswith("place:"):
            # The player chose a buildable in the inventory; ask for a direction
            # and build it on that tile.
            _place_from_inventory(self.renderer, self.game_map, menu_choice[len("place:"):])
            return HANDLED
        return REDRAW

    def _open_pause_menu(self) -> TurnIntent:
        pause_choice = _draw_pause_menu(self.renderer, self.options)
        self._held_directions.clear()
        if pause_choice == "save_game":
            player_pos = first_player_position() or self._fallback_position
            save_game(self.game_map, self.save_file, player_pos, seed=world_rng().seed)
        elif pause_choice == "quit":
            return QUIT_GAME
        return REDRAW


def _register_processors(game_map: GameMap, combat_sfx: CombatSfxPlayer) -> None:
    """Install the simulation systems, in the order one turn runs them."""
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
    esper.add_processor(HousingProcessor(game_map, live_region_only=True), priority=0)
    esper.add_processor(
        NpcAiProcessor(game_map, max_entry_catchup_advances=ACTIVE_REGION_CATCHUP_STEPS_PER_INPUT),
        priority=0,
    )
    esper.add_processor(
        FishAiProcessor(game_map, max_entry_catchup_advances=ACTIVE_REGION_CATCHUP_STEPS_PER_INPUT),
        priority=0,
    )
    esper.add_processor(NeedsProcessor(game_map), priority=0)
    # Ticks registered status effects (fire, poison, ...). A no-op until an
    # effect declares behaviour; the seam lives in content.effects.
    esper.add_processor(EffectsProcessor(), priority=0)
    esper.add_processor(TreeGrowthProcessor(game_map), priority=0)
    esper.add_processor(ReproductionProcessor(), priority=0)


@contextmanager
def game_session(args: argparse.Namespace) -> Iterator[GameSession | None]:
    """Bring up a playable game and hand it to the turn loop, tearing the
    renderer and audio back down on the way out.

    Yields ``None`` when there's nothing to play: the player backed out of the
    title screen, or ``--screenshot`` captured its one frame and is done.
    """
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

    # The world itself is built by the startup flow: loading a save reconstructs the
    # size it was made at, and a new game asks the player how big an archipelago to
    # generate, so there is nothing to build until one of those has been answered.
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
            )
            if not startup_ok or game_map is None or player_position is None:
                yield None
                return

            if args.screenshot is None:
                pygame_module = start_background_music(options)
                combat_sfx = CombatSfxPlayer(pygame_module, options)

            rat_count = _setup_world(game_map, player_position, rat_flood=args.rat_flood)
            if args.rat_flood:
                print(f"Rat flood mode enabled: spawned {rat_count} cave rats.", file=sys.stderr)
            _register_processors(game_map, combat_sfx)

            # Pre-simulate the world behind a "Generating world..." screen so the
            # startup building boom happens during loading, not as lag on the first
            # turns of play. Runs before the RenderProcessor is registered, so these
            # settle turns advance the world without drawing the game. Skipped for
            # screenshot capture (needs an immediate frame) and the rat-flood stress
            # test (no villagers to settle).
            if args.screenshot is None and not args.rat_flood:
                if not run_world_generation(renderer, WORLD_SETTLE_TURNS):
                    yield None
                    return

            esper.add_processor(RenderProcessor(renderer, game_map), priority=0)

            esper.process()  # initial frame
            if args.screenshot is not None:
                _capture_frame_screenshot(renderer, args.screenshot)
                yield None
                return

            yield GameSession(
                renderer,
                game_map,
                options,
                selected_save_file,
                fallback_position=player_position,
            )
    finally:
        stop_background_music(pygame_module)
