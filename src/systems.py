"""ECS processors (systems).

esper 3.x uses module-level state and esper.Processor subclasses whose
process() receives whatever args are passed to esper.process().
"""
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
import textwrap
import time

import esper

from components import Actor, Age, Asleep, Bed, BerryBush, Blueprint, BlocksMovement, Camp, Chest, ConstructionSite, Corpse, Deer, Dialogue, Diet, Enemy, Equipment, Family, Fish, Friendly, Furniture, Gender, Home, Inventory, Mating, Meat, Name, Needs, NPC, OnFire, Owned, Personality, Player, Position, Pregnant, Relationships, Renderable, Resident, Sapling, Seaweed, Stove, Tree, Vision, WorldClock
from game_map import GameMap, LAND_HEIGHT, LAND_WIDTH
from items import RAW_MEAT, WOOD, cook_meat, hunger_restored, is_cooked_meat, is_raw_meat
from action import BASE_ACTION_COST, action_cost
from onymancer import make_onymancer
from regions import RegionId, RegionScheduler, all_region_ids, in_region_with_margin
from renderer.base import Renderer, memory_color
from rng import world_rng

_ACTION_DELTAS = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
    "move_up_left": (-1, -1),
    "move_up_right": (1, -1),
    "move_down_left": (-1, 1),
    "move_down_right": (1, 1),
}

# The "wait" action passes a turn without moving the player. Time-advancing
# systems (needs, NPC AI) tick on any of these; movement/combat only on a move.
WAIT_ACTION = "wait"
_TURN_ACTIONS = set(_ACTION_DELTAS) | {WAIT_ACTION}

_DIR_TO_ARROW = {
    (-1, -1): "↖",
    (0, -1): "↑",
    (1, -1): "↗",
    (-1, 0): "←",
    (0, 0): "•",
    (1, 0): "→",
    (-1, 1): "↙",
    (0, 1): "↓",
    (1, 1): "↘",
}

_ARROW_TO_WORD = {
    "↖": "northwest",
    "↑": "north",
    "↗": "northeast",
    "←": "west",
    "•": "here",
    "→": "east",
    "↙": "southwest",
    "↓": "south",
    "↘": "southeast",
}

_WALL_BROWN = (124, 88, 56)
_WATER_BLUE = (64, 118, 190)
_DOOR_BROWN = (150, 106, 58)
_WINDOW_CYAN = (128, 186, 206)

# Tile memory ("fog of war"): a tile the player has seen but no longer has line
# of sight to is drawn desaturated (static terrain + scenery), while anything
# that can move or be picked up (NPCs, corpses, loot) is hidden until seen again.
# ``memory_color`` and its tuning constants live on the renderer seam so the
# backend applies the identical transform to whole sprites/surfaces.
_memory_color = memory_color


# Component types that mark an entity as static scenery worth remembering. This
# is a deliberate positive whitelist: creatures, the player, corpses and loot
# (chests) are left off so a remembered tile can never lie about where something
# takeable or moving is *now* -- you only ever remember terrain and fixtures.
_MEMORABLE_SCENERY: tuple[type, ...] = (
    Tree,
    BerryBush,
    Sapling,
    Seaweed,
    Furniture,
    Bed,
    Stove,
    Blueprint,
)


def is_memorable_scenery(ent: int) -> bool:
    """Whether an entity is static enough to leave an imprint on a tile the
    player has explored but can no longer see. NPCs, the player, corpses and
    loot are never memorable (they return ``False``)."""
    return any(esper.has_component(ent, comp) for comp in _MEMORABLE_SCENERY)
_TREE_GREEN = (58, 138, 66)
_SAPLING_GREEN = (120, 176, 90)
_BUSH_GREEN = (86, 140, 78)   # a bush that has been picked bare
_SEAWEED_GREEN = (54, 148, 116)  # must match the ocean seaweed spawned in main
_BERRY_RED = (200, 74, 96)    # a bush heavy with ripe berries
_BUSH_GLYPH = "%"


def _set_bush_appearance(ent: int, has_berries: bool) -> None:
    """Colour a berry bush by whether it's ripe: reddish with berries, plain
    green once picked. The glyph stays ``%``."""
    if not esper.has_component(ent, Renderable):
        return
    rend = esper.component_for_entity(ent, Renderable)
    rend.glyph = _BUSH_GLYPH
    rend.fg = _BERRY_RED if has_berries else _BUSH_GREEN


def pick_berries(bush_ent: int, clock: WorldClock | None) -> bool:
    """Take a bush's berries if it is ripe. Marks it bare and stamps the harvest
    time so the berries regrow 7 days later. Returns True if berries were taken."""
    if not esper.has_component(bush_ent, BerryBush):
        return False
    bush = esper.component_for_entity(bush_ent, BerryBush)
    if not bush.has_berries:
        return False
    bush.has_berries = False
    bush.harvested_turn = clock.turn if clock is not None else 0
    _set_bush_appearance(bush_ent, False)
    return True

# A section must be at least this big for the section camera to engage; below it
# (tiny test maps) the renderer keeps its plain centred camera.
_MIN_SECTION_W = 12
_MIN_SECTION_H = 8

# The render section (40x20) -- only the one the player stands in is drawn.
# The coarser 120x60 simulation region (a 3x3 block of render sections) that
# region-aware processors use is defined in regions.py.
_RENDER_SECTION_W = LAND_WIDTH // 3   # 40
_RENDER_SECTION_H = LAND_HEIGHT // 3  # 20

# --- Status identifier animation -------------------------------------------
# A character's rendered tile cycles through its own glyph followed by an
# identifier for each active status, each shown for a configurable length of
# time, then repeats. Statuses stack sequentially: e.g. swimming + on fire ->
# own tile (1.0s) -> "~" (0.5s) -> red "F" (0.5s) -> loop.
#
# How long the character's own tile shows before the status identifiers cycle.
_STATUS_BASE_SECONDS = 1.0
# status name -> (identifier glyph, fg override or None to keep the character's
# colour, seconds shown). Tune these freely.
_STATUS_DISPLAY: dict[str, tuple[str, tuple[int, int, int] | None, float]] = {
    "swimming": ("~", None, 0.5),
    "on_fire": ("F", (224, 74, 44), 0.5),
    "asleep": ("Z", (150, 170, 220), 0.6),
}
# Order the identifiers cycle in, after the base tile.
_STATUS_ORDER: tuple[str, ...] = ("swimming", "on_fire", "asleep")

# Human-readable status names for status/examine screens.
_STATUS_LABELS: dict[str, str] = {
    "swimming": "Swimming",
    "on_fire": "On fire",
    "asleep": "Asleep",
}


def status_label(name: str) -> str:
    return _STATUS_LABELS.get(name, name.replace("_", " ").capitalize())


def active_statuses(game_map: GameMap, ent: int, pos: Position) -> list[str]:
    """The status keys currently affecting an entity, in display order. Swimming
    is derived from standing on water; on_fire from the OnFire component."""
    active: set[str] = set()
    if game_map.is_water(pos.x, pos.y):
        active.add("swimming")
    if esper.has_component(ent, OnFire):
        active.add("on_fire")
    if esper.has_component(ent, Asleep):
        active.add("asleep")
    return [name for name in _STATUS_ORDER if name in active]


def player_is_animated(game_map: GameMap) -> bool:
    """True when the player has any active timed status, so the turn loop knows
    to keep re-rendering while idle for the animation to play."""
    for ent, (pos, _player) in esper.get_components(Position, Player):
        return bool(active_statuses(game_map, ent, pos))
    return False


# --- Day / night cycle -----------------------------------------------------
# A day runs dawn -> day -> dusk -> night as a fraction of ``WorldClock``. The
# boundaries below are fractions of ``day_length``; tune them freely.
_PHASE_BOUNDS: tuple[tuple[float, str], ...] = (
    (0.10, "Dawn"),
    (0.45, "Day"),
    (0.55, "Day"),
    (0.70, "Dusk"),
    (1.00, "Night"),
)
# Phases counted as "night" for the tiredness multiplier and the screen tint.
_NIGHTFALL_PHASES: frozenset[str] = frozenset({"Dusk", "Night"})
# Tiredness climbs this much faster while it is night.
_NIGHT_TIREDNESS_MULTIPLIER = 2.0
# Colour of the darkening overlay drawn over the world after dusk.
_NIGHT_TINT = (12, 20, 54)


def world_clock() -> WorldClock | None:
    """The singleton world clock, or ``None`` on maps that never created one
    (e.g. focused unit tests)."""
    for _ent, (clock,) in esper.get_components(WorldClock):
        return clock
    return None


def _current_turn() -> int:
    clock = world_clock()
    return clock.turn if clock is not None else 0


def _player_entity() -> int | None:
    for ent, (_player,) in esper.get_components(Player):
        return ent
    return None


def _current_region_turn() -> int:
    """The world clock (in TU) as a count of whole baseline turns -- the unit the
    region scheduler steps in. One region-turn == ``BASE_ACTION_COST`` TU, so a
    baseline player action (100 TU) advances every region by exactly one turn,
    preserving the old lockstep cadence."""
    return _current_turn() // BASE_ACTION_COST


def day_fraction(clock: WorldClock) -> float:
    """Where we are in the current day, in ``[0.0, 1.0)``."""
    if clock.day_length <= 0:
        return 0.0
    return (clock.turn % clock.day_length) / clock.day_length


def time_phase(clock: WorldClock | None) -> str:
    """Human-readable phase of day (``"Dawn"``/``"Day"``/``"Dusk"``/``"Night"``).
    A missing clock reads as plain ``"Day"``."""
    if clock is None:
        return "Day"
    fraction = day_fraction(clock)
    for upper, label in _PHASE_BOUNDS:
        if fraction < upper:
            return label
    return _PHASE_BOUNDS[-1][1]


def is_night(clock: WorldClock | None) -> bool:
    """True during the phases when tiredness climbs faster and the world dims."""
    return time_phase(clock) in _NIGHTFALL_PHASES


# --- Calendar --------------------------------------------------------------
# A year is 4 months, each 4 weeks of 7 days -> 4*4*7 = 112 days.
_DAYS_PER_WEEK = 7
_WEEKS_PER_MONTH = 4
_MONTHS_PER_YEAR = 4
_DAYS_PER_MONTH = _DAYS_PER_WEEK * _WEEKS_PER_MONTH  # 28
_DAYS_PER_YEAR = _DAYS_PER_MONTH * _MONTHS_PER_YEAR  # 112


def calendar(clock: WorldClock) -> tuple[int, int, int, int, int, int]:
    """Break the clock down into ``(year, month, week, day, hour, minute)``.

    Year/month/week/day are 1-based for display; hour/minute are the clock time
    within the current day (24h), derived from the fraction of the day elapsed.
    """
    day_length = max(1, clock.day_length)
    total_days = clock.turn // day_length
    day_of_year = total_days % _DAYS_PER_YEAR
    day_in_month = day_of_year % _DAYS_PER_MONTH
    year = total_days // _DAYS_PER_YEAR + 1
    month = day_of_year // _DAYS_PER_MONTH + 1
    week = day_in_month // _DAYS_PER_WEEK + 1
    day = day_in_month % _DAYS_PER_WEEK + 1
    minutes_into_day = int(day_fraction(clock) * 24 * 60)
    return year, month, week, day, minutes_into_day // 60, minutes_into_day % 60


def format_datetime(clock: WorldClock | None) -> str:
    """Compact date + clock-time + phase string for the status line, e.g.
    ``"Y1 M2 W3 D4 13:45 Night"``. A missing clock reads as ``"Day"``."""
    if clock is None:
        return "Day"
    year, month, week, day, hour, minute = calendar(clock)
    return f"Y{year} M{month} W{week} D{day} {hour:02d}:{minute:02d} {time_phase(clock)}"


def current_day(clock: WorldClock | None) -> int:
    """The absolute day number the clock is on (0-based, counting from turn 0).
    Used to gate once-per-day behaviour like a man's mating cooldown."""
    if clock is None:
        return 0
    return clock.turn // max(1, clock.day_length)


# --- Aging -----------------------------------------------------------------
# A person is an adult (may court, marry, and reproduce) from this age on. Age is
# derived from ``Age.born_turn`` and the clock, so nobody is ticked each turn.
_ADULT_AGE_YEARS = 17


def age_years(ent: int, clock: WorldClock | None) -> float:
    """A person's age in years, from their ``Age.born_turn`` and the world clock.
    Returns 0.0 for a being with no ``Age`` or no clock."""
    if clock is None or not esper.has_component(ent, Age):
        return 0.0
    born = esper.component_for_entity(ent, Age).born_turn
    turns_per_year = _DAYS_PER_YEAR * max(1, clock.day_length)
    return max(0.0, (clock.turn - born) / turns_per_year)


def is_adult(ent: int, clock: WorldClock | None) -> bool:
    """Whether ``ent`` has reached adulthood (``_ADULT_AGE_YEARS``)."""
    return age_years(ent, clock) >= _ADULT_AGE_YEARS


def born_turn_for_age(clock: WorldClock | None, age: float) -> int:
    """The ``Age.born_turn`` that makes someone ``age`` years old right now -- used
    to give the starting cast their ages (adults and children alike)."""
    turn = clock.turn if clock is not None else 0
    day_length = clock.day_length if clock is not None else 240
    return int(turn - age * _DAYS_PER_YEAR * max(1, day_length))


def night_overlay_alpha(clock: WorldClock | None) -> int:
    """Alpha (0-255) for the darkening night overlay: none by day, a light wash
    at dusk/dawn, deeper at night. Zero means draw nothing."""
    phase = time_phase(clock)
    if phase == "Night":
        return 110
    if phase in ("Dusk", "Dawn"):
        return 55
    return 0


class TimeProcessor(esper.Processor):
    """Advances the world clock one tick per real turn and announces the day's
    phase changes (``"Night falls."`` and friends) in the log."""

    _PHASE_MESSAGES = {
        "Dawn": "The sky lightens as dawn breaks.",
        "Day": "The sun climbs into the sky.",
        "Dusk": "The light fades as dusk settles in.",
        "Night": "Night falls over the land.",
    }

    def __init__(self) -> None:
        self._last_phase: str | None = None

    def process(self, action: str | None = None) -> None:
        if action not in _TURN_ACTIONS:
            return
        clock = world_clock()
        if clock is None:
            return
        # Advance world time by how long the player's action takes, not a flat 1:
        # a quicker action costs fewer TU (so the player gets relatively more of
        # them per day), a slower one costs more. A baseline action is exactly
        # BASE_ACTION_COST, preserving the old one-turn-per-action cadence.
        player = _player_entity()
        clock.turn += action_cost(player, action) if player is not None else BASE_ACTION_COST
        phase = time_phase(clock)
        if phase != self._last_phase:
            if self._last_phase is not None:
                _push_turn_event(self._PHASE_MESSAGES.get(phase, ""))
            self._last_phase = phase


# --- Sleep, camps, and waking ----------------------------------------------
_CAMP_GLYPH = "^"
_CAMP_ORANGE = (210, 130, 60)


def _camp_at(x: int, y: int) -> int | None:
    for ent, (pos, _camp) in esper.get_components(Position, Camp):
        if (pos.x, pos.y) == (x, y):
            return ent
    return None


def go_to_sleep(ent: int, in_camp: bool) -> None:
    """Put a character to sleep. Camping pitches a campfire on its tile (if one
    isn't already there); sleeping at home just marks it ``Asleep``."""
    if not esper.has_component(ent, Asleep):
        esper.add_component(ent, Asleep(in_camp=in_camp))
    if in_camp and esper.has_component(ent, Position):
        pos = esper.component_for_entity(ent, Position)
        if _camp_at(pos.x, pos.y) is None:
            esper.create_entity(
                Position(pos.x, pos.y),
                Renderable(_CAMP_GLYPH, fg=_CAMP_ORANGE),
                Name("Camp"),
                Camp(),
            )


def wake_up(ent: int) -> None:
    """Wake a sleeper: drop the ``Asleep`` tag, break any camp it pitched, and
    (for the player) log that it rose. Safe to call on an already-awake entity."""
    in_camp = False
    if esper.has_component(ent, Asleep):
        in_camp = esper.component_for_entity(ent, Asleep).in_camp
        esper.remove_component(ent, Asleep)
    if in_camp and esper.has_component(ent, Position):
        pos = esper.component_for_entity(ent, Position)
        camp_ent = _camp_at(pos.x, pos.y)
        if camp_ent is not None:
            esper.delete_entity(camp_ent, immediate=True)
    if esper.has_component(ent, Player):
        _push_turn_event("You wake, feeling rested.")


# Tiredness (percent of max) at which an NPC drops what it's doing to sleep, and
# the level past which it gives up on reaching home and camps where it stands.
_SLEEP_THRESHOLD = 70.0
_EXHAUSTED_THRESHOLD = 95.0


# --- Houses: detection cache, furnishing, and building ---------------------
_FURNITURE_TAN = (150, 116, 74)
_CHEST_BROWN = (168, 120, 66)
_BOOKSHELF_RED = (150, 92, 70)


def houses_for(game_map: GameMap, cache: dict) -> list[frozenset[tuple[int, int]]]:
    """Enclosed house interiors, recomputed only when the map has changed since
    the cache was filled. ``cache`` is any dict the caller keeps around."""
    revision = getattr(game_map, "revision", 0)
    if cache.get("revision") != revision:
        cache["revision"] = revision
        cache["houses"] = game_map.find_enclosed_rooms()
    return cache["houses"]


def _bed_in_interior(interior: frozenset[tuple[int, int]]) -> int | None:
    for ent, (pos, _bed) in esper.get_components(Position, Bed):
        if (pos.x, pos.y) in interior:
            return ent
    return None


def bed_owner(bed_ent: int) -> int | None:
    """The person who owns a bed (and thus the house), or ``None`` if it's
    unowned. An owner whose entity no longer exists is treated as unowned."""
    if bed_ent is not None and esper.has_component(bed_ent, Owned):
        owner = esper.component_for_entity(bed_ent, Owned).owner
        if esper.entity_exists(owner):
            return owner
    return None


def set_bed_owner(bed_ent: int, owner: int) -> None:
    if esper.has_component(bed_ent, Owned):
        esper.component_for_entity(bed_ent, Owned).owner = owner
    else:
        esper.add_component(bed_ent, Owned(owner))


def owned_bed_of(ent: int) -> int | None:
    """The bed a person owns (their house), or ``None`` if they own none."""
    for bed_ent, (owned, _bed) in esper.get_components(Owned, Bed):
        if owned.owner == ent and esper.entity_exists(bed_ent):
            return bed_ent
    return None


def house_is_owned(interior: frozenset[tuple[int, int]]) -> bool:
    """A house is owned when its bed belongs to someone (a living owner)."""
    bed_ent = _bed_in_interior(interior)
    return bed_ent is not None and bed_owner(bed_ent) is not None


def set_house_ownership(interior: frozenset[tuple[int, int]], owner: int) -> None:
    """Assign a house's ownable furnishings -- its bed and any chests -- to
    ``owner``, so trespassers get warned before using them."""
    for ent, (pos, _bed) in esper.get_components(Position, Bed):
        if (pos.x, pos.y) in interior:
            set_bed_owner(ent, owner)
    for ent, (pos, _chest) in esper.get_components(Position, Chest):
        if (pos.x, pos.y) in interior:
            set_bed_owner(ent, owner)


# Furnishing spec: (glyph, colour, name, blocks, factory-for-extra-components).
# Order matters -- the bed is placed first (farthest from the door), the rest
# fill remaining interior tiles.
_FURNISHINGS: list[tuple[str, tuple[int, int, int], str, bool, Callable[[], list[object]]]] = [
    ("=", (156, 116, 72), "Bed", False, lambda: [Bed()]),
    ("#", (120, 120, 128), "Oven", True, lambda: [Stove()]),
    ("n", _CHEST_BROWN, "Chest", True, lambda: [Chest(), Inventory(items=["Bread", "Wood"])]),
    ("T", _FURNITURE_TAN, "Table", True, lambda: [Furniture("table")]),
    ("W", _FURNITURE_TAN, "Wardrobe", True, lambda: [Furniture("wardrobe")]),
    ("B", _BOOKSHELF_RED, "Bookshelf", True, lambda: [Furniture("bookshelf")]),
]


def _interior_reachable(
    interior_set: set[tuple[int, int]],
    seeds: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Interior tiles reachable (8-connected, matching movement) from the door
    ``seeds`` without stepping on a ``blocked`` (furniture) tile."""
    frontier = [t for t in seeds if t not in blocked]
    seen = set(frontier)
    queue: deque[tuple[int, int]] = deque(frontier)
    while queue:
        cx, cy = queue.popleft()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                t = (cx + dx, cy + dy)
                if t in interior_set and t not in blocked and t not in seen:
                    seen.add(t)
                    queue.append(t)
    return seen


def furnish_house(game_map: GameMap, interior: frozenset[tuple[int, int]]) -> tuple[int, int] | None:
    """Populate a house interior with a bed, oven, chest, table, wardrobe, and
    bookshelf on distinct floor tiles, keeping the doorway clear. Returns the bed
    tile (a resident's sleep spot) or ``None`` if the room was too small.

    Blocking furniture is only placed where it keeps the whole interior -- the bed
    especially -- reachable from the door, so a resident can always walk in and
    lie down (never sealed behind its own furniture)."""
    interior_set = set(interior)
    door_adjacent: set[tuple[int, int]] = set()
    for (x, y) in interior:
        if any(game_map.tile_at(nx, ny) == game_map.DOOR for nx, ny in game_map.neighbors_4(x, y)):
            door_adjacent.add((x, y))

    occupied = {
        (pos.x, pos.y)
        for _ent, (pos, _blocks) in esper.get_components(Position, BlocksMovement)
    }
    occupied |= {(pos.x, pos.y) for _ent, (pos, _bed) in esper.get_components(Position, Bed)}

    def door_distance(tile: tuple[int, int]) -> int:
        if not door_adjacent:
            return 0
        return min(_chebyshev(tile, d) for d in door_adjacent)

    # Bed goes farthest from the door (cosy corner); it's non-blocking, so it
    # never affects whether the room stays connected.
    free = sorted(t for t in interior if t not in occupied)
    if not free:
        return None

    bed_tile = max(free, key=door_distance)
    bed_glyph, bed_color, bed_name, _bed_blocks, bed_extra = _FURNISHINGS[0]
    esper.create_entity(Position(*bed_tile), Renderable(bed_glyph, fg=bed_color), Name(bed_name), *bed_extra())
    occupied.add(bed_tile)

    # Try to place blocking furniture against the walls first (keeps the middle
    # open); accept a spot only if it doesn't cut any passable tile off from the
    # door. Non-blocking pieces (none by default) could go anywhere.
    def wall_adjacent(tile: tuple[int, int]) -> bool:
        return any((nx, ny) not in interior_set for nx, ny in game_map.neighbors_4(tile[0], tile[1]))

    candidates = [t for t in free if t != bed_tile and t not in door_adjacent]
    candidates.sort(key=lambda t: (not wall_adjacent(t), -door_distance(t)))
    blockers: set[tuple[int, int]] = set()
    seeds = door_adjacent if door_adjacent else {bed_tile}

    for glyph, color, name, blocks, extra in _FURNISHINGS[1:]:
        chosen: tuple[int, int] | None = None
        for i, tile in enumerate(candidates):
            if blocks:
                trial = blockers | {tile}
                passable = interior_set - trial
                if not passable <= _interior_reachable(interior_set, seeds, trial):
                    continue  # this spot would seal off part of the room
                blockers.add(tile)
            chosen = candidates.pop(i)
            break
        if chosen is None:
            continue  # nowhere safe for this piece; skip it
        components: list[object] = [Position(*chosen), Renderable(glyph, fg=color), Name(name)]
        if blocks:
            components.append(BlocksMovement())
        components.extend(extra())
        esper.create_entity(*components)
        occupied.add(chosen)

    return bed_tile


# Preset house the villagers build when no home is free: a 6x5 walled cabin with
# a door in the south wall. Coordinates are relative to the top-left corner.
_BLUEPRINT_W = 6
_BLUEPRINT_H = 5
_BLUEPRINT_DOOR = (2, _BLUEPRINT_H - 1)


def _blueprint_tiles(game_map: GameMap, ox: int, oy: int) -> tuple[list[tuple[int, int, str]], list[tuple[int, int]]]:
    """The (x, y, tile) wall/door pieces and the interior floor tiles for a cabin
    whose top-left corner is at (ox, oy)."""
    build: list[tuple[int, int, str]] = []
    interior: list[tuple[int, int]] = []
    for dy in range(_BLUEPRINT_H):
        for dx in range(_BLUEPRINT_W):
            x, y = ox + dx, oy + dy
            edge = dx in (0, _BLUEPRINT_W - 1) or dy in (0, _BLUEPRINT_H - 1)
            if (dx, dy) == _BLUEPRINT_DOOR:
                build.append((x, y, game_map.DOOR))
            elif edge:
                build.append((x, y, game_map.WALL))
            else:
                interior.append((x, y))
    return build, interior


def choose_build_site(game_map: GameMap, near: tuple[int, int], occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
    """Find a clear top-left corner for a cabin near ``near``: a WxH block of
    open floor tiles (no water, walls, borders, or occupants) with a one-tile
    margin so cabins don't fuse. Searches outward in rings; returns ``None`` if
    nowhere fits within range."""
    nx, ny = near
    for radius in range(3, 22):
        for oy in range(ny - radius, ny + radius + 1):
            for ox in range(nx - radius, nx + radius + 1):
                if _site_is_clear(game_map, ox, oy, occupied):
                    return (ox, oy)
    return None


def _site_is_clear(game_map: GameMap, ox: int, oy: int, occupied: set[tuple[int, int]]) -> bool:
    # One-tile margin around the footprint keeps cabins from sharing walls.
    for y in range(oy - 1, oy + _BLUEPRINT_H + 1):
        for x in range(ox - 1, ox + _BLUEPRINT_W + 1):
            if not game_map.in_bounds(x, y):
                return False
            if x == 0 or y == 0 or x == game_map.width - 1 or y == game_map.height - 1:
                return False
            if game_map.tile_at(x, y) != game_map.FLOOR:
                return False
            if (x, y) in occupied:
                return False
    return True


# A blueprint ghost previews the tile it will become in blue: a dim outline
# while it still needs materials, a brighter fill once its wood has been hauled
# in and it is ready to be raised. How many wood a builder carries per haul trip.
_BLUEPRINT_BLUE_DIM = (72, 108, 168)
_BLUEPRINT_BLUE = (96, 158, 240)
_HAUL_BATCH = 4
# Placeholder "occupant" id parked on the tile an NPC must not immediately
# reverse onto (see the anti-oscillation guard in ``_advance_region``). Any
# non-entity value the step code reads as "blocked, and not me" works; kept
# distinct from the -1 used for in-progress blueprint tiles.
_OSC_GUARD = -2

# Safety cap on how many times a single NPC may act in one region-turn, so a
# very quick creature (or a rounding edge) can never monopolise the loop.
_MAX_ACTIONS_PER_REGION_TURN = 4


# Flora/remains that must never be sealed inside a wall or a finished house.
# When a ghost is raised (or a cabin completes) any of these on the affected
# tiles are cleared so nothing ends up "stuck" in a wall or trapped indoors.
_OBSTRUCTION_COMPONENTS = (Tree, Corpse, BerryBush, Sapling)


def _spawn_blueprint(x: int, y: int, tile: str, site: int | None = None, stocked: bool = False) -> int:
    """Place a single blue-tinted ghost tile that previews ``tile`` at (x, y).
    Ghosts carry no ``BlocksMovement`` -- workers and passers-by walk right
    through a proto-structure until its walls are actually raised. ``site`` links
    it to a cabin ``ConstructionSite`` (``None`` for a loose player-placed piece)."""
    return esper.create_entity(
        Position(x, y),
        Renderable(tile, fg=_BLUEPRINT_BLUE if stocked else _BLUEPRINT_BLUE_DIM),
        Name("Blueprint"),
        Blueprint(tile=tile, stocked=stocked, site=site),
    )


def _set_blueprint_stocked(ghost_ent: int, stocked: bool) -> None:
    """Flip a ghost between "needs materials" (dim) and "ready to raise" (bright)."""
    if not esper.has_component(ghost_ent, Blueprint):
        return
    esper.component_for_entity(ghost_ent, Blueprint).stocked = stocked
    if esper.has_component(ghost_ent, Renderable):
        rend = esper.component_for_entity(ghost_ent, Renderable)
        rend.fg = _BLUEPRINT_BLUE if stocked else _BLUEPRINT_BLUE_DIM


def stock_blueprint(ghost_ent: int) -> None:
    """Mark a ghost's materials as delivered (public hook for the player)."""
    _set_blueprint_stocked(ghost_ent, True)


def _clear_tile_obstructions(x: int, y: int) -> None:
    """Delete any tree/corpse/bush/sapling sitting on (x, y) so it can't get
    sealed into a wall or trapped inside a house being raised over it."""
    for ent, (pos,) in list(esper.get_components(Position)):
        if (pos.x, pos.y) != (x, y):
            continue
        if any(esper.has_component(ent, comp) for comp in _OBSTRUCTION_COMPONENTS):
            esper.delete_entity(ent, immediate=True)


def create_construction_site(game_map: GameMap, origin: tuple[int, int]) -> int:
    """Stake out a cabin blueprint at ``origin`` (its top-left corner): a world
    ``ConstructionSite`` entity plus a ghost tile for every wall/door piece.
    Returns the site entity. Anybody can then haul wood to the ghosts and raise
    them; the last piece furnishes the cabin (see ``raise_blueprint``)."""
    build, interior = _blueprint_tiles(game_map, origin[0], origin[1])
    bed = interior[0] if interior else origin
    site_ent = esper.create_entity(ConstructionSite(interior=list(interior), bed=bed))
    site = esper.component_for_entity(site_ent, ConstructionSite)
    for (x, y, tile) in build:
        site.pieces[(x, y)] = _spawn_blueprint(x, y, tile, site=site_ent)
    return site_ent


def _complete_site(game_map: GameMap, site_ent: int) -> None:
    """Finish a cabin whose every ghost has been raised: clear anything trapped
    inside, furnish it, and leave it **unowned** for the nearest homeless
    resident to claim (the founder no longer owns the labour)."""
    if not esper.has_component(site_ent, ConstructionSite):
        return
    site = esper.component_for_entity(site_ent, ConstructionSite)
    interior = frozenset(site.interior)
    for (ix, iy) in interior:
        _clear_tile_obstructions(ix, iy)  # nothing gets trapped indoors
    furnish_house(game_map, interior)
    esper.delete_entity(site_ent, immediate=True)
    _push_turn_event("A new cabin is finished.")


def raise_blueprint(game_map: GameMap, ghost_ent: int) -> bool:
    """Raise one stocked ghost into its real tile: clear anything on the tile,
    set the tile, delete the ghost, and -- if it was the last piece of a cabin
    ``ConstructionSite`` -- furnish the finished house. Public so the player and
    the NPC AI raise pieces the same way."""
    if not esper.has_component(ghost_ent, Blueprint):
        return False
    bp = esper.component_for_entity(ghost_ent, Blueprint)
    pos = esper.component_for_entity(ghost_ent, Position)
    x, y = pos.x, pos.y
    _clear_tile_obstructions(x, y)
    game_map.set_tile(x, y, bp.tile)
    site_ent = bp.site
    esper.delete_entity(ghost_ent, immediate=True)
    if site_ent is not None and esper.entity_exists(site_ent) and esper.has_component(site_ent, ConstructionSite):
        site = esper.component_for_entity(site_ent, ConstructionSite)
        site.pieces.pop((x, y), None)
        if not site.pieces:
            _complete_site(game_map, site_ent)
    return True


_AUTOTILE_DIRECTION_OFFSETS = {
    1: (-1, -1),
    2: (0, -1),
    3: (1, -1),
    4: (1, 0),
    5: (1, 1),
    6: (0, 1),
    7: (-1, 1),
    8: (-1, 0),
}

_AUTOTILE_DIRECTION_MASKS = {
    1: 1,    # NW
    2: 2,    # N
    3: 4,    # NE
    4: 16,   # E
    5: 128,  # SE
    6: 64,   # S
    7: 32,   # SW
    8: 8,    # W
}

_CORPSE_GLYPH = "x"

_TURN_EVENTS: list[str] = []

_NearbyEntry = tuple[
    int,
    str,
    str,
    str,
    int,
    int,
    str,
    tuple[int, int, int] | None,
    tuple[int, int, int] | None,
]


def _push_turn_event(text: str) -> None:
    _TURN_EVENTS.append(text)


def _pull_turn_events() -> list[str]:
    events = list(_TURN_EVENTS)
    _TURN_EVENTS.clear()
    return events


def queue_message(text: str) -> None:
    """Public hook for non-processor code (e.g. the turn loop's interaction
    handlers) to push a line into the on-screen log. The RenderProcessor drains
    the queue at the top of every ``process`` call."""
    _push_turn_event(text)


def slay_entity(target_ent: int) -> int:
    """Turn ``target_ent`` into a corpse and return the new corpse entity.

    Shared by player melee (MovementProcessor) and predator kills (the AI). The
    corpse carries butcherable meat (named per creature via ``Meat``, falling
    back to generic Raw Meat) plus whatever the creature was carrying/wearing.
    """
    pos = esper.component_for_entity(target_ent, Position)
    target_name = "Unknown"
    if esper.has_component(target_ent, Name):
        target_name = esper.component_for_entity(target_ent, Name).value

    meat_name = RAW_MEAT
    if esper.has_component(target_ent, Meat):
        meat_name = esper.component_for_entity(target_ent, Meat).name

    corpse_items: list[str] = [meat_name]
    if esper.has_component(target_ent, Inventory):
        corpse_items.extend(esper.component_for_entity(target_ent, Inventory).items)

    corpse_components: list[object] = [
        Position(pos.x, pos.y),
        Renderable(_CORPSE_GLYPH),
        Name(f"Corpse of {target_name}"),
        Corpse(),
        Inventory(items=corpse_items),
    ]

    if esper.has_component(target_ent, Equipment):
        looted_slots = {
            slot_name: item_name
            for slot_name, item_name in esper.component_for_entity(target_ent, Equipment).slots.items()
            if item_name
        }
        if looted_slots:
            corpse_components.append(Equipment(slots=looted_slots))

    esper.delete_entity(target_ent, immediate=True)
    return esper.create_entity(*corpse_components)


class MovementProcessor(esper.Processor):
    """Applies a movement action to every player-controlled entity."""

    def __init__(
        self,
        game_map: GameMap,
        on_melee_attack: Callable[[], None] | None = None,
        on_enemy_death: Callable[[], None] | None = None,
    ):
        self.game_map = game_map
        self.on_melee_attack = on_melee_attack
        self.on_enemy_death = on_enemy_death

    def process(self, action: str | None = None) -> None:
        delta = _ACTION_DELTAS.get(action)
        if delta is None:
            return
        dx, dy = delta

        occupied = {
            (pos.x, pos.y): ent
            for ent, (pos, _blocks) in esper.get_components(Position, BlocksMovement)
        }

        for _ent, (pos, _player) in esper.get_components(Position, Player):
            nx, ny = pos.x + dx, pos.y + dy
            target_occupied = (nx, ny) in occupied and occupied[(nx, ny)] != _ent
            # The player can swim, so movement uses is_passable (land + water);
            # walls still block. NPCs keep using is_walkable (land only).
            if self.game_map.is_passable(nx, ny):
                if not target_occupied:
                    pos.x, pos.y = nx, ny
                    continue

                target_ent = occupied[(nx, ny)]
                target_name = "Unknown"
                if esper.has_component(target_ent, Name):
                    target_name = esper.component_for_entity(target_ent, Name).value

                # Hostiles are fought; deer are wild game the player can hunt.
                is_huntable = esper.has_component(target_ent, Enemy) or esper.has_component(
                    target_ent, Deer
                )
                if is_huntable:
                    _push_turn_event(f"You attack {target_name}.")
                    if self.on_melee_attack is not None:
                        self.on_melee_attack()
                    slay_entity(target_ent)
                    if self.on_enemy_death is not None:
                        self.on_enemy_death()
                    continue

                if esper.has_component(target_ent, Friendly):
                    _push_turn_event(f"{target_name} blocks your way. Press Enter to interact.")
                    continue

                _push_turn_event("Something blocks your way.")


# How far into a neighbouring region an NPC's goal search still looks, so
# standing near a simulation-region seam doesn't hide an otherwise-closer
# resource just across it. Matches ``_SOCIAL_SIGHT`` -- the widest existing
# "how far can a creature notice something" range in this file.
_REGION_BORDER_MARGIN = 8
# Wall-clock budget (seconds) a real turn spends nudging non-current NPC
# regions along in the background, on top of whatever the main loop's
# idle-time pump adds -- bounded so a busy world can never stall input.
_NPC_BACKGROUND_BUDGET = 0.004
# How many ``_step_toward`` calls a cached goal-rooted distance field is
# trusted for before it's rebuilt anyway (in case blockers have settled
# somewhere that changes the *effectively* best route enough to matter, even
# though the underlying static walkability hasn't).
_PATH_FIELD_REFRESH_CALLS = 20
# How many _advance_region calls a cached world snapshot (occupied tiles +
# goal candidates + NPCs, bucketed by region) is trusted for before it's
# rebuilt from scratch. A catch-up burst can call _advance_region thousands
# of times; this bounds how often that costs a full-world scan instead of a
# region-local lookup.
_WORLD_SNAPSHOT_REFRESH_CALLS = 25
# Need level (percent of max) at which an NPC stops what it's doing and forages.
_FORAGE_THRESHOLD = 55.0
# How much a single grazing/drinking/feeding action restores.
_GRAZE_RESTORE = 45.0
_DRINK_RESTORE = 100.0
_FEED_RESTORE = 60.0
# How much hunger a fish recovers from one bite of seaweed, and how far it will
# notice a frond and start swimming toward it.
_FISH_GRAZE_RESTORE = 40.0
_FISH_SIGHT = 12
# Wall-clock budget (seconds) a real turn spends nudging non-current regions
# along in the background, on top of whatever the main loop's idle-time pump
# adds -- bounded so ocean work can never stall input.
_FISH_BACKGROUND_BUDGET = 0.004
# Chance a fish drifts to a random neighbouring water tile on an idle turn, so
# the shoals mill about gently instead of holding perfectly still.
_FISH_WANDER_CHANCE = 0.6


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Whether two ``(x0, y0, x1, y1)`` rectangles intersect."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# --- Personalities, friendships, and social interactions -------------------
# Sentient beings (villagers) carry a Personality of one or more named traits.
# Each trait contributes:
#   * sociability -- how strongly it drives the being to seek out company;
#   * warmth      -- how pleasant the being is in an interaction. Positive warmth
#                    makes chats build friendship; negative warmth (Grumpy/Aloof)
#                    can turn an interaction sour (a "--").
@dataclass(frozen=True)
class _Trait:
    sociability: float
    warmth: float


_TRAITS: dict[str, _Trait] = {
    "Cheerful": _Trait(sociability=1.0, warmth=1.6),
    "Kind":     _Trait(sociability=0.6, warmth=1.4),
    "Outgoing": _Trait(sociability=1.6, warmth=0.6),
    "Playful":  _Trait(sociability=1.1, warmth=1.0),
    "Shy":      _Trait(sociability=-1.0, warmth=0.2),
    "Aloof":    _Trait(sociability=-1.4, warmth=-0.6),
    "Grumpy":   _Trait(sociability=-0.4, warmth=-1.6),
}

# Baseline warmth of any interaction: two neutral strangers chatting still drift
# gently friendlier, so relationships build over repeated contact.
_INTERACTION_BASE = 1.4
# Each being's own warmth and its partner's both feed the friendship change.
_SELF_WARMTH_WEIGHT = 0.7
_OTHER_WARMTH_WEIGHT = 0.7
# A single interaction never moves friendship more than this, so it stays gradual.
_INTERACTION_STEP_CAP = 6.0
# Friendship bounds.
_FRIENDSHIP_MIN = -100.0
_FRIENDSHIP_MAX = 100.0

# Time (TU) a being waits between voluntary interactions (so villagers don't
# chatter every single turn), how far it looks for company, and how much each tile
# of distance counts against a candidate when weighed against friendship. Expressed
# in TU (6 baseline turns) since social timestamps live on the world clock.
_SOCIAL_COOLDOWN = 6 * BASE_ACTION_COST
_SOCIAL_SIGHT = 8
_SOCIAL_DISTANCE_PENALTY = 2.0

# --- Courtship, mating, and marriage ---------------------------------------
# Two adults of opposite gender grow from friends to lovers to spouses as their
# mutual friendship climbs. Lovers may have sex (in private); a very close pair
# marries and shares a bed. Thresholds are on the [-100, 100] friendship scale.
_LOVERS_FRIENDSHIP = 60.0     # mutual friendship at which a pair may have sex
_MARRIAGE_FRIENDSHIP = 85.0   # mutual friendship at which a pair weds
# Sex happens only when nobody else is watching: no other person may stand within
# this many tiles of either partner.
_MATING_PRIVACY_RADIUS = 6
# The odds a single act of sex results in pregnancy.
_PREGNANCY_CHANCE = 0.01
# Gestation: "9 months" read as 3/4 of this world's 112-day year (a full 9 of the
# 28-day months would run past a year). Measured in days; scaled to turns in use.
_GESTATION_DAYS = 84
# The heart that floats over a couple who have just been intimate.
_HEART = "♥"
_HEART_PINK = (235, 110, 150)


def personality_warmth(traits: list[str]) -> float:
    """Sum of the warmth of a being's traits (0.0 for a being with none)."""
    return sum(_TRAITS[name].warmth for name in traits if name in _TRAITS)


def trait_sociability(traits: list[str]) -> float:
    """Sum of the sociability of a being's traits."""
    return sum(_TRAITS[name].sociability for name in traits if name in _TRAITS)


def interaction_delta(self_traits: list[str], other_traits: list[str]) -> float:
    """How much one interaction shifts a being's friendship toward a partner,
    given both personalities. Positive warms the relationship (``++``), negative
    sours it (``--``). Deterministic (no RNG) so outcomes are testable/stable."""
    raw = (
        _INTERACTION_BASE
        + _SELF_WARMTH_WEIGHT * personality_warmth(self_traits)
        + _OTHER_WARMTH_WEIGHT * personality_warmth(other_traits)
    )
    return max(-_INTERACTION_STEP_CAP, min(_INTERACTION_STEP_CAP, raw))


def friendship(rel: Relationships | None, other: int) -> float:
    """The friendship score in ``rel`` toward ``other`` (0.0 for a stranger or a
    being with no Relationships component)."""
    if rel is None:
        return 0.0
    return rel.scores.get(other, 0.0)


def _relationships(ent: int) -> Relationships:
    if esper.has_component(ent, Relationships):
        return esper.component_for_entity(ent, Relationships)
    rel = Relationships()
    esper.add_component(ent, rel)
    return rel


def adjust_friendship(ent: int, other: int, delta: float) -> float:
    """Move ``ent``'s friendship toward ``other`` by ``delta`` (clamped to the
    friendship bounds), creating a Relationships component if needed. Returns the
    new score."""
    rel = _relationships(ent)
    current = rel.scores.get(other, 0.0)
    updated = max(_FRIENDSHIP_MIN, min(_FRIENDSHIP_MAX, current + delta))
    rel.scores[other] = updated
    return updated


# --- Speech bubbles --------------------------------------------------------
# Floating labels drawn above a character for a short wall-clock time, so they
# linger and fade across idle animation ticks (mirroring the _TURN_EVENTS log
# queue, but positional and self-expiring). The RenderProcessor draws whatever is
# live each frame; the turn loop keeps re-rendering while any are active.
_BUBBLE_TTL = 2.5  # seconds a bubble stays on screen
# A bubble fades to nothing over its final ``_BUBBLE_FADE_SECONDS`` instead of
# vanishing abruptly.
_BUBBLE_FADE_SECONDS = 0.8
# A busy tile stacks bubbles upward (newest nearest the character); the oldest
# scroll off once the stack is this tall.
_MAX_BUBBLES_PER_CELL = 3

# The ``++``/``--``/``+``/``-`` indicator that trails a bubble's gibberish. Most
# bubbles show none: friendship keeps changing every exchange, but a being's
# reaction accumulates silently and only surfaces once it reaches a milestone
# (which tends to fall late in a conversation) or, now and then, at random in the
# middle. Green warms, red sours; the doubled form marks a big built-up reaction.
_INDICATOR_MIN = 0.5         # below this |pending|, never show an indicator
_INDICATOR_MILESTONE = 9.0   # |pending| that reliably surfaces an indicator
_INDICATOR_DOUBLE = 16.0     # |pending| at/above which it doubles (++ / --)
_MID_CHAT_INDICATOR_CHANCE = 0.08  # odds of surfacing a mild indicator mid-chat
_INDICATOR_GREEN = (110, 215, 120)
_INDICATOR_RED = (232, 96, 96)

_GIBBERISH_TOKENS = (
    "ba", "zo", "ki", "mu", "lo", "ta", "wu", "ne", "ry", "sh", "gorp", "bl", "eek", "mm",
)


@dataclass
class _Bubble:
    x: int
    y: int
    text: str
    indicator: str = ""
    indicator_color: tuple[int, int, int] | None = None
    born: float = 0.0
    ttl: float = _BUBBLE_TTL


_SPEECH_BUBBLES: list[_Bubble] = []


def gibberish(words: int = 2) -> str:
    """A short nonsense utterance in the beings' 'language' -- a couple of made-up
    syllable clusters (e.g. ``"bazo ki!"``). Cosmetic only."""
    rng = world_rng().stream("gibberish")
    parts: list[str] = []
    for _ in range(max(1, words)):
        syllables = rng.randint(1, 2)
        parts.append("".join(rng.choice(_GIBBERISH_TOKENS) for _ in range(syllables)))
    return " ".join(parts) + rng.choice(("", "?", "!", "..."))


def spawn_speech_bubble(
    x: int,
    y: int,
    text: str,
    indicator: str = "",
    indicator_color: tuple[int, int, int] | None = None,
    ttl: float = _BUBBLE_TTL,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Pop a floating bubble above world cell ``(x, y)`` for ``ttl`` seconds. Keeps
    at most ``_MAX_BUBBLES_PER_CELL`` bubbles per cell (oldest dropped) so a chatty
    character can't spawn an unbounded column. Overlap between neighbouring bubbles
    is resolved at draw time (newer bubbles shove older ones up).
    ``indicator``/``indicator_color`` render as a coloured suffix."""
    _SPEECH_BUBBLES.append(
        _Bubble(x, y, text, indicator, indicator_color, born=clock(), ttl=ttl)
    )
    at_cell = [b for b in _SPEECH_BUBBLES if b.x == x and b.y == y]
    for stale in at_cell[:-_MAX_BUBBLES_PER_CELL]:
        _SPEECH_BUBBLES.remove(stale)


def active_bubbles(now: float | None = None) -> list[_Bubble]:
    """Prune expired bubbles and return the live ones (used by the renderer)."""
    current = now if now is not None else time.monotonic()
    live = [b for b in _SPEECH_BUBBLES if current - b.born < b.ttl]
    if len(live) != len(_SPEECH_BUBBLES):
        _SPEECH_BUBBLES[:] = live
    return live


def bubbles_active(now: float | None = None) -> bool:
    """True while any speech bubble is still on screen, so the turn loop keeps
    re-rendering to fade them out even with no input."""
    return bool(active_bubbles(now))


def bubble_alpha(bubble: _Bubble, now: float) -> int:
    """A bubble's current opacity (0-255): full for most of its life, then a
    linear fade to nothing over its final ``_BUBBLE_FADE_SECONDS``."""
    remaining = bubble.ttl - (now - bubble.born)
    if remaining <= 0:
        return 0
    if remaining >= _BUBBLE_FADE_SECONDS:
        return 255
    return max(0, min(255, int(255 * remaining / _BUBBLE_FADE_SECONDS)))


def _react(
    rel: Relationships,
    other: int,
    delta: float,
    rng: Callable[[], float] | None = None,
) -> tuple[str, tuple[int, int, int] | None]:
    """Fold ``delta`` into ``rel``'s pending reaction toward ``other`` and decide
    whether this bubble shows an indicator. Returns the ``(text, colour)`` to draw,
    or ``("", None)`` for a plain (indicator-less) bubble.

    Most exchanges return nothing: the reaction accumulates. It surfaces once the
    pending total reaches ``_INDICATOR_MILESTONE`` (so a run of chatting culminates
    in a ``+``/``-``, doubled past ``_INDICATOR_DOUBLE``) or, occasionally, at random
    once it is at least ``_INDICATOR_MIN``. Surfacing resets the pending total."""
    if rng is None:
        rng = world_rng().stream("social").random
    pending = rel.pending.get(other, 0.0) + delta
    magnitude = abs(pending)

    if magnitude >= _INDICATOR_MILESTONE:
        show, strong = True, magnitude >= _INDICATOR_DOUBLE
    elif magnitude >= _INDICATOR_MIN and rng() < _MID_CHAT_INDICATOR_CHANCE:
        show, strong = True, False
    else:
        rel.pending[other] = pending
        return "", None

    rel.pending[other] = 0.0
    if pending > 0:
        return ("++" if strong else "+"), _INDICATOR_GREEN
    return ("--" if strong else "-"), _INDICATOR_RED


def _traits_of(ent: int) -> list[str]:
    if esper.has_component(ent, Personality):
        return esper.component_for_entity(ent, Personality).traits
    return []


def interact(
    a: int,
    b: int,
    turn: int,
    clock: Callable[[], float] = time.monotonic,
    rng: Callable[[], float] | None = None,
) -> tuple[float, float]:
    """Run one social interaction between beings ``a`` and ``b``: adjust each one's
    friendship toward the other by its personality-driven delta, stamp the cooldown
    on both, and pop a gibberish speech bubble above each -- most bubbles are plain,
    with a ``+``/``-`` indicator surfacing only now and then (see ``_react``).
    Returns ``(delta_a, delta_b)``. Shared by the NPC social AI and the player Talk
    action."""
    if rng is None:
        rng = world_rng().stream("social").random
    delta_a = interaction_delta(_traits_of(a), _traits_of(b))
    delta_b = interaction_delta(_traits_of(b), _traits_of(a))
    rel_a = _relationships(a)
    rel_b = _relationships(b)
    adjust_friendship(a, b, delta_a)
    adjust_friendship(b, a, delta_b)

    for ent, other, rel, delta in ((a, b, rel_a, delta_a), (b, a, rel_b, delta_b)):
        if esper.has_component(ent, Personality):
            esper.component_for_entity(ent, Personality).last_social_turn = turn
        if esper.has_component(ent, Position):
            pos = esper.component_for_entity(ent, Position)
            indicator, indicator_color = _react(rel, other, delta, rng)
            spawn_speech_bubble(
                pos.x, pos.y, gibberish(),
                indicator=indicator, indicator_color=indicator_color, clock=clock,
            )
    return delta_a, delta_b


# --- Courtship: lovers, sex, and marriage ----------------------------------


def _gender_of(ent: int) -> str | None:
    if esper.has_component(ent, Gender):
        return esper.component_for_entity(ent, Gender).value
    return None


def _family(ent: int) -> Family:
    """The entity's ``Family`` component, created (with an empty surname) if it
    somehow lacks one -- so marriage can always record a spouse."""
    if esper.has_component(ent, Family):
        return esper.component_for_entity(ent, Family)
    fam = Family(surname="")
    esper.add_component(ent, fam)
    return fam


def _surname_of(name: str) -> str:
    """The trailing surname of a ``"Given Surname"`` display name (or the whole
    string if it has no space)."""
    return name.rsplit(" ", 1)[-1]


def are_courtship_eligible(a: int, b: int, clock: WorldClock | None) -> bool:
    """Whether ``a`` and ``b`` could be a romantic couple at all: two different,
    living, adult, opposite-gender, friendly beings. Friendship level (lovers vs
    spouses) is checked separately by the callers."""
    if a == b or not esper.entity_exists(a) or not esper.entity_exists(b):
        return False
    if not (is_adult(a, clock) and is_adult(b, clock)):
        return False
    ga, gb = _gender_of(a), _gender_of(b)
    if ga is None or gb is None or ga == gb:
        return False
    return esper.has_component(a, Friendly) and esper.has_component(b, Friendly)


def _mutual_friendship(a: int, b: int) -> float:
    """The weaker of the two friendship scores -- courtship needs both to feel it,
    so we gate on the lower direction."""
    rel_a = esper.component_for_entity(a, Relationships) if esper.has_component(a, Relationships) else None
    rel_b = esper.component_for_entity(b, Relationships) if esper.has_component(b, Relationships) else None
    return min(friendship(rel_a, b), friendship(rel_b, a))


def is_private(a: int, b: int, sentients: list[tuple[int, Position]]) -> bool:
    """True when no third person stands within ``_MATING_PRIVACY_RADIUS`` of either
    partner -- the couple is alone enough to be intimate. ``sentients`` is the
    per-turn snapshot of everyone with a ``Personality``."""
    if not (esper.has_component(a, Position) and esper.has_component(b, Position)):
        return False
    pa = esper.component_for_entity(a, Position)
    pb = esper.component_for_entity(b, Position)
    for other, opos in sentients:
        if other == a or other == b:
            continue
        if (_chebyshev((pa.x, pa.y), (opos.x, opos.y)) <= _MATING_PRIVACY_RADIUS
                or _chebyshev((pb.x, pb.y), (opos.x, opos.y)) <= _MATING_PRIVACY_RADIUS):
            return False
    return True


def try_marry(a: int, b: int, clock: WorldClock | None) -> bool:
    """Wed ``a`` and ``b`` if they are courtship-eligible, both currently single,
    and their mutual friendship has reached ``_MARRIAGE_FRIENDSHIP``. The wife
    takes the husband's surname (her display name updates to match), the couple
    come to share a home, and the wedding is logged. Returns True if they married."""
    if not are_courtship_eligible(a, b, clock):
        return False
    fam_a, fam_b = _family(a), _family(b)
    if fam_a.spouse is not None or fam_b.spouse is not None:
        return False
    if _mutual_friendship(a, b) < _MARRIAGE_FRIENDSHIP:
        return False

    fam_a.spouse = b
    fam_b.spouse = a

    # The wife adopts the husband's family name so they -- and their children --
    # share one surname.
    husband = a if _gender_of(a) == "male" else b
    wife = b if husband == a else a
    surname = _family(husband).surname or _surname_of(_entity_display_name(husband))
    _adopt_surname(wife, surname)

    # Share a bed: whoever already has a home takes the other in.
    _merge_homes(husband, wife)

    _push_turn_event(f"{_entity_display_name(a)} and {_entity_display_name(b)} are wed.")
    return True


def try_mate(
    a: int,
    b: int,
    turn: int,
    clock: WorldClock | None,
    sentients: list[tuple[int, Position]],
    rng: Callable[[], float] | None = None,
    bubble_clock: Callable[[], float] = time.monotonic,
) -> bool:
    """A private moment between lovers. If ``a`` and ``b`` are courtship-eligible,
    their mutual friendship is at least ``_LOVERS_FRIENDSHIP``, they are alone
    (``is_private``), and the man hasn't already mated today (women are unbounded),
    they have sex: both get a mating timestamp, a heart floats over each, and with
    probability ``_PREGNANCY_CHANCE`` the woman (if not already pregnant) conceives.
    Returns True if the act happened."""
    if rng is None:
        rng = world_rng().stream("repro").random
    if not are_courtship_eligible(a, b, clock):
        return False
    if _mutual_friendship(a, b) < _LOVERS_FRIENDSHIP:
        return False
    if not is_private(a, b, sentients):
        return False

    man = a if _gender_of(a) == "male" else b
    woman = b if man == a else a

    # Men may only mate once per day; women as often as they like.
    man_mating = _mating(man)
    if current_day_for_turn(man_mating.last_turn, clock) >= current_day(clock):
        return False

    man_mating.last_turn = turn
    _mating(woman).last_turn = turn

    for person in (man, woman):
        if esper.has_component(person, Position):
            pos = esper.component_for_entity(person, Position)
            spawn_speech_bubble(pos.x, pos.y, _HEART, clock=bubble_clock)
    _push_turn_event(f"{_entity_display_name(a)} and {_entity_display_name(b)} slip away together.")

    if rng() < _PREGNANCY_CHANCE and not esper.has_component(woman, Pregnant):
        esper.add_component(woman, Pregnant(conceived_turn=turn, father=man))
    return True


def _mating(ent: int) -> Mating:
    if esper.has_component(ent, Mating):
        return esper.component_for_entity(ent, Mating)
    m = Mating()
    esper.add_component(ent, m)
    return m


def current_day_for_turn(turn: int, clock: WorldClock | None) -> int:
    """The day number a given ``turn`` falls on -- for comparing a past mating turn
    against today. A never-mated sentinel (large negative) lands on a past day."""
    if clock is None:
        return turn
    return turn // max(1, clock.day_length)


def _entity_display_name(ent: int) -> str:
    if esper.has_component(ent, Name):
        return esper.component_for_entity(ent, Name).value
    return "Someone"


def _adopt_surname(ent: int, surname: str) -> None:
    """Give ``ent`` the family ``surname``, updating both its ``Family`` record and
    the surname shown in its display ``Name``."""
    _family(ent).surname = surname
    if esper.has_component(ent, Name):
        name = esper.component_for_entity(ent, Name)
        given = name.value.rsplit(" ", 1)[0] if " " in name.value else name.value
        name.value = f"{given} {surname}"


def _merge_homes(husband: int, wife: int) -> None:
    """Bring a newly-wed couple under one roof so they share a bed. Whoever already
    has a home hosts; if only one does, the other moves in; if neither, nothing to
    do yet (the housing system settles them later, spouse-aware)."""
    h_home = esper.component_for_entity(husband, Home) if esper.has_component(husband, Home) else None
    w_home = esper.component_for_entity(wife, Home) if esper.has_component(wife, Home) else None
    if h_home is not None:
        _set_home(wife, (h_home.x, h_home.y))
    elif w_home is not None:
        _set_home(husband, (w_home.x, w_home.y))


def _set_home(ent: int, xy: tuple[int, int]) -> None:
    if esper.has_component(ent, Home):
        home = esper.component_for_entity(ent, Home)
        home.x, home.y = xy
    else:
        esper.add_component(ent, Home(xy[0], xy[1]))


class NpcAiProcessor(esper.Processor):
    """Drives NPC behaviour each turn.

    Priority order per creature:
      1. Satisfy an urgent need -- drink at water, or (by diet) graze a tree or
         hunt prey.
      2. Otherwise, hostiles chase the player when they can see them.

    Water tiles and the walkable "shore" tiles beside them are precomputed once
    (the map is static) so a thirsty animal is a cheap nearest-lookup plus one
    pathfind, not a full-map rescan every turn.
    """

    def __init__(self, game_map: GameMap, wall_clock: Callable[[], float] | None = None):
        self.game_map = game_map
        self._shore_tiles: list[tuple[int, int]] = self._compute_shore_tiles()
        self._wall_clock = wall_clock if wall_clock is not None else time.monotonic
        self.scheduler = RegionScheduler(game_map, _current_region_turn())
        self.scheduler.register("npc_ai", self._advance_region)
        # A world-wide, region-bucketed snapshot (occupied tiles + every goal
        # kind + every NPC), rebuilt from scratch only every
        # _WORLD_SNAPSHOT_REFRESH_CALLS advances instead of on every single
        # one -- a catch-up burst (region-entry, sleep) advances many regions
        # many turns each, and re-scanning literally every entity in the world
        # for each of those individual turns is the dominant cost at any real
        # population (bounded staleness traded for that no longer happening).
        self._world_snapshot: dict[str, dict] | None = None
        self._world_snapshot_calls_left = 0
        # goal xy -> (edit revision near goal when cached, calls left before a
        # routine refresh, the flow field itself). Shared across every NPC
        # heading to the same goal, not per-entity -- a distance field rooted
        # at a (largely static) goal stays valid for any traveller approaching
        # it from anywhere, so many NPCs reuse the one flood.
        self._field_cache: dict[tuple[int, int], tuple[int, int, dict[tuple[int, int], int]]] = {}
        # ent -> the tile it stood on at the start of its previous turn. Used to
        # forbid an immediate one-tile reversal (see ``_advance_region``), which
        # is the only way an NPC ends up flip-flopping between two tiles forever.
        self._prev_turn_pos: dict[int, tuple[int, int]] = {}

    def _compute_shore_tiles(self) -> list[tuple[int, int]]:
        shore: list[tuple[int, int]] = []
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                if not self.game_map.is_walkable(x, y):
                    continue
                if any(self.game_map.is_water(nx, ny) for nx, ny in self.game_map.neighbors_8(x, y)):
                    shore.append((x, y))
        return shore

    def _find_player_position(self) -> tuple[int, int] | None:
        for _ent, (pos, _player) in esper.get_components(Position, Player):
            return (pos.x, pos.y)
        return None

    @staticmethod
    def _nearest(origin: tuple[int, int], candidates: list[tuple[int, int]]) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        best_dist = None
        for cand in candidates:
            dist = _chebyshev(origin, cand)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = cand
        return best

    def _distance_field_for(self, goal: tuple[int, int]) -> dict[tuple[int, int], int]:
        """A cached flow field to ``goal`` (see ``GameMap.distance_field``),
        rebuilt only when an edit lands near ``goal`` or the routine refresh
        interval elapses -- not on every call, and not on an edit anywhere
        else in the world."""
        edit_revision = self.game_map.region_edit_revision(goal[0], goal[1])
        cached = self._field_cache.get(goal)
        if cached is not None:
            cached_revision, calls_left, field = cached
            if cached_revision == edit_revision and calls_left > 0:
                self._field_cache[goal] = (cached_revision, calls_left - 1, field)
                return field
        field = self.game_map.distance_field(goal)
        self._field_cache[goal] = (edit_revision, _PATH_FIELD_REFRESH_CALLS, field)
        return field

    def _greedy_step_toward(
        self,
        ent: int,
        xy: tuple[int, int],
        goal: tuple[int, int],
        occupied: dict[tuple[int, int], int],
    ) -> tuple[int, int] | None:
        """The best available neighbour of ``xy`` toward ``goal`` per the
        cached flow field, or ``None`` if ``goal`` isn't in it (out of its
        walkable region, the field hasn't been built for it, or literally
        every closer neighbour is currently blocked).

        Ranks *every* closer neighbour, not just the single nearest one, and
        skips ones a live occupant blocks -- so a crowded goal (e.g. a
        builder's own other, not-yet-placed blueprint pieces sitting right
        next to this one) doesn't force an expensive fallback pathfind merely
        because the closest step happens to be taken; the second- or
        third-closest is usually just as good and is right here in the same
        cached field.
        """
        field = self._distance_field_for(goal)
        here = field.get(xy)
        if here is None:
            return None
        candidates = [
            (dist, nxy)
            for nxy in self.game_map.neighbors_8(xy[0], xy[1])
            if (dist := field.get(nxy)) is not None and dist < here
        ]
        candidates.sort(key=lambda c: c[0])
        for _dist, nxy in candidates:
            blocked_by_goal = nxy == goal and nxy in occupied
            blocked_by_other = nxy in occupied and occupied[nxy] != ent
            if not blocked_by_goal and not blocked_by_other:
                return nxy
        return None

    def _step_toward(
        self,
        ent: int,
        pos: Position,
        goal: tuple[int, int],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Move one step toward ``goal``. Returns True if it moved.

        The common case is a cheap lookup in a cached, goal-rooted flow field
        (shared across every NPC currently heading to that goal) instead of a
        fresh BFS. If the goal is outside the cached field entirely, or every
        viable step toward it is currently blocked, this falls back to a
        one-off occupant-aware pathfind -- exactly what ran here before this
        cache existed.
        """
        # A non-walkable goal can never be reached (find_path can't discover
        # it either -- it's excluded during the BFS itself), so don't spend a
        # full flood or fallback pathfind finding that out the hard way. Comes
        # up for a stale/duplicate blueprint tile that's already the right type
        # (see ``_reachable_ghosts``) and cheaply guards any other goal kind that
        # could end up unreachable the same way.
        if not self.game_map.is_walkable(goal[0], goal[1]):
            return False

        xy = (pos.x, pos.y)
        step = self._greedy_step_toward(ent, xy, goal, occupied)
        if step is not None:
            self._commit_step(ent, pos, step, occupied)
            return True

        blocked = {xy2 for xy2, occ_ent in occupied.items() if occ_ent != ent}
        path = self.game_map.find_path(xy, goal, blocked_tiles=blocked)
        if not path:
            return False
        next_x, next_y = path[0]
        if (next_x, next_y) == goal and (next_x, next_y) in occupied:
            # The goal tile itself is occupied (e.g. a tree/prey we path *to*);
            # don't step onto it -- the caller handles the adjacent interaction.
            return False
        if (next_x, next_y) in occupied and occupied[(next_x, next_y)] != ent:
            return False
        self._commit_step(ent, pos, (next_x, next_y), occupied)
        return True

    @staticmethod
    def _commit_step(
        ent: int, pos: Position, next_xy: tuple[int, int], occupied: dict[tuple[int, int], int]
    ) -> None:
        old_xy = (pos.x, pos.y)
        pos.x, pos.y = next_xy
        occupied.pop(old_xy, None)
        occupied[next_xy] = ent

    def _reachable(
        self, pos: Position, items: list[tuple[tuple[int, int], int]]
    ) -> list[tuple[tuple[int, int], int]]:
        """Keep only ``(xy, ent)`` targets in the same walkable region as ``pos``,
        so a creature never fixates on food/water across a river it can't cross."""
        return [item for item in items if self.game_map.same_region((pos.x, pos.y), item[0])]

    def _seek_water(
        self, ent: int, pos: Position, needs: Needs, occupied: dict[tuple[int, int], int]
    ) -> bool:
        if any(self.game_map.is_water(nx, ny) for nx, ny in self.game_map.neighbors_8(pos.x, pos.y)):
            needs.thirst = max(0.0, needs.thirst - _DRINK_RESTORE)
            return True
        # Drink from a shore tile we can actually stand on -- skip shores blocked
        # by a tree/creature (trees cluster by water), or the animal would fixate
        # on an unreachable spot and thrash on the bank without ever drinking.
        reachable = [
            s
            for s in self._shore_tiles
            if s not in occupied and self.game_map.same_region((pos.x, pos.y), s)
        ]
        target = self._nearest((pos.x, pos.y), reachable)
        if target is None:
            return False
        return self._step_toward(ent, pos, target, occupied)

    def _graze(
        self,
        ent: int,
        pos: Position,
        needs: Needs,
        trees: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        trees = self._reachable(pos, trees)
        if not trees:
            return False
        target_xy, target_ent = min(trees, key=lambda item: _chebyshev((pos.x, pos.y), item[0]))
        if _chebyshev((pos.x, pos.y), target_xy) == 1:
            needs.hunger = max(0.0, needs.hunger - _GRAZE_RESTORE)
            if esper.entity_exists(target_ent) and esper.has_component(target_ent, Tree):
                tree = esper.component_for_entity(target_ent, Tree)
                tree.wood -= 1
                if tree.wood <= 0:
                    esper.delete_entity(target_ent, immediate=True)  # grazed bare
                    occupied.pop(target_xy, None)
            return True
        return self._step_toward(ent, pos, target_xy, occupied)

    def _forage_berries(
        self,
        ent: int,
        pos: Position,
        needs: Needs,
        bushes: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
        clock: WorldClock | None,
    ) -> bool:
        """Head to the nearest ripe berry bush and pick it clean. Bushes block
        their tile, so the forager eats from an adjacent one; the bush regrows a
        fresh crop days later (see ``TreeGrowthProcessor``).

        Takes ``clock`` rather than calling ``world_clock()`` itself: during a
        region catch-up burst this is an as-of clock for the turn being
        replayed, not the true "now" -- using the real clock here would let a
        bush regrow based on time that, for this region, hasn't happened yet.
        """
        bushes = self._reachable(pos, bushes)
        if not bushes:
            return False
        target_xy, target_ent = min(bushes, key=lambda item: _chebyshev((pos.x, pos.y), item[0]))
        if _chebyshev((pos.x, pos.y), target_xy) == 1:
            if pick_berries(target_ent, clock):
                needs.hunger = max(0.0, needs.hunger - _GRAZE_RESTORE)
            return True
        return self._step_toward(ent, pos, target_xy, occupied)

    def _seek_food(
        self,
        ent: int,
        pos: Position,
        needs: Needs,
        prey: list[tuple[tuple[int, int], int]],
        corpses: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """A hungry meat-eater heads to the nearest food: an existing corpse it
        can scavenge, or a live deer it can hunt."""
        # (xy, kind, target_ent). Corpses are already filtered to ones with meat;
        # keep only what's actually reachable from here.
        corpses = self._reachable(pos, corpses)
        prey = self._reachable(pos, prey)
        candidates: list[tuple[tuple[int, int], str, int]] = [
            (xy, "corpse", corpse_ent) for xy, corpse_ent in corpses
        ]
        candidates.extend(
            (xy, "prey", prey_ent)
            for xy, prey_ent in prey
            if prey_ent != ent and esper.entity_exists(prey_ent)
        )
        if not candidates:
            return False

        target_xy, kind, target_ent = min(
            candidates, key=lambda c: _chebyshev((pos.x, pos.y), c[0])
        )
        distance = _chebyshev((pos.x, pos.y), target_xy)

        if kind == "prey":
            # Deer block their tile, so feed from an adjacent tile.
            if distance == 1:
                slay_entity(target_ent)  # kill and butcher; predator feeds
                occupied.pop(target_xy, None)
                needs.hunger = max(0.0, needs.hunger - _FEED_RESTORE)
                return True
            return self._step_toward(ent, pos, target_xy, occupied)

        # Corpses don't block, so "reach" is standing on or next to them.
        if distance <= 1:
            self._eat_from_corpse(target_ent, needs)
            return True
        return self._step_toward(ent, pos, target_xy, occupied)

    def _eat_from_corpse(self, corpse_ent: int, needs: Needs) -> None:
        if not esper.has_component(corpse_ent, Inventory):
            return
        inventory = esper.component_for_entity(corpse_ent, Inventory)
        for index, item in enumerate(inventory.items):
            if is_raw_meat(item) or is_cooked_meat(item):
                inventory.items.pop(index)
                needs.hunger = max(0.0, needs.hunger - _FEED_RESTORE)
                return

    def _eat_from_inventory(self, ent: int, needs: Needs) -> bool:
        """A hungry creature eats a *prepared* item it is carrying (cooked meat,
        bread, ...) before foraging. Raw meat is skipped -- meat must be cooked
        first. Returns True if it ate."""
        if not esper.has_component(ent, Inventory):
            return False
        inventory = esper.component_for_entity(ent, Inventory)
        for index, item in enumerate(inventory.items):
            if is_raw_meat(item):
                continue
            restored = hunger_restored(item)
            if restored is not None:
                inventory.items.pop(index)
                needs.hunger = max(0.0, needs.hunger - restored)
                return True
        return False

    # --- "cook" diet: the full loop a villager runs to feed itself ---------
    # get raw meat (hunt/scavenge into pack) -> get wood (chop a tree) ->
    # carry both to a stove and cook -> eat the cooked meat (via
    # _eat_from_inventory next turn). Each call advances one step of that plan.

    def _forage_meat(
        self,
        ent: int,
        pos: Position,
        inventory: Inventory,
        prey: list[tuple[tuple[int, int], int]],
        corpses: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Put raw meat in the pack: scavenge a corpse, or kill a deer (which
        leaves a corpse to scavenge next). Unlike a predator, does not eat here."""
        corpses = self._reachable(pos, corpses)
        prey = self._reachable(pos, prey)
        candidates: list[tuple[tuple[int, int], str, int]] = [
            (xy, "corpse", corpse_ent) for xy, corpse_ent in corpses
        ]
        candidates.extend(
            (xy, "prey", prey_ent)
            for xy, prey_ent in prey
            if prey_ent != ent and esper.entity_exists(prey_ent)
        )
        if not candidates:
            return False
        target_xy, kind, target_ent = min(candidates, key=lambda c: _chebyshev((pos.x, pos.y), c[0]))
        distance = _chebyshev((pos.x, pos.y), target_xy)

        if kind == "prey":
            if distance == 1:
                slay_entity(target_ent)  # leaves a corpse to butcher next turn
                occupied.pop(target_xy, None)
                return True
            return self._step_toward(ent, pos, target_xy, occupied)

        if distance <= 1:
            self._take_meat_from_corpse(target_ent, inventory)
            return True
        return self._step_toward(ent, pos, target_xy, occupied)

    def _take_meat_from_corpse(self, corpse_ent: int, inventory: Inventory) -> None:
        if not esper.has_component(corpse_ent, Inventory):
            return
        corpse_inventory = esper.component_for_entity(corpse_ent, Inventory)
        for index, item in enumerate(corpse_inventory.items):
            if is_raw_meat(item):
                corpse_inventory.items.pop(index)
                inventory.items.append(item)
                return

    def _gather_wood(
        self,
        ent: int,
        pos: Position,
        inventory: Inventory,
        trees: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        trees = self._reachable(pos, trees)
        if not trees:
            return False
        target_xy, tree_ent = min(trees, key=lambda t: _chebyshev((pos.x, pos.y), t[0]))
        if _chebyshev((pos.x, pos.y), target_xy) == 1:
            inventory.items.append(WOOD)
            if esper.entity_exists(tree_ent) and esper.has_component(tree_ent, Tree):
                tree = esper.component_for_entity(tree_ent, Tree)
                tree.wood -= 1
                if tree.wood <= 0:
                    esper.delete_entity(tree_ent, immediate=True)
                    occupied.pop(target_xy, None)
            return True
        return self._step_toward(ent, pos, target_xy, occupied)

    def _cook_at_stove(
        self,
        ent: int,
        pos: Position,
        inventory: Inventory,
        stoves: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        stoves = self._reachable(pos, stoves)
        if not stoves:
            return False
        target_xy, _stove_ent = min(stoves, key=lambda s: _chebyshev((pos.x, pos.y), s[0]))
        if _chebyshev((pos.x, pos.y), target_xy) == 1:
            raw = next((item for item in inventory.items if is_raw_meat(item)), None)
            if raw is not None and WOOD in inventory.items:
                inventory.items.remove(WOOD)
                inventory.items.remove(raw)
                inventory.items.append(cook_meat(raw))
            return True
        return self._step_toward(ent, pos, target_xy, occupied)

    def _feed_cook(
        self,
        ent: int,
        pos: Position,
        prey: list[tuple[tuple[int, int], int]],
        corpses: list[tuple[tuple[int, int], int]],
        trees: list[tuple[tuple[int, int], int]],
        stoves: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        if not esper.has_component(ent, Inventory):
            return False
        inventory = esper.component_for_entity(ent, Inventory)
        needs = esper.component_for_entity(ent, Needs) if esper.has_component(ent, Needs) else None
        has_raw = any(is_raw_meat(item) for item in inventory.items)
        if not has_raw:
            if self._forage_meat(ent, pos, inventory, prey, corpses, occupied):
                return True
            # No reachable game or meat in this area: rather than starve, forage
            # food from the trees around (nuts/berries -- renewable, unlike deer).
            if needs is not None:
                return self._graze(ent, pos, needs, trees, occupied)
            return False
        if WOOD in inventory.items:
            return self._cook_at_stove(ent, pos, inventory, stoves, occupied)
        return self._gather_wood(ent, pos, inventory, trees, occupied)

    def _seek_sleep(
        self, ent: int, pos: Position, needs: Needs, occupied: dict[tuple[int, int], int]
    ) -> bool:
        """A tired NPC heads for bed. It prefers its home tile, walking there a
        step at a time; homeless creatures (and any too exhausted to make it
        home) camp where they stand."""
        home = None
        if esper.has_component(ent, Home):
            home = esper.component_for_entity(ent, Home)

        if home is None:
            go_to_sleep(ent, in_camp=True)
            return True

        home_xy = (home.x, home.y)
        if (pos.x, pos.y) == home_xy:
            go_to_sleep(ent, in_camp=False)
            return True

        if self._step_toward(ent, pos, home_xy, occupied):
            return True

        # Couldn't advance toward home. Camp where we stand if we're spent, or if
        # home is genuinely unreachable (blocked by water/walls) -- better to camp
        # and recover than to idle at a barrier, pinning tiredness and starving.
        if needs.tiredness >= _EXHAUSTED_THRESHOLD or not self.game_map.same_region((pos.x, pos.y), home_xy):
            go_to_sleep(ent, in_camp=True)
            return True
        return False

    def _ensure_inventory(self, ent: int) -> Inventory:
        if esper.has_component(ent, Inventory):
            return esper.component_for_entity(ent, Inventory)
        inventory = Inventory(items=[])
        esper.add_component(ent, inventory)
        return inventory

    def _actor_of(self, ent: int) -> Actor:
        """This NPC's action-economy bookkeeping (its per-region-turn energy),
        created on first use so creatures don't need one at spawn."""
        if esper.has_component(ent, Actor):
            return esper.component_for_entity(ent, Actor)
        actor = Actor()
        esper.add_component(ent, actor)
        return actor

    def _should_build(self, ent: int) -> bool:
        """True when raising a home is this NPC's job right now: a resident that
        owns no bed. Such a villager pitches in on the nearest blueprint -- its
        own staked-out cabin or a neighbour's -- so building is shared labour."""
        return esper.has_component(ent, Resident) and owned_bed_of(ent) is None

    def _reachable_ghosts(self, pos: Position) -> list[tuple[int, tuple[int, int], Blueprint]]:
        """Every blueprint ghost in the same walkable region as ``pos`` -- the
        pieces this worker can actually get to. Anybody's ghosts count, so
        villagers converge on whatever proto-structure is nearest."""
        here = (pos.x, pos.y)
        out: list[tuple[int, tuple[int, int], Blueprint]] = []
        for g_ent, (g_pos, bp) in esper.get_components(Position, Blueprint):
            gxy = (g_pos.x, g_pos.y)
            if self.game_map.tile_at(gxy[0], gxy[1]) == bp.tile:
                continue  # already raised elsewhere; ignore this stale ghost
            if here == gxy or self.game_map.same_region(here, gxy):
                out.append((g_ent, gxy, bp))
        return out

    def _work_blueprints(
        self,
        ent: int,
        pos: Position,
        trees: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Take one turn of work on the nearest reachable blueprint. While any
        reachable piece still lacks materials the worker **hauls wood** to it
        (lighting it up as its wood arrives); once the reachable pieces are all
        stocked it **raises** them into real tiles, a chunk a turn.

        Returns False when there's nothing this worker can actually do on the
        site right now -- no wood to haul *and* nothing stocked left to raise --
        so a builder who's run the woods dry drops out of build mode and gets on
        with other things instead of freezing beside a site it can't advance.
        Any pieces it already stocked still get raised (below), so the labour
        isn't wasted and the shell keeps rising as far as the materials reached;
        an unstocked piece stays a walkable gap, so raising the stocked ones
        never seals off the rest.
        """
        ghosts = self._reachable_ghosts(pos)
        if not ghosts:
            return False
        unstocked = {gxy: g_ent for g_ent, gxy, bp in ghosts if not bp.stocked}
        # Keep hauling while the woods can still supply the unfinished pieces;
        # _haul_to_ghosts returns False once there's no wood on hand and nothing
        # reachable to fell, at which point we raise whatever is already stocked.
        if unstocked and self._haul_to_ghosts(ent, pos, unstocked, trees, occupied):
            return True
        stocked = {gxy: g_ent for g_ent, gxy, bp in ghosts if bp.stocked}
        if stocked:
            return self._raise_nearby_ghost(ent, pos, stocked, occupied)
        return False

    def _haul_to_ghosts(
        self,
        ent: int,
        pos: Position,
        unstocked: dict[tuple[int, int], int],
        trees: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Carry wood to the proto-structure: gather a batch, walk it to the
        nearest ghost lacking materials, and drop it off -- each wood delivered
        lights one ghost up as "ready"."""
        inventory = self._ensure_inventory(ent)
        wood = inventory.items.count(WOOD)
        batch = min(_HAUL_BATCH, len(unstocked))
        # Top up the load before making the trip, unless the woods are tapped out.
        if wood < batch and self._reachable(pos, trees):
            return self._gather_wood(ent, pos, inventory, trees, occupied)
        if wood == 0:
            return False  # nothing to carry and no reachable trees -- can't progress

        by_distance = sorted(unstocked, key=lambda xy: _chebyshev((pos.x, pos.y), xy))
        if _chebyshev((pos.x, pos.y), by_distance[0]) <= 1:
            stocked_any = False
            for gxy in by_distance:
                if WOOD not in inventory.items:
                    break
                inventory.items.remove(WOOD)
                _set_blueprint_stocked(unstocked[gxy], True)
                stocked_any = True
            return stocked_any
        # Ghost tiles don't block, so the worker can walk right up to the site.
        return self._step_toward(ent, pos, by_distance[0], occupied)

    def _raise_nearby_ghost(
        self,
        ent: int,
        pos: Position,
        stocked: dict[tuple[int, int], int],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Raise a stocked ghost we're standing next to (a chunk a turn); a
        finished cabin furnishes itself. Otherwise approach the nearest ghost,
        treating still-standing ghosts as blocked so the worker stops beside them
        instead of on them (a raised wall's tile is unwalkable anyway)."""
        here = (pos.x, pos.y)
        for gxy, ghost in stocked.items():
            if _chebyshev(here, gxy) == 1:
                raise_blueprint(self.game_map, ghost)
                return True

        # Standing *on* a stocked ghost with none adjacent (it delivered wood to
        # this piece from here): a wall can't be raised under our own feet, so
        # step off onto any open tile and raise it from beside next turn. Without
        # this the nearest ghost is the one we're on -- distance 0 -- and every
        # "approach" below is a no-op step toward our own tile, pinning the
        # builder there for good.
        if here in stocked:
            for nxy in self.game_map.neighbors_8(here[0], here[1]):
                if (
                    nxy not in occupied
                    and nxy not in stocked
                    and self.game_map.is_walkable(nxy[0], nxy[1])
                ):
                    self._commit_step(ent, pos, nxy, occupied)
                    return True
            return False  # boxed in on our own ghost; nothing to do this turn

        target = min(stocked, key=lambda xy: _chebyshev(here, xy))
        added: list[tuple[int, int]] = []
        for gxy in stocked:
            if gxy not in occupied:
                occupied[gxy] = -1  # sentinel: not a real entity, just blocked
                added.append(gxy)
        moved = self._step_toward(ent, pos, target, occupied)
        for xy in added:
            if occupied.get(xy) == -1:
                del occupied[xy]
        return moved

    def _chase_player(
        self,
        ent: int,
        pos: Position,
        player_xy: tuple[int, int],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        vision_radius = 8
        if esper.has_component(ent, Vision):
            vision_radius = esper.component_for_entity(ent, Vision).radius
        if _chebyshev((pos.x, pos.y), player_xy) > vision_radius:
            return False
        if not self.game_map.has_line_of_sight((pos.x, pos.y), player_xy):
            return False
        # Line of sight crosses water/gaps a walker can't cross (it only blocks
        # on walls) -- so a hostile that can *see* the player across a river it
        # can't reach would otherwise retry a full, expensive "no path exists"
        # pathfind every single turn it keeps watching. Reachability is a
        # cheap, cached lookup (GameMap.region_of); check it before ever
        # calling into pathfinding.
        if not self.game_map.same_region((pos.x, pos.y), player_xy):
            return False
        if _chebyshev((pos.x, pos.y), player_xy) == 1:
            return False  # adjacent: hold (player-facing combat is player-driven)
        return self._step_toward(ent, pos, player_xy, occupied)

    def _pick_social_partner(
        self,
        ent: int,
        pos: Position,
        sentients: list[tuple[int, Position]],
    ) -> tuple[int, Position] | None:
        """Who ``ent`` would most like to approach: the nearest awake, reachable
        being it can see, weighted so higher current friendship wins over a
        slightly closer stranger. Returns ``(entity, position)`` or ``None``."""
        rel = (
            esper.component_for_entity(ent, Relationships)
            if esper.has_component(ent, Relationships)
            else None
        )
        best: tuple[int, Position] | None = None
        best_score: float | None = None
        for other, other_pos in sentients:
            if other == ent or esper.has_component(other, Asleep):
                continue
            dist = _chebyshev((pos.x, pos.y), (other_pos.x, other_pos.y))
            if dist > _SOCIAL_SIGHT:
                continue
            if not self.game_map.same_region((pos.x, pos.y), (other_pos.x, other_pos.y)):
                continue
            # Prefer friends: friendship pulls the score up, distance pushes it
            # down. A stranger (0 friendship) is still chosen when nobody
            # closer or friendlier is around.
            score = friendship(rel, other) - dist * _SOCIAL_DISTANCE_PENALTY
            if best_score is None or score > best_score:
                best_score = score
                best = (other, other_pos)
        return best

    def _socialize(
        self,
        ent: int,
        pos: Position,
        occupied: dict[tuple[int, int], int],
        sentients: list[tuple[int, Position]],
        turn: int,
        clock: WorldClock | None,
    ) -> bool:
        """A being seeks out someone to talk to, preferring beings it already
        likes. Interacts when adjacent (adjusting friendship + popping bubbles),
        otherwise steps toward the chosen partner. Honours a per-being cooldown so
        villagers don't chatter every turn.

        ``turn``/``clock`` are an as-of turn number and clock during a region
        catch-up burst, not necessarily the true "now" -- see ``_forage_berries``.
        """
        personality = esper.component_for_entity(ent, Personality)
        if turn - personality.last_social_turn < _SOCIAL_COOLDOWN:
            return False

        partner = self._pick_social_partner(ent, pos, sentients)
        if partner is None:
            return False
        partner_ent, partner_pos = partner

        if _chebyshev((pos.x, pos.y), (partner_pos.x, partner_pos.y)) == 1:
            interact(ent, partner_ent, turn)
            # Growing close turns friends into lovers and lovers into spouses: a
            # very close single pair weds; a close pair alone together may be
            # intimate. Both are no-ops unless the pair is adult, opposite-sex,
            # and friendly enough, so this fires only for genuine couples.
            try_marry(ent, partner_ent, clock)
            try_mate(ent, partner_ent, turn, clock, sentients)
            return True
        return self._step_toward(ent, pos, (partner_pos.x, partner_pos.y), occupied)

    def _build_world_snapshot(self) -> dict[str, dict]:
        """One full-world scan, bucketed by each entity's *exact* region (no
        margin -- that's applied at lookup time in ``_region_bucket``). This
        is the expensive part; ``_advance_region`` reuses the result across
        many calls instead of repeating it per region-turn."""
        region_at = self.scheduler.region_at
        occupied = {
            (pos.x, pos.y): ent
            for ent, (pos, _blocks) in esper.get_components(Position, BlocksMovement)
        }
        trees: dict[RegionId, list] = {}
        for ent, (pos, _t) in esper.get_components(Position, Tree):
            trees.setdefault(region_at(pos.x, pos.y), []).append(((pos.x, pos.y), ent))
        prey: dict[RegionId, list] = {}
        for ent, (pos, _d) in esper.get_components(Position, Deer):
            prey.setdefault(region_at(pos.x, pos.y), []).append(((pos.x, pos.y), ent))
        corpses: dict[RegionId, list] = {}
        for ent, (pos, _c, inv) in esper.get_components(Position, Corpse, Inventory):
            if any(is_raw_meat(item) or is_cooked_meat(item) for item in inv.items):
                corpses.setdefault(region_at(pos.x, pos.y), []).append(((pos.x, pos.y), ent))
        stoves: dict[RegionId, list] = {}
        for ent, (pos, _s) in esper.get_components(Position, Stove):
            stoves.setdefault(region_at(pos.x, pos.y), []).append(((pos.x, pos.y), ent))
        bushes: dict[RegionId, list] = {}
        for ent, (pos, bush) in esper.get_components(Position, BerryBush):
            if bush.has_berries:
                bushes.setdefault(region_at(pos.x, pos.y), []).append(((pos.x, pos.y), ent))
        # Sentients/NPCs keep the live Position object, not a frozen (x, y) --
        # they move every turn, so a snapshot reused for many calls must keep
        # reading their *current* position, not the one at snapshot time.
        sentients: dict[RegionId, list] = {}
        for ent, (pos, _p) in esper.get_components(Position, Personality):
            sentients.setdefault(region_at(pos.x, pos.y), []).append((ent, pos))
        npcs: dict[RegionId, list] = {}
        for ent, (pos, _npc) in esper.get_components(Position, NPC):
            npcs.setdefault(region_at(pos.x, pos.y), []).append((ent, pos))
        return {
            "occupied": occupied,
            "trees": trees,
            "prey": prey,
            "corpses": corpses,
            "stoves": stoves,
            "bushes": bushes,
            "sentients": sentients,
            "npcs": npcs,
        }

    def _world_snapshot_for_this_call(self) -> dict[str, dict]:
        if self._world_snapshot is None or self._world_snapshot_calls_left <= 0:
            self._world_snapshot = self._build_world_snapshot()
            self._world_snapshot_calls_left = _WORLD_SNAPSHOT_REFRESH_CALLS
        else:
            self._world_snapshot_calls_left -= 1
        return self._world_snapshot

    def _region_bucket(
        self, buckets: dict[RegionId, list], region_id: RegionId, xy_of: Callable[[object], tuple[int, int]]
    ) -> list:
        """This region's own bucketed items, widened by ``_REGION_BORDER_MARGIN``
        into the (up to 8) neighbouring regions' buckets -- so a creature near a
        seam still sees a resource one tile into the next region. Only the
        handful of neighbouring buckets are scanned, never the whole world."""
        game_map = self.game_map
        margin = _REGION_BORDER_MARGIN
        cx, cy = region_id
        items = list(buckets.get(region_id, ()))
        for nx in range(cx - 1, cx + 2):
            for ny in range(cy - 1, cy + 2):
                if (nx, ny) == region_id:
                    continue
                for item in buckets.get((nx, ny), ()):
                    x, y = xy_of(item)
                    if in_region_with_margin(game_map, region_id, x, y, margin):
                        items.append(item)
        return items

    def _advance_region(self, region_id: RegionId) -> None:
        """Run one turn of NPC AI for the NPCs standing in ``region_id``,
        using a periodically-refreshed world snapshot rather than rescanning
        every entity in the world on every single call (see
        ``_world_snapshot_for_this_call``). ``player_xy`` is fetched fresh
        every call regardless -- it's a single cheap lookup, not a full scan,
        and hostile-chase distance is short-range enough that staleness here
        would actually be noticeable.
        """
        snapshot = self._world_snapshot_for_this_call()
        occupied = snapshot["occupied"]

        # This region's own logical turn: during a catch-up burst this replays
        # turn N, N+1, N+2 ... in order, each with its own as-of clock, so
        # turn-stamped state (berry regrowth, courtship cooldowns, ages) reads
        # correctly relative to *this* region's history, not the true "now".
        # The scheduler counts in whole region-turns; the world clock and every
        # turn-stamped value are in TU, so convert (one region-turn == one
        # baseline action == BASE_ACTION_COST TU) before stamping.
        logical_turn = (self.scheduler.region_turn[region_id] + 1) * BASE_ACTION_COST
        real_clock = world_clock()
        clock = replace(real_clock, turn=logical_turn) if real_clock is not None else None

        player_xy = self._find_player_position()

        xy_of_pair = lambda item: item[0]  # noqa: E731 -- ((x, y), ent) items
        xy_of_pos = lambda item: (item[1].x, item[1].y)  # noqa: E731 -- (ent, Position) items

        trees = self._region_bucket(snapshot["trees"], region_id, xy_of_pair)
        prey = self._region_bucket(snapshot["prey"], region_id, xy_of_pair)
        corpses = self._region_bucket(snapshot["corpses"], region_id, xy_of_pair)
        stoves = self._region_bucket(snapshot["stoves"], region_id, xy_of_pair)
        bushes = self._region_bucket(snapshot["bushes"], region_id, xy_of_pair)
        sentients = self._region_bucket(snapshot["sentients"], region_id, xy_of_pos)

        # NPCs acting this turn are exactly this region's own bucket (no
        # margin) -- a border-straddling NPC must belong to exactly one
        # region's turn, never both, or it would act twice.
        for ent, pos in list(snapshot["npcs"].get(region_id, ())):
            if not esper.entity_exists(ent):
                continue
            # Sleepers skip their turn; NeedsProcessor recovers and wakes them.
            if esper.has_component(ent, Asleep):
                continue

            # Anti-oscillation guard: forbid an immediate one-tile reversal back
            # onto the tile this NPC started its *previous* turn on. A productive
            # route never needs to reverse in one step (the flow field is
            # monotonic toward its goal); a two-tile ping-pong only arises when a
            # blocked higher drive (say, seeking water walled off by trees the
            # pathfinder sees straight through) hands off to a lower one pulling
            # the other way, turn after turn. Blocking that single tile for just
            # this NPC's turn -- via the same occupant sentinel the step code
            # already respects -- collapses the jitter into standing still, which
            # is what an NPC with nowhere better to go should do anyway.
            start_xy = (pos.x, pos.y)
            guard = self._prev_turn_pos.get(ent)
            if guard is not None and (guard == start_xy or guard in occupied):
                guard = None
            if guard is not None:
                occupied[guard] = _OSC_GUARD
            try:
                # Action economy: this region-turn is one baseline action's worth
                # of time, so grant BASE_ACTION_COST energy and let the NPC act as
                # many times as its speed allows -- a quicker creature (higher
                # dexterity, lower action cost) banks the surplus and acts again.
                # A baseline NPC (cost == BASE_ACTION_COST) acts exactly once,
                # preserving the old one-turn-per-region-turn cadence.
                actor = self._actor_of(ent)
                actor.energy += BASE_ACTION_COST
                cost = action_cost(ent, None)
                acted = 0
                while actor.energy >= cost and acted < _MAX_ACTIONS_PER_REGION_TURN:
                    self._take_turn(
                        ent, pos, occupied, player_xy, diet_buckets=(
                            trees, prey, corpses, stoves, bushes, sentients
                        ), logical_turn=logical_turn, clock=clock,
                    )
                    actor.energy -= cost
                    acted += 1
                    if not esper.entity_exists(ent):
                        break
            finally:
                if guard is not None and occupied.get(guard) == _OSC_GUARD:
                    del occupied[guard]
                self._prev_turn_pos[ent] = start_xy

    def _take_turn(
        self,
        ent: int,
        pos: Position,
        occupied: dict[tuple[int, int], int],
        player_xy: tuple[int, int] | None,
        diet_buckets: tuple,
        logical_turn: int,
        clock: "WorldClock | None",
    ) -> None:
        """One NPC's single-turn behaviour: the priority ladder of survival
        drives, then building, then leisure, then hostile pursuit. Split out of
        ``_advance_region`` so the per-turn anti-oscillation guard there can wrap
        it cleanly."""
        trees, prey, corpses, stoves, bushes, sentients = diet_buckets
        acted = False

        if esper.has_component(ent, Needs):
            needs = esper.component_for_entity(ent, Needs)
            diet_kind = None
            if esper.has_component(ent, Diet):
                diet_kind = esper.component_for_entity(ent, Diet).kind

            # Sleep is the strongest drive: a spent creature beds down before
            # it forages, preferring its home over camping.
            if needs.tiredness >= _SLEEP_THRESHOLD:
                acted = self._seek_sleep(ent, pos, needs, occupied)
            # Thirst wins ties -- a parched animal drinks before it eats.
            elif needs.thirst >= _FORAGE_THRESHOLD and needs.thirst >= needs.hunger:
                acted = self._seek_water(ent, pos, needs, occupied)
            elif needs.hunger >= _FORAGE_THRESHOLD:
                # Eat prepared food already carried; otherwise forage by diet.
                if self._eat_from_inventory(ent, needs):
                    acted = True
                elif diet_kind == "herbivore":
                    # Ripe berries first (quick food); else graze a tree.
                    acted = self._forage_berries(ent, pos, needs, bushes, occupied, clock)
                    if not acted:
                        acted = self._graze(ent, pos, needs, trees, occupied)
                elif diet_kind == "carnivore":
                    # Predator: eats raw meat on the spot.
                    acted = self._seek_food(ent, pos, needs, prey, corpses, occupied)
                elif diet_kind == "cook":
                    # Villager: pick ripe berries when handy, else cook meat.
                    acted = self._forage_berries(ent, pos, needs, bushes, occupied, clock)
                    if not acted:
                        acted = self._feed_cook(ent, pos, prey, corpses, trees, stoves, occupied)

        # Building a house is the lowest-priority survival drive: a homeless
        # resident helps raise the nearest blueprint only once fed, watered,
        # and rested. Labour is shared -- several villagers can work one site.
        if not acted and self._should_build(ent):
            acted = self._work_blueprints(ent, pos, trees, occupied)

        # Leisure: a fed, rested, non-hostile being with a personality seeks
        # out company -- preferring beings it already likes. Lowest priority
        # of all, so survival and building always come first.
        if (
            not acted
            and esper.has_component(ent, Personality)
            and esper.has_component(ent, Friendly)
        ):
            acted = self._socialize(ent, pos, occupied, sentients, logical_turn, clock)

        if acted:
            return

        if player_xy is not None and esper.has_component(ent, Enemy):
            self._chase_player(ent, pos, player_xy, occupied)

    def process(self, action: str | None = None) -> None:
        if action not in _TURN_ACTIONS:
            return

        player_xy = self._find_player_position()
        player_region = (
            self.scheduler.region_at(player_xy[0], player_xy[1]) if player_xy is not None else None
        )

        if player_region is None:
            # No player (unit tests construct this processor directly): bring
            # every region up to date once, matching an unpartitioned pass.
            for region_id in all_region_ids(self.game_map):
                self.scheduler.advance_region(region_id)
            return

        target_turn = self.scheduler.next_turn_for(player_region, _current_region_turn())

        # The player's own region is always fully live -- this also covers
        # "just entered a new region": catch_up_region replays every turn the
        # region missed, in order, right here.
        self.scheduler.catch_up_region(player_region, target_turn)

        # Background: nudge the *nearest* other lagging regions along, closest
        # to the player first (never the stalest). Bounded, so it can never
        # stall input; the main loop's idle-time pump tops this up further,
        # and sleep/region-entry fully resolve everything else.
        self.scheduler.pump_background(
            _NPC_BACKGROUND_BUDGET, player_region, target_turn, self._wall_clock
        )


class FishAiProcessor(esper.Processor):
    """Swims the fish each turn. A fish is the aquatic mirror of a grazing deer:
    when hungry it eats seaweed it is next to, otherwise it drifts toward the
    nearest seaweed it can see, and failing that mills about at random.

    Fish move on water tiles only, so -- unlike the land creatures the
    ``NpcAiProcessor`` drives with ``find_path`` -- they step greedily between
    adjacent water cells and never strand themselves ashore. They are not tagged
    ``NPC``, so the land AI ignores them entirely.

    Uses a ``RegionScheduler`` (see ``regions.py``) so the player's own 120x60
    region is always fully live, distant regions pay down simulation debt in
    the background (nearest-to-player first), and either a region-entry or a
    sleep can force a region -- or the whole world -- fully up to date.
    """

    def __init__(
        self,
        game_map: GameMap,
        rng: Callable[[], float] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.game_map = game_map
        self._rng = rng if rng is not None else world_rng().stream("ai").random
        self._wall_clock = clock if clock is not None else time.monotonic
        self.scheduler = RegionScheduler(game_map, _current_region_turn())
        self.scheduler.register("fish", self._advance_region)
        # Rebuilt from scratch only every _WORLD_SNAPSHOT_REFRESH_CALLS
        # advances rather than on every single one -- see the identical cache
        # on NpcAiProcessor for why a catch-up burst makes this matter.
        self._world_snapshot: dict[str, dict] | None = None
        self._world_snapshot_calls_left = 0

    def _build_world_snapshot(self) -> dict[str, dict]:
        region_at = self.scheduler.region_at
        fish_by_region: dict[RegionId, list] = {}
        occupied: dict[tuple[int, int], int] = {}
        for ent, (pos, _f) in esper.get_components(Position, Fish):
            fish_by_region.setdefault(region_at(pos.x, pos.y), []).append((ent, pos))
            occupied[(pos.x, pos.y)] = ent
        seaweed_by_region: dict[RegionId, list] = {}
        for ent, (pos, _s) in esper.get_components(Position, Seaweed):
            seaweed_by_region.setdefault(region_at(pos.x, pos.y), []).append(((pos.x, pos.y), ent))
        return {"fish": fish_by_region, "seaweed": seaweed_by_region, "occupied": occupied}

    def _world_snapshot_for_this_call(self) -> dict[str, dict]:
        if self._world_snapshot is None or self._world_snapshot_calls_left <= 0:
            self._world_snapshot = self._build_world_snapshot()
            self._world_snapshot_calls_left = _WORLD_SNAPSHOT_REFRESH_CALLS
        else:
            self._world_snapshot_calls_left -= 1
        return self._world_snapshot

    def _advance_region(self, region_id: RegionId) -> None:
        snapshot = self._world_snapshot_for_this_call()
        fish = list(snapshot["fish"].get(region_id, ()))
        seaweed = list(snapshot["seaweed"].get(region_id, ()))
        # Shared across the whole world so a fish never swims onto a resting
        # neighbour, even one in a region we're not simulating right now.
        occupied = snapshot["occupied"]
        self._simulate_area(fish, seaweed, occupied)

    def process(self, action: str | None = None) -> None:
        # Fish move only on real turns -- never on idle/menu ticks -- so the
        # animation loop's rapid ``process(None)`` calls can never block.
        if action not in _TURN_ACTIONS:
            return

        player_region = None
        for _ent, (pos, _player) in esper.get_components(Position, Player):
            player_region = self.scheduler.region_at(pos.x, pos.y)
            break

        if player_region is None:
            # No player (unit tests construct this processor directly): bring
            # every region up to date once, matching an unpartitioned pass.
            for region_id in all_region_ids(self.game_map):
                self.scheduler.advance_region(region_id)
            return

        target_turn = self.scheduler.next_turn_for(player_region, _current_region_turn())

        # The player's own region is always fully live -- this also covers
        # "just entered a new region": catch_up_region replays every turn the
        # region missed, in order, right here.
        self.scheduler.catch_up_region(player_region, target_turn)

        # Background: spend a small time budget nudging the *nearest* other
        # lagging regions along, closest to the player first. Bounded, so it
        # can never stall input; regions it doesn't reach get a bigger budget
        # from the main loop's idle-time pump, and are fully resolved the
        # moment the player enters them (above) or the player sleeps.
        self.scheduler.pump_background(
            _FISH_BACKGROUND_BUDGET, player_region, target_turn, self._wall_clock
        )

    def _simulate_area(
        self,
        fish: list[tuple[int, Position]],
        seaweed: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> None:
        """Swim one area's fish for a single step against that area's seaweed."""
        seaweed_at = {xy: sw_ent for xy, sw_ent in seaweed}
        for ent, pos in fish:
            if not esper.entity_exists(ent):
                continue
            needs = (
                esper.component_for_entity(ent, Needs)
                if esper.has_component(ent, Needs)
                else None
            )
            hungry = needs is not None and needs.hunger >= _FORAGE_THRESHOLD

            if hungry:
                bite = next(
                    (
                        (nx, ny)
                        for nx, ny in self.game_map.neighbors_8(pos.x, pos.y)
                        if (nx, ny) in seaweed_at
                    ),
                    None,
                )
                if bite is not None:
                    self._graze_seaweed(seaweed_at[bite], bite, needs, seaweed_at, occupied)
                    continue
                target = self._nearest_seaweed(pos, seaweed)
                if target is not None and self._step_toward(ent, pos, target, occupied):
                    continue

            self._wander(ent, pos, occupied)

    def _water_neighbors(
        self, x: int, y: int, occupied: dict[tuple[int, int], int]
    ) -> list[tuple[int, int]]:
        return [
            (nx, ny)
            for nx, ny in self.game_map.neighbors_8(x, y)
            if self.game_map.is_water(nx, ny) and (nx, ny) not in occupied
        ]

    def _nearest_seaweed(
        self, pos: Position, seaweed: list[tuple[tuple[int, int], int]]
    ) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        best_dist: int | None = None
        for xy, _ent in seaweed:
            dist = _chebyshev((pos.x, pos.y), xy)
            if dist > _FISH_SIGHT:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = xy
        return best

    def _step_toward(
        self, ent: int, pos: Position, goal: tuple[int, int], occupied: dict[tuple[int, int], int]
    ) -> bool:
        """Greedily step to the open water neighbour nearest the goal."""
        options = self._water_neighbors(pos.x, pos.y, occupied)
        if not options:
            return False
        nx, ny = min(options, key=lambda xy: _chebyshev(xy, goal))
        self._move(ent, pos, nx, ny, occupied)
        return True

    def _wander(self, ent: int, pos: Position, occupied: dict[tuple[int, int], int]) -> None:
        if self._rng() >= _FISH_WANDER_CHANCE:
            return
        options = self._water_neighbors(pos.x, pos.y, occupied)
        if not options:
            return
        nx, ny = options[min(len(options) - 1, int(self._rng() * len(options)))]
        self._move(ent, pos, nx, ny, occupied)

    def _graze_seaweed(
        self,
        sw_ent: int,
        xy: tuple[int, int],
        needs: Needs,
        seaweed_at: dict[tuple[int, int], int],
        occupied: dict[tuple[int, int], int],
    ) -> None:
        needs.hunger = max(0.0, needs.hunger - _FISH_GRAZE_RESTORE)
        if esper.entity_exists(sw_ent) and esper.has_component(sw_ent, Seaweed):
            frond = esper.component_for_entity(sw_ent, Seaweed)
            frond.food -= 1
            if frond.food <= 0:
                esper.delete_entity(sw_ent, immediate=True)  # eaten bare
                seaweed_at.pop(xy, None)

    def _move(
        self, ent: int, pos: Position, nx: int, ny: int, occupied: dict[tuple[int, int], int]
    ) -> None:
        occupied.pop((pos.x, pos.y), None)
        pos.x, pos.y = nx, ny
        occupied[(nx, ny)] = ent


class HousingProcessor(esper.Processor):
    """Settles residents into homes. Houses belong to people: a resident who owns
    one is left alone. A resident who owns none **claims the nearest unowned
    house** it can reach (marking it as theirs); only if none is free does it
    **stake out a cabin blueprint** (a world ``ConstructionSite``) -- but only if
    there isn't already one it can reach, so several homeless villagers share a
    single site and raise it together rather than each starting their own.

    House detection is cached and only rebuilt when the map changes, so this is a
    cheap per-turn pass on a static map.
    """

    def __init__(self, game_map: GameMap):
        self.game_map = game_map
        self._cache: dict = {}

    def process(self, action: str | None = None) -> None:
        if action not in _TURN_ACTIONS:
            return
        houses = houses_for(self.game_map, self._cache)
        sites = [(e, comp) for e, (comp,) in esper.get_components(ConstructionSite)]

        for ent, (_res,) in list(esper.get_components(Resident)):
            if owned_bed_of(ent) is not None:
                continue  # already owns a house -- never re-claim or rebuild
            if self._handle_spouse_housing(ent):
                continue  # married couples settle into one shared home
            if self._claim_unowned_house(ent, houses):
                continue
            self._ensure_site_for(ent, sites)

    def _handle_spouse_housing(self, ent: int) -> bool:
        """Keep a married couple under one roof. Returns True (handled: skip
        claiming/building) when ``ent`` has a spouse and either:
          * the spouse already owns a home -- move in and share the bed, or
          * the spouse is the one settling the household (building, or simply the
            designated partner) -- wait this turn and move in once it's ready.
        Returns False only when ``ent`` itself should go get the home, so exactly
        one spouse ever claims or builds."""
        if not esper.has_component(ent, Family):
            return False
        spouse = esper.component_for_entity(ent, Family).spouse
        if spouse is None or not esper.entity_exists(spouse):
            return False

        if owned_bed_of(spouse) is not None and esper.has_component(spouse, Home):
            home = esper.component_for_entity(spouse, Home)
            _set_home(ent, (home.x, home.y))
            return True

        # Neither owns a home yet. Let just one partner do the settling so they
        # don't claim two separate houses: the lower entity id is the settler; the
        # other waits and moves in once its partner has a home.
        return ent > spouse

    def _claim_unowned_house(self, ent: int, houses: list[frozenset[tuple[int, int]]]) -> bool:
        pos = esper.component_for_entity(ent, Position) if esper.has_component(ent, Position) else None
        candidates: list[tuple[frozenset[tuple[int, int]], tuple[int, int]]] = []
        for interior in houses:
            bed_ent = _bed_in_interior(interior)
            if bed_ent is None or bed_owner(bed_ent) is not None:
                continue  # no bed, or already someone's house
            bed_pos = esper.component_for_entity(bed_ent, Position)
            candidates.append((interior, (bed_pos.x, bed_pos.y)))
        if not candidates:
            return False
        if pos is not None:
            candidates.sort(key=lambda c: _chebyshev((pos.x, pos.y), c[1]))

        for interior, bed_xy in candidates:
            # Only claim a house the villager can actually walk to -- otherwise it
            # would strand itself commuting to a home across water/walls and never
            # get back to foraging. If none are reachable, it builds one instead.
            if pos is not None and (pos.x, pos.y) != bed_xy and not self.game_map.same_region((pos.x, pos.y), bed_xy):
                continue
            set_house_ownership(interior, ent)  # bed + chests now belong to them
            if esper.has_component(ent, Home):
                home = esper.component_for_entity(ent, Home)
                home.x, home.y = bed_xy
            else:
                esper.add_component(ent, Home(bed_xy[0], bed_xy[1]))
            return True
        return False

    def _ensure_site_for(self, ent: int, sites: list[tuple[int, ConstructionSite]]) -> None:
        """Make sure this homeless resident has a blueprint to work on. If one it
        can reach already exists, leave it -- the NPC AI will send the villager to
        help raise it. Otherwise stake out a fresh cabin site nearby."""
        if not esper.has_component(ent, Position):
            return
        pos = esper.component_for_entity(ent, Position)
        here = (pos.x, pos.y)
        for _site_ent, site in sites:
            if site.pieces and self._site_reachable(here, site):
                return  # a neighbour already staked one out -- go help build it

        occupied = {(p.x, p.y) for _e, (p, _b) in esper.get_components(Position, BlocksMovement)}
        # Non-blocking things must not be walled in either -- keep new sites off
        # corpses, saplings, and other blueprints.
        for comp in (Corpse, Sapling, Blueprint):
            occupied |= {(p.x, p.y) for _e, (p, _c) in esper.get_components(Position, comp)}
        origin = choose_build_site(self.game_map, here, occupied)
        if origin is None:
            return
        site_ent = create_construction_site(self.game_map, origin)
        sites.append((site_ent, esper.component_for_entity(site_ent, ConstructionSite)))
        _push_turn_event("A villager stakes out a blueprint for a new cabin.")

    def _site_reachable(self, here: tuple[int, int], site: ConstructionSite) -> bool:
        for gxy in site.pieces:
            return here == gxy or self.game_map.same_region(here, gxy)
        return False


# Daily, per-tile odds that shape the flora. A sapling can sprout on any open
# outdoor ground tile; a mature tree/bush can die (rot/fall) at half that rate.
# Berry bushes sprout less often than trees.
_DAILY_SPROUT_CHANCE = 0.0001       # 0.01% per outdoor ground tile per day (trees)
_DAILY_BUSH_SPROUT_CHANCE = 0.00004  # bushes are rarer than trees
_DAILY_DEATH_CHANCE = 0.00005       # 0.005% per plant per day (half the sprout rate)
# Fresh seaweed sprouts faster than land flora so grazing fish never strip the
# ocean bare (they only eat what they swim past).
_DAILY_SEAWEED_SPROUT_CHANCE = 0.0006
# Days after a bush's berries are picked before a fresh crop ripens.
_BERRY_REGROW_DAYS = 7


class TreeGrowthProcessor(esper.Processor):
    """Ages the flora one **day** at a time (it acts on the turn a new day
    begins, matching the per-day odds below).

    Each day: any sapling that has lived a full year (112 days) matures (into a
    tree or a berry bush, per its kind); every tree/bush has a small chance of
    dying; picked bushes regrow their berries once 7 days have passed; and every
    open outdoor ground tile has a small chance of sprouting a fresh sapling. A
    soft cap (scaled to the map) keeps the world from filling solid. The RNG is
    injectable so tests can force or suppress growth.
    """

    def __init__(self, game_map: GameMap, rng: Callable[[], float] | None = None):
        self.game_map = game_map
        self._rng = rng if rng is not None else world_rng().stream("flora").random
        # Flora fills the land, so its soft cap scales to the land area (not the
        # whole map -- otherwise the ocean's tiles would inflate the tree budget).
        land_area = getattr(game_map, "land_w", game_map.width) * getattr(
            game_map, "land_h", game_map.height
        )
        self._cap = max(40, land_area * 3 // 10)
        # Seaweed fills the open sea; its cap scales to the ocean area.
        ocean_area = max(0, (game_map.width * game_map.height) - land_area)
        self._seaweed_cap = ocean_area // 28
        self._last_day: int | None = None
        # Cached lists of outdoor ground tiles (floor, not inside a house) and of
        # open-sea tiles, rebuilt only when the map changes.
        self._ground_cache: dict = {}
        self._ocean_cache: dict = {}

    def process(self, action: str | None = None) -> None:
        if action not in _TURN_ACTIONS:
            return
        clock = world_clock()
        if clock is None:
            return
        day = clock.turn // max(1, clock.day_length)
        if self._last_day is None:
            self._last_day = day  # establish a baseline; age the flora from here
            return
        if day == self._last_day:
            return
        self._last_day = day
        self._mature_saplings(clock)
        self._kill_flora()
        self._regrow_berries(clock)
        self._sprout_saplings(clock)
        if getattr(self.game_map, "has_ocean", False):
            self._sprout_seaweed()

    def _mature_saplings(self, clock: WorldClock) -> None:
        mature_age = _DAYS_PER_YEAR * max(1, clock.day_length)
        blockers = {
            (pos.x, pos.y) for _e, (pos, _b) in esper.get_components(Position, BlocksMovement)
        }
        for ent, (pos, sapling) in list(esper.get_components(Position, Sapling)):
            if clock.turn - sapling.planted_turn < mature_age:
                continue
            if (pos.x, pos.y) in blockers:
                continue  # occupied -- let it mature once the tile clears
            esper.remove_component(ent, Sapling)
            esper.add_component(ent, BlocksMovement())
            if sapling.kind == "bush":
                esper.add_component(ent, BerryBush())
                _set_bush_appearance(ent, True)
                if esper.has_component(ent, Name):
                    esper.component_for_entity(ent, Name).value = "Berry Bush"
            else:
                esper.add_component(ent, Tree())
                if esper.has_component(ent, Renderable):
                    rend = esper.component_for_entity(ent, Renderable)
                    rend.glyph = "T"
                    rend.fg = _TREE_GREEN
                if esper.has_component(ent, Name):
                    esper.component_for_entity(ent, Name).value = "Tree"

    def _kill_flora(self) -> None:
        for ent, (_pos, _tree) in list(esper.get_components(Position, Tree)):
            if self._rng() < _DAILY_DEATH_CHANCE:
                esper.delete_entity(ent, immediate=True)
        for ent, (_pos, _bush) in list(esper.get_components(Position, BerryBush)):
            if self._rng() < _DAILY_DEATH_CHANCE:
                esper.delete_entity(ent, immediate=True)

    def _regrow_berries(self, clock: WorldClock) -> None:
        ready_after = _BERRY_REGROW_DAYS * max(1, clock.day_length)
        for ent, (bush,) in esper.get_components(BerryBush):
            if bush.has_berries or bush.harvested_turn is None:
                continue
            if clock.turn - bush.harvested_turn >= ready_after:
                bush.has_berries = True
                bush.harvested_turn = None
                _set_bush_appearance(ent, True)

    def _outdoor_ground(self) -> list[tuple[int, int]]:
        """Every regular ground tile that is outdoors (floor, and not inside an
        enclosed house). Cached until the map changes."""
        revision = getattr(self.game_map, "revision", 0)
        if self._ground_cache.get("revision") != revision:
            interiors: set[tuple[int, int]] = set()
            for interior in self.game_map.find_enclosed_rooms():
                interiors |= interior
            tiles = [
                (x, y)
                for y in range(1, self.game_map.height - 1)
                for x in range(1, self.game_map.width - 1)
                if self.game_map.tiles[y][x] == self.game_map.FLOOR and (x, y) not in interiors
            ]
            self._ground_cache = {"revision": revision, "tiles": tiles}
        return self._ground_cache["tiles"]

    def _sprout_saplings(self, clock: WorldClock) -> None:
        total = (
            sum(1 for _e, _c in esper.get_components(Tree))
            + sum(1 for _e, _c in esper.get_components(BerryBush))
            + sum(1 for _e, _c in esper.get_components(Sapling))
        )
        if total >= self._cap:
            return
        occupied = {(pos.x, pos.y) for _e, (pos,) in esper.get_components(Position)}
        for x, y in self._outdoor_ground():
            if total >= self._cap:
                break
            if (x, y) in occupied:
                continue
            roll = self._rng()
            if roll < _DAILY_SPROUT_CHANCE:
                kind, glyph, name = "tree", "t", "Sapling"
            elif roll < _DAILY_SPROUT_CHANCE + _DAILY_BUSH_SPROUT_CHANCE:
                kind, glyph, name = "bush", ",", "Bush Seedling"
            else:
                continue
            esper.create_entity(
                Position(x, y),
                Renderable(glyph, fg=_SAPLING_GREEN),
                Name(name),
                Sapling(planted_turn=clock.turn, kind=kind),
            )
            occupied.add((x, y))
            total += 1

    def _ocean_tiles(self) -> list[tuple[int, int]]:
        """Every open-sea tile (water outside the land). Cached until the map
        changes; the ocean is static so this is built at most once."""
        revision = getattr(self.game_map, "revision", 0)
        if self._ocean_cache.get("revision") != revision:
            tiles = [
                (x, y)
                for y in range(1, self.game_map.height - 1)
                for x in range(1, self.game_map.width - 1)
                if self.game_map.is_ocean(x, y)
            ]
            self._ocean_cache = {"revision": revision, "tiles": tiles}
        return self._ocean_cache["tiles"]

    def _sprout_seaweed(self) -> None:
        """Grow fresh seaweed on open water so grazing fish keep the sea fed."""
        total = sum(1 for _e, _c in esper.get_components(Seaweed))
        if total >= self._seaweed_cap:
            return
        occupied = {(pos.x, pos.y) for _e, (pos,) in esper.get_components(Position)}
        for x, y in self._ocean_tiles():
            if total >= self._seaweed_cap:
                break
            if (x, y) in occupied:
                continue
            if self._rng() < _DAILY_SEAWEED_SPROUT_CHANCE:
                esper.create_entity(
                    Position(x, y),
                    Renderable('"', fg=_SEAWEED_GREEN, bg=_WATER_BLUE),
                    Name("Seaweed"),
                    Seaweed(),
                )
                occupied.add((x, y))
                total += 1


# The glyph/colour a newborn villager shares with the adult cast.
_VILLAGER_GLYPH = "v"
_BABY_SKIN = (222, 184, 156)


class ReproductionProcessor(esper.Processor):
    """Delivers babies. Once a day (a cheap day-boundary pass, like the flora)
    it checks every pregnant woman: when the gestation period has elapsed a child
    is born beside her -- named by the onymancer, sharing the family surname, and
    wired to both parents. Newborns live at the mother's home and grow up on the
    world clock; they are not ``Resident`` (a baby doesn't build its own cabin)."""

    def __init__(self) -> None:
        self._last_day: int | None = None
        # A private onymancer for naming newborns, seeded off the world seed so
        # the whole reproduction chain (conception -> birth -> name) is
        # reproducible for a given seed.
        self._onymancer = make_onymancer(world_rng().int_seed("names_newborn"))

    def process(self, action: str | None = None) -> None:
        if action not in _TURN_ACTIONS:
            return
        clock = world_clock()
        if clock is None:
            return
        day = current_day(clock)
        if self._last_day is None:
            self._last_day = day
            return
        if day == self._last_day:
            return
        self._last_day = day
        self._deliver_due_pregnancies(clock)

    def _deliver_due_pregnancies(self, clock: WorldClock) -> None:
        term = _GESTATION_DAYS * max(1, clock.day_length)
        for mother, (pregnant,) in list(esper.get_components(Pregnant)):
            if clock.turn - pregnant.conceived_turn < term:
                continue
            esper.remove_component(mother, Pregnant)
            self._give_birth(mother, pregnant.father, clock)

    def _give_birth(self, mother: int, father: int, clock: WorldClock) -> None:
        if not esper.entity_exists(mother):
            return
        birth_xy = self._birth_tile(mother)
        surname = _family(mother).surname or _surname_of(_entity_display_name(mother))
        gender = self._rng_gender()
        given = self._onymancer.given_name(gender)
        full = f"{given} {surname}"

        repro_rng = world_rng().stream("repro")
        traits = repro_rng.sample(list(_TRAITS), k=repro_rng.randint(1, 2))
        baby = esper.create_entity(
            Position(*birth_xy),
            Renderable(_VILLAGER_GLYPH, fg=_BABY_SKIN),
            Name(full),
            Age(born_turn=clock.turn),
            Gender(gender),
            Family(surname=surname, parents=[father, mother]),
            NPC(),
            Friendly(),
            Dialogue("##!/$*~# GH01^@"),
            BlocksMovement(),
            Inventory(items=[]),
            Equipment(slots={}),
            Diet("cook"),
            Needs(),
            Personality(traits=traits),
            Relationships(),
        )

        # Record the child on both parents (siblings are derived from shared
        # parents, so nothing else to link).
        for parent in (father, mother):
            if esper.entity_exists(parent):
                _family(parent).children.append(baby)

        # The baby lives with its mother.
        if esper.has_component(mother, Home):
            home = esper.component_for_entity(mother, Home)
            _set_home(baby, (home.x, home.y))

        _push_turn_event(
            f"A child, {full}, is born to {_entity_display_name(father)} and {_entity_display_name(mother)}."
        )

    @staticmethod
    def _rng_gender() -> str:
        return world_rng().stream("repro").choice(("male", "female"))

    def _birth_tile(self, mother: int) -> tuple[int, int]:
        """The mother's own tile if we can't do better; otherwise a free adjacent
        tile so the newborn doesn't share a cell (it blocks movement)."""
        if not esper.has_component(mother, Position):
            return (0, 0)
        mpos = esper.component_for_entity(mother, Position)
        occupied = {
            (pos.x, pos.y) for _e, (pos, _b) in esper.get_components(Position, BlocksMovement)
        }
        for nx, ny in _neighbours_8(mpos.x, mpos.y):
            if (nx, ny) not in occupied:
                return (nx, ny)
        return (mpos.x, mpos.y)


def _neighbours_8(x: int, y: int) -> list[tuple[int, int]]:
    return [
        (x + dx, y + dy)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dx == 0 and dy == 0)
    ]


# Hunger/thirst thresholds (percent of max) that emit an escalating warning the
# first turn each is crossed going up. Ordered high -> low so we report the most
# severe newly-crossed level.
_HUNGER_WARNINGS = (
    (100.0, "You are starving!"),
    (80.0, "Your stomach growls with hunger."),
    (50.0, "You are getting hungry."),
)
_THIRST_WARNINGS = (
    (100.0, "You are dying of thirst!"),
    (80.0, "Your throat is parched."),
    (50.0, "You are getting thirsty."),
)
_TIREDNESS_WARNINGS = (
    (100.0, "You can barely keep your eyes open."),
    (80.0, "You are exhausted and need sleep."),
    (50.0, "You are getting tired."),
)

# How much tiredness a single turn of sleep restores.
_SLEEP_RECOVERY = 3.0


def _crossed_warning(
    previous: float,
    current: float,
    warnings: tuple[tuple[float, str], ...],
) -> str | None:
    """Return the message for the highest threshold newly crossed upward this
    turn (``previous < threshold <= current``), or ``None`` if none was."""
    for threshold, message in warnings:
        if previous < threshold <= current:
            return message
    return None


class NeedsProcessor(esper.Processor):
    """Advances hunger/thirst/tiredness by the *time elapsed* since the last turn.

    Needs are a function of world time, not action count: a turn that consumed 200
    TU accrues twice as much as a baseline 100-TU turn, so a slow action makes you
    proportionally hungrier. The per-turn rates on ``Needs`` are defined per
    ``BASE_ACTION_COST`` TU, so a baseline turn accrues exactly the old amount.

    Menu refreshes call ``esper.process(None)``; those must not starve the player,
    so the tick is gated on a turn-advancing action.
    """

    def __init__(self) -> None:
        # World-clock TU at the last accrual, so we can charge only the elapsed span.
        self._last_turn: int | None = None

    def process(self, action: str | None = None) -> None:
        if action not in _TURN_ACTIONS:
            return

        clock = world_clock()
        now = clock.turn if clock is not None else 0
        if self._last_turn is None:
            # First tick: charge exactly one baseline turn (matches old behaviour
            # and a fresh unit test with no advancing clock).
            self._last_turn = now - BASE_ACTION_COST
        elapsed = now - self._last_turn
        self._last_turn = now
        # No advancing clock (e.g. a processor driven directly in a unit test):
        # fall back to one baseline turn's worth so needs still tick per call.
        if elapsed <= 0:
            elapsed = BASE_ACTION_COST
        scale = elapsed / BASE_ACTION_COST

        night = is_night(clock)
        woke: list[int] = []

        for ent, (needs,) in esper.get_components(Needs):
            prev_hunger = needs.hunger
            prev_thirst = needs.thirst
            prev_tiredness = needs.tiredness
            asleep = esper.has_component(ent, Asleep)

            # Hunger and thirst creep up whether awake or asleep.
            needs.hunger = min(needs.max_value, needs.hunger + needs.hunger_rate * scale)
            needs.thirst = min(needs.max_value, needs.thirst + needs.thirst_rate * scale)

            if asleep:
                # Sleeping pays down tiredness; waking is handled after the loop.
                needs.tiredness = max(0.0, needs.tiredness - _SLEEP_RECOVERY * scale)
                if needs.tiredness <= 0.0:
                    woke.append(ent)
            else:
                rate = needs.tiredness_rate * (_NIGHT_TIREDNESS_MULTIPLIER if night else 1.0)
                needs.tiredness = min(needs.max_value, needs.tiredness + rate * scale)

            # Every creature accumulates needs, but only the player's are
            # surfaced as log warnings -- a hungry goblin shouldn't print
            # "You are starving!" to the player.
            if not esper.has_component(ent, Player):
                continue

            hunger_msg = _crossed_warning(prev_hunger, needs.hunger, _HUNGER_WARNINGS)
            if hunger_msg is not None:
                _push_turn_event(hunger_msg)
            thirst_msg = _crossed_warning(prev_thirst, needs.thirst, _THIRST_WARNINGS)
            if thirst_msg is not None:
                _push_turn_event(thirst_msg)
            # Tiredness warnings only make sense while awake (it falls in sleep).
            if not asleep:
                tired_msg = _crossed_warning(prev_tiredness, needs.tiredness, _TIREDNESS_WARNINGS)
                if tired_msg is not None:
                    _push_turn_event(tired_msg)

        for ent in woke:
            wake_up(ent)


class RenderProcessor(esper.Processor):
    """Draws the map, then all Renderable entities on top of it."""

    def __init__(self, renderer: Renderer, game_map: GameMap):
        self.renderer = renderer
        self.game_map = game_map
        self.sidebar_x = game_map.width + 2
        self.sidebar_width = 22
        self._view_origin_x = 0
        self._view_origin_y = 0
        self._view_width = game_map.width
        self._view_height = game_map.height
        # "Look" mode overlay: a world-cell cursor to outline and a status line to
        # show while examining. Both are None during normal play (set/cleared by
        # the turn loop's look handler).
        self.look_cursor: tuple[int, int] | None = None
        self.look_info: str | None = None
        self._status_y = game_map.height
        self._message_log: list[str] = ["You enter the area."]
        self._max_log_lines = 200
        self._last_player_pos: tuple[int, int] | None = None
        self._seen_entity_ids: set[int] = set()
        self._visible_tiles: set[tuple[int, int]] = set()
        # Tile memory ("fog of war"): every tile ever in view is "explored" and
        # keeps being drawn -- desaturated -- once it drops out of line of sight.
        # ``_tile_memory`` remembers the last static scenery (tree, furniture,
        # ...) seen on a tile so it can be redrawn from memory; NPCs and loot are
        # deliberately never stored, so they vanish the moment they leave view.
        self._explored_tiles: set[tuple[int, int]] = set()
        self._tile_memory: dict[
            tuple[int, int],
            tuple[str, str, tuple[int, int, int] | None, tuple[int, int, int] | None],
        ] = {}
        # Last-seen terrain char per explored tile. The living world edits the map
        # out of sight (villagers raise walls, etc.); remembered terrain must show
        # what the player last *saw*, not the current map, so a wall built in an
        # unseen area doesn't appear in memory until it's actually seen.
        self._seen_terrain: dict[tuple[int, int], str] = {}
        # Per-view memory geometry, recomputed whenever the player moves (mirrors
        # the FOV memo below): the desaturated region to blit, the never-seen
        # holes inside it to black out, the explored-but-unseen cells, the
        # remembered scenery to overlay on them, and the terrain overrides (cells
        # whose live terrain has since diverged from what was last seen).
        self._memory_bbox: tuple[int, int, int, int] | None = None
        self._memory_holes: tuple[tuple[int, int], ...] = ()
        self._memory_cells: frozenset[tuple[int, int]] = frozenset()
        self._memory_scenery: tuple[
            tuple[tuple[int, int], tuple[str, str, tuple[int, int, int] | None, tuple[int, int, int] | None]],
            ...,
        ] = ()
        self._memory_terrain_overrides: tuple[tuple[tuple[int, int], str], ...] = ()
        # Render caches (big win on web, where every re-render is main-thread work
        # that can stutter audio). Field-of-view only changes when the player
        # moves; wall autotile masks never change on a static map.
        self._visible_cache_key: tuple[int | None, int, int] | None = None
        self._wall_mask_cache: dict[tuple[int, int], int] = {}
        self._water_mask_cache: dict[tuple[int, int], int] = {}
        # Last map revision drawn. When the map mutates at runtime (a wall or door
        # gets built), the cached map surface, autotile masks, and FOV memo all go
        # stale, so we drop them and let the next frame rebuild.
        self._map_revision = getattr(game_map, "revision", 0)
        # Per-player-position FOV memo: pos -> (visible set, view bbox, shadow
        # cells). Lets a walking step blit one cached map region + a few shadow
        # fills instead of re-drawing every visible tile.
        self._visible_bbox: tuple[int, int, int, int] | None = None
        self._shadow_cells: tuple[tuple[int, int], ...] = ()
        self._fov_cache: dict[
            tuple[int | None, int, int],
            tuple[set[tuple[int, int]], tuple[int, int, int, int] | None, tuple[tuple[int, int], ...]],
        ] = {}
        # Only the fixed-size render section (40x20) the player stands in is
        # drawn; the camera snaps to it and the view/FOV are clipped to its
        # bounds. The map is a grid of these sections (a 120x60 world is 3x3 of
        # them; the 360x180 ocean world is 9x9). Sectioning only engages on a map
        # big enough for it to matter; tiny test maps keep the centred camera.
        self._section_w = _RENDER_SECTION_W
        self._section_h = _RENDER_SECTION_H
        self._sections_enabled = (
            game_map.width >= 3 * _MIN_SECTION_W and game_map.height >= 3 * _MIN_SECTION_H
        )
        self._section_bounds: tuple[int, int, int, int] | None = None
        self._current_section: tuple[int, int] | None = None
        # Wall clock driving the swimming glyph bob: it flips every second of
        # real time, independent of turns (so an idle swimmer still shimmers).
        # Injectable so tests can pin the time.
        self._clock: Callable[[], float] = time.monotonic

    @staticmethod
    def _clamp_sign(value: int) -> int:
        if value < 0:
            return -1
        if value > 0:
            return 1
        return 0

    def _direction_arrow(self, source: Position, target: Position) -> str:
        dx = self._clamp_sign(target.x - source.x)
        dy = self._clamp_sign(target.y - source.y)
        return _DIR_TO_ARROW[(dx, dy)]

    def _neighbor_mask(self, x: int, y: int, tile_char: str) -> int:
        connected: dict[int, bool] = {}
        for direction, (dx, dy) in _AUTOTILE_DIRECTION_OFFSETS.items():
            nx = x + dx
            ny = y + dy
            connected[direction] = self.game_map.in_bounds(nx, ny) and self.game_map.tile_at(nx, ny) == tile_char

        mask_value = 0
        if connected.get(2, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[2]  # N
        if connected.get(4, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[4]  # E
        if connected.get(6, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[6]  # S
        if connected.get(8, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[8]  # W

        if connected.get(1, False) and connected.get(2, False) and connected.get(8, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[1]  # NW
        if connected.get(3, False) and connected.get(2, False) and connected.get(4, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[3]  # NE
        if connected.get(7, False) and connected.get(6, False) and connected.get(8, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[7]  # SW
        if connected.get(5, False) and connected.get(6, False) and connected.get(4, False):
            mask_value |= _AUTOTILE_DIRECTION_MASKS[5]  # SE

        return mask_value

    def _append_message(self, text: str) -> None:
        self._message_log.append(text)
        if len(self._message_log) > self._max_log_lines:
            self._message_log = self._message_log[-self._max_log_lines :]

    def _describe_movement(
        self,
        action: str | None,
        player_pos: Position | None,
        suppress_wall_message: bool,
    ) -> None:
        if action not in _ACTION_DELTAS or player_pos is None:
            return

        if self._last_player_pos is None:
            self._last_player_pos = (player_pos.x, player_pos.y)
            return

        moved = self._last_player_pos != (player_pos.x, player_pos.y)
        if not moved and not suppress_wall_message:
            dx, dy = _ACTION_DELTAS[action]
            target_x = player_pos.x + dx
            target_y = player_pos.y + dy
            if not self.game_map.is_walkable(target_x, target_y):
                self._append_message("You bump into a wall.")

        self._last_player_pos = (player_pos.x, player_pos.y)

    def _collect_nearby_objects(
        self,
        player_pos: Position,
    ) -> list[_NearbyEntry]:
        nearby: list[_NearbyEntry] = []

        for ent, (pos, rend) in esper.get_components(Position, Renderable):
            if pos.x == player_pos.x and pos.y == player_pos.y:
                continue
            if (pos.x, pos.y) not in self._visible_tiles:
                continue

            dx = pos.x - player_pos.x
            dy = pos.y - player_pos.y
            if max(abs(dx), abs(dy)) != 1:
                continue

            arrow = self._direction_arrow(player_pos, pos)
            name = "Unknown"
            if esper.has_component(ent, Name):
                name = esper.component_for_entity(ent, Name).value

            classification = "default"
            if esper.has_component(ent, Friendly):
                classification = "friendly"
            elif esper.has_component(ent, Enemy):
                classification = "enemy"

            nearby.append(
                (
                    ent,
                    rend.glyph,
                    arrow,
                    name,
                    abs(dx) + abs(dy),
                    max(abs(dx), abs(dy)),
                    classification,
                    rend.fg,
                    rend.bg,
                )
            )

        nearby.sort(key=lambda item: (item[4], item[5], item[3]))
        return nearby

    def _compute_visible_tiles(self, player_ent: int | None, player_pos: Position | None) -> set[tuple[int, int]]:
        if player_pos is None:
            return set()

        radius = max(self.game_map.width // 2, self.game_map.height // 2)
        if player_ent is not None and esper.has_component(player_ent, Vision):
            radius = esper.component_for_entity(player_ent, Vision).radius

        # Clip the scan (and thus visibility) to the current section: other
        # sections are simulated but never drawn.
        if self._section_bounds is not None:
            sx, sy, sw, sh = self._section_bounds
            x_lo, x_hi = sx, sx + sw
            y_lo, y_hi = sy, sy + sh
        else:
            x_lo, x_hi = 0, self.game_map.width
            y_lo, y_hi = 0, self.game_map.height

        visible: set[tuple[int, int]] = set()
        origin = (player_pos.x, player_pos.y)
        for y in range(y_lo, y_hi):
            for x in range(x_lo, x_hi):
                if max(abs(x - player_pos.x), abs(y - player_pos.y)) > radius:
                    continue
                if self.game_map.has_line_of_sight(origin, (x, y)):
                    visible.add((x, y))
        return visible

    @staticmethod
    def _fov_bbox_and_shadows(
        visible: set[tuple[int, int]],
    ) -> tuple[tuple[int, int, int, int] | None, tuple[tuple[int, int], ...]]:
        """Bounding box of the visible set plus the cells inside that box that
        are *not* visible (wall shadows). Used to composite the map as one blit
        of the box followed by a few shadow blackouts."""
        if not visible:
            return None, ()
        xs = [c[0] for c in visible]
        ys = [c[1] for c in visible]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        shadows = tuple(
            (x, y)
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if (x, y) not in visible
        )
        return bbox, shadows

    def _compute_memory_geometry(self) -> None:
        """Recompute the remembered-tile geometry for the current view. Cheap set
        work over the viewport (at most one render section), refreshed only when
        the player moves -- the same cadence as the FOV memo -- so a walking step
        stays a handful of region blits.

        Uses ``_explored_tiles`` as of the *previous* frame (currently-visible
        tiles are folded into it after drawing), which is exactly right: a tile
        only needs remembering once it is no longer lit.
        """
        ox, oy = self._view_origin_x, self._view_origin_y
        vw, vh = self._view_width, self._view_height
        explored = self._explored_tiles
        visible = self._visible_tiles

        explored_in_view: list[tuple[int, int]] = []
        memory_cells: set[tuple[int, int]] = set()
        for vy in range(vh):
            wy = oy + vy
            for vx in range(vw):
                wx = ox + vx
                cell = (wx, wy)
                if cell in explored:
                    explored_in_view.append(cell)
                    if cell not in visible:
                        memory_cells.add(cell)

        if not explored_in_view:
            self._memory_bbox = None
            self._memory_holes = ()
            self._memory_cells = frozenset()
            self._memory_scenery = ()
            self._memory_terrain_overrides = ()
            return

        xs = [c[0] for c in explored_in_view]
        ys = [c[1] for c in explored_in_view]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        self._memory_bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        explored_set = set(explored_in_view)
        self._memory_holes = tuple(
            (x, y)
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if (x, y) not in explored_set
        )
        self._memory_cells = frozenset(memory_cells)
        # Precompute the (few) remembered-scenery overlays so the render path
        # doesn't scan the whole explored region for the handful with scenery.
        self._memory_scenery = tuple(
            (cell, self._tile_memory[cell])
            for cell in memory_cells
            if cell in self._tile_memory
        )
        # Terrain overrides: unseen cells whose live terrain has diverged from
        # what was last seen (e.g. a villager raised a wall out of sight). These
        # get overdrawn with the remembered char so memory shows the old terrain.
        overrides: list[tuple[tuple[int, int], str]] = []
        for cell in memory_cells:
            seen = self._seen_terrain.get(cell)
            if seen is not None and seen != self.game_map.tile_at(cell[0], cell[1]):
                overrides.append((cell, seen))
        self._memory_terrain_overrides = tuple(overrides)

    def _update_tile_memory(self, seen_scenery: dict[tuple[int, int], tuple]) -> None:
        """Fold this frame's field of view into the persistent tile memory: mark
        every visible tile explored, record its current terrain char (so memory
        tracks what was last seen, not later out-of-sight edits), then record the
        static scenery seen there (from ``seen_scenery``, gathered during the
        entity draw pass) or clear stale memory if it is gone (e.g. a felled tree).
        """
        self._explored_tiles |= self._visible_tiles
        for cell in self._visible_tiles:
            self._seen_terrain[cell] = self.game_map.tile_at(cell[0], cell[1])
            appearance = seen_scenery.get(cell)
            if appearance is not None:
                self._tile_memory[cell] = appearance
            else:
                self._tile_memory.pop(cell, None)

    def _draw_map_tile(
        self,
        r,
        vx: int,
        vy: int,
        wx: int,
        wy: int,
        draw_autotile_variant,
        dim: bool = False,
        tile_char: str | None = None,
    ) -> None:
        """Draw the base terrain tile at world cell (wx, wy) to view cell (vx, vy).
        With ``dim`` the tile is toned into its remembered (desaturated) colour --
        used by renderers that can't composite a whole desaturated map surface
        (the test double); the pygame path desaturates the cached surface instead.
        ``tile_char`` overrides the terrain char (last-seen terrain for remembered
        cells the live map has since changed); autotile masks still read the live
        neighbours, which is fine for the rare divergent cell.
        """
        tone = _memory_color if dim else (lambda colour: colour)
        tile = tile_char if tile_char is not None else self.game_map.tile_at(wx, wy)
        if tile == self.game_map.WALL and callable(draw_autotile_variant):
            wall_mask = self._wall_mask_cache.get((wx, wy))
            if wall_mask is None:
                wall_mask = self._neighbor_mask(wx, wy, self.game_map.WALL)
                self._wall_mask_cache[(wx, wy)] = wall_mask
            if bool(draw_autotile_variant(vx, vy, "wall", wall_mask, fg=tone(_WALL_BROWN), bg=None)):
                return
        if tile == self.game_map.WATER and callable(draw_autotile_variant):
            water_mask = self._water_mask_cache.get((wx, wy))
            if water_mask is None:
                water_mask = self._neighbor_mask(wx, wy, self.game_map.WATER)
                self._water_mask_cache[(wx, wy)] = water_mask
            if bool(draw_autotile_variant(vx, vy, "water", water_mask, fg=tone(_WATER_BLUE), bg=None)):
                return
        classification = "wall" if tile == self.game_map.WALL else "default"
        if tile == self.game_map.WALL:
            tile_fg: tuple[int, int, int] | None = _WALL_BROWN
        elif tile == self.game_map.WATER:
            tile_fg = _WATER_BLUE
        elif tile == self.game_map.DOOR:
            tile_fg = _DOOR_BROWN
        elif tile == self.game_map.WINDOW:
            tile_fg = _WINDOW_CYAN
        else:
            tile_fg = None
        # ``_memory_color`` maps a colourless floor (None) to a neutral dim grey,
        # so remembered floor stays faintly visible rather than turning black.
        draw_fg = _memory_color(tile_fg) if dim else tile_fg
        r.draw_glyph_classified(vx, vy, tile, classification, fg=draw_fg, bg=None)

    def _draw_memory_glyph(
        self,
        r,
        vx: int,
        vy: int,
        glyph: str,
        classification: str,
        fg: tuple[int, int, int] | None,
        bg: tuple[int, int, int] | None,
    ) -> None:
        """Draw a remembered scenery glyph desaturated. Prefers the renderer's
        sprite-aware ``draw_glyph_memory`` (desaturates the whole tile); falls
        back to a memory-toned foreground for the plain test double."""
        draw_memory = getattr(r, "draw_glyph_memory", None)
        if callable(draw_memory):
            draw_memory(vx, vy, glyph, classification, fg=fg, bg=bg)
            return
        r.draw_glyph_classified(vx, vy, glyph, classification, fg=_memory_color(fg), bg=bg)

    def _draw_memory_tile(self, r, vx: int, vy: int, wx: int, wy: int, draw_autotile_variant) -> None:
        """Draw one remembered tile (used by the per-tile fallback path): the
        desaturated last-seen terrain, then any static scenery last seen there."""
        self._draw_map_tile(
            r, vx, vy, wx, wy, draw_autotile_variant, dim=True,
            tile_char=self._seen_terrain.get((wx, wy)),
        )
        appearance = self._tile_memory.get((wx, wy))
        if appearance is not None:
            glyph, classification, fg, bg = appearance
            self._draw_memory_glyph(r, vx, vy, glyph, classification, fg, bg)

    def _render_map_layer(self, r, draw_autotile_variant) -> None:
        """Draw the map for the current FOV. When the renderer supports a cached
        map surface, a step becomes one region blit + a few shadow fills instead
        of re-drawing every visible tile (the dominant walking cost on web)."""
        has_map_surface = getattr(r, "has_map_surface", None)
        build_map_surface = getattr(r, "build_map_surface", None)
        blit_map_region = getattr(r, "blit_map_region", None)
        fill_cell_bg = getattr(r, "fill_cell_bg", None)
        can_composite = all(
            callable(fn) for fn in (has_map_surface, build_map_surface, blit_map_region, fill_cell_bg)
        )

        ox, oy = self._view_origin_x, self._view_origin_y

        if not can_composite:
            # Fallback (e.g. test double renderer): per-tile draw. Lit tiles show
            # in full colour; explored-but-unseen tiles are drawn from memory
            # (desaturated terrain + last-seen scenery).
            for vy in range(self._view_height):
                wy = oy + vy
                for vx in range(self._view_width):
                    wx = ox + vx
                    if (wx, wy) in self._visible_tiles:
                        self._draw_map_tile(r, vx, vy, wx, wy, draw_autotile_variant)
                    elif (wx, wy) in self._explored_tiles:
                        self._draw_memory_tile(r, vx, vy, wx, wy, draw_autotile_variant)
            return

        if not has_map_surface():
            # One-time: render every tile fully lit to a world-sized off-screen
            # surface. World coords stay valid when the camera scrolls at zoom.
            def _draw_full_map() -> None:
                for wy in range(self.game_map.height):
                    for wx in range(self.game_map.width):
                        self._draw_map_tile(r, wx, wy, wx, wy, draw_autotile_variant)

            build_map_surface(self.game_map.width, self.game_map.height, _draw_full_map)

        if self._visible_bbox is None and self._memory_bbox is None:
            return

        # Remembered terrain reuses the lit map surface and is faded to its memory
        # tone in place, per on-screen region (no per-mutation full-surface
        # greyscale). Feature-detected so renderers without it just fall back to
        # black beyond the FOV.
        apply_memory_fade = getattr(r, "apply_memory_fade", None)
        can_remember = callable(apply_memory_fade)

        # Screen was cleared at the top of process(); composite the map in layers
        # (offset by the scroll origin). Clip to the viewport so a big region blit
        # can't spill into the sidebar/status area when a box is larger than the
        # visible map.
        set_map_clip = getattr(r, "set_map_clip", None)
        clear_clip = getattr(r, "clear_clip", None)
        clipped = callable(set_map_clip) and callable(clear_clip)
        if clipped:
            set_map_clip(self._view_width, self._view_height)
        try:
            # 1. Remembered terrain: blit the lit map over the explored region,
            #    fade it to the memory tone, then black out never-seen holes.
            if can_remember and self._memory_bbox is not None:
                mbx, mby, mbw, mbh = self._memory_bbox
                blit_map_region(mbx, mby, mbw, mbh, ox, oy)
                apply_memory_fade(mbx - ox, mby - oy, mbw, mbh)
                for wx, wy in self._memory_holes:
                    fill_cell_bg(wx - ox, wy - oy)
            # 2. The lit field of view on top of the remembered terrain.
            if self._visible_bbox is not None:
                bx, by, bw, bh = self._visible_bbox
                blit_map_region(bx, by, bw, bh, ox, oy)
                # Shadow cells sit inside the FOV box but are blocked from view;
                # the rectangular blit just painted them lit, so fade the explored
                # ones back to their remembered look and black out the rest.
                for wx, wy in self._shadow_cells:
                    if can_remember and (wx, wy) in self._explored_tiles:
                        apply_memory_fade(wx - ox, wy - oy, 1, 1)
                    else:
                        fill_cell_bg(wx - ox, wy - oy)
        finally:
            if clipped:
                clear_clip()

        # 3. Terrain overrides: the region blit above showed *live* terrain, so
        #    unseen cells the map has since changed (a wall raised out of sight)
        #    are redrawn from the last-seen char and faded, hiding the change
        #    until the tile is actually seen again.
        if can_remember:
            for (wx, wy), seen_char in self._memory_terrain_overrides:
                if (wx, wy) in self._visible_tiles:
                    continue
                view_xy = self._world_to_view(wx, wy)
                if view_xy is None:
                    continue
                self._draw_map_tile(
                    r, view_xy[0], view_xy[1], wx, wy, draw_autotile_variant, tile_char=seen_char
                )
                apply_memory_fade(view_xy[0], view_xy[1], 1, 1)

        # 4. Remembered scenery (trees, furniture, ...) overlaid on the
        #    desaturated terrain. Every cell is inside the viewport by
        #    construction, so this is safe outside the map clip.
        for (wx, wy), appearance in self._memory_scenery:
            view_xy = self._world_to_view(wx, wy)
            if view_xy is None:
                continue
            glyph, classification, fg, bg = appearance
            self._draw_memory_glyph(r, view_xy[0], view_xy[1], glyph, classification, fg, bg)

    def _update_sighting_events(self, nearby: list[tuple[int, str, str, str, int, int, int, int]]) -> None:
        newly_seen = [item for item in nearby if item[0] not in self._seen_entity_ids]

        for ent_id, _glyph, arrow, name, _md, _cd, _x, _y in newly_seen[:2]:
            direction = _ARROW_TO_WORD.get(arrow, "nearby")
            self._append_message(f"You notice {name} to the {direction}.")
            self._seen_entity_ids.add(ent_id)

    def _collect_visible_npcs(
        self,
        player_pos: Position,
    ) -> list[tuple[int, str, str, str, int, int, int, int]]:
        visible: list[tuple[int, str, str, str, int, int, int, int]] = []

        for ent, (pos, rend, _npc) in esper.get_components(Position, Renderable, NPC):
            if (pos.x, pos.y) == (player_pos.x, player_pos.y):
                continue
            if (pos.x, pos.y) not in self._visible_tiles:
                continue

            dx = pos.x - player_pos.x
            dy = pos.y - player_pos.y
            arrow = self._direction_arrow(player_pos, pos)
            name = "Unknown"
            if esper.has_component(ent, Name):
                name = esper.component_for_entity(ent, Name).value

            visible.append((ent, rend.glyph, arrow, name, abs(dx) + abs(dy), max(abs(dx), abs(dy)), pos.x, pos.y))

        visible.sort(key=lambda item: (item[4], item[5], item[3]))
        return visible

    def _draw_sidebar(self, player_pos: Position | None) -> None:
        r = self.renderer
        draw_text_clipped = getattr(r, "draw_text_clipped", None)
        draw_text_tinted = getattr(r, "draw_text_tinted", None)
        fill_cells = getattr(r, "fill_cells", None)
        draw_ui_glyph = getattr(r, "draw_ui_glyph", None)

        def draw_line(
            x: int,
            y: int,
            text: str,
            width_cells: int,
            color: tuple[int, int, int] | None = None,
        ) -> None:
            clipped_text = text[: max(0, width_cells)]
            if color is not None and callable(draw_text_tinted):
                draw_text_tinted(x, y, clipped_text, color)
                return
            if callable(draw_text_clipped):
                draw_text_clipped(x, y, text, width_cells)
                return
            r.draw_text(x, y, clipped_text)

        panel_x = max(0, self.sidebar_x)
        panel_y = 0
        panel_w = max(14, self.sidebar_width)
        panel_h = max(14, self._grid_h)

        x0 = panel_x + 2
        content_width = max(6, panel_w - 4)
        wrap_width = content_width
        get_text_cols = getattr(r, "text_columns_for_cells", None)
        if callable(get_text_cols):
            cols = get_text_cols(content_width)
            if isinstance(cols, int):
                wrap_width = max(1, cols)

        nearby_data: list[_NearbyEntry] = []
        if player_pos is not None:
            visible_npcs = self._collect_visible_npcs(player_pos)
            self._update_sighting_events(visible_npcs)
            nearby_data = self._collect_nearby_objects(player_pos)

        nearby_lines: list[str] = []

        if player_pos is None:
            nearby_lines.append("none")
        else:
            if not nearby_data:
                nearby_lines.append("none")
            else:
                for _ent_id, glyph, arrow, name, _mdist, _cdist, _classification, _fg, _bg in nearby_data:
                    line = f"{glyph} {arrow} {name}"
                    nearby_lines.append(line)

        wrapped_nearby: list[str] = []
        for line in nearby_lines:
            wrapped = textwrap.wrap(line, width=wrap_width, break_long_words=True, break_on_hyphens=False)
            if wrapped:
                wrapped_nearby.extend(wrapped)
            else:
                wrapped_nearby.append("")

        panel_gap = 1
        min_box_h = 6
        nearby_content_needed = max(1, len(wrapped_nearby))
        nearby_h_needed = nearby_content_needed + 4
        max_nearby_h = max(min_box_h, panel_h - min_box_h - panel_gap)
        nearby_h = max(min_box_h, min(max_nearby_h, nearby_h_needed))
        log_h = panel_h - nearby_h - panel_gap
        if log_h < min_box_h:
            log_h = min_box_h
            nearby_h = max(min_box_h, panel_h - log_h - panel_gap)

        nearby_y = panel_y
        log_y = nearby_y + nearby_h + panel_gap

        draw_panel = getattr(r, "draw_panel", None)
        has_custom_panels = callable(draw_panel)
        if has_custom_panels:
            draw_panel(panel_x, nearby_y, panel_w, nearby_h, title="NEARBY")
            draw_panel(panel_x, log_y, panel_w, log_h, title="LOG")
        else:
            def draw_ascii_panel(px: int, py: int, pw: int, ph: int, title: str) -> None:
                top = "+" + ("-" * max(0, pw - 2)) + "+"
                r.draw_text(px, py, top)
                for row in range(1, max(1, ph - 1)):
                    r.draw_text(px, py + row, "|" + (" " * max(0, pw - 2)) + "|")
                if ph > 1:
                    r.draw_text(px, py + ph - 1, top)
                r.draw_text(px + 2, py, title)

            draw_ascii_panel(panel_x, nearby_y, panel_w, nearby_h, "NEARBY")
            draw_ascii_panel(panel_x, log_y, panel_w, log_h, "LOG")

        if callable(fill_cells) and not has_custom_panels:
            header_bg = (18, 18, 18)
            fill_cells(panel_x + 1, nearby_y + 1, max(1, panel_w - 2), 1, header_bg)
            fill_cells(panel_x + 1, log_y + 1, max(1, panel_w - 2), 1, header_bg)

        header_color = (236, 236, 236)
        text_color = (214, 214, 214)
        muted_color = (168, 168, 168)

        if not has_custom_panels:
            draw_line(x0, nearby_y + 1, "[NEARBY]", content_width, color=header_color)
            nearby_content_y = nearby_y + 3
            nearby_content_h = max(1, nearby_h - 4)
        else:
            nearby_content_y = nearby_y + 2
            nearby_content_h = max(1, nearby_h - 3)

        if callable(draw_ui_glyph) and nearby_data:
            icon_cells = 2
            text_x = x0 + icon_cells + 1
            text_width = max(1, content_width - icon_cells - 1)
            for idx, entry in enumerate(nearby_data[:nearby_content_h]):
                _ent_id, glyph, arrow, name, _mdist, _cdist, classification, fg, bg = entry
                row_y = nearby_content_y + idx
                icon_drawn = bool(draw_ui_glyph(
                    x0,
                    row_y,
                    glyph,
                    classification=classification,
                    fg=fg,
                    bg=bg,
                    cell_span=icon_cells,
                ))
                if icon_drawn:
                    line_text = f"{arrow} {name}"
                else:
                    line_text = f"{glyph} {arrow} {name}"
                draw_line(text_x, row_y, line_text, text_width, color=text_color)
        else:
            for idx, line in enumerate(wrapped_nearby[:nearby_content_h]):
                draw_line(x0, nearby_content_y + idx, line, content_width, color=text_color)

        if not has_custom_panels:
            draw_line(x0, log_y + 1, "[LOG]", content_width, color=header_color)
            log_content_y = log_y + 3
            log_content_h = max(1, log_h - 4)
        else:
            log_content_y = log_y + 2
            log_content_h = max(1, log_h - 3)

        log_messages: list[str] = []
        for message in self._message_log:
            wrapped = textwrap.wrap(message, width=wrap_width, break_long_words=True, break_on_hyphens=False)
            if wrapped:
                log_messages.extend(wrapped)
            else:
                log_messages.append("")

        if not log_messages:
            log_messages = ["none"]

        visible_log = log_messages[-log_content_h:]
        for idx, line in enumerate(visible_log):
            color = muted_color if line == "none" else text_color
            draw_line(x0, log_content_y + idx, line, content_width, color=color)

    def _compute_layout(self, player_pos: Position | None) -> None:
        get_screen_px = getattr(self.renderer, "get_screen_size_px", None)
        get_tile_cell_px = getattr(self.renderer, "get_tile_cell_size_px", None)
        get_ui_grid = getattr(self.renderer, "get_ui_grid_size", None)
        get_ui_cell_px = getattr(self.renderer, "get_ui_cell_size_px", None)
        get_sidebar_px = getattr(self.renderer, "get_sidebar_width_px", None)

        if (
            callable(get_screen_px)
            and callable(get_tile_cell_px)
            and callable(get_ui_grid)
            and callable(get_ui_cell_px)
            and callable(get_sidebar_px)
        ):
            screen_w, screen_h = get_screen_px()
            tile_w, tile_h = get_tile_cell_px()
            ui_cols, ui_rows = get_ui_grid()
            ui_cell_w, ui_cell_h = get_ui_cell_px()

            tile_w = max(1, tile_w)
            tile_h = max(1, tile_h)
            ui_cell_w = max(1, ui_cell_w)
            ui_cell_h = max(1, ui_cell_h)

            status_px = ui_cell_h
            sidebar_px = max(14 * ui_cell_w, int(get_sidebar_px(22 * ui_cell_w)))
            sidebar_px = min(max(sidebar_px, 14 * ui_cell_w), max(14 * ui_cell_w, screen_w - 8 * tile_w))

            map_px_w = max(8 * tile_w, screen_w - sidebar_px)
            map_px_h = max(5 * tile_h, screen_h - status_px)

            self._view_width = min(self.game_map.width, max(8, map_px_w // tile_w))
            self._view_height = min(self.game_map.height, max(5, map_px_h // tile_h))

            self._grid_w = max(1, ui_cols)
            self._grid_h = max(1, ui_rows)
            self.sidebar_width = max(14, sidebar_px // ui_cell_w)
            self.sidebar_x = max(0, self._grid_w - self.sidebar_width)
            self._status_y = max(0, self._grid_h - 1)

            max_ox = max(0, self.game_map.width - self._view_width)
            max_oy = max(0, self.game_map.height - self._view_height)
            if player_pos is None:
                self._view_origin_x = 0
                self._view_origin_y = 0
                return

            self._view_origin_x = min(max_ox, max(0, player_pos.x - self._view_width // 2))
            self._view_origin_y = min(max_oy, max(0, player_pos.y - self._view_height // 2))
            return

        grid_w = self.game_map.width + self.sidebar_width + 3
        grid_h = self.game_map.height + 1
        get_grid = getattr(self.renderer, "get_grid_size", None)
        if callable(get_grid):
            w, h = get_grid()
            if isinstance(w, int) and isinstance(h, int):
                grid_w = max(20, w)
                grid_h = max(8, h)

        get_sidebar_w = getattr(self.renderer, "get_sidebar_width_cells", None)
        if callable(get_sidebar_w):
            w = get_sidebar_w(self.sidebar_width)
            if isinstance(w, int):
                self.sidebar_width = max(12, min(w, max(12, grid_w - 8)))

        self._grid_w = grid_w
        self._grid_h = grid_h

        self.sidebar_x = max(0, grid_w - self.sidebar_width)
        self._view_width = min(self.game_map.width, max(8, self.sidebar_x))
        self._view_height = min(self.game_map.height, max(5, grid_h - 1))
        self._status_y = max(0, grid_h - 1)

        max_ox = max(0, self.game_map.width - self._view_width)
        max_oy = max(0, self.game_map.height - self._view_height)
        if player_pos is None:
            self._view_origin_x = 0
            self._view_origin_y = 0
            return

        self._view_origin_x = min(max_ox, max(0, player_pos.x - self._view_width // 2))
        self._view_origin_y = min(max_oy, max(0, player_pos.y - self._view_height // 2))

    def _world_to_view(self, wx: int, wy: int) -> tuple[int, int] | None:
        vx = wx - self._view_origin_x
        vy = wy - self._view_origin_y
        if vx < 0 or vy < 0 or vx >= self._view_width or vy >= self._view_height:
            return None
        return (vx, vy)

    def view_bounds(self) -> tuple[int, int, int, int]:
        """(origin_x, origin_y, width, height) of the currently drawn viewport in
        world cells. Used by the look cursor to stay on-screen."""
        return (self._view_origin_x, self._view_origin_y, self._view_width, self._view_height)

    def tile_is_visible(self, x: int, y: int) -> bool:
        """Whether world cell (x, y) is in the player's current field of view."""
        return (x, y) in self._visible_tiles

    def _apply_section_camera(self, player_pos: Position | None) -> None:
        """Lock the camera to the render section the player occupies. Sets
        ``_section_bounds`` (used to clip FOV) and clamps the view origin inside
        that section, so only the current section is ever drawn. Emits a log line
        when the player crosses into a new section."""
        if player_pos is None or not self._sections_enabled:
            self._section_bounds = None
            return

        sw, sh = self._section_w, self._section_h
        cols = max(1, self.game_map.width // sw)
        rows = max(1, self.game_map.height // sh)
        col = min(cols - 1, max(0, player_pos.x // sw))
        row = min(rows - 1, max(0, player_pos.y // sh))
        sec_ox, sec_oy = col * sw, row * sh
        # The last column/row absorbs any remainder when the map isn't evenly
        # divisible by the section size, so the whole map stays covered.
        sec_w = self.game_map.width - sec_ox if col == cols - 1 else sw
        sec_h = self.game_map.height - sec_oy if row == rows - 1 else sh
        self._section_bounds = (sec_ox, sec_oy, sec_w, sec_h)

        if self._current_section is not None and self._current_section != (col, row):
            self._append_message("You cross into a new area.")
        self._current_section = (col, row)

        # Never render wider/taller than the section; clamp the origin inside it.
        self._view_width = min(self._view_width, sec_w)
        self._view_height = min(self._view_height, sec_h)
        max_ox = sec_ox + sec_w - self._view_width
        max_oy = sec_oy + sec_h - self._view_height
        self._view_origin_x = min(max_ox, max(sec_ox, player_pos.x - self._view_width // 2))
        self._view_origin_y = min(max_oy, max(sec_oy, player_pos.y - self._view_height // 2))

    def _status_appearance(
        self,
        ent: int,
        pos: Position,
        glyph: str,
        fg: tuple[int, int, int] | None,
    ) -> tuple[str, tuple[int, int, int] | None, bool]:
        """The (glyph, colour, is_status_identifier) to draw for a character this
        instant. With no active status it's just its own tile; with statuses it
        cycles through the base tile then each status identifier in turn, on a
        wall clock, so the animation plays even while idle. The bool marks the
        status-identifier frames, which the renderer must draw as literal glyphs
        (not the character's classification sprite)."""
        statuses = active_statuses(self.game_map, ent, pos)
        if not statuses:
            return glyph, fg, False

        # (glyph, fg, seconds, is_status_identifier)
        frames: list[tuple[str, tuple[int, int, int] | None, float, bool]] = [
            (glyph, fg, _STATUS_BASE_SECONDS, False)
        ]
        for name in statuses:
            id_glyph, id_fg, seconds = _STATUS_DISPLAY[name]
            frames.append((id_glyph, id_fg if id_fg is not None else fg, seconds, True))

        total = sum(frame[2] for frame in frames)
        if total <= 0:
            return glyph, fg, False

        elapsed = self._clock() % total
        cursor = 0.0
        for frame_glyph, frame_fg, seconds, is_status in frames:
            cursor += seconds
            if elapsed < cursor:
                return frame_glyph, frame_fg, is_status
        last = frames[-1]
        return last[0], last[1], last[3]

    def _invalidate_map_caches_if_changed(self) -> None:
        """Refresh map-derived caches when the map has been edited since last
        frame, so a newly built wall/door/window shows up immediately.

        Field-of-view can change anywhere, so its memo is always dropped. The
        cached map *surface*, though, is repainted incrementally -- only the
        edited cells (and their neighbours, whose autotile masks depend on them)
        -- because re-rendering the whole world surface on every single tile edit
        is hundreds of milliseconds on a large map (an NPC building a house edits
        many tiles in quick succession)."""
        revision = getattr(self.game_map, "revision", 0)
        if revision == self._map_revision:
            return
        self._map_revision = revision
        self._fov_cache.clear()
        self._visible_cache_key = None

        consume = getattr(self.game_map, "consume_dirty_tiles", None)
        redraw = getattr(self.renderer, "redraw_map_cells", None)
        has_surface = getattr(self.renderer, "has_map_surface", None)
        dirty = consume() if callable(consume) else None

        if dirty and callable(redraw) and callable(has_surface) and has_surface():
            # An edited cell and its 8 neighbours are the only cells whose drawn
            # appearance can change (autotile masks read neighbours).
            affected: set[tuple[int, int]] = set()
            for x, y in dirty:
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if self.game_map.in_bounds(nx, ny):
                            affected.add((nx, ny))
            for cell in affected:
                self._wall_mask_cache.pop(cell, None)
                self._water_mask_cache.pop(cell, None)
            draw_autotile_variant = getattr(self.renderer, "draw_autotile_variant", None)
            redraw(
                affected,
                lambda wx, wy: self._draw_map_tile(
                    self.renderer, wx, wy, wx, wy, draw_autotile_variant
                ),
            )
            return

        # No incremental path (test double, or surface not built yet): fall back
        # to a full invalidate and let the next frame rebuild.
        self._wall_mask_cache.clear()
        self._water_mask_cache.clear()
        invalidate_surface = getattr(self.renderer, "invalidate_map_surface", None)
        if callable(invalidate_surface):
            invalidate_surface()

    def process(self, action: str | None = None) -> None:
        r = self.renderer
        r.clear()

        self._invalidate_map_caches_if_changed()

        # Just locate the player; a whole-world position/name index was built
        # here every frame but nothing read it -- a needless full-entity scan
        # that hurt once the ocean added a couple thousand fish/seaweed.
        player_pos: Position | None = None
        player_ent: int | None = None
        for ent, (pos, _player) in esper.get_components(Position, Player):
            player_pos = pos
            player_ent = ent
            break

        if player_pos is not None and self._last_player_pos is None:
            self._last_player_pos = (player_pos.x, player_pos.y)

        events = _pull_turn_events()
        for event in events:
            self._append_message(event)

        self._describe_movement(action, player_pos, suppress_wall_message=bool(events))
        self._compute_layout(player_pos)
        self._apply_section_camera(player_pos)

        # Only recompute field-of-view when the player actually moved; the LOS
        # raycast over every tile is one of the heaviest per-frame costs. Results
        # are memoized per position (the map is static, so FOV is deterministic).
        vis_key = None if player_pos is None else (player_ent, player_pos.x, player_pos.y)
        if vis_key != self._visible_cache_key:
            self._visible_cache_key = vis_key
            if vis_key is None:
                self._visible_tiles = set()
                self._visible_bbox = None
                self._shadow_cells = ()
            else:
                cached = self._fov_cache.get(vis_key)
                if cached is None:
                    visible = self._compute_visible_tiles(player_ent, player_pos)
                    cached = (visible, *self._fov_bbox_and_shadows(visible))
                    self._fov_cache[vis_key] = cached
                self._visible_tiles, self._visible_bbox, self._shadow_cells = cached
            # Field of view changed, so the remembered region around it has too.
            # (Unlike FOV this can't be memoized per position: the explored set
            # keeps growing, so a revisited tile may recall more than last time.)
            self._compute_memory_geometry()
        draw_autotile_variant = getattr(r, "draw_autotile_variant", None)

        self._render_map_layer(r, draw_autotile_variant)

        # draw tuple: (vx, vy, glyph, classification, fg, bg, force_glyph)
        DrawData = tuple[int, int, str, str, tuple[int, int, int] | None, tuple[int, int, int] | None, bool]
        player_draw: DrawData | None = None
        character_draws: list[DrawData] = []
        # Static scenery seen this frame, keyed by tile -- folded into tile memory
        # after drawing so it can be recalled once the tile leaves view. Gathered
        # here to piggyback on the entity scan rather than sweep every entity twice.
        seen_scenery: dict[tuple[int, int], tuple[str, str, tuple[int, int, int] | None, tuple[int, int, int] | None]] = {}
        for ent, (pos, rend) in esper.get_components(Position, Renderable):
            is_player = esper.has_component(ent, Player)
            is_character = is_player or esper.has_component(ent, NPC)
            if (pos.x, pos.y) not in self._visible_tiles and not is_player:
                continue

            if (pos.x, pos.y) in self._visible_tiles and is_memorable_scenery(ent):
                seen_scenery[(pos.x, pos.y)] = (rend.glyph, "default", rend.fg, rend.bg)

            view_xy = self._world_to_view(pos.x, pos.y)
            if view_xy is None and not is_player:
                continue

            classification = "default"
            if is_player or esper.has_component(ent, Friendly):
                classification = "friendly"
            elif esper.has_component(ent, Enemy):
                classification = "enemy"

            if is_character:
                glyph, fg, force_glyph = self._status_appearance(ent, pos, rend.glyph, rend.fg)
            else:
                glyph, fg, force_glyph = rend.glyph, rend.fg, False

            if is_player:
                if view_xy is not None:
                    player_draw = (view_xy[0], view_xy[1], glyph, classification, fg, rend.bg, force_glyph)
                continue

            if view_xy is not None:
                draw_data = (view_xy[0], view_xy[1], glyph, classification, fg, rend.bg, force_glyph)
                if is_character:
                    character_draws.append(draw_data)
                else:
                    r.draw_glyph_classified(
                        draw_data[0], draw_data[1], draw_data[2], draw_data[3],
                        fg=draw_data[4], bg=draw_data[5], force_glyph=draw_data[6],
                    )

        for draw_data in character_draws:
            r.draw_glyph_classified(
                draw_data[0], draw_data[1], draw_data[2], draw_data[3],
                fg=draw_data[4], bg=draw_data[5], force_glyph=draw_data[6],
            )

        if player_draw is not None:
            r.draw_glyph_classified(
                player_draw[0], player_draw[1], player_draw[2], player_draw[3],
                fg=player_draw[4], bg=player_draw[5], force_glyph=player_draw[6],
            )

        # Commit this frame's sightings to tile memory (marks tiles explored and
        # records/refreshes their scenery), ready to be recalled next time they
        # fall out of view.
        self._update_tile_memory(seen_scenery)

        self._draw_sidebar(player_pos)

        # Dim the world when night draws in (a translucent wash over everything
        # already drawn). Skipped on renderers without the overlay hook (tests).
        clock = world_clock()
        overlay_alpha = night_overlay_alpha(clock)
        draw_overlay = getattr(r, "draw_overlay", None)
        if overlay_alpha > 0 and callable(draw_overlay):
            draw_overlay(_NIGHT_TINT, overlay_alpha)

        # Speech bubbles float above characters in world space (aligned to the
        # tile grid, not the UI text grid), drawn on top of the night wash so they
        # stay legible. Skipped on renderers without the hook (the test double).
        draw_world_label = getattr(r, "draw_world_label", None)
        measure_world_label = getattr(r, "measure_world_label", None)
        get_cell_px = getattr(r, "get_tile_cell_size_px", None)
        if callable(draw_world_label):
            now = self._clock()
            bubbles = active_bubbles(now)
            if callable(measure_world_label) and callable(get_cell_px):
                # Lay bubbles out newest-first: each one takes the lowest vertical
                # slot (nearest its character) that doesn't overlap an already-placed
                # bubble, lifting up a row at a time until it fits. So a newer bubble
                # sits below and shoves older/neighbouring ones up -- no two overlap.
                cell_w, cell_h = get_cell_px()
                cell_w, cell_h = max(1, cell_w), max(1, cell_h)
                placed: list[tuple[int, int, int, int]] = []
                for bubble in reversed(bubbles):
                    view = self._world_to_view(bubble.x, bubble.y)
                    if view is None:
                        continue
                    vx, vy = view
                    body_w, body_h, tail = measure_world_label(bubble.text, bubble.indicator)
                    step = body_h + max(2, cell_h // 8)
                    x0 = vx * cell_w + cell_w // 2 - body_w // 2
                    lift = 0
                    for _ in range(64):  # bounded so a dense pile can't spin forever
                        top = vy * cell_h - body_h - tail - lift
                        rect = (x0, top, x0 + body_w, top + body_h)
                        if not any(_rects_overlap(rect, other) for other in placed):
                            break
                        lift += step
                    placed.append(rect)
                    draw_world_label(
                        vx, vy, bubble.text,
                        indicator=bubble.indicator,
                        indicator_color=bubble.indicator_color,
                        lift=lift,
                        alpha=bubble_alpha(bubble, now),
                    )
            else:
                # No measurement hook (test double): draw without overlap layout.
                for bubble in bubbles:
                    view = self._world_to_view(bubble.x, bubble.y)
                    if view is not None:
                        draw_world_label(
                            view[0], view[1], bubble.text,
                            indicator=bubble.indicator,
                            indicator_color=bubble.indicator_color,
                            alpha=bubble_alpha(bubble, now),
                        )

        # The look cursor rides on top of the night wash so it stays bright while
        # examining. Only drawn when the cursor cell is inside the viewport.
        if self.look_cursor is not None:
            draw_cursor = getattr(r, "draw_cursor", None)
            cursor_view = self._world_to_view(*self.look_cursor)
            if cursor_view is not None and callable(draw_cursor):
                draw_cursor(cursor_view[0], cursor_view[1])

        draw_text_clipped = getattr(r, "draw_text_clipped", None)
        time_text = f"{format_datetime(clock)}  "
        needs_text = ""
        if player_ent is not None and esper.has_component(player_ent, Needs):
            needs = esper.component_for_entity(player_ent, Needs)
            needs_text = (
                f"Hunger {int(needs.hunger)}%  Thirst {int(needs.thirst)}%  "
                f"Tired {int(needs.tiredness)}%    "
            )
        # While looking, the examine line replaces the usual hint bar so the
        # player can read what's under the cursor and which actions it affords.
        if self.look_info:
            status_line = self.look_info
        else:
            status_line = f"{time_text}{needs_text}I inv  C status  L look  R sleep  Esc menu"
        if callable(draw_text_clipped):
            draw_text_clipped(0, self._status_y, status_line, self._grid_w)
        else:
            r.draw_text(0, self._status_y, status_line)
        r.present()
