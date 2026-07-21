"""ECS processors (systems).

esper 3.x uses module-level state and esper.Processor subclasses whose
process() receives whatever args are passed to esper.process().
"""
from collections import deque
from collections.abc import Callable
import random
import textwrap
import time

import esper

from components import Asleep, Bed, BerryBush, BlocksMovement, BuildPlan, Camp, Chest, Corpse, Deer, Diet, Enemy, Equipment, Friendly, Furniture, Home, Inventory, Meat, Name, Needs, NPC, OnFire, Owned, Player, Position, Renderable, Resident, Sapling, Stove, Tree, Vision, WorldClock
from game_map import GameMap
from items import RAW_MEAT, WOOD, cook_meat, hunger_restored, is_cooked_meat, is_raw_meat
from renderer.base import Renderer

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
_TREE_GREEN = (58, 138, 66)
_SAPLING_GREEN = (120, 176, 90)
_BUSH_GREEN = (86, 140, 78)   # a bush that has been picked bare
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

# A section must be at least this big for the 3x3 section camera to engage;
# below it (tiny test maps) the renderer keeps its plain centred camera.
_MIN_SECTION_W = 12
_MIN_SECTION_H = 8

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
        clock.turn += 1
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


# Need level (percent of max) at which an NPC stops what it's doing and forages.
_FORAGE_THRESHOLD = 55.0
# How much a single grazing/drinking/feeding action restores.
_GRAZE_RESTORE = 45.0
_DRINK_RESTORE = 100.0
_FEED_RESTORE = 60.0


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


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

    def __init__(self, game_map: GameMap):
        self.game_map = game_map
        self._shore_tiles: list[tuple[int, int]] = self._compute_shore_tiles()

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

    def _step_toward(
        self,
        ent: int,
        pos: Position,
        goal: tuple[int, int],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Move one step along a path to ``goal``. Returns True if it moved."""
        blocked = {xy for xy, occ_ent in occupied.items() if occ_ent != ent}
        path = self.game_map.find_path((pos.x, pos.y), goal, blocked_tiles=blocked)
        if not path:
            return False
        next_x, next_y = path[0]
        if (next_x, next_y) == goal and (next_x, next_y) in occupied:
            # The goal tile itself is occupied (e.g. a tree/prey we path *to*);
            # don't step onto it -- the caller handles the adjacent interaction.
            return False
        if (next_x, next_y) in occupied and occupied[(next_x, next_y)] != ent:
            return False
        old_xy = (pos.x, pos.y)
        pos.x, pos.y = next_x, next_y
        occupied.pop(old_xy, None)
        occupied[(next_x, next_y)] = ent
        return True

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

    def _build(
        self,
        ent: int,
        pos: Position,
        trees: list[tuple[tuple[int, int], int]],
        occupied: dict[tuple[int, int], int],
    ) -> bool:
        """Advance a resident's ``BuildPlan`` by one step: gather wood if short,
        place an adjacent pending piece, or walk toward the nearest one. Finishing
        the last piece furnishes and claims the new house."""
        plan = esper.component_for_entity(ent, BuildPlan)
        if not plan.remaining:
            self._complete_build(ent, plan)
            return True

        inventory = self._ensure_inventory(ent)
        if WOOD not in inventory.items:
            return self._gather_wood(ent, pos, inventory, trees, occupied)

        # Place a piece we're already standing next to.
        for index, (bx, by, tile) in enumerate(plan.remaining):
            if _chebyshev((pos.x, pos.y), (bx, by)) == 1:
                if self.game_map.set_tile(bx, by, tile):
                    inventory.items.remove(WOOD)
                plan.remaining.pop(index)
                if not plan.remaining:
                    self._complete_build(ent, plan)
                return True

        # Otherwise approach the nearest pending piece, treating all pending
        # tiles as blocked so the builder stops beside them instead of on them.
        target = min(plan.remaining, key=lambda t: _chebyshev((pos.x, pos.y), (t[0], t[1])))
        added: list[tuple[int, int]] = []
        for bx, by, _tile in plan.remaining:
            if (bx, by) not in occupied:
                occupied[(bx, by)] = -1  # sentinel: not a real entity, just blocked
                added.append((bx, by))
        moved = self._step_toward(ent, pos, (target[0], target[1]), occupied)
        for xy in added:
            if occupied.get(xy) == -1:
                del occupied[xy]
        return moved

    def _complete_build(self, ent: int, plan: BuildPlan) -> None:
        interior = frozenset(plan.interior)
        bed = furnish_house(self.game_map, interior)
        if bed is not None:
            if esper.has_component(ent, Home):
                home = esper.component_for_entity(ent, Home)
                home.x, home.y = bed
            else:
                esper.add_component(ent, Home(bed[0], bed[1]))
            # The builder owns the house it just raised (bed and chest).
            set_house_ownership(interior, ent)
        if esper.has_component(ent, BuildPlan):
            esper.remove_component(ent, BuildPlan)
        _push_turn_event("A villager finishes building a new cabin.")

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
        if _chebyshev((pos.x, pos.y), player_xy) == 1:
            return False  # adjacent: hold (player-facing combat is player-driven)
        return self._step_toward(ent, pos, player_xy, occupied)

    def process(self, action: str | None = None) -> None:
        if action not in _TURN_ACTIONS:
            return

        player_xy = self._find_player_position()

        occupied = {
            (pos.x, pos.y): ent
            for ent, (pos, _blocks) in esper.get_components(Position, BlocksMovement)
        }

        # Snapshot dynamic foraging targets once per turn.
        trees = [((pos.x, pos.y), ent) for ent, (pos, _t) in esper.get_components(Position, Tree)]
        prey = [((pos.x, pos.y), ent) for ent, (pos, _d) in esper.get_components(Position, Deer)]
        corpses = [
            ((pos.x, pos.y), ent)
            for ent, (pos, _c, inv) in esper.get_components(Position, Corpse, Inventory)
            if any(is_raw_meat(item) or is_cooked_meat(item) for item in inv.items)
        ]
        stoves = [((pos.x, pos.y), ent) for ent, (pos, _s) in esper.get_components(Position, Stove)]

        # Materialise the list: a predator can kill (delete) another NPC mid-loop
        # via _seek_food, so we snapshot up front and skip anything already gone.
        for ent, (pos, _npc) in list(esper.get_components(Position, NPC)):
            if not esper.entity_exists(ent):
                continue
            # Sleepers skip their turn; NeedsProcessor recovers and wakes them.
            if esper.has_component(ent, Asleep):
                continue
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
                        acted = self._graze(ent, pos, needs, trees, occupied)
                    elif diet_kind == "carnivore":
                        # Predator: eats raw meat on the spot.
                        acted = self._seek_food(ent, pos, needs, prey, corpses, occupied)
                    elif diet_kind == "cook":
                        # Villager: cooks meat before eating (multi-turn plan).
                        acted = self._feed_cook(ent, pos, prey, corpses, trees, stoves, occupied)

            # Building a house is the lowest-priority drive: a resident works its
            # BuildPlan only once fed, watered, and rested.
            if not acted and esper.has_component(ent, BuildPlan):
                acted = self._build(ent, pos, trees, occupied)

            if acted:
                continue

            if player_xy is not None and esper.has_component(ent, Enemy):
                self._chase_player(ent, pos, player_xy, occupied)


class HousingProcessor(esper.Processor):
    """Settles residents into homes. Houses belong to people: a resident who owns
    one is left alone. A resident who owns none **claims the nearest unowned
    house** it can reach (marking it as theirs); only if none is free does it get
    a ``BuildPlan`` for a preset cabin, which the NPC AI then builds over many
    turns and moves into.

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

        for ent, (_res,) in list(esper.get_components(Resident)):
            if owned_bed_of(ent) is not None:
                continue  # already owns a house -- never re-claim or rebuild
            if esper.has_component(ent, BuildPlan):
                continue  # already building its own home
            if self._claim_unowned_house(ent, houses):
                continue
            self._start_build(ent)

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

    def _start_build(self, ent: int) -> None:
        if not esper.has_component(ent, Position):
            return
        pos = esper.component_for_entity(ent, Position)
        occupied = {
            (p.x, p.y) for _e, (p, _b) in esper.get_components(Position, BlocksMovement)
        }
        origin = choose_build_site(self.game_map, (pos.x, pos.y), occupied)
        if origin is None:
            return
        build, interior = _blueprint_tiles(self.game_map, origin[0], origin[1])
        bed = interior[0] if interior else origin
        esper.add_component(ent, BuildPlan(remaining=build, interior=interior, bed=bed))


# Daily, per-tile odds that shape the flora. A sapling can sprout on any open
# outdoor ground tile; a mature tree/bush can die (rot/fall) at half that rate.
# Berry bushes sprout less often than trees.
_DAILY_SPROUT_CHANCE = 0.0001       # 0.01% per outdoor ground tile per day (trees)
_DAILY_BUSH_SPROUT_CHANCE = 0.00004  # bushes are rarer than trees
_DAILY_DEATH_CHANCE = 0.00005       # 0.005% per plant per day (half the sprout rate)
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
        self._rng = rng if rng is not None else random.random
        self._cap = max(40, (game_map.width * game_map.height) // 20)
        self._last_day: int | None = None
        # Cached list of outdoor ground tiles (floor, not inside a house),
        # rebuilt only when the map changes.
        self._ground_cache: dict = {}

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
    """Advances hunger/thirst once per real turn (a move, or a ``wait``).

    Menu refreshes call ``esper.process(None)``; those must not starve the
    player, so the tick is gated on a turn-advancing action, matching how the
    NPC AI only reacts on real turns.
    """

    def process(self, action: str | None = None) -> None:
        if action not in _TURN_ACTIONS:
            return

        night = is_night(world_clock())
        woke: list[int] = []

        for ent, (needs,) in esper.get_components(Needs):
            prev_hunger = needs.hunger
            prev_thirst = needs.thirst
            prev_tiredness = needs.tiredness
            asleep = esper.has_component(ent, Asleep)

            # Hunger and thirst creep up whether awake or asleep.
            needs.hunger = min(needs.max_value, needs.hunger + needs.hunger_rate)
            needs.thirst = min(needs.max_value, needs.thirst + needs.thirst_rate)

            if asleep:
                # Sleeping pays down tiredness; waking is handled after the loop.
                needs.tiredness = max(0.0, needs.tiredness - _SLEEP_RECOVERY)
                if needs.tiredness <= 0.0:
                    woke.append(ent)
            else:
                rate = needs.tiredness_rate * (_NIGHT_TIREDNESS_MULTIPLIER if night else 1.0)
                needs.tiredness = min(needs.max_value, needs.tiredness + rate)

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
        self._status_y = game_map.height
        self._message_log: list[str] = ["You enter the area."]
        self._max_log_lines = 200
        self._last_player_pos: tuple[int, int] | None = None
        self._seen_entity_ids: set[int] = set()
        self._visible_tiles: set[tuple[int, int]] = set()
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
        # The world is a 3x3 grid of sections. Only the section the player stands
        # in is rendered (the whole map keeps simulating); the camera snaps to it
        # and the view/FOV are clipped to its bounds. Sectioning only engages on a
        # map big enough for the grid to be meaningful (the real 120x60 world);
        # smaller maps (e.g. tests) keep the plain centred camera.
        self._section_w = max(1, game_map.width // 3)
        self._section_h = max(1, game_map.height // 3)
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

    def _draw_map_tile(self, r, vx: int, vy: int, wx: int, wy: int, draw_autotile_variant) -> None:
        tile = self.game_map.tile_at(wx, wy)
        if tile == self.game_map.WALL and callable(draw_autotile_variant):
            wall_mask = self._wall_mask_cache.get((wx, wy))
            if wall_mask is None:
                wall_mask = self._neighbor_mask(wx, wy, self.game_map.WALL)
                self._wall_mask_cache[(wx, wy)] = wall_mask
            if bool(draw_autotile_variant(vx, vy, "wall", wall_mask, fg=_WALL_BROWN, bg=None)):
                return
        if tile == self.game_map.WATER and callable(draw_autotile_variant):
            water_mask = self._water_mask_cache.get((wx, wy))
            if water_mask is None:
                water_mask = self._neighbor_mask(wx, wy, self.game_map.WATER)
                self._water_mask_cache[(wx, wy)] = water_mask
            if bool(draw_autotile_variant(vx, vy, "water", water_mask, fg=_WATER_BLUE, bg=None)):
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
        r.draw_glyph_classified(vx, vy, tile, classification, fg=tile_fg, bg=None)

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
            # Fallback (e.g. test double renderer): original per-visible-tile draw.
            for vy in range(self._view_height):
                wy = oy + vy
                for vx in range(self._view_width):
                    wx = ox + vx
                    if (wx, wy) in self._visible_tiles:
                        self._draw_map_tile(r, vx, vy, wx, wy, draw_autotile_variant)
            return

        if not has_map_surface():
            # One-time: render every tile fully lit to a world-sized off-screen
            # surface. World coords stay valid when the camera scrolls at zoom.
            def _draw_full_map() -> None:
                for wy in range(self.game_map.height):
                    for wx in range(self.game_map.width):
                        self._draw_map_tile(r, wx, wy, wx, wy, draw_autotile_variant)

            build_map_surface(self.game_map.width, self.game_map.height, _draw_full_map)

        if self._visible_bbox is None:
            return
        # Screen was cleared at the top of process(); blit only the visible FOV
        # box (offset by the scroll origin) and black out wall-shadowed cells.
        # Clip to the viewport so the single big blit can't spill into the
        # sidebar/status area when the FOV box is larger than the visible map.
        set_map_clip = getattr(r, "set_map_clip", None)
        clear_clip = getattr(r, "clear_clip", None)
        clipped = callable(set_map_clip) and callable(clear_clip)
        if clipped:
            set_map_clip(self._view_width, self._view_height)
        try:
            bx, by, bw, bh = self._visible_bbox
            blit_map_region(bx, by, bw, bh, ox, oy)
            for wx, wy in self._shadow_cells:
                fill_cell_bg(wx - ox, wy - oy)
        finally:
            if clipped:
                clear_clip()

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

    def _draw_sidebar(
        self,
        player_pos: Position | None,
        _entity_lookup: dict[tuple[int, int], tuple[str, str]],
    ) -> None:
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

    def _apply_section_camera(self, player_pos: Position | None) -> None:
        """Lock the camera to the 3x3 section the player occupies. Sets
        ``_section_bounds`` (used to clip FOV) and clamps the view origin inside
        that section, so only the current section is ever drawn. Emits a log line
        when the player crosses into a new section."""
        if player_pos is None or not self._sections_enabled:
            self._section_bounds = None
            return

        sw, sh = self._section_w, self._section_h
        col = min(2, max(0, player_pos.x // sw))
        row = min(2, max(0, player_pos.y // sh))
        sec_ox, sec_oy = col * sw, row * sh
        # The bottom/right sections absorb any remainder when the map isn't
        # evenly divisible by 3, so the whole map is covered.
        sec_w = self.game_map.width - sec_ox if col == 2 else sw
        sec_h = self.game_map.height - sec_oy if row == 2 else sh
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
        """Drop map-derived caches when the map has been edited since last frame,
        so a newly built wall/door/window shows up immediately."""
        revision = getattr(self.game_map, "revision", 0)
        if revision == self._map_revision:
            return
        self._map_revision = revision
        self._wall_mask_cache.clear()
        self._water_mask_cache.clear()
        self._fov_cache.clear()
        self._visible_cache_key = None
        invalidate_surface = getattr(self.renderer, "invalidate_map_surface", None)
        if callable(invalidate_surface):
            invalidate_surface()

    def process(self, action: str | None = None) -> None:
        r = self.renderer
        r.clear()

        self._invalidate_map_caches_if_changed()

        entity_lookup: dict[tuple[int, int], tuple[str, str]] = {}
        player_pos: Position | None = None
        player_ent: int | None = None

        for ent, (pos, rend) in esper.get_components(Position, Renderable):
            name = "Unknown"
            if esper.has_component(ent, Name):
                name = esper.component_for_entity(ent, Name).value
            entity_lookup[(pos.x, pos.y)] = (rend.glyph, name)
            if esper.has_component(ent, Player):
                player_pos = pos
                player_ent = ent

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
        draw_autotile_variant = getattr(r, "draw_autotile_variant", None)

        self._render_map_layer(r, draw_autotile_variant)

        # draw tuple: (vx, vy, glyph, classification, fg, bg, force_glyph)
        DrawData = tuple[int, int, str, str, tuple[int, int, int] | None, tuple[int, int, int] | None, bool]
        player_draw: DrawData | None = None
        character_draws: list[DrawData] = []
        for ent, (pos, rend) in esper.get_components(Position, Renderable):
            is_player = esper.has_component(ent, Player)
            is_character = is_player or esper.has_component(ent, NPC)
            if (pos.x, pos.y) not in self._visible_tiles and not is_player:
                continue

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

        self._draw_sidebar(player_pos, entity_lookup)

        # Dim the world when night draws in (a translucent wash over everything
        # already drawn). Skipped on renderers without the overlay hook (tests).
        clock = world_clock()
        overlay_alpha = night_overlay_alpha(clock)
        draw_overlay = getattr(r, "draw_overlay", None)
        if overlay_alpha > 0 and callable(draw_overlay):
            draw_overlay(_NIGHT_TINT, overlay_alpha)

        draw_text_clipped = getattr(r, "draw_text_clipped", None)
        time_text = f"{format_datetime(clock)}  "
        needs_text = ""
        if player_ent is not None and esper.has_component(player_ent, Needs):
            needs = esper.component_for_entity(player_ent, Needs)
            needs_text = (
                f"Hunger {int(needs.hunger)}%  Thirst {int(needs.thirst)}%  "
                f"Tired {int(needs.tiredness)}%    "
            )
        status_line = f"{time_text}{needs_text}I inv  C status  R sleep  Esc menu"
        if callable(draw_text_clipped):
            draw_text_clipped(0, self._status_y, status_line, self._grid_w)
        else:
            r.draw_text(0, self._status_y, status_line)
        r.present()
