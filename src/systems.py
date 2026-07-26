"""ECS processors (systems).

esper 3.x uses module-level state and esper.Processor subclasses whose
process() receives whatever args are passed to esper.process().
"""
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from math import ceil
import textwrap
import time

import esper

from components import Actor, Age, Asleep, Bed, BerryBush, Blueprint, BlocksMovement, Camp, Chest, ConstructionSite, Corpse, Deer, Dialogue, Diet, Enemy, Equipment, Family, Fish, Friendly, Furniture, Gender, Home, Inventory, Mating, Meat, Name, Needs, NPC, Owned, Personality, Player, Position, Pregnant, Relationships, Renderable, Resident, Sapling, SeaSprout, Seaweed, Settled, Stove, Tree, Vision, WorldClock
from game_map import GameMap, LAND_HEIGHT, LAND_WIDTH
from items import RAW_MEAT, WOOD, cook_meat, hunger_restored, is_cooked_meat, is_raw_meat
from action import BASE_ACTION_COST, action_cost
from onymancer import make_onymancer
from regions import RegionId, RegionScheduler, all_region_ids, in_region_with_margin, region_at
from renderer.base import Renderer, memory_color
from rng import bernoulli_hits, world_rng
import spatial
from content.effects import (
    STATUS_BASE_SECONDS,
    active_effects,
    effect_display,
    effect_label,
)

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
_SEAWEED_SPROUT_GREEN = (86, 178, 148)  # a young frond, paler than a grown one
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
    # Tell the flora processor this bush now has a regrow deadline, so its daily
    # pass never has to ask every bush in the region whether it is bare.
    _PICKED_BUSHES.append(bush_ent)
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
# The status registry (display metadata + which statuses are active) now lives in
# ``content.effects`` so effects are data-driven and moddable. These aliases keep the
# original ``systems`` names (imported by ``main`` and the renderer) working.
active_statuses = active_effects           # (game_map, ent, pos) -> ordered status ids
status_label = effect_label                # status id -> human-readable label


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


def _night_start_fraction() -> float:
    """Where in the day nightfall begins, read off ``_PHASE_BOUNDS`` rather than
    hardcoded, so retuning the phases can't silently desync the closed form below
    from ``is_night``. Assumes the nightfall phases run to the end of the day,
    which is what "night" means; ``test_regions`` pins the two together."""
    lower = 0.0
    for upper, label in _PHASE_BOUNDS:
        if label in _NIGHTFALL_PHASES:
            return lower
        lower = upper
    return 1.0  # no night at all


def night_turns_in(clock: WorldClock, first_turn: int, turns: int) -> int:
    """How many of the ``turns`` baseline turns starting at TU ``first_turn`` fall
    at night -- in closed form, without visiting one.

    This is what lets a lagging region's tiredness be settled for a span of turns
    at once: the only thing that varies across the span is the night multiplier,
    and how many night turns a span contains is arithmetic on the day cycle.

    Counting ``t`` in ``[a, b)`` with ``t mod D >= S`` is
    ``(n // D) * (D - S) + max(0, n % D - S)`` evaluated at both ends.
    """
    day_length = clock.day_length
    step = BASE_ACTION_COST
    if day_length <= 0 or turns <= 0:
        return 0
    if day_length % step or first_turn % step:
        # A day that isn't a whole number of baseline turns (or a span that
        # doesn't start on one) has no clean closed form. Nothing in the game
        # produces that today; count it out rather than get it subtly wrong.
        return sum(
            1
            for i in range(turns)
            if is_night(replace(clock, turn=first_turn + i * step))
        )
    day_turns = day_length // step
    # The first turn of the day that counts as night.
    night_from = -(-int(_night_start_fraction() * day_length) // step)  # ceil-divide
    first = first_turn // step

    def nights_before(turn_index: int) -> int:
        """How many turn-of-day indices in ``[0, turn_index)`` are night."""
        whole, rest = divmod(turn_index, day_turns)
        return whole * (day_turns - night_from) + max(0, rest - night_from)

    return nights_before(first + turns) - nights_before(first)


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


def _camp_at(x: int, y: int, game_map: GameMap | None = None) -> int | None:
    """The campfire on a tile, if any. With a map this asks the index for that
    tile's own region -- a camp pitched on another island can't be on this tile,
    and every camp in the world is a lot to walk through to learn that."""
    if game_map is not None:
        for ent, (pos, _camp) in spatial.ensure(game_map).components(
            region_at(game_map, x, y), Position, Camp
        ):
            if pos.x == x and pos.y == y:
                return ent
        return None
    for ent, (pos, _camp) in esper.get_components(Position, Camp):
        if (pos.x, pos.y) == (x, y):
            return ent
    return None


def go_to_sleep(ent: int, in_camp: bool, game_map: GameMap | None = None) -> None:
    """Put a character to sleep. Camping pitches a campfire on its tile (if one
    isn't already there); sleeping at home just marks it ``Asleep``. Pass the map
    so the "is one already here" check stays regional."""
    if not esper.has_component(ent, Asleep):
        esper.add_component(ent, Asleep(in_camp=in_camp))
    if in_camp and esper.has_component(ent, Position):
        pos = esper.component_for_entity(ent, Position)
        if _camp_at(pos.x, pos.y, game_map) is None:
            esper.create_entity(
                Position(pos.x, pos.y),
                Renderable(_CAMP_GLYPH, fg=_CAMP_ORANGE),
                Name("Camp"),
                Camp(),
            )


def wake_up(ent: int, game_map: GameMap | None = None) -> None:
    """Wake a sleeper: drop the ``Asleep`` tag, break any camp it pitched, and
    (for the player) log that it rose. Safe to call on an already-awake entity."""
    in_camp = False
    if esper.has_component(ent, Asleep):
        in_camp = esper.component_for_entity(ent, Asleep).in_camp
        esper.remove_component(ent, Asleep)
    # Anything that wakes a sleeper early -- being attacked, the player shaking it
    # -- cancels the rest of a compacted sleep. Leaving the receipt behind would
    # make an awake creature invisible to the per-turn systems for the remainder.
    if esper.has_component(ent, Settled):
        esper.remove_component(ent, Settled)
    if in_camp and esper.has_component(ent, Position):
        pos = esper.component_for_entity(ent, Position)
        camp_ent = _camp_at(pos.x, pos.y, game_map)
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
    """Enclosed house interiors. Delegates to ``GameMap.enclosed_rooms``, which
    caches per island and rebuilds only the island(s) edited since the last call --
    so this stays cheap even on the 100-island world. ``cache`` is accepted for
    backward compatibility but no longer needed (the map owns the cache now)."""
    enclosed = getattr(game_map, "enclosed_rooms", None)
    if callable(enclosed):
        return enclosed()
    # Fallback for a map object without the cached API (older test doubles).
    revision = getattr(game_map, "revision", 0)
    if cache.get("revision") != revision:
        cache["revision"] = revision
        cache["houses"] = game_map.find_enclosed_rooms()
    return cache["houses"]


def beds_by_position() -> dict[tuple[int, int], int]:
    """Map every bed's tile to its entity. Build once per turn and reuse for many
    ``_bed_in_interior`` lookups, rather than rescanning all beds per house."""
    return {(pos.x, pos.y): ent for ent, (pos, _bed) in esper.get_components(Position, Bed)}


def _bed_in_interior(
    interior: frozenset[tuple[int, int]],
    beds_by_pos: dict[tuple[int, int], int] | None = None,
) -> int | None:
    """The bed entity inside ``interior``, or ``None``. Pass ``beds_by_pos`` (from
    ``beds_by_position``) to look up the room's own tiles -- O(room) -- instead of
    scanning every bed in the world."""
    if beds_by_pos is not None:
        for tile in interior:
            ent = beds_by_pos.get(tile)
            if ent is not None:
                return ent
        return None
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


# Who owns which bed, the other way round: owner -> bed. ``owned_bed_of`` is asked
# once per resident per turn and used to answer it by walking every owned bed in the
# world; at a hundred islands that is a few hundred beds per villager per turn to
# learn one fact this dict holds outright. Verified on read (the bed must still
# exist and still be theirs), so a stale entry can never hand back a wrong home.
_bed_of_owner: dict[int, int] = {}


def set_bed_owner(bed_ent: int, owner: int) -> None:
    if esper.has_component(bed_ent, Owned):
        esper.component_for_entity(bed_ent, Owned).owner = owner
    else:
        esper.add_component(bed_ent, Owned(owner))
    # Chests are owned through this same call; only a bed is somebody's *home*.
    if esper.has_component(bed_ent, Bed):
        _bed_of_owner[owner] = bed_ent


# The population of ``Owned`` at the last full sweep. While it holds, the map above
# is known complete -- which is what lets "this villager owns nothing" (the common
# answer, asked of every homeless resident every turn) be answered without a scan.
_bed_owner_sweep_population = -1


def owned_bed_of(ent: int) -> int | None:
    """The bed a person owns (their house), or ``None`` if they own none."""
    global _bed_owner_sweep_population
    bed_ent = _bed_of_owner.get(ent)
    if bed_ent is not None:
        if (
            esper.entity_exists(bed_ent)
            and esper.has_component(bed_ent, Bed)
            and esper.has_component(bed_ent, Owned)
            and esper.component_for_entity(bed_ent, Owned).owner == ent
        ):
            return bed_ent
        del _bed_of_owner[ent]  # the bed burned down, or changed hands

    population = spatial.component_population(Owned)
    if population == _bed_owner_sweep_population:
        return None  # nothing has been claimed since the sweep: they own nothing
    _bed_of_owner.clear()
    for owned_ent, (owned, _bed) in esper.get_components(Owned, Bed):
        if esper.entity_exists(owned_ent):
            _bed_of_owner[owned.owner] = owned_ent
    _bed_owner_sweep_population = population
    return _bed_of_owner.get(ent)


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


def furnish_house(
    game_map: GameMap,
    interior: frozenset[tuple[int, int]],
    occupied: set[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    """Populate a house interior with a bed, oven, chest, table, wardrobe, and
    bookshelf on distinct floor tiles, keeping the doorway clear. Returns the bed
    tile (a resident's sleep spot) or ``None`` if the room was too small.

    Blocking furniture is only placed where it keeps the whole interior -- the bed
    especially -- reachable from the door, so a resident can always walk in and
    lie down (never sealed behind its own furniture).

    ``occupied`` may be a shared, caller-maintained set of taken tiles (furnishing
    updates it in place as it drops furniture). When omitted it is built here from a
    scan of blocking/bed entities -- fine for a one-off house, but a bulk caller
    (e.g. furnishing every island of the archipelago) should pass one shared set to
    avoid rescanning the whole entity table per house."""
    interior_set = set(interior)
    door_adjacent: set[tuple[int, int]] = set()
    for (x, y) in interior:
        if any(game_map.tile_at(nx, ny) == game_map.DOOR for nx, ny in game_map.neighbors_4(x, y)):
            door_adjacent.add((x, y))

    if occupied is None:
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


# How far from a villager ``choose_build_site`` can possibly place a cabin: its
# outermost search ring (21) plus the cabin's own footprint and margin. Anything
# occupied further out than this cannot affect the choice, so gathering blockers
# is bounded by this window rather than by the size of the world.
_BUILD_SEARCH_REACH = 30


def occupied_near(
    game_map: GameMap, center: tuple[int, int], radius: int
) -> set[tuple[int, int]]:
    """Every tile within a Chebyshev window of ``center`` that something sits on.

    The static blockers come off the spatial index's tile map; corpses, saplings
    and blueprint ghosts don't block movement but must not be built over either,
    so they come from the index's buckets for the window's regions. This used to
    be built by scanning every blocking entity in the world -- for a search that
    can only ever look thirty tiles.
    """
    index = spatial.ensure(game_map)
    cx, cy = center
    blockers = index.blockers
    occupied = {
        (x, y)
        for y in range(cy - radius, cy + radius + 1)
        for x in range(cx - radius, cx + radius + 1)
        if (x, y) in blockers
    }
    home_region = region_at(game_map, cx, cy)
    rx, ry = home_region
    for ny in range(ry - 1, ry + 2):
        for nx in range(rx - 1, rx + 2):
            for kind in (Corpse, Sapling, Blueprint):
                for ent in index.of_kind((nx, ny), kind):
                    if not esper.entity_exists(ent) or not esper.has_component(ent, Position):
                        continue
                    pos = esper.component_for_entity(ent, Position)
                    if abs(pos.x - cx) <= radius and abs(pos.y - cy) <= radius:
                        occupied.add((pos.x, pos.y))
    return occupied


def choose_build_site(game_map: GameMap, near: tuple[int, int], occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
    """Find a clear top-left corner for a cabin near ``near``: a WxH block of
    open floor tiles (no water, walls, borders, or occupants) with a one-tile
    margin so cabins don't fuse. Searches outward in rings; returns ``None`` if
    nowhere fits within range.

    The innermost radius sweeps a solid block; every larger radius then only
    tests the *fresh perimeter ring*, because a tile the loop already rejected on
    a smaller radius (with the same map and ``occupied``) can't pass on a larger
    one. This visits the same candidate corners in the same order a full
    per-radius square scan would -- so the chosen site is identical -- while
    touching each tile once instead of re-testing the whole interior at every
    radius (~10x fewer ``_site_is_clear`` calls out at the far rings)."""
    nx, ny = near
    # Innermost solid block (radius 3): nothing was scanned before it.
    for oy in range(ny - 3, ny + 4):
        for ox in range(nx - 3, nx + 4):
            if _site_is_clear(game_map, ox, oy, occupied):
                return (ox, oy)
    # Wider radii: only the tiles newly reachable at this radius -- the ring. Its
    # cells are visited top row (left->right), then each middle row's left then
    # right edge, then bottom row (left->right): the exact order a full row-major
    # square scan meets these same cells, so ties resolve identically.
    for radius in range(4, 22):
        top, bottom = ny - radius, ny + radius
        left, right = nx - radius, nx + radius
        for ox in range(left, right + 1):  # top edge
            if _site_is_clear(game_map, ox, top, occupied):
                return (ox, top)
        for oy in range(top + 1, bottom):  # side columns of the middle rows
            if _site_is_clear(game_map, left, oy, occupied):
                return (left, oy)
            if _site_is_clear(game_map, right, oy, occupied):
                return (right, oy)
        for ox in range(left, right + 1):  # bottom edge
            if _site_is_clear(game_map, ox, bottom, occupied):
                return (ox, bottom)
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


def _obstructions_at(game_map: GameMap, x: int, y: int) -> list[int]:
    """Trees, corpses, bushes and saplings sitting on one tile.

    Asks the spatial index for that tile's own region rather than indexing every
    obstruction in the world: a wall is raised on one tile, and what stands a
    hundred tiles of ocean away can never be on it.
    """
    index = spatial.ensure(game_map)
    region = region_at(game_map, x, y)
    found: list[int] = []
    for comp in _OBSTRUCTION_COMPONENTS:
        for ent in sorted(index.of_kind(region, comp)):
            if not esper.entity_exists(ent) or not esper.has_component(ent, Position):
                continue
            pos = esper.component_for_entity(ent, Position)
            if pos.x == x and pos.y == y:
                found.append(ent)
    return found


def _clear_tile_obstructions(game_map: GameMap, x: int, y: int) -> None:
    """Delete any tree/corpse/bush/sapling sitting on (x, y) so it can't get
    sealed into a wall or trapped inside a house being raised over it."""
    for ent in _obstructions_at(game_map, x, y):
        # Guard entity_exists: a listed obstruction may have been eaten/cleared
        # earlier in this same building burst.
        if esper.entity_exists(ent) and any(
            esper.has_component(ent, comp) for comp in _OBSTRUCTION_COMPONENTS
        ):
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
        _clear_tile_obstructions(game_map, ix, iy)  # nothing gets trapped indoors
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
    _clear_tile_obstructions(game_map, x, y)
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

        for _ent, (pos, _player) in esper.get_components(Position, Player):
            nx, ny = pos.x + dx, pos.y + dy
            # Who is on the tile we're stepping onto. This used to build a dict of
            # every blocking entity in the world on every single keypress -- ~85k
            # entries on the big archipelago, for one tile's worth of question.
            target_ent = self._blocker_at(nx, ny)
            target_occupied = target_ent is not None and target_ent != _ent
            # The player can swim, so movement uses is_passable (land + water);
            # walls still block. NPCs keep using is_walkable (land only).
            if self.game_map.is_passable(nx, ny):
                if not target_occupied:
                    old_xy = (pos.x, pos.y)
                    pos.x, pos.y = nx, ny
                    spatial.moved(_ent, old_xy, (nx, ny))
                    continue

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

    def _blocker_at(self, x: int, y: int) -> int | None:
        """Whoever blocks the tile ``(x, y)``, or ``None``.

        Two cheap questions instead of one whole-world dict: the static furniture
        (trees, wells, beds, walls-in-a-box) comes straight off the index by tile,
        and creatures -- which move too often to key by tile -- are checked against
        that tile's own region only, a few dozen entities.
        """
        index = spatial.ensure(self.game_map)
        static = index.blocker_at(x, y)
        if static is not None:
            return static
        for ent, (pos, _npc, _blocks) in index.components(
            region_at(self.game_map, x, y), Position, NPC, BlocksMovement
        ):
            if pos.x == x and pos.y == y:
                return ent
        return None


# How far into a neighbouring region an NPC's goal search still looks, so
# standing near a simulation-region seam doesn't hide an otherwise-closer
# resource just across it. Matches ``_SOCIAL_SIGHT`` -- the widest existing
# "how far can a creature notice something" range in this file.
_REGION_BORDER_MARGIN = 8
# Wall-clock budget (seconds) a real *active* turn (a keypress) spends nudging
# non-current NPC regions along in the background. Zero by design: while the
# player is actively taking turns we simulate only their own region, so a
# keypress stays cheap and the distant world is free to fall behind. That debt
# is paid down off the keypress path -- the main loop's idle-time pump advances
# the nearest lagging regions whenever the player pauses, region entry
# (``catch_up_region``) forces a cell current the moment the player steps into
# it, and sleep (``catch_up_all``) brings the whole world up to date. Raise this
# above zero only to trade keypress latency for less lag during held movement.
_NPC_BACKGROUND_BUDGET = 0.0
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
# The *static* buckets (trees, stoves) barely change -- trees only grow/die on a
# day boundary, stoves never move -- so they are scanned far less often than the
# dynamic buckets (moving prey/NPCs, toggling berries, appearing corpses). This is
# what keeps the big tree scan (80% of all entities) off the per-turn hot path.
_STATIC_SNAPSHOT_REFRESH_CALLS = 400
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
# How fast a fish gets hungry, per BASE_ACTION_COST of world time.
#
# Set so a fish needs exactly **one bite a day**: hunger must climb
# ``_FISH_GRAZE_RESTORE`` over a day, and a day is ``day_length /
# BASE_ACTION_COST`` region-turns (240 at the shipping numbers), giving
# 40/240 = 1/6. The generic ``Needs.hunger_rate`` of 1.0 made that six bites a
# day, which put 1121 fish 3.4x beyond what the sea's seaweed regrowth could
# feed -- a deficit hidden only by wandering fish rarely finding their food.
# ``wildlife.FISH.bites_per_day`` is the same figure for the population model,
# so a fish the player is watching eats on the schedule the stocks assume.
_FISH_HUNGER_RATE = _FISH_GRAZE_RESTORE / (WorldClock.day_length / BASE_ACTION_COST)
# Ocean mirror of ``_NPC_BACKGROUND_BUDGET``: zero, so an active turn swims only
# the player's own region and distant shoals fall behind, catching up via the
# idle pump / region entry / sleep. See that constant for the full rationale.
_FISH_BACKGROUND_BUDGET = 0.0
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

    def __init__(self, game_map: GameMap, live_region_only: bool = False):
        self.game_map = game_map
        self.live_region_only = live_region_only
        self._cache: dict = {}
        # Back-off memo for the (expensive) build-site search: entity -> the
        # (region cell, that region's edit-count) at which its last search found
        # nowhere to build. While the villager stays in that cell and no tile in
        # it has changed, re-running the ~thousand-check search would just fail
        # again, so we skip it until it moves to a new cell or the terrain there
        # changes (either of which could open a spot). See ``_ensure_site_for``.
        self._no_site: dict[int, tuple[RegionId, int]] = {}
        # Region buckets for the two things that aren't positioned entities and so
        # can't live in the spatial index: house interiors (rebuilt on a map edit)
        # and construction sites (rebuilt when one is staked or finished).
        self._houses_by_region: dict[RegionId, list[frozenset[tuple[int, int]]]] | None = None
        self._houses_by_region_rev: int = -1
        self._sites_by_region: dict[RegionId, list[tuple[int, ConstructionSite]]] = {}
        self._all_sites: list[tuple[int, ConstructionSite]] = []
        self._sites_population: int = -1

    def process(self, action: str | None = None) -> None:
        if action not in _TURN_ACTIONS:
            return
        active_region = self._player_region() if self.live_region_only else None
        houses = self._houses_in(active_region)
        sites = self._sites_in(active_region)
        # Compute the unowned houses ONCE per turn (interior + bed tile), rather than
        # having every homeless resident rescan every house. Each resident then just
        # picks the nearest reachable one and pops it from this shared list, so the
        # whole housing pass is O(houses + residents), not O(houses x residents).
        beds_by_pos = self._beds_in(active_region)
        unowned_houses: list[tuple[frozenset[tuple[int, int]], tuple[int, int]]] = []
        for interior in houses:
            bed_ent = _bed_in_interior(interior, beds_by_pos)
            if bed_ent is None or bed_owner(bed_ent) is not None:
                continue  # no bed, or already someone's house
            bed_pos = esper.component_for_entity(bed_ent, Position)
            unowned_houses.append((interior, (bed_pos.x, bed_pos.y)))

        for ent, _res in self._residents_in(active_region):
            if owned_bed_of(ent) is not None:
                continue  # already owns a house -- never re-claim or rebuild
            if self._handle_spouse_housing(ent):
                continue  # married couples settle into one shared home
            if self._claim_unowned_house(ent, unowned_houses):
                continue
            self._ensure_site_for(ent, sites)

    def _player_region(self) -> RegionId | None:
        for _ent, (pos, _player) in esper.get_components(Position, Player):
            return region_at(self.game_map, pos.x, pos.y)
        return None

    # The four things a housing turn needs -- houses, build sites, beds, residents --
    # each scoped to the live region. Whole-world versions of all four used to run
    # every turn and then throw away everything that wasn't local.

    def _residents_in(self, active_region: RegionId | None):
        if active_region is None:  # whole-world mode (tests, headless tools)
            yield from ((ent, res) for ent, (res,) in list(esper.get_components(Resident)))
            return
        index = spatial.ensure(self.game_map)
        yield from ((ent, comps[0]) for ent, comps in list(index.components(active_region, Resident)))

    def _houses_in(self, active_region: RegionId | None) -> list[frozenset[tuple[int, int]]]:
        """The region's houses, bucketed once per map revision. House interiors are
        tiles, not entities, so this is a geometry cache rather than an index --
        but it is rebuilt on a map edit, not walked every turn."""
        houses = houses_for(self.game_map, self._cache)
        if active_region is None:
            return houses
        revision = getattr(self.game_map, "revision", 0)
        if self._houses_by_region_rev != revision or self._houses_by_region is None:
            buckets: dict[RegionId, list[frozenset[tuple[int, int]]]] = {}
            for interior in houses:
                if not interior:
                    continue
                region = region_at(self.game_map, *next(iter(interior)))
                buckets.setdefault(region, []).append(interior)
            self._houses_by_region = buckets
            self._houses_by_region_rev = revision
        return self._houses_by_region.get(active_region, [])

    def _sites_in(self, active_region: RegionId | None) -> list[tuple[int, ConstructionSite]]:
        """This region's construction sites. ``ConstructionSite`` has no position of
        its own (it is a set of tiles), so it is bucketed here and rebuilt only when
        a site is created or finished -- an O(1) population check, not a scan."""
        population = spatial.component_population(ConstructionSite)
        if self._sites_population != population:
            buckets: dict[RegionId, list[tuple[int, ConstructionSite]]] = {}
            everything: list[tuple[int, ConstructionSite]] = []
            for ent, (site,) in esper.get_components(ConstructionSite):
                everything.append((ent, site))
                if site.pieces:
                    region = region_at(self.game_map, *next(iter(site.pieces)))
                    buckets.setdefault(region, []).append((ent, site))
            self._sites_by_region = buckets
            self._all_sites = everything
            self._sites_population = population
        if active_region is None:
            return list(self._all_sites)
        return list(self._sites_by_region.get(active_region, []))

    def _beds_in(self, active_region: RegionId | None) -> dict[tuple[int, int], int]:
        """Tile -> bed entity for the live region. Beds never move, so with an index
        this is that region's bed bucket instead of every bed in the world."""
        if active_region is None:
            return beds_by_position()
        beds: dict[tuple[int, int], int] = {}
        for ent, (pos, _bed) in spatial.ensure(self.game_map).components(active_region, Position, Bed):
            beds[(pos.x, pos.y)] = ent
        return beds

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

    def _claim_unowned_house(
        self,
        ent: int,
        unowned_houses: list[tuple[frozenset[tuple[int, int]], tuple[int, int]]],
    ) -> bool:
        """Claim the nearest reachable unowned house for ``ent``, removing it from
        the shared ``unowned_houses`` list so no one else claims it this turn."""
        if not unowned_houses:
            return False
        pos = esper.component_for_entity(ent, Position) if esper.has_component(ent, Position) else None
        if pos is not None:
            unowned_houses.sort(key=lambda c: _chebyshev((pos.x, pos.y), c[1]))

        for index, (interior, bed_xy) in enumerate(unowned_houses):
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
            unowned_houses.pop(index)  # taken -- keep other settlers off it
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
                self._no_site.pop(ent, None)  # a site exists now -- drop any back-off
                return  # a neighbour already staked one out -- go help build it

        # Back off if we already searched from this same region cell at this same
        # terrain revision and came up empty: nothing there fits and nothing has
        # changed, so re-running the whole outward scan (and rebuilding the
        # occupied set) would only fail again. Moving to a new cell or any tile
        # edit in this one clears the memo below by yielding a different key.
        cell = region_at(self.game_map, pos.x, pos.y)
        rev = self.game_map.region_edit_revision(pos.x, pos.y)
        if self._no_site.get(ent) == (cell, rev):
            return

        origin = choose_build_site(
            self.game_map, here, occupied_near(self.game_map, here, _BUILD_SEARCH_REACH)
        )
        if origin is None:
            self._no_site[ent] = (cell, rev)  # remember: nowhere here, for now
            return
        self._no_site.pop(ent, None)
        site_ent = create_construction_site(self.game_map, origin)
        sites.append((site_ent, esper.component_for_entity(site_ent, ConstructionSite)))
        _push_turn_event("A villager stakes out a blueprint for a new cabin.")

    def _site_reachable(self, here: tuple[int, int], site: ConstructionSite) -> bool:
        for gxy in site.pieces:
            return here == gxy or self.game_map.same_region(here, gxy)
        return False


# Daily, per-tile odds that shape the flora. A seedling can sprout on any open
# tile of its habitat; a mature plant can die (rot, fall, be swept away).
_DAILY_SPROUT_CHANCE = 0.0001       # 0.01% per outdoor ground tile per day (trees)
_DAILY_BUSH_SPROUT_CHANCE = 0.00004  # bushes are rarer than trees
_DAILY_DEATH_CHANCE = 0.00005       # 0.005% per plant per day (half the sprout rate)
# Seaweed sprouts far more often than land flora and matures in a week rather
# than a year: it is the sea's food supply, and fish graze it hard enough that a
# forest's pace would strip the ocean bare. Doubled from the pre-spreading rate
# because spreading (below) makes a sparse reef *slower* to recover, and this is
# the rate at which a healthy reef outgrows the shoal that eats it.
_DAILY_SEAWEED_SPROUT_CHANCE = 0.0012
_SEAWEED_MATURE_DAYS = 7
# Days after a bush's berries are picked before a fresh crop ripens.
_BERRY_REGROW_DAYS = 7

# How much of a plant's sprouting happens regardless of what is already growing
# nearby -- drifting spores, a seed carried in by a bird. The rest is *spreading*
# from the region's standing stock (see ``_spread_factor``), so a wood or a reef
# fills in from what is already there rather than appearing uniformly out of
# nowhere. Without a floor a region grazed or logged to nothing could never come
# back at all, which is worse than the uniform model it replaces.
_COLONISE_SHARE = 0.05


def _spread_factor(standing: int, capacity: int) -> float:
    """How vigorously a plant is spreading in a region right now, in ``0..1``.

    Logistic: growth is fastest at half capacity and tails off at both ends --
    nothing to spread from when the region is bare, nowhere to spread to when it
    is full. This is the ``r*S*(1 - S/K)`` shape, written as a multiplier on the
    per-tile chance so sprouting stays one binomial draw over the region's tiles
    rather than anything that walks the plants.

    Scaled so the peak equals 1.0, which means the per-tile constants keep their
    old meaning: they are now the rate a *thriving* region grows at, rather than
    the rate every region grows at always.
    """
    if capacity <= 0:
        return _COLONISE_SHARE
    filled = min(1.0, max(0.0, standing / capacity))
    return _COLONISE_SHARE + (1.0 - _COLONISE_SHARE) * 4.0 * filled * (1.0 - filled)


# Bushes picked since the flora processor last filed them, awaiting a regrow
# deadline. Picking is the *only* thing that makes a bush bare, so it is also
# the only place a regrow needs scheduling -- which is what lets the daily pass
# look at the handful of bushes that are actually waiting instead of every bush
# in the region. Ids are validated when drained, so a stale one is harmless.
_PICKED_BUSHES: list[int] = []


def reset_flora_queue() -> None:
    """Forget bushes picked in a previous world. Entity ids restart with each
    world, so a leftover id could alias an unrelated entity in the next one;
    the test fixture clears this alongside esper's database."""
    _PICKED_BUSHES.clear()


class TreeGrowthProcessor(esper.Processor):
    """Ages the flora one **day** at a time (it acts on the turn a new day
    begins, matching the per-day odds below).

    Each day: any sapling that has lived a full year (112 days) matures (into a
    tree or a berry bush, per its kind); every tree/bush has a small chance of
    dying; picked bushes regrow their berries once 7 days have passed; and every
    open outdoor ground tile has a small chance of sprouting a fresh sapling. A
    soft cap (scaled to the map) keeps the world from filling solid. The RNG is
    injectable so tests can force or suppress growth.

    Nothing here scans
    ------------------
    Read literally, that description is four sweeps over a region every day:
    every sapling, every plant, every bush, every ground tile. At archipelago
    scale that is ~7200 tiles and a few hundred plants **per region per day**,
    to produce roughly one sprout and one death a week. This processor states
    the same rules without any of the sweeps, in two moves:

    * **Chance is counted, not rolled.** A per-tile/per-plant chance over N
      candidates is a binomial, so the day asks *how many* sprout or die and
      then acts on that many -- see ``rng.bernoulli_hits``. Cost is what happened,
      not what could have.
    * **Deadlines are filed, not searched for.** A sapling's maturity day and a
      picked bush's regrow day are both known the moment they are created, so
      each is filed under the day it comes due (``_mature_due``,
      ``_berry_due``). The daily pass pops the day's entries instead of asking
      every plant whether its time has come yet.

    What a region's flora costs per day is therefore the flora that actually
    changed, plus O(1) bookkeeping -- which is what makes full-world simulation
    affordable. Population counts come from the spatial index
    (``spatial.of_kind``) and the world-wide soft-cap totals from
    ``spatial.component_population``; both are O(1) reads, so not even the caps
    require a scan.
    """

    def __init__(self, game_map: GameMap, rng: Callable[[], float] | None = None):
        self.game_map = game_map
        self._rng = rng if rng is not None else world_rng().stream("flora").random
        # Flora fills the land, so its soft cap scales to the land area (not the
        # whole map -- otherwise the ocean's tiles would inflate the tree budget).
        # ``land_area`` is the whole world's land (the sum over every island on the
        # archipelago); fall back to the single land rect, then the map size.
        land_area = getattr(game_map, "land_area", None) or getattr(
            game_map, "land_w", game_map.width
        ) * getattr(game_map, "land_h", game_map.height)
        self._cap = max(40, land_area * 3 // 10)
        # Seaweed fills the open sea; its cap scales to the ocean area.
        ocean_area = max(0, (game_map.width * game_map.height) - land_area)
        self._seaweed_cap = ocean_area // 28
        # The day flora last aged, tracked PER region so growth can lag behind the
        # world clock and be paid down off the keypress path -- exactly like NPC
        # region sim (see NpcAiProcessor). ``_baseline_day`` is the day of the
        # first ``process`` call: a fresh world has no growth history to replay, so
        # every region starts caught up to "today", not day zero. A region absent
        # from ``_region_day`` is assumed to sit at the baseline.
        self._baseline_day: int | None = None
        self._region_day: dict[RegionId, int] = {}
        # Cached lists of outdoor ground tiles (floor, not inside a house) and of
        # open-sea tiles, rebuilt only when the map changes.
        self._ground_cache: dict = {}
        self._ocean_cache: dict = {}
        # The same two lists bucketed by simulation region, so a single region's
        # daily flora pass touches only its own tiles (the whole point of the
        # region split -- one island's seaweed scan, not the whole sea's). They
        # are *lists*, deliberately: sprouting indexes straight into them (see
        # ``rng.bernoulli_hits``), so a candidate tile is picked in O(1) and the
        # region's tiles are never walked.
        self._ground_by_region: dict = {}
        self._ocean_by_region: dict = {}
        # Timed flora events, filed by the day they come due:
        # ``{region: {due_day: [entity, ...]}}``. A day's pass pops its own key
        # rather than asking every sapling/bush whether it is ready yet.
        self._mature_due: dict[RegionId, dict[int, list[int]]] = {}
        self._berry_due: dict[RegionId, dict[int, list[int]]] = {}
        # The saplings currently filed in ``_mature_due``, per region -- kept so
        # the queue can be checked against the world in O(1) (see
        # ``_sync_sapling_queue``).
        self._filed_saplings: dict[RegionId, set[int]] = {}
        # Regions whose pre-existing flora has been filed into the queues above.
        # A region is seeded once, on its first day pass, which is what picks up
        # plants that world-gen (or a test) created directly.
        self._seeded: set[RegionId] = set()
        # Other systems that advance on the same per-region *day* cursor this
        # processor owns, run in registration order after the flora itself.
        #
        # This exists because a day model that is coupled to another one cannot
        # keep its own cursor. Wild populations eat the flora, so if the two
        # caught up separately -- as they briefly did -- a region could grow
        # thirty days of seaweed with nothing grazing it and then graze thirty
        # days with nothing growing, and land somewhere the interleaved
        # simulation never would. One cursor, one order, every day.
        #
        # It is also the seam the shared world scheduler in ``next.md`` wants:
        # today ``TreeGrowthProcessor`` happens to be where the day cursor lives.
        self._day_steps: list[tuple[str, Callable[[RegionId, int], None]]] = []

    def register_day_step(
        self, name: str, step: Callable[[RegionId, int], None]
    ) -> None:
        """Run ``step(region_id, day)`` as part of each region-day, after growth.

        The registered system must not keep a day cursor of its own; this one is
        authoritative, and it is what keeps the two in lockstep however the debt
        was paid down (a keypress, the idle pump, or a night's sleep).
        """
        self._day_steps.append((name, step))

    def process(self, action: str | None = None) -> None:
        if action not in _TURN_ACTIONS:
            return
        clock = world_clock()
        if clock is None:
            return
        day = clock.turn // max(1, clock.day_length)
        if self._baseline_day is None:
            self._baseline_day = day  # establish a baseline; age the flora from here
            return
        # During live play (a player exists) NO flora ages on the keypress: even
        # a scan-free daily pass is a whole *world's* worth of regions, and the
        # keypress owes the player a frame, not a growing season. It is deferred
        # entirely to spare time (``pump_flora``, which walks the
        # nearest-to-player region first, so the player's own island greens up
        # the instant they pause) and to sleep (``catch_up_all_flora``). With no
        # player at all (unit tests / headless) there is no idle loop to defer to,
        # so age everything inline, matching the old unpartitioned daily pass.
        if self._player_region() is not None:
            return
        self._advance_regions_flora(all_region_ids(self.game_map), day, clock)

    def _player_region(self) -> RegionId | None:
        for _e, (pos, _p) in esper.get_components(Position, Player):
            return region_at(self.game_map, pos.x, pos.y)
        return None

    def pump_flora(
        self,
        budget_seconds: float,
        player_xy: tuple[int, int] | None,
        wall_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Spend up to ``budget_seconds`` aging the *nearest* region whose flora
        lags the world clock, closest to the player first -- the flora mirror of
        ``RegionScheduler.pump_background``. Called from the main loop's idle-time
        pump, so growth the player never waited on happens in spare cycles."""
        clock = world_clock()
        if clock is None or self._baseline_day is None:
            return
        day = clock.turn // max(1, clock.day_length)
        # Cheap early-out for the common idle tick: nothing lags, so no scan.
        lagging = [
            r for r in all_region_ids(self.game_map)
            if self._region_day.get(r, self._baseline_day) < day
        ]
        if not lagging:
            return
        player_region = (
            region_at(self.game_map, player_xy[0], player_xy[1]) if player_xy else None
        )
        if player_region is not None:
            lagging.sort(
                key=lambda r: max(
                    abs(r[0] - player_region[0]), abs(r[1] - player_region[1])
                )
            )
        # One day-context feeds this whole pump; the time budget then governs how
        # many nearby regions we advance a day from it (always at least one).
        # Regions further behind than one day come back around on a later pump;
        # the nearest-first order means the player's own island is always first
        # in line.
        ctx = self._build_day_context(clock)
        deadline = wall_clock() + budget_seconds
        for region_id in lagging:
            self._advance_one_day(region_id, ctx)
            if wall_clock() >= deadline:
                break

    def warm_region_caches(self) -> None:
        """Build the static per-region ocean buckets up front. That scan is a
        one-off ~0.5s at archipelago scale; doing it here (behind the world-gen
        screen) keeps it off the first idle pump that would otherwise trigger it."""
        if getattr(self.game_map, "has_ocean", False):
            self._ocean_buckets()

    def catch_up_all_flora(self) -> None:
        """Bring every region's flora fully up to the world clock. Wired to sleep,
        where the world is expected to jump forward all at once."""
        clock = world_clock()
        if clock is None or self._baseline_day is None:
            return
        day = clock.turn // max(1, clock.day_length)
        self._advance_regions_flora(all_region_ids(self.game_map), day, clock)

    def _advance_regions_flora(
        self, regions: list[RegionId], target_day: int, clock: WorldClock
    ) -> None:
        """Age each of ``regions`` one day at a time up to ``target_day``, in
        strict order so state one day builds is consistent for the next. The
        day-context (chiefly the world-wide soft-cap totals) is built once per
        day-step and shared by every region advancing that step."""
        while True:
            due = [
                r for r in regions if self._region_day.get(r, self._baseline_day) < target_day
            ]
            if not due:
                return
            ctx = self._build_day_context(clock)
            for region_id in due:
                self._advance_one_day(region_id, ctx)

    def _build_day_context(self, clock: WorldClock) -> dict:
        """What a day's flora pass needs that isn't per-region: the clock, the
        spatial index it asks for regional populations, and the running
        world-wide totals the soft caps compare against.

        All three are O(1) to obtain -- the totals come from
        ``spatial.component_population``, which reads a length rather than
        counting the world's plants. The regional detail this used to precompute
        (trees, bushes, saplings and every occupied tile, bucketed by region, off
        five whole-world entity scans) is gone: each pass now asks the index for
        just its own region, and only when it has something to do.
        """
        ctx = {
            "clock": clock,
            "index": spatial.ensure(self.game_map),
            # Each habitat's cap counts only its own plants, grown and young.
            # Sea sprouts are a separate component precisely so 4000 young fronds
            # cannot tell the *forests* they are full (see SeaSprout).
            "flora_total": (
                spatial.component_population(Tree)
                + spatial.component_population(BerryBush)
                + spatial.component_population(Sapling)
            ),
            "seaweed_total": (
                spatial.component_population(Seaweed)
                + spatial.component_population(SeaSprout)
            ),
        }
        self._drain_picked_bushes(ctx)
        return ctx

    def _advance_one_day(self, region_id: RegionId, ctx: dict) -> None:
        """Run ``region_id``'s next day of growth and move its cursor onto it."""
        day = self._region_day.get(region_id, self._baseline_day) + 1
        self._run_region_day(region_id, day, ctx)
        self._region_day[region_id] = day

    def _run_region_day(self, region_id: RegionId, day: int, ctx: dict) -> None:
        """One region's worth of a single day's growth.

        ``day`` is the region's *own* day -- the one it is being advanced onto,
        which for a lagging region is behind the world clock. Deadlines are
        judged against it, so a region that has not been simulated for a season
        still ripens its berries on the day they were due, not all at once when
        someone finally looks.
        """
        self._seed_region(region_id, ctx)
        self._sync_sapling_queue(region_id, ctx)
        self._mature_saplings(region_id, day, ctx)
        self._kill_flora(region_id, ctx)
        self._regrow_berries(region_id, day, ctx)
        self._sprout_saplings(region_id, day, ctx)
        if getattr(self.game_map, "has_ocean", False):
            self._sprout_seaweed(region_id, day, ctx)
        # Whatever else lives on this day cursor -- wild populations grazing what
        # just grew. Last, so a day's growth is on the table before it is eaten.
        for _name, step in self._day_steps:
            step(region_id, day)

    # --- deadline queues ---------------------------------------------------

    @staticmethod
    def _due_day(turn: int, delay_days: int, day_length: int) -> int:
        """The day ``delay_days`` after the one ``turn`` fell in.

        Growth is quantized to days, so a deadline is counted in days from the
        day it was set -- "berries regrow seven days after they were picked",
        whichever hour of that day they were taken. This is the timing the old
        per-plant test (``turn - stamped >= delay``) produced whenever the pass
        ran on schedule; unlike that test it does not also depend on how far
        behind the world clock the region happens to be.
        """
        return turn // max(1, day_length) + delay_days

    @staticmethod
    def _file(queue: dict[int, list[int]], due_day: int, ent: int) -> None:
        queue.setdefault(due_day, []).append(ent)

    @staticmethod
    def _pop_due(queue: dict[int, list[int]], day: int) -> list[int]:
        """Everything filed for ``day`` or any earlier day, removed from the
        queue. Earlier days are swept up too so a region that jumped forward
        (or was seeded with something already overdue) never strands an entry."""
        overdue = [d for d in queue if d <= day]
        if not overdue:
            return []
        ready: list[int] = []
        for d in sorted(overdue):
            ready.extend(queue.pop(d))
        return ready

    def _seed_region(self, region_id: RegionId, ctx: dict) -> None:
        """File the regrow deadlines of bushes this processor never saw picked.

        World-gen spawns ripe bushes, so in a fresh world this finds nothing; it
        exists for a bush made bare some other way (a tool, a save, a test) that
        would otherwise sit bare forever. Runs once per region.

        Juveniles need no equivalent: ``_sync_sapling_queue`` notices an unfiled
        seedling by a length comparison and rebuilds, so a first visit to a region
        full of saplings files them by that route.
        """
        if region_id in self._seeded:
            return
        self._seeded.add(region_id)
        clock, index = ctx["clock"], ctx["index"]
        berry_queue = self._berry_due.setdefault(region_id, {})
        # ``_drain_picked_bushes`` has already run for this day-step, so anything
        # picked through ``pick_berries`` is on the queue -- which is every bush
        # in normal play. Skipping those is what keeps a bush from being filed
        # twice; what is left for this scan is a bush made bare some other way.
        already = {ent for ents in berry_queue.values() for ent in ents}
        for ent in sorted(index.of_kind(region_id, BerryBush) - already):
            bush = esper.component_for_entity(ent, BerryBush)
            if not bush.has_berries and bush.harvested_turn is not None:
                self._file(
                    berry_queue,
                    self._due_day(bush.harvested_turn, _BERRY_REGROW_DAYS, clock.day_length),
                    ent,
                )

    def _drain_picked_bushes(self, ctx: dict) -> None:
        """Move bushes picked since the last pass onto their regrow deadline.

        ``pick_berries`` is the only way a bush goes bare, so this is the whole
        of "which bushes are waiting to regrow" -- no bush is ever asked.
        """
        if not _PICKED_BUSHES:
            return
        clock, index = ctx["clock"], ctx["index"]
        picked, _PICKED_BUSHES[:] = list(_PICKED_BUSHES), []
        for ent in picked:
            if not esper.entity_exists(ent) or not esper.has_component(ent, BerryBush):
                continue
            bush = esper.component_for_entity(ent, BerryBush)
            if bush.has_berries or bush.harvested_turn is None:
                continue  # already regrown (or never really picked)
            region_id = index.region_of(ent)
            if region_id is None:
                continue
            self._file(
                self._berry_due.setdefault(region_id, {}),
                self._due_day(bush.harvested_turn, _BERRY_REGROW_DAYS, clock.day_length),
                ent,
            )

    def _sync_sapling_queue(self, region_id: RegionId, ctx: dict) -> None:
        """Repair the maturity queue if it has drifted from the world.

        Sprouting files a sapling and maturing unfiles it, so in normal play the
        queue and the region's sapling population move together and this costs
        one length comparison. A mismatch means something created or destroyed a
        sapling behind our back, and the queue is rebuilt from the index -- the
        same "check a cheap length, repair by rebuilding" discipline the spatial
        index itself uses, and for the same reason: a missed hook then costs one
        rebuild instead of a sapling that never grows up.
        """
        index = ctx["index"]
        saplings = index.of_kind(region_id, Sapling)
        sprouts = index.of_kind(region_id, SeaSprout)
        filed = self._filed_saplings.setdefault(region_id, set())
        if len(filed) == len(saplings) + len(sprouts):
            return
        clock = ctx["clock"]
        queue = self._mature_due.setdefault(region_id, {})
        queue.clear()
        filed.clear()
        for ent in sorted(saplings):
            sapling = esper.component_for_entity(ent, Sapling)
            self._file(
                queue,
                self._due_day(sapling.planted_turn, _DAYS_PER_YEAR, clock.day_length),
                ent,
            )
            filed.add(ent)
        for ent in sorted(sprouts):
            sprout = esper.component_for_entity(ent, SeaSprout)
            self._file(
                queue,
                self._due_day(sprout.planted_turn, _SEAWEED_MATURE_DAYS, clock.day_length),
                ent,
            )
            filed.add(ent)

    def _mature_saplings(self, region_id: RegionId, day: int, ctx: dict) -> None:
        """Grow up the saplings whose year is up today -- the ones filed under
        this day, not every sapling in the region asked how old it is."""
        queue = self._mature_due.setdefault(region_id, {})
        due = self._pop_due(queue, day)
        if not due:
            return
        blockers = self._region_blockers(region_id, ctx)
        filed = self._filed_saplings.setdefault(region_id, set())
        for ent in due:
            filed.discard(ent)
            if not esper.entity_exists(ent):
                continue
            # Land and sea juveniles share this queue -- a deadline is a deadline
            # -- but carry different components so the two habitats stay countable
            # apart (see ``SeaSprout``).
            if esper.has_component(ent, SeaSprout):
                kind = "seaweed"
            elif esper.has_component(ent, Sapling):
                kind = esper.component_for_entity(ent, Sapling).kind
            else:
                continue
            pos = esper.component_for_entity(ent, Position)
            if (pos.x, pos.y) in blockers:
                # Something stands on it -- try again tomorrow rather than turn
                # into a wall of wood under whoever is there.
                self._file(queue, day + 1, ent)
                filed.add(ent)
                continue
            esper.remove_component(ent, SeaSprout if kind == "seaweed" else Sapling)
            if kind == "bush":
                esper.add_component(ent, BlocksMovement())
                esper.add_component(ent, BerryBush())
                _set_bush_appearance(ent, True)
                if esper.has_component(ent, Name):
                    esper.component_for_entity(ent, Name).value = "Berry Bush"
            elif kind == "seaweed":
                # Seaweed is the one flora that doesn't block: fish swim over it,
                # which is how they graze it.
                esper.add_component(ent, Seaweed())
                if esper.has_component(ent, Renderable):
                    rend = esper.component_for_entity(ent, Renderable)
                    rend.glyph = '"'
                    rend.fg = _SEAWEED_GREEN
                    rend.bg = _WATER_BLUE
                if esper.has_component(ent, Name):
                    esper.component_for_entity(ent, Name).value = "Seaweed"
            else:
                esper.add_component(ent, BlocksMovement())
                esper.add_component(ent, Tree())
                if esper.has_component(ent, Renderable):
                    rend = esper.component_for_entity(ent, Renderable)
                    rend.glyph = "T"
                    rend.fg = _TREE_GREEN
                if esper.has_component(ent, Name):
                    esper.component_for_entity(ent, Name).value = "Tree"
            # It was a seedling and is now a grown plant of some kind: the index
            # buckets it by what it is, so it has to be told.
            spatial.reclassify(ent)

    def _kill_flora(self, region_id: RegionId, ctx: dict) -> None:
        """Rot and fell the day's share of the region's mature flora.

        How many die is a binomial draw over the region's plant count (an O(1)
        read off the spatial index), so the usual answer -- none -- costs one
        random number and the region's plants are never touched at all. Only
        once a death is drawn is the population materialised, in entity order,
        so which plant dies is reproducible from the seed.
        """
        index = ctx["index"]
        # Seaweed rots and is swept away like anything else that grows -- before,
        # a frond could only ever leave the world down a fish.
        for kind in (Tree, BerryBush, Seaweed):
            population = index.of_kind(region_id, kind)
            doomed = list(bernoulli_hits(len(population), _DAILY_DEATH_CHANCE, self._rng))
            if not doomed:
                continue
            ordered = sorted(population)  # snapshot: we are about to delete from it
            for i in doomed:
                ent = ordered[i]
                if esper.entity_exists(ent):
                    esper.delete_entity(ent, immediate=True)

    def _regrow_berries(self, region_id: RegionId, day: int, ctx: dict) -> None:
        """Ripen the bushes whose seven days are up today. Only bushes that were
        actually picked are ever on the queue, so a region full of ripe bushes
        costs nothing."""
        for ent in self._pop_due(self._berry_due.setdefault(region_id, {}), day):
            if not esper.entity_exists(ent) or not esper.has_component(ent, BerryBush):
                continue
            bush = esper.component_for_entity(ent, BerryBush)
            if bush.has_berries:
                continue  # picked, regrown and picked again? the later filing wins
            bush.has_berries = True
            bush.harvested_turn = None
            _set_bush_appearance(ent, True)

    def _outdoor_ground(self) -> list[tuple[int, int]]:
        """Every regular ground tile that is outdoors (floor, and not inside an
        enclosed house). Cached until the map changes."""
        revision = getattr(self.game_map, "revision", 0)
        if self._ground_cache.get("revision") != revision:
            interiors: set[tuple[int, int]] = set()
            for interior in self.game_map.enclosed_rooms():
                interiors |= interior
            tiles = [
                (x, y)
                for y in range(1, self.game_map.height - 1)
                for x in range(1, self.game_map.width - 1)
                if self.game_map.tiles[y][x] == self.game_map.FLOOR and (x, y) not in interiors
            ]
            self._ground_cache = {"revision": revision, "tiles": tiles}
        return self._ground_cache["tiles"]

    def _ground_buckets(self) -> dict[RegionId, list[tuple[int, int]]]:
        """``_outdoor_ground`` split by simulation region, rebuilt when the map
        (house interiors) changes."""
        revision = getattr(self.game_map, "revision", 0)
        cache = self._ground_by_region
        if cache.get("revision") != revision:
            buckets: dict[RegionId, list[tuple[int, int]]] = {}
            for x, y in self._outdoor_ground():
                buckets.setdefault(region_at(self.game_map, x, y), []).append((x, y))
            self._ground_by_region = {"revision": revision, "buckets": buckets}
        return self._ground_by_region["buckets"]

    def _sprout_saplings(self, region_id: RegionId, day: int, ctx: dict) -> None:
        """Seed the region's day of new growth.

        The old shape of this was "for every outdoor ground tile, roll" -- 7200
        rolls a day per region to plant about one sapling. Here the day asks the
        distribution how many tiles sprout and gets the tiles back with them
        (``rng.bernoulli_hits``), so the work is the saplings, not the ground. Each
        one is still checked against the soft cap and against what is already on
        its tile, and the ratio of trees to bushes is the same ratio the two
        per-tile chances gave.

        The tile check is two O(1) index reads and a shrug: a candidate that is
        already grown over or built on is simply given up on, the way the
        per-tile version skipped an occupied tile. Creatures are deliberately not
        consulted -- a sapling under a passing villager is harmless (it doesn't
        block, and ``_mature_saplings`` won't let it become a tree while anyone
        is standing there), and checking for them is what would drag a
        region's whole population back into a pass that no longer needs it.
        """
        if ctx["flora_total"] >= self._cap:
            return
        tiles = self._ground_buckets().get(region_id, ())
        clock, index = ctx["clock"], ctx["index"]
        # A wood fills in from the wood that is already there: the per-tile odds
        # are scaled by how vigorously this region is spreading right now (see
        # ``_spread_factor``). Standing stock is the region's *mature* plants --
        # a seedling can't seed -- and both are O(1) index reads, so this costs
        # nothing and sprouting stays a single binomial draw.
        standing = len(index.of_kind(region_id, Tree)) + len(
            index.of_kind(region_id, BerryBush)
        )
        spread = _spread_factor(standing, self._region_capacity(region_id, tiles))
        # Dated to the region's own day, not the world clock: growth is a
        # day-quantized simulation, and a region that is a season behind must
        # plant its saplings on the day it is living, or they would mature a
        # season late (see ``_run_region_day``).
        planted_turn = day * max(1, clock.day_length)
        sprout_chance = (_DAILY_SPROUT_CHANCE + _DAILY_BUSH_SPROUT_CHANCE) * spread
        tree_share = _DAILY_SPROUT_CHANCE / (
            _DAILY_SPROUT_CHANCE + _DAILY_BUSH_SPROUT_CHANCE
        )
        for i in bernoulli_hits(len(tiles), sprout_chance, self._rng):
            if ctx["flora_total"] >= self._cap:
                break
            x, y = tiles[i]
            if index.rooted_at(x, y) is not None or index.blocker_at(x, y) is not None:
                continue  # already grown over, or somebody built on it
            if self._rng() < tree_share:
                kind, glyph, name = "tree", "t", "Sapling"
            else:
                kind, glyph, name = "bush", ",", "Bush Seedling"
            ent = esper.create_entity(
                Position(x, y),
                Renderable(glyph, fg=_SAPLING_GREEN),
                Name(name),
                Sapling(planted_turn=planted_turn, kind=kind),
            )
            # Its maturity is a known date, so file it now; the day it comes due
            # is the only day this sapling is ever looked at again.
            self._file(
                self._mature_due.setdefault(region_id, {}), day + _DAYS_PER_YEAR, ent
            )
            self._filed_saplings.setdefault(region_id, set()).add(ent)
            ctx["flora_total"] += 1

    def _region_capacity(
        self, region_id: RegionId, tiles, sea: bool = False
    ) -> int:
        """This region's share of the world's soft cap for a habitat.

        The caps are world-wide numbers; spreading is a *local* question -- a bare
        stretch of coast should fill in even while the world as a whole is near
        its ceiling. So the world cap is spread over the world's habitat tiles and
        the region gets its share of that density.
        """
        if sea:
            total = len(self._ocean_tiles())
            cap = self._seaweed_cap
        else:
            total = len(self._outdoor_ground())
            cap = self._cap
        if total <= 0:
            return 0
        return int(len(tiles) * cap / total)

    def _region_blockers(self, region_id: RegionId, ctx: dict) -> set[tuple[int, int]]:
        """The region's tiles that something blocking is standing on -- what a
        maturing sapling must not turn into a tree underneath.

        The one place a region's population is still read in bulk, and the only
        one that has to be: unlike sprouting, this genuinely cares about
        *creatures*, which move and so cannot be kept in a tile map cheaply. It
        is built only when a sapling actually comes due (about once a day in the
        few regions that have land at all) and cached for the day-step.
        """
        cache = ctx.setdefault("blockers", {})
        tiles = cache.get(region_id)
        if tiles is None:
            tiles = {
                (pos.x, pos.y)
                for _e, (pos, _b) in ctx["index"].components(
                    region_id, Position, BlocksMovement
                )
            }
            cache[region_id] = tiles
        return tiles

    def _ocean_buckets(self) -> dict[RegionId, list[tuple[int, int]]]:
        """``_ocean_tiles`` split by simulation region. Static like its source, so
        built once -- this is what lets one region's seaweed pass scan its own few
        thousand sea tiles instead of the whole ~557k-tile ocean every day."""
        key = (self.game_map.width, self.game_map.height)
        cache = self._ocean_by_region
        if cache.get("key") != key:
            buckets: dict[RegionId, list[tuple[int, int]]] = {}
            for x, y in self._ocean_tiles():
                buckets.setdefault(region_at(self.game_map, x, y), []).append((x, y))
            self._ocean_by_region = {"key": key, "buckets": buckets}
        return self._ocean_by_region["buckets"]

    def _ocean_tiles(self) -> list[tuple[int, int]]:
        """Every open-sea tile (water outside the land). Built once and kept: the
        ocean is genuinely static -- ``is_ocean`` is a pure function of the map's
        fixed island grid (or land rect), and building only ever edits land tiles,
        so no open-sea tile can appear or vanish mid-game. Keying this on
        ``game_map.revision`` (as the outdoor-ground cache must, since houses do
        change it) would rebuild this ~1.1M-tile scan every in-game day the
        villagers lay a single wall -- a ~0.5s hitch for a set that never moves."""
        key = (self.game_map.width, self.game_map.height)
        if self._ocean_cache.get("key") != key:
            tiles = [
                (x, y)
                for y in range(1, self.game_map.height - 1)
                for x in range(1, self.game_map.width - 1)
                if self.game_map.is_ocean(x, y)
            ]
            self._ocean_cache = {"key": key, "tiles": tiles}
        return self._ocean_cache["tiles"]

    def _sprout_seaweed(self, region_id: RegionId, day: int, ctx: dict) -> None:
        """Seed one region's open water, so grazing fish keep the sea fed.

        The same shape as ``_sprout_saplings`` in every respect -- counted rather
        than rolled, spreading from what already stands, and seeding a *seedling*
        that takes ``_SEAWEED_MATURE_DAYS`` to become food rather than a full
        frond appearing from nothing. That delay is what gives a grazed reef a
        recovery time instead of an instant one.
        """
        if ctx["seaweed_total"] >= self._seaweed_cap:
            return
        tiles = self._ocean_buckets().get(region_id, ())
        clock, index = ctx["clock"], ctx["index"]
        standing = len(index.of_kind(region_id, Seaweed))
        spread = _spread_factor(standing, self._region_capacity(region_id, tiles, sea=True))
        planted_turn = day * max(1, clock.day_length)
        for i in bernoulli_hits(len(tiles), _DAILY_SEAWEED_SPROUT_CHANCE * spread, self._rng):
            if ctx["seaweed_total"] >= self._seaweed_cap:
                break
            x, y = tiles[i]
            if index.rooted_at(x, y) is not None:
                continue  # something already grows there; fish swim over, so that
                          # is the only thing an open-sea tile can be taken by
            ent = esper.create_entity(
                Position(x, y),
                Renderable("'", fg=_SEAWEED_SPROUT_GREEN, bg=_WATER_BLUE),
                Name("Seaweed Sprout"),
                SeaSprout(planted_turn=planted_turn),
            )
            self._file(
                self._mature_due.setdefault(region_id, {}),
                day + _SEAWEED_MATURE_DAYS,
                ent,
            )
            self._filed_saplings.setdefault(region_id, set()).add(ent)
            ctx["seaweed_total"] += 1


# The glyph/colour a newborn villager shares with the adult cast.
_VILLAGER_GLYPH = "v"
_BABY_SKIN = (222, 184, 156)


class ReproductionProcessor(esper.Processor):
    """Delivers babies. Once a day (a cheap day-boundary pass, like the flora)
    it checks every pregnant woman: when the gestation period has elapsed a child
    is born beside her -- named by the onymancer, sharing the family surname, and
    wired to both parents. Newborns live at the mother's home and grow up on the
    world clock; they are not ``Resident`` (a baby doesn't build its own cabin)."""

    def __init__(self, game_map: GameMap | None = None) -> None:
        self._last_day: int | None = None
        # In live play, pass a map: only the region the player is in delivers on the
        # keypress. Everywhere else a birth is a thing that happened while nobody was
        # looking, and is delivered when that region is caught up (``pump_births``,
        # region entry, or sleep). Omit the map for the old whole-world behaviour,
        # which is what the unit tests drive.
        self.game_map = game_map
        # The day each region last delivered, so a region can lag the world clock
        # and be brought current off the keypress path -- the same bookkeeping the
        # flora uses (see TreeGrowthProcessor).
        self._baseline_day: int | None = None
        self._region_day: dict[RegionId, int] = {}
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
            self._baseline_day = day
            return
        if day == self._last_day:
            return
        self._last_day = day
        active = self._player_region()
        if active is None:
            self._deliver_due_pregnancies(clock)  # no player: unpartitioned pass
            return
        # Strictly the live region only. The rest of the world's babies arrive when
        # their region does (pump_births / catch_up_all_births).
        self._deliver_due_pregnancies(clock, region_id=active)
        self._region_day[active] = day

    def _player_region(self) -> RegionId | None:
        if self.game_map is None:
            return None
        for _ent, (pos, _player) in esper.get_components(Position, Player):
            return region_at(self.game_map, pos.x, pos.y)
        return None

    def pump_births(self, player_xy: tuple[int, int] | None = None) -> None:
        """Deliver the babies owed by regions the player isn't standing in --
        nearest first, like every other background catch-up. Called from the main
        loop's idle pump, so an off-screen birth costs spare time, never a turn."""
        clock = world_clock()
        if clock is None or self.game_map is None or self._baseline_day is None:
            return
        day = current_day(clock)
        lagging = [
            r for r in all_region_ids(self.game_map)
            if self._region_day.get(r, self._baseline_day) < day
        ]
        if not lagging:
            return
        if player_xy is not None:
            here = region_at(self.game_map, player_xy[0], player_xy[1])
            lagging.sort(key=lambda r: max(abs(r[0] - here[0]), abs(r[1] - here[1])))
        for region_id in lagging:
            self._deliver_due_pregnancies(clock, region_id=region_id)
            self._region_day[region_id] = day

    def catch_up_all_births(self) -> None:
        """Bring the whole world's pregnancies current -- what sleeping does."""
        self.pump_births(None)

    def _deliver_due_pregnancies(
        self, clock: WorldClock, region_id: RegionId | None = None
    ) -> None:
        term = _GESTATION_DAYS * max(1, clock.day_length)
        for mother, pregnant in self._pregnancies(region_id):
            if clock.turn - pregnant.conceived_turn < term:
                continue
            esper.remove_component(mother, Pregnant)
            self._give_birth(mother, pregnant.father, clock)

    def _pregnancies(self, region_id: RegionId | None):
        """Expectant mothers, in one region or (with no map) the whole world.

        ``Pregnant`` comes and goes at runtime so it isn't a bucketed kind; the
        region's *people* are, and there are only ever a few dozen of those.
        """
        if region_id is None or self.game_map is None:
            yield from (
                (ent, comps[0]) for ent, comps in list(esper.get_components(Pregnant))
            )
            return
        index = spatial.ensure(self.game_map)
        for ent in sorted(index.of_kind(region_id, Personality)):
            if esper.entity_exists(ent) and esper.has_component(ent, Pregnant):
                yield ent, esper.component_for_entity(ent, Pregnant)

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


def sleep_turns_needed(needs: Needs) -> int:
    """How many region-turns of unbroken sleep it takes to get ``needs`` rested.

    ``NeedsProcessor._accrue`` pays ``_SLEEP_RECOVERY`` off tiredness per turn and
    the caller wakes the sleeper the turn tiredness reaches zero, so this is just
    that division rounded up -- always at least one turn, since even a barely tired
    creature that lies down sleeps a turn.
    """
    return max(1, ceil(needs.tiredness / _SLEEP_RECOVERY))


def settle_sleep(needs: Needs, turns: int) -> None:
    """Apply ``turns`` region-turns of sleep to ``needs`` in one go.

    The closed form of ``NeedsProcessor._accrue``'s asleep branch: tiredness falls
    by ``_SLEEP_RECOVERY`` a turn, hunger and thirst climb at their own rates
    whether awake or asleep, and all three clamp at the same bounds. Sleep is the
    purest compactable activity in the game -- nothing about it depends on
    anything that happens during it -- so the N turns can be run as arithmetic
    instead of N loop iterations.

    Lives here, beside ``_accrue``, precisely so the two can't drift apart; the
    test suite pins them together (``test_activity_compaction``).
    """
    needs.tiredness = max(0.0, needs.tiredness - _SLEEP_RECOVERY * turns)
    needs.hunger = min(needs.max_value, needs.hunger + needs.hunger_rate * turns)
    needs.thirst = min(needs.max_value, needs.thirst + needs.thirst_rate * turns)


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

    def __init__(self, game_map: GameMap | None = None) -> None:
        # In live play, pass a map: needs then belong to a *region's* turn rather
        # than the world's, and are advanced wherever that region's turn is run.
        # Unit tests and tools can keep the old whole-world pass by omitting it.
        self.game_map = game_map
        # World-clock TU at the last accrual, so we can charge only the elapsed span.
        self._last_turn: int | None = None
        # Set once this processor is registered as a region step (see
        # ``register_region_step``). From then on a *turn* only ticks the player --
        # every other creature gets hungry as its own region is simulated, which is
        # what lets a region that has been asleep for a thousand turns wake up with
        # a thousand turns of hunger instead of the hunger it had when you left.
        self._scheduler_driven = False
        # Set alongside it, so a replayed turn can be dated to the region's own
        # cursor (``_as_of_clock``) rather than to whenever the replay happened.
        self._scheduler = None

    def register_region_step(self, scheduler) -> None:
        """Make needs part of a region's turn.

        Registered on the region scheduler, so hunger accrues in exactly the places
        a region is allowed to simulate: the live region each turn, and everywhere
        else in the background pump, on region entry, and during sleep.
        """
        scheduler.register("needs", self.advance_region, bulk=self.advance_region_by)
        self._scheduler_driven = True
        # Kept so a replayed turn can be dated to the region's own history rather
        # than to the moment the replay happens -- see ``_as_of_clock``.
        self._scheduler = scheduler

    def _as_of_clock(self, region_id: RegionId) -> WorldClock | None:
        """The world clock **as of the region-turn being run**, not the true "now".

        A lagging region replays turn N, N+1, N+2 … long after they happened, so
        dating them by the real clock would charge a turn that happened at midnight
        as though it were noon. Worse, it would not even be *stable*: how far a
        region lags depends on the wall-clock budget the background pump got, so
        the same seed and the same player inputs would accrue different tiredness
        on a faster machine. Reading the region's own cursor instead makes the
        answer depend only on which turn it is, which is the whole determinism
        requirement (see ``RegionScheduler``).

        Mirrors what ``NpcAiProcessor._advance_region`` does with the same cursor,
        and reads it at the same point -- the scheduler bumps it only after every
        step has run -- so the AI and the needs it drives agree on the time of day.
        """
        clock = world_clock()
        if clock is None or self._scheduler is None:
            return clock
        logical_turn = (self._scheduler.region_turn[region_id] + 1) * BASE_ACTION_COST
        return replace(clock, turn=logical_turn)

    def advance_region(self, region_id: RegionId) -> None:
        """One region-turn of hunger, thirst and tiredness for one region.

        A region-turn is one baseline action's worth of time (``BASE_ACTION_COST``),
        so replaying a region's backlog accrues exactly the needs those turns would
        have accrued had they been simulated as they happened.

        The player is excluded: their needs follow the world clock in ``process``,
        because a slow action must make *them* proportionally hungrier -- an NPC has
        no such thing as a slow action, it has region-turns.

        This is ``advance_region_by(region_id, 1)`` and nothing else. Keeping a
        separate one-turn implementation would mean two ways to age a creature that
        have to be proved equal forever; a span of one is a span.
        """
        self.advance_region_by(region_id, 1)

    def advance_region_by(self, region_id: RegionId, turns: int) -> None:
        """``turns`` region-turns of needs for one region, in closed form.

        Needs are the easy half of analytic catch-up: hunger and thirst are linear
        in elapsed turns, sleep pays tiredness down linearly, and the only thing
        that varies across the span is the night multiplier -- which
        ``night_turns_in`` counts without visiting a turn. So a region that has
        been asleep for a thousand turns gets a thousand turns of appetite for the
        price of one pass over its creatures.

        Identical to running ``advance_region`` ``turns`` times, and the suite
        pins that (``test_regions``). The clamps don't spoil it: hunger and thirst
        only rise and tiredness only falls here, so clamping the endpoint is the
        same as clamping every step.
        """
        if self.game_map is None or turns <= 0:
            return
        clock = self._as_of_clock(region_id)
        nights = (
            night_turns_in(clock, clock.turn, turns) if clock is not None else 0
        )
        woke: list[int] = []
        for ent, (needs,) in spatial.ensure(self.game_map).components(region_id, Needs):
            if esper.has_component(ent, Player):
                continue
            if esper.has_component(ent, Settled):
                if self._burn_settled_turns(ent, turns):
                    woke.append(ent)
                continue
            needs.hunger = min(needs.max_value, needs.hunger + needs.hunger_rate * turns)
            needs.thirst = min(needs.max_value, needs.thirst + needs.thirst_rate * turns)
            if esper.has_component(ent, Asleep):
                needs.tiredness = max(0.0, needs.tiredness - _SLEEP_RECOVERY * turns)
                if needs.tiredness <= 0.0:
                    woke.append(ent)
            else:
                # Day turns at the plain rate, night turns at the night multiplier.
                weighted = (turns - nights) + nights * _NIGHT_TIREDNESS_MULTIPLIER
                needs.tiredness = min(
                    needs.max_value, needs.tiredness + needs.tiredness_rate * weighted
                )
        for ent in woke:
            wake_up(ent, self.game_map)

    @staticmethod
    def _burn_settled_turns(ent: int, turns: int) -> bool:
        """Consume ``turns`` region-turns of a compacted activity at once. Returns
        True when that used the receipt up and the entity was asleep."""
        settled = esper.component_for_entity(ent, Settled)
        settled.turns -= turns
        if settled.turns > 0:
            return False
        esper.remove_component(ent, Settled)
        return esper.has_component(ent, Asleep)

    @staticmethod
    def _burn_settled_turn(ent: int) -> bool:
        """Consume one region-turn of a compacted activity. Returns True when that
        was the last one and the entity was asleep -- i.e. the sleep it settled has
        now really elapsed and it should wake, on exactly the turn it would have
        woken had every turn been simulated one at a time."""
        return NeedsProcessor._burn_settled_turns(ent, 1)

    def _accrue(self, ent: int, needs: Needs, scale: float, night: bool) -> bool:
        """Age one creature's needs by ``scale`` baseline turns. Returns True if it
        has now slept itself rested and should wake."""
        prev_hunger = needs.hunger
        prev_thirst = needs.thirst
        prev_tiredness = needs.tiredness
        asleep = esper.has_component(ent, Asleep)
        rested = False

        # Hunger and thirst creep up whether awake or asleep.
        needs.hunger = min(needs.max_value, needs.hunger + needs.hunger_rate * scale)
        needs.thirst = min(needs.max_value, needs.thirst + needs.thirst_rate * scale)

        if asleep:
            # Sleeping pays down tiredness; waking is handled by the caller.
            needs.tiredness = max(0.0, needs.tiredness - _SLEEP_RECOVERY * scale)
            rested = needs.tiredness <= 0.0
        else:
            rate = needs.tiredness_rate * (_NIGHT_TIREDNESS_MULTIPLIER if night else 1.0)
            needs.tiredness = min(needs.max_value, needs.tiredness + rate * scale)

        # Every creature accumulates needs, but only the player's are surfaced as
        # log warnings -- a hungry goblin shouldn't print "You are starving!".
        if esper.has_component(ent, Player):
            hunger_msg = _crossed_warning(prev_hunger, needs.hunger, _HUNGER_WARNINGS)
            if hunger_msg is not None:
                _push_turn_event(hunger_msg)
            thirst_msg = _crossed_warning(prev_thirst, needs.thirst, _THIRST_WARNINGS)
            if thirst_msg is not None:
                _push_turn_event(thirst_msg)
            # Tiredness warnings only make sense while awake (it falls in sleep).
            if not asleep:
                tired_msg = _crossed_warning(
                    prev_tiredness, needs.tiredness, _TIREDNESS_WARNINGS
                )
                if tired_msg is not None:
                    _push_turn_event(tired_msg)
        return rested

    def _player_region(self) -> RegionId | None:
        if self.game_map is None:
            return None
        for _ent, (pos, _player) in esper.get_components(Position, Player):
            return region_at(self.game_map, pos.x, pos.y)
        return None

    def _ticking_entities(self, active_region: RegionId | None):
        """Whose needs this *turn* advances.

        Driven by the region scheduler, that is the player alone -- everyone else
        is the business of their own region's turn. Without a scheduler (a
        processor built directly in a unit test) it is the old whole-world pass,
        which is what those tests assert.
        """
        if self._scheduler_driven:
            for ent, (needs, _player) in esper.get_components(Needs, Player):
                yield ent, needs
            return
        if self.game_map is None or active_region is None:
            yield from ((ent, needs) for ent, (needs,) in esper.get_components(Needs))
            return
        for ent, (needs,) in spatial.ensure(self.game_map).components(active_region, Needs):
            yield ent, needs

    def resync_to(self, turn: int) -> None:
        """Treat world time up to ``turn`` as already charged.

        For a caller that settled a span of the player's needs in closed form and
        jumped the clock over it (sleep does exactly that). Without this the next
        turn would look back at a clock that leapt a whole night and charge the
        player for it a second time.
        """
        self._last_turn = turn

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
        for ent, needs in self._ticking_entities(self._player_region()):
            # Same receipt as the region pass: turns a compacted activity already
            # applied must not be applied again by the whole-world tick either.
            if esper.has_component(ent, Settled):
                if self._burn_settled_turn(ent):
                    woke.append(ent)
                continue
            if self._accrue(ent, needs, scale, night):
                woke.append(ent)

        for ent in woke:
            wake_up(ent, self.game_map)


# RenderProcessor live in render.py; imported here so
# `from systems import ...` still resolves them. Always import them via systems.
from render import RenderProcessor  # noqa: E402


# NpcAiProcessor, FishAiProcessor live in ai.py; imported here so
# `from systems import ...` still resolves them. Always import them via systems.
from ai import NpcAiProcessor, FishAiProcessor  # noqa: E402
