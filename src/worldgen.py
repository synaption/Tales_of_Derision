"""World setup: populate a fresh map with the player, the starting village, wildlife,
flora, and survival stations. All spawning goes through content prefabs
(``content.registry.spawn`` / ``build_components``); the villager/player bundles use
``content.kits.person_kit`` plus the clock-dependent pieces (age, per-name colour).

Seed-deterministic: the onymancer and the forest/ocean scatter draw from
``rng.world_rng`` streams, so a given seed always regenerates the same world. Keep
the RNG draw order stable when editing -- worldgen reproducibility is tested
(``test_determinism.py``).
"""
from __future__ import annotations

import esper

from components import (
    Age, Attributes, Bed, BlocksMovement, Equipment, Family, Gender, Inventory,
    Name, Needs, Player, Position, Renderable, Vision, WorldClock,
)
from game_map import GameMap
from items import WOOD
from onymancer import make_onymancer
from queries import first_player_entity
from rng import world_rng
from systems import born_turn_for_age, furnish_house, set_bed_owner
from content.items import default_equipment_slots
from content.kits import person_kit
from content.loader import load_all_content
from content.registry import build_components, spawn

# Human skin tones, indexed by a stable hash of a person's name so a given villager
# keeps the same complexion across runs (no RNG -- purely name-derived).
_HUMAN_SKIN_TONES: list[tuple[int, int, int]] = [
    (255, 238, 220),
    (248, 227, 208),
    (241, 216, 196),
    (234, 205, 184),
    (227, 194, 172),
    (220, 183, 160),
    (213, 172, 148),
    (206, 161, 136),
    (194, 149, 123),
    (182, 137, 110),
    (170, 125, 98),
    (158, 113, 86),
    (146, 101, 74),
    (134, 89, 62),
    (122, 77, 50),
    (110, 65, 38),
]


def _human_skin_tone(seed_text: str, offset: int = 0) -> tuple[int, int, int]:
    seed = 0
    for index, char in enumerate(seed_text):
        seed += (index + 1) * ord(char)
    return _HUMAN_SKIN_TONES[(seed + offset) % len(_HUMAN_SKIN_TONES)]


def _spawn_cave_rat(x: int, y: int, *, include_loot: bool) -> None:
    # A quick, scurrying carnivore (high dexterity -> acts ~1.2x as often; see
    # action.actor_speed). The base creature is the "cave_rat" prefab; the single
    # scripted rat carries a bit of loot, the rat-flood swarm does not.
    components = build_components("cave_rat", x, y)
    if include_loot:
        components.extend(
            [
                Inventory(items=["String", "Pebble"]),
                Equipment(slots=default_equipment_slots()),
            ]
        )
    esper.create_entity(*components)


def _nearest_walkable(game_map: GameMap, x: int, y: int) -> Position:
    """The closest walkable (land) tile to ``(x, y)``, searched in growing rings.
    On the classic single island the target is already land, so this is a no-op;
    on the archipelago it nudges a spawn off a narrow water margin onto the island
    it was meant for, so land NPCs never start stranded in the sea. Falls back to
    the original tile if nothing walkable is found."""
    if game_map.is_walkable(x, y):
        return Position(x, y)
    for radius in range(1, max(game_map.width, game_map.height)):
        for ny in range(y - radius, y + radius + 1):
            for nx in range(x - radius, x + radius + 1):
                # Only the ring at this radius (Chebyshev), so nearer tiles win first.
                if max(abs(nx - x), abs(ny - y)) != radius:
                    continue
                if game_map.is_walkable(nx, ny):
                    return Position(nx, ny)
    return Position(x, y)


def _setup_world(game_map: GameMap, player_position: Position, rat_flood: bool = False) -> int:
    # Register all content (prefabs/kits/effects/items) before anything is spawned.
    # Idempotent, so tests that call _setup_world directly get a populated registry.
    load_all_content()

    # Seat the player at the centre of the island nearest the map centre (on the
    # archipelago the geometric centre is open water between islands). On the classic
    # single-island / room maps there is one island and its centre is the map centre,
    # so this reproduces the original spawn. The Position object is shared with the
    # player entity created below, so setting it here places that entity.
    islands = _islands_of(game_map)
    landing = _nearest_walkable(game_map, player_position.x, player_position.y)
    player_island = _island_containing(islands, landing.x, landing.y)
    if player_island is None:
        player_island = 0
    center = _nearest_walkable(game_map, *_island_center(islands[player_island]))
    player_position.x, player_position.y = center.x, center.y

    # Singleton world clock driving the day/night cycle and the night-time
    # tiredness ramp. Created before any creature so the first turn has a time.
    # The game opens mid-morning so the player starts a fresh day in daylight.
    start_clock = WorldClock()
    start_clock.turn = int(start_clock.day_length * 0.2)
    esper.create_entity(start_clock)

    player_name = "You"
    player_skin = _human_skin_tone(player_name)

    player_equipment = default_equipment_slots()
    player_equipment["main hand"] = "Rusty Sword"
    player_equipment["chest"] = "Traveler Tunic"
    esper.create_entity(
        player_position,
        Renderable("@", fg=player_skin),
        Name(player_name),
        # Placeholder until character creation lets the player choose; the future
        # reproduction system reads this.
        Gender("male"),
        Age(born_turn_for_age(start_clock, 25.0)),
        Player(),
        Vision(10),
        BlocksMovement(),
        # Average attributes for now; dexterity drives how long the player's
        # actions take in the action economy (see action.action_cost).
        Attributes(),
        # Some starting wood so the player can craft walls/doors/windows right
        # away (Crafting tab in the Tab menu, then place from the inventory).
        Inventory(items=["Bandage", "Torch", "Apple", WOOD, WOOD, WOOD, WOOD, WOOD]),
        Equipment(slots=player_equipment),
        Needs(),
    )

    if rat_flood:
        rats_spawned = 0
        player_xy = (player_position.x, player_position.y)
        for y in range(game_map.height):
            for x in range(game_map.width):
                if not game_map.is_walkable(x, y):
                    continue
                if (x, y) == player_xy:
                    continue
                _spawn_cave_rat(x, y, include_loot=False)
                rats_spawned += 1
        return rats_spawned

    # Populate every island with the same starting content the lone island got: a
    # village of two households, a goblin scout, a cave rat, and a clustered forest
    # with roaming deer. Each island draws from its own seeded RNG streams (keyed by
    # index) so the archipelago is reproducible yet the islands are not carbon
    # copies. This is the scale test: ~100x the entities and AI of one island.
    # One running set of occupied tiles threaded through every island, so placement
    # never rescans the (fast-growing) entity table -- crucial at 100-island scale.
    occupied: set[tuple[int, int]] = {(player_position.x, player_position.y)}
    for idx, rect in enumerate(islands):
        _populate_island(game_map, rect, idx, start_clock, occupied)

    # The player's own island also gets the survival starter kit (well, stove, owned
    # bed, a few trees/bushes) laid out around the player.
    _place_player_kit(game_map, player_position, occupied)

    if getattr(game_map, "has_ocean", False):
        _spawn_ocean_life(game_map)

    # Furnish the pre-built houses (bed, oven, chest, table, wardrobe, bookshelf)
    # across every island. Residents claim these unowned houses as homes at runtime.
    # The shared ``occupied`` set is passed through so 200 houses don't each rescan
    # the whole (80k-entity) table.
    for interior in game_map.find_enclosed_rooms():
        furnish_house(game_map, interior, occupied)

    return 1


# The starting cast of one island: two households (a family of four and a couple),
# so relationships include spouses, parents/children, and siblings from the outset.
# Contrasting personalities keep the social AI producing both warming (++) and
# souring (--) interactions. Each entry is (gender, traits, offset-from-island-centre,
# age_years); parents are adults, children too young to marry until they grow up.
_Person = tuple[str, list[str], tuple[int, int], float]
_HOUSEHOLDS: list[tuple[_Person, _Person, list[_Person]]] = [
    (
        ("male", ["Cheerful", "Outgoing"], (-2, 1), 34.0),
        ("female", ["Kind", "Shy"], (-3, 2), 31.0),
        [
            ("male", ["Grumpy"], (-4, 1), 9.0),
            ("female", ["Playful", "Kind"], (-2, 3), 6.0),
        ],
    ),
    (
        ("male", ["Aloof"], (-5, 2), 28.0),
        ("female", ["Outgoing", "Playful"], (-3, 3), 26.0),
        [],
    ),
]


def _islands_of(game_map: GameMap) -> list[tuple[int, int, int, int]]:
    """Every island's land rect. The archipelago records these directly; a lone
    island or plain room is treated as a one-island world (its single land rect)."""
    islands = getattr(game_map, "islands", None)
    if islands:
        return list(islands)
    return [(
        getattr(game_map, "land_x0", 0),
        getattr(game_map, "land_y0", 0),
        getattr(game_map, "land_w", game_map.width),
        getattr(game_map, "land_h", game_map.height),
    )]


def _island_center(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    lx, ly, lw, lh = rect
    return (lx + lw // 2, ly + lh // 2)


def _island_containing(
    islands: list[tuple[int, int, int, int]], x: int, y: int
) -> int | None:
    """Index of the island whose land rect contains ``(x, y)``, or ``None`` if the
    point is in open water between islands."""
    for idx, (lx, ly, lw, lh) in enumerate(islands):
        if lx <= x < lx + lw and ly <= y < ly + lh:
            return idx
    return None


def _populate_island(
    game_map: GameMap,
    rect: tuple[int, int, int, int],
    idx: int,
    clock: WorldClock,
    occupied: set[tuple[int, int]],
) -> None:
    """Spawn one island's village, lone monsters, forest, and deer -- the same
    content the single-island world produced, offset to this island and seeded off
    its index so each island is deterministic but distinct. ``occupied`` is the
    shared running set of taken tiles (updated as blockers are placed)."""
    cx, cy = _island_center(rect)
    # The onymancer names this island's villagers; seeding by index keeps names
    # reproducible per seed while differing island to island.
    onymancer = make_onymancer(world_rng().int_seed(f"names_{idx}"))

    def place(offset: tuple[int, int]) -> Position:
        dx, dy = offset
        return _nearest_walkable(game_map, cx + dx, cy + dy)

    def spawn_villager(pos: Position, gender: str, traits: list[str], surname: str, age_years: float) -> int:
        # The shared villager bundle (gender, family, cook diet, resident,
        # personality, inventory, ...) comes from ``person_kit``; name, colour, and
        # age (which needs the clock) are added here. Family links wired below.
        _given, _surname, full = onymancer.full_name(gender, surname)
        occupied.add((pos.x, pos.y))
        return esper.create_entity(
            pos,
            Renderable("v", fg=_human_skin_tone(full, offset=7)),
            Name(full),
            Age(born_turn_for_age(clock, age_years)),
            *person_kit(gender=gender, traits=traits, surname=surname),
        )

    for father_spec, mother_spec, child_specs in _HOUSEHOLDS:
        surname = onymancer.surname()
        fg, ftraits, foff, fage = father_spec
        mg, mtraits, moff, mage = mother_spec
        father = spawn_villager(place(foff), fg, ftraits, surname, fage)
        mother = spawn_villager(place(moff), mg, mtraits, surname, mage)
        father_fam = esper.component_for_entity(father, Family)
        mother_fam = esper.component_for_entity(mother, Family)
        father_fam.spouse = mother
        mother_fam.spouse = father
        for cg, ctraits, coff, cage in child_specs:
            child = spawn_villager(place(coff), cg, ctraits, surname, cage)
            child_fam = esper.component_for_entity(child, Family)
            child_fam.parents = [father, mother]
            father_fam.children.append(child)
            mother_fam.children.append(child)

    guard = place((-5, 0))
    occupied.add((guard.x, guard.y))
    spawn("goblin_scout", guard.x, guard.y)
    rat = place((6, -2))
    occupied.add((rat.x, rat.y))
    _spawn_cave_rat(rat.x, rat.y, include_loot=True)

    _grow_island_forest(game_map, rect, idx, occupied)


def _grow_island_forest(
    game_map: GameMap,
    rect: tuple[int, int, int, int],
    idx: int,
    occupied: set[tuple[int, int]],
) -> None:
    """Grow one island's forest in clustered stands (~10% of its walkable land) with
    the odd berry bush, then scatter deer near the stands. Reproducible per seed via
    an index-keyed RNG stream. ``occupied`` is the shared running set of taken tiles."""
    lx, ly, lw, lh = rect

    def place_prefab(x: int, y: int, prefab_id: str) -> bool:
        if not game_map.is_walkable(x, y) or (x, y) in occupied:
            return False
        occupied.add((x, y))
        spawn(prefab_id, x, y)
        return True

    rng = world_rng().stream(f"worldgen_env_{idx}")
    walkable = [
        (x, y)
        for y in range(ly, ly + lh)
        for x in range(lx, lx + lw)
        if game_map.is_walkable(x, y)
    ]
    tree_target = int(len(walkable) * 0.10)
    planted = 0
    guard = 0
    while walkable and planted < tree_target and guard < tree_target * 50:
        guard += 1
        scx, scy = rng.choice(walkable)
        radius = rng.randint(2, 6)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if planted >= tree_target:
                    break
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > radius:
                    continue
                # Denser at the core, thinning out toward the stand's edge.
                if rng.random() < dist / (radius + 1):
                    continue
                if rng.random() < 0.04:
                    place_prefab(scx + dx, scy + dy, "berry_bush")
                elif place_prefab(scx + dx, scy + dy, "tree"):
                    planted += 1

    # Deer near the island's stands / water so they can graze and drink.
    for fx, fy in ((0.25, 0.3), (0.5, 0.75), (0.75, 0.5), (0.35, 0.6)):
        dcx, dcy = lx + int(lw * fx), ly + int(lh * fy)
        for off_x, off_y in ((-2, 0), (3, 2)):
            _spawn_deer(game_map, dcx + off_x, dcy + off_y)


def _spawn_ocean_life(game_map: GameMap) -> None:
    """Scatter seaweed and fish across the open sea that surrounds the island.
    Fish graze the seaweed (see ``FishAiProcessor``). Placement draws from the
    world seed so the ocean is reproducible for a given seed."""
    rng = world_rng().stream("worldgen_ocean")
    occupied = {(pos.x, pos.y) for _ent, (pos,) in esper.get_components(Position)}

    for y in range(1, game_map.height - 1):
        for x in range(1, game_map.width - 1):
            if not game_map.is_ocean(x, y) or (x, y) in occupied:
                continue
            roll = rng.random()
            if roll < 0.028:
                spawn("seaweed", x, y)
                occupied.add((x, y))
            elif roll < 0.028 + 0.001:
                spawn("fish", x, y)
                occupied.add((x, y))


def _spawn_deer(game_map: GameMap, x: int, y: int) -> None:
    """Create a wild deer: prey that grazes trees and drinks water, and yields
    Deer Meat when hunted."""
    if not game_map.is_walkable(x, y):
        return
    spawn("deer", x, y)


def _place_player_kit(
    game_map: GameMap, player_position: Position, occupied: set[tuple[int, int]]
) -> None:
    """Lay the survival starter kit around the player on their home island: a well
    and stove for the survival loop, an owned bed, and a few trees/bushes within
    reach. Anything that would land in water (a tight island edge) is simply skipped.
    ``occupied`` is the shared running set of taken tiles."""
    # Safety on the classic maps: never leave the player walled by a lake/river. On
    # the archipelago the player is seated at a whole island's centre, so skip it --
    # carving the surrounding sea would just erode that island's coastline.
    if not getattr(game_map, "is_archipelago", False):
        game_map.clear_water_around(player_position.x, player_position.y, radius=2)

    def place_prefab(x: int, y: int, prefab_id: str) -> bool:
        """Spawn a content prefab on a clear, walkable tile (never the player's)."""
        if not game_map.is_walkable(x, y):
            return False
        if (x, y) in occupied or (x, y) == (player_position.x, player_position.y):
            return False
        occupied.add((x, y))
        spawn(prefab_id, x, y)
        return True

    def place_rel(dx: int, dy: int, prefab_id: str) -> bool:
        return place_prefab(player_position.x + dx, player_position.y + dy, prefab_id)

    place_rel(2, 2, "well")
    place_rel(4, 2, "stove")
    # The player's bed: sleep beside it to rest at home instead of camping. It
    # belongs to the player, so villagers never claim it (even if you wall it in).
    if place_rel(3, 2, "bed"):
        player_ent = first_player_entity()
        bed_xy = (player_position.x + 3, player_position.y + 2)
        if player_ent is not None:
            for bed_ent, (bpos, _bed) in esper.get_components(Position, Bed):
                if (bpos.x, bpos.y) == bed_xy:
                    set_bed_owner(bed_ent, player_ent)
                    break

    # A few starter trees and berry bushes within reach so the player has wood/food
    # from turn one (the island's wider forest is grown by ``_grow_island_forest``).
    for tree_dx, tree_dy in (
        (-3, -2), (-4, -2), (-3, 3), (5, 3), (6, 3),
        (-5, -2), (-4, 3), (6, 4), (-5, 2), (5, 4),
    ):
        place_prefab(player_position.x + tree_dx, player_position.y + tree_dy, "tree")
    for bush_dx, bush_dy in ((-2, 3), (4, -2)):
        place_prefab(player_position.x + bush_dx, player_position.y + bush_dy, "berry_bush")
