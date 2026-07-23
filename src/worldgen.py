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


def _setup_world(game_map: GameMap, player_position: Position, rat_flood: bool = False) -> int:
    # Register all content (prefabs/kits/effects/items) before anything is spawned.
    # Idempotent, so tests that call _setup_world directly get a populated registry.
    load_all_content()

    # Singleton world clock driving the day/night cycle and the night-time
    # tiredness ramp. Created before any creature so the first turn has a time.
    # The game opens mid-morning so the player starts a fresh day in daylight.
    start_clock = WorldClock()
    start_clock.turn = int(start_clock.day_length * 0.2)
    esper.create_entity(start_clock)

    # The onymancer names every villager procedurally, seeded off the world seed
    # so the same seed always conjures the same starting village.
    onymancer = make_onymancer(world_rng().int_seed("names_startup"))

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

    guard_pos = Position(max(2, player_position.x - 5), player_position.y)
    rat_pos = Position(min(game_map.width - 3, player_position.x + 6), max(2, player_position.y - 2))

    def spawn_villager(pos: Position, gender: str, traits: list[str], surname: str, age_years: float) -> int:
        # The onymancer coins the given name (flavoured by gender) and joins it to
        # the family surname. Skin tone still seeds off the final name so it stays
        # stable across runs. The shared villager bundle (gender, family, cook diet,
        # resident, personality, inventory, ...) comes from ``person_kit``; the
        # per-person pieces (name, colour, and age -- which needs the clock) are
        # added here. Family links are wired reciprocally after the whole household
        # is spawned.
        _given, _surname, full = onymancer.full_name(gender, surname)
        return esper.create_entity(
            pos,
            Renderable("v", fg=_human_skin_tone(full, offset=7)),
            Name(full),
            # Born far enough in the past to be their starting age now; parents are
            # adults, the children too young to marry or reproduce until they grow.
            Age(born_turn_for_age(start_clock, age_years)),
            *person_kit(gender=gender, traits=traits, surname=surname),
        )

    def _place(offset: tuple[int, int]) -> Position:
        dx, dy = offset
        vx = min(game_map.width - 3, max(2, player_position.x + dx))
        vy = min(game_map.height - 3, max(2, player_position.y + dy))
        return Position(vx, vy)

    # The starting cast forms two households (a full family of four and a couple),
    # so relationships include spouses, parents/children, and siblings from the
    # outset. Traits are the original contrasting personalities -- so the social AI
    # still produces both warming (++) and souring (--) interactions -- while names,
    # genders, ages, and family ties are new. Each entry is
    # (gender, traits, offset, age_years): the parents are adults, the children too
    # young to marry or reproduce until they grow up. Placement stays a
    # deterministic loose cluster near the player.
    Person = tuple[str, list[str], tuple[int, int], float]
    households: list[tuple[Person, Person, list[Person]]] = [
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

    for father_spec, mother_spec, child_specs in households:
        surname = onymancer.surname()
        fg, ftraits, foff, fage = father_spec
        mg, mtraits, moff, mage = mother_spec
        father = spawn_villager(_place(foff), fg, ftraits, surname, fage)
        mother = spawn_villager(_place(moff), mg, mtraits, surname, mage)
        father_fam = esper.component_for_entity(father, Family)
        mother_fam = esper.component_for_entity(mother, Family)
        father_fam.spouse = mother
        mother_fam.spouse = father
        for cg, ctraits, coff, cage in child_specs:
            child = spawn_villager(_place(coff), cg, ctraits, surname, cage)
            child_fam = esper.component_for_entity(child, Family)
            child_fam.parents = [father, mother]
            father_fam.children.append(child)
            mother_fam.children.append(child)

    spawn("goblin_scout", guard_pos.x, guard_pos.y)
    _spawn_cave_rat(rat_pos.x, rat_pos.y, include_loot=True)

    _spawn_environment_features(game_map, player_position)
    if getattr(game_map, "has_ocean", False):
        _spawn_ocean_life(game_map)

    # Furnish the pre-built houses (bed, oven, chest, table, wardrobe,
    # bookshelf). Residents claim these unowned houses as homes at runtime.
    for interior in game_map.find_enclosed_rooms():
        furnish_house(game_map, interior)

    return 1


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


def _spawn_environment_features(game_map: GameMap, player_position: Position) -> None:
    """Populate the world: a well and stove by the player for the survival loop,
    tree stands and roaming deer scattered across the map, with the wildlife
    biased toward the water so grazing/drinking is nearby."""
    # Safety: never leave the player standing in (or walled by) a lake/river.
    game_map.clear_water_around(player_position.x, player_position.y, radius=2)

    occupied = {
        (pos.x, pos.y)
        for _ent, (pos, _blocks) in esper.get_components(Position, BlocksMovement)
    }

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

    def plant_tree(x: int, y: int) -> bool:
        return place_prefab(x, y, "tree")

    def plant_bush(x: int, y: int) -> bool:
        return place_prefab(x, y, "berry_bush")

    # A few starter trees within reach, plus a forest scattered across the land
    # so both the player and grazing deer have wood/food.
    for tree_dx, tree_dy in (
        (-3, -2), (-4, -2), (-3, 3), (5, 3), (6, 3),
        (-5, -2), (-4, 3), (6, 4), (-5, 2), (5, 4),
    ):
        plant_tree(player_position.x + tree_dx, player_position.y + tree_dy)

    # A couple of ripe berry bushes within reach of the player to start.
    for bush_dx, bush_dy in ((-2, 3), (4, -2)):
        plant_bush(player_position.x + bush_dx, player_position.y + bush_dy)

    # Grow the forest in clustered stands rather than a uniform sprinkle: cover
    # roughly 10% of the walkable land in trees while leaving open clearings
    # between the stands so villagers can still find room to raise cabins. Each
    # stand is a dense core that thins toward its edges, with the odd ripe berry
    # bush mixed in for foragers. Reproducible per world seed.
    rng = world_rng().stream("worldgen_env")
    walkable = [
        (x, y)
        for y in range(game_map.land_y0, game_map.land_y0 + game_map.land_h)
        for x in range(game_map.land_x0, game_map.land_x0 + game_map.land_w)
        if game_map.is_walkable(x, y)
    ]
    tree_target = int(len(walkable) * 0.10)
    planted = 0
    guard = 0
    while walkable and planted < tree_target and guard < tree_target * 50:
        guard += 1
        cx, cy = rng.choice(walkable)
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
                    plant_bush(cx + dx, cy + dy)
                elif plant_tree(cx + dx, cy + dy):
                    planted += 1

    stand_centers = [
        (int(game_map.width * fx), int(game_map.height * fy))
        for fx, fy in ((0.15, 0.25), (0.4, 0.8), (0.7, 0.6), (0.85, 0.75), (0.55, 0.2))
    ]

    # Deer near the tree stands / water so they can graze and drink.
    for cx, cy in stand_centers:
        for dxy in ((-2, 0), (3, 2)):
            _spawn_deer(game_map, cx + dxy[0], cy + dxy[1])
