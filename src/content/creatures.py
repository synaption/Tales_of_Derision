"""Core creature prefabs. Each is a base glyph + a kit + a couple of extras -- add a
new creature by copying one of these (data path) or registering a factory (Python
path). Villagers and the player are assembled in ``worldgen`` because they need the
clock (age) and per-name colour; they use ``content.kits.person_kit`` for the bundle.
"""
from __future__ import annotations

from components import Equipment, Inventory
from content.items import default_equipment_slots
from content.palette import DEER_TAN, FISH_SILVER, GOBLIN_GREEN, OCEAN_WATER_BG, RAT_BROWN
from content.registry import PrefabDef, register_prefab


def register() -> None:
    register_prefab(PrefabDef(
        id="cave_rat", glyph="r", name="Cave Rat", fg=RAT_BROWN,
        kits=[("predator", dict(meat="Rat Meat", vision=6, dexterity=16))],
    ))
    register_prefab(PrefabDef(
        id="goblin_scout", glyph="g", name="Goblin Scout", fg=GOBLIN_GREEN,
        kits=[("predator", dict(meat="Goblin Meat", vision=8))],
        components=[
            Inventory(items=["Copper Coin", "Bone Charm"]),
            Equipment(slots={**default_equipment_slots(), "main hand": "Jagged Dagger"}),
        ],
    ))
    register_prefab(PrefabDef(
        id="deer", glyph="d", name="Deer", fg=DEER_TAN,
        kits=[("grazer", dict(meat="Deer Meat", vision=8, hunger=30.0, thirst=30.0))],
    ))
    register_prefab(PrefabDef(
        id="fish", glyph="f", name="Fish", fg=FISH_SILVER, bg=OCEAN_WATER_BG,
        kits=[("fish", dict(hunger=25.0))],
    ))


register()
