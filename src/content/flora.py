"""Core flora prefabs: trees, berry bushes, and sea seaweed. The ecosystem processor
(``TreeGrowthProcessor``) grows/regrows/kills these at runtime; here we just declare
what each is made of."""
from __future__ import annotations

from components import BerryBush, BlocksMovement, Seaweed, Tree
from content.palette import BERRY_RED, OCEAN_WATER_BG, SEAWEED_GREEN, TREE_GREEN
from content.registry import PrefabDef, register_prefab


def register() -> None:
    register_prefab(PrefabDef(
        id="tree", glyph="T", name="Tree", fg=TREE_GREEN,
        components=[Tree, BlocksMovement],
    ))
    register_prefab(PrefabDef(
        id="berry_bush", glyph="%", name="Berry Bush", fg=BERRY_RED,
        components=[BerryBush, BlocksMovement],
    ))
    register_prefab(PrefabDef(
        id="seaweed", glyph='"', name="Seaweed", fg=SEAWEED_GREEN, bg=OCEAN_WATER_BG,
        components=[Seaweed],
    ))


register()
