"""Core world-feature prefabs: the survival stations placed near the player and in
furnished houses. Beds get an owner wired at the call site (see ``worldgen`` /
``systems.furnish_house``); the prefab just makes the object."""
from __future__ import annotations

from components import Bed, BlocksMovement, Stove, Well
from content.palette import BED_WOOD, STOVE_IRON, WELL_STONE
from content.registry import PrefabDef, register_prefab


def register() -> None:
    register_prefab(PrefabDef(
        id="well", glyph="O", name="Stone Well", fg=WELL_STONE,
        components=[Well, BlocksMovement],
    ))
    register_prefab(PrefabDef(
        id="stove", glyph="#", name="Iron Stove", fg=STOVE_IRON,
        components=[Stove, BlocksMovement],
    ))
    register_prefab(PrefabDef(
        id="bed", glyph="=", name="Bed", fg=BED_WOOD,
        components=[Bed],
    ))


register()
