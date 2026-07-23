"""Tests for the content/mod layer: prefabs, kits, effects, items, and the loader.

All headless and renderer-free -- ``build_components`` returns a plain list to assert
on, and ``spawn``/``apply_effect`` drive the ECS directly.
"""
from __future__ import annotations

from pathlib import Path
import re

import esper
import pytest

from components import (
    Attributes, BlocksMovement, Diet, Enemy, Equipment, Fish, Inventory, Meat,
    NPC, Name, Needs, OnFire, Position, Renderable, Vision,
)
import content.effects as effects_mod
from content.effects import (
    EffectDef, EffectsProcessor, active_effects, apply_effect, effect_display,
    effect_label, register_effect, remove_effect,
)
from content.items import (
    COOKED_MEAT_HUNGER, RAW_MEAT_HUNGER, craft_cost, default_equipment_slots,
    hunger_restored, is_placeable, item_def, placed_tile, thirst_restored,
)
from content.kits import person_kit, predator_kit
from content.loader import _prefab_from_json, load_all_content
from content.registry import (
    PrefabDef, build_components, has_prefab, register_prefab, spawn,
)

pytestmark = pytest.mark.unrendered


@pytest.fixture(autouse=True)
def _content_loaded() -> None:
    load_all_content()


def _types(components: list) -> set[type]:
    return {type(c) for c in components}


# --- Prefabs ----------------------------------------------------------------

def test_build_components_assembles_a_cave_rat() -> None:
    comps = build_components("cave_rat", 5, 7)
    by_type = {type(c): c for c in comps}

    assert by_type[Position] == Position(5, 7)
    assert by_type[Renderable].glyph == "r"
    assert by_type[Name].value == "Cave Rat"
    assert by_type[Meat].name == "Rat Meat"
    assert by_type[Vision].radius == 6
    assert by_type[Attributes].dexterity == 16  # quick vermin
    assert {NPC, Enemy, Diet, BlocksMovement, Needs} <= by_type.keys()
    assert by_type[Diet].kind == "carnivore"


def test_spawn_creates_a_live_entity() -> None:
    ent = spawn("deer", 3, 4)
    assert esper.has_component(ent, Position)
    assert esper.component_for_entity(ent, Diet).kind == "herbivore"
    assert esper.component_for_entity(ent, Meat).name == "Deer Meat"


def test_fish_prefab_is_not_an_npc_and_never_thirsts() -> None:
    comps = {type(c): c for c in build_components("fish", 1, 1)}
    assert Fish in comps
    assert NPC not in comps  # fish run their own AI, not the land brain
    assert comps[Needs].thirst_rate == 0.0


def test_prefab_instances_are_not_shared_between_entities() -> None:
    """A prefab's mutable components must be copied per spawn -- else two goblins
    would share one Inventory and looting one would loot both."""
    a = {type(c): c for c in build_components("goblin_scout", 0, 0)}
    b = {type(c): c for c in build_components("goblin_scout", 1, 1)}
    assert a[Inventory] is not b[Inventory]
    assert a[Equipment].slots is not b[Equipment].slots
    a[Inventory].items.append("Loot")
    assert "Loot" not in b[Inventory].items


def test_unknown_prefab_raises() -> None:
    with pytest.raises(KeyError):
        build_components("nope", 0, 0)


def test_register_factory_prefab_python_path() -> None:
    """The registry also accepts a factory callable (the Python authoring path)."""
    def imp(x: int, y: int, overrides: dict) -> list:
        return [Position(x, y), Renderable("i", fg=(1, 2, 3)), Name("Imp"), NPC(), Enemy()]

    register_prefab(imp, prefab_id="imp")
    ent = spawn("imp", 9, 9)
    assert esper.component_for_entity(ent, Name).value == "Imp"
    assert esper.has_component(ent, Enemy)


# --- Kits -------------------------------------------------------------------

def test_predator_kit_optional_dexterity() -> None:
    assert Attributes not in _types(predator_kit(meat="X Meat", vision=8))
    assert Attributes in _types(predator_kit(meat="X Meat", dexterity=14))


def test_person_kit_is_a_cook_resident_with_a_fresh_equipment_layout() -> None:
    a = {type(c): c for c in person_kit(gender="female", traits=["Kind"], surname="Ash")}
    b = {type(c): c for c in person_kit(gender="male", traits=["Grumpy"], surname="Oak")}
    assert a[Diet].kind == "cook"
    assert a[Equipment].slots is not b[Equipment].slots  # never shared


# --- Effects ----------------------------------------------------------------

class _FakeMap:
    def __init__(self, water: set[tuple[int, int]]):
        self._water = water

    def is_water(self, x: int, y: int) -> bool:
        return (x, y) in self._water


def test_on_fire_is_a_component_effect() -> None:
    ent = esper.create_entity(Position(2, 2), Renderable("@"))
    apply_effect(ent, "on_fire")
    assert esper.has_component(ent, OnFire)
    assert active_effects(_FakeMap(set()), ent, Position(2, 2)) == ["on_fire"]
    remove_effect(ent, "on_fire")
    assert not esper.has_component(ent, OnFire)


def test_swimming_is_a_derived_effect() -> None:
    ent = esper.create_entity(Position(4, 4), Renderable("@"))
    game_map = _FakeMap({(4, 4)})
    assert active_effects(game_map, ent, Position(4, 4)) == ["swimming"]
    assert active_effects(game_map, ent, Position(0, 0)) == []


def test_effects_stack_in_registration_order() -> None:
    ent = esper.create_entity(Position(4, 4), Renderable("@"))
    apply_effect(ent, "on_fire")
    order = active_effects(_FakeMap({(4, 4)}), ent, Position(4, 4))
    assert order == ["swimming", "on_fire"]  # swimming registered before on_fire


def test_effect_display_and_label() -> None:
    glyph, fg, seconds = effect_display("on_fire")
    assert glyph == "F" and fg == (224, 74, 44) and seconds > 0
    assert effect_label("on_fire") == "On fire"


def test_effects_processor_ticks_component_effects() -> None:
    ticked: list[int] = []
    register_effect(EffectDef("_test_burn", glyph="b", seconds=0.1, label="Burn",
                              component=OnFire, on_tick=ticked.append))
    try:
        ent = esper.create_entity(Position(1, 1), OnFire())
        EffectsProcessor().process("move_up")
        assert ticked == [ent]
        ticked.clear()
        EffectsProcessor().process(None)  # menu refresh advances nothing
        assert ticked == []
    finally:
        effects_mod._EFFECTS.pop("_test_burn", None)


# --- Items ------------------------------------------------------------------

def test_item_def_lookups() -> None:
    assert item_def("Bread").eat == 30.0
    assert thirst_restored("Waterskin") == 40.0
    assert is_placeable("Wall")
    assert placed_tile("Door") == "+"
    assert craft_cost("Door") == 3


def test_meat_hunger_by_shape() -> None:
    assert hunger_restored("Cooked Rat Meat") == COOKED_MEAT_HUNGER
    assert hunger_restored("Rat Meat") == RAW_MEAT_HUNGER
    assert hunger_restored("Wood") is None


def test_items_facade_reexports_content_items() -> None:
    import items as items_facade
    from content import items as content_items
    assert items_facade.WOOD == content_items.WOOD
    assert items_facade.hunger_restored is content_items.hunger_restored


def test_default_equipment_slots_are_fresh_each_call() -> None:
    assert default_equipment_slots() is not default_equipment_slots()
    assert set(default_equipment_slots()) == {
        "head", "chest", "hands", "legs", "feet", "main hand", "off hand", "ring",
    }


# --- Loader -----------------------------------------------------------------

def test_loader_registers_core_prefabs() -> None:
    for prefab_id in ("cave_rat", "goblin_scout", "deer", "fish", "tree", "berry_bush", "well", "stove", "bed"):
        assert has_prefab(prefab_id)


def test_json_prefab_composes_from_kits_and_spawns() -> None:
    """The JSON (data) authoring path builds a PrefabDef that spawns like any other."""
    defn = _prefab_from_json({
        "id": "dire_rat", "glyph": "r", "name": "Dire Rat", "fg": [150, 60, 60],
        "kits": [["predator", {"meat": "Dire Rat Meat", "vision": 7}]],
    })
    register_prefab(defn)
    ent = spawn("dire_rat", 2, 2)
    assert esper.component_for_entity(ent, Meat).name == "Dire Rat Meat"
    assert esper.component_for_entity(ent, Renderable).fg == (150, 60, 60)


# --- Determinism gate over the content (simulation) modules -----------------

_IMPORT_RANDOM = re.compile(r"^\s*(?:import random\b|from random import)", re.MULTILINE)
_BARE_RANDOM_CALL = re.compile(r"(?:^|[^.\w])random\.[A-Za-z_]")


@pytest.mark.parametrize("module_name", [
    "content.registry", "content.kits", "content.effects", "content.items",
    "content.creatures", "content.flora", "content.features", "content.loader",
    "content.palette",
])
def test_content_modules_do_not_use_global_random(module_name: str) -> None:
    """Content is simulation code now; any randomness must flow through rng.world_rng."""
    import importlib
    source = Path(importlib.import_module(module_name).__file__).read_text(encoding="utf-8")
    assert not _IMPORT_RANDOM.search(source), f"{module_name} imports global random"
    assert not _BARE_RANDOM_CALL.search(source), f"{module_name} calls global random"
