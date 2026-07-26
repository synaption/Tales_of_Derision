"""The regional entity index must keep telling the truth.

Every system now trusts ``spatial`` instead of scanning the world, so the index
being wrong would be a silent, world-corrupting bug rather than a slow one. These
tests hold it against a fresh whole-world scan after real simulation: entities
move between regions, are born, and die throughout.
"""
from __future__ import annotations

import esper
import pytest

from components import BlocksMovement, NPC, Position, Sapling, Seaweed, Tree
from game_map import GameMap, archipelago_size
from regions import region_at
from rng import set_world_rng
from systems import (
    FishAiProcessor, HousingProcessor, MovementProcessor, NeedsProcessor,
    NpcAiProcessor, ReproductionProcessor, TimeProcessor, TreeGrowthProcessor,
)
from worldgen import _setup_world
import spatial

pytestmark = pytest.mark.headless_renderer


def _world(grid: int = 2):
    esper.clear_database()
    spatial.detach()
    set_world_rng(0x7A1E5)
    width, height = archipelago_size(grid)
    game_map = GameMap(width, height, layout="islands")
    _setup_world(game_map, Position(width // 2, height // 2))
    esper.add_processor(TimeProcessor(), priority=2)
    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(HousingProcessor(game_map, live_region_only=True), priority=0)
    esper.add_processor(NpcAiProcessor(game_map, max_entry_catchup_advances=1), priority=0)
    esper.add_processor(FishAiProcessor(game_map, max_entry_catchup_advances=1), priority=0)
    esper.add_processor(NeedsProcessor(game_map), priority=0)
    esper.add_processor(TreeGrowthProcessor(game_map), priority=0)
    esper.add_processor(ReproductionProcessor(), priority=0)
    return game_map


def test_index_matches_a_fresh_scan_of_a_new_world() -> None:
    game_map = _world()
    assert spatial.index_for(game_map).audit() == []


def test_index_stays_true_while_the_world_simulates() -> None:
    game_map = _world()
    index = spatial.index_for(game_map)

    moves = ["move_right", "move_right", "move_down", "move_down"]
    for i in range(120):
        esper.process(moves[i % len(moves)])

    assert index.audit() == []


def test_a_walking_entity_changes_region_bucket() -> None:
    game_map = _world()
    index = spatial.index_for(game_map)
    ent = esper.create_entity(Position(5, 5), NPC(), BlocksMovement())
    index.sync()  # creation is noticed by the population check

    assert ent in index.entities_in(region_at(game_map, 5, 5))
    pos = esper.component_for_entity(ent, Position)
    far = (game_map.width - 5, game_map.height - 5)
    spatial.moved(ent, (pos.x, pos.y), far)
    pos.x, pos.y = far

    assert ent not in index.entities_in(region_at(game_map, 5, 5))
    assert ent in index.entities_in(region_at(game_map, *far))
    assert index.audit() == []


def test_static_blockers_are_indexed_by_tile_not_scanned() -> None:
    game_map = _world()
    index = spatial.index_for(game_map)

    trees = [
        (ent, esper.component_for_entity(ent, Position))
        for ent, (_t,) in esper.get_components(Tree)
    ]
    assert trees, "the world should have trees to check"
    for ent, pos in trees[:20]:
        assert index.blocker_at(pos.x, pos.y) == ent
    # Creatures move every turn and are deliberately NOT in the tile map.
    for ent, (pos, _npc) in esper.get_components(Position, NPC):
        assert index.blocker_at(pos.x, pos.y) != ent
        break


def test_plants_are_indexed_by_tile_whether_or_not_they_block() -> None:
    """``rooted`` exists because half the flora doesn't block: seaweed and
    saplings occupy a tile invisibly to ``blockers``, and flora growth asks by
    tile before it seeds one."""
    game_map = _world()
    index = spatial.index_for(game_map)

    for ent, (_s,) in list(esper.get_components(Seaweed))[:20]:
        pos = esper.component_for_entity(ent, Position)
        assert index.rooted_at(pos.x, pos.y) == ent
        assert index.blocker_at(pos.x, pos.y) is None  # a frond blocks nothing

    for ent, (_t,) in list(esper.get_components(Tree))[:20]:
        pos = esper.component_for_entity(ent, Position)
        assert index.rooted_at(pos.x, pos.y) == ent  # trees are in both maps

    for ent, (pos, _npc) in esper.get_components(Position, NPC):
        assert index.rooted_at(pos.x, pos.y) != ent  # people are not plants
        break


def test_a_sapling_maturing_into_a_tree_stays_rooted_to_its_tile() -> None:
    """The reclassify hook has to carry a plant across a kind change: a sapling
    becoming a tree keeps its tile, and the map must not lose it in between."""
    game_map = _world()
    index = spatial.index_for(game_map)
    x, y = 7, 7
    ent = esper.create_entity(Position(x, y), Sapling(planted_turn=0))
    assert index.rooted_at(x, y) == ent

    esper.remove_component(ent, Sapling)
    esper.add_component(ent, Tree())
    esper.add_component(ent, BlocksMovement())
    spatial.reclassify(ent)

    assert index.rooted_at(x, y) == ent
    assert index.blocker_at(x, y) == ent  # and now it blocks, too
    assert index.audit() == []

    esper.delete_entity(ent, immediate=True)
    assert index.rooted_at(x, y) is None
    assert index.audit() == []


def test_entities_created_without_a_hook_are_picked_up() -> None:
    game_map = _world()
    index = spatial.index_for(game_map)
    rebuilds_before = index.rebuilds

    ent = esper.create_entity(Position(7, 7), Tree(), BlocksMovement())

    assert ent in index.entities_in(region_at(game_map, 7, 7))
    assert index.blocker_at(7, 7) == ent
    # Folded in from the entity's own id -- a birth must never cost a world scan,
    # which is what made the index slower than the scans it replaced.
    assert index.rebuilds == rebuilds_before


def test_a_dead_entity_leaves_the_index() -> None:
    game_map = _world()
    index = spatial.index_for(game_map)
    ent = esper.create_entity(Position(9, 9), Tree(), BlocksMovement())
    assert index.blocker_at(9, 9) == ent

    esper.delete_entity(ent, immediate=True)

    assert index.blocker_at(9, 9) is None
    assert index.audit() == []


def test_index_is_not_shared_between_worlds() -> None:
    game_map = _world()
    assert spatial.index_for(game_map) is not None
    other_map = GameMap(60, 30)
    assert spatial.index_for(other_map) is None


def test_audit_reflects_the_index_as_callers_see_it() -> None:
    """``audit`` must sync before comparing, like every other query.

    The index's contract is *repaired on read*, so an entity deleted since the
    last read is legitimately still in the buckets until something asks again.
    An audit that skipped the sync measured a state no caller can observe and
    reported false staleness for any deletion after the last read -- which is
    exactly what dissolving a shoal into a population count does.
    """
    game_map = _world()
    index = spatial.index_for(game_map)
    doomed = [ent for ent, (_t,) in list(esper.get_components(Tree))[:5]]
    assert doomed
    index.sync()

    for ent in doomed:
        esper.delete_entity(ent, immediate=True)
    # No read in between: the buckets still hold them, by design.
    assert index.audit() == [], "audit must fold in the deletions before judging"
    for ent in doomed:
        assert index.blocker_at(*(0, 0)) != ent
