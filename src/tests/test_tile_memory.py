"""Remembered ("fog of war") tiles: terrain and static scenery the player has
seen are redrawn desaturated once they leave line of sight, while NPCs and loot
are hidden until seen again.

These exercise the backend-agnostic path against the FakeRenderer, which has no
cached map surface -- so ``RenderProcessor`` falls back to its per-tile draw and
we can read exactly which glyph/colour landed on each remembered cell.
"""
from __future__ import annotations

import esper
import pytest

from components import (
    BlocksMovement,
    Corpse,
    Enemy,
    Furniture,
    Name,
    NPC,
    Player,
    Position,
    Renderable,
    Tree,
    Vision,
)
from game_map import GameMap
from renderer.base import memory_color
from systems import MovementProcessor, RenderProcessor, is_memorable_scenery
from fakes import FakeRenderer


pytestmark = pytest.mark.headless_renderer


_TREE_GLYPH = "T"
_TREE_FG = (58, 138, 66)


def _setup() -> tuple[GameMap, FakeRenderer, Position, int, int]:
    """An open bordered room with the player near a tree and an NPC, all in view.
    Vision is deliberately short (radius 2) so a couple of steps drop the tree and
    NPC out of sight without needing walls to block line of sight."""
    game_map = GameMap(25, 5)
    renderer = FakeRenderer()

    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(RenderProcessor(renderer, game_map), priority=0)

    player_pos = Position(3, 2)
    esper.create_entity(player_pos, Renderable("@"), Name("You"), Player(), Vision(2), BlocksMovement())
    tree = esper.create_entity(Position(4, 2), Renderable(_TREE_GLYPH, fg=_TREE_FG), Name("Tree"), Tree())
    npc = esper.create_entity(
        Position(5, 2), Renderable("r", fg=(200, 60, 60)), Name("Rat"), NPC(), Enemy(), BlocksMovement()
    )
    return game_map, renderer, player_pos, tree, npc


def test_is_memorable_scenery_whitelists_static_objects_only() -> None:
    tree = esper.create_entity(Position(0, 0), Renderable("T"), Tree())
    table = esper.create_entity(Position(1, 0), Renderable("T"), Furniture("table"))
    rat = esper.create_entity(Position(2, 0), Renderable("r"), NPC(), Enemy())
    corpse = esper.create_entity(Position(3, 0), Renderable("x"), Corpse(), Name("Corpse"))

    assert is_memorable_scenery(tree)
    assert is_memorable_scenery(table)
    assert not is_memorable_scenery(rat)
    assert not is_memorable_scenery(corpse)


def test_tree_is_seen_lit_then_remembered_desaturated_when_out_of_sight() -> None:
    _game_map, renderer, _player_pos, _tree, _npc = _setup()

    # In view: the tree is drawn at its true (lit) colour.
    esper.process(None)
    assert renderer.glyphs[(4, 2)] == _TREE_GLYPH
    assert renderer.glyph_colors[(4, 2)][0] == _TREE_FG

    # Walk out of the tree's line of sight (radius 2): (3,2) -> (2,2) -> (1,2).
    esper.process("move_left")
    esper.process("move_left")

    # The tree tile is no longer visible but is remembered: same glyph, but the
    # colour has been toned down to its desaturated "memory" tone.
    assert (4, 2) not in _render_processor()._visible_tiles
    assert renderer.glyphs[(4, 2)] == _TREE_GLYPH
    remembered_fg = renderer.glyph_colors[(4, 2)][0]
    assert remembered_fg == memory_color(_TREE_FG)
    assert remembered_fg != _TREE_FG


def test_npc_is_not_remembered_when_it_leaves_view() -> None:
    game_map, renderer, _player_pos, _tree, _npc = _setup()

    esper.process(None)
    assert renderer.glyphs[(5, 2)] == "r"  # the NPC is visible to start

    esper.process("move_left")
    esper.process("move_left")

    # The NPC's tile was explored, so its terrain is remembered -- but the NPC
    # itself is gone from the frame, leaving only the (desaturated) floor.
    assert renderer.glyphs[(5, 2)] == game_map.FLOOR


def test_remembered_terrain_is_dimmed_but_present() -> None:
    game_map, renderer, _player_pos, _tree, _npc = _setup()

    esper.process(None)
    esper.process("move_left")
    esper.process("move_left")

    # A remembered plain-floor tile (the tree's neighbour) still draws its floor
    # glyph, tinted to the memory tone rather than left uncoloured (None).
    assert renderer.glyphs[(5, 2)] == game_map.FLOOR
    floor_fg = renderer.glyph_colors[(5, 2)][0]
    assert floor_fg is not None
    assert floor_fg == memory_color(None)


def test_felled_tree_clears_its_memory_while_still_in_view() -> None:
    _game_map, renderer, _player_pos, tree, _npc = _setup()

    esper.process(None)  # see and remember the tree

    # The tree is chopped down while the player still has it in sight.
    esper.delete_entity(tree, immediate=True)
    esper.process(None)  # re-observe the now-empty tile: memory should clear

    esper.process("move_left")
    esper.process("move_left")

    # With the tree gone before the tile left view, nothing scenery-like is
    # recalled there -- just remembered floor.
    proc = _render_processor()
    assert (4, 2) not in proc._tile_memory
    assert renderer.glyphs[(4, 2)] == _game_map.FLOOR


def _render_processor() -> RenderProcessor:
    proc = esper.get_processor(RenderProcessor)
    assert proc is not None, "no RenderProcessor registered"
    return proc
