"""Tests for the 3x3 section camera and the swimming glyph bob. These use the
FakeRenderer test double -- no live window -- but exercise RenderProcessor."""
from __future__ import annotations

import esper
import pytest

from components import BlocksMovement, Name, OnFire, Player, Position, Renderable, Vision
from game_map import GameMap
from systems import MovementProcessor, RenderProcessor
from fakes import FakeRenderer

pytestmark = pytest.mark.headless_renderer


def _world(player_x: int, player_y: int, width: int = 120, height: int = 60):
    game_map = GameMap(width, height)
    renderer = FakeRenderer()
    esper.add_processor(MovementProcessor(game_map), priority=1)
    render_processor = RenderProcessor(renderer, game_map)
    esper.add_processor(render_processor, priority=0)
    player_pos = Position(player_x, player_y)
    esper.create_entity(
        player_pos, Renderable("@"), Name("You"), Player(), Vision(10), BlocksMovement()
    )
    return game_map, renderer, render_processor, player_pos


def test_section_camera_locks_view_to_the_players_section() -> None:
    _gm, _r, rp, _p = _world(60, 30)
    esper.process(None)
    # A 120x60 map splits into 40x20 sections; (60,30) is the middle one.
    assert rp._section_bounds == (40, 20, 40, 20)
    assert rp._current_section == (1, 1)


def test_only_the_current_section_is_visible() -> None:
    _gm, _r, rp, _p = _world(60, 30)
    esper.process(None)
    sx, sy, sw, sh = rp._section_bounds
    # Nothing outside the section rectangle is ever marked visible/drawn.
    assert rp._visible_tiles
    assert all(sx <= x < sx + sw and sy <= y < sy + sh for x, y in rp._visible_tiles)


def test_crossing_a_section_edge_snaps_camera_and_logs_transition() -> None:
    _gm, _r, rp, player_pos = _world(79, 30)
    esper.process(None)
    assert rp._current_section == (1, 1)

    esper.process("move_right")  # 79 -> 80 crosses into the right column
    assert player_pos.x == 80
    assert rp._current_section == (2, 1)
    assert rp._section_bounds == (80, 20, 40, 20)
    assert "You cross into a new area." in rp._message_log


def test_sections_disabled_on_small_maps() -> None:
    # Tiny maps (the render tests' maps) keep the plain centred camera.
    _gm, _r, rp, _p = _world(5, 3, width=10, height=6)
    esper.process(None)
    assert rp._sections_enabled is False
    assert rp._section_bounds is None


def test_swimming_status_shows_own_tile_then_water_on_a_timed_cycle() -> None:
    # Cycle: own tile 1.0s, then "~" 0.5s -> total 1.5s.
    game_map, _r, rp, _p = _world(60, 30)
    swimmer_ent = esper.create_entity(Position(61, 30), Renderable("@"))
    swimmer = esper.component_for_entity(swimmer_ent, Position)
    game_map.tiles[30][61] = GameMap.WATER

    rp._clock = lambda: 0.5   # within the 1.0s base window
    assert rp._status_appearance(swimmer_ent, swimmer, "@", None) == ("@", None, False)
    rp._clock = lambda: 1.2   # within the 0.5s swim window: literal-glyph frame
    assert rp._status_appearance(swimmer_ent, swimmer, "@", None) == ("~", None, True)
    rp._clock = lambda: 1.6   # wrapped back into the base window
    assert rp._status_appearance(swimmer_ent, swimmer, "@", None) == ("@", None, False)

    # On dry land there is no status and no animation.
    on_land_ent = esper.create_entity(Position(60, 30), Renderable("@"))
    on_land = esper.component_for_entity(on_land_ent, Position)
    rp._clock = lambda: 1.2
    assert rp._status_appearance(on_land_ent, on_land, "@", None) == ("@", None, False)


def test_statuses_stack_sequentially_swim_then_fire() -> None:
    # Swimming AND on fire: own tile 1.0s -> "~" 0.5s -> red "F" 0.5s (total 2.0s).
    game_map, _r, rp, _p = _world(60, 30)
    ent = esper.create_entity(Position(61, 30), Renderable("@"), OnFire())
    pos = esper.component_for_entity(ent, Position)
    game_map.tiles[30][61] = GameMap.WATER

    rp._clock = lambda: 0.4
    assert rp._status_appearance(ent, pos, "@", (10, 10, 10)) == ("@", (10, 10, 10), False)
    rp._clock = lambda: 1.2
    assert rp._status_appearance(ent, pos, "@", (10, 10, 10)) == ("~", (10, 10, 10), True)
    rp._clock = lambda: 1.7
    glyph, color, is_status = rp._status_appearance(ent, pos, "@", (10, 10, 10))
    assert glyph == "F" and color == (224, 74, 44) and is_status is True


def test_status_identifier_is_drawn_as_a_literal_glyph_not_the_class_sprite() -> None:
    # Regression: a status glyph ("~") must be force-drawn, not resolved back to
    # the friendly/enemy classification sprite (which would hide it).
    game_map, renderer, rp, player_pos = _world(60, 30)
    game_map.tiles[30][60] = GameMap.WATER  # player starts on water

    rp._clock = lambda: 0.3  # base frame: normal sprite draw
    esper.process(None)
    vx, vy = rp._world_to_view(player_pos.x, player_pos.y)
    assert renderer.glyphs[(vx, vy)] == "@"
    assert renderer.forced_glyphs[(vx, vy)] is False

    rp._clock = lambda: 1.2  # swim frame: literal glyph
    esper.process(None)
    vx, vy = rp._world_to_view(player_pos.x, player_pos.y)
    assert renderer.glyphs[(vx, vy)] == "~"
    assert renderer.forced_glyphs[(vx, vy)] is True
