from __future__ import annotations

import esper
import pytest

from components import BlocksMovement, Corpse, Dialogue, Enemy, Equipment, Friendly, Inventory, NPC, Name, Player, Position, Renderable, Vision
from game_map import GameMap
from systems import MovementProcessor, NpcAiProcessor, RenderProcessor
from fakes import FakeRenderer


pytestmark = pytest.mark.headless_renderer


def _setup_world(width: int = 10, height: int = 6) -> tuple[GameMap, FakeRenderer, Position]:
    game_map = GameMap(width, height)
    renderer = FakeRenderer()

    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(NpcAiProcessor(game_map), priority=0)
    esper.add_processor(RenderProcessor(renderer, game_map), priority=0)

    player_pos = Position(width // 2, height // 2)
    esper.create_entity(
        player_pos,
        Renderable("@"),
        Name("You"),
        Player(),
        Vision(10),
        BlocksMovement(),
        Inventory(items=["Bandage"]),
        Equipment(slots={"main hand": "Rusty Sword"}),
    )
    esper.create_entity(
        Position(player_pos.x + 1, player_pos.y),
        Renderable("g"),
        Name("Goblin"),
        NPC(),
        Enemy(),
        BlocksMovement(),
        Vision(8),
        Inventory(items=["Copper Coin"]),
        Equipment(slots={"main hand": "Jagged Dagger"}),
    )

    return game_map, renderer, player_pos


def test_initial_tick_renders_map_player_and_status_line() -> None:
    game_map, renderer, player_pos = _setup_world()

    esper.process(None)

    assert renderer.present_calls == 1
    assert renderer.glyphs[(0, 0)] == game_map.WALL
    assert renderer.glyphs[(1, 1)] == game_map.FLOOR
    assert renderer.glyphs[(player_pos.x, player_pos.y)] == "@"
    assert (0, game_map.height, "move: arrows/hjkl/wasd   inventory: i   menu: esc") in renderer.text
    sidebar_x = game_map.width + 2
    assert (sidebar_x, 0, "== NEARBY ==") in renderer.text
    assert any(text.startswith("g → Goblin") for _x, _y, text in renderer.text)
    assert any(text == "== LOG ==" for _x, _y, text in renderer.text)


def test_nearby_section_is_dynamic_and_log_starts_after_items() -> None:
    game_map = GameMap(30, 14)
    renderer = FakeRenderer()

    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(RenderProcessor(renderer, game_map), priority=0)

    player_pos = Position(15, 7)
    esper.create_entity(player_pos, Renderable("@"), Name("You"), Player())
    esper.create_entity(Position(16, 7), Renderable("g"), Name("Goblin"))
    esper.create_entity(Position(14, 7), Renderable("o"), Name("Orc"))
    esper.create_entity(Position(15, 8), Renderable("r"), Name("Rat"))

    esper.process(None)

    sidebar_x = game_map.width + 2
    nearby_header_y = next(y for x, y, text in renderer.text if x == sidebar_x and text == "== NEARBY ==")
    log_header_y = next(y for x, y, text in renderer.text if x == sidebar_x and text == "== LOG ==")
    nearby_rows = [
        y
        for x, y, text in renderer.text
        if x == sidebar_x and nearby_header_y < y < log_header_y and text and "==" not in text
    ]

    assert len(nearby_rows) >= 3
    assert log_header_y == max(nearby_rows) + 2


def test_move_action_updates_position_and_rendered_player_location() -> None:
    _game_map, renderer, player_pos = _setup_world()

    start = (player_pos.x, player_pos.y)
    esper.process("move_right")

    assert (player_pos.x, player_pos.y) == (start[0] + 1, start[1])
    assert renderer.glyphs[(player_pos.x, player_pos.y)] == "@"


def test_blocked_movement_adds_log_event() -> None:
    _game_map, renderer, player_pos = _setup_world(width=8, height=20)

    player_pos.x = 1
    esper.process(None)
    esper.process("move_left")

    assert any(text == "You bump into a wall." for _x, _y, text in renderer.text)


def test_player_does_not_move_through_wall() -> None:
    _game_map, _renderer, player_pos = _setup_world(width=8, height=6)

    player_pos.x = 1
    start = (player_pos.x, player_pos.y)

    esper.process("move_left")

    assert (player_pos.x, player_pos.y) == start


def test_visible_npc_sighting_logs_once_even_when_not_adjacent() -> None:
    game_map = GameMap(12, 20)
    renderer = FakeRenderer()

    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(RenderProcessor(renderer, game_map), priority=0)

    player_pos = Position(5, 10)
    esper.create_entity(player_pos, Renderable("@"), Name("You"), Player(), Vision(10), BlocksMovement())
    esper.create_entity(Position(8, 10), Renderable("r"), Name("Rat"), NPC(), Enemy(), BlocksMovement())

    esper.process(None)
    esper.process("move_left")
    esper.process("move_right")

    notice_count = sum(1 for _x, _y, text in renderer.text if text == "You notice Rat to the east.")
    assert notice_count == 1


def test_npc_moves_toward_player_when_in_sight() -> None:
    game_map = GameMap(16, 10)
    renderer = FakeRenderer()

    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(NpcAiProcessor(game_map), priority=0)
    esper.add_processor(RenderProcessor(renderer, game_map), priority=0)

    player_pos = Position(10, 5)
    npc_pos = Position(4, 5)

    esper.create_entity(player_pos, Renderable("@"), Name("You"), Player(), Vision(10), BlocksMovement())
    esper.create_entity(npc_pos, Renderable("g"), Name("Goblin"), NPC(), Enemy(), Vision(10), BlocksMovement())

    start = (npc_pos.x, npc_pos.y)
    esper.process("move_right")

    assert (npc_pos.x, npc_pos.y) != start
    assert npc_pos.x > start[0]


def test_player_attacks_enemy_on_collision() -> None:
    game_map = GameMap(12, 8)
    renderer = FakeRenderer()

    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(NpcAiProcessor(game_map), priority=0)
    esper.add_processor(RenderProcessor(renderer, game_map), priority=0)

    player_pos = Position(5, 4)
    enemy_pos = Position(6, 4)
    enemy_ent = esper.create_entity(
        enemy_pos,
        Renderable("g"),
        Name("Goblin"),
        NPC(),
        Enemy(),
        BlocksMovement(),
        Vision(8),
    )
    esper.create_entity(player_pos, Renderable("@"), Name("You"), Player(), Vision(10), BlocksMovement())

    esper.process("move_right")

    corpses = [
        (pos, name)
        for _ent, (pos, _corpse, name) in esper.get_components(Position, Corpse, Name)
    ]

    assert (player_pos.x, player_pos.y) == (6, 4)
    assert not esper.entity_exists(enemy_ent)
    assert len(corpses) == 1
    assert (corpses[0][0].x, corpses[0][0].y) == (6, 4)
    assert corpses[0][1].value == "Corpse of Goblin"
    assert any(text == "You attack Goblin." for _x, _y, text in renderer.text)


def test_player_bumps_friendly_and_gets_dialogue() -> None:
    game_map = GameMap(12, 8)
    renderer = FakeRenderer()

    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(NpcAiProcessor(game_map), priority=0)
    esper.add_processor(RenderProcessor(renderer, game_map), priority=0)

    player_pos = Position(5, 4)
    esper.create_entity(player_pos, Renderable("@"), Name("You"), Player(), Vision(10), BlocksMovement())
    esper.create_entity(
        Position(6, 4),
        Renderable("v"),
        Name("Friendly Villager"),
        NPC(),
        Friendly(),
        Dialogue("##!/$*~# GH01^@"),
        BlocksMovement(),
    )

    esper.process(None)
    esper.process("move_right")

    assert (player_pos.x, player_pos.y) == (5, 4)
    assert any(
        text.startswith('Friendly Villager says: "##!')
        for _x, _y, text in renderer.text
    )
    assert not any(text == "You bump into a wall." for _x, _y, text in renderer.text)


def test_all_characters_have_inventory_and_equipment_components() -> None:
    _game_map, _renderer, _player_pos = _setup_world(width=12, height=8)

    actor_ents = {
        ent
        for ent, (_pos, _name) in esper.get_components(Position, Name)
        if esper.has_component(ent, Player) or esper.has_component(ent, NPC)
    }

    with_inventory = {ent for ent, (_inv,) in esper.get_components(Inventory)}
    with_equipment = {ent for ent, (_equip,) in esper.get_components(Equipment)}

    assert actor_ents
    assert actor_ents.issubset(with_inventory)
    assert actor_ents.issubset(with_equipment)
