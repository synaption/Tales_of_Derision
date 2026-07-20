"""Tests for the tabbed player menu (Tab/last-tab, direct-key jump/toggle, and
inventory actions inside a tab). Driven headlessly with a scripted renderer."""
from __future__ import annotations

import asyncio
from collections import deque

import esper
import pytest

from components import BlocksMovement, Equipment, Inventory, Name, Player, Position, Renderable
from main import _draw_player_menu
from game_map import GameMap
from fakes import FakeRenderer

pytestmark = pytest.mark.headless_renderer


class ScriptedRenderer(FakeRenderer):
    """FakeRenderer that replays a queued list of input actions."""

    def __init__(self, actions: list[str | None]):
        super().__init__()
        self._actions: deque[str | None] = deque(actions)

    def poll_action(self) -> str | None:
        return self._actions.popleft() if self._actions else "open_pause_menu"


def _player(items: list[str] | None = None, main_hand: str | None = None) -> int:
    return esper.create_entity(
        Position(5, 5),
        Renderable("@"),
        Name("You"),
        Player(),
        BlocksMovement(),
        Inventory(items=list(items or [])),
        Equipment(slots={"main hand": main_hand}),
    )


def _run(actions: list[str | None], start_tab: str = "inventory") -> tuple[str, str]:
    renderer = ScriptedRenderer(actions)
    return asyncio.run(_draw_player_menu(renderer, GameMap(24, 14), start_tab))


def test_tab_cycles_and_close_reports_the_current_tab() -> None:
    _player()
    # Start on inventory, Tab once -> status, then close.
    result, last_tab = _run(["open_menu", "open_pause_menu"], start_tab="inventory")
    assert result == "close"
    assert last_tab == "status"


def test_direct_key_jumps_to_a_tab_then_toggles_closed() -> None:
    _player()
    # Start on status; C is a no-op jump onto status... use I to jump to
    # inventory, then I again toggles the menu closed.
    result, last_tab = _run(["open_inventory", "open_inventory"], start_tab="status")
    assert result == "close"
    assert last_tab == "inventory"


def test_open_status_jumps_straight_to_status_tab() -> None:
    _player()
    result, last_tab = _run(["open_status", "open_pause_menu"], start_tab="inventory")
    assert result == "close"
    assert last_tab == "status"


def test_inventory_actions_work_inside_the_menu_tab() -> None:
    player = _player(items=["Rusty Sword"], main_hand=None)
    # Move to the items panel, equip the sword, then close.
    _run(["move_right", "menu_select", "open_pause_menu"], start_tab="inventory")
    equipment = esper.component_for_entity(player, Equipment)
    inventory = esper.component_for_entity(player, Inventory)
    assert equipment.slots["main hand"] == "Rusty Sword"
    assert inventory.items == []
