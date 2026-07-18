from __future__ import annotations

import pytest

from renderer import terminal

pytestmark = pytest.mark.unrendered


def test_inventory_hotkey_maps_to_open_inventory_action() -> None:
    assert terminal._KEY_TO_ACTION[ord("i")] == "open_inventory"
    assert terminal._KEY_TO_ACTION[ord("I")] == "open_inventory"


def test_space_maps_to_confirm_action() -> None:
    assert terminal._KEY_TO_ACTION[ord(" ")] == "confirm_action"


def test_hjkl_and_arrows_are_not_movement_inputs() -> None:
    assert ord("h") not in terminal._KEY_TO_ACTION
    assert ord("j") not in terminal._KEY_TO_ACTION
    assert ord("k") not in terminal._KEY_TO_ACTION
    assert ord("l") not in terminal._KEY_TO_ACTION

    assert terminal.curses.KEY_UP not in terminal._KEY_TO_ACTION
    assert terminal.curses.KEY_DOWN not in terminal._KEY_TO_ACTION
    assert terminal.curses.KEY_LEFT not in terminal._KEY_TO_ACTION
    assert terminal.curses.KEY_RIGHT not in terminal._KEY_TO_ACTION
