from __future__ import annotations

import pytest

from renderer import terminal

pytestmark = pytest.mark.unrendered


def test_inventory_hotkey_maps_to_open_inventory_action() -> None:
    assert terminal._KEY_TO_ACTION[ord("i")] == "open_inventory"
    assert terminal._KEY_TO_ACTION[ord("I")] == "open_inventory"
