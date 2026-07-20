from __future__ import annotations

from types import SimpleNamespace

import pytest

from renderer.pygame_renderer import _build_key_mappings

pytestmark = pytest.mark.unrendered


def _fake_pygame_keys() -> SimpleNamespace:
    return SimpleNamespace(
        K_w=1,
        K_s=2,
        K_a=3,
        K_d=4,
        K_q=14,
        K_i=5,
        K_ESCAPE=6,
        K_RETURN=7,
        K_KP_ENTER=8,
        K_SPACE=9,
        K_UP=10,
        K_DOWN=11,
        K_LEFT=12,
        K_RIGHT=13,
        K_EQUALS=15,
        K_KP_PLUS=16,
        K_MINUS=17,
        K_KP_MINUS=18,
    )


def test_inventory_hotkey_maps_to_open_inventory_action() -> None:
    pygame_keys = _fake_pygame_keys()
    keydown_map, _keyup_map = _build_key_mappings(pygame_keys)
    assert keydown_map[pygame_keys.K_i] == "open_inventory"


def test_space_maps_to_confirm_action() -> None:
    pygame_keys = _fake_pygame_keys()
    keydown_map, _keyup_map = _build_key_mappings(pygame_keys)
    assert keydown_map[pygame_keys.K_SPACE] == "confirm_action"


def test_arrow_keys_are_not_movement_inputs() -> None:
    pygame_keys = _fake_pygame_keys()
    keydown_map, _keyup_map = _build_key_mappings(pygame_keys)
    assert pygame_keys.K_UP not in keydown_map
    assert pygame_keys.K_DOWN not in keydown_map
    assert pygame_keys.K_LEFT not in keydown_map
    assert pygame_keys.K_RIGHT not in keydown_map


def test_wasd_keyup_maps_to_release_actions() -> None:
    pygame_keys = _fake_pygame_keys()
    _keydown_map, keyup_map = _build_key_mappings(pygame_keys)
    assert keyup_map[pygame_keys.K_w] == "release_up"
    assert keyup_map[pygame_keys.K_s] == "release_down"
    assert keyup_map[pygame_keys.K_a] == "release_left"
    assert keyup_map[pygame_keys.K_d] == "release_right"


def test_tile_scale_actions_have_default_keybinds() -> None:
    pygame_keys = _fake_pygame_keys()
    keydown_map, _keyup_map = _build_key_mappings(pygame_keys)
    assert keydown_map[pygame_keys.K_EQUALS] == "tile_scale_up"
    assert keydown_map[pygame_keys.K_KP_PLUS] == "tile_scale_up"
    assert keydown_map[pygame_keys.K_MINUS] == "tile_scale_down"
    assert keydown_map[pygame_keys.K_KP_MINUS] == "tile_scale_down"


def test_custom_keybinds_override_defaults() -> None:
    pygame_keys = _fake_pygame_keys()
    keydown_map, _keyup_map = _build_key_mappings(
        pygame_keys,
        options={
            "keybinds": {
                "move_up": ["up"],
                "confirm_action": ["enter"],
            }
        },
    )
    assert pygame_keys.K_w not in keydown_map
    assert keydown_map[pygame_keys.K_UP] == "move_up"
    assert keydown_map[pygame_keys.K_RETURN] == "confirm_action"


def test_legacy_direction_aliases_are_supported() -> None:
    pygame_keys = _fake_pygame_keys()
    keydown_map, keyup_map = _build_key_mappings(
        pygame_keys,
        options={
            "keybinds": {
                "up": ["up"],
            }
        },
    )
    assert keydown_map[pygame_keys.K_UP] == "move_up"
    assert keyup_map[pygame_keys.K_UP] == "release_up"
