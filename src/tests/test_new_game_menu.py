"""Tests for the new-game world-size screen: the grid the player picks decides how
many islands the world holds. Driven headlessly with a scripted renderer."""
from __future__ import annotations

from collections import deque

import pytest

from config import DEFAULT_WORLD_GRID, WORLD_GRID_CHOICES
from game_map import archipelago_size
from ui import _draw_new_game_menu, new_game_map
from fakes import FakeRenderer

pytestmark = pytest.mark.headless_renderer


class ScriptedRenderer(FakeRenderer):
    """FakeRenderer that replays a queued list of input actions."""

    def __init__(self, actions: list[str | None]):
        super().__init__()
        self._actions: deque[str | None] = deque(actions)

    def poll_action(self) -> str | None:
        return self._actions.popleft() if self._actions else "open_pause_menu"


def test_enter_takes_the_default_world_size() -> None:
    assert _draw_new_game_menu(ScriptedRenderer(["menu_select"])) == DEFAULT_WORLD_GRID


def test_moving_the_selection_picks_a_different_grid() -> None:
    default_index = list(WORLD_GRID_CHOICES).index(DEFAULT_WORLD_GRID)
    expected = WORLD_GRID_CHOICES[default_index - 1]
    assert _draw_new_game_menu(ScriptedRenderer(["move_up", "confirm_action"])) == expected


def test_every_choice_is_reachable_by_walking_the_list() -> None:
    default_index = list(WORLD_GRID_CHOICES).index(DEFAULT_WORLD_GRID)
    for index, grid in enumerate(WORLD_GRID_CHOICES):
        steps = ["move_up"] * (default_index - index) + ["move_down"] * (index - default_index)
        assert _draw_new_game_menu(ScriptedRenderer([*steps, "menu_select"])) == grid


def test_back_entry_and_escape_both_return_to_the_main_menu() -> None:
    # "Back" sits after the grid choices, so one step up from the top choice.
    steps_to_back = ["move_up"] * (list(WORLD_GRID_CHOICES).index(DEFAULT_WORLD_GRID) + 1)
    assert _draw_new_game_menu(ScriptedRenderer([*steps_to_back, "menu_select"])) == "back"
    assert _draw_new_game_menu(ScriptedRenderer(["open_pause_menu"])) == "back"


def test_closing_the_window_quits_instead_of_going_back() -> None:
    assert _draw_new_game_menu(ScriptedRenderer(["quit"])) == "quit"


@pytest.mark.parametrize("grid", [1, 2, 3])
def test_new_game_map_builds_the_chosen_number_of_islands(grid: int) -> None:
    game_map, player_position = new_game_map(grid)

    assert (game_map.width, game_map.height) == archipelago_size(grid)
    assert len(game_map.islands) == grid * grid
    assert game_map.in_bounds(player_position.x, player_position.y)
