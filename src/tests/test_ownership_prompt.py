"""Headless tests for the 'that belongs to someone else' confirmation prompt
used when sleeping in a bed or opening a chest that isn't yours."""
from __future__ import annotations

from collections import deque

import esper
import pytest

from components import Bed, Name, Player, Position
from fakes import FakeRenderer
from ui import _confirm, _confirm_if_owned_by_other
from systems import set_bed_owner

pytestmark = pytest.mark.headless_renderer


class ScriptedRenderer(FakeRenderer):
    """FakeRenderer that replays a queued list of input actions."""

    def __init__(self, actions: list[str | None]):
        super().__init__()
        self._actions: deque[str | None] = deque(actions)

    def poll_action(self) -> str | None:
        return self._actions.popleft() if self._actions else "open_pause_menu"


def _run(result):
    # The menu helpers are now plain synchronous calls; kept as a thin wrapper so
    # the call sites below read unchanged.
    return result


def test_confirm_defaults_to_no() -> None:
    # Selecting straight away accepts the default (No).
    assert _run(_confirm(ScriptedRenderer(["menu_select"]), "T", ["line"])) is False


def test_confirm_yes_when_selected() -> None:
    # Move down to Yes, then confirm.
    assert _run(_confirm(ScriptedRenderer(["move_down", "menu_select"]), "T", ["line"])) is True


def test_confirm_escape_cancels() -> None:
    assert _run(_confirm(ScriptedRenderer(["open_pause_menu"]), "T", ["line"])) is False


def test_no_prompt_for_your_own_property() -> None:
    player = esper.create_entity(Position(5, 5), Player(), Name("You"))
    bed = esper.create_entity(Position(3, 3), Bed())
    set_bed_owner(bed, player)
    # No actions needed: it's yours, so it proceeds without asking.
    assert _run(_confirm_if_owned_by_other(ScriptedRenderer([]), bed, "bed", "sleep here")) is True


def test_no_prompt_for_unowned_property() -> None:
    esper.create_entity(Position(5, 5), Player(), Name("You"))
    bed = esper.create_entity(Position(3, 3), Bed())  # nobody owns it
    assert _run(_confirm_if_owned_by_other(ScriptedRenderer([]), bed, "bed", "sleep here")) is True


def test_prompt_when_it_belongs_to_someone_else() -> None:
    esper.create_entity(Position(5, 5), Player(), Name("You"))
    villager = esper.create_entity(Name("Friendly Villager"))
    bed = esper.create_entity(Position(3, 3), Bed())
    set_bed_owner(bed, villager)

    # Declining (default No) refuses.
    assert _run(_confirm_if_owned_by_other(ScriptedRenderer(["menu_select"]), bed, "bed", "sleep here")) is False
    # Explicitly choosing Yes proceeds.
    assert _run(
        _confirm_if_owned_by_other(ScriptedRenderer(["move_down", "menu_select"]), bed, "bed", "sleep here")
    ) is True
