"""Unrendered tests for personalities, friendships, and social interactions:
the trait-driven friendship math, partner preference, the NPC social drive, the
speech-bubble store, and the player's Talk action."""
from __future__ import annotations

import esper
import pytest

from components import (
    Friendly,
    NPC,
    Needs,
    Personality,
    Player,
    Position,
    Relationships,
)
from game_map import GameMap
import systems
from systems import (
    NpcAiProcessor,
    WAIT_ACTION,
    active_bubbles,
    adjust_friendship,
    bubbles_active,
    friendship,
    gibberish,
    interact,
    interaction_delta,
    spawn_speech_bubble,
)
from main import _player_talk

pytestmark = pytest.mark.unrendered


def _make_being(x: int, y: int, traits: list[str]) -> int:
    return esper.create_entity(
        Position(x, y),
        NPC(),
        Friendly(),
        Needs(),
        Personality(traits=list(traits)),
        Relationships(),
    )


# --- Trait-driven friendship math ------------------------------------------


def test_interaction_delta_sign_follows_warmth() -> None:
    # Two cheerful beings warm to each other; two grumps sour.
    assert interaction_delta(["Cheerful"], ["Cheerful"]) > 0
    assert interaction_delta(["Grumpy"], ["Grumpy"]) < 0


def test_adjust_friendship_clamps_and_creates_component() -> None:
    ent = esper.create_entity(Position(1, 1))
    other = esper.create_entity(Position(2, 2))

    assert adjust_friendship(ent, other, 200.0) == 100.0  # clamped high
    assert esper.has_component(ent, Relationships)  # created on demand
    assert adjust_friendship(ent, other, -300.0) == -100.0  # clamped low


# --- Partner preference -----------------------------------------------------


def test_pick_social_partner_prefers_a_friend_over_a_nearer_stranger() -> None:
    game_map = GameMap(24, 14)
    chooser = _make_being(5, 5, ["Outgoing"])
    friend = _make_being(9, 5, ["Kind"])       # distance 4, but already liked
    stranger = _make_being(7, 5, ["Kind"])     # distance 2, unknown

    esper.component_for_entity(chooser, Relationships).scores[friend] = 40.0

    proc = NpcAiProcessor(game_map)
    sentients = [(friend, esper.component_for_entity(friend, Position)),
                 (stranger, esper.component_for_entity(stranger, Position))]
    chosen = proc._pick_social_partner(
        chooser, esper.component_for_entity(chooser, Position), sentients
    )
    assert chosen is not None
    assert chosen[0] == friend


# --- The NPC social drive ---------------------------------------------------


def test_adjacent_villagers_interact_and_pop_bubbles() -> None:
    systems._SPEECH_BUBBLES.clear()
    game_map = GameMap(24, 14)
    a = _make_being(5, 5, ["Cheerful"])
    b = _make_being(6, 5, ["Cheerful"])

    NpcAiProcessor(game_map).process(WAIT_ACTION)

    assert esper.component_for_entity(a, Relationships).scores.get(b, 0.0) > 0
    assert esper.component_for_entity(b, Relationships).scores.get(a, 0.0) > 0
    # One interaction pops a bubble above each participant.
    assert len(active_bubbles()) == 2


def test_social_cooldown_blocks_immediate_repeat() -> None:
    systems._SPEECH_BUBBLES.clear()
    game_map = GameMap(24, 14)
    a = _make_being(5, 5, ["Cheerful"])
    b = _make_being(6, 5, ["Cheerful"])

    proc = NpcAiProcessor(game_map)
    proc.process(WAIT_ACTION)
    score_after_first = esper.component_for_entity(a, Relationships).scores[b]

    proc.process(WAIT_ACTION)  # same turn -> both still on cooldown
    assert esper.component_for_entity(a, Relationships).scores[b] == score_after_first


# --- Speech bubbles ---------------------------------------------------------


def test_bubbles_expire_after_ttl() -> None:
    systems._SPEECH_BUBBLES.clear()
    now = 100.0
    spawn_speech_bubble(1, 1, "bazo!", ttl=2.0, clock=lambda: now)
    assert len(active_bubbles(now=100.5)) == 1
    assert bubbles_active(now=100.5) is True
    assert active_bubbles(now=103.0) == []
    assert bubbles_active(now=103.0) is False


def test_gibberish_is_non_empty() -> None:
    assert gibberish().strip() != ""


def test_bubbles_cap_per_cell_and_keep_newest() -> None:
    systems._SPEECH_BUBBLES.clear()
    now = 500.0
    cap = systems._MAX_BUBBLES_PER_CELL
    for text in (f"say{i}" for i in range(cap + 2)):  # overflow the cell
        spawn_speech_bubble(4, 4, text, clock=lambda: now)

    at_cell = [b for b in active_bubbles(now=now) if (b.x, b.y) == (4, 4)]
    # Oldest scrolled off; exactly the cap remain, and the newest is kept.
    assert len(at_cell) == cap
    assert any(b.text == f"say{cap + 1}" for b in at_cell)
    assert all(b.text != "say0" for b in at_cell)
    # A different cell keeps its own bubble alongside.
    spawn_speech_bubble(9, 9, "elsewhere", clock=lambda: now)
    assert any((b.x, b.y) == (9, 9) for b in active_bubbles(now=now))


def test_social_indicator_scales_with_magnitude() -> None:
    strong_text, strong_color = systems._social_indicator(4.0)
    mild_text, mild_color = systems._social_indicator(1.0)
    sour_text, sour_color = systems._social_indicator(-1.0)
    none_text, none_color = systems._social_indicator(0.2)

    assert (strong_text, strong_color) == ("++", systems._INDICATOR_GREEN)
    assert (mild_text, mild_color) == ("+", systems._INDICATOR_GREEN)
    assert (sour_text, sour_color) == ("-", systems._INDICATOR_RED)
    # A near-zero interaction shows no indicator at all.
    assert (none_text, none_color) == ("", None)


# --- Player Talk ------------------------------------------------------------


def test_player_talk_builds_friendship() -> None:
    systems._SPEECH_BUBBLES.clear()
    player = esper.create_entity(Position(5, 5), Player())
    villager = _make_being(5, 6, ["Kind"])

    outcome = _player_talk(player, villager)

    assert isinstance(outcome, str) and outcome != ""
    rel = esper.component_for_entity(player, Relationships)
    assert friendship(rel, villager) > 0
