"""Small cross-cutting ECS queries shared across modules (worldgen, UI, the turn
loop). Kept in a neutral, dependency-light module so both ``main`` and ``worldgen``
can use them without importing each other."""
from __future__ import annotations

import esper

from components import Player, Position


def first_player_entity() -> int | None:
    """The player entity's id, or ``None`` if the world has no player yet."""
    for ent, (_pos, _player) in esper.get_components(Position, Player):
        return ent
    return None
