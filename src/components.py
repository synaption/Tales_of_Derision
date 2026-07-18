"""ECS components. Pure data, no behaviour.

Components stay renderer-agnostic: a Renderable holds a glyph, not a
curses/pygame handle. Systems and renderers translate these into pixels.
"""
from dataclasses import dataclass
from dataclasses import field


@dataclass
class Position:
    x: int
    y: int


@dataclass
class Renderable:
    glyph: str


@dataclass
class Player:
    """Tag component: marks the entity the input controls."""


@dataclass
class Name:
    value: str


@dataclass
class NPC:
    """Tag component: marks an entity controlled by AI."""


@dataclass
class Enemy:
    """Tag component: marks an NPC as hostile to the player."""


@dataclass
class Friendly:
    """Tag component: marks an NPC as non-hostile."""


@dataclass
class BlocksMovement:
    """Tag component: entity occupies and blocks its current tile."""


@dataclass
class Vision:
    radius: int = 8


@dataclass
class Dialogue:
    line: str


@dataclass
class Corpse:
    """Tag component: marks an entity as a dead body."""


@dataclass
class Inventory:
    items: list[str] = field(default_factory=list)


@dataclass
class Equipment:
    slots: dict[str, str | None] = field(default_factory=dict)
