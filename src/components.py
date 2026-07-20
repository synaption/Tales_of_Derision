"""ECS components. Pure data, no behaviour.

Components stay renderer-agnostic: a Renderable holds a glyph, not a
renderer handle. Systems and renderers translate these into pixels.
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
    fg: tuple[int, int, int] | None = None
    bg: tuple[int, int, int] | None = None


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
class Needs:
    """Survival needs. Values rise each turn; 0 is sated, ``max_value`` is dire.

    Only entities that get hungry/thirsty (currently the player) carry this. The
    per-turn increase is stored here so different creatures could tick at
    different rates later without touching the processor.
    """
    hunger: float = 0.0
    thirst: float = 0.0
    hunger_rate: float = 1.0
    thirst_rate: float = 1.4
    max_value: float = 100.0


@dataclass
class OnFire:
    """Tag: the entity is on fire. Drives a status identifier in the render
    animation (a red 'F'); no damage/spread gameplay is wired yet."""


@dataclass
class Deer:
    """Tag component: a wild grazing animal. Prey for carnivores and the player;
    grazes trees and drinks from water when its needs rise."""


@dataclass
class Diet:
    """What an NPC eats when hungry. ``"herbivore"`` grazes trees; ``"carnivore"``
    hunts prey. Drives the needs-based AI in ``systems.NpcAiProcessor``."""
    kind: str = "herbivore"


@dataclass
class Meat:
    """The meat item a creature's corpse yields when butchered, e.g.
    ``Meat("Rat Meat")``. Corpses fall back to a generic ``Raw Meat`` when the
    creature had no ``Meat`` component."""
    name: str = "Raw Meat"


@dataclass
class Tree:
    """A choppable tree. ``wood`` is how many more chops it yields before it
    falls."""
    wood: int = 3


@dataclass
class Well:
    """Tag component: a renewable water source the player can drink from."""


@dataclass
class Stove:
    """Tag component: a cooking station that turns Wood + Raw Meat into a
    Cooked Meat."""


@dataclass
class Inventory:
    items: list[str] = field(default_factory=list)


@dataclass
class Equipment:
    slots: dict[str, str | None] = field(default_factory=dict)
