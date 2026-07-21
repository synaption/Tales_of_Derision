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

    Every creature that eats/drinks/sleeps carries this. The per-turn increase is
    stored here so different creatures could tick at different rates without
    touching the processor. ``tiredness`` climbs like the others but faster at
    night (the multiplier lives in ``systems``) and is reduced by sleeping rather
    than by eating/drinking.
    """
    hunger: float = 0.0
    thirst: float = 0.0
    tiredness: float = 0.0
    hunger_rate: float = 1.0
    thirst_rate: float = 1.4
    tiredness_rate: float = 0.5
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
class Sapling:
    """A young plant seeded on open ground. ``planted_turn`` is the world-clock
    turn it sprouted; after one year (112 days) it matures. ``kind`` decides what
    it becomes -- ``"tree"`` -> ``Tree``, ``"bush"`` -> ``BerryBush``. Saplings
    don't block movement or yield anything yet."""
    planted_turn: int = 0
    kind: str = "tree"


@dataclass
class BerryBush:
    """A mature berry bush. When ripe (``has_berries``) its berries can be
    picked; once taken they regrow 7 days later. ``harvested_turn`` records when
    they were last taken (``None`` while ripe)."""
    has_berries: bool = True
    harvested_turn: int | None = None


@dataclass
class Well:
    """Tag component: a renewable water source the player can drink from."""


@dataclass
class Stove:
    """Tag component: a cooking station that turns Wood + Raw Meat into a
    Cooked Meat."""


@dataclass
class WorldClock:
    """Singleton component tracking the passage of time. ``turn`` advances once
    per real turn; a full day is ``day_length`` turns. ``systems`` derives the
    time-of-day phase (and whether it is night) from ``turn % day_length``."""
    turn: int = 0
    day_length: int = 240


@dataclass
class Home:
    """The tile a character returns to in order to sleep. NPCs prefer their home
    over camping; the player's bed sits on (or beside) theirs."""
    x: int
    y: int


@dataclass
class Bed:
    """Tag component: a place the player interacts with to sleep at home."""


@dataclass
class Camp:
    """Tag component: a temporary campsite a character sets up to sleep away from
    home. Broken (removed) when the sleeper wakes."""


@dataclass
class Asleep:
    """Tag: the character is sleeping. Sleepers skip their turn while their
    tiredness recovers; ``in_camp`` marks a camp to break on waking."""
    in_camp: bool = False


@dataclass
class Chest:
    """Tag component: a storage container. Pairs with ``Inventory`` to hold
    items the player (or an NPC) can loot, like a corpse's inventory."""


@dataclass
class Furniture:
    """Decorative house furnishing (``"table"``, ``"wardrobe"``, ``"bookshelf"``,
    ...). Purely for flavour/occupancy; functional pieces use their own tags
    (``Bed``, ``Stove``, ``Chest``)."""
    kind: str = "furniture"


@dataclass
class Resident:
    """Tag component: an NPC that wants to live in a house -- it will claim an
    unowned house or, failing that, build one from a preset design."""


@dataclass
class BuildPlan:
    """A house a resident NPC is constructing from a preset design. ``remaining``
    is the list of ``(x, y, tile)`` still to place; ``interior`` are the inside
    floor tiles to furnish and ``bed`` the tile it will sleep on once done."""
    remaining: list[tuple[int, int, str]] = field(default_factory=list)
    interior: list[tuple[int, int]] = field(default_factory=list)
    bed: tuple[int, int] = (0, 0)


@dataclass
class Inventory:
    items: list[str] = field(default_factory=list)


@dataclass
class Equipment:
    slots: dict[str, str | None] = field(default_factory=dict)
