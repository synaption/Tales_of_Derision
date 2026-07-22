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
class Gender:
    """A person's gender. ``"male"`` or ``"female"`` for now; the onymancer and
    (future) reproduction read it. Left as a free string so a future monogender
    race can carry a single value of its own."""
    value: str


@dataclass
class Family:
    """A person's place in a family. ``surname`` is shared by the whole household;
    ``spouse`` and the ``parents``/``children`` lists hold other people's entity
    ids. Kept reciprocal by whoever wires the family (spouses point at each other,
    parents and children point back). Siblings are derived from shared parents, so
    they aren't stored. The future reproduction system fills these in for newborns.
    """
    surname: str
    spouse: int | None = None
    parents: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)


@dataclass
class Age:
    """A person's age, stored as the world-clock ``turn`` they were born on. Age
    in years is derived from the current clock (see ``systems.age_years``), so
    nobody has to be ticked each turn -- the same trick flora maturity uses.
    Members of the starting cast are given a ``born_turn`` in the past (often
    negative) so they begin as adults or children of a chosen age."""
    born_turn: int


@dataclass
class Pregnant:
    """Carried by a woman from conception until the baby is born. ``conceived_turn``
    is the world-clock turn sex resulted in pregnancy; ``father`` is the other
    parent's entity id. ``systems.ReproductionProcessor`` delivers the child once
    the gestation period has elapsed and then removes this component."""
    conceived_turn: int
    father: int


@dataclass
class Mating:
    """Bookkeeping for reproduction cooldowns: ``last_turn`` is the world-clock
    turn this person last had sex. Men may only mate once per day (the day is
    derived from the turn); women are unbounded, so the rule is applied per gender
    in ``systems.try_mate`` rather than baked in here."""
    last_turn: int = -10_000


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
class Fish:
    """Tag component: a fish that roams the open sea. It swims on water only (its
    own AI in ``systems.FishAiProcessor``, never the land pathfinder) and grazes
    seaweed when hungry, the aquatic mirror of a deer grazing trees."""


@dataclass
class Seaweed:
    """An underwater plant rooted on an open-sea tile. Fish graze it like deer
    graze trees; ``food`` is how many bites remain before it is eaten bare and
    removed. Fresh fronds sprout again over time (``TreeGrowthProcessor``)."""
    food: int = 3


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
    """Singleton component tracking the passage of time in **time units (TU)**.
    ``turn`` advances by an action's *cost* each turn (a baseline action is
    ``action.BASE_ACTION_COST`` TU), not by 1, so faster/slower actions consume
    proportionally less/more world time. A full day is ``day_length`` TU;
    ``systems`` derives the time-of-day phase (and whether it is night) from
    ``turn % day_length``. The default is 240 baseline actions per day
    (240 * BASE_ACTION_COST), preserving the old one-action-per-turn day length."""
    turn: int = 0
    day_length: int = 24000


@dataclass
class Home:
    """The tile a character returns to in order to sleep. NPCs prefer their home
    over camping; the player's bed sits on (or beside) theirs."""
    x: int
    y: int


@dataclass
class Bed:
    """Tag component: a place the player interacts with to sleep at home. A bed
    stands in for the house around it -- whoever owns the bed owns the house."""


@dataclass
class Owned:
    """Marks a house (via its ``Bed``) as belonging to a person. ``owner`` is the
    entity id of the owner. An unowned house has no ``Owned`` on its bed; a house
    whose owner no longer exists counts as unowned again."""
    owner: int


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
class Blueprint:
    """A single ghost tile of a structure under construction: a blue-tinted
    preview of the ``tile`` it will eventually become. ``stocked`` flips True
    once the wood for this piece has been hauled in (its tint brightens); any
    worker then raises stocked ghosts into real tiles one at a time.

    ``site`` is the ``ConstructionSite`` entity this piece belongs to, or
    ``None`` for a loose piece a player staked out on its own. A ghost is a plain
    world entity, so **anybody** -- a villager or the player -- can walk up and
    work it."""
    tile: str = "#"
    stocked: bool = False
    site: int | None = None


@dataclass
class ConstructionSite:
    """A staked-out structure being raised from blue-tinted ``Blueprint`` ghosts.

    It is a **world** entity, owned by no one: the whole footprint is first laid
    out as ghost tiles, workers haul wood to it (lighting each ghost up as its
    materials arrive), and finally raise the stocked ghosts into real tiles one
    chunk at a time. Any build-minded villager -- or the player -- can pitch in;
    when the last piece goes up the cabin is furnished and left unowned for the
    nearest homeless resident to claim.

    ``pieces`` maps each ``(x, y)`` build coordinate -> its ``Blueprint`` ghost
    entity; ``interior`` are the inside floor tiles furnished on completion and
    ``bed`` the tile the eventual resident sleeps on."""
    pieces: dict[tuple[int, int], int] = field(default_factory=dict)
    interior: list[tuple[int, int]] = field(default_factory=list)
    bed: tuple[int, int] = (0, 0)


@dataclass
class Personality:
    """A sentient being's named personality traits (e.g. ``"Cheerful"``,
    ``"Grumpy"``, ``"Shy"``), drawn from ``systems._TRAITS``. Traits drive how
    sociable the being is and whether a given interaction warms or sours a
    relationship.

    ``last_social_turn`` is bookkeeping for the social cooldown: the world-clock
    turn this being last interacted with someone, so it doesn't chatter every
    turn. It lives here (rather than a module global) so it clears with the ECS
    database between games/tests.
    """
    traits: list[str] = field(default_factory=list)
    last_social_turn: int = -10_000


@dataclass
class Relationships:
    """Sims-like friendship scores toward other beings. ``scores`` maps another
    entity id -> friendship in ``[-100, 100]``; 0 is a stranger, positive a
    friend, negative a rival. Missing entries read as 0. Repeated interactions
    nudge the score up or down gradually (see ``systems.interact``).

    ``pending`` accumulates the not-yet-shown reaction toward each other being --
    friendship keeps changing every exchange, but the ``+``/``-`` indicator only
    surfaces once this builds to a milestone (or occasionally at random), so most
    chat bubbles carry no indicator. It resets to 0 whenever an indicator shows.
    """
    scores: dict[int, float] = field(default_factory=dict)
    pending: dict[int, float] = field(default_factory=dict)


@dataclass
class Inventory:
    items: list[str] = field(default_factory=list)


@dataclass
class Equipment:
    slots: dict[str, str | None] = field(default_factory=dict)


@dataclass
class Attributes:
    """Morrowind-style core attributes (see the readme's leveling design). For now
    only ``dexterity`` is read -- it drives how quickly an actor takes its turns in
    the action economy (see ``action.actor_speed``) -- but the full set is here so
    skills/leveling and future derived stats have a home. 10 is the human average.
    """
    strength: int = 10
    dexterity: int = 10
    constitution: int = 10
    intelligence: int = 10
    wisdom: int = 10
    charisma: int = 10


@dataclass
class Actor:
    """Scheduling bookkeeping for the time-based action economy. ``next_time`` is
    the world-clock time unit (TU) at which this entity next gets to act; the
    scheduler always runs whoever has the smallest ``next_time``. ``last_acted`` is
    the TU it last acted, so time-based effects (needs) can accrue the exact
    elapsed span when it acts again. Every creature that takes turns carries one.

    ``energy`` is the region-simulation counterpart: NPCs are driven per
    region-turn rather than by the global ``next_time`` queue, so each region-turn
    grants ``BASE_ACTION_COST`` energy and the NPC spends its action cost per
    action -- a quicker creature banks the surplus and acts again."""
    next_time: int = 0
    last_acted: int = 0
    energy: float = 0.0
