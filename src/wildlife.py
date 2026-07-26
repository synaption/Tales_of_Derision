"""Wild animal populations as **per-region counts**, not as individuals.

Why this exists
---------------
A 1400x800 world holds ~1121 fish. Swimming each one every turn cost ~900k
fish-turns for a single night's sleep -- 74% of the entire world catch-up -- and
almost all of it was discarded, because an unobserved fish's position means
nothing. What *does* mean something is how many fish there are, and whether the
sea can feed them.

So a species lives in one of two forms:

* **Stocked** -- the normal case. A region holds an integer: "310 fish". No
  entities, no positions, no per-turn work at all. The population changes here,
  once a day, against the region's food supply.
* **Resident** -- the exception. A region someone can watch (the player's, or
  one holding an NPC that might hunt) materialises its stock into real entities,
  which the per-turn AI then swims or walks exactly as before.

Crossing between the two is ``materialize`` / ``dissolve``. Individuals are
therefore not persistent: the fish you swam past yesterday is not *the same
fish* today. That is the deliberate trade -- the population is the thing being
modelled, and the population is what persists.

The model
---------
Each day, for every region (see ``advance_day``):

    demand    = count * bites_per_day
    eaten     = min(demand, standing_food * forage_efficiency)
    deaths   ~= starve_chance * (1 - eaten/demand)   -- famine kills
              + predation_chance * predators present
    births   ~= breed_chance, if fed and under the region's density cap

Deaths and births are binomial draws over the population (``rng.binomial``), so
a day costs the animals that were actually born or died rather than the animals
that exist -- and a rate far below one animal a day still works, because five
fish at a 1.5% chance is a 7.2% chance of a calf today rather than a rounding
error.

``forage_efficiency`` is worth calling out. Before this module fish "balanced"
against their food only because a wandering fish usually failed to *find*
seaweed within its sight radius -- the ecology was propped up by a bad search.
Here that inefficiency is an explicit number you can tune.

Batch independence, and one cursor
----------------------------------
``advance_day`` is one day, applied a day at a time. A region a season behind
runs a season of day-steps and lands where it would have landed had it never
fallen behind. Its randomness is drawn from ``rng.region_day_rng(seed, region,
day)`` rather than a shared stream, so a region-day's outcome does not depend on
the order regions are visited or on how many the idle pump managed this frame --
the rule ``regions.RegionScheduler`` documents, held a little tighter than it
demands.

That is still not enough on its own, because this model *reads another day model's
state*: animals eat the flora. Given a day cursor of its own, a long catch-up
could advance the flora a month and then advance the animals a month, growing a
month of seaweed with nothing grazing it and then grazing a month with nothing
growing -- an outcome the interleaved simulation never produces. So there is no
cursor here. ``WildlifeProcessor.register_on`` hands ``advance_day`` to the flora
processor, which owns the one region-day cursor, and plants and animals advance
together whatever pays the debt down.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import esper

from components import Deer, Fish, NPC, Player, Position, Seaweed, Tree
from game_map import GameMap
from regions import RegionId, all_region_ids, region_at, region_bounds
from rng import binomial, region_day_rng, world_rng
import spatial
from content.registry import spawn


@dataclass(frozen=True)
class Species:
    """Everything the stock model needs about one kind of animal.

    Declared as data (see ``FISH`` below) so adding a species is a declaration
    rather than a new code path. ``food`` names the component it eats; a
    region's supply of that is an O(1) index read, and eating deletes some.
    """

    name: str                    # stock key and RNG substream name
    component: type              # the marker component (Fish, Deer, ...)
    prefab: str                  # content prefab id used to materialise one
    habitat: Callable[[GameMap, int, int], bool]   # tiles it may occupy
    food: type | None            # what it eats (Seaweed, Tree, ...)
    food_per_unit: int           # bites one of those yields
    bites_per_day: float         # what one animal must eat per day
    forage_efficiency: float     # share of standing food it can actually find
    starve_chance: float         # daily death chance under total famine
    breed_chance: float          # daily birth chance when fed and under cap
    density_cap: float           # ceiling, in animals per habitat tile
    predators: tuple[type, ...] = ()   # components that hunt it
    predation_chance: float = 0.0      # daily death chance per predator present


def _ocean(game_map: GameMap, x: int, y: int) -> bool:
    return game_map.is_ocean(x, y)


def _land(game_map: GameMap, x: int, y: int) -> bool:
    return game_map.is_walkable(x, y)


# The sea's grazer.
#
# ``bites_per_day = 1`` is the tuned figure. At the old appetite (6 bites a day,
# from hunger climbing 240/day against a 40-point graze) 1121 fish demanded 6726
# bites a day against ~1995 of seaweed regrowth -- a 3.4x deficit that only went
# unnoticed because wandering fish rarely found their food. At one bite a day
# the same sea feeds a shoal comfortably larger than world-gen stocks, so the
# population is shaped by space and breeding rather than pinned against famine
# from the first morning. ``systems._FISH_HUNGER_RATE`` is set to match, so a
# fish the player is actually watching eats on the same schedule.
FISH = Species(
    name="fish",
    component=Fish,
    prefab="fish",
    habitat=_ocean,
    food=Seaweed,
    food_per_unit=3,
    bites_per_day=1.0,
    forage_efficiency=0.35,
    starve_chance=0.08,
    breed_chance=0.015,
    density_cap=0.02,          # at most one fish per 50 open-sea tiles
)

# The land's grazer. Declared so the model covers it, but deer are left resident
# everywhere for now (see ``WildlifeStocks.live_regions``): there are only eight
# in a world, so stocking them saves nothing, and NPC hunting reaches for deer
# *entities* in its own region. Flip them over when deer scale with the land.
DEER = Species(
    name="deer",
    component=Deer,
    prefab="deer",
    habitat=_land,
    food=Tree,
    food_per_unit=4,
    bites_per_day=1.0,
    forage_efficiency=0.5,
    starve_chance=0.05,
    breed_chance=0.01,
    density_cap=0.004,
    predators=(NPC,),
    predation_chance=0.0,      # villagers hunt live deer; no aggregate term yet
)


@dataclass
class Stock:
    """One species' population in one region.

    ``count`` is the whole truth while the region is stocked; while it is
    *resident* the entities are the truth and this is refreshed from them.

    Deliberately just a number. Rates far below one animal a day need no
    accumulator to work: five fish at a 1.5% breeding chance is a 7.2% chance of
    a calf *today*, which ``rng.binomial`` gives directly. Carrying a fractional
    credit between days would only trade that honest randomness for a smoother
    ramp, and add state that has to be saved and restored.
    """

    count: int = 0


class WildlifeStocks:
    """Every species' population in every region, and the day model that moves it.

    One of these belongs to a world (see ``ensure``), like the spatial index.
    """

    def __init__(self, game_map: GameMap, species: tuple[Species, ...] = ()) -> None:
        self.game_map = game_map
        self.species = species or (FISH,)
        self._stocks: dict[tuple[RegionId, str], Stock] = {}
        # Regions whose animals exist as entities right now.
        self.resident: set[RegionId] = set()
        # Habitat tiles per (species, region). Static -- the sea does not move --
        # so built once and kept; materialising picks tiles straight out of it.
        self._habitat: dict[tuple[str, RegionId], list[tuple[int, int]]] = {}
        self._seeds = {s.name: world_rng().int_seed(f"wildlife:{s.name}") for s in self.species}

    # --- stocks -----------------------------------------------------------

    def stock(self, region_id: RegionId, species: Species) -> Stock:
        key = (region_id, species.name)
        found = self._stocks.get(key)
        if found is None:
            found = Stock()
            self._stocks[key] = found
        return found

    def population(self, region_id: RegionId, species: Species) -> int:
        """How many of ``species`` are in ``region_id``, however it is stored --
        the entities if the region is resident, the stock count if not."""
        if region_id in self.resident:
            return len(spatial.ensure(self.game_map).of_kind(region_id, species.component))
        return self.stock(region_id, species).count

    def world_population(self, species: Species) -> int:
        return sum(
            self.population(r, species) for r in all_region_ids(self.game_map)
        )

    def habitat(self, species: Species, region_id: RegionId) -> list[tuple[int, int]]:
        """The tiles of ``region_id`` this species can occupy. Static, so built
        once per (species, region) and kept -- it is also the region's carrying
        capacity, via ``density_cap``."""
        key = (species.name, region_id)
        tiles = self._habitat.get(key)
        if tiles is None:
            x0, y0, x1, y1 = region_bounds(self.game_map, region_id)
            x1 = min(x1, self.game_map.width - 1)
            y1 = min(y1, self.game_map.height - 1)
            tiles = [
                (x, y)
                for y in range(max(1, y0), y1)
                for x in range(max(1, x0), x1)
                if species.habitat(self.game_map, x, y)
            ]
            self._habitat[key] = tiles
        return tiles

    def capacity(self, species: Species, region_id: RegionId) -> int:
        return int(len(self.habitat(species, region_id)) * species.density_cap)

    # --- seeding ----------------------------------------------------------

    def seed_from_world(self) -> None:
        """Adopt the animals world-gen already scattered, then stock down.

        Called once the world exists: every animal entity is counted into its
        region's stock, and the regions that are not resident give their entities
        up. That is what turns a freshly generated world of ~1121 swimming fish
        into ~1121 fish that mostly exist as numbers.
        """
        index = spatial.ensure(self.game_map)
        for species in self.species:
            for region_id in all_region_ids(self.game_map):
                count = len(index.of_kind(region_id, species.component))
                if count:
                    self.stock(region_id, species).count = count
        for region_id in all_region_ids(self.game_map):
            if region_id not in self.resident:
                self._dissolve(region_id)

    # --- residency --------------------------------------------------------

    def live_regions(self) -> set[RegionId]:
        """The regions that must hold real entities: wherever the player is, and
        wherever an NPC is (so hunting, conversation and everything else that
        reaches for a *creature* still finds one). Everything else is a number.

        Walked from the *people* rather than from the regions -- there are a
        couple of dozen of them and a hundred and forty regions, and this runs
        every turn to notice the moment somebody crosses a seam.
        """
        live: set[RegionId] = set()
        for _ent, (pos, _p) in esper.get_components(Position, Player):
            live.add(region_at(self.game_map, pos.x, pos.y))
        for _ent, (pos, _npc) in esper.get_components(Position, NPC):
            live.add(region_at(self.game_map, pos.x, pos.y))
        return live

    def sync_residency(self, live: set[RegionId] | None = None) -> None:
        """Materialise the regions that just became live and dissolve the ones
        that stopped being. Cheap when nothing changed, which is most turns."""
        target = self.live_regions() if live is None else set(live)
        for region_id in target - self.resident:
            self._materialize(region_id)
        for region_id in self.resident - target:
            self._dissolve(region_id)

    def _materialize(self, region_id: RegionId) -> None:
        """Turn a region's stocks into entities, scattered over its habitat."""
        self.resident.add(region_id)
        index = spatial.ensure(self.game_map)
        for species in self.species:
            stock = self.stock(region_id, species)
            existing = len(index.of_kind(region_id, species.component))
            wanted = max(0, stock.count - existing)
            if wanted <= 0:
                continue
            tiles = self.habitat(species, region_id)
            if not tiles:
                continue
            # Placement draws from a shared stream, not a region-day one: *where*
            # a materialised fish surfaces is cosmetic, changes nothing about the
            # population, and happens at moments (walking through a door) that
            # have no day attached. The population maths is the part that has to
            # be order-independent.
            rng = world_rng().stream(f"wildlife:place:{species.name}")
            taken: set[tuple[int, int]] = set()
            placed = 0
            # Rejection-sample tiles: the habitat is far larger than the shoal,
            # so a free tile is found in one or two tries and this stays O(fish)
            # rather than O(sea).
            for _ in range(wanted * 8):
                if placed >= wanted:
                    break
                x, y = tiles[int(rng.random() * len(tiles))]
                if (x, y) in taken or index.blocker_at(x, y) is not None:
                    continue
                if index.rooted_at(x, y) is not None:
                    continue  # don't surface a fish on top of a frond
                spawn(species.prefab, x, y)
                taken.add((x, y))
                placed += 1

    def _dissolve(self, region_id: RegionId) -> None:
        """Count a region's animals back into stocks and delete the entities."""
        self.resident.discard(region_id)
        index = spatial.ensure(self.game_map)
        for species in self.species:
            members = sorted(index.of_kind(region_id, species.component))
            self.stock(region_id, species).count = len(members)
            for ent in members:
                if esper.entity_exists(ent):
                    esper.delete_entity(ent, immediate=True)

    # --- the day model ----------------------------------------------------

    def advance_day(self, region_id: RegionId, day: int) -> None:
        """One day of population change for one region, for every species."""
        for species in self.species:
            self._advance_species_day(region_id, species, day)

    def _advance_species_day(
        self, region_id: RegionId, species: Species, day: int
    ) -> None:
        count = self.population(region_id, species)
        stock = self.stock(region_id, species)
        if count <= 0:
            stock.count = 0
            return
        rng = region_day_rng(self._seeds[species.name], region_id, day).random
        index = spatial.ensure(self.game_map)

        # --- eat ---------------------------------------------------------
        demand = count * species.bites_per_day
        eaten = 0.0
        if species.food is not None and demand > 0:
            standing = len(index.of_kind(region_id, species.food)) * species.food_per_unit
            eaten = min(demand, standing * species.forage_efficiency)
            self._consume(region_id, species, eaten, rng)
        fed = 1.0 if demand <= 0 else eaten / demand

        # --- die ---------------------------------------------------------
        starve_rate = species.starve_chance * (1.0 - fed)
        predator_rate = 0.0
        if species.predation_chance > 0.0 and species.predators:
            hunters = sum(
                len(index.of_kind(region_id, kind)) for kind in species.predators
            )
            predator_rate = min(1.0, species.predation_chance * hunters)
        deaths = min(count, binomial(count, min(1.0, starve_rate + predator_rate), rng))

        # --- breed -------------------------------------------------------
        births = 0
        survivors = count - deaths
        room = self.capacity(species, region_id) - survivors
        if fed >= 1.0 and room > 0:
            births = min(room, binomial(survivors, species.breed_chance, rng))

        self._apply(region_id, species, births - deaths, rng)

    def _consume(
        self, region_id: RegionId, species: Species, bites: float, rng
    ) -> None:
        """Eat ``bites`` worth of the region's food, removing whole units of it.

        Count-based: it deletes ``bites / food_per_unit`` food entities chosen in
        entity order, rather than walking the region's food looking for something
        to nibble. Grazing detail (which frond, how many bites left on it) is a
        thing only a *resident* region's per-turn AI does; out here the sea just
        gets thinner.
        """
        units = int(bites // species.food_per_unit)
        if units <= 0:
            return
        index = spatial.ensure(self.game_map)
        available = sorted(index.of_kind(region_id, species.food))
        if not available:
            return
        start = int(rng() * len(available))  # don't always eat the lowest ids
        for offset in range(min(units, len(available))):
            ent = available[(start + offset) % len(available)]
            if esper.entity_exists(ent):
                esper.delete_entity(ent, immediate=True)

    def _apply(self, region_id: RegionId, species: Species, delta: int, rng) -> None:
        """Move a region's population by ``delta``, in whichever form it is kept."""
        if delta == 0:
            return
        stock = self.stock(region_id, species)
        if region_id not in self.resident:
            stock.count = max(0, stock.count + delta)
            return
        index = spatial.ensure(self.game_map)
        if delta < 0:
            members = sorted(index.of_kind(region_id, species.component))
            for ent in members[: -delta]:
                if esper.entity_exists(ent):
                    esper.delete_entity(ent, immediate=True)
        else:
            tiles = self.habitat(species, region_id)
            for _ in range(delta * 6):
                if delta <= 0:
                    break
                x, y = tiles[int(rng() * len(tiles))] if tiles else (0, 0)
                if not tiles or index.blocker_at(x, y) is not None:
                    continue
                if index.rooted_at(x, y) is not None:
                    continue
                spawn(species.prefab, x, y)
                delta -= 1
        stock.count = len(index.of_kind(region_id, species.component))


class WildlifeProcessor(esper.Processor):
    """Keeps residency honest every turn. Population is not its clock.

    Residency has to run **every turn**: the player crossing a seam must
    materialise the sea they just swam into before it is drawn, so this cannot
    lag. It is cheap -- a walk over the couple of dozen people in the world, not
    over the hundred-odd regions.

    Population moves on a **day**, and deliberately not on a cursor of this
    processor's own. Animals eat the flora, so the two models have to advance in
    lockstep or a region can grow a month of seaweed with nothing grazing it and
    then graze a month with nothing growing. ``register_on`` hands the day step
    to the flora processor, which owns the region-day cursor; from then on a
    region's plants and its animals move together however the debt is paid down
    -- a keypress, the idle pump, or a night's sleep.
    """

    def __init__(self, game_map: GameMap) -> None:
        self.game_map = game_map
        self.stocks = ensure(game_map)

    def register_on(self, flora) -> None:
        """Ride ``flora``'s per-region day cursor. See ``TreeGrowthProcessor.
        register_day_step`` for why this must not be a second cursor."""
        flora.register_day_step("wildlife", self.stocks.advance_day)

    def process(self, action: str | None = None) -> None:
        from systems import _TURN_ACTIONS

        if action not in _TURN_ACTIONS:
            return
        self.stocks.sync_residency()


# --- per-world instances (mirrors spatial.ensure) ---------------------------

_STOCKS: dict[GameMap, WildlifeStocks] = {}


def ensure(game_map: GameMap) -> WildlifeStocks:
    """The stocks for ``game_map``, created on first ask."""
    found = _STOCKS.get(game_map)
    if found is None:
        found = WildlifeStocks(game_map)
        _STOCKS[game_map] = found
    return found


def stocks_for(game_map: GameMap) -> WildlifeStocks | None:
    return _STOCKS.get(game_map)


def detach() -> None:
    """Forget every world's stocks. Tests start clean with this."""
    _STOCKS.clear()
