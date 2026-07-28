from __future__ import annotations

import argparse
import bisect
import random
import shlex
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple


TURNS_PER_DAY = 1200
DAYS_PER_WEEK = 7
MONTHS_PER_YEAR = 4
WEEKS_PER_MONTH = 1
DAYS_PER_MONTH = DAYS_PER_WEEK * WEEKS_PER_MONTH
DAYS_PER_YEAR = DAYS_PER_MONTH * MONTHS_PER_YEAR
TURNS_PER_MONTH = TURNS_PER_DAY * DAYS_PER_MONTH
TURNS_PER_YEAR = TURNS_PER_DAY * DAYS_PER_YEAR

AVERAGE_LIFESPAN_YEARS = 80
ADULTHOOD_YEARS = 17
FEMALE_FERTILITY_END_YEARS = 45
MALE_FERTILITY_END_YEARS = 65
SEX_ACTS_PER_YEAR = 30
PREGNANCY_CHANCE_PER_ACT = 0.01

NAME_PREFIXES = (
	"Ae",
	"Al",
	"Ar",
	"Bel",
	"Cor",
	"Da",
	"Eld",
	"Fen",
	"Gal",
	"Ith",
	"Kae",
	"Lor",
	"Mor",
	"Nim",
	"Or",
	"Per",
	"Qua",
	"Ryn",
	"Syl",
	"Tor",
	"Ul",
	"Va",
	"Wyn",
	"Xan",
	"Ysa",
	"Zor",
)
NAME_MIDDLES = (
	"a",
	"ae",
	"e",
	"ei",
	"i",
	"ia",
	"o",
	"oa",
	"u",
	"y",
	"ar",
	"en",
	"ir",
	"or",
	"un",
)
NAME_SUFFIXES = (
	"dell",
	"drin",
	"fal",
	"gorn",
	"hild",
	"ion",
	"kas",
	"lith",
	"mir",
	"nor",
	"rith",
	"selle",
	"thas",
	"vash",
	"wyn",
)

SEXES = ("F", "M")

# Box-drawing glyphs for tree rendering. Swapped for pure ASCII by --ascii-tree
# so legacy Windows consoles (cp1252) do not blow up on encode.
TREE_GLYPHS = {
	"branch": "├─ ",
	"last": "└─ ",
	"pipe": "│  ",
	"blank": "   ",
	"kept": "✓",
	"erased": "✗",
}
ASCII_TREE_GLYPHS = {
	"branch": "|- ",
	"last": "`- ",
	"pipe": "|  ",
	"blank": "   ",
	"kept": "+",
	"erased": "x",
}
ACTIVE_GLYPHS = dict(TREE_GLYPHS)

TIME_UNIT_TO_TURNS = {
	"turn": 1,
	"turns": 1,
	"day": TURNS_PER_DAY,
	"days": TURNS_PER_DAY,
	"month": TURNS_PER_MONTH,
	"months": TURNS_PER_MONTH,
	"year": TURNS_PER_YEAR,
	"years": TURNS_PER_YEAR,
}


def stable_hash(*parts: object) -> int:
	# 64-bit FNV-1a style hash: deterministic and much faster than cryptographic hashing.
	value = 1469598103934665603
	prime = 1099511628211
	mask = (1 << 64) - 1
	for part in parts:
		payload = str(part).encode("utf-8")
		for byte in payload:
			value ^= byte
			value = (value * prime) & mask
		value ^= 0xFF
		value = (value * prime) & mask
	return value


def stable_float(*parts: object) -> float:
	return stable_hash(*parts) / float(1 << 64)


def stable_rng(*parts: object) -> random.Random:
	return random.Random(stable_hash(*parts))


def clamp_turn(target_turn: int) -> int:
	return max(0, target_turn)


def format_turn(turn: int) -> str:
	year = turn // TURNS_PER_YEAR
	year_turn = turn % TURNS_PER_YEAR
	month = year_turn // TURNS_PER_MONTH
	month_turn = year_turn % TURNS_PER_MONTH
	day = month_turn // TURNS_PER_DAY
	day_turn = month_turn % TURNS_PER_DAY
	return f"Y{year} M{month + 1} D{day + 1} T{day_turn}"


def turns_from_amount(amount: int, unit: str) -> int:
	if unit not in TIME_UNIT_TO_TURNS:
		raise ValueError(f"unknown time unit: {unit}")
	return amount * TIME_UNIT_TO_TURNS[unit]


def fantasy_name(*parts: object) -> str:
	rng = stable_rng("name", *parts)
	prefix = rng.choice(NAME_PREFIXES)
	middle = rng.choice(NAME_MIDDLES)
	suffix = rng.choice(NAME_SUFFIXES)
	return f"{prefix}{middle}{suffix}"


@dataclass
class Person:
	person_id: int
	name: str
	sex: str
	birth_turn: int
	natural_death_turn: int
	mother_id: Optional[int] = None
	father_id: Optional[int] = None


@dataclass
class Snapshot:
	current_turn: int
	people: Dict[int, Person]
	partner_map: Dict[int, int]
	pair_birth_counts: Dict[Tuple[int, int], int]
	next_person_id: int


@dataclass
class SimulationStats:
	current_turn: int
	living: int
	adults: int
	children: int
	males: int
	females: int
	couples: int
	births: int
	deaths: int
	interventions: int
	oldest_age_years: int
	family_count: int


class Simulation:
	def __init__(
		self,
		seed: str,
		founder_count: int = 24,
		checkpoint_interval_years: int = 10,
		carrying_capacity: int = 5000,
	) -> None:
		self.seed = seed
		self.founder_count = founder_count
		self.checkpoint_interval_years = max(1, checkpoint_interval_years)
		self.carrying_capacity = max(1, carrying_capacity)
		self.kill_interventions: Dict[int, int] = {}
		self.snapshots: Dict[int, Snapshot] = {}
		self.current_turn = 0
		self.people: Dict[int, Person] = {}
		self.partner_map: Dict[int, int] = {}
		self.pair_birth_counts: Dict[Tuple[int, int], int] = {}
		self.next_person_id = 1
		self._bootstrap()

	def _birth_chance_for_year(self, living_count: int) -> float:
		if living_count <= self.carrying_capacity:
			return PREGNANCY_CHANCE_PER_ACT
		pressure = self.carrying_capacity / living_count
		return PREGNANCY_CHANCE_PER_ACT * pressure

	def _bootstrap(self) -> None:
		self.current_turn = 0
		self.people = {}
		self.partner_map = {}
		self.pair_birth_counts = {}
		self.next_person_id = 1
		for _ in range(self.founder_count):
			self._create_founder()
		self.snapshots = {0: self._snapshot()}

	def _snapshot(self) -> Snapshot:
		return Snapshot(
			current_turn=self.current_turn,
			# Person records are immutable after creation, so a shallow dict copy is safe.
			people=self.people.copy(),
			partner_map=self.partner_map.copy(),
			pair_birth_counts=self.pair_birth_counts.copy(),
			next_person_id=self.next_person_id,
		)

	def _restore_snapshot(self, year: int) -> None:
		snapshot = self.snapshots[year]
		self.current_turn = snapshot.current_turn
		self.people = snapshot.people.copy()
		self.partner_map = snapshot.partner_map.copy()
		self.pair_birth_counts = snapshot.pair_birth_counts.copy()
		self.next_person_id = snapshot.next_person_id

	def _create_founder(self) -> None:
		person_id = self.next_person_id
		self.next_person_id += 1
		sex = SEXES[stable_hash(self.seed, "founder-sex", person_id) % len(SEXES)]
		age_years = 17 + (stable_hash(self.seed, "founder-age", person_id) % 19)
		age_offset_turns = age_years * TURNS_PER_YEAR
		birth_turn = -age_offset_turns
		person = Person(
			person_id=person_id,
			name=fantasy_name(self.seed, "founder", person_id),
			sex=sex,
			birth_turn=birth_turn,
			natural_death_turn=birth_turn + (AVERAGE_LIFESPAN_YEARS * TURNS_PER_YEAR),
		)
		self.people[person_id] = person

	def _effective_death_turn(self, person_id: int) -> int:
		person = self.people[person_id]
		kill_turn = self.kill_interventions.get(person_id)
		if kill_turn is None:
			return person.natural_death_turn
		return min(person.natural_death_turn, kill_turn)

	def _is_alive_at(self, person_id: int, turn: int) -> bool:
		person = self.people[person_id]
		return person.birth_turn <= turn < self._effective_death_turn(person_id)

	def _age_years_at(self, person_id: int, turn: int) -> int:
		person = self.people[person_id]
		return max(0, (turn - person.birth_turn) // TURNS_PER_YEAR)

	def _is_adult_at(self, person_id: int, turn: int) -> bool:
		person = self.people[person_id]
		adult_turn = person.birth_turn + (ADULTHOOD_YEARS * TURNS_PER_YEAR)
		return adult_turn <= turn < self._effective_death_turn(person_id)

	def _is_fertile_at(self, person_id: int, turn: int) -> bool:
		person = self.people[person_id]
		fertile_start_turn = person.birth_turn + (ADULTHOOD_YEARS * TURNS_PER_YEAR)
		if person.sex == "F":
			fertile_end_turn = person.birth_turn + ((FEMALE_FERTILITY_END_YEARS + 1) * TURNS_PER_YEAR)
		else:
			fertile_end_turn = person.birth_turn + ((MALE_FERTILITY_END_YEARS + 1) * TURNS_PER_YEAR)
		fertile_end_turn = min(fertile_end_turn, self._effective_death_turn(person_id))
		return fertile_start_turn <= turn < fertile_end_turn

	def _pair_key(self, female_id: int, male_id: int) -> Tuple[int, int]:
		return (female_id, male_id)

	def _normalize_parents(self, person_a: int, person_b: int) -> Tuple[int, int]:
		first = self.people[person_a]
		second = self.people[person_b]
		if first.sex == second.sex:
			raise ValueError("birth pairs must contain one female and one male")
		if first.sex == "F":
			return self._pair_key(person_a, person_b)
		return self._pair_key(person_b, person_a)

	def _compatibility_score(self, first_id: int, second_id: int) -> int:
		low, high = sorted((first_id, second_id))
		return stable_hash(self.seed, "pair-compatibility", low, high)

	def _clear_invalid_partnerships(self, turn: int) -> None:
		for person_id, partner_id in list(self.partner_map.items()):
			if person_id >= partner_id:
				continue
			person = self.people[person_id]
			partner = self.people[partner_id]
			person_adult_turn = person.birth_turn + (ADULTHOOD_YEARS * TURNS_PER_YEAR)
			partner_adult_turn = partner.birth_turn + (ADULTHOOD_YEARS * TURNS_PER_YEAR)
			if not (
				person_adult_turn <= turn < self._effective_death_turn(person_id)
				and partner_adult_turn <= turn < self._effective_death_turn(partner_id)
			):
				self.partner_map.pop(person_id, None)
				self.partner_map.pop(partner_id, None)

	def _refresh_partnerships(self, turn: int) -> None:
		self._clear_invalid_partnerships(turn)
		unpaired_females = [
			person_id
			for person_id, person in self.people.items()
			if (
				person.sex == "F"
				and (person.birth_turn + (ADULTHOOD_YEARS * TURNS_PER_YEAR)) <= turn < self._effective_death_turn(person_id)
				and person_id not in self.partner_map
			)
		]
		unpaired_males = [
			person_id
			for person_id, person in self.people.items()
			if (
				person.sex == "M"
				and (person.birth_turn + (ADULTHOOD_YEARS * TURNS_PER_YEAR)) <= turn < self._effective_death_turn(person_id)
				and person_id not in self.partner_map
			)
		]
		year = turn // TURNS_PER_YEAR
		unpaired_females.sort(key=lambda person_id: (stable_hash(self.seed, "pair-order", year, "F", person_id), person_id))
		unpaired_males.sort(key=lambda person_id: (stable_hash(self.seed, "pair-order", year, "M", person_id), person_id))
		for female_id, male_id in zip(unpaired_females, unpaired_males):
			self.partner_map[female_id] = male_id
			self.partner_map[male_id] = female_id

	def _current_couples(self) -> List[Tuple[int, int]]:
		couples = []
		for person_id, partner_id in self.partner_map.items():
			if person_id < partner_id:
				if self.people[person_id].sex == "F":
					couples.append((person_id, partner_id))
				else:
					couples.append((partner_id, person_id))
		couples.sort()
		return couples

	def _create_child(self, mother_id: int, father_id: int, birth_turn: int) -> None:
		pair_key = self._pair_key(mother_id, father_id)
		birth_index = self.pair_birth_counts.get(pair_key, 0) + 1
		self.pair_birth_counts[pair_key] = birth_index
		person_id = self.next_person_id
		self.next_person_id += 1
		sex = SEXES[stable_hash(self.seed, "child-sex", mother_id, father_id, birth_index) % len(SEXES)]
		person = Person(
			person_id=person_id,
			name=fantasy_name(self.seed, "child", mother_id, father_id, birth_index),
			sex=sex,
			birth_turn=birth_turn,
			natural_death_turn=birth_turn + (AVERAGE_LIFESPAN_YEARS * TURNS_PER_YEAR),
			mother_id=mother_id,
			father_id=father_id,
		)
		self.people[person_id] = person

	def _sample_birth_turns_for_pair_year(
		self,
		mother_id: int,
		father_id: int,
		year: int,
		pregnancy_chance: float,
	) -> List[int]:
		base_turn = year * TURNS_PER_YEAR
		chance_bucket = int(round(pregnancy_chance * 1_000_000_000))
		if chance_bucket <= 0:
			return []
		cdf = self._binomial_cdf_30(pregnancy_chance)
		roll = stable_float(
			self.seed,
			"pair-year-roll",
			mother_id,
			father_id,
			year,
			chance_bucket,
		)
		birth_count = bisect.bisect_left(cdf, roll)
		birth_turns = [
			base_turn + (stable_hash(self.seed, "pair-year-turn", mother_id, father_id, year, idx) % TURNS_PER_YEAR)
			for idx in range(birth_count)
		]
		birth_turns.sort()
		return birth_turns

	def _binomial_cdf_30(self, pregnancy_chance: float) -> List[float]:
		# CDF for Binomial(n=30, p) built once per year-level p and reused across pairs.
		p = min(max(pregnancy_chance, 0.0), 1.0)
		if p <= 0.0:
			return [0.0] * SEX_ACTS_PER_YEAR + [1.0]
		if p >= 1.0:
			return [0.0] * SEX_ACTS_PER_YEAR + [1.0]
		q = 1.0 - p
		pmf = q ** SEX_ACTS_PER_YEAR
		cdf = [pmf]
		ratio = p / q
		for k in range(SEX_ACTS_PER_YEAR):
			pmf = pmf * (SEX_ACTS_PER_YEAR - k) / (k + 1) * ratio
			cdf.append(cdf[-1] + pmf)
		cdf[-1] = 1.0
		return cdf

	def _acts_for_pair_year(self, mother_id: int, father_id: int, year: int) -> List[Tuple[int, int]]:
		events = []
		base_turn = year * TURNS_PER_YEAR
		for act_index in range(SEX_ACTS_PER_YEAR):
			offset = stable_hash(self.seed, "act-turn", mother_id, father_id, year, act_index) % TURNS_PER_YEAR
			act_turn = base_turn + offset
			events.append((act_turn, act_index))
		events.sort()
		return events

	def _process_year_segment(self, year: int, segment_start: int, segment_end: int) -> None:
		year_start = year * TURNS_PER_YEAR
		if segment_start == year_start:
			self._refresh_partnerships(segment_start)
		living_count = sum(1 for person_id in self.people if self._is_alive_at(person_id, segment_start))
		pregnancy_chance = self._birth_chance_for_year(living_count)
		for mother_id, father_id in self._current_couples():
			birth_turns = self._sample_birth_turns_for_pair_year(
				mother_id,
				father_id,
				year,
				pregnancy_chance,
			)
			for birth_turn in birth_turns:
				if birth_turn < segment_start or birth_turn >= segment_end:
					continue
				if not self._is_fertile_at(mother_id, birth_turn) or not self._is_fertile_at(father_id, birth_turn):
					continue
				self._create_child(mother_id, father_id, birth_turn)

	def _replay_from(self, start_turn: int, target_turn: int) -> None:
		if target_turn <= start_turn:
			self.current_turn = target_turn
			return
		start_year = start_turn // TURNS_PER_YEAR
		end_year = target_turn // TURNS_PER_YEAR
		for year in range(start_year, end_year + 1):
			year_start = year * TURNS_PER_YEAR
			segment_start = max(start_turn, year_start)
			segment_end = min(target_turn, (year + 1) * TURNS_PER_YEAR)
			if segment_start >= segment_end:
				continue
			self._process_year_segment(year, segment_start, segment_end)
			if segment_end == (year + 1) * TURNS_PER_YEAR:
				completed_year = year + 1
				self.current_turn = segment_end
				if completed_year % self.checkpoint_interval_years == 0:
					self.snapshots[completed_year] = self._snapshot()
		self.current_turn = target_turn

	def goto_turn(self, target_turn: int) -> None:
		target_turn = clamp_turn(target_turn)
		target_year = target_turn // TURNS_PER_YEAR
		available_years = [year for year in self.snapshots if year <= target_year]
		checkpoint_year = max(available_years) if available_years else 0
		self._restore_snapshot(checkpoint_year)
		self._replay_from(self.current_turn, target_turn)

	def advance(self, amount: int, unit: str) -> None:
		self.goto_turn(self.current_turn + turns_from_amount(amount, unit))

	def rewind(self, amount: int, unit: str) -> None:
		self.goto_turn(self.current_turn - turns_from_amount(amount, unit))

	def kill(self, person_id: int, when_turn: Optional[int] = None) -> str:
		if person_id not in self.people:
			raise ValueError(f"unknown person id: {person_id}")
		kill_turn = self.current_turn if when_turn is None else clamp_turn(when_turn)
		person = self.people[person_id]
		if kill_turn < person.birth_turn:
			raise ValueError("cannot kill a person before they are born")
		previous = self.kill_interventions.get(person_id)
		if previous is None or kill_turn < previous:
			self.kill_interventions[person_id] = kill_turn
		self.snapshots = {0: self.snapshots[0]}
		target_turn = self.current_turn
		self.goto_turn(target_turn)
		return f"{person.name} now dies at {format_turn(self.kill_interventions[person_id])}."

	def undo_last_kill(self) -> str:
		if not self.kill_interventions:
			return "No interventions to undo."
		person_id, _ = max(self.kill_interventions.items(), key=lambda item: item[1])
		removed_turn = self.kill_interventions.pop(person_id)
		self.snapshots = {0: self.snapshots[0]}
		target_turn = self.current_turn
		self.goto_turn(target_turn)
		name = self.people[person_id].name if person_id in self.people else f"Person {person_id}"
		return f"Removed intervention for {name} at {format_turn(removed_turn)}."

	def fork(self, extra_kill: Optional[Tuple[int, int]] = None) -> "Simulation":
		# A what-if copy: same seed and interventions, optionally plus one more kill.
		# Founders are bootstrapped identically regardless of interventions, so the
		# clone only diverges from the point the extra kill lands.
		clone = Simulation(
			seed=self.seed,
			founder_count=self.founder_count,
			checkpoint_interval_years=self.checkpoint_interval_years,
			carrying_capacity=self.carrying_capacity,
		)
		clone.kill_interventions = dict(self.kill_interventions)
		if extra_kill is not None:
			person_id, kill_turn = extra_kill
			previous = clone.kill_interventions.get(person_id)
			if previous is None or kill_turn < previous:
				clone.kill_interventions[person_id] = kill_turn
		clone.goto_turn(self.current_turn)
		return clone

	def living_people(self) -> List[Person]:
		return [person for person in self.people.values() if self._is_alive_at(person.person_id, self.current_turn)]

	def stats(self) -> SimulationStats:
		living_people = self.living_people()
		adults = [person for person in living_people if self._is_adult_at(person.person_id, self.current_turn)]
		males = [person for person in living_people if person.sex == "M"]
		females = [person for person in living_people if person.sex == "F"]
		families = {
			self._normalize_parents(person.mother_id, person.father_id)
			for person in self.people.values()
			if person.birth_turn <= self.current_turn and person.mother_id is not None and person.father_id is not None
		}
		oldest_age_years = 0
		if living_people:
			oldest_age_years = max(self._age_years_at(person.person_id, self.current_turn) for person in living_people)
		births = sum(1 for person in self.people.values() if person.birth_turn <= self.current_turn)
		deaths = sum(1 for person in self.people if self._effective_death_turn(person) <= self.current_turn)
		return SimulationStats(
			current_turn=self.current_turn,
			living=len(living_people),
			adults=len(adults),
			children=len(living_people) - len(adults),
			males=len(males),
			females=len(females),
			couples=len(self._current_couples()),
			births=births,
			deaths=deaths,
			interventions=len(self.kill_interventions),
			oldest_age_years=oldest_age_years,
			family_count=len(families),
		)

	def describe_person(self, person_id: int) -> str:
		if person_id not in self.people:
			raise ValueError(f"unknown person id: {person_id}")
		person = self.people[person_id]
		age_years = self._age_years_at(person_id, self.current_turn)
		status = "alive" if self._is_alive_at(person_id, self.current_turn) else "dead"
		partner_id = self.partner_map.get(person_id)
		partner_text = "none"
		if partner_id is not None and self._is_alive_at(partner_id, self.current_turn):
			partner = self.people[partner_id]
			partner_text = f"{partner.name}#{partner.person_id}"
		mother_text = "unknown" if person.mother_id is None else f"{self.people[person.mother_id].name}#{person.mother_id}"
		father_text = "unknown" if person.father_id is None else f"{self.people[person.father_id].name}#{person.father_id}"
		return (
			f"{person.name}#{person.person_id} sex={person.sex} age={age_years} status={status} "
			f"born={format_turn(person.birth_turn)} dies={format_turn(self._effective_death_turn(person_id))} "
			f"partner={partner_text} mother={mother_text} father={father_text}"
		)

	def list_people(self, limit: int = 20, living_only: bool = True) -> List[str]:
		people = self.living_people() if living_only else [
			person for person in self.people.values() if person.birth_turn <= self.current_turn
		]
		people.sort(key=lambda person: (person.birth_turn, person.person_id))
		return [self.describe_person(person.person_id) for person in people[:limit]]

	def descendants_count(self, person_id: int) -> int:
		children_by_parent: Dict[int, List[int]] = {}
		for person in self.people.values():
			if person.birth_turn > self.current_turn:
				continue
			for parent_id in (person.mother_id, person.father_id):
				if parent_id is None:
					continue
				children_by_parent.setdefault(parent_id, []).append(person.person_id)
		seen = set()
		stack = list(children_by_parent.get(person_id, []))
		while stack:
			current = stack.pop()
			if current in seen:
				continue
			seen.add(current)
			stack.extend(children_by_parent.get(current, []))
		return len(seen)

	def top_bloodlines(self, limit: int = 5) -> List[str]:
		founders = [
			person for person in self.people.values() if person.birth_turn <= self.current_turn and person.mother_id is None
		]
		founders.sort(key=lambda person: (-self.descendants_count(person.person_id), person.person_id))
		lines = []
		for founder in founders[:limit]:
			lines.append(
				f"{founder.name}#{founder.person_id} descendants={self.descendants_count(founder.person_id)}"
			)
		return lines


@dataclass
class TimelineIndex:
	turn: int
	people: Dict[int, Person]
	lineage_key: Dict[int, int]
	by_lineage: Dict[int, int]
	children: Dict[int, List[int]]


def build_timeline_index(people: Dict[int, Person], turn: int) -> TimelineIndex:
	# Person ids are allocated during replay, so an intervention renumbers everyone
	# born after it. Lineage keys identify a person by their position in the family
	# tree instead, which survives renumbering and lets two timelines be compared.
	born = {person_id: person for person_id, person in people.items() if person.birth_turn <= turn}
	ordered_ids = sorted(born)

	pair_children: Dict[Tuple[int, int], List[int]] = {}
	children: Dict[int, List[int]] = {}
	for person_id in ordered_ids:
		person = born[person_id]
		if person.mother_id is None or person.father_id is None:
			continue
		pair_children.setdefault((person.mother_id, person.father_id), []).append(person_id)
		children.setdefault(person.mother_id, []).append(person_id)
		children.setdefault(person.father_id, []).append(person_id)

	birth_index: Dict[int, int] = {}
	for sibling_ids in pair_children.values():
		# Within one pair, ids increase with birth order, so position is the birth index.
		for offset, person_id in enumerate(sibling_ids):
			birth_index[person_id] = offset + 1

	lineage_key: Dict[int, int] = {}
	for person_id in ordered_ids:
		person = born[person_id]
		if person.mother_id is None or person.father_id is None:
			lineage_key[person_id] = stable_hash("lineage-founder", person_id)
			continue
		# Parents are always created before their children, so their keys exist already.
		lineage_key[person_id] = stable_hash(
			"lineage-child",
			lineage_key[person.mother_id],
			lineage_key[person.father_id],
			birth_index[person_id],
		)

	by_lineage = {key: person_id for person_id, key in lineage_key.items()}
	return TimelineIndex(
		turn=turn,
		people=born,
		lineage_key=lineage_key,
		by_lineage=by_lineage,
		children=children,
	)


def _birth_year(person: Person) -> int:
	return person.birth_turn // TURNS_PER_YEAR


def descendant_tree_lines(
	index: TimelineIndex,
	root_id: int,
	label: Callable[[int], str],
	max_generations: int,
	max_children: int,
	glyphs: Dict[str, str],
) -> List[str]:
	lines = [label(root_id)]

	def walk(person_id: int, prefix: str, depth: int) -> None:
		child_ids = index.children.get(person_id, [])
		if not child_ids:
			return
		if depth >= max_generations:
			lines.append(prefix + glyphs["last"] + f"... {len(child_ids)} more below this depth")
			return
		shown = child_ids[:max_children]
		hidden = len(child_ids) - len(shown)
		for offset, child_id in enumerate(shown):
			is_last = (offset == len(shown) - 1) and hidden == 0
			lines.append(prefix + (glyphs["last"] if is_last else glyphs["branch"]) + label(child_id))
			walk(child_id, prefix + (glyphs["blank"] if is_last else glyphs["pipe"]), depth + 1)
		if hidden:
			lines.append(prefix + glyphs["last"] + f"(+{hidden} more children)")

	walk(root_id, "", 0)
	return lines


def render_tree(
	simulation: Simulation,
	root_id: int,
	max_generations: int = 4,
	max_children: int = 6,
	glyphs: Optional[Dict[str, str]] = None,
) -> List[str]:
	if root_id not in simulation.people:
		raise ValueError(f"unknown person id: {root_id}")
	glyphs = glyphs or TREE_GLYPHS
	index = build_timeline_index(simulation.people, simulation.current_turn)
	if root_id not in index.people:
		raise ValueError(f"person {root_id} is not born yet at {format_turn(simulation.current_turn)}")

	def label(person_id: int) -> str:
		person = index.people[person_id]
		alive = simulation._is_alive_at(person_id, simulation.current_turn)
		state = f"age {simulation._age_years_at(person_id, simulation.current_turn)}" if alive else "dead"
		return f"{person.name}#{person_id} ({person.sex}, b.Y{_birth_year(person)}, {state})"

	header = [
		f"Descendants of {index.people[root_id].name}#{root_id} at {format_turn(simulation.current_turn)} "
		f"({max_generations} generations, {simulation.descendants_count(root_id)} total descendants):"
	]
	return header + descendant_tree_lines(index, root_id, label, max_generations, max_children, glyphs)


def render_impact(
	simulation: Simulation,
	person_id: int,
	kill_turn: Optional[int] = None,
	max_generations: int = 4,
	max_children: int = 6,
	glyphs: Optional[Dict[str, str]] = None,
) -> List[str]:
	if person_id not in simulation.people:
		raise ValueError(f"unknown person id: {person_id}")
	glyphs = glyphs or TREE_GLYPHS
	victim = simulation.people[person_id]
	kill_turn = simulation.current_turn if kill_turn is None else clamp_turn(kill_turn)
	if kill_turn < victim.birth_turn:
		raise ValueError("cannot kill a person before they are born")

	before = build_timeline_index(simulation.people, simulation.current_turn)
	if person_id not in before.people:
		raise ValueError(f"person {person_id} is not born yet at {format_turn(simulation.current_turn)}")

	alternate = simulation.fork(extra_kill=(person_id, kill_turn))
	after = build_timeline_index(alternate.people, alternate.current_turn)

	before_keys = set(before.by_lineage)
	after_keys = set(after.by_lineage)
	erased = before_keys - after_keys
	unchanged = before_keys & after_keys
	created = after_keys - before_keys

	def label(node_id: int) -> str:
		person = before.people[node_id]
		base = f"{person.name}#{node_id} ({person.sex}, b.Y{_birth_year(person)})"
		if node_id == person_id:
			return f"{base} [KILLED at {format_turn(kill_turn)}]"
		if before.lineage_key[node_id] in erased:
			return f"{base} {glyphs['erased']} erased"
		return f"{base} {glyphs['kept']}"

	lines = [
		f"What if {victim.name}#{person_id} died at {format_turn(kill_turn)} "
		f"(natural death {format_turn(simulation._effective_death_turn(person_id))})?",
		f"Viewed from {format_turn(simulation.current_turn)}. This is a preview; the live timeline is unchanged.",
		"",
	]
	lines.extend(
		descendant_tree_lines(before, person_id, label, max_generations, max_children, glyphs)
	)

	before_stats = simulation.stats()
	after_stats = alternate.stats()
	living_delta = after_stats.living - before_stats.living
	lines.extend(
		[
			"",
			"Impact:",
			f"  erased    {len(erased):>7}  people never born",
			f"  unchanged {len(unchanged):>7}  people still born",
			f"  created   {len(created):>7}  people born who never existed",
			f"  living    {before_stats.living:>7} -> {after_stats.living} ({living_delta:+d})",
			f"  bloodlines{before_stats.family_count:>7} -> {after_stats.family_count} "
			f"({after_stats.family_count - before_stats.family_count:+d})",
		]
	)
	if not erased and not created:
		lines.append("")
		lines.append(
			"Nothing changed: the kill lands at or after every birth they influenced. "
			"Try an earlier turn, e.g. 'impact <id> year <n>'."
		)
	return lines


def render_stats(stats: SimulationStats) -> str:
	return (
		f"time={format_turn(stats.current_turn)} living={stats.living} adults={stats.adults} children={stats.children} "
		f"males={stats.males} females={stats.females} couples={stats.couples} births={stats.births} "
		f"deaths={stats.deaths} bloodlines={stats.family_count} interventions={stats.interventions} "
		f"oldest={stats.oldest_age_years}y"
	)


def print_help() -> None:
	print("Commands:")
	print("  help")
	print("  stats")
	print("  advance <amount> <turns|days|months|years>")
	print("  rewind <amount> <turns|days|months|years>")
	print("  goto year <n>")
	print("  goto turn <n>")
	print("  list [count] [alive|all]")
	print("  show <person_id>")
	print("  bloodlines [count]")
	print("  tree <person_id> [generations] [max_children]")
	print("  impact <person_id> [year <n>] [gens <n>]")
	print("  kill <person_id>")
	print("  undo")
	print("  quit")


def run_command(simulation: Simulation, command: str) -> bool:
	tokens = shlex.split(command)
	if not tokens:
		return True
	verb = tokens[0].lower()
	try:
		if verb == "help":
			print_help()
		elif verb == "stats":
			print(render_stats(simulation.stats()))
		elif verb == "advance":
			amount = int(tokens[1])
			unit = tokens[2].lower()
			simulation.advance(amount, unit)
			print(render_stats(simulation.stats()))
		elif verb == "rewind":
			amount = int(tokens[1])
			unit = tokens[2].lower()
			simulation.rewind(amount, unit)
			print(render_stats(simulation.stats()))
		elif verb == "goto":
			scope = tokens[1].lower()
			target = int(tokens[2])
			if scope == "year":
				simulation.goto_turn(target * TURNS_PER_YEAR)
			elif scope == "turn":
				simulation.goto_turn(target)
			else:
				raise ValueError("goto expects 'year' or 'turn'")
			print(render_stats(simulation.stats()))
		elif verb == "list":
			limit = int(tokens[1]) if len(tokens) > 1 else 20
			mode = tokens[2].lower() if len(tokens) > 2 else "alive"
			for line in simulation.list_people(limit=limit, living_only=(mode != "all")):
				print(line)
		elif verb == "show":
			print(simulation.describe_person(int(tokens[1])))
		elif verb == "bloodlines":
			limit = int(tokens[1]) if len(tokens) > 1 else 5
			for line in simulation.top_bloodlines(limit):
				print(line)
		elif verb == "tree":
			root_id = int(tokens[1])
			generations = int(tokens[2]) if len(tokens) > 2 else 4
			max_children = int(tokens[3]) if len(tokens) > 3 else 6
			for line in render_tree(simulation, root_id, generations, max_children, ACTIVE_GLYPHS):
				print(line)
		elif verb == "impact":
			target_id = int(tokens[1])
			kill_turn = None
			generations = 4
			options = tokens[2:]
			if len(options) % 2 != 0:
				raise ValueError("impact options must be pairs, e.g. 'impact 42 year 120 gens 5'")
			for name, value in zip(options[0::2], options[1::2]):
				key = name.lower()
				if key == "year":
					kill_turn = int(value) * TURNS_PER_YEAR
				elif key == "turn":
					kill_turn = int(value)
				elif key == "gens":
					generations = int(value)
				else:
					raise ValueError(f"unknown impact option: {name}")
			for line in render_impact(simulation, target_id, kill_turn, generations, glyphs=ACTIVE_GLYPHS):
				print(line)
		elif verb == "kill":
			print(simulation.kill(int(tokens[1])))
			print(render_stats(simulation.stats()))
		elif verb == "undo":
			print(simulation.undo_last_kill())
			print(render_stats(simulation.stats()))
		elif verb in {"quit", "exit"}:
			return False
		else:
			print("Unknown command. Use 'help'.")
	except (IndexError, ValueError) as exc:
		print(f"error: {exc}")
	return True


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Deterministic lineage simulation")
	parser.add_argument("--seed", default="chronicle")
	parser.add_argument("--founders", type=int, default=24)
	parser.add_argument("--checkpoint-years", type=int, default=50)
	parser.add_argument("--carrying-capacity", type=int, default=5000)
	parser.add_argument(
		"--ascii-tree",
		action="store_true",
		help="Draw trees with plain ASCII instead of box-drawing characters.",
	)
	parser.add_argument(
		"--command",
		action="append",
		default=[],
		help="Run a command non-interactively. May be passed multiple times.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.ascii_tree:
		ACTIVE_GLYPHS.update(ASCII_TREE_GLYPHS)
	simulation = Simulation(
		seed=args.seed,
		founder_count=args.founders,
		checkpoint_interval_years=args.checkpoint_years,
		carrying_capacity=args.carrying_capacity,
	)
	if args.command:
		for command in args.command:
			if not run_command(simulation, command):
				break
		return
	print("Deterministic time-travel simulation")
	print(render_stats(simulation.stats()))
	print("Type 'help' for commands.")
	while True:
		try:
			command = input("> ")
		except EOFError:
			print()
			break
		if not run_command(simulation, command):
			break


if __name__ == "__main__":
	main()
