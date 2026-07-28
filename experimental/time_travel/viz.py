"""Pygame family-tree visualizer for the deterministic lineage simulation.

Run:
	python3 viz.py --year 80
	python3 viz.py --year 200 --gens 6 --max-nodes 4000

The simulation itself lives in sim.py; this file only reads from it.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pygame

from sim import (
	TURNS_PER_YEAR,
	Simulation,
	build_timeline_index,
	format_turn,
	TimelineIndex,
)


SIBLING_SPACING = 46.0
LEVEL_HEIGHT = 96.0
NODE_RADIUS = 13.0
SIDEBAR_WIDTH = 330

# A generation fans out far wider than the tree is deep -- hundreds of cousins
# over a handful of generations -- so a layered layout collapses into an
# unreadable horizontal line at fit zoom. Radial spends that breadth around a
# circle instead, and is the default. Layered stays useful once zoomed into a
# small subtree.
LAYOUT_RADIAL = "radial"
LAYOUT_LAYERED = "layered"

# Low enough that fit-to-view can still frame a 4000-node radial graph, whose
# span runs to tens of thousands of world units.
MIN_SCALE = 0.002
MAX_SCALE = 4.0
LABEL_SCALE = 0.55
DETAIL_SCALE = 0.95

BACKGROUND = (16, 18, 24)
SIDEBAR_BG = (24, 27, 35)
SIDEBAR_LINE = (48, 53, 66)
TEXT = (222, 226, 235)
TEXT_DIM = (138, 146, 163)
TEXT_HEAD = (255, 255, 255)

EDGE = (62, 70, 88)
EDGE_COLLAPSE = (108, 78, 148)
EDGE_HIGHLIGHT = (236, 197, 92)

FEMALE = (226, 122, 150)
MALE = (104, 165, 232)
DEAD_MIX = (44, 48, 58)

ERASED = (222, 78, 78)
KEPT = (94, 196, 128)
VICTIM = (245, 214, 96)
SELECT_RING = (255, 255, 255)


def mix(color: Tuple[int, int, int], other: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
	return (
		int(color[0] + (other[0] - color[0]) * amount),
		int(color[1] + (other[1] - color[1]) * amount),
		int(color[2] + (other[2] - color[2]) * amount),
	)


@dataclass
class Layout:
	"""Positions for a depth-limited descendant DAG rooted at one person."""

	root_id: int
	positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
	depth: Dict[int, int] = field(default_factory=dict)
	primary_edges: List[Tuple[int, int]] = field(default_factory=list)
	collapse_edges: List[Tuple[int, int]] = field(default_factory=list)
	truncated: int = 0
	total_descendants: int = 0
	bounds: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
	mode: str = LAYOUT_RADIAL


def count_descendants(index: TimelineIndex, root_id: int) -> int:
	seen: Set[int] = set()
	stack = list(index.children.get(root_id, []))
	while stack:
		current = stack.pop()
		if current in seen:
			continue
		seen.add(current)
		stack.extend(index.children.get(current, []))
	return len(seen)


def build_layout(
	index: TimelineIndex,
	root_id: int,
	max_generations: int,
	max_nodes: int,
	mode: str = LAYOUT_RADIAL,
) -> Layout:
	"""Breadth-first spanning tree over the descendant DAG.

	Each person is placed exactly once, at the shallowest generation that reaches
	them. The parent that discovered them becomes the primary (tree) edge; a link
	from their other parent, when that parent is also on screen, becomes a
	collapse edge -- the visual signature of an interbred bloodline.
	"""
	layout = Layout(root_id=root_id, mode=mode)
	layout.total_descendants = count_descendants(index, root_id)

	primary_children: Dict[int, List[int]] = {}
	discovered: Set[int] = {root_id}
	layout.depth[root_id] = 0
	queue: deque = deque([root_id])
	truncated = 0

	while queue:
		person_id = queue.popleft()
		generation = layout.depth[person_id]
		if generation >= max_generations:
			continue
		for child_id in index.children.get(person_id, []):
			if child_id in discovered:
				continue
			if len(discovered) >= max_nodes:
				truncated += 1
				continue
			discovered.add(child_id)
			layout.depth[child_id] = generation + 1
			primary_children.setdefault(person_id, []).append(child_id)
			layout.primary_edges.append((person_id, child_id))
			queue.append(child_id)

	layout.truncated = truncated

	# Second parent links, only where both endpoints are actually drawn.
	primary_parent = {child: parent for parent, child in layout.primary_edges}
	for child_id in discovered:
		person = index.people[child_id]
		for parent_id in (person.mother_id, person.father_id):
			if parent_id is None or parent_id not in discovered:
				continue
			if primary_parent.get(child_id) == parent_id:
				continue
			layout.collapse_edges.append((parent_id, child_id))

	# Tidy layout in index space: leaves take the next free slot, parents centre
	# over their children. The slot value is then mapped to a column (layered) or
	# an angle (radial).
	next_slot = [0.0]
	slot: Dict[int, float] = {}

	def place(person_id: int) -> float:
		kids = primary_children.get(person_id)
		if not kids:
			value = next_slot[0]
			next_slot[0] += SIBLING_SPACING
			slot[person_id] = value
			return value
		child_slots = [place(child_id) for child_id in kids]
		value = (child_slots[0] + child_slots[-1]) / 2.0
		slot[person_id] = value
		return value

	sys.setrecursionlimit(max(10000, max_nodes * 4))
	place(root_id)

	span = max(next_slot[0], SIBLING_SPACING)
	max_depth = max(layout.depth.values()) if layout.depth else 0

	if mode == LAYOUT_RADIAL and max_depth > 0:
		# Ring spacing is chosen so the outermost ring's circumference equals the
		# full index-space span, i.e. leaves keep SIBLING_SPACING apart.
		ring_gap = max(110.0, span / (2.0 * math.pi) / max_depth)
		for person_id, value in slot.items():
			angle = (value / span) * 2.0 * math.pi
			radius = layout.depth[person_id] * ring_gap
			layout.positions[person_id] = (radius * math.cos(angle), radius * math.sin(angle))
	else:
		for person_id, value in slot.items():
			layout.positions[person_id] = (value, layout.depth[person_id] * LEVEL_HEIGHT)

	xs = [position[0] for position in layout.positions.values()]
	ys = [position[1] for position in layout.positions.values()]
	layout.bounds = (min(xs), min(ys), max(max(xs) - min(xs), 1.0), max(max(ys) - min(ys), 1.0))
	return layout


@dataclass
class ImpactView:
	person_id: int
	kill_turn: int
	erased: Set[int]
	created: int
	unchanged: int
	living_before: int
	living_after: int


class Visualizer:
	def __init__(self, simulation: Simulation, args: argparse.Namespace) -> None:
		self.simulation = simulation
		self.max_generations = args.gens
		self.max_nodes = args.max_nodes
		self.show_labels = True
		self.show_collapse = True

		pygame.init()
		pygame.display.set_caption("Lineage Visualizer")
		self.screen = pygame.display.set_mode((args.width, args.height), pygame.RESIZABLE)
		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont("consolas,dejavusansmono,couriernew,monospace", 13)
		self.font_small = pygame.font.SysFont("consolas,dejavusansmono,couriernew,monospace", 11)
		self.font_head = pygame.font.SysFont("consolas,dejavusansmono,couriernew,monospace", 16, bold=True)
		self.text_cache: Dict[Tuple[str, Tuple[int, int, int], int], pygame.Surface] = {}

		self.index: TimelineIndex = build_timeline_index(simulation.people, simulation.current_turn)
		self.root_id = args.root if args.root is not None else self.default_root()
		self.root_history: List[int] = []
		self.selected_id: Optional[int] = None
		self.impact: Optional[ImpactView] = None
		self.kill_year_offset = 0
		self.status = ""

		self.scale = 1.0
		self.offset = [0.0, 0.0]
		self.dragging = False
		self.drag_anchor = (0, 0)

		self.layout_mode = LAYOUT_LAYERED if args.layered else LAYOUT_RADIAL
		self.relayout()
		self.fit_view()

	# ---------------------------------------------------------------- helpers

	def relayout(self) -> None:
		self.layout = build_layout(
			self.index,
			self.root_id,
			self.max_generations,
			self.max_nodes,
			self.layout_mode,
		)

	def default_root(self) -> int:
		founders = [
			person_id
			for person_id, person in self.index.people.items()
			if person.mother_id is None or person.father_id is None
		]
		if not founders:
			return min(self.index.people)
		return max(founders, key=lambda person_id: (count_descendants(self.index, person_id), -person_id))

	def text(self, message: str, color: Tuple[int, int, int], size: int = 13) -> pygame.Surface:
		key = (message, color, size)
		cached = self.text_cache.get(key)
		if cached is None:
			font = self.font_small if size == 11 else (self.font_head if size == 16 else self.font)
			cached = font.render(message, True, color)
			if len(self.text_cache) > 6000:
				self.text_cache.clear()
			self.text_cache[key] = cached
		return cached

	def viewport(self) -> Tuple[int, int]:
		return (max(1, self.screen.get_width() - SIDEBAR_WIDTH), self.screen.get_height())

	def to_screen(self, x: float, y: float) -> Tuple[int, int]:
		return (int(x * self.scale + self.offset[0]), int(y * self.scale + self.offset[1]))

	def fit_view(self) -> None:
		width, height = self.viewport()
		min_x, min_y, span_x, span_y = self.layout.bounds
		padding = 60
		self.scale = min((width - padding) / span_x, (height - padding) / span_y)
		self.scale = max(MIN_SCALE, min(MAX_SCALE, self.scale))
		self.offset[0] = width / 2 - (min_x + span_x / 2) * self.scale
		self.offset[1] = height / 2 - (min_y + span_y / 2) * self.scale

	def zoom_at(self, factor: float, pivot: Tuple[int, int]) -> None:
		new_scale = max(MIN_SCALE, min(MAX_SCALE, self.scale * factor))
		if new_scale == self.scale:
			return
		ratio = new_scale / self.scale
		self.offset[0] = pivot[0] - (pivot[0] - self.offset[0]) * ratio
		self.offset[1] = pivot[1] - (pivot[1] - self.offset[1]) * ratio
		self.scale = new_scale

	def busy(self, message: str) -> None:
		width, height = self.viewport()
		box = self.text(message, TEXT_HEAD, 16)
		self.screen.fill(BACKGROUND, (0, 0, width, height))
		self.screen.blit(box, (width // 2 - box.get_width() // 2, height // 2 - box.get_height() // 2))
		pygame.display.flip()

	# ----------------------------------------------------------------- state

	def rebuild(self, keep_view: bool = False) -> None:
		self.index = build_timeline_index(self.simulation.people, self.simulation.current_turn)
		if self.root_id not in self.index.people:
			self.root_id = self.default_root()
			self.status = "root not born yet; jumped to strongest bloodline"
		self.relayout()
		if not keep_view:
			self.fit_view()

	def set_root(self, person_id: int) -> None:
		if person_id == self.root_id or person_id not in self.index.people:
			return
		self.root_history.append(self.root_id)
		self.root_id = person_id
		self.impact = None
		self.relayout()
		self.fit_view()

	def step_year(self, delta_years: int) -> None:
		target = self.simulation.current_turn + delta_years * TURNS_PER_YEAR
		if target < 0:
			target = 0
		self.busy(f"replaying to Y{target // TURNS_PER_YEAR} ...")
		self.simulation.goto_turn(target)
		self.impact = None
		self.rebuild()
		self.status = f"now at {format_turn(self.simulation.current_turn)}"

	def toggle_impact(self) -> None:
		if self.impact is not None:
			self.impact = None
			self.status = "impact preview off"
			return
		if self.selected_id is None:
			self.status = "select a person first (click a node)"
			return
		self.compute_impact()

	def compute_impact(self) -> None:
		person_id = self.selected_id
		if person_id is None:
			return
		person = self.index.people[person_id]
		kill_turn = max(person.birth_turn, person.birth_turn + self.kill_year_offset * TURNS_PER_YEAR)
		kill_turn = min(kill_turn, self.simulation.current_turn)
		self.busy(f"forking timeline: {person.name}#{person_id} dies {format_turn(kill_turn)} ...")

		alternate = self.simulation.fork(extra_kill=(person_id, kill_turn))
		after = build_timeline_index(alternate.people, alternate.current_turn)
		before_keys = set(self.index.by_lineage)
		after_keys = set(after.by_lineage)
		erased_keys = before_keys - after_keys
		erased_ids = {
			other_id
			for other_id, key in self.index.lineage_key.items()
			if key in erased_keys
		}
		self.impact = ImpactView(
			person_id=person_id,
			kill_turn=kill_turn,
			erased=erased_ids,
			created=len(after_keys - before_keys),
			unchanged=len(before_keys & after_keys),
			living_before=self.simulation.stats().living,
			living_after=alternate.stats().living,
		)
		self.status = f"impact preview: {person.name}#{person_id}"

	def node_at(self, position: Tuple[int, int]) -> Optional[int]:
		radius = max(4.0, NODE_RADIUS * self.scale)
		best_id = None
		best_distance = radius * radius
		for person_id, (x, y) in self.layout.positions.items():
			screen_x, screen_y = self.to_screen(x, y)
			dx = screen_x - position[0]
			dy = screen_y - position[1]
			distance = dx * dx + dy * dy
			if distance <= best_distance:
				best_distance = distance
				best_id = person_id
		return best_id

	# ---------------------------------------------------------------- drawing

	def node_color(self, person_id: int) -> Tuple[int, int, int]:
		if self.impact is not None:
			if person_id == self.impact.person_id:
				return VICTIM
			return ERASED if person_id in self.impact.erased else KEPT
		person = self.index.people[person_id]
		base = FEMALE if person.sex == "F" else MALE
		if not self.simulation._is_alive_at(person_id, self.simulation.current_turn):
			return mix(base, DEAD_MIX, 0.62)
		return base

	def draw_graph(self) -> None:
		width, height = self.viewport()
		surface = self.screen.subsurface((0, 0, width, height))
		surface.fill(BACKGROUND)

		margin = 80
		def visible(point: Tuple[int, int]) -> bool:
			return -margin <= point[0] <= width + margin and -margin <= point[1] <= height + margin

		screen_points = {
			person_id: self.to_screen(x, y) for person_id, (x, y) in self.layout.positions.items()
		}

		# Collapse edges go down first so the primary tree stays legible on top of
		# them; at high node counts they otherwise swamp the whole view.
		if self.show_collapse:
			for parent_id, child_id in self.layout.collapse_edges:
				start = screen_points[parent_id]
				end = screen_points[child_id]
				if not visible(start) and not visible(end):
					continue
				pygame.draw.line(surface, EDGE_COLLAPSE, start, end, 1)

		for parent_id, child_id in self.layout.primary_edges:
			start = screen_points[parent_id]
			end = screen_points[child_id]
			if not visible(start) and not visible(end):
				continue
			color = EDGE
			if self.impact is not None and child_id in self.impact.erased:
				color = mix(EDGE, ERASED, 0.45)
			elif self.selected_id in (parent_id, child_id):
				color = EDGE_HIGHLIGHT
			pygame.draw.line(surface, color, start, end, 1)

		radius = max(2, int(NODE_RADIUS * self.scale))
		draw_labels = self.show_labels and self.scale >= LABEL_SCALE
		for person_id, point in screen_points.items():
			if not visible(point):
				continue
			pygame.draw.circle(surface, self.node_color(person_id), point, radius)
			if person_id == self.root_id:
				pygame.draw.circle(surface, TEXT_HEAD, point, radius + 3, 2)
			if person_id == self.selected_id:
				pygame.draw.circle(surface, SELECT_RING, point, radius + 6, 2)
			if draw_labels:
				person = self.index.people[person_id]
				name = person.name if self.scale < DETAIL_SCALE else f"{person.name}#{person_id}"
				label = self.text(name, TEXT_DIM if self.scale < DETAIL_SCALE else TEXT, 11)
				surface.blit(label, (point[0] - label.get_width() // 2, point[1] + radius + 3))

	def draw_sidebar(self) -> None:
		width, height = self.viewport()
		panel = pygame.Rect(width, 0, SIDEBAR_WIDTH, height)
		self.screen.fill(SIDEBAR_BG, panel)
		pygame.draw.line(self.screen, SIDEBAR_LINE, (width, 0), (width, height), 1)

		x = width + 14
		y = 14

		def line(message: str, color: Tuple[int, int, int] = TEXT, size: int = 13, gap: int = 17) -> None:
			nonlocal y
			if y > height - 16:
				return
			self.screen.blit(self.text(message, color, size), (x, y))
			y += gap

		def rule() -> None:
			nonlocal y
			y += 5
			pygame.draw.line(self.screen, SIDEBAR_LINE, (x, y), (width + SIDEBAR_WIDTH - 14, y), 1)
			y += 9

		stats = self.simulation.stats()
		line("LINEAGE VISUALIZER", TEXT_HEAD, 16, 24)
		line(format_turn(self.simulation.current_turn), TEXT_DIM, 11)
		line(f"living {stats.living}  births {stats.births}  deaths {stats.deaths}", TEXT_DIM, 11)
		line(f"couples {stats.couples}  bloodlines {stats.family_count}", TEXT_DIM, 11)
		if stats.interventions:
			line(f"interventions {stats.interventions}", VICTIM, 11)
		rule()

		root = self.index.people[self.root_id]
		line("ROOT", TEXT_DIM, 11)
		line(f"{root.name}#{self.root_id}", TEXT)
		line(f"{self.layout.total_descendants} descendants", TEXT_DIM, 11)
		line(f"showing {len(self.layout.positions)} over {self.max_generations} gens", TEXT_DIM, 11)
		line(f"{self.layout_mode} layout", TEXT_DIM, 11)
		if self.layout.truncated:
			line(f"{self.layout.truncated}+ hidden by node cap", VICTIM, 11)
		if self.layout.collapse_edges:
			line(f"{len(self.layout.collapse_edges)} collapse links", EDGE_COLLAPSE, 11)
		rule()

		if self.selected_id is not None and self.selected_id in self.index.people:
			person = self.index.people[self.selected_id]
			alive = self.simulation._is_alive_at(self.selected_id, self.simulation.current_turn)
			line("SELECTED", TEXT_DIM, 11)
			line(f"{person.name}#{self.selected_id}", TEXT_HEAD)
			line(f"{'female' if person.sex == 'F' else 'male'}, born Y{person.birth_turn // TURNS_PER_YEAR}", TEXT_DIM, 11)
			if alive:
				line(f"alive, age {self.simulation._age_years_at(self.selected_id, self.simulation.current_turn)}", KEPT, 11)
			else:
				line(f"died {format_turn(self.simulation._effective_death_turn(self.selected_id))}", TEXT_DIM, 11)
			for label, parent_id in (("mother", person.mother_id), ("father", person.father_id)):
				if parent_id is None:
					line(f"{label}: founder line", TEXT_DIM, 11)
				else:
					line(f"{label}: {self.simulation.people[parent_id].name}#{parent_id}", TEXT_DIM, 11)
			line(f"children: {len(self.index.children.get(self.selected_id, []))}", TEXT_DIM, 11)
		else:
			line("SELECTED", TEXT_DIM, 11)
			line("click a node", TEXT_DIM, 11)
		rule()

		if self.impact is not None:
			victim = self.index.people[self.impact.person_id]
			delta = self.impact.living_after - self.impact.living_before
			line("IMPACT PREVIEW", VICTIM, 11)
			line(f"{victim.name}#{self.impact.person_id}", TEXT)
			line(f"dies {format_turn(self.impact.kill_turn)}", TEXT_DIM, 11)
			line(f"erased    {len(self.impact.erased)}", ERASED, 11)
			line(f"unchanged {self.impact.unchanged}", KEPT, 11)
			line(f"created   {self.impact.created}", MALE, 11)
			line(f"living    {self.impact.living_before} -> {self.impact.living_after} ({delta:+d})", TEXT_DIM, 11)
			line("timeline not modified", TEXT_DIM, 11)
		else:
			line("IMPACT PREVIEW", TEXT_DIM, 11)
			line(f"K = what if selected dies", TEXT_DIM, 11)
			line(f"kill offset: birth +{self.kill_year_offset}y", TEXT_DIM, 11)
		rule()

		line("drag/arrows  pan", TEXT_DIM, 11, 14)
		line("wheel +/-    zoom", TEXT_DIM, 11, 14)
		line("click        select", TEXT_DIM, 11, 14)
		line("R / B / H    root, back, home", TEXT_DIM, 11, 14)
		line("K            impact preview", TEXT_DIM, 11, 14)
		line("[ ]          kill year -/+", TEXT_DIM, 11, 14)
		line(", .          sim year -/+10", TEXT_DIM, 11, 14)
		line("G / shift-G  generations -/+", TEXT_DIM, 11, 14)
		line("T            radial/layered", TEXT_DIM, 11, 14)
		line("C            collapse links", TEXT_DIM, 11, 14)
		line("F / L        fit, labels", TEXT_DIM, 11, 14)
		line("Q            quit", TEXT_DIM, 11, 14)

		if self.status:
			message = self.status[:42]
			self.screen.blit(self.text(message, VICTIM, 11), (x, height - 20))

	def draw(self) -> None:
		self.draw_graph()
		self.draw_sidebar()
		pygame.display.flip()

	# ------------------------------------------------------------------ loop

	def handle_key(self, event: pygame.event.Event) -> bool:
		shift = bool(event.mod & pygame.KMOD_SHIFT)
		if event.key in (pygame.K_q, pygame.K_ESCAPE):
			return False
		elif event.key == pygame.K_f:
			self.fit_view()
		elif event.key == pygame.K_l:
			self.show_labels = not self.show_labels
		elif event.key == pygame.K_c:
			self.show_collapse = not self.show_collapse
			self.status = f"collapse links {'on' if self.show_collapse else 'off'}"
		elif event.key == pygame.K_t:
			self.layout_mode = LAYOUT_LAYERED if self.layout_mode == LAYOUT_RADIAL else LAYOUT_RADIAL
			self.relayout()
			self.fit_view()
			self.status = f"{self.layout_mode} layout"
		elif event.key == pygame.K_r and self.selected_id is not None:
			self.set_root(self.selected_id)
		elif event.key == pygame.K_b and self.root_history:
			self.root_id = self.root_history.pop()
			self.impact = None
			self.relayout()
			self.fit_view()
		elif event.key == pygame.K_h:
			self.root_history.clear()
			self.root_id = self.default_root()
			self.impact = None
			self.relayout()
			self.fit_view()
		elif event.key == pygame.K_k:
			self.toggle_impact()
		elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
			step = 1 if event.key == pygame.K_RIGHTBRACKET else -1
			self.kill_year_offset = max(0, self.kill_year_offset + step * 5)
			if self.impact is not None:
				self.compute_impact()
		elif event.key == pygame.K_COMMA:
			self.step_year(-10)
		elif event.key == pygame.K_PERIOD:
			self.step_year(10)
		elif event.key == pygame.K_g:
			self.max_generations = max(1, self.max_generations + (1 if shift else -1))
			self.relayout()
			self.fit_view()
			self.status = f"{self.max_generations} generations"
		elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
			width, height = self.viewport()
			self.zoom_at(1.2, (width // 2, height // 2))
		elif event.key == pygame.K_MINUS:
			width, height = self.viewport()
			self.zoom_at(1 / 1.2, (width // 2, height // 2))
		elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
			step = 80
			if event.key == pygame.K_LEFT:
				self.offset[0] += step
			elif event.key == pygame.K_RIGHT:
				self.offset[0] -= step
			elif event.key == pygame.K_UP:
				self.offset[1] += step
			else:
				self.offset[1] -= step
		return True

	def run(self) -> None:
		running = True
		while running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False
				elif event.type == pygame.VIDEORESIZE:
					self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
				elif event.type == pygame.KEYDOWN:
					running = self.handle_key(event)
				elif event.type == pygame.MOUSEBUTTONDOWN:
					if event.pos[0] < self.viewport()[0]:
						if event.button == 1:
							hit = self.node_at(event.pos)
							if hit is not None:
								self.selected_id = hit
								if self.impact is not None:
									self.compute_impact()
							self.dragging = True
							self.drag_anchor = event.pos
						elif event.button == 4:
							self.zoom_at(1.15, event.pos)
						elif event.button == 5:
							self.zoom_at(1 / 1.15, event.pos)
				elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
					self.dragging = False
				elif event.type == pygame.MOUSEMOTION and self.dragging:
					self.offset[0] += event.pos[0] - self.drag_anchor[0]
					self.offset[1] += event.pos[1] - self.drag_anchor[1]
					self.drag_anchor = event.pos
				elif event.type == pygame.MOUSEWHEEL:
					position = pygame.mouse.get_pos()
					if position[0] < self.viewport()[0]:
						self.zoom_at(1.15 ** event.y, position)

			self.draw()
			self.clock.tick(60)
		pygame.quit()

	def screenshot(self, path: str) -> None:
		self.draw()
		pygame.image.save(self.screen, path)
		pygame.quit()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Pygame family-tree visualizer")
	parser.add_argument("--seed", default="chronicle")
	parser.add_argument("--founders", type=int, default=24)
	parser.add_argument("--checkpoint-years", type=int, default=50)
	parser.add_argument("--carrying-capacity", type=int, default=5000)
	parser.add_argument("--year", type=int, default=80, help="Year to open the view at.")
	parser.add_argument("--root", type=int, default=None, help="Person id to root on.")
	parser.add_argument("--gens", type=int, default=5, help="Generations to draw.")
	parser.add_argument("--max-nodes", type=int, default=3000, help="Node budget per view.")
	parser.add_argument("--width", type=int, default=1500)
	parser.add_argument("--height", type=int, default=900)
	parser.add_argument("--layered", action="store_true", help="Open in layered mode instead of radial.")
	parser.add_argument("--select", type=int, default=None, help="Pre-select a person id.")
	parser.add_argument("--impact", action="store_true", help="Open with the impact preview on the selection.")
	parser.add_argument("--screenshot", default=None, help="Render one frame to a PNG and exit.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	simulation = Simulation(
		seed=args.seed,
		founder_count=args.founders,
		checkpoint_interval_years=args.checkpoint_years,
		carrying_capacity=args.carrying_capacity,
	)
	simulation.goto_turn(args.year * TURNS_PER_YEAR)
	visualizer = Visualizer(simulation, args)
	if args.select is not None and args.select in visualizer.index.people:
		visualizer.selected_id = args.select
		if args.impact:
			visualizer.compute_impact()
	if args.screenshot:
		visualizer.screenshot(args.screenshot)
		return
	visualizer.run()


if __name__ == "__main__":
	main()
