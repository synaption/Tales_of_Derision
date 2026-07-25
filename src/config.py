"""Shared game constants, in a neutral module so ``main`` and ``ui`` (and anything
else) can import them without importing each other."""
from __future__ import annotations

from game_map import ARCHIPELAGO_GRID, ARCHIPELAGO_HEIGHT, ARCHIPELAGO_WIDTH

# Scale-test world: an archipelago of copies of the classic 120x60 island, each with
# its own coastline, houses, lakes/river, village and wildlife, all in open sea with
# no wall border. Swap WORLD_LAYOUT back to "auto" for the original single-island
# 360x180 world. See ``GameMap`` layout="islands".
# MAP_WIDTH/MAP_HEIGHT are the *default* archipelago size: the size a new game starts
# on when nothing else is chosen, and the fallback for a save that doesn't record one.
# A new game asks for its own grid size (WORLD_GRID_CHOICES) and builds the map at
# ``game_map.archipelago_size`` of that choice.
MAP_WIDTH = ARCHIPELAGO_WIDTH
MAP_HEIGHT = ARCHIPELAGO_HEIGHT
WORLD_LAYOUT = "islands"

# Island-grid sizes offered on the new-game screen: an N x N grid is N**2 islands, so
# the largest is a hundred-island scale test that takes a while to generate, and the
# smallest is a single classic island for quick play/testing.
WORLD_GRID_CHOICES = (1, 2, 3, 5, 10)
DEFAULT_WORLD_GRID = ARCHIPELAGO_GRID

# Turns to pre-simulate behind a "Generating world..." screen before play, so the
# startup building boom (every homeless villager raising a home at once) happens
# during loading instead of as lag on the first turns. 0 disables the pre-sim.
WORLD_SETTLE_TURNS = 150

# Fixed world seed for a new game with no --seed, so every new game regenerates the
# same world for now. (Swap to rng.new_seed() for randomized worlds.)
DEFAULT_WORLD_SEED = 0x7A1E5  # "TALES"

# Live-play catch-up budget for region-aware AI when the player enters or walks
# inside a region that is behind. ``None`` would replay the whole debt in one
# input frame; a small deterministic cap prioritizes the player's action and
# amortizes old off-screen debt over subsequent turns/idle pumps.
ACTIVE_REGION_CATCHUP_STEPS_PER_INPUT = 1
