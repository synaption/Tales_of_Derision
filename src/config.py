"""Shared game constants, in a neutral module so ``main`` and ``ui`` (and anything
else) can import them without importing each other."""
from __future__ import annotations

from game_map import ARCHIPELAGO_HEIGHT, ARCHIPELAGO_WIDTH

# Scale-test world: an archipelago of 100 copies of the classic 120x60 island
# (10x10 grid), each with its own coastline, houses, lakes/river, village and
# wildlife, all in open sea with no wall border. Swap WORLD_LAYOUT back to "auto"
# for the original single-island 360x180 world. See ``GameMap`` layout="islands".
MAP_WIDTH = ARCHIPELAGO_WIDTH
MAP_HEIGHT = ARCHIPELAGO_HEIGHT
WORLD_LAYOUT = "islands"

# Turns to pre-simulate behind a "Generating world..." screen before play, so the
# startup building boom (every homeless villager raising a home at once) happens
# during loading instead of as lag on the first turns. 0 disables the pre-sim.
WORLD_SETTLE_TURNS = 150

# Fixed world seed for a new game with no --seed, so every new game regenerates the
# same world for now. (Swap to rng.new_seed() for randomized worlds.)
DEFAULT_WORLD_SEED = 0x7A1E5  # "TALES"
