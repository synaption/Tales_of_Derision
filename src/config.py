"""Shared game constants, in a neutral module so ``main`` and ``ui`` (and anything
else) can import them without importing each other."""
from __future__ import annotations

from game_map import LAND_HEIGHT, LAND_WIDTH

# The world is a 3x3 grid of 120x60 sections: the habitable island sits in the
# centre section, ringed by coastline, and the surrounding eight sections are open
# ocean full of fish and seaweed. The land itself is the classic 120x60 world.
MAP_WIDTH = LAND_WIDTH * 3
MAP_HEIGHT = LAND_HEIGHT * 3

# Fixed world seed for a new game with no --seed, so every new game regenerates the
# same world for now. (Swap to rng.new_seed() for randomized worlds.)
DEFAULT_WORLD_SEED = 0x7A1E5  # "TALES"
