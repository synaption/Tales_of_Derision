#!/usr/bin/env python3
"""Stress/profile harness for the per-turn simulation cost.

Builds the real 360x180 ocean world, walks the player across a region boundary for
N turns under cProfile, and prints the hottest functions. Sim-only by default (no
RenderProcessor) so the pygame map-surface build never masks simulation cost -- the
trustworthy signal per wiki/Performance.md. Determinism-affecting metrics
(per-call time, node counts) are what to trust; whole-turn A/B timings are
confounded because A*'s tie-break changes NPC routes and the worlds diverge.

    python3 scripts/profile_turns.py [turns] [--flood] [--render] [--sort tottime|cumulative]

--flood spawns a cave rat on every walkable tile (thousands of NPCs); --render also
registers the RenderProcessor with a headless fake renderer.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import esper  # noqa: E402

from components import Position  # noqa: E402
from config import MAP_HEIGHT, MAP_WIDTH  # noqa: E402
from game_map import GameMap  # noqa: E402
from rng import set_world_rng  # noqa: E402
from renderer.base import Renderer  # noqa: E402
from worldgen import _setup_world  # noqa: E402
from systems import (  # noqa: E402
    FishAiProcessor, HousingProcessor, MovementProcessor, NeedsProcessor,
    NpcAiProcessor, RenderProcessor, ReproductionProcessor, TimeProcessor,
    TreeGrowthProcessor,
)
from content.effects import EffectsProcessor  # noqa: E402


class _NullRenderer(Renderer):
    """Headless renderer that discards everything (for --render profiling)."""
    def setup(self): ...
    def teardown(self): ...
    def clear(self): ...
    def draw_glyph(self, *a, **k): ...
    def draw_text(self, *a, **k): ...
    def present(self): ...
    def poll_action(self): return None


def build(flood: bool, render: bool) -> GameMap:
    esper.clear_database()
    set_world_rng(0x7A1E5)
    game_map = GameMap(MAP_WIDTH, MAP_HEIGHT)
    player = Position(MAP_WIDTH // 2, MAP_HEIGHT // 2)
    count = _setup_world(game_map, player, rat_flood=flood)
    esper.add_processor(TimeProcessor(), priority=2)
    esper.add_processor(MovementProcessor(game_map), priority=1)
    esper.add_processor(HousingProcessor(game_map), priority=0)
    esper.add_processor(NpcAiProcessor(game_map), priority=0)
    esper.add_processor(FishAiProcessor(game_map), priority=0)
    esper.add_processor(NeedsProcessor(), priority=0)
    esper.add_processor(EffectsProcessor(), priority=0)
    esper.add_processor(TreeGrowthProcessor(game_map), priority=0)
    esper.add_processor(ReproductionProcessor(), priority=0)
    if render:
        esper.add_processor(RenderProcessor(_NullRenderer(), game_map), priority=0)
    npcs = sum(1 for _e, _c in esper.get_components(Position))
    print(f"world built: {count if flood else 'scripted'} spawn return, {npcs} positioned entities")
    return game_map


def run(turns: int) -> None:
    # Alternate right/down so the player crosses section seams (triggers region
    # catch-up bursts -- the realistic worst case).
    moves = ["move_right", "move_down"]
    for i in range(turns):
        esper.process(moves[i % len(moves)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("turns", nargs="?", type=int, default=200)
    ap.add_argument("--flood", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--sort", default="tottime", choices=["tottime", "cumulative", "ncalls"])
    ap.add_argument("--lines", type=int, default=25)
    args = ap.parse_args()

    build(args.flood, args.render)
    profiler = cProfile.Profile()
    profiler.enable()
    run(args.turns)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(args.sort)
    stats.print_stats(args.lines)
    print(stream.getvalue())
    total = sum(v[2] for v in stats.stats.values())  # tottime sum
    print(f"{args.turns} turns; total profiled time ~{total*1000:.0f} ms "
          f"(~{total/args.turns*1000:.2f} ms/turn)")


if __name__ == "__main__":
    main()
