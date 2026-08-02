"""Builds the city, the squad and the player, deterministically from a seed.

Kept apart from :mod:`sim` because generation is a one-shot that happens
before the first frame, and apart from :mod:`render` because a headless test
wants the same streets the game has. The only rule the layout has to respect
is that the player's spawn is in the open -- everything else is allowed to be
ugly, and at 320x240 with fog at 150m, it will be.
"""

from __future__ import annotations

import math
import random

from panda3d.core import Vec3

import entities
from components import Building, Transform
from sim import ARENA, Sim

# One city block, kerb to kerb, including the street on two of its sides.
BLOCK = 46.0
STREET = 15.0

# Nothing is generated inside this radius of the origin, so the Wing Diver
# always starts on open tarmac with room to take off.
PLAZA = 26.0


def build_city(sim: Sim, seed: int | None = None) -> list[int]:
    """Fill the arena with boxes on a street grid. Returns the building ids.

    Each block gets one to three towers of differing heights rather than a
    single slab, which gives the skyline enough variation to read as a city
    and gives ants enough corners to break around.
    """
    rng = random.Random(sim.seed if seed is None else seed)
    made: list[int] = []
    usable = BLOCK - STREET  # footprint budget inside one block

    steps = int(ARENA // BLOCK)
    for gx in range(-steps, steps + 1):
        for gy in range(-steps, steps + 1):
            cx, cy = gx * BLOCK, gy * BLOCK
            if math.hypot(cx, cy) < PLAZA + usable * 0.5:
                continue
            if rng.random() < 0.12:
                continue  # an empty lot, for variety and for cover

            for _ in range(rng.randint(1, 3)):
                half_x = rng.uniform(5.0, usable * 0.5)
                half_y = rng.uniform(5.0, usable * 0.5)
                slack_x = max(0.0, usable * 0.5 - half_x)
                slack_y = max(0.0, usable * 0.5 - half_y)
                x = cx + rng.uniform(-slack_x, slack_x)
                y = cy + rng.uniform(-slack_y, slack_y)
                if math.hypot(x, y) < PLAZA:
                    continue

                # Taller toward the middle of town, with a long tail so the
                # occasional tower breaks the fog line.
                downtown = 1.0 - min(1.0, math.hypot(cx, cy) / ARENA)
                height = rng.uniform(9.0, 22.0) + downtown * rng.uniform(4.0, 46.0)
                style = rng.randrange(4)

                ent = entities.spawn_building(Vec3(x, y, 0.0), half_x, half_y, height, style)
                sim.register_building(
                    ent,
                    Transform(pos=Vec3(x, y, 0.0)),
                    Building(half_x=half_x, half_y=half_y, height=height, style=style),
                )
                made.append(ent)
    return made


def populate(sim: Sim, allies: int = 8, ants: int = 14) -> int:
    """Place the player, a squad around them, and an opening ant patrol.

    Returns the player entity, which is also recorded on the ``Sim`` so the
    wave spawner and the camera can find it without being told.
    """
    rng = random.Random(sim.seed ^ 0x5EED)

    player = entities.spawn_player(Vec3(0.0, 0.0, 0.0))
    sim.player = player

    for i in range(allies):
        angle = math.tau * i / max(allies, 1) + rng.uniform(-0.2, 0.2)
        dist = rng.uniform(6.0, 16.0)
        pos = Vec3(math.cos(angle) * dist, math.sin(angle) * dist, 0.0)
        pos.z = sim.ground_height(pos.x, pos.y)
        entities.spawn_ally(pos, rally=Vec3(pos), courage=rng.uniform(0.4, 1.4), seed=i)

    for i in range(ants):
        angle = rng.uniform(0.0, math.tau)
        dist = rng.uniform(55.0, 95.0)
        x = math.cos(angle) * dist
        y = math.sin(angle) * dist
        entities.spawn_ant(
            Vec3(x, y, sim.ground_height(x, y)),
            spitter=(i % 5 == 0),
            seed=i,
        )

    return player


def new_game(seed: int = 1234, allies: int = 8, ants: int = 14) -> Sim:
    """The whole world, ready to step. No renderer involved."""
    sim = Sim(seed=seed)
    build_city(sim)
    populate(sim, allies=allies, ants=ants)
    return sim
