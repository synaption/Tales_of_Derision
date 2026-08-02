"""Entity factories: the only place that knows which components go together.

Every one of these is a plain function that composes a bag of components and
returns an entity id. There is no ``Actor`` base class and no ``Ant(Enemy)``
hierarchy -- an ant is "a thing with a Walker, an AntBrain and a bite", and a
Wing Diver is "a thing with a Flight and an Energy". Give a soldier a
:class:`~components.Flight` and he flies; that is the whole argument for doing
it this way.

Tuning constants live next to the factory that uses them so that balance is
one file to read.
"""

from __future__ import annotations

import esper
from panda3d.core import Vec3

from components import (
    AllyBrain,
    AntBrain,
    Body,
    Building,
    Dead,
    Energy,
    FactionTag,
    Faction,
    Flight,
    Gait,
    Health,
    Intent,
    Lifetime,
    Player,
    Projectile,
    Renderable,
    Transform,
    Velocity,
    Walker,
    Weapon,
)


def spawn_player(pos: Vec3) -> int:
    """The Wing Diver.

    Fragile, fast, and entirely governed by one number. The lance costs 12
    energy a shot against a 100 pool that also has to lift her -- shooting
    from the air is the trade the class is built around.
    """
    return esper.create_entity(
        Player(),
        Transform(pos=Vec3(pos)),
        Velocity(),
        Body(radius=0.55, height=1.75, gravity=22.0),
        Energy(),
        Flight(),
        Intent(),
        Health(maximum=260.0, current=260.0),
        FactionTag(Faction.EDF),
        Weapon(
            damage=34.0,
            speed=95.0,
            cooldown=0.28,
            range=160.0,
            energy_cost=12.0,
            kind="lance",
        ),
        Gait(rate=1.8),
        Renderable(kind="wingdiver", tint=(0.85, 0.9, 1.0)),
    )


def spawn_ally(pos: Vec3, rally: Vec3, courage: float = 1.0, seed: int = 0) -> int:
    """An EDF grunt.

    Deliberately not very good: slow rifle, wide spread, 90 health. Allies
    exist so the city sounds inhabited and so the ants have something to eat
    that is not you.
    """
    return esper.create_entity(
        Transform(pos=Vec3(pos)),
        Velocity(),
        Body(radius=0.5, height=1.8, gravity=22.0),
        Walker(accel=26.0, max_speed=5.2, friction=11.0, turn_rate=420.0),
        Intent(),
        Health(maximum=90.0, current=90.0),
        FactionTag(Faction.EDF),
        Weapon(damage=8.0, speed=70.0, cooldown=0.5, range=95.0, spread=3.5, kind="tracer"),
        AllyBrain(rally=Vec3(rally), courage=courage, standoff=14.0 + 8.0 * courage),
        Gait(rate=2.4),
        Renderable(kind="soldier", tint=(0.72, 0.76, 0.6), seed=seed),
    )


def spawn_ant(pos: Vec3, spitter: bool = False, seed: int = 0) -> int:
    """A giant ant.

    Fast, numerous, and no smarter than a straight line. Spitters trade the
    leap for a ranged glob, which is the only thing in the game that reliably
    punishes hovering.
    """
    if spitter:
        weapon = Weapon(
            damage=7.0,
            speed=30.0,
            cooldown=2.6,
            range=42.0,
            spread=2.0,
            kind="acid",
        )
        tint = (0.55, 0.75, 0.35)
    else:
        weapon = Weapon(damage=9.0, cooldown=1.1, range=2.6, melee=True, kind="bite")
        tint = (0.6, 0.22, 0.16)

    return esper.create_entity(
        Transform(pos=Vec3(pos)),
        Velocity(),
        Body(radius=0.9, height=1.5, gravity=24.0),
        Walker(accel=34.0, max_speed=8.4, friction=12.0, turn_rate=300.0, jump_speed=11.0),
        Intent(),
        Health(maximum=60.0, current=60.0),
        FactionTag(Faction.BUGS),
        weapon,
        AntBrain(spitter=spitter, bite_range=2.6 if not spitter else 34.0),
        Gait(rate=3.1),
        Renderable(kind="ant", tint=tint, seed=seed),
    )


def spawn_building(pos: Vec3, half_x: float, half_y: float, height: float, style: int) -> int:
    """A static box. No health -- the city is scenery, not a target.

    Registration with the collision grid happens in :mod:`worldgen`, because
    the grid wants every box before it is built and a factory has no business
    knowing that.
    """
    return esper.create_entity(
        Transform(pos=Vec3(pos)),
        Building(half_x=half_x, half_y=half_y, height=height, style=style),
        Renderable(kind="building", seed=style),
    )


def spawn_projectile(
    pos: Vec3,
    velocity: Vec3,
    damage: float,
    faction: Faction,
    owner: int,
    kind: str,
    life: float = 3.0,
    radius: float = 0.5,
) -> int:
    """A bullet. Carries no ``Body`` -- it does not walk and it does not fall."""
    return esper.create_entity(
        Transform(pos=Vec3(pos)),
        Velocity(vec=Vec3(velocity)),
        Projectile(damage=damage, faction=faction, life=life, radius=radius, owner=owner),
        Renderable(kind=kind),
    )


def spawn_effect(pos: Vec3, kind: str, life: float = 0.35, scale: float = 1.0) -> int:
    """A purely cosmetic entity: impact spark, dust puff, muzzle bloom.

    Effects are entities rather than a renderer-private particle list so that
    a headless run can count them, which is how the tests check that a hit
    actually registered somewhere visible.
    """
    return esper.create_entity(
        Transform(pos=Vec3(pos)),
        Lifetime(remaining=life),
        Renderable(kind=kind, scale=scale),
    )


def kill(ent: int) -> None:
    """Mark an entity dead if it is not already.

    Two-step death: :class:`~components.Dead` this frame, deletion when its
    timer runs out, so the corpse is on screen for a moment and the score is
    counted exactly once.
    """
    if not esper.has_component(ent, Dead):
        esper.add_component(ent, Dead())
