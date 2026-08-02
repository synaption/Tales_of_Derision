"""The simulation: every processor, and the object that owns them.

This module imports ``esper``, ``math`` and ``panda3d.core.Vec3``. It does not
import ``direct``, it never touches a ``NodePath``, and nothing in it will
open a window. That is the contract that lets ``headlesstest.py`` run the
entire game -- flight, collision, ants, gunfire, waves -- with no GPU in the
room, and it is why the renderer is a separate set of processors bolted on in
:mod:`render`.

Processor order is set by priority, highest first:

===  ==========================  ======================================
90   :class:`IndexProcessor`     rebuild the spatial hash for this frame
85   :class:`AntProcessor`       ant brains write Intent
84   :class:`AllyProcessor`      ally brains write Intent
80   :class:`FlightProcessor`    Intent + Energy -> jetpack velocity
75   :class:`WalkProcessor`      Intent -> ground velocity
70   :class:`WeaponProcessor`    Intent.fire -> projectiles and bites
60   :class:`PhysicsProcessor`   gravity, integration, city collision
50   :class:`ProjectileProcessor`  fly, hit, damage
40   :class:`GaitProcessor`      derive walk-cycle phase from speed
30   :class:`WaveProcessor`      keep the streets stocked with ants
10   :class:`ReaperProcessor`    lifetimes, corpses, deletion
===  ==========================  ======================================

Note that ``esper`` keeps its entity database in module-level globals, so
exactly one :class:`Sim` can be the active one at a time. :class:`Sim` gives
each instance its own named esper World and re-selects it in
:meth:`Sim.activate`, which is enough for tests to build several and step them
in turn.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

import esper
from panda3d.core import Vec3

import entities
from components import (
    AllyBrain,
    AntBrain,
    Body,
    Building,
    Dead,
    Energy,
    Faction,
    FactionTag,
    Flight,
    Gait,
    Health,
    Intent,
    Lifetime,
    Player,
    Projectile,
    Transform,
    Velocity,
    Walker,
    Weapon,
)
from spatialhash import SpatialHash, StaticGrid

# Half-width of the playable city, in metres. Bodies are clamped to it; it is
# also the radius the wave spawner works from.
ARENA = 230.0

# How far a body will be lifted onto a ledge without having to jump. Kerbs and
# rubble, not rooftops.
STEP_UP = 0.7

_world_counter = itertools.count(1)


# --------------------------------------------------------------------------
# small maths helpers
# --------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def heading_vectors(yaw_deg: float) -> tuple[float, float, float, float]:
    """Return ``(forward_x, forward_y, right_x, right_y)`` for a heading.

    Panda3D is Z-up and a heading of 0 faces +Y, with positive heading turning
    counter-clockwise seen from above.
    """
    y = math.radians(yaw_deg)
    sin_y, cos_y = math.sin(y), math.cos(y)
    return -sin_y, cos_y, cos_y, sin_y


def aim_vector(yaw_deg: float, pitch_deg: float) -> Vec3:
    """Unit vector for a yaw/pitch pair, matching :func:`heading_vectors`."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    cp = math.cos(p)
    return Vec3(-math.sin(y) * cp, math.cos(y) * cp, math.sin(p))


def yaw_to(dx: float, dy: float) -> float:
    """Heading in degrees that points along ``(dx, dy)``."""
    return math.degrees(math.atan2(-dx, dy))


def angle_delta(target: float, current: float) -> float:
    """Shortest signed turn from ``current`` to ``target``, in degrees."""
    return (target - current + 180.0) % 360.0 - 180.0


def horizontal_speed(vec: Vec3) -> float:
    return math.hypot(vec.x, vec.y)


# --------------------------------------------------------------------------
# the world object
# --------------------------------------------------------------------------


@dataclass
class Stats:
    """Scoreboard. Read by the HUD, asserted on by the tests."""

    ants_killed: int = 0
    allies_lost: int = 0
    shots_fired: int = 0
    wave: int = 0
    elapsed: float = 0.0


class Sim:
    """Owns one esper World, the collision grid, the RNG and the scoreboard.

    Everything a processor needs that is not a component hangs off here, and
    processors are handed the ``Sim`` at construction. That keeps the shared
    state explicit -- there are no module-level singletons in the simulation.
    """

    def __init__(self, seed: int = 1234) -> None:
        self.name = f"cramigula-{next(_world_counter)}"
        self.rng = random.Random(seed)
        self.seed = seed
        self.grid = StaticGrid(cell=16.0)
        self.hash = SpatialHash(cell=12.0)
        # Rebuilt every frame by IndexProcessor: live combatants per side.
        self.sides: dict[Faction, set[int]] = {Faction.EDF: set(), Faction.BUGS: set()}
        self.stats = Stats()
        self.player: int = -1
        self.ant_cap = 46
        self.wave_interval = 11.0
        self._wave_timer = 3.0

        self.activate()
        esper.clear_database()
        self._install_processors()

    # -- lifecycle ---------------------------------------------------------

    def activate(self) -> None:
        """Make this Sim's World the one esper's module functions act on."""
        if esper.current_world != self.name:
            esper.switch_world(self.name)

    def _install_processors(self) -> None:
        esper.add_processor(IndexProcessor(self), priority=90)
        esper.add_processor(AntProcessor(self), priority=85)
        esper.add_processor(AllyProcessor(self), priority=84)
        esper.add_processor(FlightProcessor(self), priority=80)
        esper.add_processor(WalkProcessor(self), priority=75)
        esper.add_processor(WeaponProcessor(self), priority=70)
        esper.add_processor(PhysicsProcessor(self), priority=60)
        esper.add_processor(ProjectileProcessor(self), priority=50)
        esper.add_processor(GaitProcessor(self), priority=40)
        esper.add_processor(WaveProcessor(self), priority=30)
        esper.add_processor(ReaperProcessor(self), priority=10)

    def step(self, dt: float) -> None:
        """Advance one frame.

        ``dt`` is clamped: a stall that hands us a half-second frame would
        otherwise teleport bodies through walls, and a paused-then-resumed
        window does exactly that.
        """
        self.activate()
        dt = clamp(dt, 0.0, 1.0 / 20.0)
        self.stats.elapsed += dt
        esper.process(dt)

    # -- queries -----------------------------------------------------------

    def enemies_of(self, side: Faction) -> set[int]:
        return self.sides[Faction.BUGS if side is Faction.EDF else Faction.EDF]

    def ground_height(self, x: float, y: float) -> float:
        """Street level, or the roof of whatever is under ``(x, y)``."""
        return self.grid.height_at(x, y)

    def register_building(self, ent: int, transform: Transform, building: Building) -> None:
        """Add a building's footprint to the static collision grid."""
        self.grid.add_box(
            ent,
            transform.pos.x - building.half_x,
            transform.pos.y - building.half_y,
            transform.pos.x + building.half_x,
            transform.pos.y + building.half_y,
            building.height,
        )


# --------------------------------------------------------------------------
# indexing
# --------------------------------------------------------------------------


class IndexProcessor(esper.Processor):
    """Rebuilds the per-frame spatial hash and the faction membership sets.

    Rebuilding from scratch beats maintaining it incrementally at this entity
    count -- a few hundred inserts is cheaper than the bookkeeping to keep a
    dirty-list correct, and it cannot go stale.
    """

    def __init__(self, sim: Sim) -> None:
        self.sim = sim

    def process(self, dt: float) -> None:
        sim = self.sim
        sim.hash.clear()
        for side in sim.sides.values():
            side.clear()
        for ent, (transform, tag, _health) in esper.get_components(Transform, FactionTag, Health):
            if esper.has_component(ent, Dead):
                continue  # corpses are neither targets nor obstacles
            sim.hash.insert(ent, transform.pos.x, transform.pos.y)
            sim.sides[tag.side].add(ent)


# --------------------------------------------------------------------------
# brains -- these only ever write Intent
# --------------------------------------------------------------------------


class AntProcessor(esper.Processor):
    """Walk at the nearest EDF thing; bite it, spit at it, or leap at it.

    There is no pathfinding. An ant that walks into a tower slides along it,
    because :class:`PhysicsProcessor` resolves the penetration along the wall
    normal and leaves the tangent alone. It looks like a swarm breaking around
    a building and costs nothing.
    """

    def __init__(self, sim: Sim) -> None:
        self.sim = sim

    def process(self, dt: float) -> None:
        sim = self.sim
        for ent, (transform, intent, brain, walker) in esper.get_components(
            Transform, Intent, AntBrain, Walker
        ):
            intent.clear()
            if esper.has_component(ent, Dead):
                continue

            brain.retarget -= dt
            brain.leap_cd = max(0.0, brain.leap_cd - dt)
            if brain.retarget <= 0.0 or not esper.entity_exists(brain.target):
                brain.retarget = 0.35 + sim.rng.random() * 0.4
                brain.target, _ = sim.hash.nearest(
                    transform.pos.x, transform.pos.y, brain.sight, sim.sides[Faction.EDF]
                )

            if brain.target < 0 or not esper.entity_exists(brain.target):
                continue

            goal = esper.component_for_entity(brain.target, Transform).pos
            dx, dy = goal.x - transform.pos.x, goal.y - transform.pos.y
            dz = goal.z - transform.pos.z
            flat = math.hypot(dx, dy)
            if flat < 1e-4:
                continue

            intent.aim_yaw = yaw_to(dx, dy)
            intent.aim_pitch = math.degrees(math.atan2(dz + 0.9, max(flat, 0.5)))

            if flat > brain.bite_range * 0.75:
                intent.move_y = 1.0
            if flat <= brain.bite_range:
                intent.fire = True

            # A biter that has been kept just out of reach leaps. This is the
            # only answer ants have to a hovering Wing Diver, so it is on a
            # generous cooldown and needs the target to be reachably low.
            if (
                not brain.spitter
                and walker.jump_speed > 0.0
                and brain.leap_cd <= 0.0
                and 4.0 < flat < 16.0
                and dz < 9.0
            ):
                intent.thrust = True
                brain.leap_cd = 2.2 + sim.rng.random() * 1.6


class AllyProcessor(esper.Processor):
    """Advance to a standoff distance, shoot, and drift home when it is quiet.

    Two behaviours, chosen by distance to the nearest ant: close in if the
    fight is far away, back off if it is too close. The dead zone between the
    two is what stops a squad from oscillating on the spot.
    """

    def __init__(self, sim: Sim) -> None:
        self.sim = sim

    def process(self, dt: float) -> None:
        sim = self.sim
        for ent, (transform, intent, brain) in esper.get_components(Transform, Intent, AllyBrain):
            intent.clear()
            if esper.has_component(ent, Dead):
                continue

            brain.retarget -= dt
            if brain.retarget <= 0.0 or not esper.entity_exists(brain.target):
                brain.retarget = 0.4 + sim.rng.random() * 0.5
                brain.target, _ = sim.hash.nearest(
                    transform.pos.x, transform.pos.y, brain.sight, sim.sides[Faction.BUGS]
                )

            if brain.target >= 0 and esper.entity_exists(brain.target):
                goal = esper.component_for_entity(brain.target, Transform).pos
                dx, dy = goal.x - transform.pos.x, goal.y - transform.pos.y
                dz = goal.z - transform.pos.z
                flat = math.hypot(dx, dy)
                intent.aim_yaw = yaw_to(dx, dy)
                intent.aim_pitch = math.degrees(math.atan2(dz + 0.7, max(flat, 0.5)))
                if flat < brain.sight:
                    intent.fire = True
                if flat > brain.standoff * 1.25:
                    intent.move_y = 1.0
                elif flat < brain.standoff * 0.6:
                    intent.move_y = -1.0  # too close, give ground while firing
                else:
                    # Strafe, so a firing line does not look like a bus queue.
                    intent.move_x = 0.35 if (ent % 2) else -0.35
                continue

            # Nothing in sight: wander back toward the rally point.
            dx, dy = brain.rally.x - transform.pos.x, brain.rally.y - transform.pos.y
            if math.hypot(dx, dy) > 4.0:
                intent.aim_yaw = yaw_to(dx, dy)
                intent.move_y = 0.6


# --------------------------------------------------------------------------
# movement
# --------------------------------------------------------------------------


class FlightProcessor(esper.Processor):
    """The Wing Diver movement model, and the reason this project exists.

    Four coupled rules, in the order they are applied:

    1. **Dash.** A flat energy charge buys a fixed-speed horizontal burst that
       ignores gravity for its duration. It is an impulse, not a state, so the
       speed it leaves behind is yours to keep or waste.
    2. **Thrust.** Held, it drains continuously and adds upward *acceleration*
       -- not an upward velocity -- capped at ``rise_max``. Because it is an
       acceleration, tapping it gives a hop and holding it gives a climb, and
       falling momentum has to be paid off before you rise.
    3. **Air control.** Airborne horizontal acceleration is high but the speed
       cap is soft: you may exceed it (a dash does), you just cannot thrust
       past it. Let go and drag bleeds you back down, which is the glide.
    4. **Recharge and overheat.** Energy only refills while not thrusting, and
       four times faster with feet on the ground. Touch exactly zero and the
       meter latches ``empty``: no flight, no dash, and a recharge at 55% rate
       that does not release until the bar is *completely* full. Nearly all
       the skill in the class is in never letting that latch close.
    """

    def __init__(self, sim: Sim) -> None:
        self.sim = sim

    def process(self, dt: float) -> None:
        for ent, (transform, vel, body, energy, flight, intent) in esper.get_components(
            Transform, Velocity, Body, Energy, Flight, Intent
        ):
            dead = esper.has_component(ent, Dead)
            flight.dash_cd = max(0.0, flight.dash_cd - dt)
            flight.dash_timer = max(0.0, flight.dash_timer - dt)
            flight.thrusting = False
            if dead:
                continue

            fwd_x, fwd_y, right_x, right_y = heading_vectors(intent.aim_yaw)
            wish_x = right_x * intent.move_x + fwd_x * intent.move_y
            wish_y = right_y * intent.move_x + fwd_y * intent.move_y
            wish_len = math.hypot(wish_x, wish_y)
            if wish_len > 1.0:
                wish_x /= wish_len
                wish_y /= wish_len
                wish_len = 1.0

            # 1. dash
            if intent.dash and flight.dash_cd <= 0.0 and not energy.empty:
                if wish_len < 1e-3:  # no stick input: dash where you are looking
                    wish_x, wish_y, wish_len = fwd_x, fwd_y, 1.0
                if energy.spend(energy.cost_dash):
                    flight.dash_dir = Vec3(wish_x / wish_len, wish_y / wish_len, 0.0)
                    flight.dash_timer = flight.dash_time
                    flight.dash_cd = flight.dash_cooldown
                    vel.vec.x = flight.dash_dir.x * flight.dash_speed
                    vel.vec.y = flight.dash_dir.y * flight.dash_speed
                    # A ground dash skims rather than scrapes.
                    vel.vec.z = max(vel.vec.z, 2.5)
                    body.grounded = False

            if flight.dash_timer > 0.0:
                # Hold the burst. Gravity is suspended for the duration by
                # PhysicsProcessor, which reads dash_timer.
                vel.vec.x = flight.dash_dir.x * flight.dash_speed
                vel.vec.y = flight.dash_dir.y * flight.dash_speed
                transform.heading = intent.aim_yaw
                transform.roll = -clamp(intent.move_x, -1.0, 1.0) * 24.0
                continue

            # 2. thrust -- pay first, and take the last drop if that is all
            # that is left, which is what arms the overheat.
            if intent.thrust and not energy.empty and energy.current > 0.0:
                cost = min(energy.drain_thrust * dt, energy.current)
                if energy.spend(cost):
                    flight.thrusting = True
                    vel.vec.z = min(vel.vec.z + flight.thrust_accel * dt, flight.rise_max)
                    if body.grounded:
                        body.grounded = False
                        vel.vec.z = max(vel.vec.z, 2.0)

            # 3. air / ground control
            if body.grounded and not flight.thrusting:
                # On foot she is an ordinary soldier, and a slow one.
                target_x = wish_x * flight.air_max * 0.45
                target_y = wish_y * flight.air_max * 0.45
                rate = clamp(flight.air_accel * 1.6 * dt, 0.0, 1.0)
                vel.vec.x += (target_x - vel.vec.x) * rate
                vel.vec.y += (target_y - vel.vec.y) * rate
            else:
                speed = horizontal_speed(vel.vec)
                if wish_len > 1e-3:
                    nvx = vel.vec.x + wish_x * flight.air_accel * dt
                    nvy = vel.vec.y + wish_y * flight.air_accel * dt
                    nspeed = math.hypot(nvx, nvy)
                    # Soft cap: thrusting cannot push past air_max, but a dash
                    # that already did is allowed to keep what it earned.
                    ceiling = max(flight.air_max, speed)
                    if nspeed > ceiling:
                        nvx *= ceiling / nspeed
                        nvy *= ceiling / nspeed
                    vel.vec.x, vel.vec.y = nvx, nvy
                else:
                    decay = math.exp(-flight.glide_drag * dt)
                    vel.vec.x *= decay
                    vel.vec.y *= decay

            # 4. recharge
            if not flight.thrusting:
                rate = energy.regen_ground if body.grounded else energy.regen_air
                if energy.empty:
                    rate *= energy.empty_penalty
                energy.current += rate * dt
                if energy.current >= energy.maximum:
                    energy.current = energy.maximum
                    energy.empty = False  # the latch only opens at full

            transform.heading = intent.aim_yaw
            # Cosmetic bank, eased so it does not snap when the stick centres.
            want_roll = -clamp(intent.move_x, -1.0, 1.0) * (14.0 if not body.grounded else 0.0)
            transform.roll += (want_roll - transform.roll) * clamp(8.0 * dt, 0.0, 1.0)
            transform.pitch = clamp(-vel.vec.z * 0.6, -18.0, 18.0)


class WalkProcessor(esper.Processor):
    """Ground locomotion for ants and grunts: accelerate, turn, sometimes leap."""

    def __init__(self, sim: Sim) -> None:
        self.sim = sim

    def process(self, dt: float) -> None:
        for ent, (transform, vel, body, walker, intent) in esper.get_components(
            Transform, Velocity, Body, Walker, Intent
        ):
            if esper.has_component(ent, Dead):
                continue

            fwd_x, fwd_y, right_x, right_y = heading_vectors(intent.aim_yaw)
            wish_x = right_x * intent.move_x + fwd_x * intent.move_y
            wish_y = right_y * intent.move_x + fwd_y * intent.move_y
            wish_len = math.hypot(wish_x, wish_y)
            if wish_len > 1.0:
                wish_x /= wish_len
                wish_y /= wish_len

            if body.grounded:
                if wish_len > 1e-3:
                    target_x = wish_x * walker.max_speed
                    target_y = wish_y * walker.max_speed
                    rate = clamp(walker.accel * dt / max(walker.max_speed, 0.1), 0.0, 1.0)
                    vel.vec.x += (target_x - vel.vec.x) * rate
                    vel.vec.y += (target_y - vel.vec.y) * rate
                else:
                    decay = math.exp(-walker.friction * dt)
                    vel.vec.x *= decay
                    vel.vec.y *= decay

                if intent.thrust and walker.jump_speed > 0.0:
                    vel.vec.z = walker.jump_speed
                    body.grounded = False
            else:
                # Ants steer a little in mid-leap; it makes them land on you.
                vel.vec.x += wish_x * walker.accel * 0.22 * dt
                vel.vec.y += wish_y * walker.accel * 0.22 * dt

            # Face the way we are actually moving, at a finite turn rate.
            speed = horizontal_speed(vel.vec)
            if speed > 0.25:
                want = yaw_to(vel.vec.x, vel.vec.y)
            elif wish_len > 1e-3:
                want = yaw_to(wish_x, wish_y)
            else:
                want = transform.heading
            step = walker.turn_rate * dt
            transform.heading += clamp(angle_delta(want, transform.heading), -step, step)


# --------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------


class PhysicsProcessor(esper.Processor):
    """Gravity, integration and collision against the city.

    Horizontal and vertical are resolved separately, in that order, which is
    the cheap trick that makes stepping onto kerbs and landing on roofs both
    fall out of the same code:

    * horizontally, a body is a circle pushed out of any box whose roof it is
      *below*; the push is along the box normal so the tangential component of
      velocity survives and things slide;
    * vertically, the support height under the body is the tallest roof that
      is at or below the feet, so "the ground" and "that office block's roof"
      are the same case.
    """

    def __init__(self, sim: Sim) -> None:
        self.sim = sim

    def process(self, dt: float) -> None:
        grid = self.sim.grid
        for ent, (transform, vel, body) in esper.get_components(Transform, Velocity, Body):
            flight = esper.try_component(ent, Flight)
            dashing = flight is not None and flight.dash_timer > 0.0

            if not dashing and not body.grounded:
                vel.vec.z -= body.gravity * dt
                fall_max = flight.fall_max if flight else 55.0
                if vel.vec.z < -fall_max:
                    vel.vec.z = -fall_max
            elif dashing:
                vel.vec.z = max(vel.vec.z - body.gravity * 0.15 * dt, -2.0)

            pos = transform.pos
            new_x = pos.x + vel.vec.x * dt
            new_y = pos.y + vel.vec.y * dt
            radius = body.radius

            # -- horizontal: push the circle out of every wall it is inside --
            for _ident, x0, y0, x1, y1, top in grid.query_aabb(
                new_x - radius, new_y - radius, new_x + radius, new_y + radius
            ):
                if pos.z >= top - STEP_UP:
                    continue  # we are on or above this roof; it is floor, not wall
                near_x = clamp(new_x, x0, x1)
                near_y = clamp(new_y, y0, y1)
                dx, dy = new_x - near_x, new_y - near_y
                dist2 = dx * dx + dy * dy
                if dist2 >= radius * radius:
                    continue
                if dist2 > 1e-9:
                    dist = math.sqrt(dist2)
                    nx, ny = dx / dist, dy / dist
                    push = radius - dist
                else:
                    # Centre is inside the box: escape through the nearest face.
                    left, right = new_x - x0, x1 - new_x
                    down, up = new_y - y0, y1 - new_y
                    smallest = min(left, right, down, up)
                    nx, ny = (
                        (-1.0, 0.0)
                        if smallest == left
                        else (1.0, 0.0)
                        if smallest == right
                        else (0.0, -1.0)
                        if smallest == down
                        else (0.0, 1.0)
                    )
                    push = smallest + radius
                new_x += nx * push
                new_y += ny * push
                into = vel.vec.x * nx + vel.vec.y * ny
                if into < 0.0:
                    vel.vec.x -= into * nx
                    vel.vec.y -= into * ny

            new_x = clamp(new_x, -ARENA, ARENA)
            new_y = clamp(new_y, -ARENA, ARENA)

            # -- vertical: find what is under us, then land on it or fall past --
            support = 0.0
            for _ident, x0, y0, x1, y1, top in grid.query_aabb(
                new_x - radius, new_y - radius, new_x + radius, new_y + radius
            ):
                if top <= pos.z + STEP_UP and top > support:
                    support = top

            new_z = pos.z + vel.vec.z * dt
            body.landed_speed = 0.0
            if new_z <= support:
                if not body.grounded and vel.vec.z < 0.0:
                    body.landed_speed = -vel.vec.z
                new_z = support
                vel.vec.z = 0.0
                body.grounded = True
                body.ground_z = support
                body.airborne_time = 0.0
            else:
                if body.grounded and new_z > support + 0.05:
                    body.grounded = False
                if not body.grounded:
                    body.airborne_time += dt

            pos.set(new_x, new_y, new_z)


class GaitProcessor(esper.Processor):
    """Advance the walk cycle from distance travelled, not from wall time.

    Phase is driven by metres covered so legs and ground speed always agree,
    and the amplitude eases toward the current speed so that stopping folds
    the cycle down instead of freezing it mid-stride.
    """

    def __init__(self, sim: Sim) -> None:
        self.sim = sim

    def process(self, dt: float) -> None:
        for ent, (vel, gait, body) in esper.get_components(Velocity, Gait, Body):
            speed = horizontal_speed(vel.vec)
            if body.grounded:
                gait.phase = (gait.phase + speed * gait.rate * dt) % 1.0
                want = clamp(speed / 4.0, 0.0, 1.0)
            else:
                want = 0.0  # legs tuck in mid-air
            gait.amplitude += (want - gait.amplitude) * clamp(9.0 * dt, 0.0, 1.0)


# --------------------------------------------------------------------------
# combat
# --------------------------------------------------------------------------


class WeaponProcessor(esper.Processor):
    """Turns ``Intent.fire`` into damage, by projectile or by bite.

    Energy weapons check the same meter the jetpack uses, so a shot taken in
    the air is a shot of altitude given up. That coupling is deliberate: it is
    the whole reason the Wing Diver plays differently from a man with a gun.
    """

    def __init__(self, sim: Sim) -> None:
        self.sim = sim

    def process(self, dt: float) -> None:
        sim = self.sim
        for ent, (transform, intent, weapon, tag) in esper.get_components(
            Transform, Intent, Weapon, FactionTag
        ):
            weapon.timer = max(0.0, weapon.timer - dt)
            weapon.muzzle_flash = max(0.0, weapon.muzzle_flash - dt)
            if esper.has_component(ent, Dead) or not intent.fire or weapon.timer > 0.0:
                continue

            energy = esper.try_component(ent, Energy)
            if weapon.energy_cost > 0.0:
                if energy is None or not energy.spend(weapon.energy_cost):
                    continue

            weapon.timer = weapon.cooldown
            weapon.muzzle_flash = 0.06
            sim.stats.shots_fired += 1

            body = esper.try_component(ent, Body)
            eye = transform.pos + Vec3(0, 0, (body.height if body else 1.6) * 0.72)

            if weapon.melee:
                self._bite(sim, ent, transform, weapon, tag)
                continue

            yaw = intent.aim_yaw + sim.rng.uniform(-weapon.spread, weapon.spread)
            pitch = intent.aim_pitch + sim.rng.uniform(-weapon.spread, weapon.spread)
            direction = aim_vector(yaw, pitch)
            entities.spawn_projectile(
                eye + direction * 0.9,
                direction * weapon.speed,
                weapon.damage,
                tag.side,
                ent,
                weapon.kind,
                life=weapon.range / max(weapon.speed, 1.0),
                radius=0.6 if weapon.kind == "acid" else 0.35,
            )

    @staticmethod
    def _bite(sim: Sim, ent: int, transform: Transform, weapon: Weapon, tag: FactionTag) -> None:
        """Melee: hit the nearest enemy inside the arc, or hit nothing.

        The vertical check is what keeps a grounded ant from biting a Wing
        Diver hovering directly overhead -- the reach is a short cylinder, not
        a sphere on the map.
        """
        enemies = sim.enemies_of(tag.side)
        target, dist2 = sim.hash.nearest(
            transform.pos.x, transform.pos.y, weapon.range, enemies
        )
        if target < 0:
            return
        other = esper.component_for_entity(target, Transform)
        if abs(other.pos.z - transform.pos.z) > 2.4:
            return
        health = esper.try_component(target, Health)
        if health is None:
            return
        health.damage(weapon.damage)
        entities.spawn_effect(other.pos + Vec3(0, 0, 1.0), "spark", life=0.2, scale=0.6)
        if health.current <= 0.0:
            _score_kill(sim, target)
        _ = dist2


class ProjectileProcessor(esper.Processor):
    """Fly the bullets, and stop them at the first thing they should stop at.

    Movement is substepped along the frame's segment so that a 95 m/s lance
    cannot pass through a 1.8m ant between two frames -- at sixty frames a
    second it would otherwise cover 1.6m per test.
    """

    def __init__(self, sim: Sim) -> None:
        self.sim = sim

    def process(self, dt: float) -> None:
        sim = self.sim
        grid = sim.grid
        for ent, (transform, vel, proj) in esper.get_components(Transform, Velocity, Projectile):
            proj.life -= dt
            if proj.life <= 0.0:
                esper.delete_entity(ent)
                continue

            travel = vel.vec.length() * dt
            steps = max(1, min(8, int(travel / 0.7) + 1))
            step_dt = dt / steps
            enemies = sim.enemies_of(proj.faction)
            struck = False

            for _ in range(steps):
                transform.pos += vel.vec * step_dt

                if transform.pos.z <= 0.0:
                    entities.spawn_effect(
                        Vec3(transform.pos.x, transform.pos.y, 0.05), "spark", life=0.18
                    )
                    struck = True
                    break

                if self._hits_building(grid, transform.pos):
                    entities.spawn_effect(transform.pos, "spark", life=0.22)
                    struck = True
                    break

                hit = self._hits_body(sim, transform.pos, proj, enemies)
                if hit >= 0:
                    health = esper.component_for_entity(hit, Health)
                    health.damage(proj.damage)
                    entities.spawn_effect(transform.pos, "blood", life=0.25, scale=1.2)
                    if health.current <= 0.0:
                        _score_kill(sim, hit)
                    struck = True
                    break

            if struck:
                esper.delete_entity(ent)

    @staticmethod
    def _hits_building(grid: StaticGrid, pos: Vec3) -> bool:
        for _ident, x0, y0, x1, y1, top in grid.query_aabb(pos.x, pos.y, pos.x, pos.y):
            if x0 <= pos.x <= x1 and y0 <= pos.y <= y1 and pos.z <= top:
                return True
        return False

    @staticmethod
    def _hits_body(sim: Sim, pos: Vec3, proj: Projectile, enemies: set[int]) -> int:
        """First enemy cylinder containing ``pos``, or -1.

        The hash is queried at a generous radius and the real test is done on
        the candidates, because the hash only knows about ground position and
        a body has height.
        """
        best, best_d2 = -1, float("inf")
        for candidate, d2 in sim.hash.query_radius(pos.x, pos.y, proj.radius + 2.0):
            if candidate not in enemies or candidate == proj.owner:
                continue
            other = esper.component_for_entity(candidate, Transform)
            body = esper.try_component(candidate, Body)
            if body is None:
                continue
            reach = body.radius + proj.radius
            if d2 > reach * reach:
                continue
            if not (other.pos.z - proj.radius <= pos.z <= other.pos.z + body.height + proj.radius):
                continue
            if d2 < best_d2:
                best, best_d2 = candidate, d2
        return best


def _score_kill(sim: Sim, ent: int) -> None:
    """Count a death exactly once and start the corpse timer."""
    if esper.has_component(ent, Dead):
        return
    tag = esper.try_component(ent, FactionTag)
    if tag is not None:
        if tag.side is Faction.BUGS:
            sim.stats.ants_killed += 1
        elif not esper.has_component(ent, Player):
            sim.stats.allies_lost += 1
    entities.kill(ent)


# --------------------------------------------------------------------------
# waves and cleanup
# --------------------------------------------------------------------------


class WaveProcessor(esper.Processor):
    """Keeps ants coming, from off the edge of the player's attention.

    Spawns land on a ring 70-110m out, snapped to street level so nothing
    materialises inside a building, and the population is capped so the frame
    time stays flat however long you survive.
    """

    def __init__(self, sim: Sim) -> None:
        self.sim = sim

    def process(self, dt: float) -> None:
        sim = self.sim
        sim._wave_timer -= dt
        if sim._wave_timer > 0.0:
            return
        sim._wave_timer = sim.wave_interval

        # The wave number is a difficulty clock, so it advances on schedule
        # whether or not there is room to spawn into. Only the head count is
        # capped -- otherwise a player who stops killing ants also stops the
        # game getting harder, which is exactly backwards.
        sim.stats.wave += 1

        alive = len(sim.sides[Faction.BUGS])
        room = sim.ant_cap - alive
        if room <= 0:
            return

        count = min(room, 6 + sim.stats.wave)
        origin = Vec3(0, 0, 0)
        if sim.player >= 0 and esper.entity_exists(sim.player):
            origin = Vec3(esper.component_for_entity(sim.player, Transform).pos)

        for _ in range(count):
            angle = sim.rng.uniform(0.0, math.tau)
            dist = sim.rng.uniform(70.0, 110.0)
            x = clamp(origin.x + math.cos(angle) * dist, -ARENA + 6, ARENA - 6)
            y = clamp(origin.y + math.sin(angle) * dist, -ARENA + 6, ARENA - 6)
            spitter = sim.rng.random() < 0.22
            entities.spawn_ant(
                Vec3(x, y, sim.ground_height(x, y)),
                spitter=spitter,
                seed=sim.rng.randrange(1000),
            )


class ReaperProcessor(esper.Processor):
    """Runs the corpse timers and the effect lifetimes, and deletes.

    Corpses keep falling while they rot -- a bug shot out of a leap should
    finish the arc -- so the physics pass still runs on them; only the brain
    and the gun are switched off, by the ``Dead`` checks upstream.
    """

    def __init__(self, sim: Sim) -> None:
        self.sim = sim

    def process(self, dt: float) -> None:
        for ent, (life,) in esper.get_components(Lifetime):
            life.remaining -= dt
            if life.remaining <= 0.0:
                esper.delete_entity(ent)

        for ent, (dead,) in esper.get_components(Dead):
            dead.timer -= dt
            if dead.timer <= 0.0:
                transform = esper.try_component(ent, Transform)
                if transform is not None:
                    entities.spawn_effect(transform.pos, "dust", life=0.5, scale=1.4)
                esper.delete_entity(ent)

        for _ent, (health,) in esper.get_components(Health):
            health.hurt_flash = max(0.0, health.hurt_flash - dt)
