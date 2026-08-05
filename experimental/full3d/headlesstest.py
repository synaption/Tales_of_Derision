"""Headless tests for Cramigula. No window is ever opened.

    python3 experimental/full3d/headlesstest.py
    python3 experimental/full3d/headlesstest.py --no-render   # sim only, no GL

Two halves, in the order they run:

**The simulation**, with no Panda3D window and no GPU at all. Flight, the
overheat latch, gliding, city collision, rooftop landings, ant and ally
behaviour, projectile damage and determinism are all asserted against the
dataclasses in :mod:`components`. This half is why :mod:`sim` is forbidden
from importing anything from ``direct``.

**The renderer**, against an offscreen buffer via ``window-type offscreen``,
so nothing appears on the desktop. It checks that the shader compiles, that
the PS1 pipeline really is producing a 320x240 image, that the dither is
quantising to five bits a channel, and that the scene graph tracks entity
births and deaths. It also drops ``cramigula_headless.png``: a six-panel
contact sheet so the art can be eyeballed without launching the game.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import esper
from panda3d.core import Vec3, loadPrcFileData

import entities
import worldgen
from components import (
    AntBrain,
    Body,
    Building,
    Dead,
    Energy,
    Faction,
    Flight,
    Gait,
    Health,
    Intent,
    Projectile,
    Renderable,
    Transform,
    Velocity,
    Weapon,
)
from sim import ARENA, Sim, aim_vector, angle_delta, heading_vectors, yaw_to
from spatialhash import SpatialHash, StaticGrid

SHEET = Path(__file__).resolve().parent / "cramigula_headless.png"
FRAME = 1.0 / 60.0

_passed: list[str] = []
_failed: list[str] = []


def check(name: str, fn) -> None:
    """Run one test, catch its assertion, keep going. Prints as it goes."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- a test runner wants them all
        _failed.append(f"{name}: {exc}")
        print(f"  FAIL  {name}: {exc}")
    else:
        _passed.append(name)
        print(f"  ok    {name}")


def step(sim: Sim, frames: int, **intent_fields) -> None:
    """Advance ``frames`` frames, driving the player's intent each one.

    The intent is cleared and rewritten every frame, exactly as
    :meth:`cramigula.Cramigula._write_intent` does -- otherwise a ``fire=True``
    handed to one call would still be held down in the next one, which is not
    how a keyboard behaves and not what any of these tests mean.
    """
    intent = esper.component_for_entity(sim.player, Intent)
    for _ in range(frames):
        intent.clear()
        for key, value in intent_fields.items():
            setattr(intent, key, value)
        sim.step(FRAME)


def bare_sim(seed: int = 7) -> Sim:
    """A Sim with a player on empty tarmac -- no city, no ants, no allies."""
    sim = Sim(seed=seed)
    sim.player = entities.spawn_player(Vec3(0, 0, 0))
    return sim


def plain_body(pos: Vec3, radius: float = 0.6) -> int:
    """A bare physics entity: no brain, no legs, no jetpack.

    Used where a test wants to exercise :class:`~sim.PhysicsProcessor` alone.
    Driving the player instead would be testing the flight model at the same
    time, since it rewrites velocity from intent every frame.
    """
    return esper.create_entity(
        Transform(pos=Vec3(pos)),
        Velocity(),
        Body(radius=radius, height=1.8, gravity=22.0, grounded=False),
    )


# --------------------------------------------------------------------------
# maths and data structures
# --------------------------------------------------------------------------


def test_heading_conventions() -> None:
    """Heading 0 faces +Y and positive heading turns left, as Panda3D does."""
    fx, fy, rx, ry = heading_vectors(0.0)
    assert abs(fx) < 1e-9 and abs(fy - 1.0) < 1e-9, f"forward at h=0 was ({fx},{fy})"
    assert abs(rx - 1.0) < 1e-9 and abs(ry) < 1e-9, f"right at h=0 was ({rx},{ry})"

    fx, fy, _, _ = heading_vectors(90.0)
    assert abs(fx + 1.0) < 1e-6, "h=90 should face -X"

    assert abs(yaw_to(0.0, 1.0)) < 1e-9
    assert abs(yaw_to(-1.0, 0.0) - 90.0) < 1e-6
    assert abs(angle_delta(-179.0, 179.0) - 2.0) < 1e-6, "turn should take the short way"

    up = aim_vector(0.0, 90.0)
    assert up.z > 0.999, "pitch +90 should point straight up"


def test_spatial_hash_finds_the_nearest() -> None:
    """A rebuilt hash answers radius and nearest queries, and filters."""
    hashed = SpatialHash(cell=5.0)
    for i in range(40):
        hashed.insert(i, math.cos(i) * 30.0, math.sin(i) * 30.0)
    hashed.insert(999, 1.0, 1.0)

    near = dict(hashed.query_radius(0.0, 0.0, 3.0))
    assert set(near) == {999}, f"radius query returned {set(near)}"

    ident, dist2 = hashed.nearest(0.0, 0.0, 100.0)
    assert ident == 999, "nearest should be the point at (1,1)"
    assert abs(dist2 - 2.0) < 1e-9

    ident, _ = hashed.nearest(0.0, 0.0, 100.0, accept={3, 4})
    assert ident in (3, 4), "the accept filter should exclude everything else"

    hashed.clear()
    assert hashed.nearest(0.0, 0.0, 100.0)[0] == -1, "a cleared hash finds nothing"


def test_static_grid_reports_roof_heights() -> None:
    grid = StaticGrid(cell=8.0)
    grid.add_box(1, -10.0, -10.0, 10.0, 10.0, 25.0)
    grid.add_box(2, 40.0, 40.0, 50.0, 50.0, 9.0)
    assert grid.height_at(0.0, 0.0) == 25.0
    assert grid.height_at(45.0, 45.0) == 9.0
    assert grid.height_at(100.0, 100.0) == 0.0, "open street is at zero"
    found = {box[0] for box in grid.query_aabb(-11.0, -11.0, -9.0, -9.0)}
    assert 1 in found and 2 not in found, f"aabb query returned {found}"


# --------------------------------------------------------------------------
# the Wing Diver
# --------------------------------------------------------------------------


def test_thrust_lifts_and_drains() -> None:
    """Holding thrust climbs, and costs the advertised energy per second."""
    sim = bare_sim()
    energy = esper.component_for_entity(sim.player, Energy)
    transform = esper.component_for_entity(sim.player, Transform)
    body = esper.component_for_entity(sim.player, Body)

    step(sim, 60, thrust=True)  # one second

    assert transform.pos.z > 4.0, f"one second of thrust only reached {transform.pos.z:.2f}m"
    assert not body.grounded, "should be airborne while thrusting"
    expected = 100.0 - energy.drain_thrust
    assert abs(energy.current - expected) < 1.5, (
        f"one second of thrust cost {100.0 - energy.current:.1f}, expected {energy.drain_thrust}"
    )


def test_thrust_is_acceleration_not_a_set_velocity() -> None:
    """A falling diver has to pay off her momentum before she rises.

    This is the difference between EDF flight and a lift, and it is entirely
    a consequence of thrust adding to ``vel.z`` rather than assigning it.
    """
    sim = bare_sim()
    transform = esper.component_for_entity(sim.player, Transform)
    vel = esper.component_for_entity(sim.player, Velocity)

    transform.pos.z = 60.0
    esper.component_for_entity(sim.player, Body).grounded = False
    step(sim, 45)  # fall for three quarters of a second
    falling = vel.vec.z
    assert falling < -10.0, f"should be falling hard, was {falling:.1f} m/s"

    step(sim, 6, thrust=True)  # a tenth of a second of thrust
    assert vel.vec.z > falling, "thrust must reduce the fall"
    assert vel.vec.z < 0.0, "a tap should not reverse a hard fall instantly"


def test_rise_speed_is_capped() -> None:
    sim = bare_sim()
    flight = esper.component_for_entity(sim.player, Flight)
    vel = esper.component_for_entity(sim.player, Velocity)
    step(sim, 90, thrust=True)
    assert vel.vec.z <= flight.rise_max + 1e-6, f"climb hit {vel.vec.z:.1f} m/s"


def test_overheat_latches_at_zero_and_only_clears_when_full() -> None:
    """The rule the whole class is built on.

    Zero locks out flight, the recharge runs at ``empty_penalty`` rate, and
    the lock does not open at 50% or at 99% -- only at the top.
    """
    sim = bare_sim()
    energy = esper.component_for_entity(sim.player, Energy)
    transform = esper.component_for_entity(sim.player, Transform)
    body = esper.component_for_entity(sim.player, Body)

    # Hold thrust until the meter latches. It should take about 100/26 = 3.8s,
    # and the frame it latches on is the frame it reads exactly zero.
    frames = 0
    while not energy.empty and frames < 60 * 8:
        step(sim, 1, thrust=True)
        frames += 1
    assert energy.empty, "holding thrust never emptied the meter"
    assert energy.current == 0.0, f"latched at {energy.current:.2f} rather than zero"
    assert 3.0 < frames / 60.0 < 4.5, f"took {frames / 60.0:.2f}s to drain, expected ~3.8s"

    # Fall back to the street, then try to take off again mid-recharge.
    while not body.grounded:
        step(sim, 1, thrust=True)
    height_on_landing = transform.pos.z
    step(sim, 30, thrust=True)
    assert 0.0 < energy.current < energy.maximum, "should be part-way recharged"
    assert energy.empty, "the latch must not open before the meter is full"
    assert transform.pos.z <= height_on_landing + 1e-6, "flight is locked out while empty"

    # Penalised rate: 45/s * 0.55 = ~24.75/s on the ground.
    before = energy.current
    step(sim, 60)
    gained = energy.current - before
    assert abs(gained - 24.75) < 2.5, f"penalised recharge gave {gained:.1f}/s, expected ~24.8"

    step(sim, 60 * 5)
    assert energy.current == energy.maximum and not energy.empty, "a full meter clears the latch"

    step(sim, 30, thrust=True)
    assert transform.pos.z > 1.0, "flight must work again once the latch has opened"


def test_ground_recharge_is_much_faster_than_air() -> None:
    sim = bare_sim()
    energy = esper.component_for_entity(sim.player, Energy)
    body = esper.component_for_entity(sim.player, Body)

    energy.current = 20.0
    step(sim, 30)
    on_ground = energy.current - 20.0
    assert body.grounded

    energy.current = 20.0
    esper.component_for_entity(sim.player, Transform).pos.z = 80.0
    body.grounded = False
    step(sim, 30)
    in_air = energy.current - 20.0

    assert on_ground > in_air * 2.5, (
        f"ground recharge {on_ground:.1f} was not much better than air {in_air:.1f}"
    )


def airborne_at(sim: Sim, height: float, speed_y: float = 0.0) -> None:
    """Park the player at ``height`` with a given forward speed, in free fall."""
    esper.component_for_entity(sim.player, Transform).pos = Vec3(0, 0, height)
    esper.component_for_entity(sim.player, Velocity).vec = Vec3(0, speed_y, 0)
    esper.component_for_entity(sim.player, Body).grounded = False


def test_glide_never_gains_height() -> None:
    """The defining property: a glide slows a fall, it never reverses one.

    This is what separates the glide from the hop it replaced. Entered at or
    below zero vertical speed, altitude must be non-increasing on every single
    frame -- there is no combination of inputs that turns the wings into lift.
    """
    for entry_speed in (0.0, -2.0, -12.0, -30.0):
        sim = bare_sim()
        airborne_at(sim, 90.0)
        transform = esper.component_for_entity(sim.player, Transform)
        velocity = esper.component_for_entity(sim.player, Velocity)
        velocity.vec.z = entry_speed

        previous = transform.pos.z
        for _ in range(120):
            step(sim, 1, glide=True, move_y=1.0, aim_yaw=0.0)
            assert transform.pos.z <= previous + 1e-9, (
                f"entering at {entry_speed:+.0f} m/s, the glide climbed "
                f"{transform.pos.z - previous:.4f}m in one frame"
            )
            assert velocity.vec.z <= 1e-9, f"glide produced a climb rate of {velocity.vec.z:+.3f}"
            previous = transform.pos.z


def test_glide_does_not_extend_a_climb() -> None:
    """Held on the way up, the wings must not buy a single extra metre.

    The subtle version of the same rule. Softening gravity during an ascent
    would stretch the arc out and let you float higher than you had momentum
    for -- which is a hop wearing a glide's name. So the apex has to come out
    identical whether or not shift was held on the way up.
    """

    def apex(glide: bool) -> float:
        sim = bare_sim()
        airborne_at(sim, 60.0)
        transform = esper.component_for_entity(sim.player, Transform)
        esper.component_for_entity(sim.player, Velocity).vec.z = 9.0
        highest = transform.pos.z
        for _ in range(90):
            step(sim, 1, glide=glide)
            highest = max(highest, transform.pos.z)
        return highest

    with_wings, without = apex(True), apex(False)
    assert with_wings <= without + 1e-6, (
        f"gliding upward reached {with_wings:.3f}m against {without:.3f}m coasting"
    )


def test_glide_slows_the_fall() -> None:
    """Wings out, she descends at roughly ``glide_fall`` instead of terminal."""
    sim = bare_sim()
    flight = esper.component_for_entity(sim.player, Flight)

    airborne_at(sim, 200.0)
    step(sim, 120)  # two seconds of plain falling
    plain = 200.0 - esper.component_for_entity(sim.player, Transform).pos.z

    sim = bare_sim()
    airborne_at(sim, 200.0)
    step(sim, 120, glide=True)
    glided = 200.0 - esper.component_for_entity(sim.player, Transform).pos.z

    assert glided < plain * 0.45, f"gliding fell {glided:.1f}m against {plain:.1f}m falling"
    settled = -esper.component_for_entity(sim.player, Velocity).vec.z
    assert abs(settled - flight.glide_fall) < 1.5, (
        f"glide settled at {settled:.2f} m/s, expected about {flight.glide_fall}"
    )


def test_glide_keeps_horizontal_speed_and_covers_ground() -> None:
    """The point of the glide: arrive with speed, keep it, go a long way."""
    sim = bare_sim()
    airborne_at(sim, 120.0, speed_y=17.0)
    step(sim, 180)  # three seconds, wings in
    coasted = esper.component_for_entity(sim.player, Transform).pos.y
    coast_speed = esper.component_for_entity(sim.player, Velocity).vec.y

    sim = bare_sim()
    airborne_at(sim, 120.0, speed_y=17.0)
    step(sim, 180, glide=True)
    glided = esper.component_for_entity(sim.player, Transform).pos.y
    glide_speed = esper.component_for_entity(sim.player, Velocity).vec.y

    assert glide_speed > coast_speed * 2.0, (
        f"glide bled speed nearly as fast as coasting ({glide_speed:.1f} vs {coast_speed:.1f})"
    )
    assert glide_speed > 14.0, f"three seconds of gliding lost {17.0 - glide_speed:.1f} m/s"
    assert glided > coasted * 1.5, (
        f"gliding covered {glided:.0f}m against {coasted:.0f}m coasting -- not much of a glide"
    )


def test_glide_is_cheap_but_not_free() -> None:
    sim = bare_sim()
    energy = esper.component_for_entity(sim.player, Energy)
    airborne_at(sim, 150.0)

    energy.current = 100.0
    step(sim, 60, glide=True)
    spent_gliding = 100.0 - energy.current

    sim = bare_sim()
    energy = esper.component_for_entity(sim.player, Energy)
    airborne_at(sim, 150.0)
    energy.current = 100.0
    step(sim, 60, thrust=True)
    spent_thrusting = 100.0 - energy.current

    assert spent_gliding > 5.0, "the glide should not be free"
    assert spent_gliding < spent_thrusting * 0.45, (
        f"a glide costs {spent_gliding:.1f}/s against thrust at {spent_thrusting:.1f}/s"
    )


def test_glide_is_blocked_while_overheated_and_on_the_ground() -> None:
    sim = bare_sim()
    energy = esper.component_for_entity(sim.player, Energy)
    flight = esper.component_for_entity(sim.player, Flight)

    airborne_at(sim, 150.0)
    energy.current, energy.empty = 0.0, True
    step(sim, 90, glide=True)  # 1.5s: free fall covers ~25m, a glide about 7m
    assert not flight.gliding, "an overheated diver has no wings"
    fell = 150.0 - esper.component_for_entity(sim.player, Transform).pos.z
    assert fell > 20.0, f"overheated, she should be falling properly, but fell {fell:.1f}m"

    # And there is nothing to glide on with your feet on the tarmac.
    sim = bare_sim()
    flight = esper.component_for_entity(sim.player, Flight)
    step(sim, 30, glide=True)
    assert not flight.gliding, "gliding on the ground"
    assert esper.component_for_entity(sim.player, Transform).pos.z == 0.0, "shift hopped"


def test_thrust_beats_glide_when_both_are_held() -> None:
    """Space and shift together should climb, not glide."""
    sim = bare_sim()
    flight = esper.component_for_entity(sim.player, Flight)
    airborne_at(sim, 60.0)

    step(sim, 30, thrust=True, glide=True)
    assert flight.thrusting and not flight.gliding, "the glide must yield to the jets"
    assert esper.component_for_entity(sim.player, Transform).pos.z > 60.0, "did not climb"


def test_firing_the_lance_spends_flight_energy() -> None:
    """The gun and the jetpack share one meter, on purpose."""
    sim = bare_sim()
    energy = esper.component_for_entity(sim.player, Energy)
    weapon = esper.component_for_entity(sim.player, Weapon)

    step(sim, 1, fire=True)
    assert abs(energy.current - (100.0 - weapon.energy_cost)) < 0.9, (
        f"a shot left {energy.current:.1f} energy"
    )

    energy.current = 1.0
    before = sim.stats.shots_fired
    weapon.timer = 0.0
    step(sim, 1, fire=True)
    assert sim.stats.shots_fired == before, "cannot fire without the energy for it"


# --------------------------------------------------------------------------
# physics and the city
# --------------------------------------------------------------------------


def test_walls_stop_you_but_let_you_slide() -> None:
    """Collision kills the velocity into a wall and leaves the tangent alone."""
    sim = bare_sim()
    sim.grid.add_box(1, 10.0, -40.0, 30.0, 40.0, 30.0)
    ent = plain_body(Vec3(0.0, 0.0, 0.0))
    transform = esper.component_for_entity(ent, Transform)
    vel = esper.component_for_entity(ent, Velocity)

    # Push into the wall at 45 degrees: +X is blocked, +Y should survive.
    for _ in range(120):
        vel.vec.x, vel.vec.y = 20.0, 20.0
        sim.step(FRAME)

    assert transform.pos.x < 10.0, f"walked through the wall to x={transform.pos.x:.2f}"
    assert transform.pos.y > 20.0, f"slid only {transform.pos.y:.2f}m along the wall"


def test_you_can_land_on_a_roof() -> None:
    """The street and a rooftop are the same case to the physics pass."""
    sim = bare_sim()
    sim.grid.add_box(1, -12.0, -12.0, 12.0, 12.0, 25.0)
    transform = esper.component_for_entity(sim.player, Transform)
    body = esper.component_for_entity(sim.player, Body)

    transform.pos = Vec3(0.0, 0.0, 40.0)
    body.grounded = False
    step(sim, 120)

    assert body.grounded, "never landed"
    assert abs(transform.pos.z - 25.0) < 1e-6, f"landed at z={transform.pos.z:.3f}, not the roof"
    assert body.ground_z == 25.0


def test_landing_records_the_impact_speed() -> None:
    sim = bare_sim()
    transform = esper.component_for_entity(sim.player, Transform)
    body = esper.component_for_entity(sim.player, Body)
    transform.pos.z = 50.0
    body.grounded = False

    speeds = []
    for _ in range(240):
        sim.step(FRAME)
        if body.landed_speed > 0.0:
            speeds.append(body.landed_speed)
    assert speeds and speeds[0] > 20.0, f"impact speeds recorded: {speeds}"


def test_bodies_stay_inside_the_arena() -> None:
    sim = bare_sim()
    transform = esper.component_for_entity(sim.player, Transform)
    vel = esper.component_for_entity(sim.player, Velocity)
    for _ in range(600):
        vel.vec.x = 60.0
        sim.step(FRAME)
    assert transform.pos.x <= ARENA + 1e-6, f"escaped to x={transform.pos.x:.1f}"


def test_gait_phase_tracks_distance_not_time() -> None:
    """Legs are driven by metres travelled, so they never skate."""
    sim = bare_sim()
    gait = esper.component_for_entity(sim.player, Gait)

    start = gait.phase
    step(sim, 60, move_y=1.0, aim_yaw=0.0)
    assert gait.phase != start and gait.amplitude > 0.5, "a walking body should cycle its legs"

    standing = gait.phase
    step(sim, 60)
    assert gait.amplitude < 0.15, "the cycle should fold down when stopped"
    assert abs(gait.phase - standing) < 0.2, "a standing body should not keep striding"


# --------------------------------------------------------------------------
# the ants, the squad, and the guns
# --------------------------------------------------------------------------


def test_ants_walk_at_you_and_bite() -> None:
    sim = bare_sim()
    ant = entities.spawn_ant(Vec3(0.0, 40.0, 0.0))
    ant_pos = esper.component_for_entity(ant, Transform)
    player_health = esper.component_for_entity(sim.player, Health)

    for _ in range(60 * 12):
        sim.step(FRAME)
        if player_health.current < player_health.maximum:
            break

    assert ant_pos.pos.y < 12.0, f"the ant closed only to y={ant_pos.pos.y:.1f}"
    assert player_health.current < player_health.maximum, "the ant never bit"


def test_a_grounded_ant_cannot_bite_a_hovering_diver() -> None:
    """Melee reach is a short cylinder, not a circle on the map.

    Without this the whole flight model would be pointless -- an ant standing
    under a hovering diver would still be chewing on her.
    """
    sim = bare_sim()
    entities.spawn_ant(Vec3(0.0, 3.0, 0.0))
    transform = esper.component_for_entity(sim.player, Transform)
    health = esper.component_for_entity(sim.player, Health)
    body = esper.component_for_entity(sim.player, Body)

    for _ in range(60 * 4):
        transform.pos.z = 30.0  # pinned, so gravity cannot bring her down
        body.grounded = False
        sim.step(FRAME)

    assert health.current == health.maximum, "bitten from 30m below"


def test_projectiles_damage_and_kill_and_score() -> None:
    sim = bare_sim()
    ant = entities.spawn_ant(Vec3(0.0, 30.0, 0.0))
    ant_health = esper.component_for_entity(ant, Health)
    esper.remove_component(ant, AntBrain)  # hold it still; this is a gunnery test

    step(sim, 1)  # let the index processor see the ant
    step(sim, 40, fire=True, aim_yaw=0.0, aim_pitch=0.0)

    assert ant_health.current < ant_health.maximum, "the lance never connected"

    for _ in range(60 * 6):
        step(sim, 1, fire=True, aim_yaw=0.0, aim_pitch=0.0)
        if sim.stats.ants_killed:
            break
    assert sim.stats.ants_killed == 1, f"kill count was {sim.stats.ants_killed}"
    assert esper.has_component(ant, Dead), "a dead ant should be marked, then reaped"


def test_projectiles_do_not_hit_their_own_side() -> None:
    sim = bare_sim()
    ally = entities.spawn_ally(Vec3(0.0, 25.0, 0.0), rally=Vec3(0, 25, 0))
    ally_health = esper.component_for_entity(ally, Health)
    step(sim, 90, fire=True, aim_yaw=0.0, aim_pitch=0.0)
    assert ally_health.current == ally_health.maximum, "friendly fire got through"


def test_projectiles_stop_at_buildings() -> None:
    sim = bare_sim()
    sim.grid.add_box(1, -20.0, 8.0, 20.0, 14.0, 40.0)
    ant = entities.spawn_ant(Vec3(0.0, 30.0, 0.0))
    ant_health = esper.component_for_entity(ant, Health)
    esper.remove_component(ant, AntBrain)

    step(sim, 120, fire=True, aim_yaw=0.0, aim_pitch=0.0)
    assert ant_health.current == ant_health.maximum, "shot straight through a building"


def test_projectiles_expire() -> None:
    """Nothing should be left flying when the range runs out."""
    sim = bare_sim()
    step(sim, 30, fire=True)
    assert esper.get_components(Projectile), "expected bolts in the air"
    step(sim, 60 * 4)
    assert not esper.get_components(Projectile), "bolts outlived their range"


def test_allies_engage_and_hold_a_standoff() -> None:
    sim = Sim(seed=3)
    sim.player = entities.spawn_player(Vec3(0.0, 0.0, 0.0))
    ally = entities.spawn_ally(Vec3(0.0, 0.0, 0.0), rally=Vec3(0, 0, 0), courage=1.0)
    ant = entities.spawn_ant(Vec3(0.0, 70.0, 0.0), spitter=False)
    esper.remove_component(ant, AntBrain)  # a stationary target to close on
    ant_health = esper.component_for_entity(ant, Health)
    ally_pos = esper.component_for_entity(ally, Transform)

    for _ in range(60 * 12):
        sim.step(FRAME)

    distance = abs(ally_pos.pos.y - 70.0)
    assert ant_health.current < ant_health.maximum, "the squad never opened fire"
    assert 5.0 < distance < 45.0, f"ally ended up {distance:.1f}m away, outside any standoff"


def test_waves_arrive_and_are_capped() -> None:
    # The wave number is a clock, so it keeps counting even once the streets
    # are full; only the head count is capped.
    sim = worldgen.new_game(seed=99, allies=0, ants=0)
    sim.ant_cap = 400
    sim.wave_interval = 1.0
    for _ in range(60 * 30):
        sim.step(FRAME)
    assert sim.stats.wave > 20, f"only {sim.stats.wave} waves in thirty seconds"
    assert len(sim.sides[Faction.BUGS]) > 100, "waves arrived but brought no ants"

    capped = worldgen.new_game(seed=99, allies=0, ants=0)
    capped.ant_cap = 12
    capped.wave_interval = 1.0
    for _ in range(60 * 30):
        capped.step(FRAME)
    assert len(capped.sides[Faction.BUGS]) <= 12, (
        f"cap of 12 was breached: {len(capped.sides[Faction.BUGS])} ants"
    )
    assert capped.stats.wave > 20, "the difficulty clock stalled once the cap was reached"


def test_wave_spawns_land_on_solid_ground() -> None:
    """Ants must arrive on the street or on a roof, never inside a tower."""
    sim = worldgen.new_game(seed=5, allies=0, ants=0)
    sim.wave_interval = 0.5
    for _ in range(60 * 20):
        sim.step(FRAME)

    inside = 0
    for ent, (transform, _brain) in esper.get_components(Transform, AntBrain):
        for _ident, x0, y0, x1, y1, top in sim.grid.query_aabb(
            transform.pos.x, transform.pos.y, transform.pos.x, transform.pos.y
        ):
            if x0 <= transform.pos.x <= x1 and y0 <= transform.pos.y <= y1:
                if transform.pos.z < top - 1.0:
                    inside += 1
    assert inside == 0, f"{inside} ants ended up buried inside buildings"


# --------------------------------------------------------------------------
# whole-game properties
# --------------------------------------------------------------------------


def test_the_city_generates_and_leaves_the_plaza_clear() -> None:
    sim = worldgen.new_game(seed=2024)
    assert len(sim.grid.boxes) > 60, f"only {len(sim.grid.boxes)} buildings"
    assert sim.ground_height(0.0, 0.0) == 0.0, "the player spawn is inside a building"
    for radius in (0.0, 5.0, 10.0):
        for i in range(12):
            angle = math.tau * i / 12.0
            x, y = math.cos(angle) * radius, math.sin(angle) * radius
            assert sim.ground_height(x, y) == 0.0, f"plaza blocked at ({x:.1f},{y:.1f})"


def test_the_same_seed_gives_the_same_game() -> None:
    """Determinism, end to end -- the city, the AI and the RNG together."""

    def run(seed: int) -> tuple:
        sim = worldgen.new_game(seed=seed)
        step(sim, 60 * 25, thrust=True, fire=True, move_y=1.0, aim_yaw=33.0)
        transform = esper.component_for_entity(sim.player, Transform)
        return (
            len(sim.grid.boxes),
            round(transform.pos.x, 6),
            round(transform.pos.y, 6),
            round(transform.pos.z, 6),
            sim.stats.ants_killed,
            sim.stats.shots_fired,
            sim.stats.wave,
        )

    first, second = run(31337), run(31337)
    assert first == second, f"diverged:\n  {first}\n  {second}"
    assert run(31338) != first, "different seeds produced identical games"


def test_worlds_are_isolated_from_each_other() -> None:
    """Two Sims can coexist; stepping one must not touch the other."""
    a = worldgen.new_game(seed=1, allies=2, ants=2)
    b = worldgen.new_game(seed=2, allies=2, ants=2)

    a.activate()
    before = esper.component_for_entity(a.player, Transform).pos.y
    b.activate()
    step(b, 120, move_y=1.0, aim_yaw=0.0)
    a.activate()
    after = esper.component_for_entity(a.player, Transform).pos.y

    assert before == after, "stepping one world moved the player in another"
    assert a.stats.elapsed == 0.0 and b.stats.elapsed > 1.0


def test_death_is_counted_once_then_reaped() -> None:
    sim = bare_sim()
    ant = entities.spawn_ant(Vec3(0.0, 8.0, 0.0))
    esper.remove_component(ant, AntBrain)
    step(sim, 1)
    esper.component_for_entity(ant, Health).current = 1.0

    for _ in range(60 * 4):
        step(sim, 1, fire=True, aim_yaw=0.0, aim_pitch=0.0)
        if not esper.entity_exists(ant):
            break

    assert sim.stats.ants_killed == 1, f"counted {sim.stats.ants_killed} kills for one ant"
    assert not esper.entity_exists(ant), "the corpse was never reaped"


def test_effects_clean_themselves_up() -> None:
    sim = bare_sim()
    for _ in range(20):
        entities.spawn_effect(Vec3(0, 0, 1), "spark", life=0.2)
    assert len(esper.get_components(Renderable)) > 20
    step(sim, 60)
    kinds = {r.kind for _e, (r,) in esper.get_components(Renderable)}
    assert "spark" not in kinds, "sparks outlived their Lifetime"


# --------------------------------------------------------------------------
# the renderer, offscreen
# --------------------------------------------------------------------------


def render_tests() -> None:
    """Build the PS1 pipeline against an offscreen buffer and photograph it.

    Everything in here needs a GL context, so it is quarantined behind
    ``--no-render`` for machines that have none.
    """
    # Forced before cramigula is imported, so its own prc block cannot open a
    # real window. Everything below then drives the actual application class
    # rather than a hand-rolled ShowBase, which is the only way cramigula.py
    # gets any coverage at all.
    loadPrcFileData("", "window-type offscreen")
    loadPrcFileData("", "framebuffer-multisample 0")
    loadPrcFileData("", "multisamples 0")

    from panda3d.core import PNMImage

    import cramigula
    import render

    app = cramigula.Cramigula(seed=8080)
    base, pipeline, sim, view = app, app.pipeline, app.sim, app.view
    camera = esper.get_processor(render.CameraProcessor)

    def draw(frames: int = 1, **intent_fields) -> None:
        for _ in range(frames):
            step(sim, 1, **intent_fields)
            base.graphicsEngine.renderFrame()

    def shot() -> PNMImage:
        return pipeline.screenshot_image()

    def place(pos: Vec3, speed: Vec3 = Vec3(0, 0, 0)) -> None:
        """Teleport the player and let the camera snap rather than fly over.

        The camera's own spring already snaps past ``CAM_SNAP`` metres of
        error, so this only has to clear the "have I started" flag for the
        cases where the teleport is short.
        """
        esper.component_for_entity(sim.player, Transform).pos = Vec3(pos)
        esper.component_for_entity(sim.player, Velocity).vec = Vec3(speed)
        esper.component_for_entity(sim.player, Body).grounded = pos.z <= 0.01
        camera._started = False

    energy = esper.component_for_entity(sim.player, Energy)
    panels: list[tuple[str, PNMImage]] = []

    # 1. street level, with the squad and the first ants closing in
    draw(90, aim_yaw=0.0, aim_pitch=0.0)
    place(Vec3(0, 0, 0))
    draw(20, aim_yaw=0.0, aim_pitch=-6.0)
    panels.append(("street", shot()))

    # 2. climbing out, jets lit
    place(Vec3(-20, -40, 48))
    draw(30, thrust=True, aim_yaw=28.0, aim_pitch=-16.0, move_y=1.0)
    panels.append(("thrust", shot()))

    # 3. gliding: wings out, descending slowly, keeping the speed she arrived
    #    with. The wings sweep down a few degrees when the glide is engaged.
    place(Vec3(-30, -70, 62), Vec3(6, 15, -3))
    draw(45, glide=True, aim_yaw=22.0, aim_pitch=-20.0, move_y=1.0)
    panels.append(("glide", shot()))

    # 4. the lance, fired at whatever is in front
    draw(14, fire=True, glide=True, aim_yaw=22.0, aim_pitch=-22.0)
    panels.append(("lance", shot()))

    # 5. overheated: the HUD goes red and both flight and glide are locked out
    place(Vec3(10, 10, 0))
    energy.current, energy.empty = 0.0, True
    draw(4, aim_yaw=150.0, aim_pitch=-4.0, thrust=True, glide=True)
    panels.append(("overheat", shot()))

    # 6. a swarm at street level, which is what the game mostly looks like
    place(Vec3(0, 0, 0))
    energy.current, energy.empty = 100.0, False
    for index in range(14):
        angle = math.tau * index / 14.0
        distance = 9.0 + (index % 4) * 4.0
        entities.spawn_ant(
            Vec3(math.cos(angle) * distance, math.sin(angle) * distance, 0.0),
            spitter=(index % 5 == 0),
            seed=index,
        )
    draw(26, aim_yaw=0.0, aim_pitch=-4.0, fire=True)
    panels.append(("swarm", shot()))

    check("pipeline renders at 320x240", lambda: _assert_size(panels))
    check("vertex snapping changes the image", lambda: _assert_snapping(base, sim, draw, shot))
    check("output is quantised to 5 bits a channel", lambda: _assert_quantised(panels[0][1]))
    check("frames are not blank", lambda: _assert_varied(panels))
    check("scene graph tracks entity births", lambda: _assert_nodes(view, sim))
    check("dead entities release their nodes", lambda: _assert_release(view, sim, base))
    check("the crosshair points where the lance goes", lambda: _assert_camera_aim(app, draw))
    check("the camera stays out of the ground and walls", lambda: _assert_camera_clear(app, draw))

    # The sheet is written before the app-level tests, which restart the game
    # and throw the photographed world away.
    _write_sheet(panels)

    print(f"\n  contact sheet -> {SHEET}\n")

    check("the keyboard drives the player", lambda: _assert_input(app))
    check("F1 toggles the vertex snapping", lambda: _assert_f1(app))
    check("restart rebuilds the world and the scene", lambda: _assert_restart(app))


def _assert_snapping(base, sim, draw, shot) -> None:
    """Turning the vertex grid off must actually move geometry.

    The snap is invisible in a still if you do not know where to look, so the
    test compares two stills instead: park the camera somewhere with a long
    receding edge, photograph it snapped and unsnapped, and require that a
    meaningful number of pixels disagree. Guards against the shader input
    being renamed or the whole thing silently no-opping.
    """
    import ps1

    draw(2, aim_yaw=17.0, aim_pitch=-11.0)
    snapped = shot()
    base.render.setShaderInput("jitterRes", (1e5, 1e5))
    draw(2, aim_yaw=17.0, aim_pitch=-11.0)
    smooth = shot()
    base.render.setShaderInput("jitterRes", (float(ps1.RES_X), float(ps1.RES_Y)))

    differing = sum(
        1
        for y in range(0, ps1.RES_Y, 2)
        for x in range(0, ps1.RES_X, 2)
        if snapped.getXel(x, y) != smooth.getXel(x, y)
    )
    assert differing > 40, f"only {differing} sampled pixels moved when snapping was disabled"


def _assert_size(panels) -> None:
    import ps1

    for name, image in panels:
        assert image.getXSize() == ps1.RES_X and image.getYSize() == ps1.RES_Y, (
            f"{name} came out {image.getXSize()}x{image.getYSize()}"
        )


def _assert_quantised(image) -> None:
    """Every world pixel should land on one of 32 levels, as 15-bit colour did.

    Sampled from a window that misses the HUD: the overlay is drawn by the
    fixed-function pipeline on a second display region and never goes through
    the fragment shader, so its pixels are legitimately off-grid.

    The tolerance covers the render target being 8 bits a channel: a 5-bit
    level of j/31 is stored as round(j*255/31)/255, which is up to half an
    8-bit step -- 0.06 on the 31 scale -- away from where it started.
    """
    offenders = []
    for y in range(20, 200, 3):
        for x in range(180, 310, 3):
            for channel in image.getXel(x, y):
                level = channel * 31.0
                if abs(level - round(level)) > 0.08:
                    offenders.append((x, y, round(level, 3)))
    assert not offenders, (
        f"{len(offenders)} samples were off the 5-bit grid, e.g. {offenders[:3]}"
    )


def _assert_varied(panels) -> None:
    for name, image in panels:
        colours = {
            tuple(round(c, 3) for c in image.getXel(x, y))
            for y in range(0, image.getYSize(), 7)
            for x in range(0, image.getXSize(), 7)
        }
        assert len(colours) > 12, f"{name} has only {len(colours)} distinct colours"


def _assert_nodes(view, sim) -> None:
    sim.activate()
    live = {
        ent
        for ent, (_t, _r) in esper.get_components(Transform, Renderable)
        if not esper.has_component(ent, Building)  # the city is baked, not per-node
    }
    missing = live - set(view.nodes)
    assert not missing, f"{len(missing)} entities have no node"


def _assert_release(view, sim, base) -> None:
    sim.activate()
    ent = entities.spawn_ant(Vec3(5.0, 5.0, 0.0))
    step(sim, 1)
    base.graphicsEngine.renderFrame()
    assert ent in view.nodes, "a new ant never got a node"

    esper.delete_entity(ent)
    step(sim, 2)
    base.graphicsEngine.renderFrame()
    assert ent not in view.nodes, "a deleted ant left its node behind"


def _assert_camera_aim(app, draw) -> None:
    """Screen centre must be the fire direction, at every aim angle.

    The crosshair is drawn dead centre, so the camera's forward vector has to
    equal :func:`sim.aim_vector` for the same angles -- otherwise the reticle
    points somewhere the lance does not go, and the error grows with pitch.
    An earlier version of this camera positioned itself from the aim and then
    looked back at the player, which put the crosshair on her head; this is
    the guard against that coming back.
    """
    for yaw, pitch in ((0.0, 0.0), (37.0, -28.0), (-120.0, 41.0), (200.0, -70.0), (95.0, 78.0)):
        draw(3, aim_yaw=yaw, aim_pitch=pitch)
        forward = app.camera.getQuat(app.render).getForward()
        wanted = aim_vector(yaw, pitch)
        error = math.degrees(
            math.acos(max(-1.0, min(1.0, forward.dot(wanted))))
        )
        assert error < 0.05, (
            f"at yaw {yaw} pitch {pitch} the camera looks {error:.2f} degrees "
            f"off the shot direction"
        )


def _assert_camera_clear(app, draw) -> None:
    """The boom must never end up underground or inside a tower.

    Flies a lap of the city at rooftop height and at street level, checking
    the camera's own position against the collision grid every frame.
    """
    sim = app.sim
    offences = 0
    worst = ""
    for index in range(90):
        yaw = index * 4.0
        draw(1, aim_yaw=yaw, aim_pitch=-18.0, move_y=1.0, thrust=(index % 30 < 8))
        position = app.camera.getPos(app.render)
        if position.z < 0.2:
            offences += 1
            worst = f"z={position.z:.2f} underground"
            continue
        for _ident, x0, y0, x1, y1, top in sim.grid.query_aabb(
            position.x, position.y, position.x, position.y
        ):
            if x0 <= position.x <= x1 and y0 <= position.y <= y1 and position.z < top - 0.5:
                offences += 1
                worst = f"{position.z:.1f}m inside a {top:.0f}m building"
                break
    assert offences == 0, f"the camera was buried on {offences} frames ({worst})"


def _assert_input(app) -> None:
    """Held keys must reach the player's Intent through the real task loop.

    Drives ``app.taskMgr.step()`` rather than ``sim.step()``, so this covers
    ``_read_mouse`` and ``_write_intent`` as well as everything below them.
    """
    app.sim.activate()
    intent = esper.component_for_entity(app.sim.player, Intent)
    energy = esper.component_for_entity(app.sim.player, Energy)
    energy.current, energy.empty = 100.0, False

    app.keys.clear()
    app.keys["w"] = True
    app.keys["space"] = True
    app.taskMgr.step()
    assert intent.move_y == 1.0, f"W did not reach the intent (move_y={intent.move_y})"
    assert intent.thrust, "space did not reach the intent"

    app.keys.clear()
    app.keys["a"] = True
    app.taskMgr.step()
    assert intent.move_x == -1.0, f"A should strafe left, got {intent.move_x}"
    assert not intent.thrust, "releasing space should drop the thrust"

    app.keys.clear()
    app.taskMgr.step()


def _assert_f1(app) -> None:
    before = app.snapping
    app._toggle_snapping()
    assert app.snapping is not before, "F1 did not flip the snapping flag"
    app._toggle_snapping()
    assert app.snapping is before, "F1 is not a toggle"


def _assert_restart(app) -> None:
    """R must build a new city and leave no scene graph behind."""
    old_sim, old_view = app.sim, app.view
    old_seed = app.seed

    app._start_game()
    app.taskMgr.step()

    assert app.sim is not old_sim, "restart reused the old Sim"
    assert app.seed == old_seed + 1, "restart did not advance the seed"
    assert old_view.root.isEmpty(), "the previous actor root was never removed"
    assert old_view.city.isEmpty(), "the previous city was never removed"
    assert len(app.sim.grid.boxes) > 60, "the new city is missing its buildings"
    assert app.view.nodes, "the new world produced no actor nodes"


def _write_sheet(panels) -> None:
    """Three across, two down, with a one-pixel gutter."""
    from panda3d.core import PNMImage

    import ps1

    if not panels:
        return
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    gutter = 2
    sheet = PNMImage(
        columns * ps1.RES_X + (columns + 1) * gutter,
        rows * ps1.RES_Y + (rows + 1) * gutter,
    )
    sheet.fill(0.05, 0.05, 0.07)
    for index, (_name, image) in enumerate(panels):
        col, row = index % columns, index // columns
        sheet.copySubImage(
            image,
            gutter + col * (ps1.RES_X + gutter),
            gutter + row * (ps1.RES_Y + gutter),
        )
    sheet.write(str(SHEET))


# --------------------------------------------------------------------------


SIM_TESTS = (
    test_heading_conventions,
    test_spatial_hash_finds_the_nearest,
    test_static_grid_reports_roof_heights,
    test_thrust_lifts_and_drains,
    test_thrust_is_acceleration_not_a_set_velocity,
    test_rise_speed_is_capped,
    test_overheat_latches_at_zero_and_only_clears_when_full,
    test_ground_recharge_is_much_faster_than_air,
    test_glide_never_gains_height,
    test_glide_does_not_extend_a_climb,
    test_glide_slows_the_fall,
    test_glide_keeps_horizontal_speed_and_covers_ground,
    test_glide_is_cheap_but_not_free,
    test_glide_is_blocked_while_overheated_and_on_the_ground,
    test_thrust_beats_glide_when_both_are_held,
    test_firing_the_lance_spends_flight_energy,
    test_walls_stop_you_but_let_you_slide,
    test_you_can_land_on_a_roof,
    test_landing_records_the_impact_speed,
    test_bodies_stay_inside_the_arena,
    test_gait_phase_tracks_distance_not_time,
    test_ants_walk_at_you_and_bite,
    test_a_grounded_ant_cannot_bite_a_hovering_diver,
    test_projectiles_damage_and_kill_and_score,
    test_projectiles_do_not_hit_their_own_side,
    test_projectiles_stop_at_buildings,
    test_projectiles_expire,
    test_allies_engage_and_hold_a_standoff,
    test_waves_arrive_and_are_capped,
    test_wave_spawns_land_on_solid_ground,
    test_the_city_generates_and_leaves_the_plaza_clear,
    test_the_same_seed_gives_the_same_game,
    test_worlds_are_isolated_from_each_other,
    test_death_is_counted_once_then_reaped,
    test_effects_clean_themselves_up,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless tests for Cramigula.")
    parser.add_argument(
        "--no-render", action="store_true", help="skip the offscreen GL half"
    )
    args = parser.parse_args(argv)

    print("simulation (no GL, no window)")
    for test in SIM_TESTS:
        check(test.__name__, test)

    if not args.no_render:
        print("\nrenderer (offscreen buffer)")
        render_tests()

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    for failure in _failed:
        print(f"  FAIL {failure}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
