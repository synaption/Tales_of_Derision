//! The simulation, tested with no renderer in the build at all.
//!
//! ```text
//! cargo test --no-default-features       # these tests, no bevy_render compiled
//! cargo test                             # these plus tests/rendered.rs
//! ```
//!
//! Every test here builds a bare [`App`], adds [`SimPlugin`] and nothing
//! else, and steps it by hand. No window, no GPU, no `Time` -- the timestep
//! is written into [`SimClock`] directly, so a test that wants nine hundred
//! exact sixtieths of a second gets exactly that and the assertions can be
//! about numbers rather than about tolerances.
//!
//! The bulk of it is the Wing Diver's movement model, because that is the
//! part of the game with a right answer.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_math::Vec3;

use cramigula::components::*;
use cramigula::sim::{
    aim_vector, angle_delta, heading_vectors, horizontal_speed, yaw_to, PlayerEntity, SimClock,
    SimPlugin, Stats, Waves,
};
use cramigula::spatial::{Slab, StaticGrid};
use cramigula::{headless_game, spawn, worldgen};

const FRAME: f32 = 1.0 / 60.0;

// --------------------------------------------------------------------------
// harness
// --------------------------------------------------------------------------

/// A world with the simulation in it and nothing else -- no city, no people,
/// and the wave spawner parked so that a physics test is not interrupted by
/// forty ants landing on it.
fn bare_app() -> App {
    let mut app = App::new();
    app.add_plugins(SimPlugin);
    app.update();
    app.world_mut().resource_mut::<Waves>().timer = 1.0e9;
    app
}

/// Advance `frames` steps, writing `intent` into `entity` at the start of
/// each one.
///
/// Re-applied every frame rather than latched, because that is what a held
/// key does: the keyboard writes the whole `Intent` sixty times a second, and
/// a test that sets it once is testing a different game.
fn step(app: &mut App, entity: Entity, frames: usize, intent: Intent) {
    for _ in 0..frames {
        app.world_mut().resource_mut::<SimClock>().set(FRAME);
        if let Some(mut slot) = app.world_mut().get_mut::<Intent>(entity) {
            *slot = intent;
        }
        app.update();
    }
}

/// Step without touching anybody's intent -- for testing the brains, which
/// write their own.
fn idle(app: &mut App, frames: usize) {
    for _ in 0..frames {
        app.world_mut().resource_mut::<SimClock>().set(FRAME);
        app.update();
    }
}

fn get<T: Component + Clone>(app: &App, entity: Entity) -> T {
    app.world()
        .get::<T>(entity)
        .cloned()
        .expect("component missing")
}

fn diver(app: &mut App, pos: Vec3) -> Entity {
    let world = app.world_mut();
    let entity = {
        let mut commands = world.commands();
        spawn::player(&mut commands, pos)
    };
    world.flush();
    world.resource_mut::<PlayerEntity>().0 = Some(entity);
    entity
}

fn bug(app: &mut App, pos: Vec3, spitter: bool) -> Entity {
    let world = app.world_mut();
    let entity = {
        let mut commands = world.commands();
        spawn::ant(&mut commands, pos, spitter)
    };
    world.flush();
    entity
}

/// A body with no `Flight` and no `Walker`, so the physics pass is the only
/// thing touching its velocity.
///
/// Needed because the movement systems run *before* physics and would
/// otherwise overwrite a velocity a test had just set, which makes a pure
/// collision test quietly measure something else.
fn plain_body(app: &mut App, pos: Vec3, vel: Vec3) -> Entity {
    let entity = app
        .world_mut()
        .spawn((
            Pose::at(pos),
            Velocity(vel),
            Body {
                radius: 0.5,
                height: 1.8,
                gravity: 22.0,
                grounded: false,
                ..Default::default()
            },
        ))
        .id();
    app.world_mut().flush();
    entity
}

/// Put a diver in the air at `height` travelling at `speed` along -Z, with a
/// vertical rate of `rise`.
fn airborne(app: &mut App, height: f32, speed: f32, rise: f32) -> Entity {
    let entity = diver(app, Vec3::new(0.0, height, 0.0));
    {
        let mut body = app.world_mut().get_mut::<Body>(entity).unwrap();
        body.grounded = false;
    }
    let mut vel = app.world_mut().get_mut::<Velocity>(entity).unwrap();
    vel.0 = Vec3::new(0.0, rise, -speed);
    entity
}

fn add_building(app: &mut App, slab: Slab) {
    app.world_mut().resource_mut::<StaticGrid>().add(slab);
}

fn thrusting() -> Intent {
    Intent {
        thrust: true,
        ..Default::default()
    }
}

fn gliding() -> Intent {
    Intent {
        glide: true,
        ..Default::default()
    }
}

// --------------------------------------------------------------------------
// the maths the rest of the game is built on
// --------------------------------------------------------------------------

#[test]
fn heading_zero_faces_bevys_forward() {
    let (fx, fz, rx, rz) = heading_vectors(0.0);
    assert!((fx - 0.0).abs() < 1e-6 && (fz + 1.0).abs() < 1e-6, "forward is -Z");
    assert!((rx - 1.0).abs() < 1e-6 && (rz - 0.0).abs() < 1e-6, "right is +X");
}

#[test]
fn aim_vector_agrees_with_heading_vectors_on_the_flat() {
    for yaw in [-170.0, -90.0, 0.0, 37.0, 90.0, 180.0] {
        let (fx, fz, _, _) = heading_vectors(yaw);
        let aim = aim_vector(yaw, 0.0);
        assert!(
            (aim.x - fx).abs() < 1e-5 && (aim.z - fz).abs() < 1e-5 && aim.y.abs() < 1e-6,
            "yaw {yaw}: {aim:?} vs ({fx}, 0, {fz})"
        );
        assert!((aim.length() - 1.0).abs() < 1e-5, "aim is not a unit vector");
    }
}

#[test]
fn yaw_to_inverts_the_aim() {
    for yaw in [-179.0, -90.0, -1.0, 0.0, 44.0, 91.0, 179.0] {
        let aim = aim_vector(yaw, 0.0);
        let recovered = yaw_to(aim.x, aim.z);
        assert!(
            angle_delta(recovered, yaw).abs() < 1e-3,
            "yaw {yaw} came back as {recovered}"
        );
    }
}

#[test]
fn angle_delta_takes_the_short_way_round() {
    assert!((angle_delta(170.0, -170.0) - -20.0).abs() < 1e-4);
    assert!((angle_delta(-170.0, 170.0) - 20.0).abs() < 1e-4);
    assert!((angle_delta(10.0, 350.0) - 20.0).abs() < 1e-4);
}

// --------------------------------------------------------------------------
// thrust
// --------------------------------------------------------------------------

#[test]
fn thrust_is_an_acceleration_not_an_assignment() {
    // A diver who has been falling has to pay the fall back before she
    // rises. If thrust assigned `vel.y` she would snap upward instead, which
    // is an elevator and not a jetpack.
    let mut app = bare_app();
    let entity = airborne(&mut app, 60.0, 0.0, -12.0);

    step(&mut app, entity, 1, thrusting());
    let vy = get::<Velocity>(&app, entity).0.y;
    assert!(
        vy < 0.0,
        "one frame of thrust reversed a 12 m/s fall: vy = {vy}"
    );
    assert!(vy > -12.0, "thrust did nothing at all: vy = {vy}");
}

#[test]
fn a_tap_hops_and_a_hold_climbs() {
    /// The highest she gets over `frames`, thrusting for the first `held` of
    /// them. A tap has to be measured at its apex: she comes back down.
    fn apex(held: usize, frames: usize) -> f32 {
        let mut app = bare_app();
        let entity = diver(&mut app, Vec3::ZERO);
        let mut peak = 0.0f32;
        for frame in 0..frames {
            let intent = if frame < held {
                thrusting()
            } else {
                Intent::default()
            };
            step(&mut app, entity, 1, intent);
            peak = peak.max(get::<Pose>(&app, entity).pos.y);
        }
        peak
    }

    let hop = apex(4, 90);
    let climb = apex(60, 60);

    assert!(hop > 0.05, "a four-frame tap did not leave the ground: {hop:.3}m");
    assert!(
        climb > hop * 3.0,
        "holding thrust for a second ({climb:.2}m) barely beat a tap ({hop:.2}m)"
    );
}

#[test]
fn the_climb_rate_is_capped() {
    let mut app = bare_app();
    let entity = diver(&mut app, Vec3::ZERO);
    let cap = get::<Flight>(&app, entity).rise_max;
    step(&mut app, entity, 120, thrusting());
    let vy = get::<Velocity>(&app, entity).0.y;
    assert!(
        vy <= cap + 1e-3,
        "climb reached {vy} m/s against a cap of {cap}"
    );
}

#[test]
fn thrust_drains_at_the_advertised_rate() {
    let mut app = bare_app();
    let entity = diver(&mut app, Vec3::ZERO);
    let drain = get::<Energy>(&app, entity).drain_thrust;

    step(&mut app, entity, 60, thrusting());
    let spent = 100.0 - get::<Energy>(&app, entity).current;
    assert!(
        (spent - drain).abs() < 0.5,
        "a second of thrust cost {spent:.2}, expected about {drain}"
    );
}

// --------------------------------------------------------------------------
// the overheat, which is the whole class
// --------------------------------------------------------------------------

#[test]
fn the_meter_latches_at_exactly_zero() {
    let mut app = bare_app();
    let entity = airborne(&mut app, 300.0, 0.0, 0.0);

    // Thrust until the latch closes, and catch the frame it happens on.
    let mut latched_at = None;
    for frame in 0..400 {
        step(&mut app, entity, 1, thrusting());
        let energy = get::<Energy>(&app, entity);
        if energy.empty {
            assert_eq!(
                energy.current, 0.0,
                "the latch closed at {} rather than at zero",
                energy.current
            );
            latched_at = Some(frame);
            break;
        }
    }
    let frame = latched_at.expect("400 frames of thrust never emptied the meter");
    let seconds = frame as f32 * FRAME;
    assert!(
        (3.0..4.5).contains(&seconds),
        "a full meter lasted {seconds:.2}s of continuous thrust"
    );
}

#[test]
fn overheating_locks_flight_out_entirely() {
    let mut app = bare_app();
    let entity = airborne(&mut app, 300.0, 0.0, 0.0);
    app.world_mut().get_mut::<Energy>(entity).unwrap().empty = true;
    app.world_mut().get_mut::<Energy>(entity).unwrap().current = 0.0;

    let before = get::<Velocity>(&app, entity).0.y;
    step(&mut app, entity, 30, thrusting());
    let after = get::<Velocity>(&app, entity).0.y;
    assert!(
        after < before,
        "an overheated diver still climbed: {before} -> {after}"
    );
    assert!(
        !get::<Flight>(&app, entity).thrusting,
        "the jets lit on an empty meter"
    );
}

#[test]
fn the_latch_only_opens_on_a_completely_full_bar() {
    let mut app = bare_app();
    let entity = diver(&mut app, Vec3::ZERO);
    {
        let mut energy = app.world_mut().get_mut::<Energy>(entity).unwrap();
        energy.current = 0.0;
        energy.empty = true;
    }

    // Recharge on the ground at 45/s, penalised to 55% -> about 24.75/s.
    // Three seconds gets to ~74%, which is a long way from full.
    step(&mut app, entity, 180, Intent::default());
    let energy = get::<Energy>(&app, entity);
    assert!(
        energy.current > 50.0 && energy.current < energy.maximum,
        "expected a partial refill, got {}",
        energy.current
    );
    assert!(energy.empty, "the latch released before the bar was full");

    step(&mut app, entity, 180, Intent::default());
    let energy = get::<Energy>(&app, entity);
    assert_eq!(energy.current, energy.maximum);
    assert!(!energy.empty, "a full bar did not release the latch");
}

#[test]
fn the_ground_recharges_faster_than_the_air() {
    let mut app = bare_app();

    let grounded = diver(&mut app, Vec3::ZERO);
    app.world_mut().get_mut::<Energy>(grounded).unwrap().current = 10.0;
    step(&mut app, grounded, 60, Intent::default());
    let on_foot = get::<Energy>(&app, grounded).current;

    let mut app = bare_app();
    let flying = airborne(&mut app, 200.0, 0.0, 0.0);
    app.world_mut().get_mut::<Energy>(flying).unwrap().current = 10.0;
    step(&mut app, flying, 60, Intent::default());
    let in_air = get::<Energy>(&app, flying).current;

    assert!(
        on_foot > in_air + 20.0,
        "ground {on_foot:.1} vs air {in_air:.1}: landing should be worth much more"
    );
}

// --------------------------------------------------------------------------
// the glide -- the half of the model that is not a jetpack
// --------------------------------------------------------------------------

#[test]
fn a_glide_never_gains_height() {
    // The defining property. From any entry that is not already climbing,
    // the wings must not lift her a single millimetre on any frame.
    for entry in [0.0f32, -1.0, -6.0, -18.0, -30.0] {
        let mut app = bare_app();
        let entity = airborne(&mut app, 400.0, 14.0, entry);

        let mut previous = get::<Pose>(&app, entity).pos.y;
        for frame in 0..240 {
            step(&mut app, entity, 1, gliding());
            let height = get::<Pose>(&app, entity).pos.y;
            assert!(
                height <= previous + 1e-5,
                "entering at {entry} m/s, frame {frame} of the glide climbed {:.5}m",
                height - previous
            );
            previous = height;
        }
    }
}

#[test]
fn a_glide_does_not_extend_a_climb() {
    // Softening gravity while still ascending would stretch the arc and let
    // her float higher than she had momentum for. That is a hop wearing a
    // glide's name, so the apex must be identical either way.
    let mut app = bare_app();
    let held = airborne(&mut app, 100.0, 0.0, 9.0);
    let mut apex_with_glide = 0.0f32;
    for _ in 0..90 {
        step(&mut app, held, 1, gliding());
        apex_with_glide = apex_with_glide.max(get::<Pose>(&app, held).pos.y);
    }

    let mut app = bare_app();
    let bare = airborne(&mut app, 100.0, 0.0, 9.0);
    let mut apex_without = 0.0f32;
    for _ in 0..90 {
        step(&mut app, bare, 1, Intent::default());
        apex_without = apex_without.max(get::<Pose>(&app, bare).pos.y);
    }

    assert!(
        (apex_with_glide - apex_without).abs() < 1e-3,
        "gliding reached {apex_with_glide:.4}m against {apex_without:.4}m without: the glide bought height"
    );
}

#[test]
fn a_glide_slows_the_fall_to_its_terminal() {
    let mut app = bare_app();
    let entity = airborne(&mut app, 400.0, 0.0, -28.0);
    let terminal = get::<Flight>(&app, entity).glide_fall;

    step(&mut app, entity, 120, gliding());
    let vy = get::<Velocity>(&app, entity).0.y;
    assert!(
        (vy + terminal).abs() < 1.0,
        "two seconds of gliding settled at {vy:.2} m/s, expected about {}",
        -terminal
    );
    assert!(
        get::<Flight>(&app, entity).gliding,
        "the wings were not out at all"
    );
}

#[test]
fn a_glide_keeps_its_speed_and_covers_ground() {
    let mut app = bare_app();
    let entity = airborne(&mut app, 400.0, 16.0, 0.0);
    let start = get::<Pose>(&app, entity).pos;

    step(&mut app, entity, 180, gliding());
    let end = get::<Pose>(&app, entity).pos;

    let speed = horizontal_speed(get::<Velocity>(&app, entity).0);
    assert!(
        speed > 13.0,
        "three seconds of gliding bled 16 m/s down to {speed:.2}"
    );
    let travelled = ((end.x - start.x).powi(2) + (end.z - start.z).powi(2)).sqrt();
    let dropped = start.y - end.y;
    assert!(
        travelled > dropped * 2.5,
        "glide ratio was {travelled:.1}m across for {dropped:.1}m down"
    );
}

#[test]
fn a_glide_is_cheap_but_not_free() {
    let mut app = bare_app();
    let entity = airborne(&mut app, 400.0, 10.0, 0.0);
    step(&mut app, entity, 60, gliding());
    let glide_cost = 100.0 - get::<Energy>(&app, entity).current;

    let mut app = bare_app();
    let entity = airborne(&mut app, 400.0, 10.0, 0.0);
    step(&mut app, entity, 60, thrusting());
    let thrust_cost = 100.0 - get::<Energy>(&app, entity).current;

    assert!(glide_cost > 0.0, "the glide was free");
    assert!(
        thrust_cost > glide_cost * 3.0,
        "thrust cost {thrust_cost:.1} against the glide's {glide_cost:.1}: \
         altitude is supposed to be much more expensive than distance"
    );
}

#[test]
fn a_glide_needs_air_and_a_charge() {
    // On the ground the wings do nothing at all...
    let mut app = bare_app();
    let entity = diver(&mut app, Vec3::ZERO);
    step(&mut app, entity, 60, gliding());
    assert!(
        !get::<Flight>(&app, entity).gliding,
        "she glided while standing on the tarmac"
    );
    assert_eq!(
        get::<Energy>(&app, entity).current,
        100.0,
        "standing still with shift held drained the meter"
    );

    // ...and overheated they do not come out either.
    let mut app = bare_app();
    let entity = airborne(&mut app, 400.0, 0.0, 0.0);
    {
        let mut energy = app.world_mut().get_mut::<Energy>(entity).unwrap();
        energy.current = 0.0;
        energy.empty = true;
    }
    let before = get::<Pose>(&app, entity).pos.y;
    step(&mut app, entity, 90, gliding());
    let fell = before - get::<Pose>(&app, entity).pos.y;
    assert!(
        !get::<Flight>(&app, entity).gliding,
        "an overheated diver deployed her wings"
    );
    assert!(
        fell > 8.0,
        "an overheated diver only fell {fell:.2}m in a second and a half"
    );
}

#[test]
fn thrust_beats_glide_when_both_are_held() {
    let mut app = bare_app();
    let entity = airborne(&mut app, 200.0, 0.0, -5.0);

    step(
        &mut app,
        entity,
        30,
        Intent {
            thrust: true,
            glide: true,
            ..Default::default()
        },
    );

    let flight = get::<Flight>(&app, entity);
    assert!(flight.thrusting, "holding both did not light the jets");
    assert!(!flight.gliding, "the wings came out with the jets lit");
}

// --------------------------------------------------------------------------
// physics against the city
// --------------------------------------------------------------------------

#[test]
fn a_body_lands_on_a_roof_and_stays_there() {
    let mut app = bare_app();
    add_building(
        &mut app,
        Slab { x0: -10.0, z0: -10.0, x1: 10.0, z1: 10.0, top: 18.0 },
    );
    let entity = plain_body(&mut app, Vec3::new(0.0, 30.0, 0.0), Vec3::ZERO);

    idle(&mut app, 120);
    let pose = get::<Pose>(&app, entity);
    let body = get::<Body>(&app, entity);
    assert!((pose.pos.y - 18.0).abs() < 1e-3, "settled at {}", pose.pos.y);
    assert!(body.grounded, "standing on a roof did not count as grounded");
    assert!((body.ground_y - 18.0).abs() < 1e-3);
}

#[test]
fn landing_records_the_impact_speed_once() {
    let mut app = bare_app();
    let entity = plain_body(&mut app, Vec3::new(0.0, 12.0, 0.0), Vec3::ZERO);

    let mut peak = 0.0f32;
    for _ in 0..180 {
        idle(&mut app, 1);
        peak = peak.max(get::<Body>(&app, entity).landed_speed);
    }
    assert!(peak > 15.0, "a 12m drop registered as {peak:.1} m/s");
    assert_eq!(
        get::<Body>(&app, entity).landed_speed,
        0.0,
        "the landing flag stayed raised after the frame it happened on"
    );
}

#[test]
fn a_body_slides_along_a_wall_instead_of_stopping_dead() {
    let mut app = bare_app();
    // A long wall lying along X, with the body driven diagonally into it.
    add_building(
        &mut app,
        Slab { x0: -40.0, z0: 0.0, x1: 40.0, z1: 6.0, top: 20.0 },
    );
    let entity = plain_body(&mut app, Vec3::new(0.0, 0.0, -6.0), Vec3::new(6.0, 0.0, 6.0));

    idle(&mut app, 60);
    let pose = get::<Pose>(&app, entity);
    assert!(
        pose.pos.z < 0.0,
        "the body walked into the building: z = {}",
        pose.pos.z
    );
    assert!(
        pose.pos.x > 3.0,
        "the body stuck to the wall instead of sliding: x = {}",
        pose.pos.x
    );
}

#[test]
fn a_kerb_is_stepped_over_and_a_wall_is_not() {
    let mut app = bare_app();
    add_building(
        &mut app,
        Slab { x0: 2.0, z0: -20.0, x1: 20.0, z1: 20.0, top: 0.5 },
    );
    let kerb = plain_body(&mut app, Vec3::new(-2.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0));
    // Five metres of travel at 5 m/s, from x = -2, is x = 3 -- well past the
    // kerb's edge at x = 2 if nothing stopped it.
    idle(&mut app, 60);
    let pose = get::<Pose>(&app, kerb);
    assert!(pose.pos.x > 2.5, "a 0.5m kerb stopped a walker: x = {}", pose.pos.x);
    assert!(
        (pose.pos.y - 0.5).abs() < 1e-3,
        "the walker crossed the kerb without climbing onto it: y = {}",
        pose.pos.y
    );

    let mut app = bare_app();
    add_building(
        &mut app,
        Slab { x0: 2.0, z0: -20.0, x1: 20.0, z1: 20.0, top: 9.0 },
    );
    let wall = plain_body(&mut app, Vec3::new(-2.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0));
    idle(&mut app, 60);
    assert!(
        get::<Pose>(&app, wall).pos.x < 2.0,
        "a 9m wall was walked through: x = {}",
        get::<Pose>(&app, wall).pos.x
    );
}

#[test]
fn the_arena_holds_everything_in() {
    let mut app = bare_app();
    let entity = plain_body(
        &mut app,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(400.0, 0.0, -400.0),
    );
    idle(&mut app, 240);
    let pos = get::<Pose>(&app, entity).pos;
    assert!(
        pos.x.abs() <= cramigula::sim::ARENA + 1e-3 && pos.z.abs() <= cramigula::sim::ARENA + 1e-3,
        "escaped to {pos:?}"
    );
}

#[test]
fn a_long_stall_cannot_teleport_a_body_through_a_wall() {
    // The clamp in SimClock is load-bearing: a half-second frame at 30 m/s
    // is fifteen metres of travel, which is straight through a building.
    let mut app = bare_app();
    add_building(
        &mut app,
        Slab { x0: 5.0, z0: -20.0, x1: 9.0, z1: 20.0, top: 30.0 },
    );
    let entity = plain_body(&mut app, Vec3::new(0.0, 0.0, 0.0), Vec3::new(30.0, 0.0, 0.0));

    for _ in 0..40 {
        app.world_mut().resource_mut::<SimClock>().set(0.5);
        app.update();
    }
    assert!(
        get::<Pose>(&app, entity).pos.x < 5.0,
        "a stalled frame put the body at x = {}",
        get::<Pose>(&app, entity).pos.x
    );
}

// --------------------------------------------------------------------------
// gunfire
// --------------------------------------------------------------------------

#[test]
fn the_lance_costs_energy_so_a_shot_costs_altitude() {
    let mut app = bare_app();
    let entity = airborne(&mut app, 100.0, 0.0, 0.0);
    let cost = get::<Weapon>(&app, entity).energy_cost;
    assert!(cost > 0.0, "the lance is supposed to be an energy weapon");

    step(
        &mut app,
        entity,
        1,
        Intent {
            fire: true,
            ..Default::default()
        },
    );
    let spent = 100.0 - get::<Energy>(&app, entity).current;
    assert!(
        (spent - cost).abs() < 0.6,
        "one shot cost {spent:.2} against an advertised {cost}"
    );
    assert_eq!(app.world().resource::<Stats>().shots_fired, 1);
}

#[test]
fn a_lance_kills_an_ant_and_scores_it_exactly_once() {
    let mut app = bare_app();
    let shooter = diver(&mut app, Vec3::ZERO);
    let target = bug(&mut app, Vec3::new(0.0, 0.0, -18.0), false);
    app.world_mut().get_mut::<Health>(target).unwrap().current = 5.0;
    // Aim flat down the -Z axis, at chest height.
    app.world_mut().get_mut::<Intent>(shooter).unwrap().aim_yaw = 0.0;

    for _ in 0..120 {
        step(
            &mut app,
            shooter,
            1,
            Intent {
                fire: true,
                ..Default::default()
            },
        );
        if app.world().get::<Dead>(target).is_some() {
            break;
        }
    }

    assert!(
        app.world().get::<Dead>(target).is_some(),
        "two seconds of point-blank lance fire did not kill an ant on 5 health"
    );
    idle(&mut app, 5);
    assert_eq!(
        app.world().resource::<Stats>().ants_killed,
        1,
        "the kill was counted more than once"
    );
}

#[test]
fn a_bolt_cannot_hit_its_own_side() {
    let mut app = bare_app();
    let shooter = diver(&mut app, Vec3::ZERO);
    let world = app.world_mut();
    let friend = {
        let mut commands = world.commands();
        spawn::ally(&mut commands, Vec3::new(0.0, 0.0, -12.0), Vec3::ZERO, 1.0, 0)
    };
    world.flush();

    let before = get::<Health>(&app, friend).current;
    for _ in 0..120 {
        step(
            &mut app,
            shooter,
            1,
            Intent {
                fire: true,
                ..Default::default()
            },
        );
    }
    assert_eq!(
        get::<Health>(&app, friend).current,
        before,
        "the lance shot a grunt standing in front of it"
    );
}

#[test]
fn a_fast_bolt_does_not_tunnel_through_a_bug() {
    // 95 m/s at 60fps is 1.58m a frame, and an ant is 1.8m wide. Without the
    // substepping in `projectiles` the lance would pass straight through
    // roughly half the time.
    let mut app = bare_app();
    let shooter = diver(&mut app, Vec3::ZERO);
    for offset in 0..12 {
        let z = -20.0 - offset as f32 * 0.13;
        let target = bug(&mut app, Vec3::new(0.0, 0.0, z), false);
        let before = get::<Health>(&app, target).current;

        for _ in 0..40 {
            step(
                &mut app,
                shooter,
                1,
                Intent {
                    fire: true,
                    ..Default::default()
                },
            );
            if get::<Health>(&app, target).current < before {
                break;
            }
        }
        assert!(
            get::<Health>(&app, target).current < before,
            "the lance missed an ant at z = {z}"
        );
        app.world_mut().entity_mut(target).despawn();
        app.world_mut().flush();
    }
}

#[test]
fn a_bite_cannot_reach_a_hovering_diver() {
    // The bite is a short cylinder, not a circle on the map. An ant standing
    // directly under a hovering Wing Diver must not be able to chew her feet.
    let mut app = bare_app();
    let target = diver(&mut app, Vec3::new(0.0, 8.0, 0.0));
    app.world_mut().get_mut::<Body>(target).unwrap().grounded = false;
    let biter = bug(&mut app, Vec3::new(0.0, 0.0, 0.0), false);

    let before = get::<Health>(&app, target).current;
    for _ in 0..120 {
        app.world_mut().resource_mut::<SimClock>().set(FRAME);
        // Pin her in the air; the point of the test is the reach, not flight.
        app.world_mut().get_mut::<Pose>(target).unwrap().pos = Vec3::new(0.0, 8.0, 0.0);
        app.world_mut().get_mut::<Velocity>(target).unwrap().0 = Vec3::ZERO;
        app.world_mut().get_mut::<Intent>(biter).unwrap().fire = true;
        app.update();
    }
    assert_eq!(
        get::<Health>(&app, target).current,
        before,
        "an ant on the ground bit a diver eight metres up"
    );

    // ...and the same ant on the same spot bites her the moment she lands.
    app.world_mut().get_mut::<Pose>(target).unwrap().pos = Vec3::new(0.0, 0.0, 1.5);
    for _ in 0..120 {
        idle(&mut app, 1);
    }
    assert!(
        get::<Health>(&app, target).current < before,
        "the ant could not bite her on the ground either, so the test proves nothing"
    );
}

// --------------------------------------------------------------------------
// brains
// --------------------------------------------------------------------------

#[test]
fn an_ant_closes_on_the_player() {
    let mut app = bare_app();
    let target = diver(&mut app, Vec3::ZERO);
    let ant = bug(&mut app, Vec3::new(0.0, 0.0, -60.0), false);

    let start = get::<Pose>(&app, ant).pos.z;
    for _ in 0..240 {
        app.world_mut().resource_mut::<SimClock>().set(FRAME);
        app.world_mut().get_mut::<Pose>(target).unwrap().pos = Vec3::ZERO;
        app.update();
    }
    let end = get::<Pose>(&app, ant).pos.z;
    assert!(
        end > start + 20.0,
        "four seconds of walking closed only {:.1}m",
        end - start
    );
}

#[test]
fn a_dead_ant_stops_deciding_things() {
    // A corpse keeps its momentum on purpose -- a bug shot out of a leap
    // should finish the arc rather than stop in mid-air -- so the thing to
    // assert is that its *brain* is off, not that it is motionless. No
    // intent, no steering toward the player, no biting.
    let mut app = bare_app();
    let target = diver(&mut app, Vec3::ZERO);
    let ant = bug(&mut app, Vec3::new(0.0, 0.0, -30.0), false);
    idle(&mut app, 60);
    assert!(
        get::<Intent>(&app, ant).move_y > 0.5,
        "the ant was not advancing before it died, so the test proves nothing"
    );

    app.world_mut().entity_mut(ant).insert(Dead { timer: 100.0 });
    let heading = get::<Pose>(&app, ant).heading;
    let health = get::<Health>(&app, target).current;

    // Move the target somewhere a live ant would immediately turn toward.
    app.world_mut().get_mut::<Pose>(target).unwrap().pos = Vec3::new(40.0, 0.0, -30.0);
    idle(&mut app, 60);

    let intent = get::<Intent>(&app, ant);
    assert_eq!((intent.move_x, intent.move_y), (0.0, 0.0), "a corpse still had a stick input");
    assert!(!intent.fire, "a corpse kept biting");
    assert!(
        angle_delta(get::<Pose>(&app, ant).heading, heading).abs() < 1.0,
        "a corpse turned to follow the player"
    );
    assert_eq!(
        get::<Health>(&app, target).current,
        health,
        "a corpse did damage"
    );
}

#[test]
fn an_ally_backs_off_when_the_fight_gets_too_close() {
    let mut app = bare_app();
    let world = app.world_mut();
    let grunt = {
        let mut commands = world.commands();
        spawn::ally(&mut commands, Vec3::ZERO, Vec3::ZERO, 1.0, 0)
    };
    world.flush();
    let ant = bug(&mut app, Vec3::new(0.0, 0.0, -4.0), false);

    for _ in 0..120 {
        app.world_mut().resource_mut::<SimClock>().set(FRAME);
        // Hold the ant still, so we measure the grunt's decision and not a
        // chase.
        app.world_mut().get_mut::<Pose>(ant).unwrap().pos = Vec3::new(0.0, 0.0, -4.0);
        app.world_mut().get_mut::<Velocity>(ant).unwrap().0 = Vec3::ZERO;
        app.update();
    }

    let pos = get::<Pose>(&app, grunt).pos;
    assert!(
        pos.z > 2.0,
        "the grunt stood its ground four metres from an ant: z = {}",
        pos.z
    );
}

// --------------------------------------------------------------------------
// waves, lifetimes and the scoreboard
// --------------------------------------------------------------------------

#[test]
fn the_difficulty_clock_runs_even_at_the_ant_cap() {
    // A player who stops killing ants must not also stop the game getting
    // harder. The wave number is a clock, not a spawn counter.
    let mut app = bare_app();
    diver(&mut app, Vec3::ZERO);
    {
        let mut waves = app.world_mut().resource_mut::<Waves>();
        waves.timer = 0.05;
        waves.interval = 0.05;
        waves.cap = 3;
    }

    idle(&mut app, 120);
    let stats = *app.world().resource::<Stats>();
    assert!(
        stats.wave > 10,
        "the wave counter stalled at {} once the cap filled",
        stats.wave
    );

    let ants = app
        .world_mut()
        .query_filtered::<Entity, With<AntBrain>>()
        .iter(app.world())
        .count();
    assert!(ants <= 3, "the cap leaked: {ants} ants alive against a cap of 3");
}

#[test]
fn corpses_and_effects_clean_themselves_up() {
    let mut app = bare_app();
    let ant = bug(&mut app, Vec3::ZERO, false);
    app.world_mut().get_mut::<Health>(ant).unwrap().current = 0.0;

    idle(&mut app, 2);
    assert!(app.world().get::<Dead>(ant).is_some(), "no corpse was made");

    idle(&mut app, 60);
    assert!(
        app.world().get_entity(ant).is_err(),
        "the corpse outlived its timer"
    );

    // The dust puff it left behind should go too.
    idle(&mut app, 120);
    let effects = app
        .world_mut()
        .query_filtered::<Entity, With<Lifetime>>()
        .iter(app.world())
        .count();
    assert_eq!(effects, 0, "{effects} effects never expired");
}

#[test]
fn the_walk_cycle_follows_the_feet() {
    let mut app = bare_app();
    let entity = plain_body(&mut app, Vec3::ZERO, Vec3::ZERO);
    app.world_mut().entity_mut(entity).insert(Gait::new(2.2));
    app.world_mut().get_mut::<Body>(entity).unwrap().grounded = true;

    // Walking: the phase advances and the legs open up.
    for _ in 0..60 {
        app.world_mut().resource_mut::<SimClock>().set(FRAME);
        app.world_mut().get_mut::<Velocity>(entity).unwrap().0 = Vec3::new(0.0, 0.0, -5.0);
        app.update();
    }
    let walking = get::<Gait>(&app, entity);
    assert!(walking.amplitude > 0.8, "legs stayed folded at 5 m/s");

    // Stopped: the amplitude eases back down rather than freezing mid-stride.
    for _ in 0..60 {
        app.world_mut().resource_mut::<SimClock>().set(FRAME);
        app.world_mut().get_mut::<Velocity>(entity).unwrap().0 = Vec3::ZERO;
        app.update();
    }
    assert!(
        get::<Gait>(&app, entity).amplitude < 0.05,
        "the stride did not fold down when the feet stopped"
    );
}

// --------------------------------------------------------------------------
// world generation
// --------------------------------------------------------------------------

/// Every building in a generated world, as comparable numbers.
fn skyline(app: &mut App) -> Vec<(i32, i32, i32)> {
    let mut out: Vec<_> = app
        .world_mut()
        .query::<(&Pose, &Building)>()
        .iter(app.world())
        .map(|(pose, b)| {
            (
                (pose.pos.x * 100.0) as i32,
                (pose.pos.z * 100.0) as i32,
                (b.height * 100.0) as i32,
            )
        })
        .collect();
    out.sort_unstable();
    out
}

#[test]
fn one_seed_builds_one_city() {
    let mut a = headless_game(4242, 8, 14);
    let mut b = headless_game(4242, 8, 14);
    assert_eq!(skyline(&mut a), skyline(&mut b));
    assert!(!skyline(&mut a).is_empty(), "the city came out empty");

    let mut c = headless_game(4243, 8, 14);
    assert_ne!(
        skyline(&mut a),
        skyline(&mut c),
        "two different seeds produced the same city"
    );
}

#[test]
fn the_same_seed_replays_the_same_fight() {
    // Determinism has to survive the simulation, not just generation --
    // otherwise a seed is only good for screenshots.
    let mut a = headless_game(77, 6, 10);
    let mut b = headless_game(77, 6, 10);
    let intent = Intent {
        move_y: 1.0,
        thrust: true,
        fire: true,
        ..Default::default()
    };
    cramigula::step(&mut a, 300, FRAME, intent);
    cramigula::step(&mut b, 300, FRAME, intent);

    let sa = *a.world().resource::<Stats>();
    let sb = *b.world().resource::<Stats>();
    assert_eq!(
        (sa.ants_killed, sa.shots_fired, sa.wave),
        (sb.ants_killed, sb.shots_fired, sb.wave),
        "five seconds diverged from the same seed"
    );

    let pa = a.world().resource::<PlayerEntity>().0.unwrap();
    let pb = b.world().resource::<PlayerEntity>().0.unwrap();
    let (ha, hb) = (get::<Pose>(&a, pa), get::<Pose>(&b, pb));
    assert!(
        (ha.pos - hb.pos).length() < 1e-4,
        "the divers ended up at {:?} and {:?}",
        ha.pos,
        hb.pos
    );
}

#[test]
fn the_plaza_is_clear_for_take_off() {
    let mut app = headless_game(1234, 8, 14);
    let grid = app.world().resource::<StaticGrid>();
    assert!(
        grid.height_at(0.0, 0.0) == 0.0,
        "the player spawns inside a building"
    );
    for slab in grid.slabs() {
        let nearest_x = 0.0f32.clamp(slab.x0, slab.x1);
        let nearest_z = 0.0f32.clamp(slab.z0, slab.z1);
        let distance = (nearest_x * nearest_x + nearest_z * nearest_z).sqrt();
        assert!(
            distance > 5.0,
            "a tower reaches to within {distance:.1}m of the spawn"
        );
    }

    // And the squad landed on the ground rather than inside the kerb.
    let count = app
        .world_mut()
        .query_filtered::<Entity, With<AllyBrain>>()
        .iter(app.world())
        .count();
    assert_eq!(count, 8);
}

#[test]
fn a_restart_rebuilds_the_world_and_moves_the_seed_on() {
    let mut app = headless_game(500, 4, 6);
    let before = skyline(&mut app);
    let player_before = app.world().resource::<PlayerEntity>().0.unwrap();

    cramigula::step(&mut app, 120, FRAME, Intent::default());
    app.world_mut().resource_mut::<worldgen::RestartRequest>().0 = true;
    cramigula::step(&mut app, 2, FRAME, Intent::default());

    let after = skyline(&mut app);
    assert!(!after.is_empty(), "the restart left no city");
    assert_ne!(before, after, "R gave back the same city");

    let player_after = app.world().resource::<PlayerEntity>().0.unwrap();
    assert_ne!(player_before, player_after, "the old player survived");
    assert!(
        app.world().get_entity(player_before).is_err(),
        "the old world was not torn down"
    );
    assert_eq!(
        app.world().resource::<Stats>().wave,
        0,
        "the scoreboard carried over"
    );
}
