//! The simulation: every system, and the plugin that orders them.
//!
//! This module imports `bevy_ecs`, `bevy_app`, `bevy_math` and
//! `bevy_platform`. It does not import `bevy_render`, it never touches a
//! `Mesh` or a `Transform`, and nothing in it can open a window. That is the
//! contract that lets `tests/headless.rs` run the entire game -- flight,
//! collision, ants, gunfire, waves -- with no GPU in the room, and it is why
//! the renderer is a separate set of systems bolted on in [`crate::render`].
//!
//! # Order
//!
//! The Python original ran on `esper`, which sorts processors by a numeric
//! priority. Bevy has no priorities; it has an explicit dependency graph, and
//! the whole simulation is one [`chain`](bevy_ecs::prelude::IntoScheduleConfigs::chain)
//! in [`SimSet::Simulate`]:
//!
//! | Order | System | What it does |
//! | --- | --- | --- |
//! | 1 | [`reindex`] | rebuild the spatial hash and the faction sets |
//! | 2 | [`ant_brains`] | ant brains write `Intent` |
//! | 3 | [`ally_brains`] | ally brains write `Intent` |
//! | 4 | [`flight`] | `Intent` + `Energy` -> jetpack velocity |
//! | 5 | [`walk`] | `Intent` -> ground velocity |
//! | 6 | [`weapons`] | `Intent.fire` -> projectiles and bites |
//! | 7 | [`physics`] | gravity, integration, city collision |
//! | 8 | [`projectiles`] | fly, hit, damage |
//! | 9 | [`deaths`] | anything at zero health becomes a corpse |
//! | 10 | [`gait`] | derive walk-cycle phase from speed |
//! | 11 | [`waves`] | keep the streets stocked with ants |
//! | 12 | [`reap`] | lifetimes, corpse timers, deletion |
//!
//! Chaining does more work here than the priority numbers did. Bevy inserts
//! a sync point between two chained systems when the earlier one queues
//! commands the later one would read, so a lance spawned by [`weapons`]
//! really is in the world by the time [`projectiles`] runs, in the same
//! frame -- the immediate-spawn behaviour of the `esper` version, recovered
//! from the ordering rather than from luck.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_math::Vec3;
use bevy_platform::collections::HashSet;

use crate::components::*;
use crate::rng::Rng;
use crate::spatial::{Slab, SpatialHash, StaticGrid};
use crate::spawn;

/// Half-width of the playable city, in metres. Bodies are clamped to it; it
/// is also the radius the wave spawner works from.
pub const ARENA: f32 = 230.0;

/// How far a body will be lifted onto a ledge without having to jump. Kerbs
/// and rubble, not rooftops.
pub const STEP_UP: f32 = 0.7;

// --------------------------------------------------------------------------
// small maths helpers
// --------------------------------------------------------------------------

/// `(forward_x, forward_z, right_x, right_z)` for a heading in degrees.
///
/// Bevy is Y-up and right-handed: a heading of 0 faces -Z, which is the
/// direction `Transform::forward` points, and positive heading turns
/// counter-clockwise seen from above.
#[inline]
pub fn heading_vectors(yaw_deg: f32) -> (f32, f32, f32, f32) {
    let (sin, cos) = yaw_deg.to_radians().sin_cos();
    (-sin, -cos, cos, -sin)
}

/// Unit vector for a yaw/pitch pair, matching [`heading_vectors`].
///
/// This is exactly the forward vector of
/// `Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0)`, which is the identity
/// the chase camera is built on: see [`crate::render::chase_camera`].
#[inline]
pub fn aim_vector(yaw_deg: f32, pitch_deg: f32) -> Vec3 {
    let (sin_y, cos_y) = yaw_deg.to_radians().sin_cos();
    let (sin_p, cos_p) = pitch_deg.to_radians().sin_cos();
    Vec3::new(-sin_y * cos_p, sin_p, -cos_y * cos_p)
}

/// Heading in degrees that points along the horizontal direction `(dx, dz)`.
#[inline]
pub fn yaw_to(dx: f32, dz: f32) -> f32 {
    (-dx).atan2(-dz).to_degrees()
}

/// Shortest signed turn from `current` to `target`, in degrees.
#[inline]
pub fn angle_delta(target: f32, current: f32) -> f32 {
    (target - current + 180.0).rem_euclid(360.0) - 180.0
}

/// Speed across the ground, ignoring climb or fall.
#[inline]
pub fn horizontal_speed(v: Vec3) -> f32 {
    (v.x * v.x + v.z * v.z).sqrt()
}

/// Turn an `Intent`'s stick into a world-space horizontal wish direction and
/// its magnitude. Length is clamped to 1 so diagonals are not faster.
#[inline]
fn wish_direction(intent: &Intent) -> (f32, f32, f32) {
    let (fx, fz, rx, rz) = heading_vectors(intent.aim_yaw);
    let mut wish_x = rx * intent.move_x + fx * intent.move_y;
    let mut wish_z = rz * intent.move_x + fz * intent.move_y;
    let mut length = (wish_x * wish_x + wish_z * wish_z).sqrt();
    if length > 1.0 {
        wish_x /= length;
        wish_z /= length;
        length = 1.0;
    }
    (wish_x, wish_z, length)
}

// --------------------------------------------------------------------------
// resources
// --------------------------------------------------------------------------

/// The frame's timestep, clamped.
///
/// Whoever drives the simulation writes this: the game from `Time`, a test
/// from a literal. That is the same seam the Python version had in
/// `Sim.step(dt)`, and it keeps `bevy_time` out of the simulation's
/// dependencies -- a test that wants a thousand exact sixtieths of a second
/// should not have to persuade a real clock to produce them.
#[derive(Resource, Debug, Clone, Copy)]
pub struct SimClock {
    dt: f32,
}

impl Default for SimClock {
    fn default() -> Self {
        SimClock { dt: 0.0 }
    }
}

impl SimClock {
    /// Longest step the collision code is willing to take. A stall that hands
    /// us half a second would otherwise teleport bodies through walls, and a
    /// paused-then-resumed window does exactly that.
    pub const MAX_STEP: f32 = 1.0 / 20.0;

    pub fn set(&mut self, dt: f32) {
        self.dt = dt.clamp(0.0, Self::MAX_STEP);
    }

    #[inline]
    pub fn dt(&self) -> f32 {
        self.dt
    }
}

/// Scoreboard. Read by the HUD, asserted on by the tests.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct Stats {
    pub ants_killed: u32,
    pub allies_lost: u32,
    pub shots_fired: u32,
    pub wave: u32,
    pub elapsed: f32,
}

/// Live combatants per side, rebuilt every frame by [`reindex`].
#[derive(Resource, Debug, Default)]
pub struct Sides {
    pub edf: HashSet<Entity>,
    pub bugs: HashSet<Entity>,
}

impl Sides {
    pub fn of(&self, side: Faction) -> &HashSet<Entity> {
        match side {
            Faction::Edf => &self.edf,
            Faction::Bugs => &self.bugs,
        }
    }

    pub fn enemies_of(&self, side: Faction) -> &HashSet<Entity> {
        self.of(side.opposing())
    }
}

/// Who the camera follows and the wave spawner aims at.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct PlayerEntity(pub Option<Entity>);

/// The ant tap.
#[derive(Resource, Debug, Clone, Copy)]
pub struct Waves {
    pub timer: f32,
    pub interval: f32,
    pub cap: usize,
}

impl Default for Waves {
    fn default() -> Self {
        Waves {
            timer: 3.0,
            interval: 11.0,
            cap: 46,
        }
    }
}

/// Ordering handle for everything in this module, so the renderer can say
/// `.after(SimSet::Simulate)` and mean it.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimSet;

/// The whole game, minus any way to look at it.
pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimClock>()
            .init_resource::<Stats>()
            .init_resource::<Sides>()
            .init_resource::<PlayerEntity>()
            .init_resource::<Waves>()
            .insert_resource(SpatialHash::new(12.0))
            .insert_resource(StaticGrid::new(16.0))
            .insert_resource(Rng::new(1234))
            .add_systems(
                Update,
                (
                    tick,
                    reindex,
                    ant_brains,
                    ally_brains,
                    flight,
                    walk,
                    weapons,
                    physics,
                    projectiles,
                    deaths,
                    gait,
                    waves,
                    reap,
                )
                    .chain()
                    .in_set(SimSet),
            );
    }
}

// --------------------------------------------------------------------------
// bookkeeping
// --------------------------------------------------------------------------

/// Advance the wall clock. The only thing that reads `SimClock` and writes
/// something other than a component.
pub fn tick(clock: Res<SimClock>, mut stats: ResMut<Stats>) {
    stats.elapsed += clock.dt();
}

/// Rebuild the per-frame spatial hash and the faction membership sets.
///
/// Corpses are indexed as neither: they are not targets, and nothing should
/// shoot them again.
pub fn reindex(
    mut hash: ResMut<SpatialHash>,
    mut sides: ResMut<Sides>,
    query: Query<(Entity, &Pose, &FactionTag), (With<Health>, Without<Dead>)>,
) {
    hash.clear();
    sides.edf.clear();
    sides.bugs.clear();
    for (entity, pose, tag) in &query {
        hash.insert(entity, pose.pos.x, pose.pos.z);
        match tag.0 {
            Faction::Edf => sides.edf.insert(entity),
            Faction::Bugs => sides.bugs.insert(entity),
        };
    }
}

// --------------------------------------------------------------------------
// brains -- these only ever write Intent
// --------------------------------------------------------------------------

/// Walk at the nearest EDF thing; bite it, spit at it, or leap at it.
///
/// There is no pathfinding. An ant that walks into a tower slides along it,
/// because [`physics`] resolves the penetration along the wall normal and
/// leaves the tangent alone. It looks like a swarm breaking around a building
/// and costs nothing.
pub fn ant_brains(
    clock: Res<SimClock>,
    mut rng: ResMut<Rng>,
    hash: Res<SpatialHash>,
    sides: Res<Sides>,
    mut ants: Query<(&Pose, &mut Intent, &mut AntBrain, &Walker, Option<&Dead>)>,
    poses: Query<&Pose>,
) {
    let dt = clock.dt();
    for (pose, mut intent, mut brain, walker, dead) in &mut ants {
        intent.clear();
        if dead.is_some() {
            continue;
        }

        brain.retarget -= dt;
        brain.leap_cd = (brain.leap_cd - dt).max(0.0);
        let stale = brain.target.is_none_or(|t| poses.get(t).is_err());
        if brain.retarget <= 0.0 || stale {
            brain.retarget = 0.35 + rng.unit() * 0.4;
            brain.target = hash
                .nearest(pose.pos.x, pose.pos.z, brain.sight, &sides.edf)
                .map(|(entity, _)| entity);
        }

        let Some(target) = brain.target else { continue };
        let Ok(goal) = poses.get(target) else { continue };

        let delta = goal.pos - pose.pos;
        let flat = (delta.x * delta.x + delta.z * delta.z).sqrt();
        if flat < 1e-4 {
            continue;
        }

        intent.aim_yaw = yaw_to(delta.x, delta.z);
        intent.aim_pitch = (delta.y + 0.9).atan2(flat.max(0.5)).to_degrees();

        if flat > brain.bite_range * 0.75 {
            intent.move_y = 1.0;
        }
        if flat <= brain.bite_range {
            intent.fire = true;
        }

        // A biter that has been kept just out of reach leaps. This is the
        // only answer ants have to a hovering Wing Diver, so it is on a
        // generous cooldown and needs the target to be reachably low.
        if !brain.spitter
            && walker.jump_speed > 0.0
            && brain.leap_cd <= 0.0
            && (4.0..16.0).contains(&flat)
            && delta.y < 9.0
        {
            intent.thrust = true;
            brain.leap_cd = 2.2 + rng.unit() * 1.6;
        }
    }
}

/// Advance to a standoff distance, shoot, and drift home when it is quiet.
///
/// Two behaviours, chosen by distance to the nearest ant: close in if the
/// fight is far away, back off if it is too close. The dead zone between the
/// two is what stops a squad oscillating on the spot.
pub fn ally_brains(
    clock: Res<SimClock>,
    mut rng: ResMut<Rng>,
    hash: Res<SpatialHash>,
    sides: Res<Sides>,
    mut allies: Query<(&Pose, &mut Intent, &mut AllyBrain, Option<&Dead>)>,
    poses: Query<&Pose>,
) {
    let dt = clock.dt();
    for (pose, mut intent, mut brain, dead) in &mut allies {
        intent.clear();
        if dead.is_some() {
            continue;
        }

        brain.retarget -= dt;
        let stale = brain.target.is_none_or(|t| poses.get(t).is_err());
        if brain.retarget <= 0.0 || stale {
            brain.retarget = 0.4 + rng.unit() * 0.5;
            brain.target = hash
                .nearest(pose.pos.x, pose.pos.z, brain.sight, &sides.bugs)
                .map(|(entity, _)| entity);
        }

        if let Some(goal) = brain.target.and_then(|t| poses.get(t).ok()) {
            let delta = goal.pos - pose.pos;
            let flat = (delta.x * delta.x + delta.z * delta.z).sqrt();
            intent.aim_yaw = yaw_to(delta.x, delta.z);
            intent.aim_pitch = (delta.y + 0.7).atan2(flat.max(0.5)).to_degrees();
            if flat < brain.sight {
                intent.fire = true;
            }
            if flat > brain.standoff * 1.25 {
                intent.move_y = 1.0;
            } else if flat < brain.standoff * 0.6 {
                intent.move_y = -1.0; // too close, give ground while firing
            } else {
                // Strafe, so a firing line does not look like a bus queue.
                intent.move_x = brain.strafe;
            }
            continue;
        }

        // Nothing in sight: wander back toward the rally point.
        let delta = brain.rally - pose.pos;
        if (delta.x * delta.x + delta.z * delta.z).sqrt() > 4.0 {
            intent.aim_yaw = yaw_to(delta.x, delta.z);
            intent.move_y = 0.6;
        }
    }
}

// --------------------------------------------------------------------------
// movement
// --------------------------------------------------------------------------

/// The Wing Diver movement model, and the reason this project exists.
///
/// Four coupled rules, in the order they are applied:
///
/// 1. **Thrust.** Held, it drains continuously and adds upward *acceleration*
///    -- not an upward velocity -- capped at `rise_max`. Because it is an
///    acceleration, tapping it gives a hop and holding it gives a climb, and
///    falling momentum has to be paid off before you rise.
/// 2. **Glide.** Held, and only in the air, the wings come out: the descent
///    eases back to `glide_fall` and the horizontal drag drops by most of an
///    order of magnitude, so the speed you arrived with is speed you keep. It
///    never adds height. A glide cannot lift you a millimetre -- that is the
///    whole difference between a glide and a hop -- it only makes the ground
///    come up slowly enough that you get somewhere first. At 7 energy a
///    second against the thrust's 26, distance is cheap and altitude is not.
/// 3. **Air control.** Airborne horizontal acceleration is high but the speed
///    cap is soft: you may exceed it and keep what you have, you just cannot
///    accelerate past it. Gliding trades some of that control away, because
///    committing to a line is what gliding is.
/// 4. **Recharge and overheat.** Energy only refills while neither thrusting
///    nor gliding, and four times faster with feet on the ground. Touch
///    exactly zero and the meter latches `empty`: no flight, no glide, and a
///    recharge at 55% rate that does not release until the bar is
///    *completely* full. Nearly all the skill in the class is in never
///    letting that latch close.
pub fn flight(
    clock: Res<SimClock>,
    mut query: Query<(
        &mut Pose,
        &mut Velocity,
        &mut Body,
        &mut Energy,
        &mut Flight,
        &Intent,
        Option<&Dead>,
    )>,
) {
    let dt = clock.dt();
    for (mut pose, mut vel, mut body, mut energy, mut flight, intent, dead) in &mut query {
        flight.thrusting = false;
        flight.gliding = false;
        if dead.is_some() {
            continue;
        }

        let (wish_x, wish_z, wish_len) = wish_direction(intent);

        // 1. thrust -- pay first, and take the last drop if that is all that
        // is left, which is what arms the overheat.
        if intent.thrust && !energy.empty && energy.current > 0.0 {
            let cost = (energy.drain_thrust * dt).min(energy.current);
            if energy.spend(cost) {
                flight.thrusting = true;
                vel.0.y = (vel.0.y + flight.thrust_accel * dt).min(flight.rise_max);
                if body.grounded {
                    body.grounded = false;
                    vel.0.y = vel.0.y.max(2.0);
                }
            }
        }

        // 2. glide -- airborne only, and never while the jets are lit.
        //
        // Note there is no `max(vel.y, ...)` anywhere in here: the wings can
        // only ever slow a descent, never reverse one.
        if intent.glide
            && !flight.thrusting
            && !body.grounded
            && !energy.empty
            && energy.current > 0.0
        {
            let cost = (energy.drain_glide * dt).min(energy.current);
            if energy.spend(cost) {
                flight.gliding = true;
                if vel.0.y < -flight.glide_fall {
                    // The wings bite: a hard fall is eased back to the glide
                    // terminal over a few tenths, not snapped to it.
                    vel.0.y += (-flight.glide_fall - vel.0.y) * (flight.glide_bite * dt).clamp(0.0, 1.0);
                }
            }
        }

        // 3. air / ground control
        if body.grounded && !flight.thrusting {
            // On foot she is an ordinary soldier, and a slow one.
            let target_x = wish_x * flight.air_max * 0.45;
            let target_z = wish_z * flight.air_max * 0.45;
            let rate = (flight.air_accel * 1.6 * dt).clamp(0.0, 1.0);
            vel.0.x += (target_x - vel.0.x) * rate;
            vel.0.z += (target_z - vel.0.z) * rate;
        } else {
            let speed = horizontal_speed(vel.0);
            let accel = if flight.gliding {
                flight.glide_accel
            } else {
                flight.air_accel
            };
            if wish_len > 1e-3 {
                let mut nx = vel.0.x + wish_x * accel * dt;
                let mut nz = vel.0.z + wish_z * accel * dt;
                let nspeed = (nx * nx + nz * nz).sqrt();
                // Soft cap: you cannot accelerate past air_max, but speed you
                // already have is yours to keep.
                let ceiling = flight.air_max.max(speed);
                if nspeed > ceiling {
                    nx *= ceiling / nspeed;
                    nz *= ceiling / nspeed;
                }
                vel.0.x = nx;
                vel.0.z = nz;
            } else {
                let drag = if flight.gliding {
                    flight.glide_drag
                } else {
                    flight.air_drag
                };
                let decay = (-drag * dt).exp();
                vel.0.x *= decay;
                vel.0.z *= decay;
            }
        }

        // 4. recharge
        if !flight.thrusting && !flight.gliding {
            let mut rate = if body.grounded {
                energy.regen_ground
            } else {
                energy.regen_air
            };
            if energy.empty {
                rate *= energy.empty_penalty;
            }
            energy.current += rate * dt;
            if energy.current >= energy.maximum {
                energy.current = energy.maximum;
                energy.empty = false; // the latch only opens at full
            }
        }

        pose.heading = intent.aim_yaw;
        // Cosmetic bank, eased so it does not snap when the stick centres. A
        // gliding diver leans into the turn harder; she is on her wings.
        let lean = if flight.gliding { 26.0 } else { 14.0 };
        let want_roll = -intent.move_x.clamp(-1.0, 1.0) * if body.grounded { 0.0 } else { lean };
        pose.roll += (want_roll - pose.roll) * (8.0 * dt).clamp(0.0, 1.0);
        pose.pitch = (-vel.0.y * 0.6).clamp(-18.0, 18.0);
    }
}

/// Ground locomotion for ants and grunts: accelerate, turn, sometimes leap.
pub fn walk(
    clock: Res<SimClock>,
    mut query: Query<(&mut Pose, &mut Velocity, &mut Body, &Walker, &Intent), Without<Dead>>,
) {
    let dt = clock.dt();
    for (mut pose, mut vel, mut body, walker, intent) in &mut query {
        let (wish_x, wish_z, wish_len) = wish_direction(intent);

        if body.grounded {
            if wish_len > 1e-3 {
                let target_x = wish_x * walker.max_speed;
                let target_z = wish_z * walker.max_speed;
                let rate = (walker.accel * dt / walker.max_speed.max(0.1)).clamp(0.0, 1.0);
                vel.0.x += (target_x - vel.0.x) * rate;
                vel.0.z += (target_z - vel.0.z) * rate;
            } else {
                let decay = (-walker.friction * dt).exp();
                vel.0.x *= decay;
                vel.0.z *= decay;
            }

            if intent.thrust && walker.jump_speed > 0.0 {
                vel.0.y = walker.jump_speed;
                body.grounded = false;
            }
        } else {
            // Ants steer a little in mid-leap; it makes them land on you.
            vel.0.x += wish_x * walker.accel * 0.22 * dt;
            vel.0.z += wish_z * walker.accel * 0.22 * dt;
        }

        // Face the way we are actually moving, at a finite turn rate.
        let speed = horizontal_speed(vel.0);
        let want = if speed > 0.25 {
            yaw_to(vel.0.x, vel.0.z)
        } else if wish_len > 1e-3 {
            yaw_to(wish_x, wish_z)
        } else {
            pose.heading
        };
        let step = walker.turn_rate * dt;
        pose.heading += angle_delta(want, pose.heading).clamp(-step, step);
    }
}

// --------------------------------------------------------------------------
// physics
// --------------------------------------------------------------------------

/// Gravity, integration and collision against the city.
///
/// Horizontal and vertical are resolved separately, in that order, which is
/// the cheap trick that makes stepping onto kerbs and landing on roofs both
/// fall out of the same code:
///
/// * horizontally, a body is a circle pushed out of any box whose roof it is
///   *below*; the push is along the box normal so the tangential component of
///   velocity survives and things slide;
/// * vertically, the support height under the body is the tallest roof at or
///   below the feet, so "the ground" and "that office block's roof" are the
///   same case.
pub fn physics(
    clock: Res<SimClock>,
    grid: Res<StaticGrid>,
    mut scratch: Local<Vec<Slab>>,
    mut query: Query<(&mut Pose, &mut Velocity, &mut Body, Option<&Flight>)>,
) {
    let dt = clock.dt();
    for (mut pose, mut vel, mut body, flight) in &mut query {
        if !body.grounded {
            // Gravity is scaled down rather than switched off while the wings
            // are out: the terminal descent then falls out of the balance
            // between this and `flight`'s easing, instead of being a hard
            // clamp that reads as an invisible floor.
            //
            // Gated on already descending, which matters more than it looks.
            // Wings that soften gravity on the way *up* stretch out a climb
            // -- press glide at the top of a thrust and you float higher than
            // you would have. That is a hop, which is the one thing a glide
            // must never be, so an ascent gets full gravity and the apex is
            // identical whether or not shift was held.
            let gliding = flight.is_some_and(|f| f.gliding);
            let scale = match flight {
                Some(f) if gliding && vel.0.y <= 0.0 => f.glide_gravity,
                _ => 1.0,
            };
            vel.0.y -= body.gravity * scale * dt;
            let fall_max = flight.map_or(55.0, |f| f.fall_max);
            if vel.0.y < -fall_max {
                vel.0.y = -fall_max;
            }
        }

        let pos = pose.pos;
        let mut new_x = pos.x + vel.0.x * dt;
        let mut new_z = pos.z + vel.0.z * dt;
        let radius = body.radius;

        // -- horizontal: push the circle out of every wall it is inside --
        grid.query_aabb(
            new_x - radius,
            new_z - radius,
            new_x + radius,
            new_z + radius,
            &mut scratch,
        );
        for slab in scratch.iter() {
            if pos.y >= slab.top - STEP_UP {
                continue; // we are on or above this roof; it is floor, not wall
            }
            let near_x = new_x.clamp(slab.x0, slab.x1);
            let near_z = new_z.clamp(slab.z0, slab.z1);
            let (dx, dz) = (new_x - near_x, new_z - near_z);
            let dist2 = dx * dx + dz * dz;
            if dist2 >= radius * radius {
                continue;
            }
            let (nx, nz, push) = if dist2 > 1e-9 {
                let dist = dist2.sqrt();
                (dx / dist, dz / dist, radius - dist)
            } else {
                // Centre is inside the box: escape through the nearest face.
                let left = new_x - slab.x0;
                let right = slab.x1 - new_x;
                let back = new_z - slab.z0;
                let front = slab.z1 - new_z;
                let smallest = left.min(right).min(back).min(front);
                let (nx, nz) = if smallest == left {
                    (-1.0, 0.0)
                } else if smallest == right {
                    (1.0, 0.0)
                } else if smallest == back {
                    (0.0, -1.0)
                } else {
                    (0.0, 1.0)
                };
                (nx, nz, smallest + radius)
            };
            new_x += nx * push;
            new_z += nz * push;
            let into = vel.0.x * nx + vel.0.z * nz;
            if into < 0.0 {
                vel.0.x -= into * nx;
                vel.0.z -= into * nz;
            }
        }

        new_x = new_x.clamp(-ARENA, ARENA);
        new_z = new_z.clamp(-ARENA, ARENA);

        // -- vertical: find what is under us, then land on it or fall past --
        grid.query_aabb(
            new_x - radius,
            new_z - radius,
            new_x + radius,
            new_z + radius,
            &mut scratch,
        );
        let mut support = 0.0f32;
        for slab in scratch.iter() {
            if slab.top <= pos.y + STEP_UP && slab.top > support {
                support = slab.top;
            }
        }

        let mut new_y = pos.y + vel.0.y * dt;
        body.landed_speed = 0.0;
        if new_y <= support {
            if !body.grounded && vel.0.y < 0.0 {
                body.landed_speed = -vel.0.y;
            }
            new_y = support;
            vel.0.y = 0.0;
            body.grounded = true;
            body.ground_y = support;
            body.airborne_time = 0.0;
        } else {
            if body.grounded && new_y > support + 0.05 {
                body.grounded = false;
            }
            if !body.grounded {
                body.airborne_time += dt;
            }
        }

        pose.pos = Vec3::new(new_x, new_y, new_z);
    }
}

/// Advance the walk cycle from distance travelled, not from wall time.
///
/// Phase is driven by metres covered so legs and ground speed always agree,
/// and the amplitude eases toward the current speed so that stopping folds
/// the cycle down instead of freezing it mid-stride.
pub fn gait(clock: Res<SimClock>, mut query: Query<(&Velocity, &mut Gait, &Body)>) {
    let dt = clock.dt();
    for (vel, mut gait, body) in &mut query {
        let speed = horizontal_speed(vel.0);
        let want = if body.grounded {
            gait.phase = (gait.phase + speed * gait.rate * dt).rem_euclid(1.0);
            (speed / 4.0).clamp(0.0, 1.0)
        } else {
            0.0 // legs tuck in mid-air
        };
        gait.amplitude += (want - gait.amplitude) * (9.0 * dt).clamp(0.0, 1.0);
    }
}

// --------------------------------------------------------------------------
// combat
// --------------------------------------------------------------------------

/// Turns `Intent.fire` into damage, by projectile or by bite.
///
/// Energy weapons check the same meter the jetpack uses, so a shot taken in
/// the air is a shot of altitude given up. That coupling is deliberate: it is
/// the whole reason the Wing Diver plays differently from a man with a gun.
pub fn weapons(
    clock: Res<SimClock>,
    mut commands: Commands,
    mut rng: ResMut<Rng>,
    mut stats: ResMut<Stats>,
    hash: Res<SpatialHash>,
    sides: Res<Sides>,
    mut shooters: Query<(
        Entity,
        &Pose,
        &Intent,
        &mut Weapon,
        &FactionTag,
        Option<&Body>,
        Option<&mut Energy>,
        Option<&Dead>,
    )>,
    mut victims: Query<(&Pose, &mut Health)>,
) {
    let dt = clock.dt();
    for (entity, pose, intent, mut weapon, tag, body, energy, dead) in &mut shooters {
        weapon.timer = (weapon.timer - dt).max(0.0);
        weapon.muzzle_flash = (weapon.muzzle_flash - dt).max(0.0);
        if dead.is_some() || !intent.fire || weapon.timer > 0.0 {
            continue;
        }

        // An energy weapon that cannot pay does not fire, and does not go on
        // cooldown either -- holding the trigger on an empty meter should
        // fire the instant it can, not on the next tick of a timer.
        if weapon.energy_cost > 0.0 {
            let Some(mut meter) = energy else { continue };
            if !meter.spend(weapon.energy_cost) {
                continue;
            }
        }

        weapon.timer = weapon.cooldown;
        weapon.muzzle_flash = 0.06;
        stats.shots_fired += 1;

        let eye_height = body.map_or(1.6, |b| b.height) * 0.72;
        let eye = pose.pos + Vec3::Y * eye_height;

        if weapon.melee {
            // Melee: hit the nearest enemy inside the arc, or hit nothing.
            //
            // The vertical check is what keeps a grounded ant from biting a
            // Wing Diver hovering directly overhead -- the reach is a short
            // cylinder, not a circle on the map.
            let enemies = sides.enemies_of(tag.0);
            let Some((target, _)) = hash.nearest(pose.pos.x, pose.pos.z, weapon.range, enemies)
            else {
                continue;
            };
            let Ok((other, mut health)) = victims.get_mut(target) else {
                continue;
            };
            if (other.pos.y - pose.pos.y).abs() > 2.4 {
                continue;
            }
            let hit = other.pos + Vec3::Y;
            health.damage(weapon.damage);
            spawn::effect(&mut commands, hit, ModelKind::Spark, 0.2, 0.6);
            continue;
        }

        let yaw = intent.aim_yaw + rng.range(-weapon.spread, weapon.spread);
        let pitch = intent.aim_pitch + rng.range(-weapon.spread, weapon.spread);
        let direction = aim_vector(yaw, pitch);
        spawn::bolt(
            &mut commands,
            eye + direction * 0.9,
            direction * weapon.speed,
            weapon.damage,
            tag.0,
            entity,
            weapon.kind,
            weapon.range / weapon.speed.max(1.0),
            if weapon.kind == BoltKind::Acid { 0.6 } else { 0.35 },
        );
    }
}

/// Fly the bullets, and stop them at the first thing they should stop at.
///
/// Movement is substepped along the frame's segment so that a 95 m/s lance
/// cannot pass through a 1.8m ant between two frames -- at sixty frames a
/// second it would otherwise cover 1.6m per test.
pub fn projectiles(
    clock: Res<SimClock>,
    mut commands: Commands,
    grid: Res<StaticGrid>,
    hash: Res<SpatialHash>,
    sides: Res<Sides>,
    mut bolts: Query<(Entity, &mut Pose, &Velocity, &mut Projectile)>,
    mut victims: Query<(&Pose, &Body, &mut Health), Without<Projectile>>,
    mut candidates: Local<Vec<(Entity, f32)>>,
) {
    let dt = clock.dt();
    for (entity, mut pose, vel, mut bolt) in &mut bolts {
        bolt.life -= dt;
        if bolt.life <= 0.0 {
            commands.entity(entity).despawn();
            continue;
        }

        let travel = vel.0.length() * dt;
        let steps = ((travel / 0.7) as i32 + 1).clamp(1, 8);
        let step_dt = dt / steps as f32;
        let enemies = sides.enemies_of(bolt.faction);
        let mut struck = false;

        for _ in 0..steps {
            pose.pos += vel.0 * step_dt;

            if pose.pos.y <= 0.0 {
                let at = Vec3::new(pose.pos.x, 0.05, pose.pos.z);
                spawn::effect(&mut commands, at, ModelKind::Spark, 0.18, 1.0);
                struck = true;
                break;
            }

            if grid.contains_point(pose.pos.x, pose.pos.y, pose.pos.z) {
                spawn::effect(&mut commands, pose.pos, ModelKind::Spark, 0.22, 1.0);
                struck = true;
                break;
            }

            // The hash is queried at a generous radius and the real test is
            // done on the candidates, because the hash only knows about
            // ground position and a body has height.
            candidates.clear();
            hash.query_radius(
                pose.pos.x,
                pose.pos.z,
                bolt.radius + 2.0,
                &mut candidates,
            );
            let mut best: Option<(Entity, f32)> = None;
            for &(candidate, d2) in candidates.iter() {
                if candidate == bolt.owner || !enemies.contains(&candidate) {
                    continue;
                }
                let Ok((other, body, _)) = victims.get(candidate) else {
                    continue;
                };
                let reach = body.radius + bolt.radius;
                if d2 > reach * reach {
                    continue;
                }
                let low = other.pos.y - bolt.radius;
                let high = other.pos.y + body.height + bolt.radius;
                if pose.pos.y < low || pose.pos.y > high {
                    continue;
                }
                if best.is_none_or(|(_, b)| d2 < b) {
                    best = Some((candidate, d2));
                }
            }

            if let Some((hit, _)) = best {
                if let Ok((_, _, mut health)) = victims.get_mut(hit) {
                    health.damage(bolt.damage);
                }
                spawn::effect(&mut commands, pose.pos, ModelKind::Blood, 0.25, 1.2);
                struck = true;
                break;
            }
        }

        if struck {
            commands.entity(entity).despawn();
        }
    }
}

/// Anything at zero health becomes a corpse, exactly once.
///
/// The Python original scored kills inline, wherever the damage happened,
/// and had to guard every site against counting the same death twice. One
/// system that owns the transition cannot double-count by construction, and
/// it is the shape the ECS wanted anyway: a query for "alive things that are
/// not alive any more".
pub fn deaths(
    mut commands: Commands,
    mut stats: ResMut<Stats>,
    query: Query<(Entity, &Health, &FactionTag, Option<&Player>), Without<Dead>>,
) {
    for (entity, health, tag, player) in &query {
        if health.current > 0.0 {
            continue;
        }
        match tag.0 {
            Faction::Bugs => stats.ants_killed += 1,
            Faction::Edf if player.is_none() => stats.allies_lost += 1,
            Faction::Edf => {}
        }
        commands.entity(entity).insert(Dead::default());
    }
}

// --------------------------------------------------------------------------
// waves and cleanup
// --------------------------------------------------------------------------

/// Keeps ants coming, from off the edge of the player's attention.
///
/// Spawns land on a ring 70-110m out, snapped to street level so nothing
/// materialises inside a building, and the population is capped so the frame
/// time stays flat however long you survive.
pub fn waves(
    clock: Res<SimClock>,
    mut commands: Commands,
    mut rng: ResMut<Rng>,
    mut stats: ResMut<Stats>,
    mut waves: ResMut<Waves>,
    grid: Res<StaticGrid>,
    sides: Res<Sides>,
    player: Res<PlayerEntity>,
    poses: Query<&Pose>,
) {
    waves.timer -= clock.dt();
    if waves.timer > 0.0 {
        return;
    }
    waves.timer = waves.interval;

    // The wave number is a difficulty clock, so it advances on schedule
    // whether or not there is room to spawn into. Only the head count is
    // capped -- otherwise a player who stops killing ants also stops the game
    // getting harder, which is exactly backwards.
    stats.wave += 1;

    let room = waves.cap.saturating_sub(sides.bugs.len());
    if room == 0 {
        return;
    }

    let count = room.min(6 + stats.wave as usize);
    let origin = player
        .0
        .and_then(|p| poses.get(p).ok())
        .map_or(Vec3::ZERO, |pose| pose.pos);

    for _ in 0..count {
        let angle = rng.range(0.0, std::f32::consts::TAU);
        let dist = rng.range(70.0, 110.0);
        let x = (origin.x + angle.cos() * dist).clamp(-ARENA + 6.0, ARENA - 6.0);
        let z = (origin.z + angle.sin() * dist).clamp(-ARENA + 6.0, ARENA - 6.0);
        let spitter = rng.chance(0.22);
        spawn::ant(
            &mut commands,
            Vec3::new(x, grid.height_at(x, z), z),
            spitter,
        );
    }
}

/// Runs the corpse timers and the effect lifetimes, and deletes.
///
/// Corpses keep falling while they rot -- a bug shot out of a leap should
/// finish the arc -- so the physics pass still runs on them; only the brain
/// and the gun are switched off, by the `Dead` checks upstream.
pub fn reap(
    clock: Res<SimClock>,
    mut commands: Commands,
    mut lifetimes: Query<(Entity, &mut Lifetime)>,
    mut corpses: Query<(Entity, &mut Dead, &Pose)>,
    mut wounded: Query<&mut Health>,
) {
    let dt = clock.dt();

    for (entity, mut life) in &mut lifetimes {
        life.remaining -= dt;
        if life.remaining <= 0.0 {
            commands.entity(entity).despawn();
        }
    }

    for (entity, mut dead, pose) in &mut corpses {
        dead.timer -= dt;
        if dead.timer <= 0.0 {
            spawn::effect(&mut commands, pose.pos, ModelKind::Dust, 0.5, 1.4);
            commands.entity(entity).despawn();
        }
    }

    for mut health in &mut wounded {
        health.hurt_flash = (health.hurt_flash - dt).max(0.0);
    }
}
