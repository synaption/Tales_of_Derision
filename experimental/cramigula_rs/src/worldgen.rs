//! Builds the city, the squad and the player, deterministically from a seed.
//!
//! Kept apart from [`crate::sim`] because generation is a one-shot that
//! happens before the first frame, and apart from [`crate::render`] because a
//! headless test wants the same streets the game has. The only rule the
//! layout has to respect is that the player's spawn is in the open --
//! everything else is allowed to be ugly, and at 320x240 with fog at 245m, it
//! will be.

use bevy_ecs::prelude::*;
use bevy_math::Vec3;

use crate::components::*;
use crate::rng::Rng;
use crate::sim::{PlayerEntity, Stats, Waves, ARENA};
use crate::spatial::{Slab, SpatialHash, StaticGrid};
use crate::spawn;

/// One city block, kerb to kerb, including the street on two of its sides.
pub const BLOCK: f32 = 46.0;
pub const STREET: f32 = 15.0;

/// Nothing is generated inside this radius of the origin, so the Wing Diver
/// always starts on open tarmac with room to take off.
pub const PLAZA: f32 = 26.0;

/// What to build. Set before the world is generated, and kept afterwards so
/// that pressing R can produce a *different* city rather than the same one.
#[derive(Resource, Debug, Clone, Copy)]
pub struct GameSetup {
    pub seed: u64,
    pub allies: usize,
    pub ants: usize,
}

impl Default for GameSetup {
    fn default() -> Self {
        GameSetup {
            seed: 1234,
            allies: 8,
            ants: 14,
        }
    }
}

/// Set to true to have the next frame throw the world away and rebuild it.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct RestartRequest(pub bool);

/// Fill the arena with boxes on a street grid, registering each with the
/// collision grid as it goes.
///
/// Each block gets one to three towers of differing heights rather than a
/// single slab, which gives the skyline enough variation to read as a city
/// and gives ants enough corners to break around.
pub fn build_city(commands: &mut Commands, grid: &mut StaticGrid, rng: &mut Rng) {
    let usable = BLOCK - STREET; // footprint budget inside one block
    let steps = (ARENA / BLOCK) as i32;

    for gx in -steps..=steps {
        for gz in -steps..=steps {
            let (cx, cz) = (gx as f32 * BLOCK, gz as f32 * BLOCK);
            if (cx * cx + cz * cz).sqrt() < PLAZA + usable * 0.5 {
                continue;
            }
            if rng.chance(0.12) {
                continue; // an empty lot, for variety and for cover
            }

            for _ in 0..rng.between(1, 3) {
                let half_x = rng.range(5.0, usable * 0.5);
                let half_z = rng.range(5.0, usable * 0.5);
                let slack_x = (usable * 0.5 - half_x).max(0.0);
                let slack_z = (usable * 0.5 - half_z).max(0.0);
                let x = cx + rng.range(-slack_x, slack_x);
                let z = cz + rng.range(-slack_z, slack_z);
                if (x * x + z * z).sqrt() < PLAZA {
                    continue;
                }

                // Taller toward the middle of town, with a long tail so the
                // occasional tower breaks the fog line.
                let downtown = 1.0 - ((cx * cx + cz * cz).sqrt() / ARENA).min(1.0);
                let height = rng.range(9.0, 22.0) + downtown * rng.range(4.0, 46.0);
                let style = rng.below(4) as usize;

                spawn::building(commands, Vec3::new(x, 0.0, z), half_x, half_z, height, style);
                grid.add(Slab {
                    x0: x - half_x,
                    z0: z - half_z,
                    x1: x + half_x,
                    z1: z + half_z,
                    top: height,
                });
            }
        }
    }
}

/// Place the player, a squad around them, and an opening ant patrol.
pub fn populate(
    commands: &mut Commands,
    grid: &StaticGrid,
    rng: &mut Rng,
    allies: usize,
    ants: usize,
) -> Entity {
    let player = spawn::player(commands, Vec3::ZERO);

    for index in 0..allies {
        let angle = std::f32::consts::TAU * index as f32 / allies.max(1) as f32
            + rng.range(-0.2, 0.2);
        let dist = rng.range(6.0, 16.0);
        let (x, z) = (angle.cos() * dist, angle.sin() * dist);
        let pos = Vec3::new(x, grid.height_at(x, z), z);
        spawn::ally(commands, pos, pos, rng.range(0.4, 1.4), index);
    }

    for index in 0..ants {
        let angle = rng.range(0.0, std::f32::consts::TAU);
        let dist = rng.range(55.0, 95.0);
        let (x, z) = (angle.cos() * dist, angle.sin() * dist);
        spawn::ant(
            commands,
            Vec3::new(x, grid.height_at(x, z), z),
            index % 5 == 0,
        );
    }

    player
}

/// The whole world, ready to step. No renderer involved.
///
/// Runs at startup and again on every restart. Two independent streams are
/// drawn from the seed -- one for the city, one for the people -- so that
/// changing the squad size does not also shuffle every building, which makes
/// the layout far easier to reason about when a test is failing.
pub fn generate(
    mut commands: Commands,
    setup: Res<GameSetup>,
    mut grid: ResMut<StaticGrid>,
    mut hash: ResMut<SpatialHash>,
    mut rng: ResMut<Rng>,
    mut player: ResMut<PlayerEntity>,
    mut stats: ResMut<Stats>,
    mut waves: ResMut<Waves>,
) {
    *grid = StaticGrid::new(16.0);
    hash.clear();
    *stats = Stats::default();
    *waves = Waves::default();
    // The shared stream the wave spawner and the brains draw from. Seeded
    // from the same number, so a replay of a seed replays the ants too.
    *rng = Rng::new(setup.seed);

    let mut city_rng = Rng::new(setup.seed);
    build_city(&mut commands, &mut grid, &mut city_rng);

    let mut people_rng = Rng::new(setup.seed ^ 0x5EED);
    player.0 = Some(populate(
        &mut commands,
        &grid,
        &mut people_rng,
        setup.allies,
        setup.ants,
    ));
}

/// Throw the world away when [`RestartRequest`] is set, and build a new one.
///
/// Runs before the simulation chain so that the frame a restart lands on is
/// a complete frame of the new world rather than half of each. Everything
/// with a [`Pose`] is scenery, a person or a bullet, so that one query is
/// the whole teardown.
pub fn restart_if_asked(
    mut request: ResMut<RestartRequest>,
    mut setup: ResMut<GameSetup>,
    mut commands: Commands,
    existing: Query<Entity, With<Pose>>,
) {
    if !request.0 {
        return;
    }
    request.0 = false;
    // A different city next time, rather than the same one again.
    setup.seed = setup.seed.wrapping_add(1);
    for entity in &existing {
        commands.entity(entity).despawn();
    }
    commands.run_system_cached(generate);
}
