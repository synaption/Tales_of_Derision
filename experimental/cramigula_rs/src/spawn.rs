//! Entity factories: the only place that knows which components go together.
//!
//! Every one of these is a plain function that composes a bundle and returns
//! an [`Entity`]. There is no `Actor` base type and no `Ant: Enemy` -- an ant
//! is "a thing with a [`Walker`], an [`AntBrain`] and a bite", and a Wing
//! Diver is "a thing with a [`Flight`] and an [`Energy`]". Give a soldier a
//! `Flight` and he flies; that is the whole argument for doing it this way.
//!
//! Tuning constants live next to the factory that uses them so that balance
//! is one file to read.

use bevy_ecs::prelude::*;
use bevy_math::Vec3;

use crate::components::*;

/// The Wing Diver.
///
/// Fragile, fast, and entirely governed by one number. The lance costs 12
/// energy a shot against a 100 pool that also has to lift her -- shooting
/// from the air is the trade the class is built around.
pub fn player(commands: &mut Commands, pos: Vec3) -> Entity {
    commands
        .spawn((
            Player,
            Pose::at(pos),
            Velocity::default(),
            Body {
                radius: 0.55,
                height: 1.75,
                gravity: 22.0,
                ..Default::default()
            },
            Energy::default(),
            Flight::default(),
            Intent::default(),
            Health::new(260.0),
            FactionTag(Faction::Edf),
            Weapon {
                damage: 34.0,
                speed: 95.0,
                cooldown: 0.28,
                range: 160.0,
                energy_cost: 12.0,
                kind: BoltKind::Lance,
                ..Default::default()
            },
            Gait::new(1.8),
            Renderable::new(ModelKind::WingDiver, [0.85, 0.9, 1.0]),
        ))
        .id()
}

/// An EDF grunt.
///
/// Deliberately not very good: slow rifle, wide spread, 90 health. Allies
/// exist so the city sounds inhabited and so the ants have something to eat
/// that is not you.
pub fn ally(commands: &mut Commands, pos: Vec3, rally: Vec3, courage: f32, index: usize) -> Entity {
    commands
        .spawn((
            Pose::at(pos),
            Velocity::default(),
            Body {
                radius: 0.5,
                height: 1.8,
                gravity: 22.0,
                ..Default::default()
            },
            Walker {
                accel: 26.0,
                max_speed: 5.2,
                friction: 11.0,
                turn_rate: 420.0,
                jump_speed: 0.0,
            },
            Intent::default(),
            Health::new(90.0),
            FactionTag(Faction::Edf),
            Weapon {
                damage: 8.0,
                speed: 70.0,
                cooldown: 0.5,
                range: 95.0,
                spread: 3.5,
                kind: BoltKind::Tracer,
                ..Default::default()
            },
            AllyBrain {
                rally,
                courage,
                standoff: 14.0 + 8.0 * courage,
                // Alternate, so a firing line does not drift as one block.
                strafe: if index.is_multiple_of(2) { 0.35 } else { -0.35 },
                ..Default::default()
            },
            Gait::new(2.4),
            Renderable::new(ModelKind::Soldier, [0.72, 0.76, 0.6]),
        ))
        .id()
}

/// A giant ant.
///
/// Fast, numerous, and no smarter than a straight line. Spitters trade the
/// leap for a ranged glob, which is the only thing in the game that reliably
/// punishes hovering.
pub fn ant(commands: &mut Commands, pos: Vec3, spitter: bool) -> Entity {
    let (weapon, tint) = if spitter {
        (
            Weapon {
                damage: 7.0,
                speed: 30.0,
                cooldown: 2.6,
                range: 42.0,
                spread: 2.0,
                kind: BoltKind::Acid,
                ..Default::default()
            },
            [0.55, 0.75, 0.35],
        )
    } else {
        (
            Weapon {
                damage: 9.0,
                cooldown: 1.1,
                range: 2.6,
                melee: true,
                kind: BoltKind::Bite,
                ..Default::default()
            },
            [0.6, 0.22, 0.16],
        )
    };

    commands
        .spawn((
            Pose::at(pos),
            Velocity::default(),
            Body {
                radius: 0.9,
                height: 1.5,
                gravity: 24.0,
                ..Default::default()
            },
            Walker {
                accel: 34.0,
                max_speed: 8.4,
                friction: 12.0,
                turn_rate: 300.0,
                jump_speed: 11.0,
            },
            Intent::default(),
            Health::new(60.0),
            FactionTag(Faction::Bugs),
            weapon,
            AntBrain {
                spitter,
                // A spitter's "bite range" is the distance it is happy to
                // stop and shoot from; a biter's is the reach of its jaws.
                bite_range: if spitter { 34.0 } else { 2.6 },
                ..Default::default()
            },
            Gait::new(3.1),
            Renderable::new(ModelKind::Ant, tint),
        ))
        .id()
}

/// A static box. No health -- the city is scenery, not a target.
///
/// Registration with the collision grid happens in [`crate::worldgen`],
/// because the grid wants every box before it is built and a factory has no
/// business knowing that.
pub fn building(
    commands: &mut Commands,
    pos: Vec3,
    half_x: f32,
    half_z: f32,
    height: f32,
    style: usize,
) -> Entity {
    commands
        .spawn((
            Pose::at(pos),
            Building {
                half_x,
                half_z,
                height,
                style,
            },
        ))
        .id()
}

/// A bullet. Carries no [`Body`] -- it does not walk and it does not fall.
pub fn bolt(
    commands: &mut Commands,
    pos: Vec3,
    velocity: Vec3,
    damage: f32,
    faction: Faction,
    owner: Entity,
    kind: BoltKind,
    life: f32,
    radius: f32,
) -> Entity {
    commands
        .spawn((
            Pose::at(pos),
            Velocity(velocity),
            Projectile {
                damage,
                faction,
                life,
                radius,
                owner,
            },
            Renderable::new(ModelKind::Bolt(kind), [1.0, 1.0, 1.0]),
        ))
        .id()
}

/// A purely cosmetic entity: impact spark, dust puff, blood.
///
/// Effects are entities rather than a renderer-private particle list so that
/// a headless run can count them, which is how the tests check that a hit
/// actually registered somewhere visible.
pub fn effect(commands: &mut Commands, pos: Vec3, kind: ModelKind, life: f32, scale: f32) -> Entity {
    commands
        .spawn((
            Pose::at(pos),
            Lifetime::new(life),
            Renderable {
                kind,
                scale,
                tint: [1.0, 1.0, 1.0],
            },
        ))
        .id()
}
