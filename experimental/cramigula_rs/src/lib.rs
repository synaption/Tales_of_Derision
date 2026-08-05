//! Cramigula -- a Wing Diver, a city, and rather too many ants.
//!
//! An Earth Defence Force flight-class movement model, rendered as if it were
//! 1998, on Bevy's ECS. The crate is deliberately in two halves, and the
//! `render` cargo feature is the wall between them:
//!
//! * **The simulation** ([`sim`], [`components`], [`spatial`], [`spawn`],
//!   [`worldgen`], [`rng`]) depends on `bevy_ecs`, `bevy_app`, `bevy_math`
//!   and `bevy_platform`. It cannot open a window, because nothing it can
//!   reach knows how.
//! * **The renderer** ([`render`], [`ps1`], [`models`], [`hud`]) depends on
//!   the full `bevy` facade and is compiled only with `--features render`,
//!   which is on by default.
//!
//! So `cargo test --no-default-features` type-checks and runs the entire game
//! with `bevy_render` not merely unused but absent from the build graph. That
//! is a stronger statement than a naming convention, and it is checked by the
//! compiler on every run.

// A Bevy system's signature *is* its list of dependencies, and a query's type
// *is* the set of components it touches. Both of these lints fire on nearly
// every system here and neither is telling us anything useful: hiding
// `Query<(&mut Pose, &mut Velocity, &mut Body, Option<&Flight>)>` behind an
// alias would conceal the one thing about `physics` worth reading at a
// glance. Bevy's own examples suppress the same two crate-wide.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod components;
pub mod rng;
pub mod sim;
pub mod spatial;
pub mod spawn;
pub mod worldgen;

#[cfg(feature = "render")]
pub mod hud;
#[cfg(feature = "render")]
pub mod models;
#[cfg(feature = "render")]
pub mod ps1;
#[cfg(feature = "render")]
pub mod render;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

/// The game with no way to look at it: world generation plus the simulation.
///
/// This is what a headless test adds to a bare [`App`]. The renderer adds
/// [`render::RenderPlugin`] on top of it and changes nothing about what is
/// below.
pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<worldgen::GameSetup>()
            .init_resource::<worldgen::RestartRequest>()
            .add_plugins(sim::SimPlugin)
            .add_systems(Startup, worldgen::generate)
            .add_systems(Update, worldgen::restart_if_asked.before(sim::SimSet));
    }
}

/// Build a headless game and run world generation, ready to be stepped.
///
/// Used by the tests and by the contact-sheet tool. Nothing here touches a
/// GPU: the returned [`App`] has no renderer, no window and no runner, and
/// `app.update()` advances exactly one simulation frame.
pub fn headless_game(seed: u64, allies: usize, ants: usize) -> App {
    let mut app = App::new();
    app.add_plugins(GamePlugin);
    app.insert_resource(worldgen::GameSetup { seed, allies, ants });
    // The first update runs `Startup`, which is where the city is built.
    app.update();
    app
}

/// Advance a headless game by `frames` steps of `dt`, writing `intent` into
/// the player's [`components::Intent`] at the start of each one.
///
/// The intent is re-applied every frame rather than latched, because that is
/// what a held key does: the keyboard writes the whole `Intent` sixty times
/// a second, and a test that sets it once is testing a different game.
pub fn step(app: &mut App, frames: usize, dt: f32, intent: components::Intent) {
    for _ in 0..frames {
        app.world_mut().resource_mut::<sim::SimClock>().set(dt);
        if let Some(player) = app.world().resource::<sim::PlayerEntity>().0 {
            if let Some(mut slot) = app.world_mut().get_mut::<components::Intent>(player) {
                *slot = intent;
            }
        }
        app.update();
    }
}
