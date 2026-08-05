// The whole file is renderer tests, and `--no-default-features` compiles the
// crate without a renderer to compile them against. An integration test file
// is always built, so the gate has to be here rather than in Cargo.toml.
#![cfg(feature = "render")]

//! The renderer, tested without a window.
//!
//! ```text
//! cargo test --test rendered
//! ```
//!
//! Two kinds of test live here, and neither opens a window.
//!
//! The camera tests need no GPU at all: [`chase_camera`] is an ordinary
//! system over `Pose`, `Intent` and the collision grid, so it runs in a bare
//! `App` alongside the simulation and can be stepped a thousand exact frames.
//!
//! The pipeline test at the bottom does build a real wgpu device -- offscreen,
//! with no primary window and no event loop -- because there is one property
//! of this renderer that only the GPU can confirm: that the PS1 shader is
//! actually running. In the Python original that shader was silently
//! discarded by a stray call and the scene rendered through fixed-function,
//! looking *nearly* right. The thing that caught it was a test measuring "is
//! the output quantised to 32 levels a channel", which is the test at the end
//! of this file.

use bevy::prelude::*;
use bevy_math::Vec3;

use cramigula::components::{Body, Intent, Pose, Velocity};
use cramigula::render::{chase_camera, ChaseBoom};
use cramigula::sim::{aim_vector, PlayerEntity, SimClock, SimSet};
use cramigula::spatial::{Slab, StaticGrid};
use cramigula::worldgen::GameSetup;
use cramigula::GamePlugin;

const FRAME: f32 = 1.0 / 60.0;

/// A generated city, the simulation, and the chase camera -- and nothing that
/// could put a window on anybody's desktop.
fn camera_app(seed: u64) -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(GamePlugin);
    app.insert_resource(GameSetup {
        seed,
        allies: 4,
        ants: 6,
    });
    app.add_systems(Update, chase_camera.after(SimSet));
    app.update();

    let camera = app
        .world_mut()
        .spawn((Transform::default(), ChaseBoom::default()))
        .id();
    (app, camera)
}

/// Step, holding the player's intent.
fn step(app: &mut App, frames: usize, intent: Intent) {
    for _ in 0..frames {
        app.world_mut().resource_mut::<SimClock>().set(FRAME);
        if let Some(player) = app.world().resource::<PlayerEntity>().0 {
            if let Some(mut slot) = app.world_mut().get_mut::<Intent>(player) {
                *slot = intent;
            }
        }
        app.update();
    }
}

fn camera_transform(app: &App, camera: Entity) -> Transform {
    *app.world().get::<Transform>(camera).expect("no camera")
}

// --------------------------------------------------------------------------
// the camera
// --------------------------------------------------------------------------

#[test]
fn the_crosshair_points_where_the_lance_goes() {
    // The whole design of the camera in one assertion. The centre of the
    // screen must be the fire direction to within a rounding error, at every
    // aim including the steep ones where a `look_at`-based camera diverges
    // most.
    let (mut app, camera) = camera_app(11);
    for (yaw, pitch) in [
        (0.0f32, 0.0f32),
        (37.0, -18.0),
        (-140.0, 42.0),
        (95.0, -74.0),
        (-12.0, 79.0),
    ] {
        step(
            &mut app,
            20,
            Intent {
                aim_yaw: yaw,
                aim_pitch: pitch,
                ..Default::default()
            },
        );
        let forward = camera_transform(&app, camera).forward().as_vec3();
        let wanted = aim_vector(yaw, pitch);
        let error = forward.dot(wanted).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(
            error < 0.05,
            "at yaw {yaw} pitch {pitch} the camera looked {error:.4} degrees off the shot"
        );
    }
}

#[test]
fn the_player_is_offset_out_from_behind_the_crosshair() {
    // If she sat on the aim line she would cover the reticle. The boom is
    // pushed right and up instead of the camera being aimed somewhere she is
    // not -- so she must be visibly off centre, but still on screen.
    let (mut app, camera) = camera_app(12);
    step(&mut app, 60, Intent::default());

    let player = app.world().resource::<PlayerEntity>().0.unwrap();
    let head = app.world().get::<Pose>(player).unwrap().pos + Vec3::Y * 1.3;
    let transform = camera_transform(&app, camera);
    let to_player = (head - transform.translation).normalize();
    let offset = to_player
        .dot(transform.forward().as_vec3())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();

    assert!(offset > 3.0, "the player sat on the crosshair: {offset:.2} degrees off");
    assert!(
        offset < 30.0,
        "the player was pushed {offset:.2} degrees off centre, which is off the side of a 72-degree frame"
    );
}

#[test]
fn the_boom_stays_out_of_the_ground_and_the_walls() {
    // A lap of the city at street level, which is where a chase camera
    // spends its time inside things.
    let (mut app, camera) = camera_app(4242);
    let mut worst_depth: f32 = 0.0;

    for frame in 0..900 {
        let yaw = frame as f32 * 0.8;
        step(
            &mut app,
            1,
            Intent {
                move_y: 1.0,
                aim_yaw: yaw,
                aim_pitch: -6.0,
                ..Default::default()
            },
        );

        let position = camera_transform(&app, camera).translation;
        assert!(
            position.y > 0.0,
            "frame {frame}: the camera went under the street at y = {}",
            position.y
        );

        let grid = app.world().resource::<StaticGrid>();
        for slab in grid.slabs() {
            if slab.contains(position.x, position.z) && position.y < slab.top {
                let depth = (slab.top - position.y)
                    .min(position.x - slab.x0)
                    .min(slab.x1 - position.x)
                    .min(position.z - slab.z0)
                    .min(slab.z1 - position.z);
                worst_depth = worst_depth.max(depth);
            }
        }
    }

    assert!(
        worst_depth < 0.6,
        "the camera got {worst_depth:.2}m inside a building"
    );
}

#[test]
fn the_boom_retracts_faster_than_it_extends() {
    // A camera that springs back out of a wall at the rate it went in lurches
    // every time you skim one, so the two rates are deliberately different.
    // Measured on the boom itself rather than on a screenshot.
    let (mut app, camera) = camera_app(4242);

    // A tower of our own, at a known place, rather than whichever one the
    // seed happened to put nearest -- the geometry of this test is the test.
    app.world_mut().resource_mut::<StaticGrid>().add(Slab {
        x0: -10.0,
        z0: -40.0,
        x1: 10.0,
        z1: -20.0,
        top: 30.0,
    });
    let player = app.world().resource::<PlayerEntity>().0.unwrap();
    app.world_mut().get_mut::<Pose>(player).unwrap().pos = Vec3::new(0.0, 0.0, -18.0);

    // Facing the tower (yaw 0 is -Z) puts the boom behind us in the open
    // street; facing away puts it four metres inside the tower.
    let toward = Intent::default();
    let away = Intent {
        aim_yaw: 180.0,
        ..Default::default()
    };

    step(&mut app, 120, toward);
    let settled = app.world().get::<ChaseBoom>(camera).unwrap().length;
    assert!(
        settled > 6.0,
        "the boom never extended in the open: {settled:.2}"
    );

    step(&mut app, 6, away);
    let pulled_in = app.world().get::<ChaseBoom>(camera).unwrap().length;
    assert!(
        pulled_in < settled - 1.0,
        "the boom did not retract out of a tower: {settled:.2} -> {pulled_in:.2}"
    );

    step(&mut app, 6, toward);
    let pushed_out = app.world().get::<ChaseBoom>(camera).unwrap().length;
    let extend = pushed_out - pulled_in;
    let retract = settled - pulled_in;
    assert!(
        extend < retract,
        "in the same six frames the boom extended {extend:.2}m and retracted {retract:.2}m"
    );
}

#[test]
fn a_teleport_snaps_the_camera_rather_than_flying_it_across_the_map() {
    let (mut app, camera) = camera_app(19);
    step(&mut app, 60, Intent::default());

    let player = app.world().resource::<PlayerEntity>().0.unwrap();
    {
        let mut pose = app.world_mut().get_mut::<Pose>(player).unwrap();
        pose.pos = Vec3::new(150.0, 40.0, -150.0);
        let mut body = app.world_mut().get_mut::<Body>(player).unwrap();
        body.grounded = false;
        let mut velocity = app.world_mut().get_mut::<Velocity>(player).unwrap();
        velocity.0 = Vec3::ZERO;
    }
    step(&mut app, 1, Intent::default());

    let position = camera_transform(&app, camera).translation;
    let head = app.world().get::<Pose>(player).unwrap().pos;
    assert!(
        (position - head).length() < 15.0,
        "one frame after a 200m teleport the camera was {:.1}m away",
        (position - head).length()
    );
}

// --------------------------------------------------------------------------
// the pipeline
// --------------------------------------------------------------------------

/// Render ten frames offscreen and read the 320x240 buffer back.
///
/// Everything about this is windowless: no `WinitPlugin`, no primary window,
/// no event loop. It is the same path `src/bin/contactsheet.rs` takes.
#[cfg(feature = "render")]
mod pipeline {
    use bevy::prelude::*;
    use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
    use bevy::window::{ExitCondition, WindowPlugin};
    use bevy::winit::WinitPlugin;

    use cramigula::components::Intent;
    use cramigula::ps1::{LowResTarget, RES_X, RES_Y};
    use cramigula::sim::{PlayerEntity, SimClock};
    use cramigula::worldgen::GameSetup;
    use cramigula::{hud, render, GamePlugin};

    #[derive(Resource, Default)]
    struct Grab {
        frames: usize,
        asked: bool,
        pixels: Option<Vec<u8>>,
    }

    fn drive(world: &mut World) {
        world.resource_mut::<SimClock>().set(1.0 / 60.0);
        if let Some(player) = world.resource::<PlayerEntity>().0 {
            if let Some(mut intent) = world.get_mut::<Intent>(player) {
                *intent = Intent {
                    aim_yaw: 30.0,
                    aim_pitch: -8.0,
                    ..Default::default()
                };
            }
        }

        let grab = world.resource::<Grab>();
        if grab.asked {
            return;
        }
        // Sixty frames of grace before the shutter. The embedded shader and
        // the painted textures arrive through the asset server over several
        // frames, and photographing the world before its material exists
        // gives a plausible-looking empty sky.
        if grab.frames < 60 {
            world.resource_mut::<Grab>().frames += 1;
            return;
        }

        let Some(target) = world.get_resource::<LowResTarget>().map(|t| t.0.clone()) else {
            return;
        };
        world.resource_mut::<Grab>().asked = true;
        world
            .spawn(Screenshot::image(target))
            .observe(|capture: On<ScreenshotCaptured>, mut grab: ResMut<Grab>| {
                grab.pixels = capture.image.data.clone();
            });
    }

    /// The frame, as RGBA bytes, or `None` if it never arrived.
    ///
    /// Driven with `App::update` rather than `App::run`, for one blunt
    /// reason: `run` moves the world out of the `App`, so anything a test
    /// wanted to read afterwards is gone. Manual updates also mean no
    /// runner plugin and no exit message -- the loop simply stops when the
    /// picture is in hand.
    fn render_one_frame() -> Option<Vec<u8>> {
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .build()
                .disable::<WinitPlugin>()
                // Two tests in one process each build an app, and the second
                // one to install a global tracing subscriber logs an error
                // about it. Neither test wants a log.
                .disable::<bevy::log::LogPlugin>()
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..Default::default()
                })
                .set(ImagePlugin::default_nearest()),
        )
        .add_plugins((GamePlugin, render::RenderPlugin, hud::HudPlugin))
        .insert_resource(GameSetup {
            seed: 8080,
            allies: 6,
            ants: 12,
        })
        .init_resource::<Grab>()
        .add_systems(Update, drive.before(cramigula::sim::SimSet));

        app.finish();
        app.cleanup();
        for _ in 0..400 {
            app.update();
            if app.world().resource::<Grab>().pixels.is_some() {
                break;
            }
        }
        app.world().resource::<Grab>().pixels.clone()
    }

    /// A rectangle of pure world, chosen to touch no part of the HUD.
    ///
    /// The HUD is `bevy_ui` drawing straight into the same 320x240 buffer and
    /// is *not* put through the PS1 shader, so a single leaked pixel of the
    /// scoreboard or the crosshair pushes the level count over 32 and fails
    /// the test for the wrong reason. The window therefore sits right of the
    /// scoreboard (which ends near x = 75), above the warning line and the
    /// meters (which start near y = 190), and clear of the crosshair at the
    /// exact centre of the frame.
    fn world_window(pixels: &[u8]) -> [Vec<u8>; 3] {
        const X: std::ops::Range<u32> = 190..305;
        const Y: std::ops::Range<u32> = 55..150;

        let mut channels = [Vec::new(), Vec::new(), Vec::new()];
        for y in Y {
            for x in X {
                let index = ((y * RES_X + x) * 4) as usize;
                for channel in 0..3 {
                    channels[channel].push(pixels[index + channel]);
                }
            }
        }
        channels
    }

    #[test]
    fn the_frame_is_quantised_to_five_bits_a_channel() {
        let Some(pixels) = render_one_frame() else {
            panic!("no graphics adapter: the renderer could not be tested at all");
        };
        assert_eq!(
            pixels.len(),
            (RES_X * RES_Y * 4) as usize,
            "the buffer is not 320x240 RGBA"
        );

        for (name, values) in ["red", "green", "blue"].iter().zip(world_window(&pixels)) {
            let mut seen = [false; 256];
            for value in &values {
                seen[*value as usize] = true;
            }
            let levels = seen.iter().filter(|s| **s).count();
            assert!(
                levels > 3,
                "the {name} channel has {levels} distinct values -- the frame is a flat colour, \
                 so nothing was drawn"
            );
            assert!(
                levels <= 32,
                "the {name} channel has {levels} distinct values. Five-bit output cannot exceed \
                 32, so the PS1 shader is not the one that drew this frame"
            );
        }
    }

    #[test]
    fn the_frame_has_a_city_in_it() {
        // Quantisation alone would be satisfied by a dithered sky. This is
        // the assertion that geometry actually reached the buffer.
        let Some(pixels) = render_one_frame() else {
            panic!("no graphics adapter: the renderer could not be tested at all");
        };
        let [red, _, blue] = world_window(&pixels);

        let brightest = *red.iter().max().unwrap();
        let darkest = *red.iter().min().unwrap();
        assert!(
            brightest as i32 - darkest as i32 > 40,
            "the world window is nearly uniform ({darkest}..{brightest}): no city was drawn"
        );

        // The sky is a cold blue-grey; lit surfaces are not. If every pixel
        // is still bluer than it is red, we are looking at an empty horizon.
        let warm = red
            .iter()
            .zip(&blue)
            .filter(|(r, b)| **r > **b)
            .count();
        assert!(
            warm > red.len() / 50,
            "only {warm} of {} sampled pixels are warmer than the sky",
            red.len()
        );
    }
}
