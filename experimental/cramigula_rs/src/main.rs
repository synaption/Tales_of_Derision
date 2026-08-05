//! Cramigula -- a Wing Diver, a city, and rather too many ants.
//!
//! ```text
//! cargo run --release
//! cargo run --release -- --seed 7 --ants 24
//! ```
//!
//! An Earth Defence Force flight-class movement model, rendered as if it were
//! 1998, on Bevy's ECS. The three concerns live in separate modules and only
//! meet here:
//!
//! | | |
//! | --- | --- |
//! | [`cramigula::GamePlugin`] | the game, with no renderer at all |
//! | [`cramigula::ps1`] | the look: 320x240, snapped, dithered, foggy |
//! | [`cramigula::render`] | the bridge: systems that read the simulation |
//!
//! # Controls
//!
//! | | |
//! | --- | --- |
//! | Mouse | aim; the screen centre is exactly where the lance goes |
//! | W A S D | move, relative to where you are looking |
//! | Space (hold) | thrust -- drains energy continuously while held |
//! | Shift (hold) | glide -- wings out: the fall slows and the speed keeps |
//! | Left mouse / Ctrl | fire the lance -- costs energy, so it costs altitude |
//! | R | restart with a fresh city |
//! | F1 | toggle the vertex snapping, to see what it is doing |
//! | Escape | release the mouse; again to quit |
//!
//! The whole class is one meter. Flight, gliding and shooting all drink from
//! it, it only refills quickly with your feet on the ground, and if you let
//! it hit exactly zero it locks out and refills at half speed until it is
//! completely full. Everything interesting about playing a Wing Diver is the
//! arithmetic of not letting that happen.
//!
//! The glide is what makes the arithmetic work. Thrust costs 26 a second and
//! gliding costs 7, and a glide never gains a millimetre of height -- so the
//! way to cross the city is one hard burn upward followed by a long flat
//! descent, not a jetpack held down the whole way.

// See the note in `lib.rs`: a Bevy system's argument list is its dependency
// list, and shortening it would only hide what the system reads.
#![allow(clippy::too_many_arguments)]

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow, WindowResolution};

use cramigula::components::{Dead, Intent};
use cramigula::sim::{PlayerEntity, SimClock, SimSet};
use cramigula::{hud, ps1, render, worldgen, GamePlugin};

/// Degrees of aim per pixel of mouse travel.
const MOUSE_SENSITIVITY: f32 = 0.13;

/// Where the mouse has aimed us. Kept out of the player's `Intent` because
/// the aim survives death, a restart and a released cursor, none of which the
/// per-frame intent does.
#[derive(Resource, Debug, Clone, Copy)]
struct Aim {
    yaw: f32,
    pitch: f32,
    captured: bool,
    /// Whether the vertex snapping is on. F1 toggles it.
    snapping: bool,
}

impl Default for Aim {
    fn default() -> Self {
        Aim {
            yaw: 0.0,
            pitch: 8.0,
            captured: true,
            snapping: true,
        }
    }
}

fn main() {
    let setup = parse_args();

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Cramigula".into(),
                        resolution: WindowResolution::new(960, 720),
                        ..Default::default()
                    }),
                    primary_cursor_options: Some(CursorOptions {
                        visible: false,
                        grab_mode: CursorGrabMode::Locked,
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                // Every texture in this game is meant to be blocky, and the
                // one that matters most is the 320x240 frame itself.
                .set(ImagePlugin::default_nearest()),
        )
        .add_plugins((GamePlugin, render::RenderPlugin, hud::HudPlugin))
        .insert_resource(setup)
        .init_resource::<Aim>()
        .add_systems(Update, drive.before(SimSet))
        .add_systems(Update, (grab_cursor, toggle_snapping))
        .run();
}

/// Read the keyboard and the mouse, and write the player's `Intent`.
///
/// This is the entire coupling between the window and the simulation. An
/// ant's brain writes the same six fields, and nothing downstream can tell
/// which of the two produced them.
fn drive(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    mut aim: ResMut<Aim>,
    mut clock: ResMut<SimClock>,
    mut restart: ResMut<worldgen::RestartRequest>,
    player: Res<PlayerEntity>,
    mut intents: Query<(&mut Intent, Option<&Dead>)>,
) {
    clock.set(time.delta_secs());

    if aim.captured && motion.delta != Vec2::ZERO {
        aim.yaw -= motion.delta.x * MOUSE_SENSITIVITY;
        aim.pitch = (aim.pitch - motion.delta.y * MOUSE_SENSITIVITY).clamp(-85.0, 85.0);
    }

    if keys.just_pressed(KeyCode::KeyR) {
        restart.0 = true;
    }

    let Some(entity) = player.0 else { return };
    let Ok((mut intent, dead)) = intents.get_mut(entity) else {
        return;
    };

    intent.clear();
    intent.aim_yaw = aim.yaw;
    intent.aim_pitch = aim.pitch;
    if dead.is_some() {
        return;
    }

    let axis = |positive: KeyCode, negative: KeyCode| {
        (keys.pressed(positive) as i32 - keys.pressed(negative) as i32) as f32
    };
    intent.move_x = axis(KeyCode::KeyD, KeyCode::KeyA);
    intent.move_y = axis(KeyCode::KeyW, KeyCode::KeyS);
    intent.thrust = keys.pressed(KeyCode::Space);
    intent.glide = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    intent.fire = buttons.pressed(MouseButton::Left)
        || keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]);
}

/// Escape hands the mouse back; a second press quits.
///
/// The cursor is locked rather than warped to the centre every frame:
/// `AccumulatedMouseMotion` reports raw deltas, so there is nothing to
/// recentre. Locking is refused by some compositors, in which case the aim
/// still works and the pointer merely escapes the window.
fn grab_cursor(
    keys: Res<ButtonInput<KeyCode>>,
    mut aim: ResMut<Aim>,
    mut exit: MessageWriter<AppExit>,
    mut windows: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if !keys.just_pressed(KeyCode::Escape) {
        return;
    }
    if !aim.captured {
        exit.write(AppExit::Success);
        return;
    }
    aim.captured = false;
    for mut cursor in &mut windows {
        cursor.visible = true;
        cursor.grab_mode = CursorGrabMode::None;
    }
}

/// Turn the vertex grid off, so it is obvious what it was doing.
///
/// Setting the snap resolution to something enormous makes the rounding a
/// no-op without needing a second shader: the frame is still 320x240 and
/// still dithered, only the geometry stops shimmering.
fn toggle_snapping(
    keys: Res<ButtonInput<KeyCode>>,
    mut aim: ResMut<Aim>,
    mut materials: ResMut<Assets<ps1::Ps1Material>>,
) {
    if !keys.just_pressed(KeyCode::F1) {
        return;
    }
    aim.snapping = !aim.snapping;
    let snap = if aim.snapping {
        Vec2::new(ps1::RES_X as f32, ps1::RES_Y as f32)
    } else {
        ps1::SNAP_OFF
    };
    let ids: Vec<_> = materials.ids().collect();
    for id in ids {
        if let Some(material) = materials.get_mut(id) {
            material.settings.snap = snap;
        }
    }
}

/// `--seed N --allies N --ants N`, and nothing else worth a dependency.
fn parse_args() -> worldgen::GameSetup {
    let mut setup = worldgen::GameSetup::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next();
        match (flag.as_str(), value) {
            ("--seed", Some(v)) => setup.seed = v.parse().unwrap_or(setup.seed),
            ("--allies", Some(v)) => setup.allies = v.parse().unwrap_or(setup.allies),
            ("--ants", Some(v)) => setup.ants = v.parse().unwrap_or(setup.ants),
            (other, _) => {
                eprintln!("cramigula: unknown option {other}");
                eprintln!("usage: cramigula [--seed N] [--allies N] [--ants N]");
                std::process::exit(2);
            }
        }
    }
    setup
}
