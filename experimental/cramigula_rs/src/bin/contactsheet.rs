//! Renders six frames of the game and writes them out as one contact sheet.
//!
//! ```text
//! cargo run --release --bin contactsheet
//! ```
//!
//! **This opens no window.** `WinitPlugin` is replaced with a plain schedule
//! runner and the primary window is `None`, so there is no event loop and
//! nothing appears on the desktop -- the whole thing runs against an
//! offscreen wgpu device, which on this machine is lavapipe. Bevy still
//! builds the real pipeline, compiles the real shader and draws the real
//! city; the only thing missing is a surface to present to.
//!
//! That matters for more than politeness. `cramigula_contactsheet.png` is how
//! the art gets reviewed without launching the game, and a shader that has
//! silently stopped being applied looks *nearly* right until you put two
//! frames side by side.
//!
//! The six panels are the six things worth being able to see at a glance:
//! street level, a thrust climb, a glide, the lance, the overheat warning,
//! and a swarm.

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::WindowPlugin;
use bevy::winit::WinitPlugin;

use cramigula::components::{Body, Energy, Intent, Pose, Velocity};
use cramigula::ps1::{LowResTarget, RES_X, RES_Y};
use cramigula::sim::{PlayerEntity, SimClock};
use cramigula::{hud, render, worldgen, GamePlugin};

const COLUMNS: usize = 3;
const OUTPUT: &str = "cramigula_contactsheet.png";

/// One panel: how to set the world up, and how long to let it run.
struct Panel {
    name: &'static str,
    /// Frames of simulation before the shutter, at a sixtieth each.
    frames: usize,
    setup: fn(&mut World),
    intent: Intent,
}

const PANELS: &[Panel] = &[
    Panel {
        name: "street",
        frames: 30,
        setup: |_| {},
        intent: Intent {
            aim_yaw: 25.0,
            aim_pitch: -4.0,
            ..blank()
        },
    },
    Panel {
        name: "thrust",
        frames: 55,
        setup: |_| {},
        intent: Intent {
            aim_yaw: 130.0,
            aim_pitch: -12.0,
            move_y: 1.0,
            thrust: true,
            ..blank()
        },
    },
    Panel {
        name: "glide",
        frames: 70,
        setup: |world| lift(world, 34.0, 16.0),
        intent: Intent {
            aim_yaw: -70.0,
            aim_pitch: -22.0,
            move_y: 1.0,
            glide: true,
            ..blank()
        },
    },
    Panel {
        name: "lance",
        frames: 40,
        setup: |world| lift(world, 9.0, 4.0),
        intent: Intent {
            aim_yaw: 200.0,
            aim_pitch: -6.0,
            fire: true,
            ..blank()
        },
    },
    Panel {
        name: "overheat",
        frames: 20,
        setup: |world| {
            lift(world, 2.0, 0.0);
            if let Some(player) = world.resource::<PlayerEntity>().0 {
                if let Some(mut energy) = world.get_mut::<Energy>(player) {
                    energy.current = 0.0;
                    energy.empty = true;
                }
            }
        },
        intent: Intent {
            aim_yaw: 15.0,
            aim_pitch: -8.0,
            ..blank()
        },
    },
    Panel {
        name: "swarm",
        frames: 600,
        setup: |_| {},
        intent: Intent {
            aim_yaw: 90.0,
            aim_pitch: -10.0,
            move_y: 1.0,
            fire: true,
            ..blank()
        },
    },
];

/// `Intent::default()` is not `const`, and every panel wants the same base.
const fn blank() -> Intent {
    Intent {
        move_x: 0.0,
        move_y: 0.0,
        aim_yaw: 0.0,
        aim_pitch: 0.0,
        thrust: false,
        glide: false,
        fire: false,
    }
}

/// Put the player in the air with some forward speed, so the panels that are
/// about flight are not all taken from the pavement.
fn lift(world: &mut World, height: f32, speed: f32) {
    let Some(player) = world.resource::<PlayerEntity>().0 else {
        return;
    };
    let yaw = world
        .get::<Intent>(player)
        .map_or(0.0, |intent| intent.aim_yaw);
    let forward = cramigula::sim::aim_vector(yaw, 0.0);
    if let Some(mut pose) = world.get_mut::<Pose>(player) {
        pose.pos.y = height;
    }
    if let Some(mut velocity) = world.get_mut::<Velocity>(player) {
        velocity.0 = forward * speed;
    }
    if let Some(mut body) = world.get_mut::<Body>(player) {
        body.grounded = false;
    }
}

/// Where the driver has got to.
#[derive(Resource)]
struct Sheet {
    panel: usize,
    /// Frames still to simulate before the shutter for the current panel.
    remaining: usize,
    /// Whether the current panel's setup has been applied yet.
    started: bool,
    /// Set while a capture is in flight, so the driver waits rather than
    /// racing ahead and photographing the next panel's setup.
    waiting: bool,
    /// One 320x240 RGBA buffer per finished panel.
    captured: Vec<Vec<u8>>,
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .build()
                // No event loop and no window: the entire run is offscreen.
                .disable::<WinitPlugin>()
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: bevy::window::ExitCondition::DontExit,
                    ..Default::default()
                })
                .set(ImagePlugin::default_nearest()),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(
            std::time::Duration::from_secs_f32(1.0 / 240.0),
        ))
        .add_plugins((GamePlugin, render::RenderPlugin, hud::HudPlugin))
        .insert_resource(worldgen::GameSetup {
            seed: 8080,
            allies: 8,
            ants: 18,
        })
        .insert_resource(Sheet {
            panel: 0,
            remaining: PANELS[0].frames,
            started: false,
            waiting: false,
            captured: Vec::new(),
        })
        .add_systems(Update, drive.before(cramigula::sim::SimSet))
        .run();
}

/// Step the world, take the shot, move on.
///
/// An exclusive system because the panel setups need the whole world, and
/// there are six of them rather than sixty thousand.
fn drive(world: &mut World) {
    let (index, remaining, started, waiting, captured) = {
        let sheet = world.resource::<Sheet>();
        (
            sheet.panel,
            sheet.remaining,
            sheet.started,
            sheet.waiting,
            sheet.captured.len(),
        )
    };

    // A screenshot is a round trip through the render world and does not land
    // on the frame it was asked for, so the driver stops until it arrives
    // rather than racing ahead and photographing the next panel's setup.
    if waiting {
        return;
    }

    if index >= PANELS.len() {
        if captured == PANELS.len() {
            match compose(&world.resource::<Sheet>().captured) {
                Ok(path) => println!("wrote {path}"),
                Err(error) => eprintln!("contactsheet: {error}"),
            }
            world.write_message(AppExit::Success);
            world.resource_mut::<Sheet>().panel = usize::MAX; // do not write it twice
        }
        return;
    }

    let panel = &PANELS[index];
    if !started {
        (panel.setup)(world);
        world.resource_mut::<Sheet>().started = true;
    }

    world.resource_mut::<SimClock>().set(1.0 / 60.0);
    if let Some(player) = world.resource::<PlayerEntity>().0 {
        if let Some(mut intent) = world.get_mut::<Intent>(player) {
            *intent = panel.intent;
        }
    }

    if remaining > 0 {
        world.resource_mut::<Sheet>().remaining -= 1;
        return;
    }

    let Some(target) = world.get_resource::<LowResTarget>().map(|t| t.0.clone()) else {
        return;
    };
    world.resource_mut::<Sheet>().waiting = true;
    world
        .spawn(Screenshot::image(target))
        .observe(|capture: On<ScreenshotCaptured>, mut sheet: ResMut<Sheet>| {
            sheet.captured.push(capture.image.data.clone().unwrap_or_default());
            sheet.panel += 1;
            sheet.started = false;
            sheet.waiting = false;
            if sheet.panel < PANELS.len() {
                sheet.remaining = PANELS[sheet.panel].frames;
            }
        });
}

/// Tile the panels into one image with a one-pixel rule between them, and
/// write it as a PNG.
fn compose(panels: &[Vec<u8>]) -> Result<String, String> {
    let rows = panels.len().div_ceil(COLUMNS);
    let (pw, ph) = (RES_X as usize, RES_Y as usize);
    let gap = 2usize;
    let width = COLUMNS * pw + (COLUMNS - 1) * gap;
    let height = rows * ph + (rows.saturating_sub(1)) * gap;

    let mut sheet = vec![0u8; width * height * 4];
    for (index, pixels) in panels.iter().enumerate() {
        if pixels.len() < pw * ph * 4 {
            return Err(format!(
                "panel {} came back with {} bytes, expected {}",
                PANELS[index].name,
                pixels.len(),
                pw * ph * 4
            ));
        }
        let ox = (index % COLUMNS) * (pw + gap);
        let oy = (index / COLUMNS) * (ph + gap);
        for y in 0..ph {
            let src = y * pw * 4;
            let dst = ((oy + y) * width + ox) * 4;
            sheet[dst..dst + pw * 4].copy_from_slice(&pixels[src..src + pw * 4]);
        }
    }

    write_png(OUTPUT, width as u32, height as u32, &sheet)?;
    Ok(OUTPUT.to_string())
}

/// Write an 8-bit RGBA PNG, by hand.
///
/// A PNG is a signature, three chunks and a zlib stream, and `flate2` is
/// already in the dependency graph via Bevy -- but reaching into a transitive
/// dependency is how a build breaks on somebody else's machine. Stored
/// (uncompressed) deflate blocks are perfectly legal zlib, which makes the
/// whole encoder forty lines and no new dependency. The sheet is 200KB
/// instead of 40KB, once, on a developer's machine.
fn write_png(path: &str, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &byte in bytes {
            crc ^= byte as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }

    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
    }

    // Scanlines with filter type 0, which is what the Adler sum runs over.
    let mut raw = Vec::with_capacity((width * height * 4 + height) as usize);
    for y in 0..height as usize {
        raw.push(0);
        let start = y * width as usize * 4;
        raw.extend_from_slice(&rgba[start..start + width as usize * 4]);
    }

    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }

    let mut zlib = vec![0x78, 0x01]; // deflate, 32K window, no preset dictionary
    for (index, block) in raw.chunks(65535).enumerate() {
        let last = (index + 1) * 65535 >= raw.len();
        zlib.push(last as u8);
        zlib.extend_from_slice(&(block.len() as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    zlib.extend_from_slice(&((b << 16) | a).to_be_bytes());

    let mut header = Vec::new();
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, truecolour with alpha

    let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    chunk(&mut png, b"IHDR", &header);
    chunk(&mut png, b"IDAT", &zlib);
    chunk(&mut png, b"IEND", &[]);

    std::fs::write(path, png).map_err(|error| format!("could not write {path}: {error}"))
}
