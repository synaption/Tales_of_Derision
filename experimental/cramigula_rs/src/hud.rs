//! The energy meter, and the four other numbers that matter.
//!
//! Drawn through the *scene* camera, whose render target is the 320x240
//! image (see [`crate::ps1`]), so the HUD is as chunky as the world instead
//! of floating above it at native resolution looking like a different decade.
//! Bevy's UI lays out against the target's size, so "12 pixels" here really
//! is twelve of the console's pixels.
//!
//! The energy bar is the important one. It changes colour on overheat and
//! flashes, because a Wing Diver who has not noticed she is empty is a Wing
//! Diver falling off a roof.

use bevy::prelude::*;

use crate::components::{Energy, Faction, Health};
use crate::sim::{PlayerEntity, SimSet, Sides, Stats};

/// Width of both meters, in the 320-pixel-wide frame.
const BAR_WIDTH: f32 = 248.0;

#[derive(Component)]
struct EnergyFill;

#[derive(Component)]
struct HealthFill;

#[derive(Component)]
struct Scoreboard;

#[derive(Component)]
struct Warning;

/// How long the overheat warning spends lit before it blinks off.
#[derive(Resource, Default)]
struct Blink(f32);

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Blink>()
            // Strictly after the renderer's start-up, which is where the
            // scene camera this UI has to be bound to comes from.
            .add_systems(Startup, build.after(crate::render::setup))
            .add_systems(Update, update.after(SimSet));
    }
}

/// A meter: a dark trough with a bright bar inside it whose width is the
/// value. Two nested nodes rather than a shader, because at 320x240 a
/// rectangle is a rectangle.
fn meter(bottom: f32, height: f32, trough: Color, fill: Color) -> (impl Bundle, impl Bundle) {
    (
        (
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(36.0),
                bottom: Val::Px(bottom),
                width: Val::Px(BAR_WIDTH + 4.0),
                height: Val::Px(height + 4.0),
                padding: UiRect::all(Val::Px(2.0)),
                ..Default::default()
            },
            BackgroundColor(trough),
        ),
        (
            Node {
                width: Val::Px(BAR_WIDTH),
                height: Val::Px(height),
                ..Default::default()
            },
            BackgroundColor(fill),
        ),
    )
}

fn build(mut commands: Commands, camera: Res<crate::render::SceneCamera>) {
    // Everything is parented to one full-screen node bound to the scene
    // camera. Without `UiTargetCamera` the UI would go to the window camera
    // and be drawn at 960x720 over the top of the blit, sharp and wrong.
    let root = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..Default::default()
            },
            UiTargetCamera(camera.0),
        ))
        .id();

    let (energy_trough, energy_fill) = meter(
        14.0,
        8.0,
        Color::srgb(0.05, 0.07, 0.10),
        Color::srgb(0.2, 0.85, 1.0),
    );
    let (health_trough, health_fill) = meter(
        4.0,
        5.0,
        Color::srgb(0.08, 0.04, 0.04),
        Color::srgb(0.9, 0.35, 0.25),
    );

    commands.entity(root).with_children(|parent| {
        parent
            .spawn(energy_trough)
            .with_children(|trough| {
                trough.spawn((energy_fill, EnergyFill));
            });
        parent
            .spawn(health_trough)
            .with_children(|trough| {
                trough.spawn((health_fill, HealthFill));
            });

        parent.spawn((
            Text::new(""),
            TextFont {
                font_size: 11.0,
                ..Default::default()
            },
            TextColor(Color::srgb(0.88, 0.96, 1.0)),
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(6.0),
                top: Val::Px(5.0),
                ..Default::default()
            },
            Scoreboard,
        ));

        parent.spawn((
            Text::new(""),
            TextFont {
                font_size: 12.0,
                ..Default::default()
            },
            TextColor(Color::srgb(1.0, 0.45, 0.3)),
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                bottom: Val::Px(34.0),
                justify_content: JustifyContent::Center,
                ..Default::default()
            },
            TextLayout::new_with_justify(Justify::Center),
            Warning,
        ));

        // Crosshair: two bars crossing at the exact centre of the frame,
        // which is exactly where the lance goes. See
        // [`crate::render::chase_camera`] for why that is true by
        // construction rather than by tuning.
        for (width, height) in [(9.0f32, 1.0f32), (1.0, 9.0)] {
            parent.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Percent(50.0),
                    top: Val::Percent(50.0),
                    width: Val::Px(width),
                    height: Val::Px(height),
                    margin: UiRect {
                        left: Val::Px(-width * 0.5),
                        top: Val::Px(-height * 0.5),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                BackgroundColor(Color::srgba(0.9, 1.0, 0.9, 0.85)),
            ));
        }
    });
}

fn update(
    time: Res<Time>,
    mut blink: ResMut<Blink>,
    stats: Res<Stats>,
    sides: Res<Sides>,
    player: Res<PlayerEntity>,
    meters: Query<(&Energy, &Health)>,
    mut fills: ParamSet<(
        Query<(&mut Node, &mut BackgroundColor), With<EnergyFill>>,
        Query<&mut Node, With<HealthFill>>,
    )>,
    mut scoreboard: Query<&mut Text, (With<Scoreboard>, Without<Warning>)>,
    mut warning: Query<&mut Text, With<Warning>>,
) {
    blink.0 = (blink.0 + time.delta_secs()) % 1.0;

    let alive = player.0.and_then(|entity| meters.get(entity).ok());

    for (mut node, mut colour) in &mut fills.p0() {
        match alive {
            Some((energy, _)) => {
                node.width = Val::Px((BAR_WIDTH * energy.fraction()).max(1.0));
                *colour = if energy.empty {
                    // Overheated: red, and flashing, because you cannot fly.
                    let bright = if blink.0 < 0.5 { 1.0 } else { 0.55 };
                    BackgroundColor(Color::srgb(bright, 0.18 * bright, 0.12 * bright))
                } else {
                    BackgroundColor(Color::srgb(0.2, 0.85, 1.0))
                };
            }
            None => node.width = Val::Px(1.0),
        }
    }

    for mut node in &mut fills.p1() {
        node.width = match alive {
            Some((_, health)) => Val::Px((BAR_WIDTH * health.current / health.maximum).max(1.0)),
            None => Val::Px(1.0),
        };
    }

    if let Ok(mut text) = scoreboard.single_mut() {
        **text = format!(
            "WAVE {}\nANTS {}\nSQUAD {}\nKILLS {}",
            stats.wave,
            sides.of(Faction::Bugs).len(),
            sides.of(Faction::Edf).len().saturating_sub(1),
            stats.ants_killed,
        );
    }

    if let Ok(mut text) = warning.single_mut() {
        **text = match alive {
            None => "K.I.A.".into(),
            Some((energy, _)) if energy.empty => "ENERGY DEPLETED".into(),
            Some(_) => String::new(),
        };
    }
}
