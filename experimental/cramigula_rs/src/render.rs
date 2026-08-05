//! The renderer: systems that read the simulation and never write it.
//!
//! The split this file exists to enforce is that [`crate::sim`] has no idea
//! any of this is here. Gameplay entities carry a
//! [`Renderable`](crate::components::Renderable), which is an enum and a
//! tint; this module owns the meshes, the materials, the chase camera and the
//! entity hierarchies. Delete every system here and the game still runs --
//! that is what `cargo test --no-default-features` does.
//!
//! Where the Python original kept a side table mapping entity ids to
//! `NodePath`s, Bevy lets the visual *be* the entity: `spawn_visuals` inserts
//! a `Transform`, a `Mesh3d` and a few children onto the same entity the
//! simulation is already stepping. Nothing has to be kept in sync, and
//! nothing has to be cleaned up -- despawning an ant despawns its legs.
//!
//! The city is handled differently. Buildings never move, so rather than one
//! entity each they are merged into one mesh per facade style at start-up.
//! Two hundred boxes become five draw calls.

use bevy::prelude::*;

use crate::components::*;
use crate::models::{self, BoxSpec, MeshBuilder};
use crate::ps1::{self, Ps1Material, Textures};
use crate::sim::{aim_vector, heading_vectors, SimSet, ARENA};
use crate::spatial::StaticGrid;

// Camera. The boom sits behind the player along her aim, offset to one side
// and slightly up so she does not stand in front of the crosshair.
const CAM_STIFFNESS: f32 = 14.0; // e-folds per second for the position spring
const CAM_SNAP: f32 = 30.0; // metres of error that mean "teleported", so do not lerp
const CAM_BOOM: f32 = 7.4; // metres behind the focus at rest
const CAM_BOOM_SPEED: f32 = 0.085; // extra metres per m/s of horizontal speed
const CAM_BOOM_EXTRA: f32 = 3.6; // cap on that extra
const CAM_SHOULDER: f32 = 0.85; // metres right of the aim line
const CAM_LIFT: f32 = 0.42; // metres above it
const CAM_FOCUS_Y: f32 = 1.30; // height up the body the boom pivots about
const CAM_CLEARANCE: f32 = 0.55; // keep this far off the ground and off walls
const CAM_PULL_IN: f32 = 26.0; // metres/sec the boom retracts out of a wall
const CAM_PUSH_OUT: f32 = 7.0; // metres/sec it is allowed back out again

/// Largest polygon, in metres, allowed on a wall or a rooftop. Bounds how
/// badly the affine texture mapping can swim -- see [`models::ground`].
/// Roofs get the finer grid because they are the one large flat surface the
/// player stands on, where a smear is most obvious.
const WALL_CELL: f32 = 8.0;
const ROOF_CELL: f32 = 5.0;

// --------------------------------------------------------------------------
// markers
// --------------------------------------------------------------------------

/// This entity has had its visuals built. Cheaper and clearer than asking
/// whether it happens to have a `Transform` yet.
#[derive(Component, Debug)]
pub struct Visualised;

/// A limb that swings with the walk cycle. `offset` is its phase within the
/// stride, so two racks at 0.0 and 0.5 counter-rotate.
#[derive(Component, Debug)]
pub struct LegRack {
    pub offset: f32,
}

/// The jetpack flame, scaled by [`Flight`].
#[derive(Component, Debug)]
pub struct Thruster;

/// The wing plates, which sweep down on a glide.
#[derive(Component, Debug)]
pub struct Wings;

/// Both arms and the gun, which kick on the muzzle flash.
#[derive(Component, Debug)]
pub struct Arms;

/// The material this actor owns, so its tint can be changed without touching
/// anybody else's.
#[derive(Component, Debug)]
pub struct ActorMaterial(pub Handle<Ps1Material>);

/// Part of the baked static city, so a restart knows what to throw away.
#[derive(Component, Debug)]
pub struct CityGeometry;

/// The chase camera's smoothed state. On the camera entity, so there is no
/// resource to keep in step with which camera is live.
#[derive(Component, Debug)]
pub struct ChaseBoom {
    pub position: Vec3,
    pub length: f32,
    pub started: bool,
}

impl Default for ChaseBoom {
    fn default() -> Self {
        ChaseBoom {
            position: Vec3::new(0.0, 6.0, 10.0),
            length: CAM_BOOM,
            started: false,
        }
    }
}

/// Every mesh and material the renderer hands out.
#[derive(Resource)]
pub struct Assets3d {
    pub ant_body: Handle<Mesh>,
    pub ant_legs: [Handle<Mesh>; 2],
    pub soldier_body: Handle<Mesh>,
    pub diver_body: Handle<Mesh>,
    pub legs: [Handle<Mesh>; 2],
    pub arms: Handle<Mesh>,
    pub wings: Handle<Mesh>,
    pub thruster: Handle<Mesh>,
    pub lance: Handle<Mesh>,
    pub tracer: Handle<Mesh>,
    pub acid: Handle<Mesh>,
    pub puff: Handle<Mesh>,

    pub chitin: Ps1Material,
    pub fatigues: Ps1Material,
    pub armour: Ps1Material,
    pub glow: Ps1Material,
    pub spark: Ps1Material,
}

/// Bolts the renderer onto the simulation.
pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ps1::Ps1Plugin)
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    rebuild_city,
                    spawn_visuals,
                    sync_transforms,
                    topple,
                    animate,
                    update_tints,
                    chase_camera,
                )
                    .chain()
                    .after(SimSet),
            );
    }
}

// --------------------------------------------------------------------------
// one-time construction
// --------------------------------------------------------------------------

pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<Ps1Material>>,
    mut images: ResMut<Assets<Image>>,
) {
    let textures = Textures::paint(&mut images);

    let target = ps1::make_target(&mut images);
    let camera = commands
        .spawn((ps1::scene_camera(target.clone()), ChaseBoom::default()))
        .id();

    // The window draws exactly one thing: the 320x240 frame, blown up. A 2D
    // camera at a later order composites it over whatever is already there.
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            ..Default::default()
        },
        Msaa::Off,
    ));
    commands.spawn((
        Sprite {
            image: target.clone(),
            custom_size: Some(Vec2::new(960.0, 720.0)),
            ..Default::default()
        },
        ps1::BlitCard,
    ));
    commands.insert_resource(ps1::LowResTarget(target));
    commands.insert_resource(SceneCamera(camera));

    // The ground is one mesh that never changes, so it is built here rather
    // than with the city.
    commands.spawn((
        Mesh3d(meshes.add(models::ground(ARENA + 60.0, 6.0))),
        MeshMaterial3d(materials.add(Ps1Material::new(textures.road.clone()))),
        Transform::default(),
    ));

    commands.insert_resource(Assets3d {
        ant_body: meshes.add(models::ant_body()),
        ant_legs: [
            meshes.add(models::ant_legs(1.0)),
            meshes.add(models::ant_legs(-1.0)),
        ],
        soldier_body: meshes.add(models::soldier_body()),
        diver_body: meshes.add(models::diver_body()),
        legs: [meshes.add(models::leg(1.0)), meshes.add(models::leg(-1.0))],
        arms: meshes.add(models::arms()),
        wings: meshes.add(models::wings()),
        thruster: meshes.add(models::thruster()),
        lance: meshes.add(models::bolt(BoltKind::Lance)),
        tracer: meshes.add(models::bolt(BoltKind::Tracer)),
        acid: meshes.add(models::bolt(BoltKind::Acid)),
        puff: meshes.add(models::puff()),

        chitin: Ps1Material::new(textures.chitin.clone()),
        fatigues: Ps1Material::new(textures.fatigues.clone()),
        armour: Ps1Material::new(textures.armour.clone()),
        glow: Ps1Material::emissive(textures.glow.clone()),
        spark: Ps1Material::emissive(textures.white.clone()),
    });
    commands.insert_resource(textures);
}

/// The camera the HUD and the world are drawn through.
#[derive(Resource, Debug, Clone, Copy)]
pub struct SceneCamera(pub Entity);

/// Merge every building into one mesh per facade style, plus one for roofs.
///
/// Buildings are static, so their transforms can be baked into the vertices.
/// The alternative -- two hundred entities each with its own transform --
/// costs a per-frame cull and draw for scenery that will never move.
///
/// Roofs are collected separately from walls. A rooftop is somewhere the Wing
/// Diver actually lands and stands, and a roof wearing the window texture
/// looks like a mistake the moment she does.
///
/// Keyed off `Added<Building>`, so it runs on the frame the city appears and
/// again after every restart, and never in between.
fn rebuild_city(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<Ps1Material>>,
    textures: Option<Res<Textures>>,
    fresh: Query<(), Added<Building>>,
    all: Query<(&Pose, &Building)>,
    old: Query<Entity, With<CityGeometry>>,
) {
    let Some(textures) = textures else { return };
    if fresh.is_empty() {
        return;
    }
    for entity in &old {
        commands.entity(entity).despawn();
    }

    let mut walls: [MeshBuilder; 4] = std::array::from_fn(|_| MeshBuilder::new(1.0 / 16.0));
    let mut roofs = MeshBuilder::new(0.25); // one gravel tile per 4m

    for (pose, building) in &all {
        let style = building.style % walls.len();
        let mut spec = BoxSpec::new(
            Vec3::new(building.half_x * 2.0, building.height, building.half_z * 2.0),
            Vec3::new(pose.pos.x, building.height * 0.5, pose.pos.z),
        );
        spec.skip_bottom = true;
        spec.skip_top = true;
        spec.max_cell = Some(WALL_CELL);
        walls[style].add_box(spec);

        let x0 = pose.pos.x - building.half_x;
        let x1 = pose.pos.x + building.half_x;
        let z0 = pose.pos.z - building.half_z;
        let z1 = pose.pos.z + building.half_z;
        let top = building.height;
        let uv = 0.25;
        roofs.add_quad(
            [
                Vec3::new(x0, top, z1),
                Vec3::new(x1, top, z1),
                Vec3::new(x1, top, z0),
                Vec3::new(x0, top, z0),
            ],
            Vec3::Y,
            [1.0, 1.0, 1.0],
            [
                [x0 * uv, z1 * uv],
                [x1 * uv, z1 * uv],
                [x1 * uv, z0 * uv],
                [x0 * uv, z0 * uv],
            ],
            ((x1 - x0).max(z1 - z0) / ROOF_CELL).ceil().max(1.0) as usize,
        );
    }

    for (style, builder) in walls.into_iter().enumerate() {
        if builder.triangles() == 0 {
            continue;
        }
        commands.spawn((
            Mesh3d(meshes.add(builder.build())),
            MeshMaterial3d(materials.add(Ps1Material::new(textures.facades[style].clone()))),
            Transform::default(),
            CityGeometry,
        ));
    }
    if roofs.triangles() > 0 {
        commands.spawn((
            Mesh3d(meshes.add(roofs.build())),
            MeshMaterial3d(materials.add(Ps1Material::new(textures.gravel.clone()))),
            Transform::default(),
            CityGeometry,
        ));
    }
}

// --------------------------------------------------------------------------
// per-entity visuals
// --------------------------------------------------------------------------

/// Give every newly spawned simulation entity a body.
///
/// Runs on `Without<Visualised>`, so it touches each entity exactly once in
/// its life and does nothing at all on a quiet frame.
fn spawn_visuals(
    mut commands: Commands,
    bank: Option<Res<Assets3d>>,
    mut materials: ResMut<Assets<Ps1Material>>,
    fresh: Query<(Entity, &Renderable), Without<Visualised>>,
) {
    let Some(bank) = bank else { return };
    for (entity, renderable) in &fresh {
        let material = materials.add(match renderable.kind {
            ModelKind::Ant => bank.chitin.clone().tinted(renderable.tint),
            ModelKind::Soldier => bank.fatigues.clone().tinted(renderable.tint),
            ModelKind::WingDiver => bank.armour.clone().tinted(renderable.tint),
            ModelKind::Bolt(_) => bank.glow.clone(),
            _ => bank.spark.clone().tinted(renderable.tint),
        });

        let mut root = commands.entity(entity);
        root.insert((
            Visualised,
            Transform::default(),
            Visibility::default(),
            ActorMaterial(material.clone()),
        ));

        match renderable.kind {
            ModelKind::Ant => {
                root.with_children(|parent| {
                    parent.spawn((
                        Mesh3d(bank.ant_body.clone()),
                        MeshMaterial3d(material.clone()),
                    ));
                    for (index, offset) in [(0usize, 0.0f32), (1, 0.5)] {
                        parent.spawn((
                            Mesh3d(bank.ant_legs[index].clone()),
                            MeshMaterial3d(material.clone()),
                            LegRack { offset },
                        ));
                    }
                });
            }
            ModelKind::Soldier | ModelKind::WingDiver => {
                let diver = renderable.kind == ModelKind::WingDiver;
                let body = if diver {
                    bank.diver_body.clone()
                } else {
                    bank.soldier_body.clone()
                };
                root.with_children(|parent| {
                    parent.spawn((Mesh3d(body), MeshMaterial3d(material.clone())));
                    for (index, offset) in [(0usize, 0.0f32), (1, 0.5)] {
                        parent.spawn((
                            Mesh3d(bank.legs[index].clone()),
                            MeshMaterial3d(material.clone()),
                            // The hip, so a rotation about X swings the leg.
                            Transform::from_xyz(0.0, 0.85, 0.0),
                            LegRack { offset },
                        ));
                    }
                    parent.spawn((
                        Mesh3d(bank.arms.clone()),
                        MeshMaterial3d(material.clone()),
                        Transform::from_xyz(0.0, 1.42, 0.0), // shoulder
                        Arms,
                    ));
                    if diver {
                        parent.spawn((
                            Mesh3d(bank.wings.clone()),
                            MeshMaterial3d(material.clone()),
                            Wings,
                        ));
                        parent.spawn((
                            Mesh3d(bank.thruster.clone()),
                            // Emissive independently of her armour: the flame
                            // is lit whatever the sun is doing.
                            MeshMaterial3d(materials.add(bank.glow.clone())),
                            Transform::from_xyz(0.0, 1.0, 0.0).with_scale(Vec3::splat(0.001)),
                            Thruster,
                        ));
                    }
                });
            }
            ModelKind::Bolt(kind) => {
                let mesh = match kind {
                    BoltKind::Lance => bank.lance.clone(),
                    BoltKind::Acid => bank.acid.clone(),
                    _ => bank.tracer.clone(),
                };
                root.insert((Mesh3d(mesh), MeshMaterial3d(material)));
            }
            ModelKind::Spark | ModelKind::Blood | ModelKind::Dust => {
                root.insert((Mesh3d(bank.puff.clone()), MeshMaterial3d(material)));
            }
        }
    }
}

/// The one place the simulation's `Pose` becomes a Bevy `Transform`.
///
/// Bolts point along their own velocity; everything else uses its heading,
/// pitch and roll. `EulerRot::YXZ` is the order that makes yaw the outermost
/// rotation, which is what "turn, then look up" means.
fn sync_transforms(
    mut query: Query<(
        &Pose,
        &mut Transform,
        Option<&Velocity>,
        Option<&Projectile>,
        Option<&Renderable>,
        Option<&Lifetime>,
    )>,
) {
    for (pose, mut transform, velocity, projectile, renderable, life) in &mut query {
        transform.translation = pose.pos;

        if projectile.is_some() {
            if let Some(velocity) = velocity {
                if velocity.0.length_squared() > 1e-6 {
                    transform.rotation =
                        Transform::default().looking_to(velocity.0, Vec3::Y).rotation;
                }
            }
        } else {
            transform.rotation = Quat::from_euler(
                EulerRot::YXZ,
                pose.heading.to_radians(),
                pose.pitch.to_radians(),
                pose.roll.to_radians(),
            );
        }

        // Effects puff outward and shrink away over their lifetime.
        if let (Some(life), Some(renderable)) = (life, renderable) {
            let age = 1.0 - (life.remaining / life.total).clamp(0.0, 1.0);
            let scale = renderable.scale * (0.35 + 1.9 * age) * (life.remaining * 3.0).max(0.05);
            transform.scale = Vec3::splat(scale.max(0.02));
        }
    }
}

/// The small amount of animation the game has.
///
/// All of it is derived from simulation state that already exists -- the
/// walk phase from [`Gait`], the flame from [`Flight`], the recoil from
/// [`Weapon::muzzle_flash`] -- so nothing here has to be stepped or
/// remembered between frames. That is what makes it safe to run after the
/// simulation and never before.
fn animate(
    time: Res<Time>,
    actors: Query<(&Children, Option<&Gait>, Option<&Flight>, Option<&Weapon>)>,
    mut legs: Query<(&LegRack, &mut Transform), (Without<Thruster>, Without<Wings>, Without<Arms>)>,
    mut thrusters: Query<&mut Transform, (With<Thruster>, Without<Wings>, Without<Arms>)>,
    mut wings: Query<&mut Transform, (With<Wings>, Without<Arms>)>,
    mut arms: Query<&mut Transform, With<Arms>>,
) {
    let flicker = time.elapsed_secs() * 54.0;

    for (children, gait, flight, weapon) in &actors {
        for &child in children {
            if let Ok((rack, mut transform)) = legs.get_mut(child) {
                let angle = gait.map_or(0.0, |g| models::swing(g.phase, g.amplitude, rack.offset));
                transform.rotation = Quat::from_rotation_x(angle);
            }

            if let Ok(mut transform) = thrusters.get_mut(child) {
                let flight = flight.copied().unwrap_or_default();
                transform.scale = if flight.thrusting {
                    Vec3::new(1.0, 0.85 + 0.3 * flicker.sin(), 1.0)
                } else if flight.gliding {
                    // Pilot light: the wings are powered, but she is coasting.
                    Vec3::new(0.45, 0.35, 0.45)
                } else {
                    Vec3::splat(0.001)
                };
            }

            if let Ok(mut transform) = wings.get_mut(child) {
                // A few degrees, but it is the only feedback that the glide
                // is engaged other than the rate the ground is arriving at,
                // and that one is easy to miss.
                let sweep = if flight.is_some_and(|f| f.gliding) { -14.0f32 } else { 0.0 };
                transform.rotation = Quat::from_rotation_x(sweep.to_radians());
            }

            if let Ok(mut transform) = arms.get_mut(child) {
                let kick = weapon
                    .filter(|w| !w.melee)
                    .map_or(0.0, |w| 6.0 + 40.0 * w.muzzle_flash / 0.06);
                transform.rotation = Quat::from_rotation_x(kick.to_radians());
            }
        }
    }
}

/// Corpses darken, the hurt flash whitens, everything else wears its tint.
///
/// Kept out of [`animate`] because it writes to the material assets rather
/// than to transforms, and mixing the two makes the borrow checker's opinion
/// of `animate` considerably more interesting than it needs to be.
fn update_tints(
    mut materials: ResMut<Assets<Ps1Material>>,
    query: Query<(
        &ActorMaterial,
        &Renderable,
        Option<&Health>,
        Option<&Dead>,
    )>,
) {
    for (handle, renderable, health, dead) in &query {
        let Some(material) = materials.get_mut(&handle.0) else {
            continue;
        };
        let [r, g, b] = renderable.tint;
        material.settings.tint = if dead.is_some() {
            Vec4::new(r * 0.45, g * 0.45, b * 0.45, 1.0)
        } else if health.is_some_and(|h| h.hurt_flash > 0.0) {
            Vec4::new(2.4, 2.2, 2.2, 1.0)
        } else {
            Vec4::new(r, g, b, 1.0)
        };
    }
}

/// Corpses topple. Applied to the root rather than to a limb, and after
/// [`sync_transforms`] has written the upright pose it is rolling away from.
pub fn topple(mut query: Query<(&Dead, &mut Transform)>) {
    for (dead, mut transform) in &mut query {
        let fall = (1.0 - dead.timer / 0.6).clamp(0.0, 1.0);
        let angle = 95.0f32.to_radians() * (fall * 3.0).min(1.0);
        transform.rotation *= Quat::from_rotation_z(angle);
    }
}

// --------------------------------------------------------------------------
// the camera
// --------------------------------------------------------------------------

/// A chase camera that looks *along* the aim, not at the player.
///
/// That distinction is the whole design. An obvious-looking chase camera puts
/// itself behind the player and then aims at her -- and the moment you pitch,
/// the centre of the screen is her head rather than the direction the lance
/// travels. The crosshair then lies about where the shot goes, by more the
/// harder you are aiming, which makes the gun feel broken in a way that is
/// very hard to attribute to the camera.
///
/// So the orientation here is set directly from the aim angles and nothing
/// else. The forward vector of `Quat::from_euler(YXZ, yaw, pitch, 0)` is
/// exactly [`aim_vector`], so screen centre *is* the fire direction, by
/// construction. The camera is then free to be shoved anywhere -- pulled out
/// of a wall, lifted off the tarmac -- without the crosshair ever drifting
/// off the shot, because position and orientation are fully decoupled.
///
/// The player is kept out from behind the crosshair by offsetting the boom to
/// the right and up, rather than by aiming somewhere she is not.
pub fn chase_camera(
    // The simulation's clock rather than `Time`: the camera should be smoothed
    // on the same timestep the player moves on, including the stall clamp,
    // and it makes the spring testable against exact sixtieths.
    clock: Res<crate::sim::SimClock>,
    grid: Res<StaticGrid>,
    player: Res<crate::sim::PlayerEntity>,
    poses: Query<(&Pose, &Intent, &Velocity)>,
    mut camera: Query<(&mut Transform, &mut ChaseBoom)>,
    mut scratch: Local<Vec<crate::spatial::Slab>>,
) {
    let dt = clock.dt();
    let Some(entity) = player.0 else { return };
    let Ok((pose, intent, velocity)) = poses.get(entity) else {
        return;
    };
    let Ok((mut transform, mut boom)) = camera.single_mut() else {
        return;
    };

    let focus = pose.pos + Vec3::Y * CAM_FOCUS_Y;
    let direction = aim_vector(intent.aim_yaw, intent.aim_pitch);
    let (_, _, right_x, right_z) = heading_vectors(intent.aim_yaw);
    let right = Vec3::new(right_x, 0.0, right_z);
    // Camera-relative up, so the offsets roll with the pitch instead of
    // sliding the player across the screen as you look down.
    let up = right.cross(direction).normalize_or(Vec3::Y);

    let speed = crate::sim::horizontal_speed(velocity.0);
    let mut wanted = CAM_BOOM + (speed * CAM_BOOM_SPEED).clamp(0.0, CAM_BOOM_EXTRA);
    wanted = wanted.min(clear_distance(&grid, &mut scratch, focus, direction, wanted));

    // Retract fast, extend slow. A camera that springs back out of a wall at
    // the same rate it went in reads as a lurch every time you skim one.
    let rate = if wanted < boom.length {
        CAM_PULL_IN
    } else {
        CAM_PUSH_OUT
    };
    boom.length += (wanted - boom.length).clamp(-rate * dt, rate * dt);

    let mut target = focus - direction * boom.length + right * CAM_SHOULDER + up * CAM_LIFT;
    let floor = grid.height_at(target.x, target.z) + CAM_CLEARANCE;
    target.y = target.y.max(floor);

    if !boom.started || (target - boom.position).length() > CAM_SNAP {
        // First frame, or the player was teleported (a restart, a test
        // harness). Springing across the map takes a visible second.
        boom.position = target;
        boom.started = true;
    } else {
        let error = target - boom.position;
        boom.position += error * (1.0 - (-CAM_STIFFNESS * dt).exp());
    }

    transform.translation = boom.position;
    transform.rotation = Quat::from_euler(
        EulerRot::YXZ,
        intent.aim_yaw.to_radians(),
        intent.aim_pitch.to_radians(),
        0.0,
    );
}

/// How far back the boom can reach before it is inside something.
///
/// Marches out from the player in half-metre steps and stops at the first
/// sample inside a building or under the street. Half a metre is finer than
/// the camera's own smoothing, so the result never quantises visibly, and the
/// march is over a handful of grid cells.
fn clear_distance(
    grid: &StaticGrid,
    scratch: &mut Vec<crate::spatial::Slab>,
    focus: Vec3,
    direction: Vec3,
    wanted: f32,
) -> f32 {
    // A pad in every direction, so the near plane never clips through a wall
    // the boom is merely touching. It is also why the grid is queried as a
    // padded box rather than as a point: a building whose padded edge reaches
    // into the sample's cell still blocks, and a point query would miss it.
    const PAD: f32 = CAM_CLEARANCE;

    let steps = ((wanted / 0.5) as usize).max(2);
    for index in 1..=steps {
        let distance = wanted * index as f32 / steps as f32;
        let sample = focus - direction * distance;
        if sample.y < CAM_CLEARANCE {
            return distance;
        }
        grid.query_aabb(
            sample.x - PAD,
            sample.z - PAD,
            sample.x + PAD,
            sample.z + PAD,
            scratch,
        );
        let blocked = scratch.iter().any(|slab| {
            sample.x >= slab.x0 - PAD
                && sample.x <= slab.x1 + PAD
                && sample.z >= slab.z0 - PAD
                && sample.z <= slab.z1 + PAD
                && sample.y <= slab.top + PAD
        });
        if blocked {
            return (distance - wanted / steps as f32).max(1.5);
        }
    }
    wanted
}
