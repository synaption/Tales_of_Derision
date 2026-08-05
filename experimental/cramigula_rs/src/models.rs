//! Procedural geometry. Every model in the game is built from boxes at start-up.
//!
//! There are no art assets in this project, which is convenient for a
//! repository and, as it happens, historically accurate: a PS1 character was
//! a few hundred triangles and a 64x64 texture, and boxes with a limb helper
//! get you most of the way to one. Everything here is authored in metres,
//! Y-up, with the origin between the feet and the model facing -Z, matching
//! [`crate::components::Pose`].
//!
//! [`MeshBuilder`] accumulates arbitrarily-oriented boxes into a single
//! [`Mesh`], so an ant's six legs are one draw call rather than six. The parts
//! that need to move independently -- leg racks, gun arms, a jetpack flame --
//! are separate meshes on child entities, which is the only reason any model
//! here is more than one entity.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

/// The six faces of a unit cube: outward normal, and the two in-plane axes
/// used to walk its corners. Written out rather than derived because it is
/// read far more often than it is changed.
///
/// The orders give a counter-clockwise winding seen from outside the box,
/// which is what Bevy's default back-face culling wants.
const FACES: [(Vec3, Vec3, Vec3); 6] = [
    (Vec3::Y, Vec3::X, Vec3::NEG_Z),
    (Vec3::NEG_Y, Vec3::NEG_X, Vec3::NEG_Z),
    (Vec3::X, Vec3::NEG_Z, Vec3::Y),
    (Vec3::NEG_X, Vec3::Z, Vec3::Y),
    (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
    (Vec3::Z, Vec3::X, Vec3::Y),
];

/// Accumulates boxes and quads into one triangle list with normals, UVs and
/// vertex colours.
///
/// `uv_scale` is in texture repeats per metre, applied in the plane of each
/// face, so a texture keeps a constant real-world size no matter how the box
/// is proportioned -- the reason one 64x64 facade works on every tower in the
/// city.
pub struct MeshBuilder {
    uv_scale: f32,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

/// How a box should be built. Most callers want [`BoxSpec::new`] and one or
/// two overrides; a struct beats eight positional arguments that are all
/// floats and tuples.
#[derive(Debug, Clone, Copy)]
pub struct BoxSpec {
    pub size: Vec3,
    pub at: Vec3,
    pub colour: [f32; 3],
    /// Applied about `at`. Used by [`MeshBuilder::add_limb`].
    pub rotation: Option<Quat>,
    pub skip_bottom: bool,
    pub skip_top: bool,
    /// Split each face into a grid no coarser than this many metres. See
    /// [`MeshBuilder::add_box`].
    pub max_cell: Option<f32>,
}

impl BoxSpec {
    pub fn new(size: Vec3, at: Vec3) -> Self {
        BoxSpec {
            size,
            at,
            colour: [1.0, 1.0, 1.0],
            rotation: None,
            skip_bottom: false,
            skip_top: false,
            max_cell: None,
        }
    }

    pub fn colour(mut self, colour: [f32; 3]) -> Self {
        self.colour = colour;
        self
    }
}

impl MeshBuilder {
    pub fn new(uv_scale: f32) -> Self {
        MeshBuilder {
            uv_scale,
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            colors: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Add a box, optionally rotated and optionally tessellated.
    ///
    /// `max_cell` is the period-correct fix for affine texture mapping: the
    /// warp is a function of how much depth a single polygon spans, so every
    /// PS1 game chopped its large flat surfaces up. Without it a road or a
    /// rooftop swims so violently it stops reading as a surface at all.
    pub fn add_box(&mut self, spec: BoxSpec) -> &mut Self {
        let half = spec.size * 0.5;
        let scale = self.uv_scale;

        for (normal, u_axis, v_axis) in FACES {
            if spec.skip_bottom && normal.y < -0.5 {
                continue;
            }
            if spec.skip_top && normal.y > 0.5 {
                continue;
            }
            // Corner offsets in the face's own plane, and how far out it sits.
            let depth = normal * half;
            let u_half = (u_axis.abs() * half).element_sum();
            let v_half = (v_axis.abs() * half).element_sum();

            let steps_u = spec
                .max_cell
                .map_or(1, |cell| ((u_half * 2.0 / cell).ceil() as usize).max(1));
            let steps_v = spec
                .max_cell
                .map_or(1, |cell| ((v_half * 2.0 / cell).ceil() as usize).max(1));

            for iu in 0..steps_u {
                for iv in 0..steps_v {
                    let u0 = -u_half + 2.0 * u_half * iu as f32 / steps_u as f32;
                    let u1 = -u_half + 2.0 * u_half * (iu + 1) as f32 / steps_u as f32;
                    let v0 = -v_half + 2.0 * v_half * iv as f32 / steps_v as f32;
                    let v1 = -v_half + 2.0 * v_half * (iv + 1) as f32 / steps_v as f32;

                    let base = self.positions.len() as u32;
                    for (u, v) in [(u0, v0), (u1, v0), (u1, v1), (u0, v1)] {
                        let mut local = depth + u_axis * u + v_axis * v;
                        let mut world_normal = normal;
                        if let Some(rotation) = spec.rotation {
                            local = rotation * local;
                            world_normal = (rotation * normal).normalize();
                        }
                        self.push(
                            spec.at + local,
                            world_normal,
                            [u * scale, v * scale],
                            spec.colour,
                        );
                    }
                    self.indices
                        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
        }
        self
    }

    /// Add a box spanning `start` to `end`. Legs, arms, mandibles.
    ///
    /// The box is built along +Y and rotated onto the segment.
    /// `Quat::from_rotation_arc` picks its own perpendicular axis for the
    /// antiparallel case, so a limb pointing straight down does not
    /// degenerate.
    pub fn add_limb(&mut self, start: Vec3, end: Vec3, thickness: f32, colour: [f32; 3]) -> &mut Self {
        let delta = end - start;
        let length = delta.length();
        if length < 1e-5 {
            return self;
        }
        let mut spec = BoxSpec::new(Vec3::new(thickness, length, thickness), (start + end) * 0.5)
            .colour(colour);
        spec.rotation = Some(Quat::from_rotation_arc(Vec3::Y, delta / length));
        self.add_box(spec)
    }

    /// Add a quad, optionally split into a `steps` x `steps` grid.
    ///
    /// Corners are given counter-clockwise from the origin corner.
    /// Subdivision interpolates position and UV bilinearly, which is exact
    /// for the rectangles this is used on (ground tiles, rooftops) and good
    /// enough for the ones it is not (the Wing Diver's wings, `steps = 1`).
    pub fn add_quad(
        &mut self,
        corners: [Vec3; 4],
        normal: Vec3,
        colour: [f32; 3],
        uvs: [[f32; 2]; 4],
        steps: usize,
    ) -> &mut Self {
        /// Bilinear blend of four corner values.
        fn blend<T>(a: T, b: T, c: T, d: T, s: f32, t: f32) -> T
        where
            T: core::ops::Mul<f32, Output = T> + core::ops::Add<Output = T> + Copy,
        {
            (a * (1.0 - s) + b * s) * (1.0 - t) + (d * (1.0 - s) + c * s) * t
        }

        let steps = steps.max(1);
        let [p0, p1, p2, p3] = corners;
        let [q0, q1, q2, q3] = uvs;
        let uv = |s: f32, t: f32| {
            [
                blend(q0[0], q1[0], q2[0], q3[0], s, t),
                blend(q0[1], q1[1], q2[1], q3[1], s, t),
            ]
        };

        for i in 0..steps {
            for j in 0..steps {
                let (s0, s1) = (i as f32 / steps as f32, (i + 1) as f32 / steps as f32);
                let (t0, t1) = (j as f32 / steps as f32, (j + 1) as f32 / steps as f32);
                let base = self.positions.len() as u32;
                for (s, t) in [(s0, t0), (s1, t0), (s1, t1), (s0, t1)] {
                    self.push(blend(p0, p1, p2, p3, s, t), normal, uv(s, t), colour);
                }
                self.indices
                    .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
        self
    }

    fn push(&mut self, position: Vec3, normal: Vec3, uv: [f32; 2], colour: [f32; 3]) {
        self.positions.push(position.to_array());
        self.normals.push(normal.to_array());
        self.uvs.push(uv);
        self.colors.push([colour[0], colour[1], colour[2], 1.0]);
    }

    pub fn triangles(&self) -> usize {
        self.indices.len() / 3
    }

    /// Bake into a [`Mesh`].
    ///
    /// `RenderAssetUsages::RENDER_WORLD` alone would free the CPU copy after
    /// upload, which is what you want for a mesh nothing ever reads back --
    /// and none of these are read back.
    pub fn build(self) -> Mesh {
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs)
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors)
            .with_inserted_indices(Indices::U32(self.indices))
    }
}

// --------------------------------------------------------------------------
// the models
// --------------------------------------------------------------------------

const DARK: [f32; 3] = [0.55, 0.5, 0.5];

/// A tessellated ground plane.
///
/// Deliberately *not* one big quad. The affine texture mapping warps in
/// proportion to how much depth a single polygon spans, and a ground plane
/// seen from a metre and a half up spans all of it -- one quad turns the road
/// into a smear. Six-metre cells keep the swim to the amount a 1998 game had,
/// which is the amount that reads as charm rather than as breakage. It also
/// gives the vertex snapping something to bite on.
pub fn ground(extent: f32, cell: f32) -> Mesh {
    let steps = ((extent * 2.0 / cell) as usize).max(1);
    let uv = 0.125; // one road tile per 8m
    let mut builder = MeshBuilder::new(1.0);
    builder.add_quad(
        [
            Vec3::new(-extent, 0.0, extent),
            Vec3::new(extent, 0.0, extent),
            Vec3::new(extent, 0.0, -extent),
            Vec3::new(-extent, 0.0, -extent),
        ],
        Vec3::Y,
        [1.0, 1.0, 1.0],
        [
            [-extent * uv, extent * uv],
            [extent * uv, extent * uv],
            [extent * uv, -extent * uv],
            [-extent * uv, -extent * uv],
        ],
        steps,
    );
    builder.build()
}

/// A giant ant's thorax, abdomen, head and mandibles. Faces -Z.
pub fn ant_body() -> Mesh {
    let mut b = MeshBuilder::new(0.9);
    b.add_box(BoxSpec::new(
        Vec3::new(1.15, 1.15, 1.65),
        Vec3::new(0.0, 0.85, 1.05),
    )); // abdomen
    b.add_box(BoxSpec::new(Vec3::new(0.55, 0.5, 0.5), Vec3::new(0.0, 0.85, 0.15)).colour(DARK));
    b.add_box(BoxSpec::new(
        Vec3::new(0.95, 0.9, 1.05),
        Vec3::new(0.0, 0.85, -0.45),
    )); // thorax
    b.add_box(BoxSpec::new(
        Vec3::new(0.85, 0.7, 0.75),
        Vec3::new(0.0, 0.88, -1.25),
    )); // head
    for side in [1.0f32, -1.0] {
        b.add_limb(
            Vec3::new(side * 0.28, 0.75, -1.55),
            Vec3::new(side * 0.5, 0.55, -2.25),
            0.16,
            DARK,
        ); // mandible
        b.add_limb(
            Vec3::new(side * 0.2, 1.15, -1.5),
            Vec3::new(side * 0.55, 1.5, -2.4),
            0.09,
            DARK,
        ); // antenna
    }
    b.build()
}

/// One side's three legs, pivoting about the body centre.
///
/// Counter-rotating two leg racks is the cheapest thing that reads as a gait
/// from more than five metres away, and at 320x240 nothing is ever closer.
pub fn ant_legs(side: f32) -> Mesh {
    let mut b = MeshBuilder::new(0.9);
    for (index, z) in [-0.75f32, -0.15, 0.5].into_iter().enumerate() {
        let reach = 1.05 + index as f32 * 0.12;
        let hip = Vec3::new(side * 0.35, 0.85, z);
        let knee = Vec3::new(side * 0.85, 1.35, z - 0.15);
        let foot = Vec3::new(side * reach, 0.0, z - 0.35 + index as f32 * 0.35);
        b.add_limb(hip, knee, 0.17, DARK);
        b.add_limb(knee, foot, 0.13, DARK);
    }
    b.build()
}

/// Shared torso/head/helmet/backpack block for both EDF body types.
fn humanoid(b: &mut MeshBuilder, pack: [f32; 3]) {
    let skin = [0.75, 0.6, 0.5];
    b.add_box(BoxSpec::new(
        Vec3::new(0.58, 0.66, 0.34),
        Vec3::new(0.0, 1.18, 0.0),
    )); // torso
    b.add_box(BoxSpec::new(
        Vec3::new(0.44, 0.14, 0.30),
        Vec3::new(0.0, 0.85, 0.0),
    )); // belt
    b.add_box(BoxSpec::new(Vec3::new(0.28, 0.28, 0.26), Vec3::new(0.0, 1.62, 0.0)).colour(skin));
    b.add_box(BoxSpec::new(
        Vec3::new(0.4, 0.16, 0.42),
        Vec3::new(0.0, 1.74, -0.02),
    )); // helmet
    b.add_box(BoxSpec::new(Vec3::new(0.36, 0.44, 0.2), Vec3::new(0.0, 1.2, 0.26)).colour(pack));
}

pub fn soldier_body() -> Mesh {
    let mut b = MeshBuilder::new(1.4);
    humanoid(&mut b, [0.5, 0.52, 0.45]);
    b.build()
}

/// The Wing Diver's torso, with the jetpack in place of a rucksack.
pub fn diver_body() -> Mesh {
    let mut b = MeshBuilder::new(1.4);
    humanoid(&mut b, [0.55, 0.6, 0.72]);
    b.build()
}

/// One leg, authored about a hip at the origin so a rotation about X swings it.
pub fn leg(side: f32) -> Mesh {
    let mut b = MeshBuilder::new(1.4);
    b.add_limb(
        Vec3::new(side * 0.16, 0.0, 0.0),
        Vec3::new(side * 0.16, -0.85, 0.0),
        0.24,
        [1.0, 1.0, 1.0],
    );
    b.build()
}

/// Both arms and the rifle in one mesh, so recoil moves the whole lot.
pub fn arms() -> Mesh {
    let mut b = MeshBuilder::new(1.4);
    for side in [1.0f32, -1.0] {
        b.add_limb(
            Vec3::new(side * 0.36, 0.0, 0.0),
            Vec3::new(side * 0.24, -0.16, -0.42),
            0.18,
            [1.0, 1.0, 1.0],
        );
    }
    b.add_box(
        BoxSpec::new(Vec3::new(0.1, 0.16, 0.95), Vec3::new(0.2, -0.16, -0.6))
            .colour([0.35, 0.35, 0.38]),
    );
    b.build()
}

/// A swept plate either side of the pack, two quads per side so it is lit
/// from both directions rather than vanishing when seen from behind.
pub fn wings() -> Mesh {
    let mut b = MeshBuilder::new(1.2);
    for side in [1.0f32, -1.0] {
        let tip = Vec3::new(side * 1.05, 1.75, 0.85);
        let root_high = Vec3::new(side * 0.16, 1.42, 0.42);
        let root_low = Vec3::new(side * 0.16, 1.05, 0.42);
        let tip_low = Vec3::new(side * 0.95, 1.2, 0.8);
        let normal = Vec3::new(0.0, 0.35, 1.0).normalize();
        let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        b.add_quad(
            [root_low, tip_low, tip, root_high],
            normal,
            [0.62, 0.68, 0.82],
            uvs,
            1,
        );
        b.add_quad(
            [root_high, tip, tip_low, root_low],
            -normal,
            [0.45, 0.5, 0.62],
            uvs,
            1,
        );
    }
    b.build()
}

/// Two flame boxes, scaled by [`crate::components::Flight`].
pub fn thruster() -> Mesh {
    let mut b = MeshBuilder::new(1.0);
    for side in [1.0f32, -1.0] {
        b.add_box(
            BoxSpec::new(
                Vec3::new(0.2, 0.7, 0.2),
                Vec3::new(side * 0.22, -0.45, 0.34),
            )
            .colour([1.0, 0.85, 0.5]),
        );
    }
    b.build()
}

/// A bolt. Long and thin along -Z, so `Transform::looking_to(velocity)`
/// points it the right way.
pub fn bolt(kind: crate::components::BoltKind) -> Mesh {
    use crate::components::BoltKind;
    let (size, colour) = match kind {
        BoltKind::Lance => (Vec3::new(0.16, 0.16, 2.4), [1.0, 0.95, 0.6]),
        BoltKind::Acid => (Vec3::splat(0.42), [0.7, 1.0, 0.35]),
        _ => (Vec3::new(0.09, 0.09, 1.1), [1.0, 0.8, 0.45]),
    };
    let mut b = MeshBuilder::new(1.0);
    b.add_box(BoxSpec::new(size, Vec3::ZERO).colour(colour));
    b.build()
}

/// A camera-facing quad, for sparks, blood and dust.
///
/// A cube rather than a billboard: billboarding needs a per-frame rotation
/// against the camera, and a one-metre cube read through a 320x240 buffer
/// with four-band vertex lighting is indistinguishable from one at the sizes
/// these are ever drawn at. The console's own explosions were a handful of
/// rotating quads, and this is cheaper than either.
pub fn puff() -> Mesh {
    let mut b = MeshBuilder::new(1.0);
    b.add_box(BoxSpec::new(Vec3::splat(1.0), Vec3::ZERO));
    b.build()
}

/// Leg angle in radians for a walk-cycle phase. Shared by ants and grunts.
pub fn swing(phase: f32, amplitude: f32, offset: f32) -> f32 {
    ((phase + offset) * std::f32::consts::TAU).sin() * 38.0f32.to_radians() * amplitude
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every attribute has to be the same length or the GPU reads garbage
    /// off the end of the shortest one.
    fn assert_consistent(mesh: &Mesh, name: &str) {
        let count = mesh.count_vertices();
        assert!(count > 0, "{name} came out empty");
        for attribute in [
            Mesh::ATTRIBUTE_POSITION.id,
            Mesh::ATTRIBUTE_NORMAL.id,
            Mesh::ATTRIBUTE_UV_0.id,
            Mesh::ATTRIBUTE_COLOR.id,
        ] {
            let values = mesh
                .attribute(attribute)
                .unwrap_or_else(|| panic!("{name} is missing an attribute"));
            assert_eq!(values.len(), count, "{name} has a ragged attribute");
        }
        let indices = mesh.indices().expect("no index buffer");
        assert_eq!(indices.len() % 3, 0, "{name} has a partial triangle");
        assert!(
            indices.iter().all(|i| i < count),
            "{name} indexes past the end of its vertex buffer"
        );
    }

    #[test]
    fn every_model_is_well_formed() {
        use crate::components::BoltKind;
        for (name, mesh) in [
            ("ant_body", ant_body()),
            ("ant_legs", ant_legs(1.0)),
            ("soldier_body", soldier_body()),
            ("diver_body", diver_body()),
            ("leg", leg(-1.0)),
            ("arms", arms()),
            ("wings", wings()),
            ("thruster", thruster()),
            ("lance", bolt(BoltKind::Lance)),
            ("acid", bolt(BoltKind::Acid)),
            ("puff", puff()),
            ("ground", ground(40.0, 6.0)),
        ] {
            assert_consistent(&mesh, name);
        }
    }

    #[test]
    fn tessellation_actually_subdivides() {
        // Without this the road is one quad and the affine warp turns it into
        // a smear, so it is worth an assertion rather than an assumption.
        let coarse = ground(60.0, 120.0).count_vertices();
        let fine = ground(60.0, 6.0).count_vertices();
        assert!(
            fine > coarse * 100,
            "6m cells gave {fine} vertices against {coarse} for one quad"
        );

        let mut plain = MeshBuilder::new(1.0);
        plain.add_box(BoxSpec::new(Vec3::splat(20.0), Vec3::ZERO));
        let mut split = MeshBuilder::new(1.0);
        let mut spec = BoxSpec::new(Vec3::splat(20.0), Vec3::ZERO);
        spec.max_cell = Some(5.0);
        split.add_box(spec);
        assert_eq!(plain.triangles(), 12, "a box is twelve triangles");
        assert_eq!(
            split.triangles(),
            12 * 16,
            "a 20m box on a 5m grid is 4x4 cells a face"
        );
    }

    #[test]
    fn skipping_faces_removes_exactly_two() {
        let mut all = MeshBuilder::new(1.0);
        all.add_box(BoxSpec::new(Vec3::splat(2.0), Vec3::ZERO));
        let mut walls = MeshBuilder::new(1.0);
        let mut spec = BoxSpec::new(Vec3::splat(2.0), Vec3::ZERO);
        spec.skip_top = true;
        spec.skip_bottom = true;
        walls.add_box(spec);
        assert_eq!(all.triangles() - walls.triangles(), 4);
    }

    #[test]
    fn a_limb_spans_its_endpoints_whichever_way_it_points() {
        // Straight down is the case that degenerates if the rotation is
        // built from a cross product with a parallel up vector -- and the
        // soldier's legs point straight down.
        for end in [Vec3::NEG_Y, Vec3::Y, Vec3::X, Vec3::new(1.0, -2.0, 0.5)] {
            let mut b = MeshBuilder::new(1.0);
            b.add_limb(Vec3::ZERO, end, 0.2, [1.0, 1.0, 1.0]);
            let mesh = b.build();
            let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION.id).unwrap() {
                bevy::mesh::VertexAttributeValues::Float32x3(values) => values.clone(),
                _ => panic!("positions are not f32x3"),
            };
            let reach = positions
                .iter()
                .map(|p| Vec3::from_array(*p).dot(end.normalize()))
                .fold(f32::MIN, f32::max);
            assert!(
                (reach - end.length()).abs() < 0.2,
                "a limb to {end:?} only reached {reach}"
            );
        }
    }
}
