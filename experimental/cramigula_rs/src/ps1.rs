//! The PlayStation look: one material, one pair of shaders, and a 320x240
//! render target.
//!
//! The console did not have an "aesthetic"; it had five limitations, and
//! every game of the era wore all five. Rather than fake the result with a
//! post-process filter, this module reproduces the causes, because the causes
//! interact -- the wobble only reads as wobble because the framebuffer is
//! 320x240, and the texture warp only shows up on large polygons near the
//! camera.
//!
//! 1. **No subpixel precision.** [`ps1.wgsl`](../src/ps1.wgsl) snaps
//!    clip-space XY to the low-res pixel grid.
//! 2. **No perspective correction.** UVs are declared `@interpolate(linear)`,
//!    so textures swim exactly as they did. This is why large surfaces are
//!    chopped into 5-8m cells in [`crate::models`]: the warp scales with how
//!    much depth a single polygon spans, and every game of the era subdivided
//!    for the same reason.
//! 3. **15-bit colour.** Five bits a channel with a 4x4 Bayer dither, applied
//!    at 320x240 so the crosshatch is coarse and visible.
//! 4. **Vertex lighting**, Gouraud, quantised to four bands.
//! 5. **Hard distance fog**, doing the job the draw distance could not.
//!
//! The one thing deliberately *not* reproduced is the missing Z-buffer. Real
//! hardware sorted per polygon and got it wrong at the seams; a real ordering
//! table would make the city unreadable, and games shipped with that artifact
//! rather than because of it.
//!
//! Everything is rendered into a 320x240 image and blown up with
//! nearest-neighbour sampling, letterboxed to 4:3 so the pixels stay square
//! whatever the window is.

use bevy::asset::{embedded_asset, RenderAssetUsages};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::image::{ImageSampler, ImageSamplerDescriptor};
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AddressMode, AsBindGroup, Extent3d, FilterMode, ShaderType, TextureDimension, TextureFormat,
    TextureUsages,
};
use bevy::shader::ShaderRef;

use crate::rng::Rng;

/// The console's most common mode. 4:3, and small enough that the dither and
/// the vertex snapping are features rather than trivia.
pub const RES_X: u32 = 320;
pub const RES_Y: u32 = 240;
pub const ASPECT: f32 = RES_X as f32 / RES_Y as f32;

/// Fog and sky. A muted teal dusk reads as "the ants came at nightfall" and
/// hides the draw distance without looking like a bug. The components are
/// exact 5-bit levels so the cleared background sits on the same colour grid
/// the dither quantises everything else onto.
pub const SKY: Color = Color::srgb(5.0 / 31.0, 6.0 / 31.0, 8.0 / 31.0);

/// Far enough out that the city reads as a city before it dissolves, close
/// enough that the draw distance never has to be honest about where it ends.
pub const FOG_NEAR: f32 = 70.0;
pub const FOG_FAR: f32 = 245.0;

/// Single directional light, aimed to rake across building faces.
pub const LIGHT_DIR: Vec3 = Vec3::new(-0.45, -0.65, -0.62);
pub const LIGHT_COLOR: Vec4 = Vec4::new(0.95, 0.92, 0.82, 1.0);
pub const AMBIENT: Vec4 = Vec4::new(0.40, 0.42, 0.52, 1.0);

/// Field of view, vertical, in degrees.
///
/// The Panda3D original asked for 72 degrees, but Panda's `setFov` takes the
/// *horizontal* angle and derives the vertical from the aspect ratio, while
/// Bevy's `PerspectiveProjection::fov` is vertical outright. Taking the 72
/// across is the one porting mistake that does not look like a mistake: the
/// scene is merely a quarter smaller, everywhere, and the game feels oddly
/// remote without anything being obviously wrong. This is 72 horizontal at
/// 4:3, expressed the way Bevy wants it.
pub const FOV_VERTICAL: f32 = 57.18;

/// Snap resolution that switches the vertex grid off without a second shader.
/// The frame is still 320x240 and still dithered; only the geometry stops
/// shimmering, which is what makes the comparison worth having on a key.
pub const SNAP_OFF: Vec2 = Vec2::new(1.0e5, 1.0e5);

// --------------------------------------------------------------------------
// the material
// --------------------------------------------------------------------------

/// Everything the shader needs that is not a vertex or a texel.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct Ps1Settings {
    pub light_dir: Vec3,
    pub emissive: f32,
    pub light_color: Vec4,
    pub ambient: Vec4,
    pub fog_color: Vec4,
    pub tint: Vec4,
    pub snap: Vec2,
    pub fog_near: f32,
    pub fog_far: f32,
}

impl Default for Ps1Settings {
    fn default() -> Self {
        Ps1Settings {
            light_dir: LIGHT_DIR,
            emissive: 0.0,
            light_color: LIGHT_COLOR,
            ambient: AMBIENT,
            fog_color: SKY.to_linear().to_vec4(),
            tint: Vec4::ONE,
            snap: Vec2::new(RES_X as f32, RES_Y as f32),
            fog_near: FOG_NEAR,
            fog_far: FOG_FAR,
        }
    }
}

/// One texture, one tint, one lighting mode.
///
/// Every actor gets its own copy of this asset rather than sharing one,
/// because the tint is per-entity and changes every frame that something is
/// hurt, dead or flashing. Sixty small uniform buffers is a rounding error
/// next to the alternative, which is an instance-data path this game has no
/// other use for.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct Ps1Material {
    #[uniform(0)]
    pub settings: Ps1Settings,
    #[texture(1)]
    #[sampler(2)]
    pub texture: Handle<Image>,
}

impl Ps1Material {
    pub fn new(texture: Handle<Image>) -> Self {
        Ps1Material {
            settings: Ps1Settings::default(),
            texture,
        }
    }

    pub fn emissive(texture: Handle<Image>) -> Self {
        let mut material = Ps1Material::new(texture);
        material.settings.emissive = 1.0;
        material
    }

    pub fn tinted(mut self, tint: [f32; 3]) -> Self {
        self.settings.tint = Vec4::new(tint[0], tint[1], tint[2], 1.0);
        self
    }
}

impl Material for Ps1Material {
    fn vertex_shader() -> ShaderRef {
        "embedded://cramigula/ps1.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://cramigula/ps1.wgsl".into()
    }

    /// Opaque, and the cut-outs are done with `discard` in the shader.
    ///
    /// Alpha blending is the wrong tool here twice over: the console could
    /// not do it per-pixel the way a modern blend does, and a blended
    /// material would be sorted into the transparent phase where the vertex
    /// snapping makes the sort order visibly unstable.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        // The shader declares its own vertex inputs, so the pipeline has to be
        // told which mesh attributes feed them. Location 5 for colour is
        // Bevy's own convention, kept so that nothing here has to disagree
        // with the rest of the engine about what a vertex looks like.
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
            Mesh::ATTRIBUTE_COLOR.at_shader_location(5),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

// --------------------------------------------------------------------------
// procedural textures -- no asset files, and the repository stays text
// --------------------------------------------------------------------------

/// A painted image, ready to be handed to [`Assets<Image>`].
struct Canvas {
    size: u32,
    pixels: Vec<[f32; 3]>,
}

impl Canvas {
    fn new(size: u32) -> Self {
        Canvas {
            size,
            pixels: vec![[0.0; 3]; (size * size) as usize],
        }
    }

    fn set(&mut self, x: u32, y: u32, colour: [f32; 3]) {
        let index = (y % self.size * self.size + x % self.size) as usize;
        self.pixels[index] = colour;
    }

    fn get(&self, x: u32, y: u32) -> [f32; 3] {
        self.pixels[(y % self.size * self.size + x % self.size) as usize]
    }

    /// Fill with a flat colour plus per-texel grain, quantised to 5 bits so
    /// the texture is already on the palette the shader will snap it to.
    fn noise(&mut self, rng: &mut Rng, base: [f32; 3], amount: f32) {
        for index in 0..self.pixels.len() {
            let n = rng.range(-amount, amount);
            self.pixels[index] = [
                ((base[0] + n).clamp(0.0, 1.0) * 31.0).round() / 31.0,
                ((base[1] + n).clamp(0.0, 1.0) * 31.0).round() / 31.0,
                ((base[2] + n).clamp(0.0, 1.0) * 31.0).round() / 31.0,
            ];
        }
    }

    /// Wrap as a repeating, nearest-filtered, unmipmapped texture.
    ///
    /// Nearest with no mipmaps is not laziness: the console had neither, and
    /// the crawling aliasing on distant surfaces is a large part of the look.
    fn finish(self, repeat: bool) -> Image {
        let mut data = Vec::with_capacity(self.pixels.len() * 4);
        for pixel in &self.pixels {
            for channel in pixel {
                data.push((channel.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
            data.push(255);
        }
        let mut image = Image::new(
            Extent3d {
                width: self.size,
                height: self.size,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            // sRGB, because the values above were authored by eye. The
            // sampler hands the shader linear light, which is the space the
            // lighting maths wants.
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD,
        );
        let mode = if repeat {
            AddressMode::Repeat
        } else {
            AddressMode::ClampToEdge
        };
        image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: mode.into(),
            address_mode_v: mode.into(),
            mag_filter: FilterMode::Nearest.into(),
            min_filter: FilterMode::Nearest.into(),
            mipmap_filter: FilterMode::Nearest.into(),
            ..Default::default()
        });
        image
    }
}

/// A concrete slab with a grid of lit and unlit windows.
///
/// One texel is about 40cm on the building and the UVs are generated in
/// metres, so towers of every size share one texture with the windows staying
/// the same size on all of them.
fn facade(style: usize) -> Image {
    const SIZE: u32 = 64;
    const PITCH: u32 = 8;
    let tints = [
        [0.52, 0.50, 0.48],
        [0.44, 0.46, 0.50],
        [0.58, 0.52, 0.44],
        [0.40, 0.42, 0.44],
    ];
    let mut rng = Rng::new(9000 + style as u64);
    let mut canvas = Canvas::new(SIZE);
    canvas.noise(&mut rng, tints[style % tints.len()], 0.05);

    let mut wy = 2;
    while wy < SIZE - 2 {
        let mut wx = 2;
        while wx < SIZE - 2 {
            let roll = rng.unit();
            let colour = if roll < 0.18 {
                [0.95, 0.85, 0.55] // someone is still at their desk
            } else if roll < 0.34 {
                [0.30, 0.45, 0.55]
            } else {
                [0.09, 0.10, 0.13]
            };
            for y in wy..(wy + PITCH - 3).min(SIZE) {
                for x in wx..(wx + PITCH - 3).min(SIZE) {
                    canvas.set(x, y, colour);
                }
            }
            wx += PITCH;
        }
        wy += PITCH;
    }
    canvas.finish(true)
}

/// Asphalt with a kerb strip and a dashed centre line, one tile per 8m.
fn road() -> Image {
    const SIZE: u32 = 64;
    let mut rng = Rng::new(4242);
    let mut canvas = Canvas::new(SIZE);
    canvas.noise(&mut rng, [0.20, 0.21, 0.23], 0.045);
    for y in 0..SIZE {
        for x in 0..SIZE {
            if !(3..SIZE - 3).contains(&x) {
                canvas.set(x, y, [0.42, 0.42, 0.40]); // kerb
            }
        }
    }
    let mut y = 4;
    while y < SIZE - 4 {
        for dy in 0..8 {
            for x in (SIZE / 2 - 1)..(SIZE / 2 + 1) {
                canvas.set(x, y + dy, [0.72, 0.68, 0.38]); // dashed centre line
            }
        }
        y += 16;
    }
    canvas.finish(true)
}

/// Blotchy organic-ish texture, used for chitin, uniforms and gravel.
///
/// `grain` is the per-texel noise and `blotch` the amplitude of the overlaid
/// patches. Gravel wants both turned down: it covers whole rooftops at close
/// range, where the same contrast that makes an ant look like chitin makes a
/// roof look like moss.
fn mottle(base: [f32; 3], seed: u64, grain: f32, blotch: f32) -> Image {
    const SIZE: u32 = 32;
    let mut rng = Rng::new(seed);
    let mut canvas = Canvas::new(SIZE);
    canvas.noise(&mut rng, base, grain);
    for _ in 0..SIZE {
        let cx = rng.below(SIZE) as i32;
        let cy = rng.below(SIZE) as i32;
        let radius = rng.between(1, 3);
        let shade = rng.range(-blotch, blotch * 0.85);
        for y in (cy - radius)..=(cy + radius) {
            for x in (cx - radius)..=(cx + radius) {
                if (x - cx).pow(2) + (y - cy).pow(2) > radius * radius {
                    continue;
                }
                let (px, py) = (
                    x.rem_euclid(SIZE as i32) as u32,
                    y.rem_euclid(SIZE as i32) as u32,
                );
                let old = canvas.get(px, py);
                canvas.set(
                    px,
                    py,
                    [
                        (old[0] + shade).clamp(0.0, 1.0),
                        (old[1] + shade).clamp(0.0, 1.0),
                        (old[2] + shade).clamp(0.0, 1.0),
                    ],
                );
            }
        }
    }
    canvas.finish(true)
}

/// A solid colour. Needed because every mesh must carry a texture -- the
/// fragment shader samples its sampler unconditionally.
fn flat(colour: [f32; 3]) -> Image {
    let mut canvas = Canvas::new(4);
    for y in 0..4 {
        for x in 0..4 {
            canvas.set(x, y, colour);
        }
    }
    canvas.finish(false)
}

/// Every texture in the game, painted once at start-up.
#[derive(Resource, Debug)]
pub struct Textures {
    pub facades: [Handle<Image>; 4],
    pub road: Handle<Image>,
    pub gravel: Handle<Image>,
    /// Creature and uniform textures are near-white on purpose: the colour of
    /// an individual comes from its material tint, so one grey mottle serves
    /// red ants, green spitters and olive grunts alike.
    pub chitin: Handle<Image>,
    pub fatigues: Handle<Image>,
    pub armour: Handle<Image>,
    pub white: Handle<Image>,
    pub glow: Handle<Image>,
}

impl Textures {
    pub fn paint(images: &mut Assets<Image>) -> Self {
        Textures {
            facades: std::array::from_fn(|style| images.add(facade(style))),
            road: images.add(road()),
            gravel: images.add(mottle([0.44, 0.43, 0.46], 21, 0.035, 0.045)),
            chitin: images.add(mottle([0.80, 0.76, 0.72], 11, 0.07, 0.11)),
            fatigues: images.add(mottle([0.82, 0.82, 0.78], 13, 0.07, 0.11)),
            armour: images.add(mottle([0.86, 0.88, 0.92], 14, 0.07, 0.11)),
            white: images.add(flat([1.0, 1.0, 1.0])),
            glow: images.add(flat([1.0, 0.95, 0.75])),
        }
    }
}

// --------------------------------------------------------------------------
// the low-res target
// --------------------------------------------------------------------------

/// The 320x240 image everything is drawn into, world and HUD alike.
#[derive(Resource, Debug, Clone)]
pub struct LowResTarget(pub Handle<Image>);

/// Marks the sprite in the window that shows [`LowResTarget`] blown up.
#[derive(Component, Debug)]
pub struct BlitCard;

/// Build the render target: 320x240, nearest-sampled, and never mipmapped.
pub fn make_target(images: &mut Assets<Image>) -> Handle<Image> {
    let mut image = Image::new_fill(
        Extent3d {
            width: RES_X,
            height: RES_Y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC
        | TextureUsages::RENDER_ATTACHMENT;
    image.sampler = ImageSampler::nearest();
    images.add(image)
}

/// The camera settings the whole look depends on, in one place because every
/// one of them is a thing Bevy switches on by default and must not.
///
/// * `Msaa::Off` -- multisampling would smooth exactly the stair-stepped
///   edges the vertex snapping exists to produce.
/// * `Tonemapping::None` -- a filmic curve applied after a five-bit quantise
///   un-quantises it, and the console had no tonemapper.
/// * `DebandDither::Disabled` -- Bevy's own dither fighting the Bayer matrix
///   in the shader gives a mush that is neither.
pub fn scene_camera(target: Handle<Image>) -> impl Bundle {
    (
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(SKY),
            order: 0,
            ..Default::default()
        },
        // Not HDR: the `Hdr` marker component is simply absent. A floating
        // point target would be pointless under a shader whose last act is to
        // round to one of thirty-two levels.
        RenderTarget::from(target),
        Projection::Perspective(PerspectiveProjection {
            fov: FOV_VERTICAL.to_radians(),
            near: 0.35,
            far: 900.0,
            aspect_ratio: ASPECT,
            ..Default::default()
        }),
        Tonemapping::None,
        DebandDither::Disabled,
        Msaa::Off,
        Transform::default(),
    )
}

/// Scale the blit sprite so 320x240 pixels stay square in any window.
///
/// Runs every frame rather than on a resize message: it is two divisions and
/// a compare, and a resize that arrives while the sprite has not been spawned
/// yet is a stretched frame nobody can explain afterwards.
pub fn letterbox(
    windows: Query<&Window>,
    mut cards: Query<&mut Sprite, With<BlitCard>>,
) {
    let Ok(window) = windows.single() else { return };
    let (width, height) = (window.resolution.width(), window.resolution.height());
    if width <= 0.0 || height <= 0.0 {
        return;
    }
    let size = if width / height > ASPECT {
        Vec2::new(height * ASPECT, height)
    } else {
        Vec2::new(width, width / ASPECT)
    };
    for mut sprite in &mut cards {
        sprite.custom_size = Some(size);
    }
}

/// Registers the material, the shader and the letterbox.
pub struct Ps1Plugin;

impl Plugin for Ps1Plugin {
    fn build(&self, app: &mut App) {
        // Embedded rather than loaded from an `assets/` directory: the game
        // is one binary with no data files, and a shader that only loads when
        // the working directory happens to be right is a trap.
        embedded_asset!(app, "ps1.wgsl");
        app.add_plugins(MaterialPlugin::<Ps1Material>::default())
            .add_systems(Update, letterbox);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sky_sits_on_the_five_bit_grid() {
        // The cleared background has to be a colour the dither can also
        // produce, or the horizon shows a seam where the fog stops matching
        // the sky it is fading into.
        let [r, g, b, _] = SKY.to_srgba().to_f32_array();
        for channel in [r, g, b] {
            let level = channel * 31.0;
            assert!(
                (level - level.round()).abs() < 1e-4,
                "{channel} is not an exact 5-bit level"
            );
        }
    }

    #[test]
    fn painted_textures_are_the_size_they_claim() {
        for (name, image, expected) in [
            ("facade", facade(0), 64),
            ("road", road(), 64),
            ("mottle", mottle([0.5; 3], 1, 0.07, 0.11), 32),
            ("flat", flat([1.0; 3]), 4),
        ] {
            let size = image.texture_descriptor.size;
            assert_eq!(size.width, expected, "{name} width");
            assert_eq!(size.height, expected, "{name} height");
            assert_eq!(
                image.data.as_ref().map(|d| d.len()),
                Some((expected * expected * 4) as usize),
                "{name} has the wrong amount of data for its size"
            );
        }
    }

    #[test]
    fn a_facade_has_windows_in_it() {
        // A flat grey wall would still pass every other test in this file,
        // and would look like fog at any distance.
        let image = facade(1);
        let data = image.data.as_ref().unwrap();
        let mut darkest = 255u8;
        let mut brightest = 0u8;
        for texel in data.chunks_exact(4) {
            darkest = darkest.min(texel[0]);
            brightest = brightest.max(texel[0]);
        }
        assert!(
            brightest as i32 - darkest as i32 > 100,
            "the facade has no contrast: {darkest}..{brightest}"
        );
    }

    #[test]
    fn the_settings_block_matches_what_the_shader_declares() {
        use bevy::render::render_resource::ShaderType;
        // std140-ish packing: three vec4s and a vec3+f32 pair, then a vec2
        // and two floats. If this drifts, the shader reads the fog range out
        // of the tint and the whole frame goes strange in a way that is very
        // hard to attribute.
        assert_eq!(Ps1Settings::min_size().get(), 96);
    }
}
