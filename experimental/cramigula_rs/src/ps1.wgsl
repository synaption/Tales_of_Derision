// The PlayStation look, as a pair of shaders.
//
// The console did not have an aesthetic; it had five limitations. Four of
// them are in this file, reproduced as causes rather than faked with a
// post-process filter, because the causes interact -- the wobble only reads
// as wobble because the framebuffer is 320x240, and the texture warp only
// shows up on polygons that span a lot of depth.
//
// See src/ps1.rs for the fifth (no Z-buffer) and why it is deliberately not
// reproduced.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::{position_world_to_clip, position_world_to_view},
}

struct Ps1Settings {
    light_dir: vec3<f32>,
    // 1 on anything that should ignore the sun: bolts, sparks, jet flame. The
    // console had no emissive term either -- it had artists setting vertex
    // colours to white and calling it a day, which is exactly what this is.
    emissive: f32,
    light_color: vec4<f32>,
    ambient: vec4<f32>,
    fog_color: vec4<f32>,
    tint: vec4<f32>,
    // Resolution the vertex snapping quantises to. Set enormous to disable.
    snap: vec2<f32>,
    fog_near: f32,
    fog_far: f32,
}

@group(3) @binding(0) var<uniform> settings: Ps1Settings;
@group(3) @binding(1) var base_texture: texture_2d<f32>;
@group(3) @binding(2) var base_sampler: sampler;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(5) color: vec4<f32>,
};

struct Ps1Vertex {
    @builtin(position) clip_position: vec4<f32>,
    // Limitation 2, in one attribute. `linear` is WGSL's spelling of GLSL's
    // `noperspective`: interpolate in screen space with no divide by w,
    // exactly as the PSX rasteriser did, because it had no divider to spare.
    @location(0) @interpolate(linear) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) view_distance: f32,
};

@vertex
fn vertex(v: Vertex) -> Ps1Vertex {
    var out: Ps1Vertex;

    let world_from_local = mesh_functions::get_world_from_local(v.instance_index);
    let world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(v.position, 1.0),
    );
    var clip = position_world_to_clip(world_position.xyz);

    // Limitation 1: no subpixel precision. The GTE transformed vertices to
    // integer screen coordinates, so geometry shimmers as it moves and shared
    // edges crack open and close.
    //
    // Guarded on w so vertices behind the camera, where the divide flips
    // sign, are left alone rather than folded across the screen.
    if (clip.w > 0.0001) {
        let grid = settings.snap * 0.5;
        let snapped = floor((clip.xy / clip.w) * grid + 0.5) / grid * clip.w;
        clip = vec4<f32>(snapped, clip.z, clip.w);
    }
    out.clip_position = clip;
    out.view_distance = length(position_world_to_view(world_position.xyz));

    // Limitation 4: Gouraud lighting, quantised to four bands. Smooth ramps
    // are a later console's luxury; banding is what the hardware produced.
    let n = normalize(mesh_functions::mesh_normal_local_to_world(v.normal, v.instance_index));
    var lambert = max(dot(n, -normalize(settings.light_dir)), 0.0);
    lambert = floor(lambert * 4.0) / 4.0;
    let lit = mix(
        settings.ambient.rgb + lambert * settings.light_color.rgb,
        vec3<f32>(1.35),
        settings.emissive,
    );

    out.color = vec4<f32>(v.color.rgb * settings.tint.rgb * lit, v.color.a * settings.tint.a);
    out.uv = v.uv;
    return out;
}

// The two halves of the sRGB transfer function.
//
// The quantisation below has to happen in *display* space, not in the linear
// space the rest of the pipeline works in. The console's framebuffer stored
// five bits of an already gamma-encoded signal, so its 32 levels were evenly
// spaced to the eye. Quantising the linear value instead spends almost all
// of the levels on the darks and leaves the bright half of every gradient
// visibly stepped, which looks like a bug rather than like 1998.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let cutoff = c < vec3<f32>(0.0031308);
    let low = c * 12.92;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, cutoff);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = c < vec3<f32>(0.04045);
    let low = c / 12.92;
    let high = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, cutoff);
}

@fragment
fn fragment(in: Ps1Vertex) -> @location(0) vec4<f32> {
    let texel = textureSample(base_texture, base_sampler, in.uv);
    let alpha = texel.a * in.color.a;
    if (alpha < 0.35) {
        discard; // no blending: cut-outs, like the era
    }

    var colour = texel.rgb * in.color.rgb;

    // Limitation 5: hard distance fog, doing the job the draw distance could
    // not.
    let fog = clamp(
        (in.view_distance - settings.fog_near) / (settings.fog_far - settings.fog_near),
        0.0,
        1.0,
    );
    colour = mix(colour, settings.fog_color.rgb, fog);

    // Limitation 3: 15-bit colour. Five bits a channel with a 4x4 ordered
    // dither, which is why period screenshots have that fine crosshatch in
    // every gradient. A function-scope `var` because WGSL will not index a
    // module-level constant array with a value it cannot see at compile time.
    var bayer = array<f32, 16>(
         0.0,  8.0,  2.0, 10.0,
        12.0,  4.0, 14.0,  6.0,
         3.0, 11.0,  1.0,  9.0,
        15.0,  7.0, 13.0,  5.0,
    );
    let cell = vec2<i32>(i32(in.clip_position.x) % 4, i32(in.clip_position.y) % 4);
    let threshold = (bayer[cell.y * 4 + cell.x] / 16.0 - 0.5) / 31.0;

    var display = linear_to_srgb(clamp(colour, vec3<f32>(0.0), vec3<f32>(1.0)));
    display = floor(clamp(display + threshold, vec3<f32>(0.0), vec3<f32>(1.0)) * 31.0 + 0.5) / 31.0;

    return vec4<f32>(srgb_to_linear(display), 1.0);
}
