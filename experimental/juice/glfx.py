"""The juice bench on the graphics card: batching, deformation, lighting, post.

This is the same bench as `rogue_juice.py` with the *presentation* moved onto
the GPU. Nothing about the feel changes, because none of the feel was ever in
the renderer -- `juicefx.py` has no pygame in it and therefore no OpenGL in it
either, and the sim, the toggles and the sliders are imported unchanged. That
split was worth having on its own and it pays for itself again here.

What the card actually buys, in the order it matters:

**1. The slime stops being ten strips.** `Jelly` is drawn on the CPU as ten
bands with a surface-tension constraint holding them together, and that whole
design exists only because `Surface.blit` can move rectangles and nothing else.
Here the deformation is a continuous function evaluated per fragment: the
sprite is sampled at `q - lag * f(q)`, where `f` runs smoothly from zero at the
leading edge to one at the trailing edge. There are no bands to tear apart, no
`link` clamp holding them together and no band count to tune. The same springs
drive it; they now drive a field instead of ten samples of one.

**2. Sprites are lit, not tinted.** Each sprite carries a normal map generated
from its own silhouette, so a torch wraps around a creature instead of
uniformly brightening it, and a hit flash is a light *in the room* that shades
everything near it. The software bench multiplies a flat frame by a light map,
which is the best a blit can do and is why it needed a compensating additive
pass; here there is nothing to compensate for.

**3. Pixel art can be crisp *and* move smoothly.** The software bench has to
offer three pixel modes because rounding a blit destination is the only control
it has. A textured quad sampled with a "sharp bilinear" filter -- bilinear
across a texel boundary, flat within it -- is crisp at rest and smooth at any
sub-pixel speed. The three-way trade is a `Surface.blit` artefact, not a
property of pixel art.

**4. Everything is one draw call.** Sprites, particles, shadows, decals,
shockwave rings, weapon arcs and slash cuts are all instanced quads in one
buffer, and the fragment shader branches on a shape id to draw a ring or an arc
procedurally rather than filling a polygon on the CPU. Particle counts stop
being a budget.

And the whole `screen` group of the software bench -- bloom, RGB split,
vignette, heat haze -- collapses into one fragment shader, where it costs
microseconds instead of the better part of a frame.

The panel is the pragmatic exception. Text layout on the GPU means an atlas and
a lot of machinery to reproduce something `pygame.font` already does well, so
the panel and the log are drawn by the *existing* software renderer into an
offscreen surface and uploaded as a texture. That is one upload a frame and it
keeps every slider, blurb and scroll behaviour identical.

Context handling follows `experimental/fantacy_maps/inkfx.py`: under WSL, Mesa
quietly rasterises on the processor unless it is pointed at the card, so the
same check and the same override are applied here.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import moderngl
import numpy as np

# ---------------------------------------------------------------------------
# Getting a context, and getting it on the right device
# ---------------------------------------------------------------------------
#
# Lifted from inkfx.py, which learned it the hard way: a WSL session with a
# perfectly good card will run every shader on the CPU unless Mesa is told
# otherwise, and it decides which driver to use when the *first* context is
# created and never revisits it.

SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast")
WSL_GPU_DEVICE = Path("/dev/dxg")
WSL_GPU_DRIVER = "d3d12"
WSL_GPU_LIBRARIES = "/usr/lib/wsl/lib"
WSL_GPU_MODULE = Path("/usr/lib/x86_64-linux-gnu/dri/d3d12_dri.so")
DRIVER_SETTLED = "JUICEGL_DRIVER_SETTLED"


def wsl_card_available() -> bool:
    """Whether this is a WSL session with a card Mesa could be pointed at.

    Both the device and the driver module have to exist: forcing a driver that
    is not installed leaves Mesa with nowhere to fall back to and the context
    is never created at all.
    """
    if "JUICEGL_DRIVER" in os.environ or os.environ.get("MESA_LOADER_DRIVER_OVERRIDE"):
        return False
    return WSL_GPU_DEVICE.exists() and WSL_GPU_MODULE.exists()


def use_wsl_card() -> None:
    """Point Mesa at the card. Only has any effect before the first context."""
    os.environ[DRIVER_SETTLED] = "1"
    os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = WSL_GPU_DRIVER
    os.environ["GALLIUM_DRIVER"] = WSL_GPU_DRIVER
    libs = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{WSL_GPU_LIBRARIES}:{libs}" if libs else WSL_GPU_LIBRARIES


def is_software(renderer_name: str) -> bool:
    return any(name in renderer_name.lower() for name in SOFTWARE_RENDERERS)


def restart_on_the_card(renderer_name: str) -> None:
    """Relaunch on the graphics card if this is WSL rasterising on the CPU.

    Only ever call this from something that was run as a program: importing a
    module must never re-execute anything.
    """
    if os.environ.get(DRIVER_SETTLED) or not wsl_card_available():
        return
    if not is_software(renderer_name):
        return
    print(f"Rendering on the processor ({renderer_name}); "
          f"restarting on the graphics card.", file=sys.stderr)
    use_wsl_card()
    import pygame
    pygame.quit()
    os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ))


def open_gl_window(size: Tuple[int, int], caption: str = "juice workbench (GL)"):
    """A window with a core 3.3 context. Returns the size actually granted."""
    import pygame
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                    pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 0)
    pygame.display.set_caption(caption)
    pygame.display.set_mode(size, pygame.OPENGL | pygame.DOUBLEBUF)
    return pygame.display.get_window_size()


def create_context(headless: bool = False):
    """A ModernGL context, with or without a window behind it."""
    if headless:
        if wsl_card_available():
            use_wsl_card()
        return moderngl.create_context(standalone=True, backend="egl", require=330)
    return moderngl.create_context(require=330)


# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------
#
# One program draws everything in the world. The vertex shader turns an
# instance into a quad and applies the camera; the fragment shader branches on
# a shape id. Branching in a fragment shader is not free, but it is far cheaper
# than the state changes and draw calls that avoiding it would cost at this
# scale, and it keeps every effect in one buffer in painter's order.

MAX_LIGHTS = 24
MAX_WAVES = 8

QUAD_VS = """
#version 330

// Per-vertex: the unit quad corner, -0.5 .. +0.5.
in vec2 in_corner;

// Per-instance.
in vec2 in_center;      // world pixels
in vec2 in_size;        // world pixels, before any deformation
in float in_rot;        // radians, the sprite's own spin
in vec2 in_uv0;         // atlas cell, top-left
in vec2 in_uv1;         // atlas cell, bottom-right
in vec4 in_color;       // tint, alpha in .a
in float in_flash;      // 0..1 towards white
in vec2 in_lag;         // soft-body displacement of the trailing edge, px
in float in_shape;      // 0 sprite, 1 disc, 2 ring, 3 arc, 4 cut
in vec4 in_params;      // shape-specific

// Camera. Roll and zoom are applied to the *geometry* rather than by
// resampling the finished frame, which is the one place the GL path is not
// merely faster: the software bench has to rotozoom a whole surface and eat
// the filtering, and the edges of the frame come with it.
uniform vec2 u_camera;
uniform vec2 u_half;        // half the viewport, in pixels
uniform float u_roll;       // radians
uniform float u_zoom;
uniform vec2 u_shake;       // pixels

out vec2 v_local;           // quad space, -0.5 .. +0.5 before any bloat
out vec2 v_world;           // world pixels, for lighting
flat out vec2 v_uv0;
flat out vec2 v_uv1;
flat out vec4 v_color;
flat out float v_flash;
flat out vec2 v_lag;        // in quad-space units
flat out float v_shape;
flat out vec4 v_params;

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

void main() {
    // A deformed sprite reaches outside its own tile, so the quad is grown by
    // however far the deformation can push it. Sampling then clips it back to
    // the sprite's real extent -- growing the quad costs nothing, and not
    // growing it would slice the smear off at the tile edge.
    vec2 grow = abs(in_lag) + 2.0;
    vec2 half_size = in_size * 0.5 + grow;
    v_local = in_corner * (half_size / max(in_size * 0.5, vec2(0.5)));

    vec2 local = rot(in_rot) * (in_corner * half_size * 2.0);
    vec2 world = in_center + local;
    v_world = world;

    vec2 rel = (world - u_camera) * u_zoom;
    vec2 screen = rot(u_roll) * rel + u_half + u_shake;
    gl_Position = vec4(screen / u_half - 1.0, 0.0, 1.0);
    gl_Position.y = -gl_Position.y;

    v_uv0 = in_uv0;
    v_uv1 = in_uv1;
    v_color = in_color;
    v_flash = in_flash;
    v_lag = in_lag / max(in_size, vec2(1.0));
    v_shape = in_shape;
    v_params = in_params;
}
"""

QUAD_FS = """
#version 330

in vec2 v_local;
in vec2 v_world;
flat in vec2 v_uv0;
flat in vec2 v_uv1;
flat in vec4 v_color;
flat in float v_flash;
flat in vec2 v_lag;
flat in float v_shape;
flat in vec4 v_params;

uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform vec2 u_atlas_size;

uniform vec4 u_lights[MAX_LIGHTS];      // xy world, z radius, w strength
uniform vec3 u_light_color;
uniform int u_light_count;
uniform vec3 u_ambient;
uniform float u_light_height;           // how far the lights float above the plane
uniform float u_lit;                    // 0 = lighting off, sprites are flat

uniform float u_soft_power;             // shape of the soft-body gradient
uniform float u_soft_pinch;             // cross-axis squeeze per unit of lag
uniform float u_sharpness;              // 0 = bilinear, 1 = sharp bilinear

out vec4 frag_color;

const float PI = 3.14159265;

/* Sharp bilinear.

   Pixel art has two failure modes and every CPU blitter makes you pick one:
   round the destination and slow movement stair-steps, do not round it and
   everything shimmers. Neither is inherent to pixel art -- they are both
   artefacts of only being able to place a sprite on whole pixels.

   This samples flat *inside* a texel and interpolates only across the boundary
   between two, over a distance of one screen pixel. A resting sprite is exactly
   as crisp as nearest-neighbour; a sprite creeping along at a third of a pixel
   a frame moves smoothly, with one blended row of pixels at its edges instead
   of a jump. `fwidth` is what makes it scale-aware: zoom in and the blend
   region stays one screen pixel wide, so it never turns into a blur. */
vec2 sharp_uv(vec2 uv, vec2 tex_size) {
    vec2 p = uv * tex_size;
    vec2 fw = max(fwidth(p), vec2(1e-5));
    // `p` is in texels, and in GL a texel *centre* sits at a half-integer, so
    // the integers are the seams between texels. Snapping to the nearest seam
    // and then clamping the offset one screen pixel either side of it puts
    // every fragment except the ones straddling a seam exactly on a texel
    // centre -- which is nearest sampling -- and blends only across the seam.
    //
    // Snapping to the texel centre instead, which is the intuitive reading,
    // does the exact opposite: it parks every fragment on a seam, where a
    // linear sampler averages two texels, and the "sharp" filter comes out
    // blurrier than plain bilinear at every magnification.
    vec2 seam = floor(p + 0.5);
    vec2 f = clamp((p - seam) / fw, -0.5, 0.5);
    return mix(uv, (seam + f) / tex_size, u_sharpness);
}

vec3 light_at(vec2 world, vec3 n) {
    vec3 sum = u_ambient;
    for (int i = 0; i < u_light_count; ++i) {
        vec4 L = u_lights[i];
        vec2 d = L.xy - world;
        float dist = length(d);
        if (dist > L.z) continue;
        // Squared falloff, clipped at the radius so a light has a real edge
        // and the loop can be cut short.
        float atten = 1.0 - dist / L.z;
        atten *= atten;
        vec3 dir = normalize(vec3(d, u_light_height));
        // Half-lambert: a creature's unlit side goes dim, not black. Fully
        // black silhouettes read as holes at this sprite size.
        float ndl = 0.35 + 0.65 * max(dot(n, dir), 0.0);
        sum += u_light_color * L.w * atten * ndl;
    }
    return sum;
}

void main() {
    vec2 q = v_local;
    float alpha = 1.0;
    vec3 rgb = v_color.rgb;

    if (v_shape < 0.5) {
        /* A sprite, with continuous soft-body deformation.

           The CPU version cuts the sprite into ten bands and slides each one,
           and then has to stop the bands coming apart. Here the displacement
           is a function: every fragment asks where the material it is showing
           came from, which is the same question the bands were approximating
           ten answers to. No band count, no tearing, nothing to hold together. */
        vec2 p = q;
        float m = length(v_lag);
        if (m > 1e-4) {
            /* Invert the deformation rather than approximating it.

               The forward map is the easy direction: material at distance `a`
               back from the leading edge is *drawn* at `0.5 - a - m*a^power`,
               which is a pure stretch -- its derivative is 1 + m*power*a^(p-1),
               never less than one, so nothing is ever squeezed.

               Sampling needs the other direction, and the tempting shortcut is
               to reuse the forward expression with the *drawn* position in
               place of the source one. That is not the inverse: it compresses
               the middle of the body by up to ten to one, which shows up as a
               smear of blur rather than a stretch, and past a certain lag it
               folds the sprite back through itself. Three Newton steps on a
               function this smooth land within a texel, and they cost less than
               the artefact costs. */
            vec2 back = v_lag / m;                     // away from the travel
            vec2 perp = vec2(-back.y, back.x);
            float x = -dot(q, back);                   // +0.5 at the leading edge
            float across = dot(q, perp);

            float a = clamp(0.5 - x, 0.0, 1.0 + m);    // undeformed first guess
            for (int i = 0; i < 3; ++i) {
                float ap = pow(max(a, 0.0), u_soft_power);
                float g = 0.5 - a - m * ap - x;
                float dg = -1.0 - m * u_soft_power * pow(max(a, 1e-3),
                                                         u_soft_power - 1.0);
                a = clamp(a - g / dg, 0.0, 1.0 + m);
            }
            // The cross-axis pinch, for the same reason as on the CPU: a blob
            // squeezed lengthways gets thinner all over. Uniform, so it stays a
            // scale rather than a field.
            across *= 1.0 + clamp(u_soft_pinch * m, 0.0, 0.6);
            p = back * -(0.5 - a) + perp * across;
        }
        // Outside the sprite's own extent there is nothing to show. Without
        // this the grown quad would sample the atlas neighbours.
        if (abs(p.x) > 0.5 || abs(p.y) > 0.5) discard;

        vec2 uv = mix(v_uv0, v_uv1, p + 0.5);
        vec2 suv = sharp_uv(uv, u_atlas_size);
        vec4 texel = texture(u_albedo, suv);
        alpha = texel.a;
        if (alpha < 0.004) discard;
        rgb = texel.rgb * v_color.rgb;

        // params.x marks a quad that must not be lit at all -- an afterimage
        // is an echo rather than a body, and a damage number is not in the
        // room. params.y asks for a flat normal, which is what "normal-mapped
        // sprites" being switched off actually means: still lit, but lit like
        // a piece of paper.
        if (u_lit > 0.5 && v_params.x < 0.5) {
            vec3 n = v_params.y > 0.5
                ? vec3(0.0, 0.0, 1.0)
                : normalize(texture(u_normal, suv).xyz * 2.0 - 1.0);
            rgb *= light_at(v_world, n);
        }
    } else if (v_shape < 1.5) {
        // A soft disc: particles, shadows, decals, dust.
        float d = length(q) * 2.0;
        alpha = 1.0 - smoothstep(1.0 - v_params.x, 1.0, d);
    } else if (v_shape < 2.5) {
        // A ring. params.x is the thickness as a fraction of the radius.
        float d = length(q) * 2.0;
        float t = max(v_params.x, 0.02);
        alpha = 1.0 - smoothstep(0.0, t, abs(d - 1.0 + t * 0.5));
    } else if (v_shape < 3.5) {
        /* A weapon arc, drawn procedurally rather than as a filled polygon.

           The CPU version builds a twenty-two point lens and stamps it through
           a scratch surface. Here it is four lines of maths per fragment and
           the shape is exact at any size. */
        float d = length(q) * 2.0;
        float a = atan(q.y, q.x);
        float half_span = v_params.y * 0.5;
        float da = abs(a);
        if (da > half_span) discard;
        float along = 1.0 - da / max(half_span, 1e-4);   // 1 in the middle
        float thick = v_params.z * sin(along * PI * 0.5);
        alpha = 1.0 - smoothstep(0.0, max(thick, 0.02), abs(d - v_params.x));
        alpha *= pow(along, 0.35);
    } else {
        /* A slash cut: a lens that tapers to a point at both ends and wipes on
           from the tail. params.x is how far it has been drawn. */
        float t = q.x + 0.5;
        if (t > v_params.x) discard;
        float w = pow(max(sin(t * PI), 0.0), 0.6) * 0.5;
        float e = abs(q.y) / max(w, 1e-4);
        if (e > 1.0) discard;
        alpha = 1.0 - smoothstep(0.55, 1.0, e);
        // A white core inside the coloured body, which is the cheapest way to
        // make a flat shape look like it is glowing.
        rgb = mix(rgb, vec3(1.6), 1.0 - smoothstep(0.0, 0.45, e));
    }

    rgb = mix(rgb, vec3(1.4), v_flash);
    frag_color = vec4(rgb * v_color.a * alpha, v_color.a * alpha);
}
"""

# The floor is one quad with the baked arena on it, so the ripple can be a uv
# displacement instead of a per-tile repaint -- which is not just cheaper, it is
# *continuous*: the wave bends the floor between tiles rather than shuffling
# whole tiles about.
FLOOR_VS = """
#version 330
in vec2 in_corner;
uniform vec2 u_camera;
uniform vec2 u_half;
uniform float u_roll;
uniform float u_zoom;
uniform vec2 u_shake;
uniform vec2 u_extent;      // world size of the arena, px
out vec2 v_world;
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
void main() {
    vec2 world = (in_corner + 0.5) * u_extent;
    v_world = world;
    vec2 rel = (world - u_camera) * u_zoom;
    vec2 screen = rot(u_roll) * rel + u_half + u_shake;
    gl_Position = vec4(screen / u_half - 1.0, 0.0, 1.0);
    gl_Position.y = -gl_Position.y;
}
"""

FLOOR_FS = """
#version 330
in vec2 v_world;
uniform sampler2D u_floor;
uniform sampler2D u_floor_normal;
uniform vec2 u_extent;
uniform vec2 u_floor_size;
uniform float u_sharpness;

uniform vec4 u_waves[MAX_WAVES];   // xy centre, z front radius, w amplitude
uniform float u_wave_len[MAX_WAVES];
uniform int u_wave_count;

uniform vec4 u_lights[MAX_LIGHTS];
uniform vec3 u_light_color;
uniform int u_light_count;
uniform vec3 u_ambient;
uniform float u_light_height;
uniform float u_lit;
uniform float u_unlit_scale;
uniform float u_ambient_sway;
uniform float u_time;

out vec4 frag_color;
const float PI = 3.14159265;

vec2 sharp_uv(vec2 uv, vec2 tex_size) {
    vec2 p = uv * tex_size;
    vec2 fw = max(fwidth(p), vec2(1e-5));
    // `p` is in texels, and in GL a texel *centre* sits at a half-integer, so
    // the integers are the seams between texels. Snapping to the nearest seam
    // and then clamping the offset one screen pixel either side of it puts
    // every fragment except the ones straddling a seam exactly on a texel
    // centre -- which is nearest sampling -- and blends only across the seam.
    //
    // Snapping to the texel centre instead, which is the intuitive reading,
    // does the exact opposite: it parks every fragment on a seam, where a
    // linear sampler averages two texels, and the "sharp" filter comes out
    // blurrier than plain bilinear at every magnification.
    vec2 seam = floor(p + 0.5);
    vec2 f = clamp((p - seam) / fw, -0.5, 0.5);
    return mix(uv, (seam + f) / tex_size, u_sharpness);
}

void main() {
    vec2 world = v_world;

    // The ripple. On the CPU this repaints every tile the wave front has
    // reached; here it is a displacement of the texture coordinate, so the
    // floor deforms smoothly instead of tile by tile.
    for (int i = 0; i < u_wave_count; ++i) {
        vec2 d = world - u_waves[i].xy;
        float dist = length(d);
        if (dist < 0.001) continue;
        float lag = dist - u_waves[i].z;
        float len = u_wave_len[i];
        if (lag > 0.0 || lag < -len * 1.5) continue;
        float amp = u_waves[i].w * sin(lag / len * PI) / (1.0 + dist / 120.0);
        world += d / dist * amp;
    }
    // A slow travelling wave over the whole floor, so the room is not a
    // screenshot. Free here; on the CPU it would have undone the baked arena.
    if (u_ambient_sway > 0.0) {
        world += vec2(sin((world.x + world.y * 0.6) / 190.0 + u_time * 0.55),
                      cos((world.x * 0.7 - world.y) / 320.0 - u_time * 0.38))
                 * u_ambient_sway;
    }

    vec2 uv = world / u_extent;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        frag_color = vec4(0.0);
        return;
    }
    vec2 suv = sharp_uv(uv, u_floor_size);
    vec3 rgb = texture(u_floor, suv).rgb;
    if (u_lit > 0.5) {
        vec3 n = normalize(texture(u_floor_normal, suv).xyz * 2.0 - 1.0);
        vec3 sum = u_ambient;
        for (int i = 0; i < u_light_count; ++i) {
            vec4 L = u_lights[i];
            vec2 d = L.xy - world;
            float dist = length(d);
            if (dist > L.z) continue;
            float atten = 1.0 - dist / L.z;
            atten *= atten;
            vec3 dir = normalize(vec3(d, u_light_height));
            // A wider wrap than the sprites get. A floor is nearly flat, so a
            // strict N.L makes the whole pool depend on the light's height and
            // a low light lands as a coin-sized dot; the wrap keeps the pool
            // the size the radius says it is and leaves the normals to pick out
            // the tile edges, which is all they are there for.
            sum += u_light_color * L.w * atten * (0.45 + 0.55 * max(dot(n, dir), 0.0));
        }
        rgb *= sum;
    } else {
        rgb *= u_unlit_scale;      // back to the palette as it was authored
    }
    frag_color = vec4(rgb, 1.0);
}
"""

FULLSCREEN_VS = """
#version 330
in vec2 in_corner;
out vec2 v_uv;
void main() {
    v_uv = in_corner + 0.5;
    gl_Position = vec4(in_corner * 2.0, 0.0, 1.0);
}
"""

# Bright pass and downsample in one. The threshold is what stops bloom reading
# as a dirty screen: without one the whole frame hazes over, which is the
# classic way to overdo it and exactly what the CPU version shipped with until
# it was measured.
BRIGHT_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_scene;
uniform float u_threshold;
uniform float u_knee;
out vec4 frag_color;
void main() {
    vec3 c = texture(u_scene, v_uv).rgb;
    float lum = max(max(c.r, c.g), c.b);
    // A soft knee, so a pixel just over the line glows faintly rather than
    // jumping in. A hard threshold makes bloom flicker on moving highlights.
    float w = clamp((lum - u_threshold) / max(u_knee, 1e-4), 0.0, 1.0);
    frag_color = vec4(c * w * w, 1.0);
}
"""

# A real separable Gaussian, which the CPU version could only approximate with
# a downscale-and-upscale box filter.
BLUR_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_source;
uniform vec2 u_direction;      // texel step, one axis at a time
out vec4 frag_color;
void main() {
    // Nine taps folded into five with linear sampling between texel pairs.
    float w[3] = float[](0.2270270270, 0.3162162162, 0.0702702703);
    float o[3] = float[](0.0, 1.3846153846, 3.2307692308);
    vec3 sum = texture(u_source, v_uv).rgb * w[0];
    for (int i = 1; i < 3; ++i) {
        sum += texture(u_source, v_uv + u_direction * o[i]).rgb * w[i];
        sum += texture(u_source, v_uv - u_direction * o[i]).rgb * w[i];
    }
    frag_color = vec4(sum, 1.0);
}
"""

COMPOSITE_FS = """
#version 330
in vec2 v_uv;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform sampler2D u_ui;

uniform float u_bloom_amount;
uniform float u_rgb_split;      // pixels
uniform vec2  u_texel;
uniform float u_flash;
uniform vec3  u_flash_color;
uniform float u_vignette;
uniform float u_vignette_pulse;
uniform float u_scanlines;
uniform float u_grade;          // warm/cool push on impact
uniform float u_time;

uniform vec4 u_haze[MAX_LIGHTS];    // xy world->screen already, z radius, w strength
uniform int  u_haze_count;
uniform float u_haze_amount;
uniform vec2 u_view;

out vec4 frag_color;

void main() {
    vec2 uv = v_uv;

    /* Heat haze as a genuine screen-space refraction.

       On the CPU this is ninety row-blits per torch, which is why it had to be
       kept to a band above each one. Here every fragment near a heat source
       asks the same question and the answer costs a sine. */
    if (u_haze_amount > 0.001) {
        vec2 px = uv * u_view;
        for (int i = 0; i < u_haze_count; ++i) {
            vec2 d = px - u_haze[i].xy;
            float dist = length(d);
            if (dist > u_haze[i].z) continue;
            float fall = 1.0 - dist / u_haze[i].z;
            // Rising: the phase runs with height, so the distortion travels
            // upwards the way hot air does rather than shivering in place.
            float wob = sin(px.y * 0.06 - u_time * 2.6 + float(i)) *
                        cos(px.x * 0.05 + u_time * 1.7);
            uv.x += wob * u_haze_amount * fall * fall * u_haze[i].w * u_texel.x;
            uv.y -= abs(wob) * u_haze_amount * 0.35 * fall * u_haze[i].w * u_texel.y;
        }
    }

    vec3 scene;
    if (u_rgb_split > 0.05) {
        vec2 d = vec2(u_rgb_split * u_texel.x, 0.0);
        scene.r = texture(u_scene, uv - d).r;
        scene.g = texture(u_scene, uv).g;
        scene.b = texture(u_scene, uv + d).b;
    } else {
        scene = texture(u_scene, uv).rgb;
    }

    scene += texture(u_bloom, uv).rgb * u_bloom_amount;

    // Grade: push warm on impact. A colour shift is a hit confirmation the eye
    // reads without noticing it, and it costs one mix.
    scene = mix(scene, scene * vec3(1.18, 1.02, 0.86), u_grade);
    scene = mix(scene, u_flash_color, u_flash);

    if (u_scanlines > 0.001) {
        float line = 0.5 + 0.5 * cos(v_uv.y * u_view.y * 3.14159);
        scene *= 1.0 - u_scanlines * line;
    }

    vec2 c = (v_uv - 0.5) * 2.0;
    float vig = 1.0 - (u_vignette + u_vignette_pulse) * dot(c, c) * 0.55;
    scene *= clamp(vig, 0.0, 1.0);

    // The panel and the log, drawn by pygame and uploaded. Straight alpha, and
    // deliberately outside everything above: UI must not bloom, shake or bend.
    vec4 ui = texture(u_ui, v_uv);
    frag_color = vec4(mix(scene, ui.rgb, ui.a), 1.0);
}
"""


def _inject(src: str) -> str:
    """Substitute the array sizes the shaders share with Python."""
    return (src.replace("MAX_LIGHTS", str(MAX_LIGHTS))
               .replace("MAX_WAVES", str(MAX_WAVES)))


# ---------------------------------------------------------------------------
# Normal maps from silhouettes
# ---------------------------------------------------------------------------


def normal_from_alpha(alpha: np.ndarray, strength: float = 2.6,
                      blur: int = 3) -> np.ndarray:
    """Turn a sprite's own alpha channel into a tangent-space normal map.

    A pixel-art sprite has no depth information in it, but it does have a
    silhouette, and a silhouette is enough: blur the mask and its gradient
    points outwards at every edge, which is exactly the normal of a shape
    inflated off the page. The middle of a sprite comes out flat and facing the
    viewer, the rim comes out facing away, and a light passing by rakes across
    it. That is the whole trick, and it needs no new art at all.

    `blur` sets how far the bevel reaches in: one pass is a hard chamfer, four
    is a pillow. Three is about right for a 16px sprite drawn at 32.
    """
    a = alpha.astype(np.float32) / 255.0
    for _ in range(max(0, blur)):
        # A 3x3 box, done as two 1D passes with edge clamping.
        a = (np.pad(a, ((0, 0), (1, 0)), mode="edge")[:, :-1] + a
             + np.pad(a, ((0, 0), (0, 1)), mode="edge")[:, 1:]) / 3.0
        a = (np.pad(a, ((1, 0), (0, 0)), mode="edge")[:-1] + a
             + np.pad(a, ((0, 1), (0, 0)), mode="edge")[1:]) / 3.0

    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * 0.5
    gy[1:-1, :] = (a[2:, :] - a[:-2, :]) * 0.5

    # Pointing *out* of the shape, so the rim faces away from the middle.
    nx, ny = -gx * strength, -gy * strength
    nz = np.ones_like(a)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.empty(a.shape + (3,), dtype=np.float32)
    out[..., 0] = nx / length
    out[..., 1] = ny / length
    out[..., 2] = nz / length
    return ((out * 0.5 + 0.5) * 255.0).astype(np.uint8)


def normal_from_luminance(rgb: np.ndarray, strength: float = 1.4,
                          blur: int = 1) -> np.ndarray:
    """The same idea applied to a painted surface: brightness stands in for
    height. Good enough for a tiled floor, where what you want is for the light
    to catch the edges of the tiles."""
    lum = rgb.astype(np.float32).mean(axis=2)
    lum = (lum - lum.min()) / max(1.0, float(lum.max() - lum.min())) * 255.0
    return normal_from_alpha(lum.astype(np.uint8), strength, blur)


# ---------------------------------------------------------------------------
# The atlas
# ---------------------------------------------------------------------------


class Atlas:
    """Every sprite, plus a white texel, in one texture -- and its normal map.

    One texture is what makes one draw call possible, and one draw call is what
    makes the particle count stop mattering. The cells are padded so that
    filtering at a cell edge cannot reach into the neighbour.

    Sprites go in *untinted*: the tint is a per-instance colour that multiplies
    in the shader, so one white goblin serves every colour of goblin without a
    second upload. That is also why the software renderer's per-tint surface
    cache has no equivalent here -- there is nothing to cache.
    """

    #: The albedo page is stored at the *source* resolution and the card does
    #: the upscale. Pre-scaling it in pygame was the obvious thing to do and it
    #: silently disables the sharp filter: a filter can only hold an edge crisp
    #: while it is magnifying, so at 1:1 sharp bilinear degenerates into plain
    #: bilinear and at a minification it is worse than useless.
    SCALE = 1
    #: The normal page is a separate, larger texture with the *same* layout.
    #: Normals want a smooth bevel and there is nothing to bevel across two
    #: texels, but the uv are fractions, so the two pages need not be the same
    #: size -- one lookup serves both.
    NORMAL_SCALE = 4
    PAD = 4
    COLUMNS = 5

    def __init__(self, ctx, sheet, keys: Sequence[str], sprite_table: dict,
                 extra: Optional[dict] = None, normal_strength: float = 2.6,
                 normal_blur: int = 1):
        import pygame
        self.ctx = ctx
        self.cell = 16 * self.SCALE
        step = self.cell + self.PAD * 2
        extra = extra or {}
        # Sprites, then one white cell, then whatever else was handed in (the
        # digits for damage numbers). Everything in one page, because one page
        # is what makes one draw call possible.
        total = len(keys) + 1 + len(extra)
        rows = (total + self.COLUMNS - 1) // self.COLUMNS
        w = step * self.COLUMNS
        h = step * rows
        page = pygame.Surface((w, h), pygame.SRCALPHA)
        page.fill((0, 0, 0, 0))

        def slot(i):
            return ((i % self.COLUMNS) * step + self.PAD,
                    (i // self.COLUMNS) * step + self.PAD)

        def cell_uv(x, y):
            return ((x / w, y / h), ((x + self.cell) / w, (y + self.cell) / h))

        self.uv: dict = {}
        for i, key in enumerate(keys):
            col, row = sprite_table[key]
            # Untinted: the tint is a per-instance colour that multiplies in the
            # shader, so one white goblin serves every colour of goblin and the
            # software renderer's per-tint surface cache has no equivalent here.
            src = sheet.sprite(col, row, None)
            x, y = slot(i)
            page.blit(pygame.transform.scale(src, (self.cell, self.cell)), (x, y))
            self.uv[key] = cell_uv(x, y)

        # A solid white cell. Particles, shadows, rings, arcs and cuts all
        # sample one texel of it, so they ride in the same buffer as the
        # sprites and cost no extra draw call.
        x, y = slot(len(keys))
        page.fill((255, 255, 255, 255), pygame.Rect(x, y, self.cell, self.cell))
        cx, cy = (x + self.cell * 0.5) / w, (y + self.cell * 0.5) / h
        self.white = ((cx, cy), (cx, cy))

        for i, (name, surf) in enumerate(extra.items()):
            x, y = slot(len(keys) + 1 + i)
            page.blit(surf, surf.get_rect(center=(x + self.cell // 2,
                                                  y + self.cell // 2)))
            self.uv[name] = cell_uv(x, y)

        self.size = (w, h)
        # Uploaded *unflipped*, unlike a full-screen background. The world is
        # drawn in pygame's y-down coordinates and the vertex shader flips NDC
        # to match, so a texture coordinate of v=0 has to be the top of the
        # page. Uploading flipped mirrors every cell, and because the page is a
        # grid the result is not an obviously upside-down sprite -- it is each
        # creature sampling a slice of the padding above its neighbour, which
        # looks like the art failing to load.
        data = pygame.image.tobytes(page, "RGBA", False)
        self.albedo = ctx.texture((w, h), 4, data)
        self.albedo.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.albedo.repeat_x = self.albedo.repeat_y = False

        rgba = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
        k = self.NORMAL_SCALE
        big = np.repeat(np.repeat(rgba[..., 3], k, axis=0), k, axis=1)
        normals = normal_from_alpha(big, strength=normal_strength,
                                    blur=normal_blur * k)
        self.normal = ctx.texture((w * k, h * k), 3, normals.tobytes())
        self.normal.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.normal.repeat_x = self.normal.repeat_y = False

    def release(self):
        self.albedo.release()
        self.normal.release()


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------

#: One instance, in floats. Kept as a module constant because the numpy view,
#: the buffer format string and the shader inputs all have to agree and there
#: is no way to make the compiler check that for you.
INSTANCE_FLOATS = 21
INSTANCE_FORMAT = "2f 2f 1f 2f 2f 4f 1f 2f 1f 4f"
INSTANCE_ATTRS = ["in_center", "in_size", "in_rot", "in_uv0", "in_uv1",
                  "in_color", "in_flash", "in_lag", "in_shape", "in_params"]

SHAPE_SPRITE, SHAPE_DISC, SHAPE_RING, SHAPE_ARC, SHAPE_CUT = 0.0, 1.0, 2.0, 3.0, 4.0


class QuadBatch:
    """Everything in the world, as instanced quads in painter's order.

    The order is the point as much as the batching: one buffer written
    front-to-back means a decal, a shadow, a creature and a spark can be
    interleaved exactly as they should be drawn without any of them needing
    their own pass. Growing is by doubling, so a fight that throws four thousand
    sparks costs one reallocation and never a stall per particle.
    """

    def __init__(self, ctx, program, capacity: int = 4096):
        self.ctx = ctx
        self.program = program
        self.capacity = capacity
        self.data = np.zeros((capacity, INSTANCE_FLOATS), dtype="f4")
        self.count = 0

        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]],
                           dtype="f4")
        self.corner_vbo = ctx.buffer(corners.tobytes())
        self.instance_vbo = ctx.buffer(reserve=self.data.nbytes, dynamic=True)
        self._build_vao()

    def _build_vao(self):
        self.vao = self.ctx.vertex_array(self.program, [
            (self.corner_vbo, "2f", "in_corner"),
            (self.instance_vbo, INSTANCE_FORMAT + "/i", *INSTANCE_ATTRS),
        ])

    def clear(self):
        self.count = 0

    def _grow(self):
        self.capacity *= 2
        bigger = np.zeros((self.capacity, INSTANCE_FLOATS), dtype="f4")
        bigger[:self.count] = self.data[:self.count]
        self.data = bigger
        self.vao.release()
        self.instance_vbo.release()
        self.instance_vbo = self.ctx.buffer(reserve=self.data.nbytes, dynamic=True)
        self._build_vao()

    def add(self, cx, cy, w, h, *, rot=0.0, uv=None, color=(1.0, 1.0, 1.0),
            alpha=1.0, flash=0.0, lag=(0.0, 0.0), shape=SHAPE_SPRITE,
            params=(0.0, 0.0, 0.0, 0.0)):
        if self.count >= self.capacity:
            self._grow()
        row = self.data[self.count]
        (u0, v0), (u1, v1) = uv if uv is not None else ((0.0, 0.0), (1.0, 1.0))
        row[0], row[1] = cx, cy
        row[2], row[3] = w, h
        row[4] = rot
        row[5], row[6] = u0, v0
        row[7], row[8] = u1, v1
        row[9], row[10], row[11], row[12] = color[0], color[1], color[2], alpha
        row[13] = flash
        row[14], row[15] = lag
        row[16] = shape
        row[17], row[18], row[19], row[20] = params
        self.count += 1

    def render(self):
        if not self.count:
            return
        self.instance_vbo.write(self.data[:self.count].tobytes())
        self.vao.render(mode=moderngl.TRIANGLE_STRIP, instances=self.count)

    def release(self):
        self.vao.release()
        self.instance_vbo.release()
        self.corner_vbo.release()


# ---------------------------------------------------------------------------
# The post chain
# ---------------------------------------------------------------------------


class PostChain:
    """Scene buffer, a bloom pyramid, and one composite pass for everything else.

    The software bench spends most of a frame on the RGB split alone and nearly
    four milliseconds on lighting. All of that -- split, bloom, vignette, flash,
    grade, scanlines, heat haze -- is one fragment shader here, and the bloom is
    a real separable Gaussian rather than a downscale pretending to be a blur.

    The scene is a half-float target, which matters more than it sounds: a hit
    flash and a torch can push a pixel well past white, and in an 8-bit buffer
    that information is gone before the bright pass can find it. Bloom off an
    LDR buffer can only ever glow things that were already at the top of the
    range.
    """

    def __init__(self, ctx, size: Tuple[int, int], bloom_divisor: int = 2):
        self.ctx = ctx
        self.size = size
        w, h = size
        bw, bh = max(1, w // bloom_divisor), max(1, h // bloom_divisor)

        self.scene_tex = ctx.texture(size, 4, dtype="f2")
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene = ctx.framebuffer(color_attachments=[self.scene_tex])

        self.bloom_tex = [ctx.texture((bw, bh), 4, dtype="f2") for _ in range(2)]
        for t in self.bloom_tex:
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
        self.bloom = [ctx.framebuffer(color_attachments=[t]) for t in self.bloom_tex]
        self.bloom_size = (bw, bh)

        self.bright_prog = ctx.program(vertex_shader=FULLSCREEN_VS,
                                       fragment_shader=_inject(BRIGHT_FS))
        self.blur_prog = ctx.program(vertex_shader=FULLSCREEN_VS,
                                     fragment_shader=_inject(BLUR_FS))
        self.comp_prog = ctx.program(vertex_shader=FULLSCREEN_VS,
                                     fragment_shader=_inject(COMPOSITE_FS))

        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]],
                           dtype="f4")
        self.quad = ctx.buffer(corners.tobytes())
        self.vaos = {name: ctx.vertex_array(prog, [(self.quad, "2f", "in_corner")])
                     for name, prog in (("bright", self.bright_prog),
                                        ("blur", self.blur_prog),
                                        ("comp", self.comp_prog))}

    def build_bloom(self, threshold: float, knee: float, passes: int = 2):
        """Bright pass, then ping-pong horizontal and vertical blurs."""
        bw, bh = self.bloom_size
        self.bloom[0].use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.scene_tex.use(0)
        self.bright_prog["u_scene"] = 0
        self.bright_prog["u_threshold"] = threshold
        self.bright_prog["u_knee"] = knee
        self.vaos["bright"].render(mode=moderngl.TRIANGLE_STRIP)

        for _ in range(passes):
            for direction in ((1.0 / bw, 0.0), (0.0, 1.0 / bh)):
                self.bloom[1].use()
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                self.bloom_tex[0].use(0)
                self.blur_prog["u_source"] = 0
                self.blur_prog["u_direction"] = direction
                self.vaos["blur"].render(mode=moderngl.TRIANGLE_STRIP)
                self.bloom[0], self.bloom[1] = self.bloom[1], self.bloom[0]
                self.bloom_tex[0], self.bloom_tex[1] = self.bloom_tex[1], self.bloom_tex[0]

    def composite(self, target, ui_texture, uniforms: dict):
        target.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.scene_tex.use(0)
        self.bloom_tex[0].use(1)
        ui_texture.use(2)
        p = self.comp_prog
        p["u_scene"] = 0
        p["u_bloom"] = 1
        p["u_ui"] = 2
        for key, value in uniforms.items():
            if key in p:
                p[key] = value
        self.vaos["comp"].render(mode=moderngl.TRIANGLE_STRIP)

    def release(self):
        for vao in self.vaos.values():
            vao.release()
        self.quad.release()
        for fbo in self.bloom:
            fbo.release()
        for tex in self.bloom_tex:
            tex.release()
        self.scene.release()
        self.scene_tex.release()
        for prog in (self.bright_prog, self.blur_prog, self.comp_prog):
            prog.release()


# ---------------------------------------------------------------------------
# Uniform helpers
# ---------------------------------------------------------------------------


def array_uniform(program, name: str):
    """An array uniform, under whichever name the driver decided to report.

    GLSL leaves this to the implementation: an array may show up as `u_lights`
    or as `u_lights[0]`, and different drivers pick differently. Checking only
    the bare name is the trap -- the lookup silently misses, the uniform is
    never written, and the shader runs perfectly happily with a zero-length
    light array. Nothing errors; the room is simply dark.
    """
    for candidate in (name, f"{name}[0]"):
        if candidate in program:
            return program[candidate]
    return None


def write_vec4_array(program, name: str, rows: Sequence[Sequence[float]],
                     limit: int) -> int:
    """Fill a `vec4[]` uniform, padding the tail. Returns how many are live."""
    uniform = array_uniform(program, name)
    if uniform is None:
        return 0
    buf = np.zeros((limit, 4), dtype="f4")
    n = min(len(rows), limit)
    for i in range(n):
        buf[i] = rows[i]
    uniform.write(buf.tobytes())
    return n


def write_float_array(program, name: str, values: Sequence[float], limit: int):
    uniform = array_uniform(program, name)
    if uniform is None:
        return
    buf = np.zeros(limit, dtype="f4")
    for i, v in enumerate(values[:limit]):
        buf[i] = v
    uniform.write(buf.tobytes())


def surface_texture(ctx, surface, components: int = 4, flip: bool = True):
    """Upload a pygame surface.

    `flip` picks the convention. A full-screen overlay sampled with a 0..1
    coordinate that runs *up* the screen wants the flip; anything sampled from
    world coordinates, which run down, does not. Getting it wrong is silent --
    the texture is there, it is just mirrored -- so it is a parameter rather
    than a constant.
    """
    import pygame
    fmt = "RGBA" if components == 4 else "RGB"
    data = pygame.image.tobytes(surface, fmt, flip)
    tex = ctx.texture(surface.get_size(), components, data)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = False
    return tex
