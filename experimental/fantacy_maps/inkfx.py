"""Wet ink on parchment: the effect, with no application around it.

A page of parchment lit by one moving light, with ink that is glossy and
iridescent while it is wet and sinks into the weave as it dries, and a pen that
can draw a sprite on to it stroke by stroke. Everything here is about the
effect itself -- no window, no menus, no artwork, no key bindings -- so it can
be driven by a workbench, by a test with no display at all, or by a game.

The three things a caller deals with:

    InkCanvas   the page as editable artwork: a coverage mask at whatever
                resolution the strokes need, and a wetness field beside it that
                fades on its own. Pure pygame surfaces, no GL, so the whole ink
                model can be tested without rendering anything.
    InkSettings the material: how fast ink dries, how deep the weave is, how
                strongly the wet coat throws colour. One flat record, so a
                caller can put a slider on any of it.
    ParchmentInkRenderer
                the GL side: builds the parchment, holds the textures, and
                draws the page. It can be created without a window at all --
                `headless=True` -- which is how it is tested.

Coordinates: every rectangle crossing `InkCanvas`'s boundary is in page pixels,
whatever the coverage mask is actually stored at. The conversion happens once,
at the last moment, inside the canvas.
"""

from __future__ import annotations

import math
import os
import random
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path

try:
    # Recovering the strokes inside a finished drawing. Optional: it needs
    # numpy, and everything except drawing a sprite on works without it.
    import numpy
    import pentrace
except ImportError:
    numpy = pentrace = None

try:
    import pygame
    import moderngl
except ImportError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing {missing!r}. Install the dependencies with:\n"
        "    python -m pip install pygame moderngl"
    ) from exc


PAPER_TEXTURE_WIDTH = 512

# The ink mask is drawn at this multiple of the window resolution. Zooming in
# magnifies the mask, so a stroke's edge is only ever as smooth as the grid it
# was drawn on; supersampling buys back that headroom for memory and stamping
# time, and costs nothing per frame. At rest the extra samples are not wasted
# either: the bilinear fetch lands exactly between texels and box-filters them.
INK_SUPERSAMPLE = 2

# How fast the drawing pen travels, in design-space units per second, and how
# long it pauses between one stroke and the next. The pause is short because a
# traced figure is dozens of small strokes, and the pen is charged for crossing
# to the next one on top of it.
PEN_SPEED = 420.0
PEN_TOUCHDOWN_SECONDS = 0.05

# How far past its traced width the nib reaches when uncovering a sprite, in
# page pixels. Reaching too far costs nothing at all -- there is no artwork
# outside the figure for it to uncover -- while reaching too short leaves a
# pixel waiting on its neighbours, so the setting is deliberately generous.
NIB_SLACK = 1.5

# The largest copy of a sprite the pen path is worked out on. The path decides
# only the order pixels appear in, so it can be coarser than the drawing itself
# without any of the artwork being lost; and it has to be, because thinning
# takes a pass per pixel of the widest stroke and the cost is in the square of
# this.
TRACE_MAX_SIZE = 320

# Mesa's software rasteriser, which every driver here falls back to when it
# cannot reach a real one. Its name in `GL_RENDERER` is the only reliable way
# to tell that a shader is about to be run on the CPU.
SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast")

# WSL exposes the host's graphics card through this device, and Mesa reaches it
# with the `d3d12` driver. It is not the default, so a WSL session that has a
# perfectly good card silently rasterises everything on the processor instead.
WSL_GPU_DEVICE = Path("/dev/dxg")
WSL_GPU_DRIVER = "d3d12"
WSL_GPU_LIBRARIES = "/usr/lib/wsl/lib"
# Checked as well as the device, because forcing a driver that is not installed
# leaves Mesa with nowhere to fall back to and the window never opens at all.
WSL_GPU_MODULE = Path(
    "/usr/lib/x86_64-linux-gnu/dri/d3d12_dri.so"
)

# Set once the driver has been chosen, so the restart below can only happen
# once however the program was launched.
DRIVER_SETTLED = "INKGL_DRIVER_SETTLED"


def wsl_graphics_card_available() -> bool:
    """Whether this is a WSL session with a card Mesa could be pointed at.

    Both the device and the driver have to be there: forcing a driver that is
    not installed leaves Mesa with nowhere to fall back to, and the context is
    never created at all.
    """
    if "INKGL_DRIVER" in os.environ or os.environ.get("MESA_LOADER_DRIVER_OVERRIDE"):
        return False
    return WSL_GPU_DEVICE.exists() and WSL_GPU_MODULE.exists()


def use_wsl_graphics_card() -> None:
    """Point Mesa at the card. Only has any effect before the first context."""
    os.environ[DRIVER_SETTLED] = "1"
    os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = WSL_GPU_DRIVER
    os.environ["GALLIUM_DRIVER"] = WSL_GPU_DRIVER
    libraries = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        f"{WSL_GPU_LIBRARIES}:{libraries}" if libraries else WSL_GPU_LIBRARIES
    )


def restart_on_the_graphics_card(renderer_name: str) -> None:
    """If this is WSL rasterising on the CPU, start again on the card.

    A page is a single full-screen quad, so the entire frame is fragment
    shader; run on the processor it costs about thirty milliseconds at 4K and
    keeps several cores busy, and on the card it costs under two. That is worth
    a relaunch, and a relaunch is the only way to take it: Mesa chooses its
    driver when the first context is created and will not revisit the decision.

    Nothing happens unless every sign points the same way -- a software
    renderer, the WSL graphics device present, and no driver already asked for
    by hand. `INKGL_DRIVER` set to anything, including empty, opts out.

    This re-executes the running program, so only ever call it from something
    that was run as a program. Importing a module must never relaunch anything,
    or a test harness that imported it would restart itself.
    """
    if os.environ.get(DRIVER_SETTLED) or not wsl_graphics_card_available():
        return
    if not any(name in renderer_name.lower() for name in SOFTWARE_RENDERERS):
        return

    print(
        f"Rendering on the processor ({renderer_name}); "
        f"restarting on the graphics card.",
        file=sys.stderr,
    )
    use_wsl_graphics_card()
    pygame.quit()
    os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ))


def open_window(size: tuple[int, int], fullscreen: bool = False) -> tuple[int, int]:
    """Open a window with a core OpenGL context; return the size granted.

    A window manager is free to refuse what it is asked for. Fullscreen usually
    lands on the desktop resolution whatever mode was requested, and growing an
    existing window past the desktop may simply not happen. Everything from the
    page outwards is sized from this, so the size that came back is the one to
    believe rather than the one that was asked for.
    """
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK,
        pygame.GL_CONTEXT_PROFILE_CORE,
    )
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 0)

    flags = pygame.OPENGL | pygame.DOUBLEBUF
    if fullscreen:
        flags |= pygame.FULLSCREEN
    pygame.display.set_mode(size, flags)
    return pygame.display.get_window_size()


# The relief of the sheet itself, as a tangent-space normal map. Without it the
# parchment falls back to the normal generated alongside its albedo.
CANVAS_NORMAL_PATH = Path(__file__).resolve().parent / "canvas_normal.png"
# The weave photograph repeats across the page rather than being stretched to
# fit it, which keeps the threads square whatever shape the window is; it very
# nearly tiles already. Its threads lie every sixteen texels, measured by
# autocorrelating the map, and that is what turns the "weave size" the user asks
# for, in page pixels between threads, into a number of repeats.
CANVAS_WEAVE_PERIOD = 16.0

# The iridescent film reflected in wet ink. Whichever of these exists is used.
OIL_SLICK_PATHS = (
    Path(__file__).resolve().parent / "oil_slick.png",
    Path(__file__).resolve().parent / "oil_slick.jpg",
)
# Magnified by the slick zoom, only a fraction of the film is on screen at once,
# so it needs the resolution to survive being spread over the sheet.
OIL_SLICK_SIZE = 1024

# Ink drying. The shine and the raised meniscus fade exponentially, which is the
# usual model for a solvent leaving a film: `exp(-age / tau)`. Half the gloss is
# gone within a second and a stroke reads as dry after two or three.
INK_DRY_TAU = 1.0
# How long a stroke's pixels stay in the wetness field. By then the fade has
# bottomed out below PAGE_WET_FLOOR, so emptying the field changes nothing.
INK_DRY_SECONDS = 4.0
# Fully dry ink keeps a trace of sheen; parchment is not perfectly matte either.
PAGE_WET_FLOOR = 0.06
# The stroke wetness field fades on this interval rather than every frame: it is
# a slow, low-contrast change, and each step costs a fill plus an upload over the
# whole wet region. Pygame's multiply blend is `(dest * src + 255) >> 8`, whose
# +1 of rounding stalls the fade once a byte reaches about `1 / (1 - factor)`;
# stepping this slowly puts that floor at 15/255, below the sheen of dry ink.
WET_UPDATE_INTERVAL = 1.0 / 15.0


VERTEX_SHADER = """
#version 330

in vec2 in_position;
in vec2 in_uv;

out vec2 v_uv;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_uv;
}
"""


FRAGMENT_SHADER = """
#version 330

uniform sampler2D u_paper_albedo;
uniform sampler2D u_paper_normal;
uniform sampler2D u_ink_grain;
uniform vec2 u_paper_normal_tiles;
uniform float u_paper_relief;
uniform float u_weave_soak;
uniform float u_wet_floor;
uniform sampler2D u_paper_roughness;
uniform sampler2D u_ink_mask;
uniform sampler2D u_ink_wet;
uniform sampler2D u_oil_slick;
uniform float u_slick_gain;

// How much of the oil film shows over the ink, how fast it swirls while the ink
// is still liquid, and how much of the sheet one copy of the photograph covers.
// All three are taste rather than physics. Everything feeding the lookup is kept
// in page units and scaled once at the end, so size cannot disturb speed.
uniform float u_slick_opacity;
uniform float u_slick_swirl;
uniform float u_slick_zoom;

// How sharply drying slows the swirl, and the factor that keeps the total travel
// the same however sharply it does. See the derivation where they are used.
uniform float u_swirl_dryness;
uniform float u_swirl_span;

uniform vec2 u_resolution;
uniform vec2 u_light_uv;
uniform float u_light_height;
uniform float u_page_wetness;
uniform float u_wet_gain;
uniform float u_exposure;
uniform int u_debug_mode;
uniform float u_zoom;
uniform vec2 u_center_uv;

in vec2 v_uv;
out vec4 frag_color;

const float PI = 3.14159265359;

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

vec3 to_linear(vec3 srgb) {
    return pow(max(srgb, vec3(0.0)), vec3(2.2));
}

vec3 to_srgb(vec3 linear_color) {
    return pow(max(linear_color, vec3(0.0)), vec3(1.0 / 2.2));
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(1.0 - saturate(cos_theta), 5.0);
}

// Tile a photograph that was never made to tile: folding the coordinate back on
// itself is seamless everywhere, and a mirrored swirl still reads as a swirl.
vec2 mirror_uv(vec2 uv) {
    return 1.0 - abs(fract(uv * 0.5) * 2.0 - 1.0);
}

void main() {
    float aspect = u_resolution.x / u_resolution.y;

    // Screen space stays put while the page zooms and pans underneath it, so
    // the lamp keeps behaving like a fixed desk light above the viewer.
    vec2 page_uv = (v_uv - 0.5) / u_zoom + u_center_uv;

    vec2 marker_delta = (v_uv - u_light_uv) * vec2(aspect, 1.0);
    float marker_distance = length(marker_delta);
    float marker = 1.0 - smoothstep(0.010, 0.012, abs(marker_distance - 0.013));

    vec3 warm_light = to_linear(vec3(1.00, 0.84, 0.61));
    vec3 ambient_light = to_linear(vec3(0.34, 0.29, 0.22));

    // Anything off the sheet is the desk it is lying on.
    if (any(lessThan(page_uv, vec2(0.0))) || any(greaterThan(page_uv, vec2(1.0)))) {
        vec3 desk = to_linear(vec3(0.078, 0.062, 0.049));
        desk *= 1.0 / (1.0 + 4.5 * marker_distance * marker_distance);
        if (u_debug_mode == 0) {
            desk += warm_light * marker * 0.22;
        }
        frag_color = vec4(to_srgb(desk * u_exposure), 1.0);
        return;
    }

    // A page pixel, not a mask texel: the relief below is tuned to the width of
    // the ramp the brush leaves, so supersampling the mask must not narrow it.
    vec2 texel = 1.0 / u_resolution;

    vec3 paper_albedo = to_linear(texture(u_paper_albedo, page_uv).rgb);
    float paper_roughness = texture(u_paper_roughness, page_uv).r;

    // The bare sheet's relief. It tiles across the page rather than stretching
    // to fit, so the weave stays square whatever shape the window is, and its
    // slopes are scaled to the depth the rest of the lighting was tuned for.
    vec3 sheet_normal = texture(
        u_paper_normal, page_uv * u_paper_normal_tiles
    ).rgb * 2.0 - 1.0;
    sheet_normal = normalize(
        vec3(sheet_normal.xy * u_paper_relief, max(sheet_normal.z, 1e-3))
    );

    // The sheet's fine grain, which is what wet ink lies on. Which of the two
    // shows through a stroke depends on how wet it is; see `under_ink` below.
    vec3 grain_normal = normalize(texture(u_ink_grain, page_uv).rgb * 2.0 - 1.0);

    float ink = texture(u_ink_mask, page_uv).a;
    float ink_left = texture(u_ink_mask, page_uv - vec2(texel.x, 0.0)).a;
    float ink_right = texture(u_ink_mask, page_uv + vec2(texel.x, 0.0)).a;
    float ink_down = texture(u_ink_mask, page_uv - vec2(0.0, texel.y)).a;
    float ink_up = texture(u_ink_mask, page_uv + vec2(0.0, texel.y)).a;

    // Wetness has two sources: the page-wide dry-down of the printed artwork,
    // and a per-pixel field that gives every fresh stroke its own drying clock.
    // The field only fades in steps, so u_wet_gain carries it the rest of the way
    // to the present moment and the steps stop being visible as steps.
    float wetness = max(
        u_page_wetness, texture(u_ink_wet, page_uv).r * u_wet_gain
    );

    // The alpha gradient makes wet strokes look microscopically raised. Drying
    // ink sinks into the fibres, so the relief nearly vanishes with the shine.
    vec2 ink_gradient = vec2(ink_right - ink_left, ink_up - ink_down);
    float edge = saturate(length(ink_gradient) * 3.2);

    // Wet ink is a film lying over the sheet: it pools into the hollows of the
    // weave and skins over them, so the coat picks up only the fibre underneath.
    // As it dries it sinks in and takes up the shape of the fabric, and by then
    // the weave is showing through the stroke as strongly as it does beside it.
    //
    // The changeover is `span^soak`, where span is the dry-down remapped to run
    // the whole way from 1 to 0 -- wetness itself stops at the floor, so raising
    // that to a power would leave the weave short of full no matter how dry the
    // ink got. A soak above 1 brings the fabric through early, below 1 holds it
    // off until the stroke has nearly finished drying, and exactly 1 is a
    // straight line.
    float wet_span = saturate(
        (wetness - u_wet_floor) / max(1.0 - u_wet_floor, 1e-4)
    );
    vec3 under_ink = normalize(
        mix(sheet_normal, grain_normal, pow(wet_span, u_weave_soak))
    );

    vec3 ink_normal = normalize(vec3(
        under_ink.xy * 0.22 - ink_gradient * (0.30 + 2.30 * wetness),
        max(0.30, under_ink.z)
    ));
    vec3 normal = normalize(mix(sheet_normal, ink_normal, ink));

    vec3 dry_ink = to_linear(vec3(0.085, 0.047, 0.026));
    vec3 wet_ink = to_linear(vec3(0.030, 0.016, 0.010));
    vec3 ink_albedo = mix(dry_ink, wet_ink, wetness);
    vec3 albedo = mix(paper_albedo, ink_albedo, ink);

    float ink_roughness = mix(0.68, 0.10, wetness);
    float roughness = mix(paper_roughness, ink_roughness, ink);

    vec3 surface_position = vec3(v_uv.x * aspect, v_uv.y, 0.0);
    vec3 light_position = vec3(u_light_uv.x * aspect, u_light_uv.y, u_light_height);
    vec3 light_vector = light_position - surface_position;
    float light_distance = length(light_vector);
    vec3 light_direction = light_vector / max(light_distance, 0.0001);
    vec3 view_direction = vec3(0.0, 0.0, 1.0);
    vec3 half_vector = normalize(light_direction + view_direction);

    float n_dot_l = saturate(dot(normal, light_direction));
    float n_dot_h = saturate(dot(normal, half_vector));
    float n_dot_v = saturate(dot(normal, view_direction));

    // A close desk-lamp falloff, softened so the whole sheet remains visible.
    float attenuation = 1.0 / (0.45 + 3.5 * light_distance * light_distance);
    attenuation = min(attenuation, 1.6);

    // High roughness gives parchment a wide, weak reflection. Wet ink gets a
    // much narrower coat highlight. This is intentionally lightweight rather
    // than a full PBR BRDF, but the controls behave like material properties.
    float shininess = mix(8.0, 420.0, pow(1.0 - roughness, 2.1));
    float normalized_specular = (shininess + 2.0) / (2.0 * PI);
    float specular_lobe = pow(n_dot_h, shininess) * normalized_specular;

    vec3 paper_f0 = vec3(0.025);
    vec3 ink_f0 = mix(vec3(0.030), vec3(0.115), wetness);
    vec3 f0 = mix(paper_f0, ink_f0, ink);
    vec3 fresnel = fresnel_schlick(n_dot_v, f0);

    float paper_specular_strength = mix(0.055, 0.025, paper_roughness);
    float ink_specular_strength = mix(0.10, 1.05, wetness);
    float specular_strength = mix(paper_specular_strength, ink_specular_strength, ink);

    // Wet ink is a thin film: it mirrors the room, and interference paints that
    // reflection with the shifting colours of an oil slick. The lookup is warped
    // by the surface normal so the film reads as curved, and it keeps swirling
    // while the ink is liquid, settling into place as the shine goes.
    float coat = ink * wetness;
    vec3 reflection = reflect(-view_direction, normal);
    // The swirl is driven by drying rather than by the clock, so each patch of
    // ink churns at its own speed: fast while it is liquid, creeping as it sets,
    // frozen once dry. A pixel's phase is the integral of its own rate, and for
    // a rate following a power of the wetness that integral closes in the
    // wetness itself, with K = u_swirl_dryness setting how sharply drying slows
    // the film down:
    //     rate(age)  = R / tau * exp(-K * age / tau)
    //     phase(age) = R / K * (1 - exp(-K * age / tau)) = R / K * (1 - wetness^K)
    // which needs no history and cannot jump, however the wetness got there.
    // u_swirl_span carries the 1/K and normalises the total travel to R, so that
    // K only redistributes the churn over the dry-down instead of adding more:
    // near zero the film creeps evenly the whole way, high up it does nearly all
    // of its moving in the first instant of being wet.
    //
    // Leaving tau out of the phase fixes the total travel and lets the drying
    // slider stretch it over more time. Keeping it in would instead fix the
    // speed, but then ink laid down one fade step apart would sit a large phase
    // apart, and a slowly drawn stroke would show stripes where the steps fell
    // -- a whole second of churn between one band of ink and the next.
    float swirl = u_slick_swirl * (1.0 - pow(wetness, u_swirl_dryness))
        * u_swirl_span;

    vec2 flow = vec2(page_uv.x * aspect, page_uv.y) + vec2(0.13, -0.21) * swirl;
    // Wet ink is also warped more strongly, not just faster.
    flow += vec2(
        sin(flow.y * 7.7 + swirl * 6.8),
        cos(flow.x * 6.6 - swirl * 5.2)
    ) * 0.055 * wetness;

    // The warp from the normal has to stay gentle: the ink normal carries the
    // paper's own grain, and a strong warp turns that into rainbow speckle
    // instead of the broad bands a film actually shows.
    vec2 slick_uv = (flow + reflection.xy * 0.21) / u_slick_zoom;

    vec3 slick = to_linear(texture(u_oil_slick, mirror_uv(slick_uv)).rgb);
    float slick_luminance = max(dot(slick, vec3(0.2126, 0.7152, 0.0722)), 1e-4);
    // Chroma alone colours the lamp's hotspot without changing how bright it is.
    vec3 highlight_tint = mix(
        vec3(1.0), slick / slick_luminance, coat * 0.9 * u_slick_opacity
    );

    vec3 diffuse = albedo * (ambient_light + warm_light * n_dot_l * attenuation * 0.68);
    vec3 specular = warm_light * highlight_tint
        * fresnel * specular_lobe * specular_strength * attenuation;

    // The film mirrors more of the room than just the lamp, and most strongly
    // where it curves away from the viewer. u_slick_gain normalises the photo to
    // a mean luminance of one, so this adds the pattern, not a brightness bias.
    // Squaring the wetness keeps it a liquid effect: iridescence is gone well
    // before the last of the shine is, and dry ink never shimmers.
    float grazing = pow(1.0 - n_dot_v, 2.0);
    specular += slick * u_slick_gain * coat * wetness
        * (0.016 + 0.055 * grazing) * attenuation * 0.85 * u_slick_opacity;

    // A small extra meniscus glint along the stroke boundary sells fresh ink.
    float edge_glint = edge * ink * wetness * pow(n_dot_h, 34.0) * attenuation;
    specular += warm_light * edge_glint * 0.16;

    // Darken the sheet perimeter without baking it into the generated texture.
    vec2 border_distance = min(page_uv, 1.0 - page_uv);
    float border = smoothstep(0.0, 0.095, min(border_distance.x, border_distance.y));
    diffuse *= mix(0.72, 1.0, border);

    vec3 color = (diffuse + specular) * u_exposure;

    if (u_debug_mode == 1) {
        color = normal * 0.5 + 0.5;
    } else if (u_debug_mode == 2) {
        color = vec3(roughness);
    } else if (u_debug_mode == 3) {
        color = vec3(ink);
    } else if (u_debug_mode == 4) {
        color = vec3(wetness);
    }

    // Draw a tiny unobtrusive ring at the movable light position.
    if (u_debug_mode == 0) {
        color += warm_light * marker * 0.22;
    }

    frag_color = vec4(to_srgb(color), 1.0);
}
"""


OVERLAY_VERTEX_SHADER = """
#version 330

in vec2 in_position;
in vec2 in_uv;

out vec2 v_uv;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_uv;
}
"""


OVERLAY_FRAGMENT_SHADER = """
#version 330

uniform sampler2D u_panel;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    frag_color = texture(u_panel, v_uv);
}
"""


@dataclass
class InkSettings:
    wetness: float = 1.00
    light_height: float = 0.24
    exposure: float = 1.00
    debug_mode: int = 0
    drying: bool = True

    # Live-tunable material controls, all exposed as sliders.
    paper_relief: float = 1.00
    weave_soak: float = 1.00
    weave_size: float = 4.00
    dry_rate: float = 1.00
    slick_opacity: float = 0.50
    slick_swirl: float = 0.35
    slick_zoom: float = 2.20
    swirl_dryness: float = 1.00
    pen_speed: float = PEN_SPEED

    @property
    def swirl_span(self) -> float:
        """Normalises the swirl's total travel against `swirl_dryness`.

        The film's phase runs from 0 at fresh to `slick_swirl` by the time the ink
        has dried to the floor, whatever shape the slowdown takes, so the two
        sliders stay independent of each other.
        """
        return 1.0 / (1.0 - PAGE_WET_FLOOR**self.swirl_dryness)

    def advance(self, elapsed: float) -> bool:
        """Dry the page-wide ink. Returns True if the wetness changed.

        Decaying incrementally is exact for an exponential -- successive factors
        multiply to `exp(-total / tau)` -- so this is frame-rate independent and
        needs no absolute clock, which also lets `[`/`]` nudge the value.
        """
        if not self.drying or self.wetness <= PAGE_WET_FLOOR + 1e-4:
            return False

        decay = math.exp(-elapsed * self.dry_rate / INK_DRY_TAU)
        self.wetness = PAGE_WET_FLOOR + (self.wetness - PAGE_WET_FLOOR) * decay
        return True

    def rewet(self) -> None:
        """Flood the page with fresh ink again and restart the dry-down."""
        self.wetness = 1.0
        self.drying = True


class View:
    """Zoom and pan over the page.

    `center` is the canvas pixel shown at the middle of the window, so the
    mapping both ways is just `canvas = (screen - middle) / zoom + center`.
    """

    MIN_ZOOM = 0.25
    MAX_ZOOM = 16.0

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.reset()

    def reset(self) -> None:
        self.zoom = 1.0
        self.center_x = self.size[0] * 0.5
        self.center_y = self.size[1] * 0.5

    @property
    def center_uv(self) -> tuple[float, float]:
        return (self.center_x / self.size[0], 1.0 - self.center_y / self.size[1])

    def screen_to_canvas_f(self, position: tuple[int, int]) -> tuple[float, float]:
        return (
            (position[0] - self.size[0] * 0.5) / self.zoom + self.center_x,
            (position[1] - self.size[1] * 0.5) / self.zoom + self.center_y,
        )

    def screen_to_canvas(self, position: tuple[int, int]) -> tuple[int, int]:
        x, y = self.screen_to_canvas_f(position)
        return (round(x), round(y))

    def pan_by(self, screen_delta: tuple[int, int]) -> None:
        """Drag the page with the cursor rather than moving the camera."""
        self.center_x -= screen_delta[0] / self.zoom
        self.center_y -= screen_delta[1] / self.zoom

    def zoom_by(self, steps: int, anchor: tuple[int, int]) -> bool:
        """Zoom about `anchor`, keeping the canvas point under it stationary."""
        previous_zoom = self.zoom
        before = self.screen_to_canvas_f(anchor)

        self.zoom = max(
            self.MIN_ZOOM, min(self.MAX_ZOOM, self.zoom * (1.15**steps))
        )
        if self.zoom == previous_zoom:
            return False

        after = self.screen_to_canvas_f(anchor)
        self.center_x += before[0] - after[0]
        self.center_y += before[1] - after[1]
        return True


@dataclass
class Stroke:
    """One hand-drawn polyline, kept as vectors so undo can recomposite."""

    points: list[tuple[int, int]]
    radius: int
    erase: bool


@dataclass
class PaperMaps:
    albedo: pygame.Surface
    normal: pygame.Surface
    roughness: pygame.Surface


def clamp_byte(value: float) -> int:
    return max(0, min(255, round(value)))


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def value_noise(
    width: int,
    height: int,
    cell_size: int,
    rng: random.Random,
) -> list[float]:
    """Generate smooth value noise without NumPy or external image assets."""
    grid_width = width // cell_size + 2
    grid_height = height // cell_size + 2
    grid = [rng.random() for _ in range(grid_width * grid_height)]
    output = [0.0] * (width * height)

    for y in range(height):
        grid_y = y // cell_size
        local_y = smoothstep((y % cell_size) / cell_size)
        row = y * width
        grid_row = grid_y * grid_width
        next_grid_row = (grid_y + 1) * grid_width

        for x in range(width):
            grid_x = x // cell_size
            local_x = smoothstep((x % cell_size) / cell_size)

            top_left = grid[grid_row + grid_x]
            top_right = grid[grid_row + grid_x + 1]
            bottom_left = grid[next_grid_row + grid_x]
            bottom_right = grid[next_grid_row + grid_x + 1]

            top = top_left + (top_right - top_left) * local_x
            bottom = bottom_left + (bottom_right - bottom_left) * local_x
            output[row + x] = top + (bottom - top) * local_y

    return output


def generate_paper_maps(
    window_size: tuple[int, int],
    seed: int,
) -> PaperMaps:
    window_width, window_height = window_size
    width = PAPER_TEXTURE_WIDTH
    height = max(1, round(width * window_height / window_width))
    rng = random.Random(seed)

    broad = value_noise(width, height, 72, rng)
    medium = value_noise(width, height, 23, rng)
    fine = value_noise(width, height, 7, rng)

    height_map = [0.0] * (width * height)
    albedo_bytes = bytearray(width * height * 4)
    roughness_bytes = bytearray(width * height * 4)

    for y in range(height):
        row = y * width
        for x in range(width):
            index = row + x
            long_fiber = math.sin(y * 0.43 + math.sin(x * 0.031) * 2.3)
            cross_fiber = math.sin(x * 0.91 + y * 0.057) * 0.35
            fiber = (long_fiber + cross_fiber) * 0.5

            height_value = (
                broad[index] * 0.53
                + medium[index] * 0.29
                + fine[index] * 0.18
                + fiber * 0.018
            )
            height_map[index] = height_value

            stain = (broad[index] - 0.5) * 0.23 + (medium[index] - 0.5) * 0.08
            fleck = (fine[index] - 0.5) * 0.055
            brightness = 1.0 + stain + fleck

            red = 226.0 * brightness + 7.0 * (broad[index] - 0.5)
            green = 202.0 * brightness
            blue = 151.0 * brightness - 9.0 * (broad[index] - 0.5)

            byte_index = index * 4
            albedo_bytes[byte_index : byte_index + 4] = bytes(
                (clamp_byte(red), clamp_byte(green), clamp_byte(blue), 255)
            )

            roughness = 0.76 + 0.17 * (1.0 - fine[index]) + 0.05 * abs(fiber)
            roughness_byte = clamp_byte(roughness * 255.0)
            roughness_bytes[byte_index : byte_index + 4] = bytes(
                (roughness_byte, roughness_byte, roughness_byte, 255)
            )

    normal_bytes = bytearray(width * height * 4)
    normal_strength = 8.0

    def sample_height(sample_x: int, sample_y: int) -> float:
        sample_x = max(0, min(width - 1, sample_x))
        sample_y = max(0, min(height - 1, sample_y))
        return height_map[sample_y * width + sample_x]

    for y in range(height):
        for x in range(width):
            dx = sample_height(x + 1, y) - sample_height(x - 1, y)
            dy_down = sample_height(x, y + 1) - sample_height(x, y - 1)

            # Texture V points upward while Pygame Y points downward.
            nx = -dx * normal_strength
            ny = dy_down * normal_strength
            nz = 1.0
            inverse_length = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
            nx *= inverse_length
            ny *= inverse_length
            nz *= inverse_length

            byte_index = (y * width + x) * 4
            normal_bytes[byte_index : byte_index + 4] = bytes(
                (
                    clamp_byte((nx * 0.5 + 0.5) * 255.0),
                    clamp_byte((ny * 0.5 + 0.5) * 255.0),
                    clamp_byte((nz * 0.5 + 0.5) * 255.0),
                    255,
                )
            )

    return PaperMaps(
        albedo=pygame.image.frombytes(bytes(albedo_bytes), (width, height), "RGBA"),
        normal=pygame.image.frombytes(bytes(normal_bytes), (width, height), "RGBA"),
        roughness=pygame.image.frombytes(bytes(roughness_bytes), (width, height), "RGBA"),
    )


_ink_image_cache: dict[tuple[str, int | None], pygame.Surface] = {}


def load_ink_image(
    path: Path,
    height: int | None = None,
) -> pygame.Surface | None:
    """Load a standalone PNG as ink: alpha is kept, colour is forced to white.

    Only the alpha channel reaches the shader, so a black-on-transparent
    silhouette and a white-on-transparent one must behave identically. Unlike
    the pixel-art tiles this is smoothed artwork, so it is resized with
    `smoothscale` rather than point sampling.
    """
    key = (str(path), height)
    cached = _ink_image_cache.get(key)
    if cached is not None:
        return cached

    try:
        source = pygame.image.load(str(path))
    except (pygame.error, FileNotFoundError) as exc:
        print(f"Could not load ink image {path}: {exc}", file=sys.stderr)
        return None

    image = pygame.Surface(source.get_size(), pygame.SRCALPHA)
    image.fill((0, 0, 0, 0))
    image.blit(source, (0, 0))

    if height is not None and image.get_height() != height:
        width = max(1, round(image.get_width() * height / image.get_height()))
        image = pygame.transform.smoothscale(image, (width, height))

    # Max against opaque white lifts RGB to white while leaving alpha alone.
    image.fill((255, 255, 255, 0), special_flags=pygame.BLEND_RGBA_MAX)

    _ink_image_cache[key] = image
    return image


def load_canvas_normal(
    path: Path = CANVAS_NORMAL_PATH,
) -> pygame.Surface | None:
    """The woven relief of the sheet, as a tangent-space normal map.

    Returns None when the file is missing, in which case the parchment keeps the
    normal generated alongside its albedo and roughness.
    """
    try:
        image = pygame.image.load(str(path))
    except (pygame.error, FileNotFoundError) as exc:
        print(f"Could not load canvas normal {path}: {exc}", file=sys.stderr)
        return None

    surface = pygame.Surface(image.get_size(), pygame.SRCALPHA)
    surface.blit(image, (0, 0))
    return surface


def mean_tangent_deviation(surface: pygame.Surface, stride: int = 17) -> float:
    """How steep a normal map's slopes are on average: the mean of |xy|.

    Comparing two maps by this puts them on the same footing, which is what
    lets a photographed weave stand in for a procedural one without the whole
    surface suddenly reading as flat or as corrugated metal.
    """
    data = pygame.image.tobytes(surface, "RGB")
    total = 0.0
    count = 0
    for index in range(0, len(data) - 2, 3 * stride):
        x = data[index] / 127.5 - 1.0
        y = data[index + 1] / 127.5 - 1.0
        total += math.hypot(x, y)
        count += 1
    return total / max(1, count)


def mean_linear_luminance(surface: pygame.Surface) -> float:
    """Average luminance of a surface in linear light, judged from a thumbnail."""
    thumbnail = pygame.transform.smoothscale(surface, (64, 64))
    to_linear = [(value / 255.0) ** 2.2 for value in range(256)]
    data = pygame.image.tobytes(thumbnail, "RGB")

    total = 0.0
    for index in range(0, len(data), 3):
        total += (
            0.2126 * to_linear[data[index]]
            + 0.7152 * to_linear[data[index + 1]]
            + 0.0722 * to_linear[data[index + 2]]
        )
    return total / (len(data) // 3)


def load_oil_slick(
    paths: tuple[Path, ...] = OIL_SLICK_PATHS,
    size: int = OIL_SLICK_SIZE,
) -> tuple[pygame.Surface, float]:
    """Load the iridescent film that wet ink reflects.

    Returns the square, downscaled image together with the gain that lifts its
    mean linear luminance to 1.0, so using it to tint a highlight adds the
    pattern without also making the highlight brighter or darker on average.
    A plain grey stands in if the photograph is missing, which switches the
    iridescence off and leaves a colourless coat.
    """
    for path in paths:
        if not path.exists():
            continue
        try:
            source = pygame.image.load(str(path))
        except pygame.error as exc:
            print(f"Could not load oil slick {path}: {exc}", file=sys.stderr)
            continue

        # Centre square, so the swirls keep their shape instead of being squashed.
        side = min(source.get_size())
        image = pygame.Surface((side, side))
        image.blit(
            source,
            (0, 0),
            pygame.Rect(
                (source.get_width() - side) // 2,
                (source.get_height() - side) // 2,
                side,
                side,
            ),
        )

        # Halve repeatedly before the final resize: one big step of smoothscale
        # samples too few source pixels and aliases the fine filaments away.
        while image.get_width() >= size * 2:
            half = image.get_width() // 2
            image = pygame.transform.smoothscale(image, (half, half))
        if image.get_width() != size:
            image = pygame.transform.smoothscale(image, (size, size))

        return image, 1.0 / max(mean_linear_luminance(image), 1e-3)

    print(
        f"No oil slick image found at {paths[0]}; wet ink will not be iridescent.",
        file=sys.stderr,
    )
    grey = pygame.Surface((size, size))
    grey.fill((128, 128, 128))
    return grey, 1.0 / max(mean_linear_luminance(grey), 1e-3)


def surface_to_texture(
    context: moderngl.Context,
    surface: pygame.Surface,
    repeat: bool = False,
    mipmap: bool = False,
) -> moderngl.Texture:
    """Upload a Pygame surface with top-left artwork correctly oriented in GL.

    `mipmap` is only for textures that are uploaded once and never patched: the
    chain would have to be rebuilt on every partial write otherwise.
    """
    data = pygame.image.tobytes(surface, "RGBA", True)
    texture = context.texture(surface.get_size(), 4, data)
    texture.repeat_x = repeat
    texture.repeat_y = repeat
    if mipmap:
        texture.build_mipmaps()
        texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    else:
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return texture


def make_brush(radius: int, softness: float = 1.5) -> pygame.Surface:
    """A round brush whose alpha fades over the outer pixel or two.

    Stamping a soft brush along a segment gives anti-aliased strokes with round
    caps and joins, which `pygame.draw.line` cannot do; the ink mask is sampled
    one-to-one with the screen, so hard edges would read as staircase aliasing.
    """
    diameter = max(2, radius * 2)
    brush = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    center = (diameter - 1) * 0.5

    for y in range(diameter):
        delta_y = y - center
        for x in range(diameter):
            delta_x = x - center
            distance = math.hypot(delta_x, delta_y)
            coverage = (radius - distance) / softness
            alpha = clamp_byte(max(0.0, min(1.0, coverage)) * 255.0)
            if alpha:
                brush.set_at((x, y), (255, 255, 255, alpha))

    return brush


class InkCanvas:
    """The ink mask as editable artwork: generated demo art plus user strokes.

    Alongside the coverage mask the canvas keeps a `wetness` field holding how
    fresh the ink at each pixel is. Keeping it in its own surface means a drying
    stroke re-uploads one small greyscale region and never touches the coverage
    mask, and it leaves the mask's own colour channels alone.

    Coverage is held at `scale` times the page resolution because its edges are
    magnified when the view zooms in. Wetness is not: it varies slowly and is
    only ever seen through the coverage mask, so it stays at page resolution and
    the cost of drying is unchanged. Every coordinate crossing this class's
    boundary -- arguments and returned rectangles alike -- is in page pixels.
    """

    def __init__(self, size: tuple[int, int], scale: int = INK_SUPERSAMPLE) -> None:
        self.size = size
        self.scale = max(1, int(scale))
        self.mask_size = (size[0] * self.scale, size[1] * self.scale)
        self.base = pygame.Surface(self.mask_size, pygame.SRCALPHA)
        self.base.fill((0, 0, 0, 0))
        self.surface = self.base.copy()
        self.wetness = pygame.Surface(size)
        self.wetness.fill((0, 0, 0))
        # Every non-zero wetness pixel lies inside `wet_bounds`, so both the
        # decay and its upload can be confined to that rectangle.
        self.wet_bounds: pygame.Rect | None = None
        self.wet_seconds_left = 0.0
        self._wet_elapsed = 0.0
        self.strokes: list[Stroke] = []
        self.active_stroke: Stroke | None = None
        self._brushes: dict[int, pygame.Surface] = {}

    def mask_rect(self, region: pygame.Rect | None) -> pygame.Rect | None:
        """Convert a page-pixel rectangle to the coverage mask's finer grid."""
        if region is None:
            return None
        scale = self.scale
        return pygame.Rect(
            region.left * scale,
            region.top * scale,
            region.width * scale,
            region.height * scale,
        ).clip(pygame.Rect((0, 0), self.mask_size))

    def _brush(self, radius: int) -> pygame.Surface:
        """The brush for a page-pixel radius, built on the mask's finer grid.

        Softness is scaled along with the radius so the ramp stays the same
        fraction of a page pixel: supersampling is meant to resolve the existing
        edge better, not to sharpen it into a different-looking stroke.
        """
        brush = self._brushes.get(radius)
        if brush is None:
            brush = make_brush(radius * self.scale, softness=1.5 * self.scale)
            self._brushes[radius] = brush
        return brush

    def _stamp_segment(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        radius: int,
        erase: bool,
    ) -> pygame.Rect:
        """Stamp the brush along one segment; returns the touched rectangle."""
        scale = self.scale
        brush = self._brush(radius)
        offset = brush.get_width() * 0.5
        blend = pygame.BLEND_RGBA_SUB if erase else pygame.BLEND_RGBA_MAX

        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        # Spacing and travel are both in mask texels, so the stamp count is
        # unchanged by supersampling and only their placement gets finer.
        distance = math.hypot(delta_x, delta_y) * scale
        step = max(1.0, radius * scale * 0.35)
        stamps = max(1, int(distance / step) + 1)

        for index in range(stamps + 1):
            travel = index / stamps
            x = (start[0] + delta_x * travel) * scale - offset
            y = (start[1] + delta_y * travel) * scale - offset
            self.surface.blit(brush, (round(x), round(y)), special_flags=blend)

        pad = radius + 2
        dirty = pygame.Rect(
            min(start[0], end[0]) - pad,
            min(start[1], end[1]) - pad,
            abs(delta_x) + pad * 2,
            abs(delta_y) + pad * 2,
        )
        return dirty.clip(pygame.Rect((0, 0), self.size))

    def _wet_segment(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        radius: int,
    ) -> pygame.Rect:
        """Flood a segment with fresh ink in the wetness field.

        Hard-edged primitives are enough here: wetness is a slowly varying
        material property, and it only ever shows through the coverage mask, so
        spilling a pixel past the stroke edge is invisible.
        """
        fresh = (255, 255, 255)
        reach = radius + 1
        if start != end:
            pygame.draw.line(self.wetness, fresh, start, end, reach * 2)
        pygame.draw.circle(self.wetness, fresh, start, reach)
        pygame.draw.circle(self.wetness, fresh, end, reach)

        pad = reach + 2
        return self.mark_wet(
            pygame.Rect(
                min(start[0], end[0]) - pad,
                min(start[1], end[1]) - pad,
                abs(end[0] - start[0]) + pad * 2,
                abs(end[1] - start[1]) + pad * 2,
            )
        )

    def mark_wet(self, region: pygame.Rect) -> pygame.Rect:
        """Note that fresh ink was put into the wetness field inside `region`.

        Whatever wrote the field -- a brush segment, or a sprite uncovering
        itself -- the drying clock is restarted and the region the decay has to
        sweep is widened to include it.
        """
        dirty = region.clip(self.wetness.get_rect())
        self.wet_bounds = (
            dirty if self.wet_bounds is None else self.wet_bounds.union(dirty)
        )
        self.wet_seconds_left = INK_DRY_SECONDS
        return dirty

    def _draw_segment(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        radius: int,
        erase: bool,
    ) -> pygame.Rect:
        """Lay down one segment of live ink: coverage plus its drying clock."""
        dirty = self._stamp_segment(start, end, radius, erase)
        if erase:
            # Lifting ink off the page leaves nothing behind to dry.
            return dirty
        return dirty.union(self._wet_segment(start, end, radius))

    def wet_gain(self, rate: float = 1.0) -> float:
        """How much further the field has faded since its last step.

        The field itself only moves in steps, which have to be far enough apart
        to keep the multiply blend's floor under dry ink. Handing the shader the
        fade for the part-step since then makes the result continuous, so a
        sixty-second dry-down still looks smooth on a field stepping once a
        second.
        """
        if self.wet_bounds is None:
            return 1.0
        return math.exp(-self._wet_elapsed * rate / INK_DRY_TAU)

    def dry(self, elapsed: float, rate: float = 1.0) -> pygame.Rect | None:
        """Fade the stroke wetness field; returns the region needing re-upload.

        Every pixel fades by the same factor each step, so a pixel stamped
        `age` seconds ago is left holding roughly `exp(-age / tau)` -- no
        per-stroke bookkeeping is needed to give each stroke its own clock.

        `rate` scales how fast ink dries. The interval between steps scales with
        it so that each step is always the same fraction of a time constant:
        that keeps the multiply blend's stall floor pinned under the sheen of dry
        ink, where a slow dry-down with frequent steps would leave it stranded at
        a visible wetness.
        """
        if self.wet_bounds is None:
            return None

        interval = max(1.0 / 60.0, WET_UPDATE_INTERVAL / rate)
        self._wet_elapsed += elapsed
        if self._wet_elapsed < interval:
            return None

        step = self._wet_elapsed
        self._wet_elapsed = 0.0
        self.wet_seconds_left -= step * rate
        region = self.wet_bounds

        if self.wet_seconds_left <= 0.0:
            # Land exactly on dry, so nothing keeps a stale trace of wetness.
            self.wetness.fill((0, 0, 0))
            self.wet_bounds = None
            return region

        # The blend divides by 256, not 255, so that is what the factor scales.
        scale = clamp_byte(math.exp(-step * rate / INK_DRY_TAU) * 256.0)
        self.wetness.subsurface(region).fill(
            (scale, scale, scale), special_flags=pygame.BLEND_RGB_MULT
        )
        return region

    def dry_now(self) -> pygame.Rect | None:
        """Take every trace of wetness off the page at once.

        The pen uses it so a drawing begins on a dry sheet whatever was there a
        moment earlier; without it the fresh nib would be laying wet ink onto a
        page that was already wet, and there would be nothing to see.
        """
        if self.wet_bounds is None:
            return None
        region = self.wet_bounds
        self.wetness.fill((0, 0, 0))
        self.wet_bounds = None
        self.wet_seconds_left = 0.0
        self._wet_elapsed = 0.0
        return region

    def begin_stroke(
        self,
        position: tuple[int, int],
        radius: int,
        erase: bool,
    ) -> pygame.Rect:
        self.active_stroke = Stroke([position], radius, erase)
        self.strokes.append(self.active_stroke)
        return self._draw_segment(position, position, radius, erase)

    def extend_stroke(self, position: tuple[int, int]) -> pygame.Rect | None:
        stroke = self.active_stroke
        if stroke is None:
            return None

        previous = stroke.points[-1]
        if previous == position:
            return None

        stroke.points.append(position)
        return self._draw_segment(previous, position, stroke.radius, stroke.erase)

    def end_stroke(self) -> None:
        self.active_stroke = None

    def undo(self) -> bool:
        if not self.strokes:
            return False
        self.strokes.pop()
        self.active_stroke = None
        self.recomposite()
        return True

    def clear_strokes(self) -> bool:
        if not self.strokes:
            return False
        self.strokes.clear()
        self.active_stroke = None
        self.recomposite()
        return True

    def set_base(self, base: pygame.Surface) -> None:
        self.base = base
        self.recomposite()

    def recomposite(self) -> None:
        """Redraw everything from the base art. Only needed for undo and clear."""
        self.surface = self.base.copy()
        for stroke in self.strokes:
            points = stroke.points
            self._stamp_segment(points[0], points[0], stroke.radius, stroke.erase)
            for start, end in zip(points, points[1:]):
                self._stamp_segment(start, end, stroke.radius, stroke.erase)


PANEL_FILL = (24, 19, 14, 234)
PANEL_EDGE = (150, 124, 86, 255)
PANEL_TITLE = (240, 218, 170)
PANEL_KEY = (248, 220, 148)
PANEL_TEXT = (224, 213, 194)


def bounds_of(flags) -> tuple[int, int, int, int] | None:
    """The box enclosing everything set in a boolean array, or None if empty.

    Reduced along each axis first, so the search is over a row and a column
    rather than the whole array; the caller uses this every frame on a field
    of a million entries.
    """
    rows = flags.any(axis=1)
    if not rows.any():
        return None
    columns = flags.any(axis=0)
    top = int(rows.argmax())
    bottom = len(rows) - int(rows[::-1].argmax())
    left = int(columns.argmax())
    right = len(columns) - int(columns[::-1].argmax())
    return left, top, right, bottom


class SpriteReveal:
    """Draws a sprite on to the page by uncovering it along a pen's path.

    What makes a traced drawing exact is that it is uncovered rather than
    redrawn. Redrawing a figure by stamping a
    round brush along its medial axis lands within a few percent of it and no
    closer, because the brush is not the tool the figure was made with. Here
    the pixels put down are the sprite's own, in an order the traced pen path
    supplies, so the drawing ends up identical to the artwork rather than a
    good likeness of it.

    Coverage is uncovered on the mask's fine grid and wetness on the page's
    coarse one, which is the same split the rest of the canvas uses; the two
    travel maps are built together so a pixel wets exactly as it appears.
    """

    def __init__(
        self,
        sprite: pygame.Surface,
        mask_rect: pygame.Rect,
        page_rect: pygame.Rect,
        travel,
        page_travel,
        scale: int = 1,
        speed: float = PEN_SPEED,
        unit: float = 1.0,
    ) -> None:
        self.sprite = sprite
        self.mask_rect = mask_rect
        self.page = page_rect
        self.scale = scale
        self.travel = travel
        self.page_travel = page_travel
        self.speed = speed
        # Mask texels per design-space unit, so the pen's speed means the same
        # thing here as it does when it is drawing with the brush.
        self.unit = unit
        self.opaque = pygame.surfarray.array_alpha(sprite)
        self.reached = 0.0
        # The first frame has to take in the pixels the nib was already on when
        # it touched down, whose travel is exactly nothing.
        self.started = False
        reachable = travel[numpy.isfinite(travel)]
        self.total = float(reachable.max()) if reachable.size else 0.0
        # Reused rather than rebuilt every frame; only its alpha changes.
        self.piece = sprite.copy()

    @property
    def finished(self) -> bool:
        return self.reached >= self.total

    @property
    def progress(self) -> float:
        return 1.0 if self.total <= 0.0 else min(1.0, self.reached / self.total)

    def update(self, elapsed: float, canvas: InkCanvas) -> pygame.Rect | None:
        """Uncover as much of the sprite as the pen reaches this frame.

        Everything below is confined to the box the nib actually moved through
        this frame. That matters more than it looks: the figure is a thousand
        texels square at a large window, and touching all of it every frame --
        blitting it, and worse, re-sending it to the card -- costs twenty-odd
        milliseconds a frame, while the sliver the pen swept costs a fraction
        of one.
        """
        if self.finished:
            return None

        previous = self.reached if self.started else -1.0
        self.started = True
        self.reached = min(
            self.total, self.reached + self.speed * self.unit * elapsed
        )

        uncovered = (self.travel > previous) & (self.travel <= self.reached)
        box = bounds_of(uncovered)
        if box is None:
            return None
        left, top, right, bottom = box

        # Rebuild the sprite's alpha only across that box, then blit the same
        # box: what is already on the page stays there, coverage being
        # accumulated with a max blend.
        alpha = pygame.surfarray.pixels_alpha(self.piece)
        numpy.multiply(
            self.opaque[left:right, top:bottom],
            (self.travel[top:bottom, left:right] <= self.reached).T,
            out=alpha[left:right, top:bottom],
            casting="unsafe",
        )
        del alpha
        patch = pygame.Rect(left, top, right - left, bottom - top)
        canvas.surface.blit(
            self.piece,
            (self.mask_rect.left + left, self.mask_rect.top + top),
            patch,
            special_flags=pygame.BLEND_RGBA_MAX,
        )

        # Wetness, on the other hand, is written only where the pen has just
        # been: what is behind it has to be left alone to dry.
        dirty = pygame.Rect(
            self.page.left + patch.left // self.scale,
            self.page.top + patch.top // self.scale,
            -(-patch.width // self.scale) + 1,
            -(-patch.height // self.scale) + 1,
        ).clip(self.page)

        fresh = (self.page_travel > previous) & (
            self.page_travel <= self.reached
        )
        wet_box = bounds_of(fresh)
        if wet_box is not None:
            field = pygame.surfarray.pixels3d(canvas.wetness)
            window = field[
                self.page.left : self.page.right,
                self.page.top : self.page.bottom,
            ]
            window[fresh.T] = 255
            del window, field
            wet_left, wet_top, wet_right, wet_bottom = wet_box
            dirty = dirty.union(
                canvas.mark_wet(
                    pygame.Rect(
                        self.page.left + wet_left,
                        self.page.top + wet_top,
                        wet_right - wet_left,
                        wet_bottom - wet_top,
                    )
                )
            )

        return dirty


_travel_cache: dict[tuple, tuple] = {}


def _page_travel(travel, scale: int):
    """Reduce a mask-resolution travel map to one value per page pixel.

    A page pixel is reached as soon as any of the mask texels inside it is, so
    the reduction is a minimum. Padded out to a whole number of page pixels
    first; the padding is unreachable and so never fires.
    """
    if scale == 1:
        return travel
    height, width = travel.shape
    tall, wide = -(-height // scale), -(-width // scale)
    padded = numpy.full((tall * scale, wide * scale), numpy.inf, numpy.float32)
    padded[:height, :width] = travel
    return padded.reshape(tall, scale, wide, scale).min(axis=(1, 3))


def sprite_reveal(
    path: Path,
    placed: pygame.Rect,
    scale: int,
    unit: float,
    speed: float = PEN_SPEED,
) -> SpriteReveal | None:
    """Work out the order a pen would draw a sprite in, and stand ready to.

    The sprite is traced at exactly the size it will appear -- no rescaling
    anywhere -- so the timings and the pixels they belong to are the same grid,
    and what ends up on the page is the artwork rather than a redrawing of it.

    Tracing costs a handful of passes over the image, so the answer is kept:
    pressing the key again asks the same question of the same sprite.
    """
    if pentrace is None:
        print("Drawing a sprite needs numpy: python -m pip install numpy",
              file=sys.stderr)
        return None

    sprite = load_ink_image(path, placed.height)
    if sprite is None:
        return None

    key = (str(path), placed.height, scale)
    cached = _travel_cache.get(key)
    if cached is None:
        # Coverage lives in alpha, and pygame hands out arrays column-first
        # while the tracer works in rows, hence the transpose.
        showing = pygame.surfarray.array_alpha(sprite).T > 0

        # The path is worked out on a copy no larger than this. It decides only
        # the order pixels appear in, never which ones or what they look like,
        # so a coarser one is invisible -- while tracing at the full size of a
        # figure a thousand texels tall takes well over a second, all of it in
        # a single stall the moment the key is pressed.
        step = max(1, -(-placed.height // TRACE_MAX_SIZE))
        coarse = pygame.surfarray.array_alpha(
            sprite if step == 1 else pygame.transform.smoothscale(
                sprite, (placed.width // step, placed.height // step)
            )
        ).T
        strokes = pentrace.trace_drawing(coarse > 127)
        travel = pentrace.reveal_travel(
            coarse > 0,
            strokes,
            lift=PEN_TOUCHDOWN_SECONDS * PEN_SPEED * unit / step,
            slack=NIB_SLACK * scale / step,
        )
        if step > 1:
            # Back up to the real grid, distances and all, and let the pixels
            # the coarse copy could not see take their time from a neighbour.
            travel = numpy.kron(travel * step, numpy.ones((step, step), numpy.float32))
            travel = numpy.pad(
                travel,
                ((0, max(0, showing.shape[0] - travel.shape[0])),
                 (0, max(0, showing.shape[1] - travel.shape[1]))),
                constant_values=numpy.inf,
            )[: showing.shape[0], : showing.shape[1]]
            travel = numpy.where(showing, travel, numpy.inf)
            travel = pentrace.fill_unreached(travel, showing, float(travel[
                numpy.isfinite(travel)].max() if numpy.isfinite(travel).any() else 0.0))

        cached = (travel, _page_travel(travel, scale))
        _travel_cache[key] = cached

    travel, page_travel = cached
    page = pygame.Rect(
        placed.left // scale,
        placed.top // scale,
        page_travel.shape[1],
        page_travel.shape[0],
    )
    return SpriteReveal(
        sprite, placed, page, travel, page_travel, scale, speed, unit
    )


def make_panel_surface(size: tuple[int, int]) -> pygame.Surface:
    """A bordered, mostly opaque card for the interface overlays to draw on."""
    panel = pygame.Surface(size, pygame.SRCALPHA)
    panel.fill(PANEL_FILL)
    pygame.draw.rect(panel, PANEL_EDGE, panel.get_rect(), 2)
    return panel


class Panel:
    """A pixel-positioned, alpha-blended quad: how the overlays reach the screen.

    The window is an OpenGL surface, so there is nothing to blit text onto. Each
    overlay lays itself out with Pygame's font renderer and hands the surface
    here to be drawn over the finished page, at a fixed size in window pixels so
    zooming and panning the page leave it alone.
    """

    def __init__(
        self,
        context: moderngl.Context,
        window_size: tuple[int, int],
        surface: pygame.Surface,
        position: tuple[int, int] | None = None,
    ) -> None:
        self.context = context
        self.visible = False

        window_width, window_height = window_size
        panel_width, panel_height = surface.get_size()
        if position is None:
            # Centre it, on whole pixels so the text stays crisp.
            position = (
                (window_width - panel_width) // 2,
                (window_height - panel_height) // 2,
            )
        self.rect = pygame.Rect(position, surface.get_size())

        self.texture = surface_to_texture(context, surface)
        self.program = context.program(
            vertex_shader=OVERLAY_VERTEX_SHADER,
            fragment_shader=OVERLAY_FRAGMENT_SHADER,
        )
        self.program["u_panel"].value = 0

        x0 = 2.0 * self.rect.left / window_width - 1.0
        x1 = 2.0 * self.rect.right / window_width - 1.0
        y0 = 1.0 - 2.0 * self.rect.bottom / window_height
        y1 = 1.0 - 2.0 * self.rect.top / window_height

        vertices = array(
            "f",
            (
                x0, y0, 0.0, 0.0,
                x1, y0, 1.0, 0.0,
                x0, y1, 0.0, 1.0,
                x1, y1, 1.0, 1.0,
            ),
        )
        self.vertex_buffer = context.buffer(vertices.tobytes())
        self.vertex_array = context.vertex_array(
            self.program,
            [(self.vertex_buffer, "2f 2f", "in_position", "in_uv")],
        )

    def update(self, surface: pygame.Surface) -> None:
        """Replace the artwork. The size is fixed by the quad, so it must match."""
        self.texture.write(pygame.image.tobytes(surface, "RGBA", True))

    def toggle(self) -> None:
        self.visible = not self.visible

    def draw(self) -> None:
        """Composite the card over whatever has already been rendered."""
        self.context.enable(moderngl.BLEND)
        self.context.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )
        self.texture.use(location=0)
        self.vertex_array.render(mode=moderngl.TRIANGLE_STRIP)
        self.context.disable(moderngl.BLEND)

    def release(self) -> None:
        self.texture.release()
        self.vertex_array.release()
        self.vertex_buffer.release()
        self.program.release()


class ParchmentInkRenderer:
    def __init__(
        self,
        size: tuple[int, int],
        seed: int = 7,
        supersample: int = INK_SUPERSAMPLE,
        headless: bool = False,
    ) -> None:
        self.size = size
        self.seed = seed
        self.headless = headless

        if headless:
            # A context with no window behind it at all, and a buffer of our
            # own to draw into. Everything downstream is identical: the page is
            # rendered by the same shader into a framebuffer of the same shape,
            # and the only thing missing is the part that would have put it on
            # screen. That is what makes it worth testing through -- it is the
            # real renderer, not a stand-in for it.
            if wsl_graphics_card_available():
                # Nothing has made a context yet, so the choice is still open.
                use_wsl_graphics_card()
            self.context = moderngl.create_context(
                standalone=True, backend="egl", require=330
            )
            self.target = self.context.simple_framebuffer(size, components=4)
        else:
            self.context = moderngl.create_context(require=330)
            self.target = self.context.screen
            # Re-opening the window for a new resolution leaves ModernGL's idea
            # of the default framebuffer at the old size, so say what it is.
            # Drawing sets the viewport itself; this is for anything that reads
            # the screen back.
            self.target.viewport = (0, 0, *size)

        self.context.disable(moderngl.DEPTH_TEST)
        self.context.disable(moderngl.CULL_FACE)

        self.program = self.context.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        vertices = array(
            "f",
            (
                -1.0, -1.0, 0.0, 0.0,
                +1.0, -1.0, 1.0, 0.0,
                -1.0, +1.0, 0.0, 1.0,
                +1.0, +1.0, 1.0, 1.0,
            ),
        )
        self.vertex_buffer = self.context.buffer(vertices.tobytes())
        self.vertex_array = self.context.vertex_array(
            self.program,
            [(self.vertex_buffer, "2f 2f", "in_position", "in_uv")],
        )

        # The page starts blank. Whatever is printed on it is the caller's
        # business: hand it in with `set_page_art`.
        self.canvas = InkCanvas(size, scale=supersample)

        self.paper_albedo: moderngl.Texture
        self.paper_normal: moderngl.Texture
        self.ink_grain: moderngl.Texture
        self.paper_roughness: moderngl.Texture
        self.ink_mask = self.context.texture(self.canvas.mask_size, 4)
        self.ink_mask.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.ink_mask.repeat_x = False
        self.ink_mask.repeat_y = False

        # One byte per pixel: how fresh the ink there is.
        self.ink_wet = self.context.texture(size, 1)
        self.ink_wet.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.ink_wet.repeat_x = False
        self.ink_wet.repeat_y = False

        slick, slick_gain = load_oil_slick()
        # The shader folds the coordinate to tile it, so clamping is what we want.
        self.oil_slick = surface_to_texture(self.context, slick)
        self.slick_gain = slick_gain

        self.upload_ink()
        self.upload_wetness()
        self.canvas_normal = load_canvas_normal()
        self.normal_gain = 1.0
        self._replace_paper_textures(seed)

        self.program["u_paper_albedo"].value = 0
        self.program["u_paper_normal"].value = 1
        self.program["u_paper_roughness"].value = 2
        self.program["u_ink_mask"].value = 3
        self.program["u_ink_wet"].value = 4
        self.program["u_oil_slick"].value = 5
        self.program["u_ink_grain"].value = 6
        self.program["u_slick_gain"].value = self.slick_gain
        self.program["u_resolution"].value = tuple(float(value) for value in size)
        # The dry-down stops here rather than at zero, which is what the weave's
        # changeover has to be measured against.
        self.program["u_wet_floor"].value = PAGE_WET_FLOOR

    def set_page_art(
        self, art: pygame.Surface, changed: pygame.Rect | None = None
    ) -> None:
        """Print artwork on to the page, underneath anything drawn on it.

        `changed` names the only part that can have moved, when the caller
        knows it -- swapping one figure in or out, say. Everything outside it
        already matches what the card is holding, and re-sending the whole mask
        is not free: at four times anti-aliasing it is thirty-three million
        texels and the best part of a tenth of a second.
        """
        self.canvas.set_base(art)
        self.upload_ink(changed)

    def _write_region(
        self,
        texture: moderngl.Texture,
        surface: pygame.Surface,
        region: pygame.Rect | None,
        red_only: bool = False,
    ) -> None:
        """Upload part of a surface into a texture, by default the whole thing.

        Texture V runs bottom-up while Pygame Y runs top-down, so the region is
        flipped and its origin mirrored to match `surface_to_texture`.
        """
        if region is None:
            region = surface.get_rect()
        else:
            region = region.clip(surface.get_rect())
            if not region.width or not region.height:
                return

        patch = pygame.transform.flip(surface.subsurface(region), False, True)
        if red_only:
            data = pygame.image.tobytes(patch, "RGB")[0::3]
        else:
            data = pygame.image.tobytes(patch, "RGBA")

        texture.write(
            data,
            viewport=(
                region.left,
                surface.get_height() - region.bottom,
                region.width,
                region.height,
            ),
        )

    def upload_ink(self, region: pygame.Rect | None = None) -> None:
        """Push the ink coverage to the GPU, only the changed rectangle.

        The region arrives in page pixels, like every rectangle the canvas
        hands out, and is converted here to the mask's own grid.
        """
        self._write_region(
            self.ink_mask, self.canvas.surface, self.canvas.mask_rect(region)
        )

    def upload_wetness(self, region: pygame.Rect | None = None) -> None:
        """Push the stroke wetness field to its single-channel texture."""
        self._write_region(self.ink_wet, self.canvas.wetness, region, red_only=True)

    def upload_stroke(self, region: pygame.Rect | None) -> None:
        """Push both halves of freshly drawn ink: what it covers and how wet."""
        if region is None:
            return
        self.upload_ink(region)
        self.upload_wetness(region)

    def _weave_tiles(self, weave_size: float) -> tuple[float, float]:
        """How often the weave repeats, for a wanted spacing between threads.

        `weave_size` is in page pixels, so asking for finer marks means more
        repeats. The generated parchment is made at the window's aspect and
        covers the page exactly once, so it stays at one repeat; a photograph
        has an aspect of its own, and stretching it would shear the weave.
        """
        if self.canvas_normal is None:
            return (1.0, 1.0)

        weave_width, weave_height = self.canvas_normal.get_size()
        page_width, page_height = self.size
        across = (
            CANVAS_WEAVE_PERIOD * page_width / (max(weave_size, 0.1) * weave_width)
        )
        down = across * (page_height / page_width) * (weave_width / weave_height)
        return (across, down)

    def _replace_paper_textures(self, seed: int) -> None:
        """Regenerate the parchment for a new seed.

        The weave, when there is one, is a photograph rather than something the
        seed produces, so it is uploaded once and left alone; the albedo, the
        roughness and the grain under the ink are rebuilt each time.
        """
        maps = generate_paper_maps(self.size, seed)
        new_textures = {
            "paper_albedo": surface_to_texture(self.context, maps.albedo),
            "paper_roughness": surface_to_texture(self.context, maps.roughness),
            # The ink is lit by the generated grain whether or not a weave was
            # loaded, so a coat of ink looks the same as it always did.
            "ink_grain": surface_to_texture(self.context, maps.normal),
        }

        if self.canvas_normal is None:
            new_textures["paper_normal"] = surface_to_texture(
                self.context, maps.normal
            )
            self.normal_gain = 1.0
        elif getattr(self, "paper_normal", None) is None:
            new_textures["paper_normal"] = surface_to_texture(
                self.context, self.canvas_normal, repeat=True, mipmap=True
            )
            # The photographed weave has far gentler slopes than the procedural
            # parchment. Matching their average puts it at the depth the rest of
            # the lighting was tuned against, so `paper_relief` reads as 100%.
            self.normal_gain = mean_tangent_deviation(maps.normal) / max(
                mean_tangent_deviation(self.canvas_normal), 1e-4
            )

        for name, texture in new_textures.items():
            old_texture = getattr(self, name, None)
            if old_texture is not None:
                old_texture.release()
            setattr(self, name, texture)

    def regenerate_paper(self) -> None:
        self.seed += 1
        self._replace_paper_textures(self.seed)

    def render(
        self,
        light_position: tuple[int, int],
        settings: InkSettings,
        view: View | None = None,
    ) -> None:
        width, height = self.size
        light_uv = (
            max(0.0, min(1.0, light_position[0] / width)),
            max(0.0, min(1.0, 1.0 - light_position[1] / height)),
        )

        self.target.use()
        self.context.viewport = (0, 0, width, height)
        # No clear: the page is one quad covering the whole viewport, drawn
        # without blending, so every pixel is written before anything reads it.
        # Clearing first is a second full pass over the framebuffer for nothing,
        # which at 4K is a sixth of the frame.

        self.paper_albedo.use(location=0)
        self.paper_normal.use(location=1)
        self.paper_roughness.use(location=2)
        self.ink_mask.use(location=3)
        self.ink_wet.use(location=4)
        self.oil_slick.use(location=5)
        self.ink_grain.use(location=6)

        self.program["u_light_uv"].value = light_uv
        self.program["u_light_height"].value = settings.light_height
        self.program["u_paper_relief"].value = settings.paper_relief * self.normal_gain
        self.program["u_weave_soak"].value = max(settings.weave_soak, 1e-3)
        self.program["u_paper_normal_tiles"].value = self._weave_tiles(
            settings.weave_size
        )
        self.program["u_page_wetness"].value = settings.wetness
        self.program["u_wet_gain"].value = self.canvas.wet_gain(settings.dry_rate)
        self.program["u_slick_opacity"].value = settings.slick_opacity
        self.program["u_slick_swirl"].value = settings.slick_swirl
        self.program["u_slick_zoom"].value = max(settings.slick_zoom, 1e-3)
        self.program["u_swirl_dryness"].value = settings.swirl_dryness
        self.program["u_swirl_span"].value = settings.swirl_span
        self.program["u_exposure"].value = settings.exposure
        self.program["u_debug_mode"].value = settings.debug_mode
        self.program["u_zoom"].value = 1.0 if view is None else view.zoom
        self.program["u_center_uv"].value = (
            (0.5, 0.5) if view is None else view.center_uv
        )

        self.vertex_array.render(mode=moderngl.TRIANGLE_STRIP)
        # Anything drawn over the page -- a panel, a card -- composites itself
        # after this, into the target that is still bound.

    def capture(self, components: int = 3) -> bytes:
        """Read the rendered page back, bottom row first as OpenGL stores it.

        The one thing a test needs that a window would otherwise have to
        provide. Reading the default framebuffer works too, so this is the same
        call either way and a test does not have to know which it is looking at.
        """
        return bytes(self.target.read(viewport=(0, 0, *self.size),
                                      components=components))

    def snapshot(self) -> pygame.Surface:
        """The rendered page as a surface, the right way up."""
        return pygame.image.frombytes(
            self.capture(3), self.size, "RGB", True
        )

    def release(self) -> None:
        if self.headless:
            self.target.release()
        self.paper_albedo.release()
        self.paper_normal.release()
        self.ink_grain.release()
        self.paper_roughness.release()
        self.ink_mask.release()
        self.ink_wet.release()
        self.oil_slick.release()
        self.vertex_array.release()
        self.vertex_buffer.release()
        self.program.release()


