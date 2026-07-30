from __future__ import annotations

import math
import random
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path

try:
    import pygame
    import moderngl
except ImportError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing {missing!r}. Install the dependencies with:\n"
        "    python -m pip install pygame moderngl"
    ) from exc


WINDOW_SIZE = (1000, 700)
PAPER_TEXTURE_WIDTH = 512

REPO_ROOT = Path(__file__).resolve().parents[2]
SPRITE_SHEET_PATH = (
    REPO_ROOT / "gfx" / "tilesets" / "Hexany" / "monochrome_32x32_transparent.png"
)
SPRITE_TILE_SIZE = 32

FEATURE_IMAGE_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "inv"
    / "creature_059_smoothed_512.png"
)
FEATURE_IMAGE_HEIGHT = 160

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
uniform sampler2D u_paper_roughness;
uniform sampler2D u_ink_mask;
uniform sampler2D u_ink_wet;
uniform sampler2D u_oil_slick;
uniform float u_slick_gain;
uniform float u_time;

// How much of the oil film shows over the ink, how fast it swirls while the ink
// is still liquid, and how much of the sheet one copy of the photograph covers.
// All three are taste rather than physics. Everything feeding the lookup is kept
// in page units and scaled once at the end, so size cannot disturb speed.
uniform float u_slick_opacity;
uniform float u_slick_swirl;
uniform float u_slick_zoom;

uniform vec2 u_resolution;
uniform vec2 u_light_uv;
uniform float u_light_height;
uniform float u_page_wetness;
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

    vec2 texel = 1.0 / vec2(textureSize(u_ink_mask, 0));

    vec3 paper_albedo = to_linear(texture(u_paper_albedo, page_uv).rgb);
    vec3 paper_normal = normalize(texture(u_paper_normal, page_uv).rgb * 2.0 - 1.0);
    float paper_roughness = texture(u_paper_roughness, page_uv).r;

    float ink = texture(u_ink_mask, page_uv).a;
    float ink_left = texture(u_ink_mask, page_uv - vec2(texel.x, 0.0)).a;
    float ink_right = texture(u_ink_mask, page_uv + vec2(texel.x, 0.0)).a;
    float ink_down = texture(u_ink_mask, page_uv - vec2(0.0, texel.y)).a;
    float ink_up = texture(u_ink_mask, page_uv + vec2(0.0, texel.y)).a;

    // Wetness has two sources: the page-wide dry-down of the printed artwork,
    // and a per-pixel field that gives every fresh stroke its own drying clock.
    float wetness = max(u_page_wetness, texture(u_ink_wet, page_uv).r);

    // The alpha gradient makes wet strokes look microscopically raised. Drying
    // ink sinks into the fibres, so the relief nearly vanishes with the shine.
    vec2 ink_gradient = vec2(ink_right - ink_left, ink_up - ink_down);
    float edge = saturate(length(ink_gradient) * 3.2);

    vec3 ink_normal = normalize(vec3(
        paper_normal.xy * 0.22 - ink_gradient * (0.30 + 2.30 * wetness),
        max(0.30, paper_normal.z)
    ));
    vec3 normal = normalize(mix(paper_normal, ink_normal, ink));

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
    // Every phase advances at a fixed rate; only the amplitude follows the
    // wetness, otherwise drying ink would jerk the swirl backwards.
    vec2 flow = vec2(page_uv.x * aspect, page_uv.y)
        + vec2(0.033, -0.052) * u_time * u_slick_swirl;
    flow += vec2(
        sin(flow.y * 7.7 + u_time * 1.7 * u_slick_swirl),
        cos(flow.x * 6.6 - u_time * 1.3 * u_slick_swirl)
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
class DemoSettings:
    wetness: float = 1.00
    light_height: float = 0.24
    exposure: float = 1.00
    debug_mode: int = 0
    brush_radius: int = 5
    drying: bool = True
    moment: float = 0.0

    # Live-tunable material controls, all exposed as sliders.
    dry_rate: float = 1.00
    slick_opacity: float = 0.50
    slick_swirl: float = 0.35
    slick_zoom: float = 2.20

    def advance(self, elapsed: float) -> bool:
        """Dry the page-wide ink. Returns True if the wetness changed.

        Decaying incrementally is exact for an exponential -- successive factors
        multiply to `exp(-total / tau)` -- so this is frame-rate independent and
        needs no absolute clock, which also lets `[`/`]` nudge the value.
        """
        # The clock the slick swirls on keeps running even when drying is held.
        self.moment += elapsed

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


class SpriteSheet:
    """A grid of equally sized tiles cut out of one image.

    The Hexany monochrome sheet is pure white on transparency, so a tile's
    alpha channel drops straight into the ink mask and the sprite is lit as
    though it had been stamped onto the parchment in the same ink as the
    hand-drawn strokes.
    """

    def __init__(self, path: Path, tile_size: int = SPRITE_TILE_SIZE) -> None:
        source = pygame.image.load(str(path))

        # Blitting onto an SRCALPHA surface resolves the palette + colour key
        # of an indexed PNG into real per-pixel alpha, and unlike
        # convert_alpha() it does not need a display surface.
        self.surface = pygame.Surface(source.get_size(), pygame.SRCALPHA)
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(source, (0, 0))

        self.tile_size = tile_size
        sheet_width, sheet_height = self.surface.get_size()
        self.columns = sheet_width // tile_size
        self.rows = sheet_height // tile_size
        self._scaled_cache: dict[tuple[int, int], pygame.Surface] = {}

    @property
    def count(self) -> int:
        return self.columns * self.rows

    def tile_rect(self, index: int) -> pygame.Rect:
        index %= self.count
        column = index % self.columns
        row = index // self.columns
        return pygame.Rect(
            column * self.tile_size,
            row * self.tile_size,
            self.tile_size,
            self.tile_size,
        )

    def tile(self, index: int, scale: int = 1) -> pygame.Surface:
        """Return one tile, optionally point-scaled to keep pixels crisp."""
        index %= self.count
        key = (index, scale)
        cached = self._scaled_cache.get(key)
        if cached is not None:
            return cached

        tile = pygame.Surface(
            (self.tile_size, self.tile_size), pygame.SRCALPHA
        )
        tile.fill((0, 0, 0, 0))
        tile.blit(self.surface, (0, 0), self.tile_rect(index))

        if scale != 1:
            size = (self.tile_size * scale, self.tile_size * scale)
            tile = pygame.transform.scale(tile, size)

        self._scaled_cache[key] = tile
        return tile

    def draw(
        self,
        target: pygame.Surface,
        index: int,
        position: tuple[int, int],
        scale: int = 1,
        opacity: int = 255,
    ) -> pygame.Rect:
        """Stamp one tile onto `target`, treating it as ink of a given density."""
        tile = self.tile(index, scale)
        if opacity != 255:
            tile = tile.copy()
            tile.fill(
                (255, 255, 255, opacity),
                special_flags=pygame.BLEND_RGBA_MULT,
            )
        return target.blit(tile, position)


def load_sprite_sheet(path: Path = SPRITE_SHEET_PATH) -> SpriteSheet | None:
    """Load the sprite sheet, or return None so the demo still runs without it."""
    try:
        return SpriteSheet(path)
    except (pygame.error, FileNotFoundError) as exc:
        print(f"Could not load sprite sheet {path}: {exc}", file=sys.stderr)
        return None


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


def make_demo_ink(
    size: tuple[int, int],
    sprites: SpriteSheet | None = None,
    sprite_page: int = 0,
) -> pygame.Surface:
    """Create anti-aliased ink artwork; its alpha channel becomes the ink mask."""
    width, height = size
    ink_mask = pygame.Surface(size, pygame.SRCALPHA)
    ink_mask.fill((0, 0, 0, 0))

    title_font = pygame.font.SysFont("serif", 58, bold=True)
    body_font = pygame.font.SysFont("serif", 28)
    small_font = pygame.font.SysFont("serif", 20, italic=True)

    def draw_text(
        text: str,
        font: pygame.font.Font,
        position: tuple[int, int],
        opacity: int = 255,
    ) -> None:
        rendered = font.render(text, True, (255, 255, 255))
        rendered.set_alpha(opacity)
        ink_mask.blit(rendered, position)

    draw_text("The Cartographer's Ledger", title_font, (90, 72))
    draw_text(
        "Move the mouse: one light, two shader materials.",
        body_font,
        (94, 152),
    )
    draw_text(
        "Rough parchment scatters light; fresh ink forms a glossy coat.",
        small_font,
        (96, 196),
    )

    white = (255, 255, 255, 255)
    pygame.draw.line(ink_mask, white, (95, 245), (width - 95, 245), 5)
    pygame.draw.circle(ink_mask, white, (width // 2, 245), 13, 3)

    points: list[tuple[int, int]] = []
    for x in range(120, width - 120, 8):
        y = 390 + int(52 * math.sin(x * 0.018)) + int(16 * math.sin(x * 0.053))
        points.append((x, y))
    pygame.draw.lines(ink_mask, white, False, points, 12)

    # The rose sits up beside the title so the lower right stays free for the
    # featured creature and the sprite strip.
    center = (width - 145, 320)
    pygame.draw.circle(ink_mask, white, center, 62, 5)
    for angle in range(0, 360, 45):
        vector = pygame.Vector2(0, -54).rotate(angle)
        endpoint = (round(center[0] + vector.x), round(center[1] + vector.y))
        pygame.draw.line(ink_mask, white, center, endpoint, 5)
    pygame.draw.circle(ink_mask, white, center, 10)

    # Thick pools produce broad dark shapes and very visible coat highlights.
    for position, radius in [((430, 520), 30), ((512, 496), 18), ((594, 528), 24)]:
        pygame.draw.circle(ink_mask, white, position, radius)

    # A few pressure-varying pen strokes.
    for index in range(5):
        y = 292 + index * 18
        start = (112, y)
        end = (340 + index * 34, y + random.Random(index).randint(-5, 5))
        pygame.draw.aaline(ink_mask, white, start, end)
        pygame.draw.line(ink_mask, white, start, end, 2 + index // 2)

    feature = load_ink_image(FEATURE_IMAGE_PATH, FEATURE_IMAGE_HEIGHT)
    if feature is not None:
        feature_rect = feature.get_rect()
        feature_rect.center = (width - 178, height - 150)
        ink_mask.blit(feature, feature_rect, special_flags=pygame.BLEND_RGBA_MAX)

    if sprites is not None:
        per_page = sprites.columns
        pages = max(1, sprites.count // per_page)
        page = sprite_page % pages
        first_tile = page * per_page

        draw_text(
            f"Hexany 32x32 - tiles {first_tile}-{first_tile + per_page - 1}"
            f" of {sprites.count}",
            small_font,
            (100, 538),
        )

        scale = 2
        spacing = sprites.tile_size * scale + 12
        for column in range(per_page):
            sprites.draw(
                ink_mask,
                first_tile + column,
                (100 + column * spacing, 566),
                scale=scale,
            )

    return ink_mask


def surface_to_texture(
    context: moderngl.Context,
    surface: pygame.Surface,
) -> moderngl.Texture:
    """Upload a Pygame surface with top-left artwork correctly oriented in GL."""
    data = pygame.image.tobytes(surface, "RGBA", True)
    texture = context.texture(surface.get_size(), 4, data)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = False
    texture.repeat_y = False
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
    """

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.base = pygame.Surface(size, pygame.SRCALPHA)
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

    def _brush(self, radius: int) -> pygame.Surface:
        brush = self._brushes.get(radius)
        if brush is None:
            brush = make_brush(radius)
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
        brush = self._brush(radius)
        offset = brush.get_width() * 0.5
        blend = pygame.BLEND_RGBA_SUB if erase else pygame.BLEND_RGBA_MAX

        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        distance = math.hypot(delta_x, delta_y)
        step = max(1.0, radius * 0.35)
        stamps = max(1, int(distance / step) + 1)

        for index in range(stamps + 1):
            travel = index / stamps
            x = start[0] + delta_x * travel - offset
            y = start[1] + delta_y * travel - offset
            self.surface.blit(brush, (round(x), round(y)), special_flags=blend)

        pad = radius + 2
        dirty = pygame.Rect(
            min(start[0], end[0]) - pad,
            min(start[1], end[1]) - pad,
            abs(delta_x) + pad * 2,
            abs(delta_y) + pad * 2,
        )
        return dirty.clip(self.surface.get_rect())

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
        dirty = pygame.Rect(
            min(start[0], end[0]) - pad,
            min(start[1], end[1]) - pad,
            abs(end[0] - start[0]) + pad * 2,
            abs(end[1] - start[1]) + pad * 2,
        ).clip(self.wetness.get_rect())

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


class HelpOverlay(Panel):
    """A card of controls, shown over the page when the user presses `?`."""

    # None starts a new group of related keys.
    ROWS: tuple[tuple[str, str] | None, ...] = (
        ("Drag", "draw wet ink"),
        ("Right-drag", "erase"),
        ("Shift+Wheel", "brush size"),
        None,
        ("Middle-drag", "pan the page"),
        ("Wheel", "zoom about the cursor"),
        ("0 / Home", "reset the view"),
        None,
        ("Move mouse", "move the lamp"),
        ("- / =", "lamp height"),
        ("W", "re-wet the whole page"),
        ("[ / ]", "hold the wetness, pausing the dry-down"),
        None,
        ("U / Ctrl+Z", "undo a stroke"),
        ("C", "clear every stroke"),
        ("Arrows", "sprite sheet page"),
        ("R", "regenerate the parchment"),
        None,
        ("Tab", "sliders for drying and the oil slick"),
        ("1 - 5", "final, normals, roughness, ink, wetness"),
        ("?", "hide this card"),
        ("Esc", "quit"),
    )

    TITLE = "Controls"
    PADDING = 24
    LINE_HEIGHT = 24
    GROUP_GAP = 12

    def __init__(self, context: moderngl.Context, window_size: tuple[int, int]) -> None:
        super().__init__(context, window_size, self._render_panel())

    def _render_panel(self) -> pygame.Surface:
        title_font = pygame.font.SysFont("serif", 22, bold=True)
        key_font = pygame.font.SysFont("monospace", 14, bold=True)
        text_font = pygame.font.SysFont("serif", 17)

        rows = [row for row in self.ROWS if row is not None]
        key_column = max(key_font.size(keys)[0] for keys, _ in rows) + 20
        body_width = max(text_font.size(text)[0] for _, text in rows)
        title = title_font.render(self.TITLE, True, PANEL_TITLE)

        width = self.PADDING * 2 + max(key_column + body_width, title.get_width())
        height = (
            self.PADDING * 2
            + title.get_height()
            + 14
            + len(rows) * self.LINE_HEIGHT
            + sum(self.GROUP_GAP for row in self.ROWS if row is None)
        )

        panel = make_panel_surface((width, height))
        panel.blit(title, (self.PADDING, self.PADDING))
        y = self.PADDING + title.get_height() + 14

        for row in self.ROWS:
            if row is None:
                y += self.GROUP_GAP
                continue
            keys, text = row
            panel.blit(key_font.render(keys, True, PANEL_KEY), (self.PADDING, y + 2))
            panel.blit(
                text_font.render(text, True, PANEL_TEXT),
                (self.PADDING + key_column, y),
            )
            y += self.LINE_HEIGHT

        return panel


@dataclass(frozen=True)
class Slider:
    """One tunable value on the slider panel.

    `logarithmic` is for the controls whose useful range is multiplicative -- the
    drying rate and the slick zoom -- so that the middle of the track is the
    default and each half is an equal factor either side of it.
    """

    field: str
    label: str
    minimum: float
    maximum: float
    style: str = "plain"
    logarithmic: bool = False

    def position(self, value: float) -> float:
        """Where a value sits along the track, 0 to 1."""
        value = max(self.minimum, min(self.maximum, value))
        if self.logarithmic:
            return math.log(value / self.minimum) / math.log(
                self.maximum / self.minimum
            )
        return (value - self.minimum) / (self.maximum - self.minimum)

    def value(self, position: float) -> float:
        """The value at a point along the track, 0 to 1."""
        position = max(0.0, min(1.0, position))
        if self.logarithmic:
            return self.minimum * (self.maximum / self.minimum) ** position
        return self.minimum + (self.maximum - self.minimum) * position

    def text(self, value: float) -> str:
        if self.style == "seconds":
            # Read out the time ink takes to dry rather than the bare multiplier.
            return f"{INK_DRY_SECONDS / value:.1f} s"
        if self.style == "percent":
            return f"{value * 100:.0f}%"
        if self.style == "times":
            return f"{value:.2f}x"
        return f"{value:.2f}"

    def widest_text(self) -> str:
        candidates = (
            self.text(self.minimum),
            self.text(self.maximum),
            self.text(self.value(0.5)),
        )
        return max(candidates, key=len)


class SliderPanel(Panel):
    """Draggable controls for the material parameters, shown on Tab.

    The panel owns its own hit testing: while it is visible it takes any mouse
    event over itself, so tuning a slider never leaves ink on the page and the
    lamp stops following the cursor.
    """

    SLIDERS = (
        Slider("dry_rate", "Ink dries in", 0.25, 4.0, "seconds", logarithmic=True),
        Slider("slick_swirl", "Swirl speed", 0.0, 1.5),
        Slider("slick_zoom", "Slick size", 0.5, 8.0, "times", logarithmic=True),
        Slider("slick_opacity", "Iridescence", 0.0, 1.5, "percent"),
    )

    TITLE = "Wet ink"
    PADDING = 18
    ROW_HEIGHT = 42
    TRACK_HEIGHT = 6
    KNOB_RADIUS = 7
    MARGIN = 20

    def __init__(
        self,
        context: moderngl.Context,
        window_size: tuple[int, int],
        settings: DemoSettings,
    ) -> None:
        self.title_font = pygame.font.SysFont("serif", 19, bold=True)
        self.label_font = pygame.font.SysFont("serif", 16)
        self.value_font = pygame.font.SysFont("monospace", 14, bold=True)

        label_width = max(
            self.label_font.size(slider.label)[0] for slider in self.SLIDERS
        )
        value_width = max(
            self.value_font.size(slider.widest_text())[0] for slider in self.SLIDERS
        )
        title = self.title_font.render(self.TITLE, True, PANEL_TITLE)

        self._label_width = label_width
        self._value_width = value_width
        self._header = title.get_height() + 12
        # A track long enough to be worth dragging, whatever the labels measure.
        self._track_width = 190
        size = (
            self.PADDING * 2 + label_width + 14 + self._track_width + 14 + value_width,
            self.PADDING * 2 + self._header + len(self.SLIDERS) * self.ROW_HEIGHT,
        )

        # Tucked into the top left, clear of the page's title and the lamp.
        super().__init__(context, window_size, make_panel_surface(size),
                         (self.MARGIN, self.MARGIN))
        self.dragging: Slider | None = None
        self.refresh(settings)

    def _track_rect(self, index: int) -> pygame.Rect:
        """Where a slider's track sits, in window pixels."""
        return pygame.Rect(
            self.rect.left + self.PADDING + self._label_width + 14,
            self.rect.top
            + self.PADDING
            + self._header
            + index * self.ROW_HEIGHT
            + self.ROW_HEIGHT // 2
            - self.TRACK_HEIGHT // 2,
            self._track_width,
            self.TRACK_HEIGHT,
        )

    def refresh(self, settings: DemoSettings) -> None:
        """Redraw the panel for the current values and push it to the GPU."""
        panel = make_panel_surface(self.rect.size)
        panel.blit(
            self.title_font.render(self.TITLE, True, PANEL_TITLE),
            (self.PADDING, self.PADDING),
        )

        for index, slider in enumerate(self.SLIDERS):
            value = getattr(settings, slider.field)
            # The track rect is in window pixels; shift it into panel space.
            track = self._track_rect(index).move(-self.rect.left, -self.rect.top)
            row_middle = track.centery

            label = self.label_font.render(slider.label, True, PANEL_TEXT)
            panel.blit(
                label, (self.PADDING, row_middle - label.get_height() // 2)
            )

            filled = round(track.width * slider.position(value))
            pygame.draw.rect(panel, (58, 47, 36, 255), track, border_radius=3)
            if filled:
                pygame.draw.rect(
                    panel,
                    (150, 112, 58, 255),
                    pygame.Rect(track.left, track.top, filled, track.height),
                    border_radius=3,
                )
            pygame.draw.circle(
                panel, PANEL_KEY, (track.left + filled, row_middle), self.KNOB_RADIUS
            )

            reading = self.value_font.render(slider.text(value), True, PANEL_KEY)
            panel.blit(
                reading,
                (
                    self.rect.width - self.PADDING - reading.get_width(),
                    row_middle - reading.get_height() // 2,
                ),
            )

        self.update(panel)

    def _set_from_mouse(self, slider: Slider, index: int, x: int,
                        settings: DemoSettings) -> None:
        track = self._track_rect(index)
        position = (x - track.left) / max(1, track.width)
        setattr(settings, slider.field, slider.value(position))
        self.refresh(settings)

    def handle(self, event: pygame.event.Event, settings: DemoSettings) -> bool:
        """Consume a mouse event aimed at the panel. Returns True if it was ours."""
        if not self.visible:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if not self.rect.collidepoint(event.pos):
                return False
            if event.button == 1:
                for index, slider in enumerate(self.SLIDERS):
                    # A generous grab area: the whole row, not just the track.
                    row = self._track_rect(index).inflate(
                        self.KNOB_RADIUS * 2, self.ROW_HEIGHT
                    )
                    if row.collidepoint(event.pos):
                        self.dragging = slider
                        self._set_from_mouse(slider, index, event.pos[0], settings)
                        break
            return True

        if event.type == pygame.MOUSEMOTION:
            if self.dragging is not None:
                index = self.SLIDERS.index(self.dragging)
                self._set_from_mouse(self.dragging, index, event.pos[0], settings)
                return True
            return self.rect.collidepoint(event.pos)

        if event.type == pygame.MOUSEBUTTONUP:
            was_dragging = self.dragging is not None
            self.dragging = None
            return was_dragging or self.rect.collidepoint(event.pos)

        return False


class ParchmentInkRenderer:
    def __init__(self, size: tuple[int, int], seed: int = 7) -> None:
        self.size = size
        self.seed = seed
        self.context = moderngl.create_context(require=330)
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

        self.sprites = load_sprite_sheet()
        self.sprite_page = 0
        self.canvas = InkCanvas(size)

        self.paper_albedo: moderngl.Texture
        self.paper_normal: moderngl.Texture
        self.paper_roughness: moderngl.Texture
        self.ink_mask = self.context.texture(size, 4)
        self.ink_mask.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.ink_mask.repeat_x = False
        self.ink_mask.repeat_y = False

        # One byte per pixel: how fresh the ink there is.
        self.ink_wet = self.context.texture(size, 1)
        self.ink_wet.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.ink_wet.repeat_x = False
        self.ink_wet.repeat_y = False

        self.help = HelpOverlay(self.context, size)
        self.sliders = SliderPanel(self.context, size, DemoSettings())

        slick, slick_gain = load_oil_slick()
        # The shader folds the coordinate to tile it, so clamping is what we want.
        self.oil_slick = surface_to_texture(self.context, slick)
        self.slick_gain = slick_gain

        self._rebuild_base_ink()
        self.upload_wetness()
        self._replace_paper_textures(seed)

        self.program["u_paper_albedo"].value = 0
        self.program["u_paper_normal"].value = 1
        self.program["u_paper_roughness"].value = 2
        self.program["u_ink_mask"].value = 3
        self.program["u_ink_wet"].value = 4
        self.program["u_oil_slick"].value = 5
        self.program["u_slick_gain"].value = self.slick_gain
        self.program["u_resolution"].value = tuple(float(value) for value in size)

    def _rebuild_base_ink(self) -> None:
        """Regenerate the demo artwork underneath the user's strokes."""
        self.canvas.set_base(make_demo_ink(self.size, self.sprites, self.sprite_page))
        self.upload_ink()

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
                self.size[1] - region.bottom,
                region.width,
                region.height,
            ),
        )

    def upload_ink(self, region: pygame.Rect | None = None) -> None:
        """Push the ink coverage to the GPU, only the changed rectangle."""
        self._write_region(self.ink_mask, self.canvas.surface, region)

    def upload_wetness(self, region: pygame.Rect | None = None) -> None:
        """Push the stroke wetness field to its single-channel texture."""
        self._write_region(self.ink_wet, self.canvas.wetness, region, red_only=True)

    def upload_stroke(self, region: pygame.Rect | None) -> None:
        """Push both halves of freshly drawn ink: what it covers and how wet."""
        if region is None:
            return
        self.upload_ink(region)
        self.upload_wetness(region)

    def change_sprite_page(self, delta: int) -> None:
        if self.sprites is None:
            return
        pages = max(1, self.sprites.count // self.sprites.columns)
        self.sprite_page = (self.sprite_page + delta) % pages
        self._rebuild_base_ink()

    def _replace_paper_textures(self, seed: int) -> None:
        maps = generate_paper_maps(self.size, seed)
        new_textures = (
            surface_to_texture(self.context, maps.albedo),
            surface_to_texture(self.context, maps.normal),
            surface_to_texture(self.context, maps.roughness),
        )

        for name in ("paper_albedo", "paper_normal", "paper_roughness"):
            old_texture = getattr(self, name, None)
            if old_texture is not None:
                old_texture.release()

        self.paper_albedo, self.paper_normal, self.paper_roughness = new_textures

    def regenerate_paper(self) -> None:
        self.seed += 1
        self._replace_paper_textures(self.seed)

    def render(
        self,
        light_position: tuple[int, int],
        settings: DemoSettings,
        view: View | None = None,
    ) -> None:
        width, height = self.size
        light_uv = (
            max(0.0, min(1.0, light_position[0] / width)),
            max(0.0, min(1.0, 1.0 - light_position[1] / height)),
        )

        self.context.screen.use()
        self.context.viewport = (0, 0, width, height)
        self.context.clear(0.0, 0.0, 0.0, 1.0)

        self.paper_albedo.use(location=0)
        self.paper_normal.use(location=1)
        self.paper_roughness.use(location=2)
        self.ink_mask.use(location=3)
        self.ink_wet.use(location=4)
        self.oil_slick.use(location=5)

        self.program["u_light_uv"].value = light_uv
        self.program["u_light_height"].value = settings.light_height
        self.program["u_page_wetness"].value = settings.wetness
        self.program["u_time"].value = settings.moment
        self.program["u_slick_opacity"].value = settings.slick_opacity
        self.program["u_slick_swirl"].value = settings.slick_swirl
        self.program["u_slick_zoom"].value = max(settings.slick_zoom, 1e-3)
        self.program["u_exposure"].value = settings.exposure
        self.program["u_debug_mode"].value = settings.debug_mode
        self.program["u_zoom"].value = 1.0 if view is None else view.zoom
        self.program["u_center_uv"].value = (
            (0.5, 0.5) if view is None else view.center_uv
        )

        self.vertex_array.render(mode=moderngl.TRIANGLE_STRIP)

        if self.sliders.visible:
            self.sliders.draw()
        if self.help.visible:
            self.help.draw()

    def release(self) -> None:
        self.help.release()
        self.sliders.release()
        self.paper_albedo.release()
        self.paper_normal.release()
        self.paper_roughness.release()
        self.ink_mask.release()
        self.ink_wet.release()
        self.oil_slick.release()
        self.vertex_array.release()
        self.vertex_buffer.release()
        self.program.release()


def update_caption(settings: DemoSettings, view: View) -> None:
    modes = ("final", "normals", "roughness", "ink mask", "wetness")
    pygame.display.set_caption(
        "ModernGL Parchment + Fresh Ink  |  "
        f"wetness {settings.wetness:.2f} "
        f"({'drying' if settings.drying else 'held'})  "
        f"light height {settings.light_height:.2f}  "
        f"brush {settings.brush_radius}px  "
        f"zoom {view.zoom * 100:.0f}%  "
        f"view {modes[settings.debug_mode]}  |  press ? for controls"
    )


def main() -> None:
    pygame.init()
    pygame.font.init()

    # Ask Pygame for a modern core OpenGL context before creating the window.
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK,
        pygame.GL_CONTEXT_PROFILE_CORE,
    )
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 0)

    try:
        pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF)
        renderer = ParchmentInkRenderer(WINDOW_SIZE)
    except Exception as exc:
        pygame.quit()
        raise SystemExit(
            "Could not create the OpenGL 3.3 / ModernGL renderer. "
            "Update the graphics driver or relax the requested context version.\n"
            f"Original error: {exc}"
        ) from exc

    clock = pygame.time.Clock()
    settings = DemoSettings()
    view = View(WINDOW_SIZE)
    update_caption(settings, view)

    light_position = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)
    drawing = False
    panning = False
    elapsed = 0.0
    shown_wetness = settings.wetness

    running = True
    while running:
        caption_changed = False
        for event in pygame.event.get():
            # The slider panel gets first refusal on the mouse, so dragging a
            # control neither draws ink nor drags the lamp along with it.
            if renderer.sliders.handle(event, settings):
                continue

            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                panning = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                panning = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                drawing = True
                dirty = renderer.canvas.begin_stroke(
                    view.screen_to_canvas(event.pos),
                    settings.brush_radius,
                    erase=event.button == 3,
                )
                renderer.upload_stroke(dirty)
            elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 3):
                drawing = False
                renderer.canvas.end_stroke()
            elif event.type == pygame.MOUSEMOTION:
                if panning:
                    view.pan_by(event.rel)
                elif drawing:
                    renderer.upload_stroke(
                        renderer.canvas.extend_stroke(view.screen_to_canvas(event.pos))
                    )
                else:
                    # The light stays put while drawing so a fresh stroke can be
                    # judged under steady lighting instead of a moving highlight.
                    light_position = event.pos
            elif event.type == pygame.MOUSEWHEEL:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    settings.brush_radius = max(
                        1, min(48, settings.brush_radius + event.y)
                    )
                else:
                    view.zoom_by(event.y, pygame.mouse.get_pos())
                caption_changed = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif getattr(event, "unicode", "") == "?" or event.key in (
                    pygame.K_QUESTION,
                    pygame.K_SLASH,
                    pygame.K_F1,
                ):
                    # The key that produces `?` moves around between layouts, so
                    # trust the character the event carries where there is one.
                    renderer.help.toggle()
                elif event.key == pygame.K_TAB:
                    renderer.sliders.toggle()
                    renderer.sliders.refresh(settings)
                elif event.key == pygame.K_r:
                    renderer.regenerate_paper()
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA):
                    # Taking manual control pauses the dry-down, otherwise the
                    # decay would pull the value straight back down again.
                    settings.drying = False
                    settings.wetness = max(0.0, settings.wetness - 0.06)
                    caption_changed = True
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD):
                    settings.drying = False
                    settings.wetness = min(1.0, settings.wetness + 0.06)
                    caption_changed = True
                elif event.key == pygame.K_w:
                    settings.rewet()
                    caption_changed = True
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    settings.light_height = max(0.06, settings.light_height - 0.025)
                    caption_changed = True
                elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    settings.light_height = min(0.70, settings.light_height + 0.025)
                    caption_changed = True
                elif event.key in (pygame.K_1, pygame.K_KP1):
                    settings.debug_mode = 0
                    caption_changed = True
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    settings.debug_mode = 1
                    caption_changed = True
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    settings.debug_mode = 2
                    caption_changed = True
                elif event.key in (pygame.K_4, pygame.K_KP4):
                    settings.debug_mode = 3
                    caption_changed = True
                elif event.key in (pygame.K_5, pygame.K_KP5):
                    settings.debug_mode = 4
                    caption_changed = True
                elif event.key == pygame.K_LEFT:
                    renderer.change_sprite_page(-1)
                elif event.key == pygame.K_RIGHT:
                    renderer.change_sprite_page(1)
                elif event.key == pygame.K_u or (
                    event.key == pygame.K_z and event.mod & pygame.KMOD_CTRL
                ):
                    if renderer.canvas.undo():
                        renderer.upload_ink()
                elif event.key == pygame.K_c:
                    if renderer.canvas.clear_strokes():
                        renderer.upload_ink()
                elif event.key in (pygame.K_0, pygame.K_KP0, pygame.K_HOME):
                    view.reset()
                    caption_changed = True

        # The page-wide ink and each individual stroke dry on the same clock.
        if settings.advance(elapsed) and abs(settings.wetness - shown_wetness) >= 0.01:
            caption_changed = True

        drying_region = renderer.canvas.dry(elapsed, settings.dry_rate)
        if drying_region is not None:
            renderer.upload_wetness(drying_region)

        if caption_changed:
            shown_wetness = settings.wetness
            update_caption(settings, view)

        renderer.render(light_position, settings, view)
        pygame.display.flip()
        elapsed = clock.tick(60) / 1000.0

    renderer.release()
    pygame.quit()


if __name__ == "__main__":
    main()