from __future__ import annotations

import math
import random
import sys
from array import array
from dataclasses import dataclass

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

uniform vec2 u_resolution;
uniform vec2 u_light_uv;
uniform float u_light_height;
uniform float u_wetness;
uniform float u_exposure;
uniform int u_debug_mode;

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

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_ink_mask, 0));

    vec3 paper_albedo = to_linear(texture(u_paper_albedo, v_uv).rgb);
    vec3 paper_normal = normalize(texture(u_paper_normal, v_uv).rgb * 2.0 - 1.0);
    float paper_roughness = texture(u_paper_roughness, v_uv).r;

    float ink = texture(u_ink_mask, v_uv).a;
    float ink_left = texture(u_ink_mask, v_uv - vec2(texel.x, 0.0)).a;
    float ink_right = texture(u_ink_mask, v_uv + vec2(texel.x, 0.0)).a;
    float ink_down = texture(u_ink_mask, v_uv - vec2(0.0, texel.y)).a;
    float ink_up = texture(u_ink_mask, v_uv + vec2(0.0, texel.y)).a;

    // The alpha gradient makes wet strokes look microscopically raised.
    vec2 ink_gradient = vec2(ink_right - ink_left, ink_up - ink_down);
    float edge = saturate(length(ink_gradient) * 3.2);

    vec3 ink_normal = normalize(vec3(
        paper_normal.xy * 0.22 - ink_gradient * (1.4 + 1.2 * u_wetness),
        max(0.30, paper_normal.z)
    ));
    vec3 normal = normalize(mix(paper_normal, ink_normal, ink));

    vec3 dry_ink = to_linear(vec3(0.085, 0.047, 0.026));
    vec3 wet_ink = to_linear(vec3(0.030, 0.016, 0.010));
    vec3 ink_albedo = mix(dry_ink, wet_ink, u_wetness);
    vec3 albedo = mix(paper_albedo, ink_albedo, ink);

    float ink_roughness = mix(0.62, 0.10, u_wetness);
    float roughness = mix(paper_roughness, ink_roughness, ink);

    float aspect = u_resolution.x / u_resolution.y;
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
    vec3 ink_f0 = mix(vec3(0.035), vec3(0.115), u_wetness);
    vec3 f0 = mix(paper_f0, ink_f0, ink);
    vec3 fresnel = fresnel_schlick(n_dot_v, f0);

    float paper_specular_strength = mix(0.055, 0.025, paper_roughness);
    float ink_specular_strength = mix(0.16, 1.05, u_wetness);
    float specular_strength = mix(paper_specular_strength, ink_specular_strength, ink);

    vec3 warm_light = to_linear(vec3(1.00, 0.84, 0.61));
    vec3 ambient_light = to_linear(vec3(0.34, 0.29, 0.22));

    vec3 diffuse = albedo * (ambient_light + warm_light * n_dot_l * attenuation * 0.68);
    vec3 specular = warm_light * fresnel * specular_lobe * specular_strength * attenuation;

    // A small extra meniscus glint along the stroke boundary sells fresh ink.
    float edge_glint = edge * ink * u_wetness * pow(n_dot_h, 34.0) * attenuation;
    specular += warm_light * edge_glint * 0.16;

    // Darken the sheet perimeter without baking it into the generated texture.
    vec2 border_distance = min(v_uv, 1.0 - v_uv);
    float border = smoothstep(0.0, 0.095, min(border_distance.x, border_distance.y));
    diffuse *= mix(0.72, 1.0, border);

    vec3 color = (diffuse + specular) * u_exposure;

    if (u_debug_mode == 1) {
        color = normal * 0.5 + 0.5;
    } else if (u_debug_mode == 2) {
        color = vec3(roughness);
    } else if (u_debug_mode == 3) {
        color = vec3(ink);
    }

    // Draw a tiny unobtrusive ring at the movable light position.
    vec2 marker_delta = (v_uv - u_light_uv) * vec2(aspect, 1.0);
    float marker_distance = length(marker_delta);
    float marker = 1.0 - smoothstep(0.010, 0.012, abs(marker_distance - 0.013));
    if (u_debug_mode == 0) {
        color += warm_light * marker * 0.22;
    }

    frag_color = vec4(to_srgb(color), 1.0);
}
"""


@dataclass
class DemoSettings:
    wetness: float = 0.82
    light_height: float = 0.24
    exposure: float = 1.00
    debug_mode: int = 0


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


def make_demo_ink(size: tuple[int, int]) -> pygame.Surface:
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

    center = (width - 190, height - 155)
    pygame.draw.circle(ink_mask, white, center, 72, 5)
    for angle in range(0, 360, 45):
        vector = pygame.Vector2(0, -62).rotate(angle)
        endpoint = (round(center[0] + vector.x), round(center[1] + vector.y))
        pygame.draw.line(ink_mask, white, center, endpoint, 5)
    pygame.draw.circle(ink_mask, white, center, 12)

    # Thick pools produce broad dark shapes and very visible coat highlights.
    for position, radius in [((165, 555), 30), ((245, 520), 18), ((325, 565), 24)]:
        pygame.draw.circle(ink_mask, white, position, radius)

    # A few pressure-varying pen strokes.
    for index in range(5):
        y = 292 + index * 18
        start = (112, y)
        end = (340 + index * 34, y + random.Random(index).randint(-5, 5))
        pygame.draw.aaline(ink_mask, white, start, end)
        pygame.draw.line(ink_mask, white, start, end, 2 + index // 2)

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

        self.paper_albedo: moderngl.Texture
        self.paper_normal: moderngl.Texture
        self.paper_roughness: moderngl.Texture
        self.ink_mask = surface_to_texture(self.context, make_demo_ink(size))
        self._replace_paper_textures(seed)

        self.program["u_paper_albedo"].value = 0
        self.program["u_paper_normal"].value = 1
        self.program["u_paper_roughness"].value = 2
        self.program["u_ink_mask"].value = 3
        self.program["u_resolution"].value = tuple(float(value) for value in size)

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

        self.program["u_light_uv"].value = light_uv
        self.program["u_light_height"].value = settings.light_height
        self.program["u_wetness"].value = settings.wetness
        self.program["u_exposure"].value = settings.exposure
        self.program["u_debug_mode"].value = settings.debug_mode

        self.vertex_array.render(mode=moderngl.TRIANGLE_STRIP)

    def release(self) -> None:
        self.paper_albedo.release()
        self.paper_normal.release()
        self.paper_roughness.release()
        self.ink_mask.release()
        self.vertex_array.release()
        self.vertex_buffer.release()
        self.program.release()


def update_caption(settings: DemoSettings) -> None:
    modes = ("final", "normals", "roughness", "ink mask")
    pygame.display.set_caption(
        "ModernGL Parchment + Fresh Ink  |  "
        f"wetness {settings.wetness:.2f}  "
        f"light height {settings.light_height:.2f}  "
        f"view {modes[settings.debug_mode]}  |  "
        "Mouse light, [ ] wetness, - = height, 1-4 views, R regenerate, Esc quit"
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
    update_caption(settings)

    running = True
    while running:
        caption_changed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    renderer.regenerate_paper()
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA):
                    settings.wetness = max(0.0, settings.wetness - 0.06)
                    caption_changed = True
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD):
                    settings.wetness = min(1.0, settings.wetness + 0.06)
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

        if caption_changed:
            update_caption(settings)

        renderer.render(pygame.mouse.get_pos(), settings)
        pygame.display.flip()
        clock.tick(60)

    renderer.release()
    pygame.quit()


if __name__ == "__main__":
    main()