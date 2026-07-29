"""PBR material grid rendered with ModernGL + pygame.

Controls
--------
Left mouse drag : orbit camera
Mouse wheel     : zoom
Space           : toggle automatic camera orbit
R               : reset camera
Escape          : quit

Grid layout
-----------
Metallic increases from left to right.
Roughness increases from bottom to top.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import moderngl
import numpy as np
import pygame


VERTEX_SHADER = r"""
#version 330 core

in vec3 in_position;
in vec3 in_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_world_position;
out vec3 v_world_normal;

void main() {
    vec4 world_position = u_model * vec4(in_position, 1.0);
    v_world_position = world_position.xyz;
    v_world_normal = normalize(mat3(transpose(inverse(u_model))) * in_normal);
    gl_Position = u_projection * u_view * world_position;
}
"""


FRAGMENT_SHADER = r"""
#version 330 core

const float PI = 3.14159265359;
const int LIGHT_COUNT = 4;

in vec3 v_world_position;
in vec3 v_world_normal;

uniform vec3 u_camera_position;
uniform vec3 u_base_color;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_ao;
uniform float u_exposure;
uniform vec3 u_light_positions[LIGHT_COUNT];
uniform vec3 u_light_colors[LIGHT_COUNT];

out vec4 frag_color;

float distribution_ggx(vec3 n, vec3 h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float n_dot_h = max(dot(n, h), 0.0);
    float n_dot_h2 = n_dot_h * n_dot_h;

    float denominator = n_dot_h2 * (a2 - 1.0) + 1.0;
    denominator = PI * denominator * denominator;
    return a2 / max(denominator, 0.000001);
}

float geometry_schlick_ggx(float n_dot_v, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.000001);
}

float geometry_smith(vec3 n, vec3 v, vec3 l, float roughness) {
    float n_dot_v = max(dot(n, v), 0.0);
    float n_dot_l = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(n_dot_v, roughness)
         * geometry_schlick_ggx(n_dot_l, roughness);
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

void main() {
    vec3 n = normalize(v_world_normal);
    vec3 v = normalize(u_camera_position - v_world_position);

    // Dielectrics use about 4% reflectance; metals tint F0 with base color.
    vec3 f0 = mix(vec3(0.04), u_base_color, u_metallic);
    vec3 direct_lighting = vec3(0.0);

    for (int i = 0; i < LIGHT_COUNT; ++i) {
        vec3 light_vector = u_light_positions[i] - v_world_position;
        float distance_squared = max(dot(light_vector, light_vector), 0.01);
        vec3 l = normalize(light_vector);
        vec3 h = normalize(v + l);
        vec3 radiance = u_light_colors[i] / distance_squared;

        float ndf = distribution_ggx(n, h, u_roughness);
        float geometry = geometry_smith(n, v, l, u_roughness);
        vec3 fresnel = fresnel_schlick(max(dot(h, v), 0.0), f0);

        vec3 numerator = ndf * geometry * fresnel;
        float denominator = 4.0
            * max(dot(n, v), 0.0)
            * max(dot(n, l), 0.0);
        vec3 specular = numerator / max(denominator, 0.0001);

        vec3 k_specular = fresnel;
        vec3 k_diffuse = (vec3(1.0) - k_specular) * (1.0 - u_metallic);
        float n_dot_l = max(dot(n, l), 0.0);

        direct_lighting += (k_diffuse * u_base_color / PI + specular)
                         * radiance * n_dot_l;
    }

    // Small constant ambient term. Replace this with IBL for production use.
    vec3 ambient = vec3(0.025) * u_base_color * u_ao;
    vec3 hdr_color = ambient + direct_lighting;

    // Exponential tone mapping followed by linear-to-sRGB conversion.
    vec3 mapped = vec3(1.0) - exp(-hdr_color * u_exposure);
    mapped = pow(mapped, vec3(1.0 / 2.2));
    frag_color = vec4(mapped, 1.0);
}
"""


@dataclass
class OrbitCamera:
    yaw: float = math.radians(22.0)
    pitch: float = math.radians(-12.0)
    radius: float = 16.0
    target: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.target is None:
            self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    @property
    def position(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        offset = np.array(
            [
                self.radius * cp * math.sin(self.yaw),
                self.radius * math.sin(self.pitch),
                self.radius * cp * math.cos(self.yaw),
            ],
            dtype=np.float32,
        )
        return self.target + offset

    def reset(self) -> None:
        self.yaw = math.radians(22.0)
        self.pitch = math.radians(-12.0)
        self.radius = 16.0


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        return vector
    return vector / length


def perspective(fov_y_radians: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y_radians * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalize(target - eye)
    side = normalize(np.cross(forward, up))
    corrected_up = np.cross(side, forward)

    matrix = np.identity(4, dtype=np.float32)
    matrix[0, 0:3] = side
    matrix[1, 0:3] = corrected_up
    matrix[2, 0:3] = -forward
    matrix[0, 3] = -float(np.dot(side, eye))
    matrix[1, 3] = -float(np.dot(corrected_up, eye))
    matrix[2, 3] = float(np.dot(forward, eye))
    return matrix


def translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.identity(4, dtype=np.float32)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix


def create_uv_sphere(
    radius: float = 0.72,
    latitude_segments: int = 48,
    longitude_segments: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interleaved position/normal vertices and uint32 triangle indices."""
    vertices: list[float] = []
    indices: list[int] = []

    for latitude in range(latitude_segments + 1):
        v = latitude / latitude_segments
        theta = v * math.pi
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        for longitude in range(longitude_segments + 1):
            u = longitude / longitude_segments
            phi = u * math.tau
            normal = (
                sin_theta * math.cos(phi),
                cos_theta,
                sin_theta * math.sin(phi),
            )
            position = tuple(component * radius for component in normal)
            vertices.extend((*position, *normal))

    row_length = longitude_segments + 1
    for latitude in range(latitude_segments):
        for longitude in range(longitude_segments):
            top_left = latitude * row_length + longitude
            bottom_left = (latitude + 1) * row_length + longitude
            top_right = top_left + 1
            bottom_right = bottom_left + 1

            indices.extend(
                [
                    top_left,
                    top_right,
                    bottom_left,
                    top_right,
                    bottom_right,
                    bottom_left,
                ]
            )

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def write_matrix(uniform: moderngl.Uniform, matrix: np.ndarray) -> None:
    # NumPy stores rows contiguously; GLSL matrices are uploaded column-major.
    uniform.write(matrix.T.astype("f4", copy=False).tobytes())


def set_vec3_array(program: moderngl.Program, name: str, values: np.ndarray) -> None:
    program[name].value = [
        tuple(float(component) for component in value) for value in values
    ]


def main() -> int:
    pygame.init()
    pygame.display.set_caption("ModernGL PBR Material Grid")

    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

    window_size = (1280, 800)
    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE

    try:
        pygame.display.set_mode(window_size, flags)
        context = moderngl.create_context(require=330)
    except Exception as exc:
        pygame.quit()
        print(f"Could not create an OpenGL 3.3 context: {exc}", file=sys.stderr)
        return 1

    context.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    context.gc_mode = "auto"

    program = context.program(
        vertex_shader=VERTEX_SHADER,
        fragment_shader=FRAGMENT_SHADER,
    )

    vertex_data, index_data = create_uv_sphere()
    vertex_buffer = context.buffer(vertex_data.tobytes())
    index_buffer = context.buffer(index_data.tobytes())
    vertex_array = context.vertex_array(
        program,
        [(vertex_buffer, "3f 3f", "in_position", "in_normal")],
        index_buffer=index_buffer,
        index_element_size=4,
    )

    # Bright values are intentional: the shader treats them as HDR intensity.
    light_positions = np.array(
        [
            [-7.0, 7.0, 8.0],
            [7.0, 7.0, 8.0],
            [-7.0, -5.0, 7.0],
            [7.0, -5.0, 7.0],
        ],
        dtype=np.float32,
    )
    light_colors = np.array(
        [
            [320.0, 270.0, 230.0],
            [230.0, 275.0, 330.0],
            [210.0, 190.0, 160.0],
            [165.0, 185.0, 220.0],
        ],
        dtype=np.float32,
    )
    set_vec3_array(program, "u_light_positions", light_positions)
    set_vec3_array(program, "u_light_colors", light_colors)

    program["u_base_color"] = (0.95, 0.37, 0.12)
    program["u_ao"] = 1.0
    program["u_exposure"] = 1.15

    camera = OrbitCamera()
    auto_orbit = False
    dragging = False
    last_mouse_position = (0, 0)
    clock = pygame.time.Clock()
    running = True

    grid_size = 5
    spacing = 1.75
    grid_extent = spacing * (grid_size - 1)
    y_center_offset = 0.0

    print(__doc__)

    while running:
        delta_time = min(clock.tick(144) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    auto_orbit = not auto_orbit
                elif event.key == pygame.K_r:
                    camera.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                last_mouse_position = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse_position[0]
                dy = event.pos[1] - last_mouse_position[1]
                last_mouse_position = event.pos
                camera.yaw -= dx * 0.006
                camera.pitch = float(
                    np.clip(camera.pitch - dy * 0.006, math.radians(-80), math.radians(80))
                )
            elif event.type == pygame.MOUSEWHEEL:
                camera.radius = float(np.clip(camera.radius * (0.90 ** event.y), 8.0, 30.0))

        if auto_orbit and not dragging:
            camera.yaw += delta_time * 0.22

        width, height = pygame.display.get_window_size()
        width = max(width, 1)
        height = max(height, 1)
        context.viewport = (0, 0, width, height)
        context.clear(0.012, 0.016, 0.024, 1.0, depth=1.0)

        eye = camera.position
        view_matrix = look_at(
            eye,
            camera.target,
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        projection_matrix = perspective(
            math.radians(45.0),
            width / height,
            0.1,
            100.0,
        )
        write_matrix(program["u_view"], view_matrix)
        write_matrix(program["u_projection"], projection_matrix)
        program["u_camera_position"] = tuple(float(component) for component in eye)

        for row in range(grid_size):
            roughness = max(row / (grid_size - 1), 0.045)
            y = row * spacing - grid_extent * 0.5 + y_center_offset

            for column in range(grid_size):
                metallic = column / (grid_size - 1)
                x = column * spacing - grid_extent * 0.5
                model_matrix = translation(x, y, 0.0)

                write_matrix(program["u_model"], model_matrix)
                program["u_metallic"] = metallic
                program["u_roughness"] = roughness
                vertex_array.render(mode=moderngl.TRIANGLES)

        pygame.display.flip()

    vertex_array.release()
    index_buffer.release()
    vertex_buffer.release()
    program.release()
    context.release()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())