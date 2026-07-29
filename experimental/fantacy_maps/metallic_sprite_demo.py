"""PBR sprite demo using Pygame and ModernGL.

This version loads four independent material textures:
    gear_albedo.png
    gear_normal.png
    gear_metallic.png
    gear_roughness.png

Install:
    python -m pip install pygame moderngl

Run:
    python metallic_sprite_demo.py

Controls:
    Mouse              Move the point light
    W / A / S / D      Move the sprite
    Q / E              Rotate the sprite
    Up / Down          Change roughness-map strength
    Left / Right       Change metallic-map strength
    1                  Final shaded material
    2                  Albedo map
    3                  Normal map
    4                  Metallic map
    5                  Roughness map
    R                  Reset
    Escape             Quit
"""

from __future__ import annotations

import struct
from pathlib import Path

import moderngl
import pygame

WINDOW_SIZE = (1000, 700)
BACKGROUND = (0.012, 0.017, 0.030, 1.0)

MAP_FILES = {
    "albedo": "gear_albedo.png",
    "normal": "gear_normal.png",
    "metallic": "gear_metallic.png",
    "roughness": "gear_roughness.png",
}

DEBUG_NAMES = {
    0: "final PBR",
    1: "albedo",
    2: "normal",
    3: "metallic",
    4: "roughness",
}


def split_shader_file(path: Path) -> tuple[str, str]:
    """Read the vertex and fragment stages stored in one GLSL file."""
    source = path.read_text(encoding="utf-8")
    vertex_marker = "// === VERTEX SHADER ==="
    fragment_marker = "// === FRAGMENT SHADER ==="

    if vertex_marker not in source or fragment_marker not in source:
        raise ValueError(
            f"{path.name} must contain {vertex_marker!r} and {fragment_marker!r}."
        )

    vertex_source = source.split(vertex_marker, 1)[1].split(fragment_marker, 1)[0]
    fragment_source = source.split(fragment_marker, 1)[1]
    return vertex_source.strip(), fragment_source.strip()


def load_rgba_texture(ctx: moderngl.Context, path: Path) -> moderngl.Texture:
    """Load any source image as RGBA so all material maps share one loader."""
    surface = pygame.image.load(path).convert_alpha()
    pixels = pygame.image.tobytes(surface, "RGBA", True)

    texture = ctx.texture(surface.get_size(), components=4, data=pixels)
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    texture.repeat_x = False
    texture.repeat_y = False
    texture.build_mipmaps()
    return texture


def create_sprite_vao(
    ctx: moderngl.Context,
    program: moderngl.Program,
) -> tuple[moderngl.Buffer, moderngl.VertexArray]:
    """Create a centered unit quad with UV coordinates."""
    # Local positions use positive Y downward. The image upload is flipped,
    # so the top vertices sample v=1 and the bottom vertices sample v=0.
    vertices = (
        -0.5, -0.5, 0.0, 1.0,
         0.5, -0.5, 1.0, 1.0,
        -0.5,  0.5, 0.0, 0.0,
         0.5,  0.5, 1.0, 0.0,
    )
    buffer = ctx.buffer(struct.pack(f"{len(vertices)}f", *vertices))
    vao = ctx.vertex_array(
        program,
        [(buffer, "2f 2f", "in_position", "in_uv")],
    )
    return buffer, vao


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def main() -> None:
    asset_dir = Path(__file__).resolve().parent
    shader_path = asset_dir / "metallic_sprite.glsl"
    map_paths = {name: asset_dir / filename for name, filename in MAP_FILES.items()}

    required_files = [shader_path, *map_paths.values()]
    missing = [path.name for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK,
        pygame.GL_CONTEXT_PROFILE_CORE,
    )
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF, vsync=1)
    pygame.display.set_caption("ModernGL PBR Sprite")

    ctx = moderngl.create_context(require=330)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    vertex_source, fragment_source = split_shader_file(shader_path)
    program = ctx.program(
        vertex_shader=vertex_source,
        fragment_shader=fragment_source,
    )

    vertex_buffer, vao = create_sprite_vao(ctx, program)
    textures = {
        name: load_rgba_texture(ctx, path)
        for name, path in map_paths.items()
    }

    texture_bindings = {
        "albedo": (0, "u_albedo_map"),
        "normal": (1, "u_normal_map"),
        "metallic": (2, "u_metallic_map"),
        "roughness": (3, "u_roughness_map"),
    }
    for name, (unit, uniform_name) in texture_bindings.items():
        textures[name].use(location=unit)
        program[uniform_name].value = unit

    program["u_alpha_cutoff"].value = 0.015
    program["u_normal_strength"].value = 1.0

    position = pygame.Vector2(WINDOW_SIZE[0] * 0.5, WINDOW_SIZE[1] * 0.51)
    sprite_size = pygame.Vector2(460.0, 460.0)
    rotation = 0.0
    metallic_scale = 1.0
    roughness_scale = 1.0
    debug_view = 0

    clock = pygame.time.Clock()
    running = True
    elapsed = 0.0

    try:
        while running:
            delta_time = min(clock.tick(144) / 1000.0, 0.05)
            elapsed += delta_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif pygame.K_1 <= event.key <= pygame.K_5:
                        debug_view = event.key - pygame.K_1
                    elif event.key == pygame.K_r:
                        position.update(WINDOW_SIZE[0] * 0.5, WINDOW_SIZE[1] * 0.51)
                        rotation = 0.0
                        metallic_scale = 1.0
                        roughness_scale = 1.0
                        debug_view = 0

            keys = pygame.key.get_pressed()

            movement = pygame.Vector2(
                float(keys[pygame.K_d]) - float(keys[pygame.K_a]),
                float(keys[pygame.K_s]) - float(keys[pygame.K_w]),
            )
            if movement.length_squared() > 0.0:
                position += movement.normalize() * 290.0 * delta_time

            rotation += (
                float(keys[pygame.K_e]) - float(keys[pygame.K_q])
            ) * 1.8 * delta_time

            roughness_scale += (
                float(keys[pygame.K_DOWN]) - float(keys[pygame.K_UP])
            ) * 0.65 * delta_time
            roughness_scale = clamp(roughness_scale, 0.05, 2.0)

            metallic_scale += (
                float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
            ) * 0.65 * delta_time
            metallic_scale = clamp(metallic_scale, 0.0, 1.0)

            half_size = sprite_size * 0.35
            position.x = clamp(position.x, half_size.x, WINDOW_SIZE[0] - half_size.x)
            position.y = clamp(position.y, half_size.y, WINDOW_SIZE[1] - half_size.y)

            mouse_x, mouse_y = pygame.mouse.get_pos()

            ctx.clear(*BACKGROUND)
            program["u_resolution"].value = tuple(map(float, WINDOW_SIZE))
            program["u_position"].value = (position.x, position.y)
            program["u_size"].value = (sprite_size.x, sprite_size.y)
            program["u_rotation"].value = rotation
            program["u_light_position"].value = (float(mouse_x), float(mouse_y))
            program["u_time"].value = elapsed
            program["u_metallic_scale"].value = metallic_scale
            program["u_roughness_scale"].value = roughness_scale
            program["u_debug_view"].value = debug_view

            for name, (unit, _) in texture_bindings.items():
                textures[name].use(location=unit)

            vao.render(mode=moderngl.TRIANGLE_STRIP)
            pygame.display.flip()

            pygame.display.set_caption(
                "ModernGL PBR Sprite | "
                f"view: {DEBUG_NAMES[debug_view]} | "
                f"metallic x{metallic_scale:.2f} | roughness x{roughness_scale:.2f} | "
                "1-5=maps"
            )
    finally:
        vao.release()
        vertex_buffer.release()
        for texture in textures.values():
            texture.release()
        program.release()
        ctx.release()
        pygame.quit()


if __name__ == "__main__":
    main()