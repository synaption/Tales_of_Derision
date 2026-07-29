"""Metallic sprite demo using Pygame + ModernGL.

Install:
    python -m pip install pygame moderngl

Run this file from the same directory as:
    metallic_sprite.glsl
    metal_sprite.png

Controls:
    Mouse          Move the point light
    W/A/S/D        Move the sprite
    Q / E          Rotate the sprite
    Up / Down      Decrease / increase roughness
    M              Toggle metallic / non-metallic
    R              Reset the material and transform
    Escape         Quit
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import moderngl
import pygame

WINDOW_SIZE = (1000, 700)
BACKGROUND = (0.012, 0.017, 0.030, 1.0)


def split_shader_file(path: Path) -> tuple[str, str]:
    """Load the two shader stages stored in one GLSL text file."""
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


def load_texture(ctx: moderngl.Context, path: Path) -> moderngl.Texture:
    """Load a Pygame RGBA image as a filtered ModernGL texture."""
    surface = pygame.image.load(path).convert_alpha()
    image_bytes = pygame.image.tobytes(surface, "RGBA", True)

    texture = ctx.texture(surface.get_size(), components=4, data=image_bytes)
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    texture.repeat_x = False
    texture.repeat_y = False
    texture.build_mipmaps()
    return texture


def create_sprite_vao(
    ctx: moderngl.Context,
    program: moderngl.Program,
) -> tuple[moderngl.Buffer, moderngl.VertexArray]:
    """Create one reusable unit quad for all sprite instances."""
    # x, y, u, v. Local Y points downward. The image data is vertically
    # flipped during upload, so the top vertices use v=1.
    vertices = (
        -0.5, -0.5, 0.0, 1.0,
         0.5, -0.5, 1.0, 1.0,
        -0.5,  0.5, 0.0, 0.0,
         0.5,  0.5, 1.0, 0.0,
    )
    vertex_buffer = ctx.buffer(struct.pack(f"{len(vertices)}f", *vertices))
    vao = ctx.vertex_array(
        program,
        [(vertex_buffer, "2f 2f", "in_position", "in_uv")],
    )
    return vertex_buffer, vao


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def main() -> None:
    asset_dir = Path(__file__).resolve().parent
    shader_path = asset_dir / "metallic_sprite.glsl"
    sprite_path = asset_dir / "metal_sprite.png"

    for required_file in (shader_path, sprite_path):
        if not required_file.exists():
            raise FileNotFoundError(f"Missing required file: {required_file}")

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK,
        pygame.GL_CONTEXT_PROFILE_CORE,
    )
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF, vsync=1)
    pygame.display.set_caption("ModernGL Metallic Sprite")

    ctx = moderngl.create_context(require=330)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    vertex_source, fragment_source = split_shader_file(shader_path)
    program = ctx.program(
        vertex_shader=vertex_source,
        fragment_shader=fragment_source,
    )

    vertex_buffer, vao = create_sprite_vao(ctx, program)
    sprite_texture = load_texture(ctx, sprite_path)
    sprite_texture.use(location=0)
    program["u_sprite"].value = 0
    program["u_texel_size"].value = (
        1.0 / sprite_texture.width,
        1.0 / sprite_texture.height,
    )
    program["u_alpha_cutoff"].value = 0.015
    program["u_normal_strength"].value = 9.0

    clock = pygame.time.Clock()
    running = True
    elapsed = 0.0

    position = pygame.Vector2(WINDOW_SIZE[0] * 0.5, WINDOW_SIZE[1] * 0.51)
    sprite_size = pygame.Vector2(380.0, 380.0)
    rotation = 0.0
    metallic = 1.0
    roughness = 0.22

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
                    elif event.key == pygame.K_m:
                        metallic = 0.0 if metallic > 0.5 else 1.0
                    elif event.key == pygame.K_r:
                        position.update(WINDOW_SIZE[0] * 0.5, WINDOW_SIZE[1] * 0.51)
                        rotation = 0.0
                        metallic = 1.0
                        roughness = 0.22

            keys = pygame.key.get_pressed()
            movement = pygame.Vector2(
                float(keys[pygame.K_d]) - float(keys[pygame.K_a]),
                float(keys[pygame.K_s]) - float(keys[pygame.K_w]),
            )
            if movement.length_squared() > 0.0:
                movement = movement.normalize()
                position += movement * 290.0 * delta_time

            rotation += (
                float(keys[pygame.K_e]) - float(keys[pygame.K_q])
            ) * 1.8 * delta_time

            roughness += (
                float(keys[pygame.K_DOWN]) - float(keys[pygame.K_UP])
            ) * 0.55 * delta_time
            roughness = clamp(roughness, 0.045, 1.0)

            half_size = sprite_size * 0.34
            position.x = clamp(position.x, half_size.x, WINDOW_SIZE[0] - half_size.x)
            position.y = clamp(position.y, half_size.y, WINDOW_SIZE[1] - half_size.y)

            mouse_position = pygame.mouse.get_pos()

            ctx.clear(*BACKGROUND)
            program["u_resolution"].value = tuple(map(float, WINDOW_SIZE))
            program["u_position"].value = (position.x, position.y)
            program["u_size"].value = (sprite_size.x, sprite_size.y)
            program["u_rotation"].value = rotation
            program["u_light_position"].value = tuple(map(float, mouse_position))
            program["u_time"].value = elapsed
            program["u_metallic"].value = metallic
            program["u_roughness"].value = roughness

            sprite_texture.use(location=0)
            vao.render(mode=moderngl.TRIANGLE_STRIP)
            pygame.display.flip()

            material_name = "metal" if metallic > 0.5 else "painted plastic"
            pygame.display.set_caption(
                "ModernGL Metallic Sprite | "
                f"material: {material_name} | roughness: {roughness:.2f} | "
                "mouse=light, WASD=move, Q/E=rotate, M=toggle"
            )
    finally:
        vao.release()
        vertex_buffer.release()
        sprite_texture.release()
        program.release()
        ctx.release()
        pygame.quit()


if __name__ == "__main__":
    main()
