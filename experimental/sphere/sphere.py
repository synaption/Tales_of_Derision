from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


BUILD_ID = "FLAT-LOCAL-SPHERE-MAP-V4-2026-08-05"

FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
# Exact cube-net atlas. Coordinates are in whole face widths and use a
# Cartesian Y axis (up is positive). All faces keep their native tile
# orientation, so every atlas cell is an axis-aligned square.
#
#             +Y
#     -X      +Z      +X      -Z
#             -Y
FACE_NET_OFFSETS = {
    1: (0, 1),  # -X
    4: (1, 1),  # +Z
    0: (2, 1),  # +X
    5: (3, 1),  # -Z
    2: (1, 2),  # +Y
    3: (1, 0),  # -Y
}
BIOME_COLORS = {
    "deep_water": (0.035, 0.12, 0.29),
    "water": (0.055, 0.23, 0.46),
    "sand": (0.78, 0.69, 0.43),
    "grass": (0.22, 0.54, 0.25),
    "forest": (0.075, 0.31, 0.16),
    "rock": (0.42, 0.43, 0.40),
    "snow": (0.88, 0.92, 0.95),
}
BLOCKED_BIOMES = {"deep_water", "water"}


def normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length <= 1e-12:
        return np.zeros_like(v, dtype=np.float64)
    return v / length


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        ((1.0, 0.0, 0.0, 0.0), (0.0, c, -s, 0.0), (0.0, s, c, 0.0), (0.0, 0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        ((c, 0.0, s, 0.0), (0.0, 1.0, 0.0, 0.0), (-s, 0.0, c, 0.0), (0.0, 0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a 4x4 rotation that maps one normalized direction to another."""
    a = normalize(np.asarray(source, dtype=np.float64))
    b = normalize(np.asarray(target, dtype=np.float64))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    axis_cross = np.cross(a, b)
    s = float(np.linalg.norm(axis_cross))

    if s <= 1e-10:
        if c > 0.0:
            return np.eye(4, dtype=np.float64)
        candidate = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        if abs(float(np.dot(a, candidate))) > 0.9:
            candidate = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        axis = normalize(np.cross(a, candidate))
        x, y, z = axis
        # Rodrigues for a 180-degree turn: R = 2 aa^T - I.
        r3 = 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    else:
        x, y, z = axis_cross / s
        k = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)
        r3 = np.eye(3, dtype=np.float64) + k * s + (k @ k) * (1.0 - c)

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = r3
    return result


def orthographic(left: float, right: float, bottom: float, top: float, near: float, far: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[0, 0] = 2.0 / (right - left)
    result[1, 1] = 2.0 / (top - bottom)
    result[2, 2] = -2.0 / (far - near)
    result[0, 3] = -(right + left) / (right - left)
    result[1, 3] = -(top + bottom) / (top - bottom)
    result[2, 3] = -(far + near) / (far - near)
    return result


def cube_point(face: int, u: float, v: float, extent: float = 1.0) -> np.ndarray:
    """Map a point on one of six cube faces to 3D cube coordinates."""
    if face == 0:  # +X
        return np.array((extent, v, -u), dtype=np.float64)
    if face == 1:  # -X
        return np.array((-extent, v, u), dtype=np.float64)
    if face == 2:  # +Y
        return np.array((u, extent, -v), dtype=np.float64)
    if face == 3:  # -Y
        return np.array((u, -extent, v), dtype=np.float64)
    if face == 4:  # +Z
        return np.array((u, v, extent), dtype=np.float64)
    if face == 5:  # -Z
        return np.array((-u, v, -extent), dtype=np.float64)
    raise ValueError(f"Unknown face index: {face}")


def hash3(ix: int, iy: int, iz: int, seed: int) -> float:
    """Deterministic integer hash mapped to [-1, 1]."""
    n = (
        ix * 0x1F123BB5
        ^ iy * 0x05491333
        ^ iz * 0x7F4A7C15
        ^ seed * 0x6C8E9CF5
    ) & 0xFFFFFFFF
    n ^= n >> 16
    n = (n * 0x7FEB352D) & 0xFFFFFFFF
    n ^= n >> 15
    n = (n * 0x846CA68B) & 0xFFFFFFFF
    n ^= n >> 16
    return (n / 0xFFFFFFFF) * 2.0 - 1.0


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def value_noise_3d(p: np.ndarray, seed: int) -> float:
    base = np.floor(p).astype(np.int64)
    frac = p - base
    fx, fy, fz = (smoothstep(float(x)) for x in frac)

    values = np.empty((2, 2, 2), dtype=np.float64)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                values[dx, dy, dz] = hash3(
                    int(base[0] + dx),
                    int(base[1] + dy),
                    int(base[2] + dz),
                    seed,
                )

    x00 = values[0, 0, 0] * (1 - fx) + values[1, 0, 0] * fx
    x10 = values[0, 1, 0] * (1 - fx) + values[1, 1, 0] * fx
    x01 = values[0, 0, 1] * (1 - fx) + values[1, 0, 1] * fx
    x11 = values[0, 1, 1] * (1 - fx) + values[1, 1, 1] * fx
    y0 = x00 * (1 - fy) + x10 * fy
    y1 = x01 * (1 - fy) + x11 * fy
    return float(y0 * (1 - fz) + y1 * fz)


def fractal_noise_3d(direction: np.ndarray, seed: int) -> float:
    total = 0.0
    amplitude = 1.0
    frequency = 1.6
    weight = 0.0
    for octave in range(5):
        total += amplitude * value_noise_3d(direction * frequency + 17.0, seed + octave * 1013)
        weight += amplitude
        frequency *= 2.03
        amplitude *= 0.5
    return total / weight


@dataclass(frozen=True)
class TileId:
    face: int
    x: int
    y: int


@dataclass
class Tile:
    tile_id: TileId
    center: np.ndarray
    corners: np.ndarray
    # Ordered by this tile's four edges:
    #   0 = south/bottom, 1 = east/right, 2 = north/top, 3 = west/left.
    # The order is local to the quad and remains meaningful across cube seams.
    neighbors: list[int]
    # For each local edge, the index of the matching edge on the neighbor.
    neighbor_edges: list[int]
    height: float = 0.0
    temperature: float = 0.0
    moisture: float = 0.0
    biome: str = "grass"

    @property
    def walkable(self) -> bool:
        return self.biome not in BLOCKED_BIOMES


class CubeSphereWorld:
    def __init__(self, face_size: int = 32, seed: int = 1337):
        if face_size < 4:
            raise ValueError("face_size must be at least 4")
        self.face_size = face_size
        self.seed = int(seed)
        self.tiles: list[Tile] = []
        self.index_by_id: dict[TileId, int] = {}
        self.average_step_angle = 0.0
        self._build_topology()
        self.regenerate(seed)

    def _grid_cube_point(self, face: int, gx: int, gy: int) -> tuple[int, int, int]:
        n = self.face_size
        u = -n + 2 * gx
        v = -n + 2 * gy
        p = cube_point(face, u, v, extent=n)
        return int(p[0]), int(p[1]), int(p[2])

    def _build_topology(self) -> None:
        edge_users: dict[
            tuple[tuple[int, int, int], tuple[int, int, int]],
            list[tuple[int, int]],
        ] = defaultdict(list)

        for face in range(6):
            for y in range(self.face_size):
                for x in range(self.face_size):
                    grid_corners = [
                        self._grid_cube_point(face, x, y),
                        self._grid_cube_point(face, x + 1, y),
                        self._grid_cube_point(face, x + 1, y + 1),
                        self._grid_cube_point(face, x, y + 1),
                    ]
                    corners = np.array([normalize(np.array(p, dtype=np.float64)) for p in grid_corners])
                    center = normalize(np.mean(corners, axis=0))
                    tile = Tile(
                        TileId(face, x, y),
                        center,
                        corners,
                        [-1, -1, -1, -1],
                        [-1, -1, -1, -1],
                    )
                    index = len(self.tiles)
                    self.tiles.append(tile)
                    self.index_by_id[tile.tile_id] = index

                    for edge_index, (a, b) in enumerate(
                        zip(grid_corners, grid_corners[1:] + grid_corners[:1])
                    ):
                        edge_users[tuple(sorted((a, b)))].append((index, edge_index))

        bad_edges = [edge for edge, users in edge_users.items() if len(users) != 2]
        if bad_edges:
            raise RuntimeError(f"Cube-sphere topology has {len(bad_edges)} non-manifold edges")

        for users in edge_users.values():
            (a, edge_a), (b, edge_b) = users
            self.tiles[a].neighbors[edge_a] = b
            self.tiles[a].neighbor_edges[edge_a] = edge_b
            self.tiles[b].neighbors[edge_b] = a
            self.tiles[b].neighbor_edges[edge_b] = edge_a

        bad_tiles = [
            tile.tile_id
            for tile in self.tiles
            if any(n < 0 for n in tile.neighbors) or any(e < 0 for e in tile.neighbor_edges)
        ]
        if bad_tiles:
            raise RuntimeError(f"Expected four neighbors per tile; bad examples: {bad_tiles[:8]}")

        angles = []
        for i, tile in enumerate(self.tiles):
            for j in tile.neighbors:
                if j > i:
                    dot = float(np.clip(np.dot(tile.center, self.tiles[j].center), -1.0, 1.0))
                    angles.append(math.acos(dot))
        self.average_step_angle = float(np.median(angles))

    def regenerate(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        heights = []
        moistures = []

        for tile in self.tiles:
            p = tile.center
            broad = fractal_noise_3d(p * 0.82, self.seed)
            detail = fractal_noise_3d(p * 2.45, self.seed + 7001)
            ridge = 1.0 - abs(fractal_noise_3d(p * 1.35, self.seed + 17011))
            height = broad * 0.68 + detail * 0.18 + (ridge - 0.5) * 0.26
            moisture = fractal_noise_3d(p * 1.18, self.seed + 31013) * 0.5 + 0.5
            latitude = abs(float(p[1]))
            temperature = 1.0 - latitude ** 1.25 - max(0.0, height - 0.2) * 0.45
            tile.height = height
            tile.moisture = moisture
            tile.temperature = temperature
            heights.append(height)
            moistures.append(moisture)

        sea_level = float(np.quantile(np.asarray(heights), 0.42))
        mountain = float(np.quantile(np.asarray(heights), 0.87))
        snowline = float(np.quantile(np.asarray(heights), 0.94))

        for tile in self.tiles:
            h = tile.height
            if h < sea_level - 0.12:
                tile.biome = "deep_water"
            elif h < sea_level:
                tile.biome = "water"
            elif h < sea_level + 0.055:
                tile.biome = "sand"
            elif h > snowline or tile.temperature < 0.12:
                tile.biome = "snow"
            elif h > mountain:
                tile.biome = "rock"
            elif tile.moisture > 0.57:
                tile.biome = "forest"
            else:
                tile.biome = "grass"

    def choose_start(self) -> int:
        candidates = []
        for i, tile in enumerate(self.tiles):
            if tile.biome not in {"grass", "forest", "sand"}:
                continue
            walkable_neighbors = sum(self.tiles[n].walkable for n in tile.neighbors)
            if walkable_neighbors >= 3:
                equator_bonus = 1.0 - abs(float(tile.center[1]))
                score = walkable_neighbors + equator_bonus + tile.moisture * 0.2
                candidates.append((score, i))
        if not candidates:
            return next(i for i, tile in enumerate(self.tiles) if tile.walkable)
        return max(candidates)[1]

    def biome_color(self, tile_index: int) -> tuple[float, float, float]:
        tile = self.tiles[tile_index]
        base = np.array(BIOME_COLORS[tile.biome], dtype=np.float64)
        shade = np.clip(0.92 + tile.height * 0.16, 0.78, 1.12)
        return tuple(np.clip(base * shade, 0.0, 1.0))

    def flat_map_center(self, tile_index: int) -> tuple[float, float]:
        """Return the center of a globe quad in the exact flat cube-net atlas."""
        tile_id = self.tiles[tile_index].tile_id
        face_x, face_y = FACE_NET_OFFSETS[tile_id.face]
        return (
            face_x * self.face_size + tile_id.x + 0.5,
            face_y * self.face_size + tile_id.y + 0.5,
        )

    @property
    def flat_map_size(self) -> tuple[int, int]:
        return 4 * self.face_size, 3 * self.face_size

    def best_neighbor(self, tile_index: int, desired_tangent: np.ndarray) -> int:
        tile = self.tiles[tile_index]
        n = tile.center
        desired = normalize(desired_tangent - n * np.dot(desired_tangent, n))
        best = tile_index
        best_score = -1e9
        for neighbor_index in tile.neighbors:
            q = self.tiles[neighbor_index].center
            tangent = normalize(q - n * np.dot(q, n))
            score = float(np.dot(tangent, desired))
            if score > best_score:
                best_score = score
                best = neighbor_index
        return best

    def validate(self) -> None:
        expected = 6 * self.face_size * self.face_size
        assert len(self.tiles) == expected
        assert all(len(t.neighbors) == 4 for t in self.tiles)
        assert all(len(t.neighbor_edges) == 4 for t in self.tiles)
        for i, tile in enumerate(self.tiles):
            for edge, n in enumerate(tile.neighbors):
                other_edge = tile.neighbor_edges[edge]
                assert 0 <= other_edge < 4
                assert self.tiles[n].neighbors[other_edge] == i
                assert self.tiles[n].neighbor_edges[other_edge] == edge
        assert 0.0 < self.average_step_angle < math.pi / 2
        assert any(tile.walkable for tile in self.tiles)
        assert any(not tile.walkable for tile in self.tiles)


class Player:
    def __init__(self, world: CubeSphereWorld):
        self.tile_index = world.choose_start()
        # Rotation from screen-space directions to this quad's ordered edges.
        # 0 means screen north/east/south/west match tile edges 2/1/0/3.
        self.view_rotation = 0
        self.steps = 0
        self.blocked_flash = 0.0

    def basis(self, world: CubeSphereWorld) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tile = world.tiles[self.tile_index]
        normal = tile.center
        north_edge = (2 + self.view_rotation) % 4
        a = tile.corners[north_edge]
        b = tile.corners[(north_edge + 1) % 4]
        edge_midpoint = normalize(a + b)
        forward = normalize(edge_midpoint - normal * np.dot(edge_midpoint, normal))
        right = normalize(np.cross(forward, normal))
        return right, forward, normal

    def move(self, world: CubeSphereWorld, forward_amount: int, right_amount: int) -> bool:
        if forward_amount > 0:
            screen_edge = 2  # north
        elif forward_amount < 0:
            screen_edge = 0  # south
        elif right_amount > 0:
            screen_edge = 1  # east
        elif right_amount < 0:
            screen_edge = 3  # west
        else:
            return False

        tile = world.tiles[self.tile_index]
        local_edge = (screen_edge + self.view_rotation) % 4
        target = tile.neighbors[local_edge]
        if not world.tiles[target].walkable:
            self.blocked_flash = 0.18
            return False

        neighbor_edge = tile.neighbor_edges[local_edge]
        self.tile_index = target
        # The edge just crossed must become the opposite screen edge on the
        # neighbor. This carries the local square-grid orientation over seams.
        self.view_rotation = (neighbor_edge - ((screen_edge + 2) % 4)) % 4
        self.steps += 1
        return True


VERTEX_2D = """
#version 330
in vec2 in_position;
in vec3 in_color;
uniform vec2 u_resolution;
out vec3 v_color;
void main() {
    vec2 ndc = vec2(
        in_position.x / u_resolution.x * 2.0 - 1.0,
        1.0 - in_position.y / u_resolution.y * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = in_color;
}
"""

FRAGMENT_COLOR = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""

VERTEX_3D = """
#version 330
in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
uniform mat4 u_mvp;
out vec3 v_normal;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_normal = normalize(in_normal);
    v_color = in_color;
}
"""

FRAGMENT_3D = """
#version 330
in vec3 v_normal;
in vec3 v_color;
out vec4 f_color;
void main() {
    vec3 light_direction = normalize(vec3(-0.45, 0.65, 1.0));
    float diffuse = max(dot(normalize(v_normal), light_direction), 0.0);
    float lighting = 0.58 + diffuse * 0.42;
    f_color = vec4(v_color * lighting, 1.0);
}
"""

VERTEX_3D_LINE = """
#version 330
in vec3 in_position;
in vec3 in_color;
uniform mat4 u_mvp;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_color = in_color;
}
"""

FRAGMENT_3D_LINE = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""


VERTEX_TEXTURE = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_uv;
}
"""

FRAGMENT_TEXTURE = """
#version 330
uniform sampler2D u_texture;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(u_texture, v_uv);
}
"""


class Renderer:
    def __init__(self, world: CubeSphereWorld, width: int, height: int):
        import moderngl
        import pygame

        self.moderngl = moderngl
        self.pygame = pygame
        self.ctx = moderngl.create_context(require=330)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.width = width
        self.height = height

        self.program_2d = self.ctx.program(vertex_shader=VERTEX_2D, fragment_shader=FRAGMENT_COLOR)
        self.vbo_2d = self.ctx.buffer(reserve=1024 * 1024)
        self.vao_2d = self.ctx.vertex_array(
            self.program_2d,
            [(self.vbo_2d, "2f 3f", "in_position", "in_color")],
        )

        self.program_3d = self.ctx.program(vertex_shader=VERTEX_3D, fragment_shader=FRAGMENT_3D)
        self.vbo_globe = self.ctx.buffer(reserve=4 * 1024 * 1024)
        self.vao_globe = self.ctx.vertex_array(
            self.program_3d,
            [(self.vbo_globe, "3f 3f 3f", "in_position", "in_normal", "in_color")],
        )
        self.program_3d_line = self.ctx.program(
            vertex_shader=VERTEX_3D_LINE,
            fragment_shader=FRAGMENT_3D_LINE,
        )
        self.vbo_globe_grid = self.ctx.buffer(reserve=2 * 1024 * 1024)
        self.vao_globe_grid = self.ctx.vertex_array(
            self.program_3d_line,
            [(self.vbo_globe_grid, "3f 3f", "in_position", "in_color")],
        )
        self.vbo_player_outline = self.ctx.buffer(reserve=1024)
        self.vao_player_outline = self.ctx.vertex_array(
            self.program_3d_line,
            [(self.vbo_player_outline, "3f 3f", "in_position", "in_color")],
        )
        self.globe_vertex_count = 0
        self.globe_grid_vertex_count = 0

        self.program_texture = self.ctx.program(
            vertex_shader=VERTEX_TEXTURE,
            fragment_shader=FRAGMENT_TEXTURE,
        )
        quad = np.array(
            [
                -1, -1, 0, 1,
                1, -1, 1, 1,
                1, 1, 1, 0,
                -1, -1, 0, 1,
                1, 1, 1, 0,
                -1, 1, 0, 0,
            ],
            dtype="f4",
        )
        self.quad_vbo = self.ctx.buffer(quad.tobytes())
        self.quad_vao = self.ctx.vertex_array(
            self.program_texture,
            [(self.quad_vbo, "2f 2f", "in_position", "in_uv")],
        )
        self.overlay_texture = self.ctx.texture((width, height), 4)
        self.overlay_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont("consolas", 27, bold=True)
        self.rebuild_world(world)

    def resize(self, width: int, height: int) -> None:
        self.width = max(1, width)
        self.height = max(1, height)
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.overlay_texture.release()
        self.overlay_texture = self.ctx.texture((self.width, self.height), 4)
        self.overlay_texture.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)

    def rebuild_world(self, world: CubeSphereWorld) -> None:
        """Upload the real curved quad mesh used only by the M-map."""
        surface: list[float] = []
        grid: list[float] = []
        triangles = ((0, 1, 2), (0, 2, 3))

        for tile_index, tile in enumerate(world.tiles):
            color = world.biome_color(tile_index)
            normal = tile.center
            for triangle in triangles:
                for corner_index in triangle:
                    position = tile.corners[corner_index]
                    surface.extend((*position, *normal, *color))

            # Each shared edge is uploaded once. A tiny radial offset keeps the
            # square-tile grid readable without changing the terrain surface.
            for edge, neighbor in enumerate(tile.neighbors):
                if tile_index > neighbor:
                    continue
                a = tile.corners[edge] * 1.0035
                b = tile.corners[(edge + 1) % 4] * 1.0035
                line_color = (0.025, 0.032, 0.045)
                grid.extend((*a, *line_color, *b, *line_color))

        surface_data = np.asarray(surface, dtype="f4")
        grid_data = np.asarray(grid, dtype="f4")
        self.vbo_globe.orphan(max(surface_data.nbytes, 36))
        self.vbo_globe.write(surface_data.tobytes())
        self.vbo_globe_grid.orphan(max(grid_data.nbytes, 24))
        self.vbo_globe_grid.write(grid_data.tobytes())
        self.globe_vertex_count = len(surface_data) // 9
        self.globe_grid_vertex_count = len(grid_data) // 6

    @staticmethod
    def _append_rect(vertices: list[float], x: float, y: float, w: float, h: float, color: Iterable[float]) -> None:
        r, g, b = color
        vertices.extend((x, y, r, g, b, x + w, y, r, g, b, x + w, y + h, r, g, b))
        vertices.extend((x, y, r, g, b, x + w, y + h, r, g, b, x, y + h, r, g, b))

    @classmethod
    def _append_outline(
        cls,
        vertices: list[float],
        x: float,
        y: float,
        w: float,
        h: float,
        thickness: float,
        color: Iterable[float],
    ) -> None:
        cls._append_rect(vertices, x, y, w, thickness, color)
        cls._append_rect(vertices, x, y + h - thickness, w, thickness, color)
        cls._append_rect(vertices, x, y, thickness, h, color)
        cls._append_rect(vertices, x + w - thickness, y, thickness, h, color)

    @staticmethod
    def _append_triangle(vertices: list[float], points: Iterable[tuple[float, float]], color: Iterable[float]) -> None:
        r, g, b = color
        for x, y in points:
            vertices.extend((x, y, r, g, b))

    @staticmethod
    def _unfold_local_patch(
        world: CubeSphereWorld,
        player: Player,
        half_width: int,
        half_height: int,
    ) -> list[tuple[int, int, int, int]]:
        """Unfold actual connected globe quads into a flat square chart.

        Each result is (tile_index, grid_x, grid_y, tile_rotation). The chart
        is centered on the player's real spherical-topology tile. Every screen
        square is therefore a one-to-one view of one world quad.

        A sphere cannot be covered by one globally consistent square chart;
        cube-sphere corner singularities require a cut. This BFS keeps that cut
        near the outside of the local viewport by accepting the nearest
        placement of each tile first.
        """
        # Screen-space edge order matches Tile.neighbors:
        # south, east, north, west.
        deltas = ((0, -1), (1, 0), (0, 1), (-1, 0))
        queue = deque([(player.tile_index, 0, 0, player.view_rotation)])
        placed_tiles: set[int] = set()
        occupied: set[tuple[int, int]] = set()
        result: list[tuple[int, int, int, int]] = []

        while queue:
            tile_index, gx, gy, rotation = queue.popleft()
            if tile_index in placed_tiles or (gx, gy) in occupied:
                continue
            if abs(gx) > half_width or abs(gy) > half_height:
                continue

            placed_tiles.add(tile_index)
            occupied.add((gx, gy))
            result.append((tile_index, gx, gy, rotation))
            tile = world.tiles[tile_index]

            for screen_edge, (dx, dy) in enumerate(deltas):
                nx, ny = gx + dx, gy + dy
                if abs(nx) > half_width or abs(ny) > half_height:
                    continue
                local_edge = (screen_edge + rotation) % 4
                neighbor = tile.neighbors[local_edge]
                neighbor_edge = tile.neighbor_edges[local_edge]
                neighbor_rotation = (
                    neighbor_edge - ((screen_edge + 2) % 4)
                ) % 4
                queue.append((neighbor, nx, ny, neighbor_rotation))

        return result

    def render_local(self, world: CubeSphereWorld, player: Player) -> None:
        self.ctx.disable(self.moderngl.DEPTH_TEST)
        self.ctx.clear(0.018, 0.024, 0.032, 1.0)

        tile_px = max(28.0, min(58.0, min(self.width, self.height) / 13.0))
        half_width = int(math.ceil(self.width / (tile_px * 2.0))) + 2
        half_height = int(math.ceil(self.height / (tile_px * 2.0))) + 2
        vertices: list[float] = []
        center_x = self.width * 0.5
        center_y = self.height * 0.5
        half = tile_px * 0.5
        grid_line = max(0.75, min(1.5, tile_px * 0.025))

        visible = self._unfold_local_patch(world, player, half_width, half_height)
        for tile_index, gx, gy, _ in visible:
            sx = center_x + gx * tile_px
            sy = center_y - gy * tile_px
            color = world.biome_color(tile_index)
            self._append_rect(
                vertices,
                sx - half + grid_line,
                sy - half + grid_line,
                tile_px - grid_line * 2.0,
                tile_px - grid_line * 2.0,
                color,
            )

        marker_color = (1.0, 0.91, 0.18) if player.blocked_flash <= 0 else (1.0, 0.24, 0.16)
        p = tile_px * 0.33
        self._append_triangle(
            vertices,
            ((center_x, center_y - p), (center_x + p * 0.78, center_y + p), (center_x - p * 0.78, center_y + p)),
            marker_color,
        )

        data = np.asarray(vertices, dtype="f4")
        self.vbo_2d.orphan(max(data.nbytes, 20))
        if data.nbytes:
            self.vbo_2d.write(data.tobytes())
        self.program_2d["u_resolution"].value = (self.width, self.height)
        self.vao_2d.render(vertices=len(data) // 5)

        tile = world.tiles[player.tile_index]
        self._draw_overlay(
            [
                (f"FLAT LOCAL VIEW  |  {BUILD_ID}", (18, 16), self.title_font, (242, 246, 255)),
                (f"Biome: {tile.biome.replace('_', ' ').title()}   Height: {tile.height:+.2f}", (20, 54), self.font, (220, 228, 240)),
                (f"Actual globe quad: {FACE_NAMES[tile.tile_id.face]} {tile.tile_id.x},{tile.tile_id.y}   Steps: {player.steps}", (20, 78), self.small_font, (170, 183, 202)),
                ("WASD / arrows: step     M: spherical world map     R: new world     Esc: quit", (20, self.height - 34), self.small_font, (195, 205, 220)),
            ],
            panel_rects=[(10, 9, min(650, self.width - 20), 93, (8, 12, 19, 205)), (10, self.height - 47, min(650, self.width - 20), 37, (8, 12, 19, 210))],
        )

    def render_globe(
        self,
        world: CubeSphereWorld,
        player: Player,
        rotation: np.ndarray,
        zoom: float,
    ) -> None:
        """Render the same world quads as an orthographic 3D sphere.

        Local play remains a strict flat grid. Only this M-map curves the tile
        corner positions back onto the globe. Orthographic projection avoids
        perspective parallax and distance-based tile scaling.
        """
        self.ctx.enable(self.moderngl.DEPTH_TEST)
        self.ctx.clear(0.008, 0.012, 0.021, 1.0)

        aspect = self.width / max(1.0, float(self.height))
        half_height = 1.18 / max(0.05, zoom)
        half_width = half_height * aspect
        projection = orthographic(-half_width, half_width, -half_height, half_height, 0.1, 10.0)
        view = np.eye(4, dtype=np.float64)
        view[2, 3] = -3.0
        mvp = projection @ view @ rotation
        mvp_bytes = np.asarray(mvp.T, dtype="f4").tobytes()

        self.program_3d["u_mvp"].write(mvp_bytes)
        self.vao_globe.render(vertices=self.globe_vertex_count)

        self.program_3d_line["u_mvp"].write(mvp_bytes)
        self.ctx.line_width = 1.0
        self.vao_globe_grid.render(
            mode=self.moderngl.LINES,
            vertices=self.globe_grid_vertex_count,
        )

        # Highlight the player's exact quad on the sphere.
        tile = world.tiles[player.tile_index]
        marker_color = (1.0, 0.9, 0.12)
        outline: list[float] = []
        for edge in range(4):
            a = tile.corners[edge] * 1.018
            b = tile.corners[(edge + 1) % 4] * 1.018
            outline.extend((*a, *marker_color, *b, *marker_color))
        outline_data = np.asarray(outline, dtype="f4")
        self.vbo_player_outline.orphan(max(outline_data.nbytes, 24))
        self.vbo_player_outline.write(outline_data.tobytes())
        self.ctx.line_width = 3.0
        self.vao_player_outline.render(mode=self.moderngl.LINES, vertices=len(outline_data) // 6)
        self.ctx.line_width = 1.0

        self.ctx.disable(self.moderngl.DEPTH_TEST)
        self._draw_overlay(
            [
                (f"ORTHOGRAPHIC SPHERICAL MAP  |  {BUILD_ID}", (22, 18), self.title_font, (242, 246, 255)),
                (f"{world.face_size}x{world.face_size} quads per cube face   {len(world.tiles):,} curved world tiles", (24, 54), self.small_font, (180, 193, 212)),
                (f"Player: {FACE_NAMES[tile.tile_id.face]} {tile.tile_id.x},{tile.tile_id.y}   Same tile mesh as local view", (24, 76), self.small_font, (220, 226, 237)),
                ("Drag/WASD: rotate globe   Wheel: zoom   C: center player   M/Esc: close map", (24, self.height - 34), self.small_font, (195, 205, 220)),
            ],
            panel_rects=[
                (12, 10, min(790, self.width - 24), 88, (8, 12, 20, 210)),
                (12, self.height - 47, min(760, self.width - 24), 37, (8, 12, 20, 215)),
            ],
        )

    def _draw_overlay(self, labels, panel_rects=()) -> None:
        pygame = self.pygame
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for x, y, w, h, color in panel_rects:
            pygame.draw.rect(surface, color, pygame.Rect(x, y, w, h), border_radius=8)
            pygame.draw.rect(surface, (80, 95, 120, 110), pygame.Rect(x, y, w, h), width=1, border_radius=8)
        for text, pos, font, color in labels:
            surface.blit(font.render(text, True, color), pos)
        rgba = pygame.image.tostring(surface, "RGBA", False)
        self.overlay_texture.write(rgba)
        self.overlay_texture.use(0)
        self.program_texture["u_texture"].value = 0
        self.quad_vao.render()


class Game:
    def __init__(self, face_size: int, seed: int):
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        self.width, self.height = 1280, 800
        pygame.display.set_mode(
            (self.width, self.height),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
            vsync=1,
        )
        pygame.display.set_caption(f"{BUILD_ID} — Flat local grid + spherical M-map")

        self.world = CubeSphereWorld(face_size=face_size, seed=seed)
        self.player = Player(self.world)
        self.renderer = Renderer(self.world, self.width, self.height)
        self.clock = pygame.time.Clock()
        self.running = True
        self.map_open = False
        self.map_zoom = 1.0
        self.map_rotation = self.rotation_centered_on_player()
        self.dragging = False

    def rotation_centered_on_player(self) -> np.ndarray:
        player_direction = self.world.tiles[self.player.tile_index].center
        # Camera looks toward the origin from +Z, so +Z is the visible center.
        return rotation_from_to(player_direction, np.array((0.0, 0.0, 1.0), dtype=np.float64))

    def regenerate(self) -> None:
        seed = random.SystemRandom().randrange(1, 2**31)
        self.world.regenerate(seed)
        self.player = Player(self.world)
        self.map_rotation = self.rotation_centered_on_player()
        self.renderer.rebuild_world(self.world)

    def rotate_map(self, horizontal: float, vertical: float) -> None:
        # Pre-multiplication makes drag axes camera-relative rather than tied to
        # whichever cube face is currently visible.
        self.map_rotation = rotation_y(horizontal) @ rotation_x(vertical) @ self.map_rotation

    def handle_event(self, event) -> None:
        pygame = self.pygame
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.VIDEORESIZE:
            self.width, self.height = max(640, event.w), max(420, event.h)
            self.renderer.resize(self.width, self.height)
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                self.map_open = not self.map_open
                if self.map_open:
                    self.map_rotation = self.rotation_centered_on_player()
                return
            if event.key == pygame.K_ESCAPE:
                if self.map_open:
                    self.map_open = False
                else:
                    self.running = False
                return
            if event.key == pygame.K_r and not self.map_open:
                self.regenerate()
                return

            if self.map_open:
                angle = math.radians(7.0)
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    self.rotate_map(-angle, 0.0)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.rotate_map(angle, 0.0)
                elif event.key in (pygame.K_UP, pygame.K_w):
                    self.rotate_map(0.0, -angle)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.rotate_map(0.0, angle)
                elif event.key == pygame.K_c:
                    self.map_rotation = self.rotation_centered_on_player()
            else:
                moves = {
                    pygame.K_UP: (1, 0),
                    pygame.K_w: (1, 0),
                    pygame.K_DOWN: (-1, 0),
                    pygame.K_s: (-1, 0),
                    pygame.K_LEFT: (0, -1),
                    pygame.K_a: (0, -1),
                    pygame.K_RIGHT: (0, 1),
                    pygame.K_d: (0, 1),
                }
                if event.key in moves:
                    self.player.move(self.world, *moves[event.key])

        if self.map_open:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                dx, dy = event.rel
                self.rotate_map(dx * 0.008, dy * 0.008)
            elif event.type == pygame.MOUSEWHEEL:
                self.map_zoom = float(np.clip(self.map_zoom * (1.14 ** event.y), 0.55, 6.0))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
                delta = 1 if event.button == 4 else -1
                self.map_zoom = float(np.clip(self.map_zoom * (1.14 ** delta), 0.55, 6.0))

    def run(self) -> None:
        pygame = self.pygame
        while self.running:
            dt = self.clock.tick(120) / 1000.0
            for event in pygame.event.get():
                self.handle_event(event)
            self.player.blocked_flash = max(0.0, self.player.blocked_flash - dt)

            if self.map_open:
                self.renderer.render_globe(
                    self.world,
                    self.player,
                    self.map_rotation,
                    self.map_zoom,
                )
            else:
                self.renderer.render_local(self.world, self.player)
            pygame.display.flip()

        pygame.quit()


def run_self_test(face_size: int, seed: int) -> None:
    world = CubeSphereWorld(face_size=face_size, seed=seed)
    world.validate()
    player = Player(world)
    right, forward, _ = player.basis(world)
    assert np.isclose(np.dot(right, forward), 0.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(right), 1.0, atol=1e-6)

    patch = Renderer._unfold_local_patch(world, player, 6, 6)
    by_position = {(gx, gy): (tile_index, rotation) for tile_index, gx, gy, rotation in patch}
    assert by_position[(0, 0)] == (player.tile_index, player.view_rotation)
    assert all(pos in by_position for pos in ((0, 1), (1, 0), (0, -1), (-1, 0)))
    for (gx, gy), (tile_index, rotation) in by_position.items():
        tile = world.tiles[tile_index]
        for screen_edge, (dx, dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
            neighbor_position = (gx + dx, gy + dy)
            if neighbor_position not in by_position:
                continue
            local_edge = (screen_edge + rotation) % 4
            expected_neighbor = tile.neighbors[local_edge]
            actual_neighbor, neighbor_rotation = by_position[neighbor_position]
            assert actual_neighbor == expected_neighbor
            expected_rotation = tile.neighbor_edges[local_edge] - ((screen_edge + 2) % 4)
            assert neighbor_rotation == expected_rotation % 4

    # The map uses every tile's four normalized spherical corners.
    assert all(np.allclose(np.linalg.norm(tile.corners, axis=1), 1.0) for tile in world.tiles)
    globe_triangle_vertices = len(world.tiles) * 6
    unique_edges = sum(1 for i, tile in enumerate(world.tiles) for n in tile.neighbors if i < n)
    assert unique_edges == len(world.tiles) * 2

    centered = rotation_from_to(
        world.tiles[player.tile_index].center,
        np.array((0.0, 0.0, 1.0), dtype=np.float64),
    )
    transformed = centered[:3, :3] @ world.tiles[player.tile_index].center
    assert np.allclose(transformed, (0.0, 0.0, 1.0), atol=1e-7)
    projection = orthographic(-1.5, 1.5, -1.0, 1.0, 0.1, 10.0)
    assert np.isfinite(projection).all()

    seam_tiles = [
        i
        for i, tile in enumerate(world.tiles)
        if any(world.tiles[n].tile_id.face != tile.tile_id.face for n in tile.neighbors)
    ]
    assert seam_tiles
    print(
        f"Self-test passed: {len(world.tiles)} tiles, "
        f"{len(seam_tiles)} seam tiles, {len(patch)} locally unfolded quads, "
        f"{globe_triangle_vertices} globe triangle vertices, {unique_edges} unique grid edges, "
        f"median step {math.degrees(world.average_step_angle):.3f}°"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-down flat-tile roguelike with an orthographic spherical M-map")
    parser.add_argument("--face-size", type=int, default=32, help="tiles along each cube face edge (default: 32)")
    parser.add_argument("--seed", type=int, default=1337, help="procedural world seed")
    parser.add_argument("--self-test", action="store_true", help="validate topology and generation without opening a window")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    print(f"Running {BUILD_ID}: flat local tiles; M opens the curved orthographic sphere.")
    if args.self_test:
        run_self_test(args.face_size, args.seed)
        return 0
    try:
        Game(args.face_size, args.seed).run()
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())