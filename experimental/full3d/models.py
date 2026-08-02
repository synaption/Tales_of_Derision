"""Procedural geometry. Every model in the game is built from boxes at import.

There are no art assets in this project, which is convenient for a repository
and, as it happens, historically accurate: a PS1 character was a few hundred
triangles and a 64x64 texture, and boxes with a limb helper get you most of
the way to one. Everything here is authored in metres, Z-up, with the origin
at the feet, matching :class:`components.Transform`.

:class:`BoxBuilder` accumulates arbitrarily-oriented boxes into a single
:class:`Geom`, so an ant's six legs are one draw call rather than six nodes.
The parts that need to move independently -- leg racks, gun arms, a jetpack
flame -- are built as separate builders and parented, which is the only reason
any model here has more than one node in it.
"""

from __future__ import annotations

import math

from panda3d.core import (
    CardMaker,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LMatrix4,
    LQuaternion,
    NodePath,
    Vec3,
    lookAt,
)

# The six faces of a unit cube: outward normal, and the two in-plane axes used
# to walk its corners. Written out rather than derived because it is read far
# more often than it is changed.
_FACES = (
    (Vec3(0, 0, 1), Vec3(1, 0, 0), Vec3(0, 1, 0)),
    (Vec3(0, 0, -1), Vec3(-1, 0, 0), Vec3(0, 1, 0)),
    (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)),
    (Vec3(-1, 0, 0), Vec3(0, -1, 0), Vec3(0, 0, 1)),
    (Vec3(0, 1, 0), Vec3(-1, 0, 0), Vec3(0, 0, 1)),
    (Vec3(0, -1, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)),
)


class BoxBuilder:
    """Accumulates boxes into one triangle list with normals, UVs and colours.

    ``uv_scale`` is in texture repeats per metre, applied in the plane of each
    face, so a texture keeps a constant real-world size no matter how the box
    is proportioned -- the reason one 64x64 facade works on every tower in the
    city.
    """

    def __init__(self, uv_scale: float = 0.25) -> None:
        self.uv_scale = uv_scale
        self._verts: list[tuple[Vec3, Vec3, tuple[float, float], tuple[float, float, float]]] = []
        self._tris: list[tuple[int, int, int]] = []

    # -- authoring ---------------------------------------------------------

    def add_box(
        self,
        size: tuple[float, float, float],
        pos: tuple[float, float, float] = (0, 0, 0),
        colour: tuple[float, float, float] = (1, 1, 1),
        matrix: LMatrix4 | None = None,
        skip_bottom: bool = False,
        skip_top: bool = False,
        uv_scale: float | None = None,
        max_cell: float | None = None,
    ) -> "BoxBuilder":
        """Add a box of ``size`` centred on ``pos``, optionally transformed.

        ``skip_bottom`` and ``skip_top`` drop faces nobody can see, or that a
        caller wants to supply itself with a different texture -- which is how
        buildings get gravel roofs without a second draw call per tower.

        ``max_cell`` splits each face into a grid no coarser than that many
        metres. This is the period-correct fix for affine texture mapping: the
        warp is a function of how much a single polygon spans in depth, so
        every PS1 game chopped its large flat surfaces up. Without it a road
        or a rooftop swims so violently it stops reading as a surface at all.
        """
        half = Vec3(size[0] * 0.5, size[1] * 0.5, size[2] * 0.5)
        centre = Vec3(*pos)
        scale = self.uv_scale if uv_scale is None else uv_scale

        for normal, u_axis, v_axis in _FACES:
            if skip_bottom and normal.z < -0.5:
                continue
            if skip_top and normal.z > 0.5:
                continue
            # Corner offsets in the face's own plane, and how far out it sits.
            depth = Vec3(normal.x * half.x, normal.y * half.y, normal.z * half.z)
            u_half = abs(u_axis.x) * half.x + abs(u_axis.y) * half.y + abs(u_axis.z) * half.z
            v_half = abs(v_axis.x) * half.x + abs(v_axis.y) * half.y + abs(v_axis.z) * half.z

            steps_u = 1 if max_cell is None else max(1, math.ceil(u_half * 2.0 / max_cell))
            steps_v = 1 if max_cell is None else max(1, math.ceil(v_half * 2.0 / max_cell))

            for iu in range(steps_u):
                for iv in range(steps_v):
                    u0 = -u_half + 2.0 * u_half * iu / steps_u
                    u1 = -u_half + 2.0 * u_half * (iu + 1) / steps_u
                    v0 = -v_half + 2.0 * v_half * iv / steps_v
                    v1 = -v_half + 2.0 * v_half * (iv + 1) / steps_v

                    base = len(self._verts)
                    for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                        local = centre + depth + u_axis * u + v_axis * v
                        world_n = Vec3(normal)
                        if matrix is not None:
                            local = matrix.xformPoint(local)
                            world_n = matrix.xformVec(world_n)
                            world_n.normalize()
                        self._verts.append((local, world_n, (u * scale, v * scale), colour))
                    self._tris.append((base, base + 1, base + 2))
                    self._tris.append((base, base + 2, base + 3))
        return self

    def add_limb(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        thickness: float,
        colour: tuple[float, float, float] = (1, 1, 1),
    ) -> "BoxBuilder":
        """Add a box spanning ``start`` to ``end``. Legs, arms, mandibles.

        The box is built along +Y and rotated onto the segment, picking an up
        vector that is not parallel to it so the orientation never degenerates
        on a vertical limb.
        """
        a, b = Vec3(*start), Vec3(*end)
        direction = b - a
        length = direction.length()
        if length < 1e-5:
            return self
        direction /= length

        quat = LQuaternion()
        up = Vec3(0, 0, 1) if abs(direction.z) < 0.95 else Vec3(0, 1, 0)
        lookAt(quat, direction, up)

        matrix = LMatrix4()
        quat.extractToMatrix(matrix)
        matrix.setRow(3, (a + b) * 0.5)
        return self.add_box((thickness, length, thickness), (0, 0, 0), colour, matrix=matrix)

    def add_quad(
        self,
        corners: tuple[Vec3, Vec3, Vec3, Vec3],
        normal: Vec3,
        colour: tuple[float, float, float] = (1, 1, 1),
        uvs: tuple[tuple[float, float], ...] = ((0, 0), (1, 0), (1, 1), (0, 1)),
        steps: int = 1,
    ) -> "BoxBuilder":
        """Add a quad, optionally split into a ``steps`` x ``steps`` grid.

        Corners are given anticlockwise from the origin corner. Subdivision
        interpolates position and UV bilinearly, which is exact for the
        rectangles this is used on (ground tiles, rooftops) and good enough
        for the ones it is not (the Wing Diver's wings, which use ``steps=1``).
        """

        def lerp2(a, b, c, d, s: float, t: float):
            """Bilinear blend of the four corner values."""
            return (a * (1 - s) + b * s) * (1 - t) + (d * (1 - s) + c * s) * t

        p0, p1, p2, p3 = (Vec3(c) for c in corners)
        (u0, v0), (u1, v1), (u2, v2), (u3, v3) = uvs

        for i in range(steps):
            for j in range(steps):
                s0, s1 = i / steps, (i + 1) / steps
                t0, t1 = j / steps, (j + 1) / steps
                base = len(self._verts)
                for s, t in ((s0, t0), (s1, t0), (s1, t1), (s0, t1)):
                    position = lerp2(p0, p1, p2, p3, s, t)
                    uv = (
                        lerp2(u0, u1, u2, u3, s, t),
                        lerp2(v0, v1, v2, v3, s, t),
                    )
                    self._verts.append((position, Vec3(normal), uv, colour))
                self._tris.append((base, base + 1, base + 2))
                self._tris.append((base, base + 2, base + 3))
        return self

    # -- output ------------------------------------------------------------

    def node(self, name: str) -> NodePath:
        """Bake into a :class:`NodePath`. Empty builders give an empty node."""
        vformat = GeomVertexFormat.getV3n3cpt2()
        vdata = GeomVertexData(name, vformat, Geom.UHStatic)
        vdata.setNumRows(max(1, len(self._verts)))

        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        colour = GeomVertexWriter(vdata, "color")
        texcoord = GeomVertexWriter(vdata, "texcoord")

        for pos, norm, uv, col in self._verts:
            vertex.addData3(pos)
            normal.addData3(norm)
            colour.addData4(col[0], col[1], col[2], 1.0)
            texcoord.addData2(uv[0], uv[1])

        tris = GeomTriangles(Geom.UHStatic)
        for a, b, c in self._tris:
            tris.addVertices(a, b, c)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode(name)
        node.addGeom(geom)
        return NodePath(node)


# --------------------------------------------------------------------------
# the models
# --------------------------------------------------------------------------


def build_ground(extent: float, cell: float = 6.0) -> NodePath:
    """A tessellated ground plane.

    Deliberately *not* one big quad. The affine texture mapping warps in
    proportion to how much depth a single polygon spans, and a ground plane
    seen from a metre and a half up spans all of it -- one quad turns the road
    into a smear. Six-metre cells keep the swim to the amount a 1998 game had,
    which is the amount that reads as charm rather than as breakage. It also
    gives the vertex snapping something to bite on.
    """
    builder = BoxBuilder()
    steps = max(1, int((extent * 2.0) / cell))
    uv = 0.125  # one road tile per 8m
    builder.add_quad(
        (
            Vec3(-extent, -extent, 0),
            Vec3(extent, -extent, 0),
            Vec3(extent, extent, 0),
            Vec3(-extent, extent, 0),
        ),
        Vec3(0, 0, 1),
        (1, 1, 1),
        uvs=(
            (-extent * uv, -extent * uv),
            (extent * uv, -extent * uv),
            (extent * uv, extent * uv),
            (-extent * uv, extent * uv),
        ),
        steps=steps,
    )
    node = builder.node("ground")
    node.flattenStrong()
    return node


def build_ant() -> NodePath:
    """A giant ant, in three pieces so the legs can walk.

    Root
      +- ``body``      thorax, abdomen, head, mandibles
      +- ``legs_l``    three left legs, pivoting about the body centre
      +- ``legs_r``    three right legs, in antiphase

    Counter-rotating two leg racks is the cheapest thing that reads as a gait
    from more than five metres away, and at 320x240 nothing is ever closer.
    """
    dark = (0.55, 0.5, 0.5)

    body = BoxBuilder(uv_scale=0.9)
    body.add_box((1.15, 1.65, 1.15), (0.0, -1.05, 0.85))  # abdomen
    body.add_box((0.55, 0.5, 0.5), (0.0, -0.15, 0.85), colour=dark)  # waist
    body.add_box((0.95, 1.05, 0.9), (0.0, 0.45, 0.85))  # thorax
    body.add_box((0.85, 0.75, 0.7), (0.0, 1.25, 0.88))  # head
    body.add_limb((0.28, 1.55, 0.75), (0.5, 2.25, 0.55), 0.16, dark)  # mandibles
    body.add_limb((-0.28, 1.55, 0.75), (-0.5, 2.25, 0.55), 0.16, dark)
    body.add_limb((0.2, 1.5, 1.15), (0.55, 2.4, 1.5), 0.09, dark)  # antennae
    body.add_limb((-0.2, 1.5, 1.15), (-0.55, 2.4, 1.5), 0.09, dark)

    racks = []
    for side in (1.0, -1.0):
        rack = BoxBuilder(uv_scale=0.9)
        for index, y in enumerate((0.75, 0.15, -0.5)):
            reach = 1.05 + index * 0.12
            knee = (side * 0.85, y + 0.15, 1.35)
            foot = (side * reach, y + 0.35 - index * 0.35, 0.0)
            rack.add_limb((side * 0.35, y, 0.85), knee, 0.17, dark)
            rack.add_limb(knee, foot, 0.13, dark)
        racks.append(rack)

    root = NodePath("ant")
    part = body.node("body")
    part.flattenStrong()
    part.reparentTo(root)
    for name, rack in zip(("legs_l", "legs_r"), racks):
        node = rack.node(name)
        node.flattenStrong()
        node.reparentTo(root)
        node.setPos(0, 0, 0)
    return root


def _humanoid(builder_body: BoxBuilder, skin: tuple[float, float, float]) -> None:
    """Shared torso/head/helmet block for both EDF body types."""
    builder_body.add_box((0.58, 0.34, 0.66), (0.0, 0.0, 1.18))  # torso
    builder_body.add_box((0.44, 0.30, 0.14), (0.0, 0.0, 0.85))  # belt
    builder_body.add_box((0.28, 0.26, 0.28), (0.0, 0.0, 1.62), colour=skin)  # head
    builder_body.add_box((0.4, 0.42, 0.16), (0.0, 0.02, 1.74))  # helmet


def build_soldier() -> NodePath:
    """An EDF grunt.

    Root
      +- ``body``   torso, head, helmet, backpack
      +- ``leg_l``  one leg, pivoting at the hip
      +- ``leg_r``
      +- ``arms``   both arms and the rifle, so recoil moves the whole lot
    """
    skin = (0.75, 0.6, 0.5)
    root = NodePath("soldier")

    body = BoxBuilder(uv_scale=1.4)
    _humanoid(body, skin)
    body.add_box((0.36, 0.2, 0.44), (0.0, -0.26, 1.2), colour=(0.5, 0.52, 0.45))  # pack
    part = body.node("body")
    part.flattenStrong()
    part.reparentTo(root)

    for name, side in (("leg_l", 1.0), ("leg_r", -1.0)):
        leg = BoxBuilder(uv_scale=1.4)
        leg.add_limb((side * 0.16, 0, 0.0), (side * 0.16, 0, -0.85), 0.24)
        node = leg.node(name)
        node.flattenStrong()
        node.reparentTo(root)
        node.setPos(0, 0, 0.85)  # hip, so a rotation about X swings the leg

    arms = BoxBuilder(uv_scale=1.4)
    for side in (1.0, -1.0):
        arms.add_limb((side * 0.36, 0.0, 0.0), (side * 0.24, 0.42, -0.16), 0.18)
    arms.add_box((0.1, 0.95, 0.16), (0.2, 0.6, -0.16), colour=(0.35, 0.35, 0.38))  # rifle
    node = arms.node("arms")
    node.flattenStrong()
    node.reparentTo(root)
    node.setPos(0, 0, 1.42)  # shoulder
    return root


def build_wingdiver() -> NodePath:
    """The player. A grunt's silhouette with the jetpack that defines her.

    Root
      +- ``body``      torso, head, backpack, wings
      +- ``leg_l`` / ``leg_r``
      +- ``arms``      lance arm
      +- ``thruster``  two flame boxes, scaled by :class:`components.Flight`
    """
    root = build_soldier()
    body = root.find("body")

    wings = BoxBuilder(uv_scale=1.2)
    wings.add_box((0.34, 0.26, 0.5), (0.0, -0.3, 1.24), colour=(0.55, 0.6, 0.72))
    for side in (1.0, -1.0):
        # A swept plate either side of the pack. Two quads, one per face, so
        # it is lit from both sides rather than vanishing when seen from behind.
        tip = Vec3(side * 1.05, -0.85, 1.75)
        root_in = Vec3(side * 0.16, -0.42, 1.42)
        root_lo = Vec3(side * 0.16, -0.42, 1.05)
        tip_lo = Vec3(side * 0.95, -0.8, 1.2)
        normal = Vec3(0, -1, 0.35)
        normal.normalize()
        corners = (root_lo, tip_lo, tip, root_in)
        wings.add_quad(corners, normal, (0.62, 0.68, 0.82))
        wings.add_quad(tuple(reversed(corners)), -normal, (0.45, 0.5, 0.62))
    plate = wings.node("wings")
    plate.flattenStrong()
    plate.reparentTo(body)

    flame = BoxBuilder(uv_scale=1.0)
    for side in (1.0, -1.0):
        flame.add_box((0.2, 0.2, 0.7), (side * 0.22, -0.34, -0.45), colour=(1.0, 0.85, 0.5))
    thruster = flame.node("thruster")
    thruster.flattenStrong()
    thruster.reparentTo(root)
    thruster.setPos(0, 0, 1.0)
    thruster.setScale(0.001)  # invisible until she lights it
    return root


def build_projectile(kind: str) -> NodePath:
    """A bolt. Long and thin along +Y so it can be pointed with ``lookAt``."""
    sizes = {
        "lance": ((0.16, 2.4, 0.16), (1.0, 0.95, 0.6)),
        "tracer": ((0.09, 1.1, 0.09), (1.0, 0.8, 0.45)),
        "acid": ((0.42, 0.42, 0.42), (0.7, 1.0, 0.35)),
    }
    size, colour = sizes.get(kind, sizes["tracer"])
    builder = BoxBuilder(uv_scale=1.0)
    builder.add_box(size, (0, 0, 0), colour=colour)
    node = builder.node(f"proj-{kind}")
    node.flattenStrong()
    return node


def build_sprite(size: float = 1.0) -> NodePath:
    """A camera-facing quad, for sparks, blood and dust.

    Billboarded rather than a real particle system: the console had no such
    thing, and every explosion of the era was a handful of rotating quads.
    """
    maker = CardMaker("sprite")
    maker.setFrame(-size * 0.5, size * 0.5, -size * 0.5, size * 0.5)
    node = NodePath(maker.generate())
    node.setBillboardPointEye()
    return node


def swing(phase: float, amplitude: float, offset: float = 0.0) -> float:
    """Leg angle in degrees for a walk-cycle phase. Shared by ants and grunts."""
    return math.sin((phase + offset) * math.tau) * 38.0 * amplitude
