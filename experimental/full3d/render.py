"""The renderer: three processors that read the simulation and never write it.

The split this file exists to enforce is that :mod:`sim` has no idea any of
this is here. Gameplay entities carry a :class:`~components.Renderable`, which
is a string and a tint; this module owns the entity-to-``NodePath`` map, the
model prototypes, the chase camera and the HUD. Delete every processor here
and the game still runs -- that is what ``headlesstest.py`` does.

The city is handled differently from everything else: buildings never move, so
rather than one node each they are merged into one flattened geom per facade
style at startup. Two hundred boxes become four draw calls.
"""

from __future__ import annotations

import math

import esper
from panda3d.core import (
    NodePath,
    TextNode,
    TransparencyAttrib,
    Vec3,
    Vec4,
)

import models
import ps1
from components import (
    Building,
    Dead,
    Energy,
    Faction,
    Flight,
    Gait,
    Health,
    Intent,
    Lifetime,
    Projectile,
    Renderable,
    Transform,
    Velocity,
    Weapon,
)
from sim import ARENA, Sim, aim_vector, clamp

# How much of the gap between the camera's current and wanted position is
# closed per second. High enough to keep up with a dash, low enough to lag.
CAM_STIFFNESS = 9.0

# Largest polygon, in metres, allowed on a wall or a rooftop. Bounds how badly
# the affine texture mapping can swim -- see :func:`models.build_ground`.
# Roofs get the finer grid because they are the one large flat surface the
# player stands on, where a smear is most obvious.
WALL_CELL = 8.0
ROOF_CELL = 5.0


class SceneView:
    """Owns the scene graph: prototypes, the static city, and the actor nodes.

    One instance is shared by all three render processors, which is why they
    are constructed with it rather than each rebuilding their own state.
    """

    def __init__(self, base, sim: Sim, pipeline: ps1.Ps1Pipeline) -> None:
        self.base = base
        self.sim = sim
        self.pipeline = pipeline
        self.textures = pipeline.textures

        self.root = base.render.attachNewNode("actors")
        self.nodes: dict[int, NodePath] = {}

        self._prototypes: dict[str, NodePath] = {}
        self._build_prototypes()
        self._build_ground()
        self._build_city()

    # -- one-time construction --------------------------------------------

    def _build_prototypes(self) -> None:
        """Build one hidden copy of every dynamic model, to clone from."""
        tex = self.textures

        def register(kind: str, node: NodePath, texture, emissive: float = 0.0) -> None:
            node.setTexture(texture, 1)
            node.setShaderInput("emissive", emissive)
            node.detachNode()
            self._prototypes[kind] = node

        register("ant", models.build_ant(), tex.chitin)
        register("soldier", models.build_soldier(), tex.fatigues)
        register("wingdiver", models.build_wingdiver(), tex.armour)
        for kind in ("lance", "tracer", "acid"):
            register(kind, models.build_projectile(kind), tex.glow, emissive=1.0)
        for kind in ("spark", "blood", "dust"):
            register(kind, models.build_sprite(1.0), tex.white, emissive=1.0)

        # The Wing Diver's flame is emissive independently of her armour.
        proto = self._prototypes["wingdiver"]
        proto.find("**/thruster").setShaderInput("emissive", 1.0)

    def _build_ground(self) -> None:
        ground = models.build_ground(ARENA + 60.0, cell=6.0)
        ground.setTexture(self.textures.road, 1)
        ground.setShaderInput("emissive", 0.0)
        ground.reparentTo(self.base.render)
        self.ground = ground

    def _build_city(self) -> None:
        """Merge every building into one geom per facade style.

        Buildings are static, so their transforms can be baked into the
        vertices. The alternative -- 226 nodes each with its own transform --
        costs a per-frame cull and draw for scenery that will never move.

        Roofs are collected separately from walls. A rooftop is somewhere the
        Wing Diver actually lands and stands, and a roof wearing the window
        texture looks like a mistake the moment she does.
        """
        builders: dict[int, models.BoxBuilder] = {}
        roofs = models.BoxBuilder(uv_scale=0.25)  # one gravel tile per 4m

        for _ent, (transform, building) in esper.get_components(Transform, Building):
            builder = builders.setdefault(building.style, models.BoxBuilder(uv_scale=1.0 / 16.0))
            builder.add_box(
                (building.half_x * 2.0, building.half_y * 2.0, building.height),
                (transform.pos.x, transform.pos.y, building.height * 0.5),
                skip_bottom=True,
                skip_top=True,
                max_cell=WALL_CELL,
            )
            x0 = transform.pos.x - building.half_x
            x1 = transform.pos.x + building.half_x
            y0 = transform.pos.y - building.half_y
            y1 = transform.pos.y + building.half_y
            top = building.height
            uv = 0.25
            roofs.add_quad(
                (Vec3(x0, y0, top), Vec3(x1, y0, top), Vec3(x1, y1, top), Vec3(x0, y1, top)),
                Vec3(0, 0, 1),
                uvs=((x0 * uv, y0 * uv), (x1 * uv, y0 * uv), (x1 * uv, y1 * uv), (x0 * uv, y1 * uv)),
                steps=max(1, int(max(x1 - x0, y1 - y0) / ROOF_CELL)),
            )

        self.city = self.base.render.attachNewNode("city")
        for style, builder in builders.items():
            node = builder.node(f"style{style}")
            node.setTexture(self.textures.facades[style % len(self.textures.facades)], 1)
            node.setShaderInput("emissive", 0.0)
            node.reparentTo(self.city)
            node.flattenStrong()

        roof_node = roofs.node("roofs")
        roof_node.setTexture(self.textures.gravel, 1)
        roof_node.setShaderInput("emissive", 0.0)
        roof_node.reparentTo(self.city)
        roof_node.flattenStrong()

    # -- per-frame node management ----------------------------------------

    def acquire(self, ent: int, renderable: Renderable) -> NodePath | None:
        """Return this entity's node, cloning a prototype the first time."""
        node = self.nodes.get(ent)
        if node is not None:
            return node
        prototype = self._prototypes.get(renderable.kind)
        if prototype is None:
            return None
        node = prototype.copyTo(self.root)
        node.setColorScale(*renderable.tint, 1.0)
        node.setScale(renderable.scale)
        self.nodes[ent] = node
        return node

    def release(self, ent: int) -> None:
        node = self.nodes.pop(ent, None)
        if node is not None:
            node.removeNode()


class RenderProcessor(esper.Processor):
    """Pushes every entity's state into its ``NodePath`` once per frame.

    Also does the small amount of animation the game has: leg swings from
    :class:`~components.Gait`, a jetpack flame from :class:`~components.Flight`,
    a white flash from :class:`~components.Health`, and corpses toppling over.
    All of that is derived from simulation state that already exists, so
    nothing here has to be stepped or remembered between frames.
    """

    def __init__(self, view: SceneView) -> None:
        self.view = view

    def process(self, dt: float) -> None:
        view = self.view
        seen: set[int] = set()

        for ent, (transform, renderable) in esper.get_components(Transform, Renderable):
            if esper.has_component(ent, Building):
                continue  # baked into the static city geometry
            node = view.acquire(ent, renderable)
            if node is None:
                continue
            seen.add(ent)

            node.setPos(transform.pos)
            projectile = esper.try_component(ent, Projectile)
            if projectile is not None:
                self._orient_projectile(ent, node, transform)
            else:
                node.setHpr(transform.heading, transform.pitch, transform.roll)

            self._animate(ent, node, transform)

        for ent in [e for e in view.nodes if e not in seen]:
            view.release(ent)

    @staticmethod
    def _orient_projectile(ent: int, node: NodePath, transform: Transform) -> None:
        """Point a bolt along its own velocity. Models are built along +Y."""
        vel = esper.try_component(ent, Velocity)
        if vel is None or vel.vec.lengthSquared() < 1e-6:
            return
        node.lookAt(transform.pos + vel.vec)

    def _animate(self, ent: int, node: NodePath, transform: Transform) -> None:
        gait = esper.try_component(ent, Gait)
        if gait is not None:
            self._walk(node, gait)

        flight = esper.try_component(ent, Flight)
        if flight is not None:
            self._thruster(node, flight, ent)

        weapon = esper.try_component(ent, Weapon)
        if weapon is not None and not weapon.melee:
            arms = node.find("**/arms")
            if not arms.isEmpty():
                # Recoil: kick the arms up for the length of the muzzle flash.
                arms.setP(-6.0 - 40.0 * weapon.muzzle_flash / 0.06 if weapon.muzzle_flash else -6.0)

        dead = esper.try_component(ent, Dead)
        health = esper.try_component(ent, Health)
        renderable = esper.component_for_entity(ent, Renderable)
        if dead is not None:
            # Topple over the first third of a second, then just lie there.
            fall = clamp(1.0 - dead.timer / 0.6, 0.0, 1.0)
            node.setR(transform.roll + 95.0 * min(1.0, fall * 3.0))
            node.setColorScale(
                renderable.tint[0] * 0.45, renderable.tint[1] * 0.45, renderable.tint[2] * 0.45, 1
            )
        elif health is not None and health.hurt_flash > 0.0:
            node.setColorScale(2.4, 2.2, 2.2, 1.0)
        elif health is not None:
            node.setColorScale(*renderable.tint, 1.0)

        life = esper.try_component(ent, Lifetime)
        if life is not None:
            # Effects puff outward and shrink away over their lifetime.
            age = clamp(1.0 - life.remaining / max(life.remaining + 1e-4, 0.35), 0.0, 1.0)
            scale = renderable.scale * (0.35 + 1.9 * age) * max(0.05, life.remaining * 3.0)
            node.setScale(max(0.02, scale))

    @staticmethod
    def _walk(node: NodePath, gait: Gait) -> None:
        """Swing whichever limb nodes this model happens to have."""
        for name, offset in (("legs_l", 0.0), ("leg_l", 0.0), ("legs_r", 0.5), ("leg_r", 0.5)):
            limb = node.find(f"**/{name}")
            if not limb.isEmpty():
                limb.setP(models.swing(gait.phase, gait.amplitude, offset))

    @staticmethod
    def _thruster(node: NodePath, flight: Flight, ent: int) -> None:
        """Flare the jets when they are lit, and hide them when they are not."""
        flame = node.find("**/thruster")
        if flame.isEmpty():
            return
        if flight.dash_timer > 0.0:
            flame.setScale(1.5, 1.5, 2.4)
        elif flight.thrusting:
            wobble = 0.85 + 0.3 * math.sin(flight.dash_cd * 90.0 + ent)
            flame.setScale(1.0, 1.0, wobble)
        else:
            flame.setScale(0.001)


class CameraProcessor(esper.Processor):
    """A chase camera on a spring, that will not let a building block the shot.

    The boom points backward along the player's *aim*, not their velocity, so
    strafing and dashing sideways keep the camera facing where the lance will
    go -- which is what EDF does and what makes the class aimable at speed.
    """

    def __init__(self, view: SceneView) -> None:
        self.view = view
        self.position = Vec3(0, -10, 6)

    def process(self, dt: float) -> None:
        sim = self.view.sim
        base = self.view.base
        if sim.player < 0 or not esper.entity_exists(sim.player):
            return

        transform = esper.component_for_entity(sim.player, Transform)
        intent = esper.component_for_entity(sim.player, Intent)
        vel = esper.component_for_entity(sim.player, Velocity)

        focus = transform.pos + Vec3(0, 0, 1.45)
        speed = math.hypot(vel.vec.x, vel.vec.y)
        boom = 7.0 + clamp(speed * 0.11, 0.0, 3.5)

        direction = aim_vector(intent.aim_yaw, clamp(intent.aim_pitch, -55.0, 55.0))
        wanted = focus - direction * boom + Vec3(0, 0, 1.15)
        wanted.z = max(wanted.z, 0.9)  # never dip below the tarmac

        blend = clamp(CAM_STIFFNESS * dt, 0.0, 1.0)
        self.position += (wanted - self.position) * blend
        self.position = self._unblock(focus, self.position)

        base.camera.setPos(self.position)
        base.camera.lookAt(focus)

    def _unblock(self, focus: Vec3, position: Vec3) -> Vec3:
        """Pull the camera in until the line back to the player is clear.

        Marches the boom in eight steps and stops at the first sample inside a
        building. Crude, but the alternative is a swept-sphere test against the
        collision grid for a camera that is already smoothed to within a metre.
        """
        grid = self.view.sim.grid
        offset = position - focus
        length = offset.length()
        if length < 1e-4:
            return position
        step = offset / length
        clear = length
        for i in range(1, 9):
            distance = length * i / 8.0
            sample = focus + step * distance
            blocked = False
            for _ident, x0, y0, x1, y1, top in grid.query_aabb(
                sample.x - 0.4, sample.y - 0.4, sample.x + 0.4, sample.y + 0.4
            ):
                if x0 - 0.4 <= sample.x <= x1 + 0.4 and y0 - 0.4 <= sample.y <= y1 + 0.4:
                    if sample.z <= top + 0.4:
                        blocked = True
                        break
            if blocked:
                clear = max(1.8, distance - length / 8.0)
                break
        return focus + step * clear


class HudProcessor(esper.Processor):
    """The energy meter, and the four other numbers that matter.

    Drawn into the same 320x240 buffer as the world (see
    :class:`ps1.Ps1Pipeline`), so it is as chunky as everything else. The
    energy bar is the important one: it changes colour on overheat and pulses,
    because a Wing Diver who has not noticed she is empty is a Wing Diver
    falling off a roof.
    """

    def __init__(self, view: SceneView) -> None:
        self.view = view
        hud = view.pipeline.hud2d
        hud.setTransparency(TransparencyAttrib.MAlpha)

        self.energy_bar = self._bar(hud, -0.62, -0.86, 1.24, 0.062, (0.2, 0.85, 1.0))
        self.energy_back = self._bar(hud, -0.63, -0.868, 1.26, 0.078, (0.05, 0.07, 0.1))
        self.energy_back.setBin("fixed", 0)
        self.energy_bar.setBin("fixed", 1)

        self.health_back = self._bar(hud, -0.63, -0.948, 1.26, 0.06, (0.08, 0.04, 0.04))
        self.health_bar = self._bar(hud, -0.62, -0.94, 1.24, 0.044, (0.9, 0.35, 0.25))
        self.health_back.setBin("fixed", 0)
        self.health_bar.setBin("fixed", 1)

        # Drop shadows, because thin light text over a city at 320x240 is
        # unreadable the moment the player flies over anything pale.
        self.text = TextNode("hudtext")
        self.text.setTextColor(0.88, 0.96, 1.0, 1.0)
        self.text.setShadow(0.07, 0.07)
        self.text.setShadowColor(0.0, 0.0, 0.0, 1.0)
        self.text_np = hud.attachNewNode(self.text)
        self.text_np.setScale(0.085)
        self.text_np.setPos(-1.29, 0, 0.85)

        self.warning = TextNode("hudwarn")
        self.warning.setTextColor(1.0, 0.45, 0.3, 1.0)
        self.warning.setShadow(0.07, 0.07)
        self.warning.setShadowColor(0.0, 0.0, 0.0, 1.0)
        self.warning.setAlign(TextNode.ACenter)
        self.warning_np = hud.attachNewNode(self.warning)
        self.warning_np.setScale(0.09)
        self.warning_np.setPos(0.0, 0, -0.76)

        self._crosshair(hud)
        self._blink = 0.0

    @staticmethod
    def _bar(parent: NodePath, x: float, y: float, w: float, h: float, colour) -> NodePath:
        """A solid rectangle whose X scale is the value it displays."""
        from panda3d.core import CardMaker

        maker = CardMaker("bar")
        maker.setFrame(0.0, 1.0, 0.0, 1.0)
        node = parent.attachNewNode(maker.generate())
        node.setPos(x, 0, y)
        node.setScale(w, 1.0, h)
        node.setColor(Vec4(colour[0], colour[1], colour[2], 1.0))
        return node

    def _crosshair(self, parent: NodePath) -> None:
        from panda3d.core import CardMaker

        maker = CardMaker("cross")
        for w, h in ((0.055, 0.006), (0.006, 0.055)):
            maker.setFrame(-w, w, -h, h)
            arm = parent.attachNewNode(maker.generate())
            arm.setColor(0.9, 1.0, 0.9, 0.85)
            arm.setPos(0, 0, 0)

    def process(self, dt: float) -> None:
        sim = self.view.sim
        self._blink = (self._blink + dt) % 1.0

        if sim.player >= 0 and esper.entity_exists(sim.player):
            energy = esper.component_for_entity(sim.player, Energy)
            health = esper.component_for_entity(sim.player, Health)
            self.energy_bar.setScale(max(0.001, 1.24 * energy.fraction), 1.0, 0.062)
            if energy.empty:
                # Overheated: red, and flashing, because you cannot fly.
                bright = 0.55 + 0.45 * (self._blink < 0.5)
                self.energy_bar.setColor(bright, 0.18 * bright, 0.12 * bright, 1.0)
                self.warning.setText("ENERGY DEPLETED")
            else:
                self.energy_bar.setColor(0.2, 0.85, 1.0, 1.0)
                self.warning.setText("")
            self.health_bar.setScale(
                max(0.001, 1.24 * (health.current / health.maximum)), 1.0, 0.044
            )
        else:
            self.energy_bar.setScale(0.001, 1.0, 0.062)
            self.health_bar.setScale(0.001, 1.0, 0.044)
            self.warning.setText("K.I.A.")

        stats = sim.stats
        self.text.setText(
            f"WAVE {stats.wave}\n"
            f"ANTS {len(sim.sides[Faction.BUGS])}\n"
            f"SQUAD {max(0, len(sim.sides[Faction.EDF]) - 1)}\n"
            f"KILLS {stats.ants_killed}"
        )


def attach(base, sim: Sim, pipeline: ps1.Ps1Pipeline) -> SceneView:
    """Bolt the renderer onto an existing simulation.

    Called once, after the world is generated. The three processors are added
    at low priority so they run after everything that might move something.
    """
    sim.activate()
    view = SceneView(base, sim, pipeline)
    esper.add_processor(RenderProcessor(view), priority=5)
    esper.add_processor(CameraProcessor(view), priority=4)
    esper.add_processor(HudProcessor(view), priority=3)
    return view
