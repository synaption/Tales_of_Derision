"""The juice workbench, rendered on the graphics card.

Same bench, same sim, same panel. The renderer is the only thing that changed,
and the import list is the argument for why the split was worth having:

    from juicefx import ...        # every effect, as maths. Unchanged.
    from rogue_juice import World  # the sim, the toggles, the sliders. Unchanged.
    from audiofx import SoundBank  # unchanged.

Not one line of `juicefx.py` knows this file exists. `Body` already carried
`ox, oy, sx, sy, angle, alpha, flash` and a lag vector, which turns out to be a
per-instance attribute block; the springs that drove ten CPU-drawn bands now
drive a continuous displacement field, and nothing about the springs changed.

What is different to look at, in the order it is worth looking at it:

* **the slime** deforms continuously instead of in ten strips, so it can smear
  as far as you like without coming apart -- there is no `link` constraint here
  because there is nothing to hold together;
* **creatures are lit**, with normals generated from their own silhouettes, so
  a torch rakes across them and a hit flash shades the room;
* **sprites move sub-pixel and stay crisp**, which the software bench has to
  offer as a three-way compromise;
* **the floor ripples continuously**, as a displacement of the texture rather
  than a repaint of the tiles it passes;
* **one draw call** for every sprite, spark, shadow, decal, ring, arc and cut.

The panel is drawn by the software renderer into an offscreen surface and
uploaded as a texture, which keeps every slider, blurb and scroll behaviour
identical and costs one upload a frame. Text layout is the one thing pygame
does better than a weekend of shader work.

    python3 rogue_juice_gl.py
    python3 rogue_juice_gl.py --headless out.png
"""

from __future__ import annotations

import math
import os
import sys
import time

if "--headless" in sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import moderngl
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glfx  # noqa: E402
import rogue_juice as rj  # noqa: E402
import tiles  # noqa: E402
from juicefx import ambient_offset, clamp, shadow_of  # noqa: E402

try:
    import audiofx
    HAVE_AUDIO = True
except Exception:                                    # pragma: no cover
    audiofx = None                                   # type: ignore
    HAVE_AUDIO = False


TILE = rj.TILE
VIEW_W, VIEW_H = rj.VIEW_W, rj.VIEW_H
WIN_W, WIN_H = rj.WIN_W, rj.WIN_H
GRID_W, GRID_H = rj.GRID_W, rj.GRID_H
PX = rj.PX
FPS = rj.FPS


# ---------------------------------------------------------------------------
# The extra controls the card makes possible
# ---------------------------------------------------------------------------
#
# Added to the existing registry rather than replacing it, so the panel, the
# blurbs, the scrolling and the drag handling all carry over untouched. Two
# software-only controls are dropped because the hardware makes the question
# they answer disappear.

GL_TOGGLES = [
    ("normals", "normal-mapped sprites", "screen",
     "Light the creatures with normals generated from their own silhouettes, "
     "so a torch rakes across a body instead of uniformly brightening it. No "
     "new art: a blurred alpha mask's gradient is the normal of an inflated "
     "shape."),
    ("softbody", "continuous soft body", "movement",
     "Deform the slime as a field rather than as ten strips. Off, it falls "
     "back to a rigid drag -- which is what a blit can do, and the reason the "
     "CPU bench needs bands and a surface-tension clamp at all."),
    ("sharp", "sharp bilinear", "screen",
     "Flat inside a texel, blended across the boundary over one screen pixel. "
     "Crisp at rest and smooth at any sub-pixel speed. Off is plain bilinear, "
     "which is the mush that made everyone snap to whole pixels in the first "
     "place."),
    ("grade", "impact grade", "screen",
     "Push the whole frame warm for a few frames on contact. A colour shift is "
     "a hit confirmation the eye reads without noticing it."),
    ("scanlines", "scanlines", "screen",
     "A CRT line pattern. Free here, and impossible on the CPU without walking "
     "every pixel.", False),
    ("refract", "heat refraction", "screen",
     "Bend the frame around each torch. The software bench re-blits ninety rows "
     "per torch to fake this; here every fragment simply asks where to sample."),
]

GL_PARAMS = [
    ("soft_power", "soft-body falloff", "movement", 1.7, 0.4, 5.0,
     "Shape of the displacement gradient across the body. 1 is a linear smear, "
     "high numbers keep the front rigid and let only the tail string out.",
     "{:.2f}"),
    ("sharpness", "filter sharpness", "screen", 1.0, 0.0, 1.0,
     "0 is plain bilinear, 1 is sharp bilinear. Drag it down while something "
     "is moving slowly to see exactly what the filter is buying.", "{:.2f}"),
    ("normal_depth", "normal depth", "screen", 1.0, 0.0, 2.0,
     "How far the generated normals turn away at a sprite's rim. Past about "
     "1.5 the creatures start to look like foil balloons.", "{:.2f}"),
    ("light_height", "light height", "screen", 82.0, 2.0, 240.0,
     "How far the lights float above the floor plane. This is the number that "
     "decides whether a torch lights a room or lights a coin: low is a raking, "
     "dramatic light that dies within a tile, high flattens everything out.",
     "{:.0f}"),
    ("bloom_knee", "bloom knee", "screen", 0.45, 0.01, 2.0,
     "How softly the threshold lets a pixel in. A hard knee makes bloom "
     "flicker on moving highlights.", "{:.2f}"),
    ("grade_amt", "grade strength", "screen", 0.5, 0.0, 1.0,
     "How far the frame is pushed warm on impact.", "{:.2f}"),
    ("scanline_amt", "scanline depth", "screen", 0.14, 0.0, 0.6,
     "", "{:.2f}"),
]

#: Software-only controls that the hardware makes meaningless.
#: `pixel_mode` existed because rounding a blit destination was the only
#: control there was; `light warmth` existed because a multiply cannot brighten.
DROP_CHOICES = ["pixel_mode"]
DROP_PARAMS = ["light_warm"]


def gl_juice() -> rj.Juice:
    """The software registry plus the hardware's own controls."""
    juice = rj.Juice()
    for spec in GL_TOGGLES:
        key, label, group, blurb = spec[:4]
        on = spec[4] if len(spec) > 4 else True
        juice.toggles[key] = rj.Toggle(key, label, group, blurb, on)
    for spec in GL_PARAMS:
        juice.params.add(*spec)
    for key in DROP_CHOICES:
        juice.choices.pop(key, None)
    for key in DROP_PARAMS:
        juice.params.params.pop(key, None)
    # Lighting is cheap enough to leave on now: on the CPU it was 3.8ms and
    # defaulted off, here it is part of the same pass that was going to run
    # anyway.
    juice.toggles["light"].on = True
    # Bigger than the software default, which was sized around a light pass
    # that cost four milliseconds. This one is part of a pass that was going to
    # run anyway, so the radius can be what looks right.
    juice.params["light_radius"].value = 7.5
    juice.params["light_radius"].default = 7.5
    return juice


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------


class GLRenderer:
    """Floor, one instanced batch, a bloom pyramid, one composite, one UI blit."""

    #: Sprites packed into the atlas, plus the digits for damage numbers --
    #: which are drawn as instanced quads in the world so they shake with it,
    #: rather than being pasted on flat afterwards.
    DIGITS = "0123456789"

    #: How much the baked floor is brightened to serve as an albedo. See
    #: `_build_floor`: a palette chosen to look right unlit is not one.
    ALBEDO_GAIN = 2.6

    #: Panel redraws per second. See `draw_ui` -- text layout is the most
    #: expensive thing left in the frame and the least in need of 60Hz.
    UI_HZ = 30.0

    def __init__(self, ctx, target, world: rj.World, headless: bool = False):
        self.ctx = ctx
        self.target = target
        self.headless = headless
        self.frame_ms = 0.0
        self.upload_ms = 0.0
        self._ui_drawn = 0.0
        self._ui_mouse = None
        # Set by anything that can change what the panel says -- which is any
        # input at all. Input is rare and a stale panel is the one thing a
        # throttle must never cause, so the flag is cheaper than being clever
        # about which events matter.
        self.ui_dirty = True

        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        # Premultiplied alpha: the fragment shader outputs `rgb * a`, so a
        # bright particle adds light instead of merely covering what is behind
        # it, and a stack of sparks reads as a glow rather than as paint.
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

        self.sheet = tiles.SpriteSheet(TILE)
        self.atlas = self._build_atlas()
        self.digit_uv = {ch: self.atlas.uv[f"digit_{ch}"] for ch in self.DIGITS
                         if f"digit_{ch}" in self.atlas.uv}

        self.quad_prog = ctx.program(vertex_shader=glfx._inject(glfx.QUAD_VS),
                                     fragment_shader=glfx._inject(glfx.QUAD_FS))
        self.floor_prog = ctx.program(vertex_shader=glfx._inject(glfx.FLOOR_VS),
                                      fragment_shader=glfx._inject(glfx.FLOOR_FS))
        self.batch = glfx.QuadBatch(ctx, self.quad_prog)

        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]],
                           dtype="f4")
        self.floor_vbo = ctx.buffer(corners.tobytes())
        self.floor_vao = ctx.vertex_array(
            self.floor_prog, [(self.floor_vbo, "2f", "in_corner")])

        self.post = glfx.PostChain(ctx, (VIEW_W, VIEW_H))

        # The panel and the log, drawn by the software renderer. Constructing it
        # costs a few surfaces we never use; what it buys is that every row,
        # blurb, slider and scroll behaviour is literally the same code.
        self.ui = rj.Renderer()
        self.ui_surface = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        self.ui_tex = ctx.texture((WIN_W, WIN_H), 4)
        self.ui_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)  # already 1:1
        self.ui_tex.repeat_x = self.ui_tex.repeat_y = False

        self._build_floor(world)

    # -- setup ------------------------------------------------------------
    def _build_atlas(self):
        """Sprites and digits in one page.

        The digits are here rather than in the UI overlay because damage
        numbers have to live in the *world*: drawn flat on top afterwards they
        would sit perfectly still while the screen shook underneath them, which
        reads worse than having no numbers at all. Putting them in the same
        atlas keeps the whole frame at one draw call.
        """
        font = pygame.font.Font(pygame.font.match_font(
            "dejavusansmono,couriernew,consolas,monospace"), 40)
        font.set_bold(True)
        digits = {f"digit_{ch}": font.render(ch, True, (255, 255, 255))
                  for ch in self.DIGITS}
        return glfx.Atlas(self.ctx, self.sheet, list(tiles.SPRITES),
                          tiles.SPRITES, extra=digits)

    def _build_floor(self, world: rj.World):
        """Bake the arena once, with its tufts, and derive a normal map from it.

        On the CPU the bake exists to avoid repainting five hundred rounded
        rects a frame, and the ripple has to punch holes in it. Here it is
        simply a texture: the ripple and the ambient sway are displacements of
        the coordinate that samples it, so nothing is ever repainted at all --
        and the sway can cover the whole floor instead of a few dozen props,
        because it costs the same either way.
        """
        surf = self.ui._bake_floor(world)
        # Tufts baked in, since the whole floor now sways as one.
        for kind, wx, wy, phase in world.props:
            if kind != "tuft":
                continue
            for i in (-3, 0, 3):
                pygame.draw.line(surf, (44, 58, 50), (wx + i, wy),
                                 (wx + i, wy - 6 - abs(i) * 0.4), 1)

        rgb = np.frombuffer(pygame.image.tobytes(surf, "RGB", False),
                            dtype=np.uint8).reshape(surf.get_height(),
                                                    surf.get_width(), 3)
        # The floor palette was picked to look right with no lighting on it,
        # which makes it far too dark to be an *albedo*: at 28/255 a torch can
        # multiply it by two and it is still black. Under a lighting model the
        # texture stops being the finished pixels and becomes what the surface
        # reflects, so it is scaled up here and the light is what brings it
        # back down. `u_unlit_scale` puts it back when lighting is switched
        # off, so the toggle still compares like with like.
        bright = np.clip(rgb.astype(np.float32) * self.ALBEDO_GAIN, 0, 255)
        self.floor_tex = self.ctx.texture(surf.get_size(), 3,
                                          bright.astype(np.uint8).tobytes())
        self.floor_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.floor_tex.repeat_x = self.floor_tex.repeat_y = False
        normals = glfx.normal_from_luminance(rgb, strength=2.2, blur=1)
        self.floor_normal = self.ctx.texture(surf.get_size(), 3, normals.tobytes())
        self.floor_normal.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.floor_size = surf.get_size()

    # -- per frame --------------------------------------------------------
    def camera_uniforms(self, world: rj.World):
        """Shake, kick, roll and zoom, applied to the geometry.

        The software bench has to `rotozoom` the finished frame, which resamples
        every pixel and drags the frame's own edges into view. Here they are
        four numbers in the vertex shader: the world is simply drawn from a
        different place, so there is no resampling and no edge to hide.
        """
        j = world.juice
        dx = dy = roll = 0.0
        if j.on("shake"):
            sx, sy, sr = world.trauma.offset(
                max_px=j.p("shake_px") * PX * j.intensity,
                max_deg=2.4 * j.intensity)
            dx, dy, roll = sx, sy, sr
        if j.on("kick"):
            dx += world.kick[0].value
            dy += world.kick[1].value
        if j.on("tilt"):
            roll += world.camera.tilt.value
        zoom = clamp(world.camera.zoom.value if j.on("zoom") else 1.0, 0.85, 1.3)
        return {
            "u_camera": (world.camera.x, world.camera.y),
            "u_half": (VIEW_W * 0.5, VIEW_H * 0.5),
            "u_roll": math.radians(roll),
            "u_zoom": zoom,
            "u_shake": (dx, dy),
        }

    def lights_of(self, world: rj.World):
        """(x, y, radius, strength) for everything giving off light."""
        j = world.juice
        if not j.on("light"):
            return []
        radius = j.p("light_radius") * TILE
        out = [(*world.player.world_pos(), radius, 1.0)]
        flicker = world.elapsed
        for kind, wx, wy, phase in world.props:
            if kind == "torch":
                # A torch that is perfectly steady is a lamp. Two sines at
                # unrelated rates keep it from looking like a pulse.
                f = 0.86 + 0.09 * math.sin(flicker * 7.3 + phase) \
                    + 0.05 * math.sin(flicker * 17.1 + phase * 2.0)
                out.append((wx, wy, radius * 0.85, f))
        for x, y, s, life, maxl in world.lights:
            out.append((x, y, radius * 1.1, s * (life / maxl)))
        return out[:glfx.MAX_LIGHTS]

    def _set_lighting(self, program, world: rj.World, lights):
        j = world.juice
        n = glfx.write_vec4_array(program, "u_lights", lights, glfx.MAX_LIGHTS)
        if "u_light_count" in program:
            program["u_light_count"] = n
        if "u_light_color" in program:
            program["u_light_color"] = (1.0, 0.86, 0.66)
        if "u_ambient" in program:
            a = j.p("light_ambient") if j.on("light") else 1.0
            program["u_ambient"] = (a, a * 1.01, a * 1.08)   # a cool shadow
        if "u_light_height" in program:
            program["u_light_height"] = j.p("light_height")
        if "u_lit" in program:
            program["u_lit"] = 1.0 if j.on("light") else 0.0
        if "u_sharpness" in program:
            program["u_sharpness"] = j.p("sharpness") if j.on("sharp") else 0.0

    def draw_floor(self, world: rj.World, lights):
        j = world.juice
        p = self.floor_prog
        for key, value in self.camera_uniforms(world).items():
            p[key] = value
        p["u_extent"] = (GRID_W * TILE, GRID_H * TILE)
        p["u_floor_size"] = self.floor_size
        p["u_time"] = world.elapsed
        p["u_ambient_sway"] = (j.p("ambient_amt") * j.intensity
                               if j.on("ambient") else 0.0)
        p["u_unlit_scale"] = 1.0 / self.ALBEDO_GAIN

        waves, lens = [], []
        if j.on("ripple"):
            for imp in world.fx.impacts[:glfx.MAX_WAVES]:
                waves.append((imp.x, imp.y, imp.speed * imp.t,
                              imp.strength * (1.0 - imp.t) ** 2))
                lens.append(imp.wavelength)
        n = glfx.write_vec4_array(p, "u_waves", waves, glfx.MAX_WAVES)
        glfx.write_float_array(p, "u_wave_len", lens, glfx.MAX_WAVES)
        p["u_wave_count"] = n

        self._set_lighting(p, world, lights)
        self.floor_tex.use(0)
        self.floor_normal.use(1)
        p["u_floor"] = 0
        p["u_floor_normal"] = 1
        self.floor_vao.render(mode=moderngl.TRIANGLE_STRIP)

    # -- the batch --------------------------------------------------------
    def fill_batch(self, world: rj.World):
        """Everything else, in painter's order, into one instance buffer."""
        b = self.batch
        j = world.juice
        b.clear()
        white = self.atlas.white

        if j.on("decals"):
            for d in world.decals.decals:
                b.add(d.x, d.y, d.radius * 2.4, d.radius * 2.4 * d.squash,
                      uv=white, color=[c / 255.0 for c in d.color],
                      alpha=0.85 * d.alpha, shape=glfx.SHAPE_DISC,
                      params=(0.55, 0, 0, 0))

        # Torch flames. The tufts are baked into the floor because the floor
        # itself sways now; a flame has to be drawn because it also flickers.
        for kind, wx, wy, phase in world.props:
            if kind != "torch":
                continue
            t = world.elapsed * 9.0 + phase
            wob = math.sin(t) * 1.2
            if j.on("ambient"):
                ox, oy = ambient_offset(wx, wy, world.elapsed + phase,
                                        j.p("ambient_amt") * j.intensity)
                wob += ox
            b.add(wx + wob, wy - 8.0, 9.0, 15.0, uv=white,
                  color=(1.6, 0.8, 0.25), alpha=0.9, shape=glfx.SHAPE_DISC,
                  params=(0.85, 0, 0, 0))
            b.add(wx + wob * 0.6, wy - 9.0, 4.5, 9.0, uv=white,
                  color=(2.4, 2.0, 1.2), alpha=1.0, shape=glfx.SHAPE_DISC,
                  params=(0.7, 0, 0, 0))

        if j.on("shadow"):
            for e in world.entities:
                if e.dying:
                    continue
                ox, size, alpha = shadow_of(e.body, base=j.p("shadow_size"))
                alpha *= j.p("shadow_alpha") * j.intensity
                if alpha <= 0.02:
                    continue
                b.add((e.body.tx + 0.5 + ox) * TILE,
                      (e.body.ty + 0.5) * TILE + TILE * 0.36,
                      TILE * size * 1.3, TILE * size * 0.55, uv=white,
                      color=(0.0, 0.0, 0.0), alpha=alpha,
                      shape=glfx.SHAPE_DISC, params=(0.7, 0, 0, 0))

        for c in world.corpses:
            uv = self.atlas.uv.get(c.sprite)
            if uv is None:
                continue
            b.add(c.x, c.y, TILE, TILE, rot=math.radians(c.angle), uv=uv,
                  color=self._tint(c), alpha=c.alpha)

        for e in world.entities:
            self._add_trail(b, e)
        for e in world.entities:
            self._add_tail(b, world, e)
            self._add_body(b, world, e)

        if j.on("shockwave"):
            for s in world.fx.shockwaves:
                r = s.radius
                if r < 2:
                    continue
                b.add(s.x, s.y, r * 2, r * 2, uv=white,
                      color=[c / 255.0 * 1.6 for c in s.color],
                      alpha=0.85 * s.alpha, shape=glfx.SHAPE_RING,
                      params=(max(0.04, s.thickness / max(r, 1.0)), 0, 0, 0))

        for s in world.fx.slashes:
            reach = s.radius * rj.lerp(0.55, 1.15, s.progress)
            b.add(s.x, s.y, (reach + 8) * 2, (reach + 8) * 2,
                  rot=-math.radians(s.angle), uv=white,
                  color=(1.7, 1.7, 1.7), alpha=0.95 * s.alpha,
                  shape=glfx.SHAPE_ARC,
                  params=(reach / (reach + 8), math.radians(s.sweep),
                          0.10 + 0.25 * (1.0 - s.t), 0.0))

        for c in world.fx.cuts:
            b.add(c.x, c.y, c.length, c.thickness * 2.4,
                  rot=-math.radians(c.angle), uv=white,
                  color=[v / 255.0 for v in c.color], alpha=c.alpha,
                  shape=glfx.SHAPE_CUT, params=(c.progress, 0, 0, 0))

        for p in world.fx.particles.particles:
            size = p.size * (1.0 - p.t) * 2.2
            if size < 0.6:
                continue
            b.add(p.x, p.y, size, size, uv=white,
                  color=[v / 255.0 * 1.5 for v in p.color],
                  alpha=(1.0 - p.t) ** 0.6, shape=glfx.SHAPE_DISC,
                  params=(0.85, 0, 0, 0))

        if j.on("numbers"):
            self._add_numbers(b, world)

    def _tint(self, e):
        return [c / 255.0 for c in e.color] if e.tint else (1.0, 1.0, 1.0)

    def _add_trail(self, b, e):
        uv = self.atlas.uv.get(e.sprite)
        if uv is None:
            return
        for g in e.trail.ghosts:
            b.add(g.x, g.y, TILE * g.sx, TILE * g.sy, rot=-math.radians(g.angle),
                  uv=uv, color=self._tint(e), alpha=0.42 * g.alpha,
                  params=(1.0, 0, 0, 0))       # unlit: an echo is not a body

    def _add_tail(self, b, world, e):
        if e.tail is None or not world.juice.on("tail"):
            return
        pts = e.tail.points(e.body)
        white = self.atlas.white
        col = [c / 255.0 for c in e.tail_color]
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            x0, y0, x1, y1 = x0 * TILE, y0 * TILE, x1 * TILE, y1 * TILE
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length < 0.5:
                continue
            t = i / max(1, len(pts) - 2)
            # Overlapping hard-edged capsules rather than soft blobs: a soft
            # ellipse per link draws as a row of dots, which reads as a string
            # of beads instead of a tail.
            b.add((x0 + x1) * 0.5, (y0 + y1) * 0.5, length * 2.0,
                  max(2.0, (1.0 - t) ** 0.8 * TILE * 0.42),
                  rot=math.atan2(dy, dx), uv=white, color=col,
                  alpha=1.0, shape=glfx.SHAPE_DISC, params=(0.35, 0, 0, 0))

    def _add_body(self, b, world, e):
        uv = self.atlas.uv.get(e.sprite)
        if uv is None:
            return
        body = e.body
        j = world.juice
        wx, wy = e.world_pos()

        # The soft body. `Jelly` hands the CPU renderer ten band offsets; the
        # same numbers describe a field, so what the shader wants is the *worst*
        # of them -- how far the trailing edge is behind -- and the axis it lies
        # along. Everything between the two edges is then interpolated by the
        # gradient rather than quantised into strips.
        lag = (0.0, 0.0)
        if body.shear and j.on("softbody") and j.on("jelly"):
            head, tail = body.shear[-1], body.shear[0]
            fx, fy = body.facing
            if abs(fx) >= abs(fy) and fx < 0:
                head, tail = tail, head
            elif abs(fy) > abs(fx) and fy < 0:
                head, tail = tail, head
            span = (tail - head) * TILE
            if body.shear_axis == 0:
                lag = (span, 0.0)
            else:
                lag = (0.0, span)
        elif body.shear and j.on("jelly"):
            # The honest fallback: a rigid drag, which is all a blit can do
            # without cutting the sprite up.
            off = sum(body.shear) / len(body.shear) * TILE
            lag = (off, 0.0) if body.shear_axis == 0 else (0.0, off)
            wx += lag[0] * 0.5
            wy += lag[1] * 0.5
            lag = (0.0, 0.0)

        # Mirroring is a swap of the *u* bounds only. Swapping the whole pair
        # flips both axes and stands the creature on its head, which is the one
        # bug this line will ever have.
        (u0, v0), (u1, v1) = uv
        cell = ((u1, v0), (u0, v1)) if e.flip else uv
        b.add(wx, wy, TILE * body.sx, TILE * body.sy,
              rot=-math.radians(body.angle), uv=cell,
              color=self._tint(e), alpha=body.alpha, flash=body.flash,
              lag=lag, params=(0.0, 0.0 if j.on("normals") else 1.0, 0, 0))

    def _add_numbers(self, b, world):
        """Damage numbers as instanced digit quads, so they live in the world.

        Drawn into the UI overlay instead, they would hold perfectly still
        while the screen shook underneath them, which reads worse than having
        no numbers at all.
        """
        if not self.digit_uv:
            return
        for f in world.fx.floaters.floaters:
            size = TILE * 0.8 * f.scale
            text = f.text
            x = f.x - (len(text) - 1) * size * 0.28
            for ch in text:
                uv = self.digit_uv.get(ch)
                if uv is not None:
                    b.add(x, f.y, size, size, uv=uv,
                          color=[c / 255.0 * 1.35 for c in f.color],
                          alpha=clamp(f.alpha), params=(1.0, 0, 0, 0))
                x += size * 0.56

    # -- UI ---------------------------------------------------------------
    def draw_ui(self, world: rj.World, mouse, force: bool = False):
        """Redraw the panel and the log, and upload them.

        Throttled, because this is the most expensive thing left in the frame
        by a wide margin: laying out ninety rows of text in pygame and encoding
        a 1180x600 surface costs about 1.2ms, against roughly 1ms for
        *everything else in the renderer put together*. A panel does not need
        sixty updates a second -- nothing on it moves faster than the eye can
        read -- so it gets thirty, and the mouse forces one whenever it moves so
        hovering and dragging stay immediate.
        """
        now = time.perf_counter()
        moved = mouse != self._ui_mouse
        if not (force or moved or self.ui_dirty
                or now - self._ui_drawn >= 1.0 / self.UI_HZ):
            return
        self._ui_drawn = now
        self.ui_dirty = False
        self._ui_mouse = mouse
        self.ui_surface.fill((0, 0, 0, 0))
        self.ui.frame_ms = self.frame_ms
        self.ui.draw_hud(self.ui_surface, world)
        self.ui.draw_panel(self.ui_surface, world, mouse)
        self.ui_tex.write(pygame.image.tobytes(self.ui_surface, "RGBA", True))
        self.upload_ms += ((time.perf_counter() - now) * 1000.0 - self.upload_ms) * 0.08

    # -- the frame --------------------------------------------------------
    def render(self, world: rj.World, mouse=(0, 0)):
        j = world.juice
        lights = self.lights_of(world)

        self.post.scene.use()
        self.ctx.clear(*[c / 255.0 for c in rj.BG], 1.0)
        self.draw_floor(world, lights)

        p = self.quad_prog
        for key, value in self.camera_uniforms(world).items():
            p[key] = value
        p["u_atlas_size"] = self.atlas.size
        p["u_soft_power"] = j.p("soft_power")
        p["u_soft_pinch"] = j.p("jelly_stretch") * j.intensity
        self._set_lighting(p, world, lights)
        self.atlas.albedo.use(0)
        self.atlas.normal.use(1)
        p["u_albedo"] = 0
        p["u_normal"] = 1
        self.fill_batch(world)
        self.batch.render()

        bloom = j.p("bloom_amt") * j.intensity if j.on("bloom") else 0.0
        if bloom > 0.001:
            self.post.build_bloom(j.p("bloom_thresh") / 255.0, j.p("bloom_knee"))

        self.draw_ui(world, mouse)

        haze = []
        if j.on("refract") or j.on("haze"):
            cam = self.camera_uniforms(world)
            for kind, wx, wy, phase in world.props:
                if kind != "torch":
                    continue
                sx = (wx - cam["u_camera"][0]) * cam["u_zoom"] + VIEW_W * 0.5
                sy = (wy - cam["u_camera"][1]) * cam["u_zoom"] + VIEW_H * 0.5
                if -200 < sx < VIEW_W + 200 and -200 < sy < VIEW_H + 200:
                    haze.append((sx, sy, TILE * 5.0, 1.0))
        n = glfx.write_vec4_array(self.post.comp_prog, "u_haze", haze,
                                  glfx.MAX_LIGHTS)

        self.post.composite(self.target, self.ui_tex, {
            "u_bloom_amount": bloom,
            "u_rgb_split": world.rgb_split if j.on("rgbsplit") else 0.0,
            "u_texel": (1.0 / VIEW_W, 1.0 / VIEW_H),
            "u_flash": clamp(world.screen_flash) if j.on("scrflash") else 0.0,
            "u_flash_color": (1.0, 0.96, 0.88),
            "u_vignette": j.p("vig_base") / 200.0 if j.on("vignette") else 0.0,
            "u_vignette_pulse": (clamp(world.vignette_pulse) * 0.7
                                 if j.on("vignette") else 0.0),
            "u_scanlines": j.p("scanline_amt") if j.on("scanlines") else 0.0,
            "u_grade": (clamp(world.screen_flash * 4.0) * j.p("grade_amt")
                        if j.on("grade") else 0.0),
            "u_time": world.elapsed,
            "u_haze_count": n,
            "u_haze_amount": (j.p("haze_amt") * j.intensity
                              if (j.on("refract") or j.on("haze")) else 0.0),
            "u_view": (float(VIEW_W), float(VIEW_H)),
        })

    def release(self):
        self.batch.release()
        self.post.release()
        self.atlas.release()
        self.floor_tex.release()
        self.floor_normal.release()
        self.floor_vao.release()
        self.floor_vbo.release()
        self.ui_tex.release()
        self.quad_prog.release()
        self.floor_prog.release()


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def handle_event(event, world: rj.World, renderer: GLRenderer) -> bool:
    """The software handler, with the panel pointed at the UI renderer.

    The panel is the software renderer's, so its own hit testing, scrolling and
    drag handling apply unchanged -- the only difference is that the surface it
    drew onto is now a texture.
    """
    renderer.ui_dirty = True                     # any input can change the panel
    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
        return True                              # pixel mode has no meaning here
    return rj.handle_event(event, world, renderer.ui)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def build(headless: bool = False, audio=True):
    """Context, world and renderer, in the order that has to happen."""
    bank = None
    if audio and HAVE_AUDIO and not headless:
        bank = audiofx.SoundBank()
        bank.init_mixer()
        bank.load_all()

    pygame.init()
    pygame.font.init()
    if headless:
        pygame.display.set_mode((WIN_W, WIN_H))
        ctx = glfx.create_context(headless=True)
        target = ctx.simple_framebuffer((WIN_W, WIN_H), components=4)
    else:
        glfx.open_gl_window((VIEW_W, VIEW_H))
        ctx = glfx.create_context()
        target = ctx.screen
        target.viewport = (0, 0, VIEW_W, VIEW_H)

    world = rj.World(gl_juice(), bank)
    renderer = GLRenderer(ctx, target, world, headless=headless)
    return ctx, target, world, renderer


def run():
    # The window is the *view*, and the panel is drawn into it as an overlay,
    # so unlike the software bench there is no second region to size: the GL
    # surface is the whole window.
    bank = None
    if HAVE_AUDIO:
        bank = audiofx.SoundBank()
        bank.init_mixer()
        bank.load_all()

    pygame.init()
    pygame.font.init()
    pygame.joystick.init()
    glfx.open_gl_window((WIN_W, WIN_H))
    ctx = glfx.create_context()
    glfx.restart_on_the_card(ctx.info["GL_RENDERER"])

    target = ctx.screen
    target.viewport = (0, 0, WIN_W, WIN_H)
    world = rj.World(gl_juice(), bank)
    if pygame.joystick.get_count():               # pragma: no cover
        world.pad = pygame.joystick.Joystick(0)
        world.pad.init()
    renderer = GLRenderer(ctx, target, world)
    print(f"rendering on {ctx.info['GL_RENDERER']}")

    pygame.key.set_repeat(180, 70)
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = min(clock.tick(FPS) / 1000.0, 1.0 / 20.0)
        for event in pygame.event.get():
            if not handle_event(event, world, renderer):
                running = False

        t0 = time.perf_counter()
        world.update(dt)
        ctx.screen.viewport = (0, 0, WIN_W, WIN_H)
        renderer.render(world, pygame.mouse.get_pos())
        ms = (time.perf_counter() - t0) * 1000.0
        renderer.frame_ms += (ms - renderer.frame_ms) * 0.08
        pygame.display.flip()

    renderer.release()
    if bank is not None:
        bank.stop_all()
    pygame.quit()


def run_headless(path: str):
    """The scripted swing, rendered with no window and read back to a PNG.

    Same regression value as the software bench's headless path: it drives the
    real shaders through the real batch, so a broken uniform shows up here
    rather than the first time somebody opens the window.
    """
    pygame.init()
    pygame.font.init()
    # `rj.Renderer` calls `Surface.convert()`, which needs a display to convert
    # *to*. Under the dummy video driver this opens nothing and costs nothing;
    # the GL context below is a standalone EGL one and never touches it.
    pygame.display.set_mode((WIN_W, WIN_H))
    ctx = glfx.create_context(headless=True)
    target = ctx.simple_framebuffer((WIN_W, WIN_H), components=4)
    juice = gl_juice()
    juice.set_all(True)
    world = rj.World(juice, None)
    renderer = GLRenderer(ctx, target, world, headless=True)
    dt = 1.0 / FPS

    def step(n):
        for _ in range(n):
            world.update(dt)

    px, py = world.player.tile
    tx, ty = rj.DUMMY_POS
    while (px, py) != (tx + 1, ty):
        dx = (tx + 1 > px) - (tx + 1 < px)
        dy = 0 if dx else (ty > py) - (ty < py)
        if not dx and not dy:
            break
        world.try_move(world.player, dx, dy)
        step(14)
        if world.player.tile == (px, py):
            break
        px, py = world.player.tile
    world.try_move(world.player, -1, 0)
    step(int((juice.p("windup_time") + juice.p("attack_time") * 0.32) / dt) + 5)

    target.viewport = (0, 0, WIN_W, WIN_H)
    renderer.render(world, (WIN_W - 200, 300))
    image = pygame.image.frombytes(
        bytes(target.read(components=3)), (WIN_W, WIN_H), "RGB", True)
    pygame.image.save(image, path)
    print(f"wrote {path} on {ctx.info['GL_RENDERER']} "
          f"({renderer.batch.count} quads in one draw call, "
          f"{len(world.fx)} live effects)")
    renderer.release()
    pygame.quit()


def main():
    if "--headless" in sys.argv:
        i = sys.argv.index("--headless")
        path = sys.argv[i + 1] if len(sys.argv) > i + 1 else "juice_gl_headless.png"
        run_headless(path)
    else:
        run()


if __name__ == "__main__":
    main()
