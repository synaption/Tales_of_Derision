"""Headless tests for the OpenGL renderer.

Deliberately hostile, in the same way `experimental/fantacy_maps/headlesstest.py`
is: the SDL video driver is pointed at something that cannot show anything and
the GL context is a standalone EGL one with no window behind it, so anything
that quietly depended on a window fails here rather than in six months.

The rule from `notes4LLMs.md` -- "use headless testing even for modernGL or else
it keeps bringing up windows on my desktop" -- is the whole design of this file.

Two kinds of test, and the second kind is the one that earns its keep:

* the parts that are plain maths (normal generation, the atlas layout, uniform
  name lookup) are checked directly;
* the shaders are checked by *rendering and reading pixels back*. A shader has
  no assertions and no stack trace -- a wrong uniform does not raise, it just
  draws something slightly wrong for ever -- so the only way to test one is to
  point it at a known input and look at what came out. Every real bug found
  while writing this renderer was of that kind: an atlas uploaded flipped, an
  array uniform whose name the driver spelled differently, a displacement that
  compressed where it should have stretched. None of them raised anything.

Run directly (`python3 gltest.py`) or under pytest. If no GL context can be
created at all the whole file skips rather than failing, so a machine with no
EGL still gets a green run out of `juicetest.py`.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------
#
# One context and one renderer for the whole file. Contexts are expensive and
# leak-prone, and every test here wants the same one.

_STATE: dict = {}


def gl_available() -> bool:
    """Whether a context can be had at all. Cached, including the failure."""
    if "available" in _STATE:
        return _STATE["available"]
    try:
        import glfx
        ctx = glfx.create_context(headless=True)
        _STATE["ctx"] = ctx
        _STATE["available"] = True
        _STATE["renderer_name"] = ctx.info["GL_RENDERER"]
    except Exception as exc:                         # pragma: no cover
        print(f"  (no GL context: {type(exc).__name__}: {exc}) -- skipping",
              file=sys.stderr)
        _STATE["available"] = False
    return _STATE["available"]


def bench():
    """(ctx, target, world, renderer), built once."""
    if "bench" in _STATE:
        return _STATE["bench"]
    import pygame
    import glfx
    import rogue_juice as rj
    import rogue_juice_gl as gl

    pygame.init()
    pygame.font.init()
    # A dummy display surface, only so `rj.Renderer` can call `Surface.convert`.
    # Nothing is ever shown through it and no window is opened.
    pygame.display.set_mode((gl.WIN_W, gl.WIN_H))
    ctx = _STATE["ctx"]
    target = ctx.simple_framebuffer((gl.WIN_W, gl.WIN_H), components=4)
    world = rj.World(gl.gl_juice(), None)
    renderer = gl.GLRenderer(ctx, target, world, headless=True)
    _STATE["bench"] = (ctx, target, world, renderer)
    return _STATE["bench"]


def frame(world=None, mouse=(0, 0)) -> np.ndarray:
    """Render one frame and read it back as (h, w, 3) uint8, top row first."""
    import rogue_juice_gl as gl
    ctx, target, default_world, renderer = bench()
    target.viewport = (0, 0, gl.WIN_W, gl.WIN_H)
    renderer.render(world or default_world, mouse)
    raw = np.frombuffer(bytes(target.read(components=3)), dtype=np.uint8)
    return raw.reshape(gl.WIN_H, gl.WIN_W, 3)[::-1]


def view_of(image: np.ndarray) -> np.ndarray:
    """Just the world, without the panel."""
    import rogue_juice_gl as gl
    return image[:, :gl.VIEW_W]


def step(world, n, dt=1 / 60):
    for _ in range(n):
        world.update(dt)


# ---------------------------------------------------------------------------
# The context itself
# ---------------------------------------------------------------------------


def test_a_context_exists_with_no_window():
    if not gl_available():
        return
    import pygame
    ctx = _STATE["ctx"]
    assert ctx.version_code >= 330, f"got GL {ctx.version_code}"
    # `bench()` opens a dummy display surface for font conversion; what must
    # never happen is a *GL* window, which is what `ctx.screen` would imply.
    assert _STATE["renderer_name"], "no renderer name"
    print(f"        (rendering on {_STATE['renderer_name']})")


def test_the_wsl_card_check_needs_both_the_device_and_the_driver():
    """Forcing a driver that is not installed leaves Mesa nowhere to fall back
    to and the context is never created at all, so both have to be present."""
    import glfx
    saved = (glfx.WSL_GPU_DEVICE, glfx.WSL_GPU_MODULE,
             os.environ.pop("MESA_LOADER_DRIVER_OVERRIDE", None))
    try:
        from pathlib import Path
        glfx.WSL_GPU_DEVICE = Path("/definitely/not/here")
        assert not glfx.wsl_card_available()
        glfx.WSL_GPU_DEVICE = saved[0]
        glfx.WSL_GPU_MODULE = Path("/definitely/not/here/either")
        assert not glfx.wsl_card_available()
    finally:
        glfx.WSL_GPU_DEVICE, glfx.WSL_GPU_MODULE = saved[0], saved[1]
        if saved[2] is not None:
            os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = saved[2]
    assert glfx.is_software("llvmpipe (LLVM 20)")
    assert not glfx.is_software("D3D12 (NVIDIA GeForce RTX 4080)")


# ---------------------------------------------------------------------------
# Maths that does not need a context
# ---------------------------------------------------------------------------


def test_normals_are_flat_inside_a_shape_and_turn_out_at_the_rim():
    """The whole trick: a silhouette is enough to light a sprite with.

    A blurred mask's gradient points outwards at every edge, which is the
    normal of the shape inflated off the page -- flat and facing the viewer in
    the middle, facing away at the rim.
    """
    import glfx
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    n = glfx.normal_from_alpha(mask, strength=3.0, blur=3).astype(np.float32)
    n = n / 127.5 - 1.0

    middle = n[32, 32]
    assert abs(middle[0]) < 0.05 and abs(middle[1]) < 0.05, "the middle is not flat"
    assert middle[2] > 0.9, "the middle does not face the viewer"

    left = n[32, 16]
    right = n[32, 47]
    assert left[0] < -0.25, f"the left rim does not face left: {left}"
    assert right[0] > 0.25, f"the right rim does not face right: {right}"
    top = n[16, 32]
    bottom = n[47, 32]
    assert top[1] < -0.25, "the top rim does not face up"
    assert bottom[1] > 0.25, "the bottom rim does not face down"

    lengths = np.linalg.norm(n, axis=2)
    assert abs(lengths.mean() - 1.0) < 0.02, "normals are not unit length"


def test_deeper_normals_turn_further():
    import glfx
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    shallow = glfx.normal_from_alpha(mask, strength=1.0).astype(np.float32) / 127.5 - 1
    deep = glfx.normal_from_alpha(mask, strength=4.0).astype(np.float32) / 127.5 - 1
    assert abs(deep[32, 16][0]) > abs(shallow[32, 16][0])


def test_an_array_uniform_is_found_under_either_spelling():
    """GLSL leaves it to the implementation whether an array uniform is
    reported as `u_lights` or `u_lights[0]`, and drivers differ.

    Checking only the bare name is the trap: the lookup misses silently, the
    uniform is never written, and the shader runs perfectly happily with a
    zero-length light array. Nothing errors -- the room is simply dark.
    """
    import glfx

    class FakeProgram:
        def __init__(self, name):
            self._name = name
            self.written = None

        def __contains__(self, key):
            return key == self._name

        def __getitem__(self, key):
            assert key == self._name
            return self

        def write(self, data):
            self.written = data

    for spelling in ("u_lights", "u_lights[0]"):
        prog = FakeProgram(spelling)
        n = glfx.write_vec4_array(prog, "u_lights", [(1, 2, 3, 4)], 8)
        assert n == 1, f"{spelling} was not found"
        assert prog.written is not None and len(prog.written) == 8 * 4 * 4
    assert glfx.write_vec4_array(FakeProgram("something_else"), "u_lights",
                                 [(1, 2, 3, 4)], 8) == 0


# ---------------------------------------------------------------------------
# The atlas
# ---------------------------------------------------------------------------


def test_the_atlas_holds_every_sprite_the_digits_and_a_white_texel():
    if not gl_available():
        return
    import tiles
    ctx, target, world, renderer = bench()
    atlas = renderer.atlas
    for key in tiles.SPRITES:
        assert key in atlas.uv, f"{key} is not in the atlas"
    for ch in renderer.DIGITS:
        assert ch in renderer.digit_uv, f"digit {ch} is not in the atlas"
    assert atlas.white[0] == atlas.white[1], "the white cell is not a single texel"
    for (u0, v0), (u1, v1) in atlas.uv.values():
        assert 0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0, "cell escaped the page"


def test_atlas_cells_do_not_overlap():
    """Padding is what stops filtering at a cell edge reaching the neighbour --
    which draws as a sliver of the wrong creature down one side of a sprite."""
    if not gl_available():
        return
    ctx, target, world, renderer = bench()
    cells = list(renderer.atlas.uv.values())
    for i, ((ax0, ay0), (ax1, ay1)) in enumerate(cells):
        for (bx0, by0), (bx1, by1) in cells[i + 1:]:
            apart = ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0
            assert apart, "two atlas cells overlap"


def test_the_atlas_is_the_right_way_up():
    """Uploaded flipped, every sprite samples a slice of the padding above its
    neighbour -- which does not look like an upside-down sprite, it looks like
    the art failing to load. Checked against the source surface directly."""
    if not gl_available():
        return
    import pygame
    import tiles
    ctx, target, world, renderer = bench()
    atlas = renderer.atlas
    w, h = atlas.size
    data = np.frombuffer(atlas.albedo.read(), dtype=np.uint8).reshape(h, w, 4)

    col, row = tiles.SPRITES["player"]
    source = renderer.sheet.sprite(col, row, None)
    source = pygame.transform.scale(source, (atlas.cell, atlas.cell))
    (u0, v0), (u1, v1) = atlas.uv["player"]
    x, y = int(u0 * w), int(v0 * h)
    patch = data[y:y + atlas.cell, x:x + atlas.cell]

    # Compare the vertical centre of mass of the alpha: a flipped upload puts
    # it on the wrong side of the cell.
    src_alpha = np.array([[source.get_at((c, r))[3] for c in range(atlas.cell)]
                          for r in range(atlas.cell)], dtype=np.float32)
    rows = np.arange(atlas.cell, dtype=np.float32)
    src_com = float((src_alpha.sum(axis=1) * rows).sum() / max(src_alpha.sum(), 1))
    dst_alpha = patch[..., 3].astype(np.float32)
    dst_com = float((dst_alpha.sum(axis=1) * rows).sum() / max(dst_alpha.sum(), 1))
    assert abs(src_com - dst_com) < 2.0, \
        f"atlas is mirrored: source centre of mass {src_com:.1f}, atlas {dst_com:.1f}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_a_frame_comes_back_and_is_not_empty():
    if not gl_available():
        return
    image = frame()
    world_part = view_of(image)
    assert world_part.mean() > 3.0, "the world is black"
    assert world_part.std() > 6.0, "the world is a flat fill"


def test_the_panel_lands_on_the_frame():
    """The UI is drawn by the software renderer and uploaded. If the upload or
    the composite's alpha is wrong the panel is simply missing, and everything
    else still looks fine."""
    if not gl_available():
        return
    import rogue_juice_gl as gl
    image = frame()
    panel = image[:, gl.VIEW_W:]
    assert panel.std() > 12.0, "the panel is blank"
    # The header is drawn light on dark: there must be near-white text in it.
    assert panel[:40].max() > 180, "no panel header text"


def test_every_effect_can_be_switched_off_without_a_broken_frame():
    """One pass with each effect solo. This is what stops a toggle being wired
    to a uniform that is only set on some other branch."""
    if not gl_available():
        return
    import rogue_juice as rj
    import rogue_juice_gl as gl
    ctx, target, _, renderer = bench()
    keys = list(gl.gl_juice().toggles)
    for solo in keys + [None]:
        juice = gl.gl_juice()
        juice.set_all(False)
        if solo:
            juice.toggles[solo].on = True
        world = rj.World(juice, None)
        world.try_move(world.player, -1, 0)
        step(world, 10)
        d = world.nearest_enemy(killable=True)
        world.land_blow(world.player, d, 1, 0)
        step(world, 8)
        image = view_of(frame(world))
        assert np.isfinite(image).all()
        assert image.max() > 0, f"{solo}: the frame went completely black"


def test_moving_a_light_changes_the_picture():
    if not gl_available():
        return
    import rogue_juice as rj
    import rogue_juice_gl as gl
    juice = gl.gl_juice()
    juice.set_all(False)
    juice.toggles["light"].on = True
    world = rj.World(juice, None)
    before = view_of(frame(world)).astype(int)
    world.lights.append([world.camera.x - 180, world.camera.y, 2.0, 1.0, 1.0])
    after = view_of(frame(world)).astype(int)
    moved = float(np.abs(after - before).mean())
    assert moved > 1.5, f"adding a light changed the frame by {moved:.2f}"


def test_lighting_falls_off_with_distance():
    """Measured by moving the light, not by comparing two places.

    Comparing the middle of the view with a corner is the obvious test and it
    is worthless: the corner is a wall and the middle is floor, so it measures
    the difference between two albedos and reports whichever is brighter.
    Moving one light and watching the *same* pixels is the only comparison that
    isolates the falloff.
    """
    if not gl_available():
        return
    import rogue_juice as rj
    import rogue_juice_gl as gl

    def patch_at(distance):
        juice = gl.gl_juice()
        juice.set_all(False)
        juice.toggles["light"].on = True
        juice.params["light_ambient"].value = 0.05
        juice.params["light_radius"].value = 7.0
        world = rj.World(juice, None)
        world.lights.clear()
        px, py = world.player.world_pos()
        # One light, moved away from a patch of floor we keep looking at.
        world.player.body.tx += 40                # the player's own light, gone
        world.lights.append([px + distance, py + 90, 4.0, 1.0, 1.0])
        return view_of(frame(world)).astype(float)[380:420, 380:420].mean()

    near, far = patch_at(0), patch_at(400)
    assert near > far * 1.8, f"no falloff: near {near:.1f}, far {far:.1f}"


def test_normal_mapping_shades_a_sprite_from_the_side():
    """The payoff of the generated normals: a light to the left of a creature
    has to look different from a light to the right of it. With flat sprites
    the two are identical, which is exactly what the software bench does.

    Rendered as a single sprite with a single light rather than through the
    world, because the world always carries the player's own light and a test
    with two lights in it measures neither.
    """
    if not gl_available():
        return

    def lit_from(dx, flat=False):
        image = _one_sprite(light=(dx, 0.0, 260.0), flat_normal=flat).astype(float)
        half = image.max(axis=2)[96:160, 96:160]
        return half[:, :32].mean(), half[:, 32:].mean()

    left_near, left_far = lit_from(-120)
    right_near, right_far = lit_from(+120)
    assert left_near > left_far * 1.05, \
        f"lighting from the left did not favour the left: {left_near:.1f} vs {left_far:.1f}"
    assert right_far > right_near * 1.05, \
        f"lighting from the right did not favour the right: {right_far:.1f} vs {right_near:.1f}"

    # And with a flat normal the two sides match, which is the thing the
    # toggle is switching between.
    flat_near, flat_far = lit_from(-120, flat=True)
    assert abs(flat_near - flat_far) < abs(left_near - left_far), \
        "a flat normal shaded the sprite as unevenly as a real one"


def test_switching_normals_off_flattens_the_sprites():
    if not gl_available():
        return
    import rogue_juice as rj
    import rogue_juice_gl as gl

    def render(normals):
        juice = gl.gl_juice()
        juice.set_all(False)
        juice.toggles["light"].on = True
        juice.toggles["normals"].on = normals
        juice.params["light_height"].value = 12.0
        world = rj.World(juice, None)
        return view_of(frame(world)).astype(int)

    diff = float(np.abs(render(True) - render(False)).max())
    assert diff > 6, f"the normals toggle changed nothing (max {diff})"


# ---------------------------------------------------------------------------
# The soft body
# ---------------------------------------------------------------------------


def _one_sprite(lag_px=0.0, power=1.7, light=None, flat_normal=False,
                sharpness=1.0, zoom=1.0):
    """Draw one sprite alone into a private buffer, with everything named.

    Testing a shader means giving it a known input and looking at the pixels
    that came out. Doing that through the whole bench means fighting the
    player's own light, the floor and the panel; this is the same program with
    one quad in front of it.
    """
    import glfx
    import rogue_juice_gl as gl
    ctx, target, world, renderer = bench()
    size = (256, 256)
    tex = ctx.texture(size, 4, dtype="f2")
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()
    ctx.clear(0.0, 0.0, 0.0, 1.0)

    p = renderer.quad_prog
    p["u_camera"] = (0.0, 0.0)
    p["u_half"] = (size[0] * 0.5, size[1] * 0.5)
    p["u_roll"] = 0.0
    p["u_zoom"] = zoom
    p["u_shake"] = (0.0, 0.0)
    p["u_atlas_size"] = renderer.atlas.size
    p["u_soft_power"] = power
    p["u_soft_pinch"] = 0.0
    p["u_sharpness"] = sharpness
    p["u_light_color"] = (1.0, 1.0, 1.0)
    p["u_light_height"] = 18.0
    if light is None:
        p["u_lit"] = 0.0
        p["u_ambient"] = (1.0, 1.0, 1.0)
        p["u_light_count"] = 0
    else:
        p["u_lit"] = 1.0
        p["u_ambient"] = (0.0, 0.0, 0.0)
        lx, ly, radius = light
        p["u_light_count"] = glfx.write_vec4_array(
            p, "u_lights", [(lx, ly, radius, 1.0)], glfx.MAX_LIGHTS)
    renderer.atlas.albedo.use(0)
    renderer.atlas.normal.use(1)
    p["u_albedo"] = 0
    p["u_normal"] = 1

    b = renderer.batch
    b.clear()
    b.add(0.0, 0.0, gl.TILE, gl.TILE, uv=renderer.atlas.uv["ogre"],
          color=(1.0, 1.0, 1.0), alpha=1.0, lag=(lag_px, 0.0),
          params=(0.0, 1.0 if flat_normal else 0.0, 0.0, 0.0))
    b.render()

    raw = np.frombuffer(bytes(fbo.read(components=3)), dtype=np.uint8)
    image = raw.reshape(size[1], size[0], 3)[::-1]
    fbo.release()
    tex.release()
    return image


def _lagged_sprite(lag_px, power=1.7):
    """Draw one sprite alone with a known lag, and return its ink footprint."""
    import glfx
    import rogue_juice_gl as gl
    ctx, target, world, renderer = bench()
    size = (256, 256)
    tex = ctx.texture(size, 4, dtype="f2")
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()
    ctx.clear(0.0, 0.0, 0.0, 1.0)

    p = renderer.quad_prog
    p["u_camera"] = (0.0, 0.0)
    p["u_half"] = (size[0] * 0.5, size[1] * 0.5)
    p["u_roll"] = 0.0
    p["u_zoom"] = 1.0
    p["u_shake"] = (0.0, 0.0)
    p["u_atlas_size"] = renderer.atlas.size
    p["u_soft_power"] = power
    p["u_soft_pinch"] = 0.0
    p["u_lit"] = 0.0
    p["u_sharpness"] = 1.0
    p["u_light_count"] = 0
    p["u_ambient"] = (1.0, 1.0, 1.0)
    p["u_light_height"] = 40.0
    p["u_light_color"] = (1.0, 1.0, 1.0)
    renderer.atlas.albedo.use(0)
    renderer.atlas.normal.use(1)
    p["u_albedo"] = 0
    p["u_normal"] = 1

    b = renderer.batch
    b.clear()
    b.add(0.0, 0.0, gl.TILE, gl.TILE, uv=renderer.atlas.uv["ogre"],
          color=(1.0, 1.0, 1.0), alpha=1.0, lag=(lag_px, 0.0))
    b.render()

    raw = np.frombuffer(bytes(fbo.read(components=3)), dtype=np.uint8)
    image = raw.reshape(size[1], size[0], 3)[::-1]
    fbo.release()
    tex.release()
    return image


def test_the_soft_body_stretches_backwards_only():
    """A body dragged left has to grow on its left and stay put on its right.
    Growing at both ends is a scale, not a lag, and reads as a pulse."""
    if not gl_available():
        return
    import rogue_juice_gl as gl
    still = _lagged_sprite(0.0)
    dragged = _lagged_sprite(-gl.TILE * 0.5)          # tail pushed left

    def extent(image):
        cols = np.where(image.max(axis=(0, 2)) > 20)[0]
        return int(cols.min()), int(cols.max())

    s0, s1 = extent(still)
    d0, d1 = extent(dragged)
    assert d0 < s0 - 6, f"did not stretch backwards: {d0} vs {s0}"
    assert abs(d1 - s1) <= 3, f"the leading edge moved: {d1} vs {s1}"


def test_the_soft_body_stretches_rather_than_compressing():
    """The bug this exists for.

    The obvious way to write the sampler is to reuse the *forward* displacement
    with the drawn position in place of the source one. That is not the
    inverse: it squeezes the middle of the body by up to ten to one, which
    draws as a smear of blur rather than a stretch, and past a certain lag it
    folds the sprite back through itself. Compression destroys ink, so counting
    it catches the whole family of mistakes.
    """
    if not gl_available():
        return
    import rogue_juice_gl as gl
    still = _lagged_sprite(0.0)
    dragged = _lagged_sprite(-gl.TILE * 0.6)

    def ink(image):
        return float((image.max(axis=2) > 20).sum())

    a, b = ink(still), ink(dragged)
    assert b > a * 1.05, f"stretching lost ink: {a:.0f} -> {b:.0f} (compressed)"
    assert b < a * 2.4, f"ink exploded: {a:.0f} -> {b:.0f}"


def test_the_soft_body_never_folds_through_itself():
    """A fold shows up as a column of the sprite appearing twice. The
    silhouette of a stretched body must stay a single connected run."""
    if not gl_available():
        return
    import rogue_juice_gl as gl
    for lag in (0.3, 0.6, 0.9):
        image = _lagged_sprite(-gl.TILE * lag)
        lit = image.max(axis=(0, 2)) > 20
        runs = np.diff(np.concatenate(([0], lit.view(np.int8), [0]))).nonzero()[0]
        assert len(runs) == 2, f"lag {lag}: silhouette broke into {len(runs) // 2} runs"


def test_a_higher_falloff_concentrates_the_stretch_in_the_tail():
    """The extent is the same either way -- the tail always ends up a full lag
    behind -- so what the exponent moves is where the *material* sits. Steep,
    the body stays bunched at the head and only a thin streamer trails; shallow,
    the whole thing is drawn out evenly."""
    if not gl_available():
        return
    import rogue_juice_gl as gl

    def centroid(power):
        image = _lagged_sprite(-gl.TILE * 0.5, power=power).max(axis=2)
        weight = image.sum(axis=0).astype(float)
        cols = np.arange(len(weight))
        return float((weight * cols).sum() / max(weight.sum(), 1.0))

    steep, shallow = centroid(4.0), centroid(0.6)
    assert steep > shallow + 1.0, \
        f"a steeper falloff did not keep the body forward: {steep:.1f} vs {shallow:.1f}"


# ---------------------------------------------------------------------------
# Filtering, batching, post
# ---------------------------------------------------------------------------


def test_sharp_bilinear_is_sharper_than_bilinear():
    """The whole reason the software bench needs three pixel modes and this one
    needs none.

    Measured as edge contrast on a magnified sprite. It only means anything
    while the shader is *magnifying*: pre-scaling the atlas in pygame and
    handing the card a 1:1 page silently turns this filter into a no-op, which
    is a mistake that leaves the picture looking perfectly reasonable.
    """
    if not gl_available():
        return

    def edges(sharpness):
        image = _one_sprite(sharpness=sharpness, zoom=3.0).astype(float)
        return float(np.abs(np.diff(image.mean(axis=2), axis=1)).max())

    sharp, smooth = edges(1.0), edges(0.0)
    assert sharp > smooth * 1.05, \
        f"sharp bilinear was not sharper: {sharp:.1f} vs {smooth:.1f}"


def test_everything_is_one_draw_call():
    if not gl_available():
        return
    import rogue_juice as rj
    import rogue_juice_gl as gl
    juice = gl.gl_juice()
    world = rj.World(juice, None)
    ctx, target, _, renderer = bench()
    d = world.nearest_enemy(killable=True)
    for _ in range(3):
        world.land_blow(world.player, d, 1, 0)
        step(world, 4)
    renderer.fill_batch(world)
    assert renderer.batch.count > 60, \
        f"only {renderer.batch.count} quads -- effects are missing"
    # Sprites, particles, shadows, decals, rings, arcs and cuts, all of them in
    # the one instance buffer.
    shapes = set(renderer.batch.data[:renderer.batch.count, 16])
    assert len(shapes) >= 3, f"only shapes {shapes} made it into the batch"


def test_the_instance_buffer_grows_instead_of_overflowing():
    if not gl_available():
        return
    import glfx
    ctx, target, world, renderer = bench()
    batch = glfx.QuadBatch(ctx, renderer.quad_prog, capacity=8)
    for i in range(200):
        batch.add(float(i), 0.0, 4.0, 4.0, uv=renderer.atlas.white,
                  shape=glfx.SHAPE_DISC)
    assert batch.count == 200
    assert batch.capacity >= 200
    assert batch.data[199][0] == 199.0, "the contents were lost while growing"
    batch.render()
    batch.release()


def test_bloom_spreads_light_beyond_a_bright_shape():
    if not gl_available():
        return
    import rogue_juice as rj
    import rogue_juice_gl as gl

    def torch_surround(bloom):
        juice = gl.gl_juice()
        juice.set_all(False)
        juice.toggles["light"].on = True
        juice.toggles["bloom"].on = bloom
        juice.params["bloom_amt"].value = 1.2
        juice.params["bloom_thresh"].value = 60.0
        world = rj.World(juice, None)
        return view_of(frame(world)).astype(float).mean()

    assert torch_surround(True) > torch_surround(False) + 0.4, \
        "bloom added no light to the frame"


def test_the_ripple_displaces_the_floor():
    if not gl_available():
        return
    import rogue_juice as rj
    import rogue_juice_gl as gl
    juice = gl.gl_juice()
    juice.set_all(False)
    juice.toggles["ripple"].on = True
    world = rj.World(juice, None)
    flat = view_of(frame(world)).astype(int)
    world.fx.impact(world.camera.x, world.camera.y,
                    strength=18.0, life=0.42, speed=270.0, wavelength=46.0)
    step(world, 8)
    rippled = view_of(frame(world)).astype(int)
    # A ripple is a thin travelling annulus by design -- one that covered the
    # whole screen would read as an earthquake -- so the mean over the frame is
    # the wrong statistic. Look at how far any pixel moved and how many did.
    delta = np.abs(rippled - flat).max(axis=2)
    assert delta.max() > 24, f"nothing moved at all (peak {delta.max()})"
    assert (delta > 6).sum() > 400, f"only {(delta > 6).sum()} pixels moved"


def test_the_camera_transform_is_geometry_not_a_resample():
    """Shake and roll are applied in the vertex shader, so the frame is drawn
    from a different place rather than rotated afterwards. The tell is that the
    corners never go black: a rotated *image* brings its own edges in."""
    if not gl_available():
        return
    import rogue_juice as rj
    import rogue_juice_gl as gl
    juice = gl.gl_juice()
    juice.set_all(False)
    juice.toggles["shake"].on = True
    juice.toggles["tilt"].on = True
    world = rj.World(juice, None)
    world.trauma.add(1.0)
    world.camera.punch(tilt=60.0)
    step(world, 3)
    image = view_of(frame(world))
    for name, patch in (("top-left", image[2:14, 2:14]),
                        ("top-right", image[2:14, -14:-2]),
                        ("bottom-left", image[-14:-2, 2:14])):
        assert patch.max() > 4, f"{name} corner is empty -- the frame was resampled"


# ---------------------------------------------------------------------------
# The bench is still the bench
# ---------------------------------------------------------------------------


def test_the_gl_bench_keeps_every_software_control():
    """The renderer changed; the bench did not. Anything dropped has to be
    dropped on purpose."""
    import rogue_juice as rj
    import rogue_juice_gl as gl
    soft = rj.Juice()
    hard = gl.gl_juice()
    for key in soft.toggles:
        assert key in hard.toggles, f"{key} vanished in the GL build"
    for key in soft.params.params:
        assert key in hard.params or key in gl.DROP_PARAMS, \
            f"{key} vanished in the GL build"
    for key in gl.DROP_CHOICES:
        assert key not in hard.choices
    assert len(hard.toggles) == len(soft.toggles) + len(gl.GL_TOGGLES)


def test_the_sim_layer_is_untouched_by_the_renderer():
    """`juicefx.py` has no pygame in it and therefore no OpenGL in it. If that
    is still true, the GL module can be imported and the maths module cannot
    tell."""
    import juicefx
    source = open(juicefx.__file__).read()
    for banned in ("import pygame", "import moderngl", "import glfx"):
        assert banned not in source, f"juicefx.py picked up {banned!r}"


def test_new_gl_controls_sit_in_groups_the_panel_draws():
    import rogue_juice as rj
    import rogue_juice_gl as gl
    juice = gl.gl_juice()
    for t in juice.toggles.values():
        assert t.group in rj.GROUPS, f"{t.key} is in unknown group {t.group!r}"
    for p in juice.params.params.values():
        assert p.group in rj.GROUPS, f"{p.key} is in unknown group {p.group!r}"


def test_the_gl_panel_still_scrolls_and_clicks():
    """The panel is the software renderer's, uploaded as a texture, so all of
    its behaviour has to carry over -- including rows that only exist at some
    scroll positions."""
    if not gl_available():
        return
    import rogue_juice_gl as gl
    ctx, target, world, renderer = bench()
    ui = renderer.ui
    ui.scroll = 0.0
    found = set()
    for _ in range(60):
        # The panel redraw is throttled to 30Hz, so a loop this tight has to
        # say that something changed -- exactly as the event handler does.
        renderer.ui_dirty = True
        frame(world)
        for rect, kind, key in ui.rows:
            found.add((kind, key))
        limit = ui.content_h - ui.panel_rect().h
        if ui.scroll >= limit:
            break
        ui.scroll = min(ui.scroll + 90, limit)
    for key in world.juice.toggles:
        assert ("t", key) in found, f"{key} has no row in the GL panel"
    for key in world.juice.params.params:
        assert ("p", key) in found, f"{key} has no slider in the GL panel"


def test_damage_numbers_live_in_the_world():
    """Drawn into the UI overlay they would sit perfectly still while the
    screen shook underneath them, so they are digit quads in the batch."""
    if not gl_available():
        return
    import rogue_juice as rj
    import rogue_juice_gl as gl
    ctx, target, _, renderer = bench()
    juice = gl.gl_juice()
    juice.set_all(False)
    juice.toggles["numbers"].on = True
    world = rj.World(juice, None)
    renderer.fill_batch(world)
    before = renderer.batch.count
    world.fx.floaters.add("42", world.camera.x, world.camera.y)
    renderer.fill_batch(world)
    assert renderer.batch.count == before + 2, "the digits are not in the batch"


def test_the_headless_render_writes_a_frame():
    if not gl_available():
        return
    import tempfile
    import pygame
    import rogue_juice_gl as gl
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "gl.png")
        gl.run_headless(path)
        assert os.path.exists(path)
        image = pygame.image.load(path)
        assert image.get_size() == (gl.WIN_W, gl.WIN_H)
    # `run_headless` calls pygame.quit(); the shared bench needs it back.
    _STATE.pop("bench", None)
    pygame.init()
    pygame.font.init()


# ---------------------------------------------------------------------------


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:                      # noqa: BLE001
            failures.append((name, exc))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
