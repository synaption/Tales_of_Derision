"""The PlayStation look, as a render pipeline and one pair of shaders.

The console did not have an "aesthetic"; it had five limitations, and every
game of the era wore all five. Rather than fake the result with a post-process
filter, this module reproduces the causes, because the causes interact -- the
wobble only reads as wobble because the framebuffer is 320x240, and the
texture warp only shows up on large polygons near the camera.

**1. No subpixel precision.** The GTE transformed vertices to integer screen
coordinates. The vertex shader snaps clip-space XY to the low-res pixel grid,
so geometry shimmers as it moves and seams open and close along shared edges.

**2. No perspective correction.** The GPU interpolated texture coordinates
linearly in screen space with no divide by w, so textures on big surfaces
swim as the camera turns. GLSL spells this ``noperspective``, which is a
one-word implementation of an entire generation's visual signature.

**3. No Z-buffer.** Real hardware sorted per polygon and got it wrong at the
edges. This is the one limitation not reproduced -- a real ordering table
would make the city unreadable, and every PS1 game shipped with the artifact
rather than because of it.

**4. 15-bit colour.** Five bits a channel with a 4x4 ordered dither, applied
at the low resolution so the dither pattern is chunky and visible, exactly as
it was on a CRT.

**5. Vertex lighting and hard fog.** Lambert per vertex, quantised to four
steps, with distance fog doing the job that draw distance could not.

Everything is rendered into a 320x240 buffer and blown up with nearest-
neighbour sampling, letterboxed to 4:3 so the pixels stay square whatever the
window is.
"""

from __future__ import annotations

import math
import random

from panda3d.core import (
    Camera,
    CardMaker,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    NodePath,
    OrthographicLens,
    PNMImage,
    PerspectiveLens,
    SamplerState,
    Shader,
    Texture,
    Vec3,
    Vec4,
    WindowProperties,
)

# The console's most common mode. 4:3, and small enough that the dither and
# the vertex snapping are features rather than trivia.
RES_X, RES_Y = 320, 240
ASPECT = RES_X / RES_Y

# Fog and sky. A muted teal night reads as "the ants came at dusk" and hides
# the draw distance without looking like a bug. The components are exact 5-bit
# levels (5/31, 6/31, 8/31) so the cleared background sits on the same colour
# grid the dither quantises everything else onto.
SKY = Vec4(5 / 31, 6 / 31, 8 / 31, 1.0)
# Far enough out that the city reads as a city before it dissolves, close
# enough that the draw distance never has to be honest about where it ends.
FOG_NEAR, FOG_FAR = 70.0, 245.0

# Single directional light, aimed to rake across building faces.
LIGHT_DIR = Vec3(-0.45, 0.62, -0.65)
LIGHT_COLOR = Vec4(0.95, 0.92, 0.82, 1.0)
AMBIENT = Vec4(0.40, 0.42, 0.52, 1.0)


VERT = """
#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrixInverseTranspose;
uniform vec4 p3d_ColorScale;

uniform vec2 jitterRes;
uniform vec3 lightDir;
uniform vec4 lightColor;
uniform vec4 ambient;
// 1 on anything that should ignore the sun: bolts, sparks, jet flame. The
// console had no emissive term either -- it had artists setting vertex
// colours to white and calling it a day, which is exactly what this is.
uniform float emissive;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;
in vec4 p3d_Color;

// The whole trick, in one qualifier: interpolate UVs linearly in screen
// space with no perspective divide, exactly as the PSX rasteriser did.
noperspective out vec2 uv;
out vec4 vcolor;
out float vdist;

void main() {
    vec4 clip = p3d_ModelViewProjectionMatrix * p3d_Vertex;

    // No subpixel precision: snap XY to the low-res pixel grid. Guarded on w
    // so vertices behind the camera, where the divide flips sign, are left
    // alone rather than folded across the screen.
    if (clip.w > 0.0001) {
        vec2 grid = jitterRes * 0.5;
        clip.xy = floor((clip.xy / clip.w) * grid + 0.5) / grid * clip.w;
    }
    gl_Position = clip;

    vdist = length((p3d_ModelViewMatrix * p3d_Vertex).xyz);

    // Gouraud, quantised to four bands. Smooth ramps are a later console's
    // luxury; banding is what the hardware actually produced.
    vec3 n = normalize((p3d_ModelMatrixInverseTranspose * vec4(p3d_Normal, 0.0)).xyz);
    float lam = max(dot(n, -normalize(lightDir)), 0.0);
    lam = floor(lam * 4.0) / 4.0;
    vec3 lit = mix(ambient.rgb + lam * lightColor.rgb, vec3(1.35), emissive);

    vcolor = vec4(p3d_Color.rgb * p3d_ColorScale.rgb * lit,
                  p3d_Color.a * p3d_ColorScale.a);
    uv = p3d_MultiTexCoord0;
}
"""


FRAG = """
#version 150

uniform sampler2D p3d_Texture0;
uniform vec4 fogColor;
uniform vec2 fogRange;

noperspective in vec2 uv;
in vec4 vcolor;
in float vdist;

out vec4 fragColor;

// Bayer 4x4. The PSX dithered on the way into its 15-bit framebuffer, which
// is why period screenshots have that fine crosshatch in every gradient.
const float bayer[16] = float[16](
     0.0,  8.0,  2.0, 10.0,
    12.0,  4.0, 14.0,  6.0,
     3.0, 11.0,  1.0,  9.0,
    15.0,  7.0, 13.0,  5.0
);

void main() {
    vec4 texel = texture(p3d_Texture0, uv);
    float alpha = texel.a * vcolor.a;
    if (alpha < 0.35) discard;          // no blending: cut-outs, like the era

    vec3 c = texel.rgb * vcolor.rgb;

    float fog = clamp((vdist - fogRange.x) / (fogRange.y - fogRange.x), 0.0, 1.0);
    c = mix(c, fogColor.rgb, fog);

    ivec2 p = ivec2(mod(gl_FragCoord.xy, 4.0));
    float d = (bayer[p.y * 4 + p.x] / 16.0 - 0.5) / 31.0;
    c = floor(clamp(c + d, 0.0, 1.0) * 31.0 + 0.5) / 31.0;   // 5 bits a channel

    fragColor = vec4(c, 1.0);
}
"""


# --------------------------------------------------------------------------
# procedural textures -- no asset files, and everything stays in one repo blob
# --------------------------------------------------------------------------


def _finish(image: PNMImage, name: str, repeat: bool = True) -> Texture:
    """Wrap a painted image as a nearest-filtered, unmipmapped texture.

    Nearest with no mipmaps is not laziness: the console had neither, and the
    crawling aliasing on distant surfaces is a large part of the look.
    """
    tex = Texture(name)
    tex.load(image)
    tex.setMagfilter(SamplerState.FT_nearest)
    tex.setMinfilter(SamplerState.FT_nearest)
    mode = Texture.WMRepeat if repeat else Texture.WMClamp
    tex.setWrapU(mode)
    tex.setWrapV(mode)
    return tex


def _noise(image: PNMImage, rng: random.Random, base: tuple[float, float, float], amount: float) -> None:
    """Fill with a flat colour plus per-texel grain, quantised to 5 bits."""
    for y in range(image.getYSize()):
        for x in range(image.getXSize()):
            n = rng.uniform(-amount, amount)
            image.setXel(
                x,
                y,
                min(1.0, max(0.0, round((base[0] + n) * 31) / 31)),
                min(1.0, max(0.0, round((base[1] + n) * 31) / 31)),
                min(1.0, max(0.0, round((base[2] + n) * 31) / 31)),
            )


def make_facade(style: int, size: int = 64) -> Texture:
    """A concrete slab with a grid of lit and unlit windows.

    One texel of the texture is about 40cm on the building, and the UVs are
    generated in metres, so towers of different sizes share one texture with
    the windows staying the same size on all of them.
    """
    rng = random.Random(9000 + style)
    tints = [(0.52, 0.50, 0.48), (0.44, 0.46, 0.50), (0.58, 0.52, 0.44), (0.40, 0.42, 0.44)]
    img = PNMImage(size, size)
    _noise(img, rng, tints[style % len(tints)], 0.05)

    pitch = 8  # window every 8 texels
    for wy in range(2, size - 2, pitch):
        for wx in range(2, size - 2, pitch):
            roll = rng.random()
            if roll < 0.18:
                colour = (0.95, 0.85, 0.55)  # someone is still at their desk
            elif roll < 0.34:
                colour = (0.30, 0.45, 0.55)
            else:
                colour = (0.09, 0.10, 0.13)
            for y in range(wy, min(wy + pitch - 3, size)):
                for x in range(wx, min(wx + pitch - 3, size)):
                    img.setXel(x, y, *colour)
    return _finish(img, f"facade{style}")


def make_road(size: int = 64) -> Texture:
    """Asphalt with a kerb strip and a dashed centre line.

    Tiled at one texture per 8m of street, so the dashes come out at a
    plausible size when the ground plane is UV-mapped in metres.
    """
    rng = random.Random(4242)
    img = PNMImage(size, size)
    _noise(img, rng, (0.20, 0.21, 0.23), 0.045)
    for y in range(size):
        for x in range(size):
            if x < 3 or x >= size - 3:  # kerb
                img.setXel(x, y, 0.42, 0.42, 0.40)
    for y in range(4, size - 4, 16):  # dashed centre line
        for dy in range(8):
            for x in range(size // 2 - 1, size // 2 + 1):
                img.setXel(x, (y + dy) % size, 0.72, 0.68, 0.38)
    return _finish(img, "road")


def make_mottle(
    name: str,
    base: tuple[float, float, float],
    seed: int,
    size: int = 32,
    grain: float = 0.07,
    blotch: float = 0.11,
) -> Texture:
    """Blotchy organic-ish texture, used for chitin, uniforms and gravel.

    ``grain`` is the per-texel noise and ``blotch`` the amplitude of the
    overlaid patches. Gravel wants both turned down: it covers whole rooftops
    at close range, where the same contrast that makes an ant look like chitin
    makes a roof look like moss.
    """
    rng = random.Random(seed)
    img = PNMImage(size, size)
    _noise(img, rng, base, grain)
    for _ in range(size):
        cx, cy = rng.randrange(size), rng.randrange(size)
        radius = rng.randint(1, 3)
        shade = rng.uniform(-blotch, blotch * 0.85)
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 > radius * radius:
                    continue
                px, py = x % size, y % size
                col = img.getXel(px, py)
                img.setXel(
                    px,
                    py,
                    min(1.0, max(0.0, col[0] + shade)),
                    min(1.0, max(0.0, col[1] + shade)),
                    min(1.0, max(0.0, col[2] + shade)),
                )
    return _finish(img, name)


def make_flat(name: str, colour: tuple[float, float, float], size: int = 4) -> Texture:
    """A solid colour. Needed because every node must carry a texture.

    The fragment shader samples ``p3d_Texture0`` unconditionally, and an
    unbound sampler is undefined behaviour, so untextured is not an option.
    """
    img = PNMImage(size, size)
    img.fill(*colour)
    return _finish(img, name)


class TextureBank:
    """Every texture in the game, built once and handed out by name."""

    def __init__(self) -> None:
        self.facades = [make_facade(i) for i in range(4)]
        self.road = make_road()
        self.gravel = make_mottle("gravel", (0.44, 0.43, 0.46), 21, grain=0.035, blotch=0.045)
        # Creature and uniform textures are near-white on purpose: the colour
        # of an individual comes from its Renderable tint, so one grey mottle
        # serves red ants, green spitters and olive grunts alike.
        self.chitin = make_mottle("chitin", (0.80, 0.76, 0.72), 11)
        self.fatigues = make_mottle("fatigues", (0.82, 0.82, 0.78), 13)
        self.armour = make_mottle("armour", (0.86, 0.88, 0.92), 14)
        self.white = make_flat("white", (1.0, 1.0, 1.0))
        self.glow = make_flat("glow", (1.0, 0.95, 0.75))


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------


class Ps1Pipeline:
    """Renders the 3D scene into a 320x240 buffer and blows it up.

    Also owns :attr:`hud2d`, an orthographic overlay drawn *into the same
    buffer* on a second display region -- so the HUD is 320x240 too, and gets
    the same chunky upscale as the world instead of floating above it at
    native resolution looking like a different decade.
    """

    def __init__(self, base) -> None:
        self.base = base
        self.textures = TextureBank()
        self.shader = Shader.make(Shader.SL_GLSL, VERT, FRAG)

        base.setBackgroundColor(SKY)
        # Note: do not follow this with setShaderAuto(False). Both calls write
        # the same ShaderAttrib on the same node, so the later one wins and
        # the explicit shader is silently discarded -- the scene still renders,
        # via fixed-function, and looks *nearly* right, which is a miserable
        # thing to debug. An explicit shader already suppresses the auto one.
        base.render.setShader(self.shader)
        self._apply_shader_inputs(base.render)

        self.buffer = self._make_buffer()
        self.texture = Texture("ps1frame")
        self.card: NodePath | None = None
        self.hud2d = NodePath("hud2d")
        self.hud2d.setDepthTest(False)
        self.hud2d.setDepthWrite(False)

        if self.buffer is None:
            # No offscreen buffer (rare, but some software GL stacks refuse).
            # Draw straight to the window; the look degrades to "correct but
            # sharp", and the game is still playable.
            self.scene_cam = base.cam
            self._setup_hud_on(base.win, sort=20)
            self.lowres = False
        else:
            self.lowres = True
            self.buffer.addRenderTexture(
                self.texture, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPColor
            )
            self.buffer.setClearColor(SKY)
            self.texture.setMagfilter(SamplerState.FT_nearest)
            self.texture.setMinfilter(SamplerState.FT_nearest)

            lens = PerspectiveLens()
            lens.setAspectRatio(ASPECT)
            lens.setFov(72.0)
            lens.setNearFar(0.35, 900.0)
            self.scene_cam = base.makeCamera(self.buffer, lens=lens, camName="ps1cam")
            base.camNode.setActive(False)  # the window draws only the blit card

            self._make_blit_card()
            self._setup_hud_on(self.buffer, sort=20)

        base.accept("window-event", self._on_window_event)

    # -- construction helpers ---------------------------------------------

    def _apply_shader_inputs(self, np: NodePath) -> None:
        np.setShaderInput("jitterRes", (float(RES_X), float(RES_Y)))
        np.setShaderInput("lightDir", LIGHT_DIR)
        np.setShaderInput("lightColor", LIGHT_COLOR)
        np.setShaderInput("ambient", AMBIENT)
        np.setShaderInput("fogColor", SKY)
        np.setShaderInput("fogRange", (FOG_NEAR, FOG_FAR))
        np.setShaderInput("emissive", 0.0)  # default; individual nodes override

    def _make_buffer(self):
        props = FrameBufferProperties()
        props.setRgbColor(True)
        props.setRgbaBits(8, 8, 8, 0)
        props.setDepthBits(24)
        return self.base.graphicsEngine.makeOutput(
            self.base.pipe,
            "ps1buffer",
            -10,
            props,
            WindowProperties.size(RES_X, RES_Y),
            GraphicsPipe.BFRefuseWindow,
            self.base.win.getGsg(),
            self.base.win,
        )

    def _make_blit_card(self) -> None:
        maker = CardMaker("ps1blit")
        maker.setFrameFullscreenQuad()
        card = self.base.render2d.attachNewNode(maker.generate())
        card.setTexture(self.texture)
        card.setDepthTest(False)
        card.setDepthWrite(False)
        card.setBin("background", 0)
        self.card = card
        self._letterbox()

    def _setup_hud_on(self, output, sort: int) -> None:
        """Attach an orthographic HUD camera to a second display region.

        Clearing colour is off so the HUD composites over the frame already
        drawn; clearing depth is on so HUD geometry never fights the world's.
        """
        lens = OrthographicLens()
        lens.setFilmSize(2.0 * ASPECT, 2.0)
        lens.setNearFar(-100.0, 100.0)
        camera = Camera("hudcam")
        camera.setLens(lens)
        self.hud_cam = self.hud2d.attachNewNode(camera)

        region = output.makeDisplayRegion(0.0, 1.0, 0.0, 1.0)
        region.setSort(sort)
        region.setClearColorActive(False)
        region.setClearDepthActive(True)
        region.setCamera(self.hud_cam)
        self.hud_region = region

    # -- window plumbing ---------------------------------------------------

    def _on_window_event(self, window) -> None:
        self._letterbox()

    def _letterbox(self) -> None:
        """Scale the blit card so 320x240 pixels stay square in any window."""
        if self.card is None or not self.base.win:
            return
        # getXSize/getYSize rather than getProperties(): an offscreen host is
        # a GraphicsBuffer, which has the sizes but not the window properties.
        width, height = self.base.win.getXSize(), self.base.win.getYSize()
        if width <= 0 or height <= 0:
            return
        window_aspect = width / height
        if window_aspect > ASPECT:
            self.card.setScale(ASPECT / window_aspect, 1.0, 1.0)
        else:
            self.card.setScale(1.0, 1.0, window_aspect / ASPECT)

    def screenshot_image(self) -> PNMImage:
        """Pull the low-res frame back as a :class:`PNMImage`.

        Used by the headless test to build its contact sheet. Reads through
        ``GraphicsOutput.getScreenshot`` rather than ``extractTextureData``:
        with ``RTMBindOrCopy`` the texture *is* the framebuffer attachment,
        and extracting it hands back whatever was last copied to RAM, which
        on this driver is a frame or more stale. ``getScreenshot`` forces the
        read, which is what a screenshot is for.
        """
        image = PNMImage()
        source = self.buffer if self.lowres else self.base.win
        source.getScreenshot(image)
        return image


def flat_shade(
    np: NodePath,
    texture: Texture,
    tint: tuple[float, float, float] = (1, 1, 1),
    emissive: float = 0.0,
) -> None:
    """Prepare a node for the PS1 shader: one texture, a tint, a lighting mode.

    Kept here rather than in :mod:`models` so that everything which knows
    about ``SamplerState`` and the shader's uniform names lives in the same
    file as the shader itself. Every node needs a texture -- the fragment
    shader samples ``p3d_Texture0`` unconditionally.
    """
    np.setTexture(texture, 1)
    np.setColorScale(tint[0], tint[1], tint[2], 1.0)
    np.setShaderInput("emissive", emissive)


def fog_distance(pos_a, pos_b) -> float:
    """Straight-line distance, exposed so gameplay can match the fog cut-off."""
    return math.dist((pos_a.x, pos_a.y, pos_a.z), (pos_b.x, pos_b.y, pos_b.z))
