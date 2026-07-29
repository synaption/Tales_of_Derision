from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame


@dataclass
class MaterialSettings:
    ambient: int = 105
    diffuse_radius: int = 430
    gloss_radius: int = 145
    paper_sheen: float = 0.16
    ink_wetness: float = 0.78


class ParchmentMaterial:
    """Pure-Pygame approximation of lit parchment and glossy fresh ink.

    The paper and ink are rendered as separate materials:
      * parchment: broad diffuse light + weak, rough sheen
      * ink: dark diffuse layer + tighter specular hotspot + edge glint

    No NumPy, shaders, OpenGL, or external texture files are required.
    """

    LIGHT_SCALE = 4  # Build gradients at low resolution, then smooth-scale them.

    def __init__(
        self,
        size: tuple[int, int],
        settings: MaterialSettings | None = None,
        seed: int = 7,
    ) -> None:
        self.size = size
        self.settings = settings or MaterialSettings()
        self.seed = seed
        self.paper = self._make_paper(seed)
        self.paper_micro_spec = self._make_paper_micro_spec(seed + 1)
        self.vignette = self._make_vignette()

    def regenerate(self) -> None:
        self.seed += 1
        self.paper = self._make_paper(self.seed)
        self.paper_micro_spec = self._make_paper_micro_spec(self.seed + 1)

    def render(
        self,
        ink_surface: pygame.Surface,
        ink_mask: pygame.Surface,
        light_positions: list[tuple[int, int]],
    ) -> pygame.Surface:
        """Return a lit parchment frame containing the supplied ink layer.

        ink_surface:
            SRCALPHA surface containing the dark ink artwork.

        ink_mask:
            Opaque RGB surface: black outside ink, white/gray inside ink.
            Anti-aliased gray edges are supported.

        light_positions:
            One or more light source positions. Their diffuse and specular
            contributions are combined (brightest wins) so multiple lights
            don't blow out the parchment where they overlap.
        """
        if not light_positions:
            light_positions = [(self.size[0] // 2, self.size[1] // 2)]

        diffuse: pygame.Surface | None = None
        specular: pygame.Surface | None = None
        rim = pygame.Surface(self.size, pygame.SRCALPHA)
        rim.fill((0, 0, 0, 0))

        for position in light_positions:
            pos_diffuse = self._radial_map(
                position,
                self.settings.diffuse_radius,
                self.settings.ambient,
                255,
            )
            pos_specular = self._radial_map(
                position,
                self.settings.gloss_radius,
                0,
                255,
                exponent=2.6,
            )
            if diffuse is None:
                diffuse, specular = pos_diffuse, pos_specular
            else:
                diffuse.blit(pos_diffuse, (0, 0), special_flags=pygame.BLEND_RGB_MAX)
                specular.blit(pos_specular, (0, 0), special_flags=pygame.BLEND_RGB_MAX)

            # Thin directional glint along the edge facing this light. This is
            # the strongest cue that the ink is wet and slightly raised.
            edge = self._light_facing_edge(ink_mask, position, thickness=3)
            edge.blit(pos_specular, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
            self._tint_and_scale_in_place(
                edge,
                tint=(255, 235, 198),
                amount=0.75 * self.settings.ink_wetness,
            )
            rim.blit(edge, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        # Rough parchment: broad diffuse response.
        frame = self.paper.copy()
        frame.blit(diffuse, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

        # Very weak paper reflection. A static fibrous map is revealed only
        # around the light, giving the paper a rough rather than glassy sheen.
        paper_glint = self.paper_micro_spec.copy()
        paper_glint.blit(specular, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        self._scale_rgb_in_place(paper_glint, self.settings.paper_sheen)
        frame.blit(paper_glint, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        # Ink receives the same broad light but remains much darker than paper.
        lit_ink = ink_surface.copy()
        lit_ink.blit(diffuse, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        frame.blit(lit_ink, (0, 0))

        # Fresh ink: compact glossy hotspot, clipped to the ink mask.
        ink_gloss = specular.copy()
        ink_gloss.blit(ink_mask, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        self._tint_and_scale_in_place(
            ink_gloss,
            tint=(255, 226, 178),
            amount=0.52 * self.settings.ink_wetness,
        )
        frame.blit(ink_gloss, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        frame.blit(rim, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        # Old parchment is normally darker around its perimeter.
        frame.blit(self.vignette, (0, 0))
        return frame

    def _make_paper(self, seed: int) -> pygame.Surface:
        rng = random.Random(seed)
        width, height = self.size
        paper = pygame.Surface(self.size).convert()
        paper.fill((222, 199, 151))

        # Low-frequency stains, produced at low resolution for soft edges.
        small_size = (max(1, width // 4), max(1, height // 4))
        stains = pygame.Surface(small_size, pygame.SRCALPHA)
        stains.fill((0, 0, 0, 0))
        for _ in range(48):
            x = rng.randrange(small_size[0])
            y = rng.randrange(small_size[1])
            radius = rng.randint(8, 42)
            if rng.random() < 0.68:
                color = (88, 52, 21, rng.randint(3, 13))
            else:
                color = (255, 241, 201, rng.randint(3, 11))
            pygame.draw.circle(stains, color, (x, y), radius)
        stains = pygame.transform.smoothscale(stains, self.size)
        paper.blit(stains, (0, 0))

        # Fine pigment grain. This is generated once, not every frame.
        pixels = pygame.PixelArray(paper)
        for _ in range((width * height) // 5):
            x = rng.randrange(width)
            y = rng.randrange(height)
            base = paper.unmap_rgb(pixels[x, y])
            delta = rng.randint(-11, 11)
            pixels[x, y] = (
                max(0, min(255, base.r + delta)),
                max(0, min(255, base.g + delta)),
                max(0, min(255, base.b + delta)),
            )
        del pixels

        # Fibers and scratches.
        fibers = pygame.Surface(self.size, pygame.SRCALPHA)
        for _ in range(1900):
            x = rng.randrange(width)
            y = rng.randrange(height)
            length = rng.randint(3, 19)
            angle = rng.uniform(-0.22, 0.22)
            end = (
                int(x + math.cos(angle) * length),
                int(y + math.sin(angle) * length),
            )
            if rng.random() < 0.56:
                color = (255, 243, 211, rng.randint(5, 18))
            else:
                color = (92, 57, 27, rng.randint(3, 12))
            pygame.draw.aaline(fibers, color, (x, y), end)
        paper.blit(fibers, (0, 0))
        return paper

    def _make_paper_micro_spec(self, seed: int) -> pygame.Surface:
        """A subtle fibrous map used only for the paper's reflection."""
        rng = random.Random(seed)
        width, height = self.size
        scale = 2
        small_size = (max(1, width // scale), max(1, height // scale))
        micro = pygame.Surface(small_size).convert()
        micro.fill((0, 0, 0))

        for _ in range(1350):
            x = rng.randrange(small_size[0])
            y = rng.randrange(small_size[1])
            length = rng.randint(1, 8)
            level = rng.randint(24, 90)
            pygame.draw.aaline(
                micro,
                (level, level, level),
                (x, y),
                (x + length, y + rng.choice((-1, 0, 1))),
            )

        return pygame.transform.smoothscale(micro, self.size)

    def _make_vignette(self) -> pygame.Surface:
        width, height = self.size
        vignette = pygame.Surface(self.size, pygame.SRCALPHA)
        vignette.fill((0, 0, 0, 0))
        border = min(width, height) // 7
        for i in range(border):
            t = 1.0 - i / max(1, border - 1)
            alpha = int(2 + 34 * t * t)
            pygame.draw.rect(
                vignette,
                (65, 35, 14, alpha),
                pygame.Rect(i, i, width - 2 * i, height - 2 * i),
                1,
            )
        return vignette

    def _radial_map(
        self,
        center: tuple[int, int],
        radius: int,
        outer_value: int,
        inner_value: int,
        exponent: float = 1.45,
    ) -> pygame.Surface:
        width, height = self.size
        scale = self.LIGHT_SCALE
        small_size = (max(1, width // scale), max(1, height // scale))
        small = pygame.Surface(small_size).convert()
        small.fill((outer_value, outer_value, outer_value))

        cx = center[0] // scale
        cy = center[1] // scale
        small_radius = max(1, radius // scale)

        # Draw large to small so each inner circle overwrites the previous one.
        for r in range(small_radius, 0, -1):
            inward = 1.0 - r / small_radius
            value = int(
                outer_value
                + (inner_value - outer_value) * (inward ** exponent)
            )
            pygame.draw.circle(small, (value, value, value), (cx, cy), r)

        return pygame.transform.smoothscale(small, self.size)

    def _light_facing_edge(
        self,
        mask: pygame.Surface,
        light_position: tuple[int, int],
        thickness: int,
    ) -> pygame.Surface:
        center = pygame.Vector2(self.size[0] / 2, self.size[1] / 2)
        direction = pygame.Vector2(light_position) - center
        if direction.length_squared() == 0:
            direction = pygame.Vector2(-1, -1)
        direction.scale_to_length(thickness)

        # Move a copy opposite the light direction. Subtracting it leaves the
        # edge that faces the light.
        shifted = pygame.Surface(self.size).convert()
        shifted.fill((0, 0, 0))
        offset = (-round(direction.x), -round(direction.y))
        shifted.blit(mask, offset)

        edge = mask.copy()
        edge.blit(shifted, (0, 0), special_flags=pygame.BLEND_RGB_SUB)
        return edge

    @staticmethod
    def _scale_rgb_in_place(surface: pygame.Surface, amount: float) -> None:
        amount = max(0.0, min(1.0, amount))
        factor = int(255 * amount)
        multiplier = pygame.Surface(surface.get_size()).convert()
        multiplier.fill((factor, factor, factor))
        surface.blit(multiplier, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

    @staticmethod
    def _tint_and_scale_in_place(
        surface: pygame.Surface,
        tint: tuple[int, int, int],
        amount: float,
    ) -> None:
        amount = max(0.0, min(1.0, amount))
        multiplier = pygame.Surface(surface.get_size()).convert()
        multiplier.fill(
            (
                int(tint[0] * amount),
                int(tint[1] * amount),
                int(tint[2] * amount),
            )
        )
        surface.blit(multiplier, (0, 0), special_flags=pygame.BLEND_RGB_MULT)


def make_demo_ink(size: tuple[int, int]) -> tuple[pygame.Surface, pygame.Surface]:
    """Create matching ink artwork and an RGB gloss mask."""
    width, height = size
    ink = pygame.Surface(size, pygame.SRCALPHA)
    mask = pygame.Surface(size).convert()
    mask.fill((0, 0, 0))

    title_font = pygame.font.SysFont("serif", 58, bold=True)
    body_font = pygame.font.SysFont("serif", 28)
    small_font = pygame.font.SysFont("serif", 20, italic=True)

    def draw_text(
        text: str,
        font: pygame.font.Font,
        position: tuple[int, int],
        ink_color: tuple[int, int, int, int],
    ) -> None:
        ink_text = font.render(text, True, ink_color)
        mask_text = font.render(text, True, (255, 255, 255))
        ink.blit(ink_text, position)
        mask.blit(mask_text, position)

    draw_text("The Cartographer's Ledger", title_font, (90, 72), (30, 20, 13, 238))
    draw_text(
        "Move the mouse: one light, two material responses.",
        body_font,
        (94, 152),
        (34, 22, 14, 232),
    )
    draw_text(
        "The parchment is rough; the fresh ink is smooth and reflective.",
        small_font,
        (96, 196),
        (43, 26, 15, 225),
    )

    ink_color = (31, 19, 12, 240)
    white = (255, 255, 255)

    # Decorative divider.
    pygame.draw.line(ink, ink_color, (95, 245), (width - 95, 245), 5)
    pygame.draw.line(mask, white, (95, 245), (width - 95, 245), 5)
    pygame.draw.circle(ink, ink_color, (width // 2, 245), 13, 3)
    pygame.draw.circle(mask, white, (width // 2, 245), 13, 3)

    # A hand-drawn river/path made from many connected strokes.
    points: list[tuple[int, int]] = []
    for x in range(120, width - 120, 10):
        y = 390 + int(52 * math.sin(x * 0.018)) + int(16 * math.sin(x * 0.053))
        points.append((x, y))
    pygame.draw.lines(ink, ink_color, False, points, 12)
    pygame.draw.lines(mask, white, False, points, 12)

    # Compass rose.
    center = (width - 190, height - 155)
    pygame.draw.circle(ink, ink_color, center, 72, 5)
    pygame.draw.circle(mask, white, center, 72, 5)
    for angle in range(0, 360, 45):
        vector = pygame.Vector2(0, -62).rotate(angle)
        endpoint = (center[0] + vector.x, center[1] + vector.y)
        pygame.draw.line(ink, ink_color, center, endpoint, 5)
        pygame.draw.line(mask, white, center, endpoint, 5)
    pygame.draw.circle(ink, ink_color, center, 12)
    pygame.draw.circle(mask, white, center, 12)

    # Dense ink pools catch the strongest highlight.
    for position, radius in [((165, 555), 30), ((245, 520), 18), ((325, 565), 24)]:
        pygame.draw.circle(ink, (24, 14, 9, 245), position, radius)
        pygame.draw.circle(mask, white, position, radius)

    return ink, mask


def main() -> None:
    pygame.init()
    size = (1000, 700)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Pure-Pygame Parchment and Fresh Ink")
    clock = pygame.time.Clock()

    material = ParchmentMaterial(size)
    ink, ink_mask = make_demo_ink(size)

    info_font = pygame.font.SysFont("sans", 18)

    # Camera: zoom is a scale factor, pan is the screen-space offset of the
    # content's top-left corner.
    zoom = 1.0
    min_zoom, max_zoom = 0.35, 4.0
    pan = pygame.Vector2(0, 0)
    panning = False
    pickup_radius_px = 22.0

    def screen_to_content(screen_pos: tuple[int, int]) -> pygame.Vector2:
        return (pygame.Vector2(screen_pos) - pan) / zoom

    def content_to_screen(content_pos: pygame.Vector2) -> pygame.Vector2:
        return content_pos * zoom + pan

    # One light always starts out "carried" (following the mouse), matching
    # the original mouse-is-the-light demo. Clicking places it; clicking
    # again with an empty hand picks a light back up.
    carried_light: pygame.Vector2 | None = screen_to_content(pygame.mouse.get_pos())
    placed_lights: list[pygame.Vector2] = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    material.regenerate()
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                    material.settings.ink_wetness = max(
                        0.0, material.settings.ink_wetness - 0.08
                    )
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS):
                    material.settings.ink_wetness = min(
                        1.0, material.settings.ink_wetness + 0.08
                    )
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    content_pos = screen_to_content(event.pos)
                    shift_held = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    if shift_held:
                        # Drop an extra light without letting go of whatever
                        # is currently being carried.
                        placed_lights.append(content_pos)
                    elif carried_light is not None:
                        # Place the carried light where it stands.
                        placed_lights.append(carried_light)
                        carried_light = None
                    else:
                        # Empty-handed: pick up whichever placed light is
                        # closest to the click, or start a fresh one.
                        nearest_index = None
                        nearest_distance = pickup_radius_px
                        for i, light in enumerate(placed_lights):
                            distance = content_to_screen(light).distance_to(event.pos)
                            if distance <= nearest_distance:
                                nearest_distance = distance
                                nearest_index = i
                        if nearest_index is not None:
                            placed_lights.pop(nearest_index)
                        carried_light = content_pos
                elif event.button == 2:
                    panning = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    panning = False
            elif event.type == pygame.MOUSEMOTION:
                if panning:
                    pan += pygame.Vector2(event.rel)
            elif event.type == pygame.MOUSEWHEEL:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    material.settings.ink_wetness = max(
                        0.0,
                        min(1.0, material.settings.ink_wetness + event.y * 0.05),
                    )
                else:
                    # Zoom, keeping the point under the cursor stationary.
                    mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
                    content_before = screen_to_content(mouse_pos)
                    zoom = max(min_zoom, min(max_zoom, zoom * (1.1**event.y)))
                    pan = mouse_pos - content_before * zoom

        if carried_light is not None:
            carried_light = screen_to_content(pygame.mouse.get_pos())

        light_positions = list(placed_lights)
        if carried_light is not None:
            light_positions.append(carried_light)

        frame = material.render(ink, ink_mask, light_positions)

        if zoom == 1.0 and pan == pygame.Vector2(0, 0):
            display_frame = frame
        else:
            scaled_size = (
                max(1, round(size[0] * zoom)),
                max(1, round(size[1] * zoom)),
            )
            display_frame = pygame.transform.smoothscale(frame, scaled_size)

        screen.fill((12, 9, 6))
        screen.blit(display_frame, pan)

        # Show every light source unobtrusively; the carried one glows a
        # touch brighter since it's still "live".
        for light in placed_lights:
            pygame.draw.circle(screen, (255, 239, 199), content_to_screen(light), 5, 1)
        if carried_light is not None:
            pygame.draw.circle(
                screen, (255, 250, 225), content_to_screen(carried_light), 6, 2
            )

        # UI is drawn after the material pass so it stays easy to read.
        label = (
            f"Click = place/pick up light   Shift+Click = drop another   "
            f"Wheel = zoom   Shift+Wheel = ink wetness ({material.settings.ink_wetness:.2f})   "
            f"Middle-drag = pan   R = new paper   Esc = quit"
        )
        text = info_font.render(label, True, (245, 236, 214))
        backing = pygame.Surface((text.get_width() + 22, text.get_height() + 12), pygame.SRCALPHA)
        backing.fill((25, 19, 14, 185))
        screen.blit(backing, (12, size[1] - backing.get_height() - 12))
        screen.blit(text, (23, size[1] - text.get_height() - 18))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()