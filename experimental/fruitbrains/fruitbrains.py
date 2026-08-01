"""Fruit Brains -- a tiny animated Pygame toy.

Run from anywhere with::

    python3 experimental/fruitbrains/fruitbrains.py

The original 32 px fruit art is kept intact. Faces, moods and rubber-hose limbs
are drawn at runtime, so every fruit has a full walk cycle and a face that can
swing between eight emotions. Happy is home base -- everything else decays back
to it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import pygame


WIDTH, HEIGHT = 960, 600
FPS = 60
GROUND_Y = 500
INK = (35, 30, 38)
CREAM = (255, 247, 218)
BLUSH = (247, 146, 152)
TEAR = (126, 197, 236)
ROOT = Path(__file__).resolve().parent
FRUIT_DIR = ROOT / "Fruits_Separated"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def thick_line(surface: pygame.Surface, color, points, width: int = 5) -> None:
    """A round-ended line, closer to the hand-inked reference than polygons."""
    int_points = [(round(x), round(y)) for x, y in points]
    pygame.draw.lines(surface, color, False, int_points, width)
    radius = width // 2
    for point in (int_points[0], int_points[-1]):
        pygame.draw.circle(surface, color, point, radius)


@dataclass(frozen=True)
class Emotion:
    """Everything that makes one mood look different from another."""

    label: str
    eyes: str       # arc | open | wide | half | heart | teary | swirl
    mouth: str      # smile | grin | frown | wavy | round | grimace | smirk
    brow: tuple     # (inner_dy, outer_dy) -- positive lowers that end
    arms: str       # swing | wave | droop | fists | up
    thought: str
    bounce: float   # extra idle bob, in pixels
    pace: float     # wander speed multiplier
    blush: bool = False


EMOTIONS: dict[str, Emotion] = {
    "happy":     Emotion("happy",     "open",  "smile",   (0, -3),   "swing", "♪", 0.8, 1.00),
    "excited":   Emotion("excited",   "wide",  "grin",    (-4, -5),  "wave",  "!",      2.2, 1.40, blush=True),
    "love":      Emotion("in love",   "heart", "smile",   (-2, -4),  "wave",  "♥", 1.4, 0.85, blush=True),
    "sad":       Emotion("sad",       "teary", "frown",   (-5, 3),   "droop", "...",    0.1, 0.50),
    "angry":     Emotion("angry",     "wide",  "grimace", (5, -5),   "fists", "#!?",    0.9, 1.25),
    "surprised": Emotion("surprised", "wide",  "round",   (-6, -6),  "up",    "?!",     0.5, 0.75),
    "sleepy":    Emotion("sleepy",    "half",  "smirk",   (3, 2),    "droop", "z",      0.2, 0.35),
    "worried":   Emotion("worried",   "open",  "wavy",    (-4, 2),   "droop", "?",      0.3, 0.65),
}
MOOD_KEYS = list(EMOTIONS)
# Moods a fruit will wander into on its own. Happy stays the default.
IDLE_MOODS = ("excited", "love", "sad", "surprised", "sleepy", "worried")


class Fruit:
    def __init__(self, path: Path, pos: tuple[float, float], index: int) -> None:
        self.name = path.stem
        raw = pygame.image.load(path).convert_alpha()
        self.image = pygame.transform.scale(raw, (96, 96))
        self.pos = pygame.Vector2(pos)
        self.velocity = pygame.Vector2()
        self.facing = -1 if index % 3 == 0 else 1
        self.phase = index * 0.83
        self.blink_at = 1.2 + index * 0.31
        self.blink = 0.0
        self.emotion = "happy"
        self.mood_left = 0.0          # time before falling back to happy
        self.mood_at = 4.0 + index * 1.7
        self.gesture = 0.0            # short burst of arm/bubble action
        self.walk_target = self.pos.x

    @property
    def mood(self) -> Emotion:
        return EMOTIONS[self.emotion]

    def feel(self, emotion: str, duration: float = 3.5) -> None:
        self.emotion = emotion
        self.mood_left = duration
        self.gesture = min(1.1, duration)

    def update(self, dt: float, now: float, controlled: bool, keys) -> None:
        self.blink = max(0.0, self.blink - dt)
        self.gesture = max(0.0, self.gesture - dt)
        if now >= self.blink_at:
            self.blink = 0.14
            self.blink_at = now + random.uniform(1.8, 4.8)

        # Moods are temporary; happy is what a fruit relaxes back into.
        if self.mood_left > 0:
            self.mood_left -= dt
            if self.mood_left <= 0:
                self.emotion = "happy"
        elif not controlled and now >= self.mood_at:
            self.feel(random.choice(IDLE_MOODS), random.uniform(2.5, 5.5))
            self.mood_at = now + random.uniform(6.0, 14.0)

        pace = self.mood.pace
        if controlled:
            direction = pygame.Vector2(
                int(keys[pygame.K_RIGHT] or keys[pygame.K_d])
                - int(keys[pygame.K_LEFT] or keys[pygame.K_a]),
                int(keys[pygame.K_DOWN] or keys[pygame.K_s])
                - int(keys[pygame.K_UP] or keys[pygame.K_w]),
            )
            if direction.length_squared():
                direction = direction.normalize()
                self.velocity += direction * 700 * dt
        elif abs(self.pos.x - self.walk_target) < 8:
            self.walk_target = random.uniform(70, WIDTH - 70)
        else:
            self.velocity.x += (1 if self.walk_target > self.pos.x else -1) * 105 * pace * dt

        self.velocity *= 0.86 ** (dt * 60)
        self.velocity.x = clamp(self.velocity.x, -180 * pace, 180 * pace)
        self.velocity.y = clamp(self.velocity.y, -115, 115)
        self.pos += self.velocity * dt
        self.pos.x = clamp(self.pos.x, 52, WIDTH - 52)
        self.pos.y = clamp(self.pos.y, 360, GROUND_Y - 5)
        if abs(self.velocity.x) > 8:
            self.facing = 1 if self.velocity.x > 0 else -1

    def draw(self, surface: pygame.Surface, now: float, selected: bool) -> None:
        mood = self.mood
        speed = self.velocity.length()
        walking = clamp(speed / 110, 0, 1)
        cycle = now * (7 + walking * 4) + self.phase
        idle_bob = math.sin(now * 3.1 + self.phase) * mood.bounce
        bob = math.sin(cycle * 2) * (2.7 * walking) + idle_bob - (7 if self.gesture else 0)
        x, y = self.pos.x, self.pos.y + bob
        foot_y = self.pos.y + 57

        # Shadow and selection halo.
        pygame.draw.ellipse(surface, (73, 104, 78), (x - 37, foot_y - 4, 74, 15))
        if selected:
            pulse = 48 + math.sin(now * 5) * 3
            pygame.draw.circle(surface, (255, 235, 116), (round(x), round(y)), round(pulse), 3)

        # Rubber-hose limbs sit behind the fruit body.
        stride = math.sin(cycle) * 16 * walking
        left_foot = (x - 16 + stride, foot_y)
        right_foot = (x + 16 - stride, foot_y)
        thick_line(surface, INK, [(x - 15, y + 32), (x - 17 - stride * .2, y + 47), left_foot], 5)
        thick_line(surface, INK, [(x + 15, y + 32), (x + 17 + stride * .2, y + 47), right_foot], 5)
        thick_line(surface, INK, [left_foot, (left_foot[0] - 8 * self.facing, foot_y)], 5)
        thick_line(surface, INK, [right_foot, (right_foot[0] - 8 * self.facing, foot_y)], 5)

        self._draw_arms(surface, x, y, now, cycle, walking, mood)

        rect = self.image.get_rect(center=(round(x), round(y)))
        surface.blit(self.image, rect)
        self._draw_face(surface, x, y, now, walking, mood)
        self._draw_mood_fx(surface, x, y, now, mood)

        if self.gesture:
            font = pygame.font.Font(None, 29)
            bubble = font.render(mood.thought, True, INK)
            bubble_rect = bubble.get_rect(center=(x + 55, y - 61))
            pygame.draw.circle(surface, CREAM, bubble_rect.center, 22)
            pygame.draw.circle(surface, INK, bubble_rect.center, 22, 2)
            surface.blit(bubble, bubble_rect)

    def _draw_arms(self, surface, x, y, now, cycle, walking, mood: Emotion) -> None:
        swing = math.sin(cycle) * 12 * walking
        pose = mood.arms if (self.gesture or mood.arms in ("droop", "fists")) else "swing"
        flap = math.sin(now * 12) * 6

        if pose == "wave":
            left = (x - 50, y + 13 - swing)
            right = (x + 50, y - 12 + flap)
            left_mid, right_mid = (x - 42, y + 18), (x + 44, y - 2)
        elif pose == "droop":
            left = (x - 40, y + 34)
            right = (x + 40, y + 34)
            left_mid, right_mid = (x - 40, y + 16), (x + 40, y + 16)
        elif pose == "fists":
            shake = math.sin(now * 22) * 2
            left = (x - 34, y - 6 + shake)
            right = (x + 34, y - 6 - shake)
            left_mid, right_mid = (x - 44, y + 14), (x + 44, y + 14)
        elif pose == "up":
            left = (x - 44, y - 26 + flap * .3)
            right = (x + 44, y - 26 - flap * .3)
            left_mid, right_mid = (x - 46, y - 2), (x + 46, y - 2)
        else:  # swing
            left = (x - 50, y + 13 - swing)
            right = (x + 50, y + 13 + swing)
            left_mid, right_mid = (x - 42, y + 18), (x + 42, y + 18)

        thick_line(surface, INK, [(x - 32, y + 3), left_mid, left], 5)
        thick_line(surface, INK, [(x + 32, y + 3), right_mid, right], 5)
        self._draw_hand(surface, left, -1, pose == "fists")
        self._draw_hand(surface, right, 1, pose == "fists")

    @staticmethod
    def _draw_hand(surface: pygame.Surface, hand, facing: int, fist: bool = False) -> None:
        hx, hy = hand
        pygame.draw.circle(surface, INK, (round(hx), round(hy)), 7 if fist else 5)
        if fist:
            return
        for dy in (-5, 0, 5):
            thick_line(surface, INK, [(hx, hy), (hx + facing * 7, hy + dy)], 2)

    def _draw_face(self, surface, x, y, now, walking, mood: Emotion) -> None:
        look = math.sin(now * .75 + self.phase) * 1.5
        eye_y = y - 5
        style = "arc" if (self.blink and mood.eyes != "half") else mood.eyes

        for side, eye_x in ((-1, x - 12), (1, x + 12)):
            self._draw_eye(surface, eye_x, eye_y, side, style, look, now)
            inner_dy, outer_dy = mood.brow
            brow_y = eye_y - 13
            inner = (eye_x + side * 6, brow_y + inner_dy)
            outer = (eye_x - side * 7, brow_y + outer_dy)
            thick_line(surface, INK, [inner, outer], 3)

        if mood.blush:
            for cheek_x in (x - 22, x + 22):
                pygame.draw.ellipse(surface, BLUSH, (cheek_x - 8, eye_y + 12, 16, 9))

        self._draw_mouth(surface, x, y + 16, mood.mouth, walking, now)

    @staticmethod
    def _draw_eye(surface, eye_x, eye_y, side, style, look, now) -> None:
        if style == "arc":       # happy closed-eye curve
            pygame.draw.arc(surface, INK, (eye_x - 7, eye_y - 4, 14, 12), math.pi * .15, math.pi * .85, 3)
            return
        if style == "half":      # sleepy lids
            pygame.draw.ellipse(surface, CREAM, (eye_x - 6, eye_y - 4, 12, 11))
            pygame.draw.ellipse(surface, INK, (eye_x - 4, eye_y - 1, 7, 8))
            thick_line(surface, INK, [(eye_x - 7, eye_y - 3), (eye_x + 7, eye_y - 1)], 3)
            return
        if style == "heart":
            for dx in (-3, 3):
                pygame.draw.circle(surface, (233, 76, 96), (round(eye_x + dx), round(eye_y - 4)), 5)
            pygame.draw.polygon(surface, (233, 76, 96), (
                (eye_x - 7, eye_y - 2), (eye_x + 7, eye_y - 2), (eye_x, eye_y + 8)))
            pygame.draw.circle(surface, CREAM, (round(eye_x - 3), round(eye_y - 5)), 2)
            return

        wide = style == "wide"
        box = (eye_x - 8, eye_y - 11, 16, 22) if wide else (eye_x - 6, eye_y - 9, 12, 18)
        pygame.draw.ellipse(surface, CREAM, box)
        pupil_w, pupil_h = (6, 9) if wide else (7, 12)
        pygame.draw.ellipse(surface, INK, (eye_x - pupil_w / 2 + look, eye_y - pupil_h / 2, pupil_w, pupil_h))
        pygame.draw.circle(surface, CREAM, (round(eye_x - 1 + look), round(eye_y - 3)), 2)

        if style == "teary":     # a tear welling up and rolling down
            drop = (now * 34 + eye_x) % 46
            pygame.draw.circle(surface, TEAR, (round(eye_x + side * 8), round(eye_y + 6 + drop)), max(2, 5 - drop // 12))

    @staticmethod
    def _draw_mouth(surface, x, mouth_y, style, walking, now) -> None:
        if style == "grin":
            pygame.draw.ellipse(surface, INK, (x - 9, mouth_y - 3, 18, 17))
            pygame.draw.arc(surface, (243, 105, 112), (x - 5, mouth_y + 6, 10, 6), math.pi, math.tau, 3)
        elif style == "frown":
            pygame.draw.arc(surface, INK, (x - 8, mouth_y + 2, 16, 12), 0, math.pi, 3)
        elif style == "wavy":
            points = [(x - 9 + i * 3, mouth_y + 4 + math.sin(i * 1.6) * 2.5) for i in range(7)]
            thick_line(surface, INK, points, 3)
        elif style == "round":
            pygame.draw.circle(surface, INK, (round(x), round(mouth_y + 5)), 6)
        elif style == "grimace":
            pygame.draw.rect(surface, INK, (x - 10, mouth_y, 20, 11), border_radius=3)
            pygame.draw.line(surface, CREAM, (x - 8, mouth_y + 5), (x + 8, mouth_y + 5), 2)
            for tooth_x in (x - 4, x, x + 4):
                pygame.draw.line(surface, CREAM, (tooth_x, mouth_y + 1), (tooth_x, mouth_y + 9), 2)
        elif style == "smirk":
            pygame.draw.arc(surface, INK, (x - 2, mouth_y - 2, 13, 11), math.pi, math.tau, 3)
        else:                    # smile -- wider when trotting along
            width = 18 if walking > .15 else 16
            pygame.draw.arc(surface, INK, (x - width / 2, mouth_y - 5, width, 14), math.pi, math.tau, 3)

    def _draw_mood_fx(self, surface, x, y, now, mood: Emotion) -> None:
        """Little cartoon annotations floating around the head."""
        if self.emotion == "angry":
            for i in range(3):
                a = now * 2.5 + i * 2.1
                px, py = x + math.cos(a) * 34, y - 46 + math.sin(a * 1.3) * 5
                pygame.draw.circle(surface, (250, 233, 233), (round(px), round(py)), 6)
                pygame.draw.circle(surface, (232, 88, 88), (round(px), round(py)), 6, 2)
        elif self.emotion == "love":
            for i in range(3):
                t = (now * .8 + i / 3) % 1.0
                px, py = x + 30 + math.sin(t * 6 + i) * 8, y - 30 - t * 46
                size = round(7 * (1 - t * .5))
                for dx in (-size // 2, size // 2):
                    pygame.draw.circle(surface, (233, 76, 96), (round(px + dx), round(py)), max(2, size // 2))
                pygame.draw.polygon(surface, (233, 76, 96), (
                    (px - size, py), (px + size, py), (px, py + size * 1.4)))
        elif self.emotion == "sleepy":
            font = pygame.font.Font(None, 26)
            for i in range(3):
                t = (now * .5 + i / 3) % 1.0
                z = font.render("z", True, INK)
                surface.blit(z, (x + 26 + t * 18, y - 44 - t * 40))
        elif self.emotion == "worried":
            sweat = (now * 2) % 1.0
            pygame.draw.circle(surface, TEAR, (round(x + 26), round(y - 26 + sweat * 16)), 5)
        elif self.emotion == "surprised":
            for i in range(6):
                a = i * math.tau / 6 + now * 1.5
                r0, r1 = 44, 44 + 9 + math.sin(now * 9) * 3
                pygame.draw.line(surface, (255, 236, 130),
                                 (x + math.cos(a) * r0, y - 6 + math.sin(a) * r0),
                                 (x + math.cos(a) * r1, y - 6 + math.sin(a) * r1), 3)


def draw_background(surface: pygame.Surface, now: float) -> None:
    surface.fill((156, 218, 216))
    pygame.draw.circle(surface, (255, 230, 119), (820, 82), 45)
    for x, y, scale in ((115, 100, 1), (470, 73, .8), (710, 155, .65)):
        for dx, dy in ((0, 0), (28, -8), (56, 2)):
            pygame.draw.circle(surface, (245, 248, 229), (x + dx, y + dy), round(22 * scale))
    pygame.draw.rect(surface, (116, 185, 105), (0, 300, WIDTH, HEIGHT - 300))
    for i in range(18):
        x = (i * 71 + 29) % WIDTH
        sway = math.sin(now * 1.6 + i) * 3
        pygame.draw.line(surface, (76, 151, 80), (x, 520), (x + sway, 499), 3)
        pygame.draw.circle(surface, (255, 232, 119) if i % 2 else (246, 172, 179), (round(x + sway), 495), 4)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Fruit Brains")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    ui_font = pygame.font.Font(None, 25)
    small_font = pygame.font.Font(None, 21)
    title_font = pygame.font.Font(None, 42)
    random.seed(7)

    paths = sorted(FRUIT_DIR.glob("*.png"))
    if not paths:
        raise SystemExit(f"No fruit PNGs found in {FRUIT_DIR}")
    fruits = [Fruit(path, (85 + (i % 6) * 158, 390 + (i // 6) * 95), i) for i, path in enumerate(paths)]
    selected = 0
    mood_hotkeys = {getattr(pygame, f"K_{i + 1}"): name for i, name in enumerate(MOOD_KEYS)}
    running = True

    while running:
        dt = min(clock.tick(FPS) / 1000.0, .05)
        now = pygame.time.get_ticks() / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                selected = (selected + (-1 if event.mod & pygame.KMOD_SHIFT else 1)) % len(fruits)
            elif event.type == pygame.KEYDOWN and event.key in mood_hotkeys:
                name = mood_hotkeys[event.key]
                # Happy is the resting state, so setting it just clears the timer.
                fruits[selected].feel(name, 0.9 if name == "happy" else 6.0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                fruits[selected].feel("excited", 2.0)
                fruits[selected].velocity.y = -105
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                selected = min(range(len(fruits)), key=lambda i: fruits[i].pos.distance_to(event.pos))

        keys = pygame.key.get_pressed()
        for i, fruit in enumerate(fruits):
            fruit.update(dt, now, i == selected, keys)
        hero = fruits[selected]

        draw_background(screen, now)
        for i in sorted(range(len(fruits)), key=lambda n: fruits[n].pos.y):
            fruits[i].draw(screen, now, i == selected)

        pygame.draw.rect(screen, (33, 40, 48), (0, 0, WIDTH, 54))
        title = title_font.render("FRUIT BRAINS", True, (255, 228, 111))
        screen.blit(title, (20, 10))
        status = ui_font.render(f"{hero.name} is feeling {hero.mood.label}", True, CREAM)
        screen.blit(status, (250, 18))
        help_text = "arrows/WASD move  •  SPACE hop  •  TAB swap  •  " + "  ".join(
            f"{i + 1}:{EMOTIONS[name].label}" for i, name in enumerate(MOOD_KEYS))
        help_surface = small_font.render(help_text, True, CREAM)
        screen.blit(help_surface, (WIDTH - help_surface.get_width() - 18, 21))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
