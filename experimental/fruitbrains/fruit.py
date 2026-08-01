"""Fruit villagers: pixel-art bodies with HD faces, limbs and moods.

The body is 32 px art blitted at the scene's pixel scale.  Everything that
gives a fruit personality -- eyes, brows, mouth, rubber-hose arms and legs,
name tag, speech -- is drawn in screen space at full resolution, so zooming in
reveals a finer face instead of a bigger pixel.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from render import (Camera, PixelArt, blit_art, clamp, hd_arc, hd_circle,
                    hd_ellipse, hd_font, hd_line, hd_text)


INK = (35, 30, 38)
CREAM = (255, 247, 218)
BLUSH = (247, 146, 152)
TEAR = (126, 197, 236)

BODY = 64.0          # villager body size in world units (32 px art at 2x)
FEET = 38.0          # world units from body centre down to the soles
ART_SPACE = BODY / 96.0   # the face/limb art was tuned in a 96 px space

PLAYER_SPEED = 235.0      # world units/second -- about 7 tiles/s at a trot
WANDER_SPEED = 150.0      # villagers mooching about are a touch slower
BOUNCE = 13.0             # hop height at a full run, in old-space pixels


@dataclass(frozen=True)
class Emotion:
    """Everything that makes one mood look different from another."""

    label: str
    eyes: str       # arc | open | wide | half | heart | teary
    mouth: str      # smile | grin | frown | wavy | round | grimace | smirk
    brow: tuple     # (inner_dy, outer_dy) -- positive lowers that end
    arms: str       # swing | wave | droop | fists | up
    thought: str
    bounce: float   # extra idle bob, in old-space pixels
    pace: float     # wander speed multiplier
    blush: bool = False


EMOTIONS: dict[str, Emotion] = {
    "happy":     Emotion("happy",     "open",  "smile",   (0, -3),   "swing", "~",      0.8, 1.00),
    "excited":   Emotion("excited",   "wide",  "grin",    (-4, -5),  "wave",  "!",      2.2, 1.40, blush=True),
    "love":      Emotion("in love",   "heart", "smile",   (-2, -4),  "wave",  "<3",     1.4, 0.85, blush=True),
    "sad":       Emotion("sad",       "teary", "frown",   (-5, 3),   "droop", "...",    0.1, 0.50),
    "angry":     Emotion("angry",     "wide",  "grimace", (5, -5),   "fists", "#!?",    0.9, 1.25),
    "surprised": Emotion("surprised", "wide",  "round",   (-6, -6),  "up",    "?!",     0.5, 0.75),
    "sleepy":    Emotion("sleepy",    "half",  "smirk",   (3, 2),    "droop", "z",      0.2, 0.35),
    "worried":   Emotion("worried",   "open",  "wavy",    (-4, 2),   "droop", "?",      0.3, 0.65),
}
MOOD_KEYS = list(EMOTIONS)
# Moods a fruit wanders into on its own. Happy stays the default.
IDLE_MOODS = ("excited", "love", "sad", "surprised", "sleepy", "worried")

LINES: dict[str, tuple[str, ...]] = {
    "happy": ("Lovely day for a stroll, isn't it?",
              "I watered the flowers by the fountain today.",
              "You're looking especially ripe."),
    "excited": ("The market stall is OPEN! Let's go, let's go!",
                "I have SO much to tell you!"),
    "love": ("Someone left a flower on my doorstep...",
             "I just adore this little town."),
    "sad": ("Nobody sat on the bench with me today.",
            "I think I'm going a bit soft."),
    "angry": ("Somebody took my spot by the fountain.",
              "Do NOT talk to me about jam."),
    "surprised": ("Oh! You startled me!",
                  "Wait -- were you there the whole time?"),
    "sleepy": ("Mmh... five more minutes...",
               "The sun's so warm right here..."),
    "worried": ("Do you think it'll rain? I bruise, you know.",
                "I can't remember if I locked the door."),
}


class Fruit:
    def __init__(self, name: str, art: PixelArt, pos, index: int, scene: str = "town") -> None:
        self.name = name
        self.art = art
        self.scene = scene
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
        self.walk_target = pygame.Vector2(pos)
        self.repath = 0.0
        self.line = ""
        self.line_left = 0.0
        self.thinking = False     # set while a local model is writing their reply
        self._overlay = None      # screen-space handoff from draw() to draw_overlay()

    # -- state ---------------------------------------------------------------

    @property
    def mood(self) -> Emotion:
        return EMOTIONS[self.emotion]

    @property
    def feet(self) -> pygame.Vector2:
        return pygame.Vector2(self.pos.x, self.pos.y + FEET)

    def footprint(self) -> pygame.Rect:
        return pygame.Rect(round(self.pos.x - 14), round(self.pos.y + FEET - 12), 28, 14)

    @property
    def depth(self) -> float:
        return self.pos.y + FEET

    def feel(self, emotion: str, duration: float = 3.5) -> None:
        self.emotion = emotion
        self.mood_left = duration
        self.gesture = min(1.1, duration)

    def say(self, text: str, seconds: float = 4.5) -> None:
        self.line = text
        self.line_left = seconds

    def greet(self, other: "Fruit") -> None:
        """Answer someone who walked up and said hello."""
        mood = random.choice(("happy", "happy", "excited", "love", "surprised")
                             if self.emotion == "happy" else (self.emotion,))
        self.feel(mood, 4.0)
        self.say(random.choice(LINES[self.emotion]))
        self.facing = 1 if other.pos.x > self.pos.x else -1

    # -- simulation ----------------------------------------------------------

    def update(self, dt: float, now: float, control: pygame.Vector2 | None, scene) -> None:
        self.blink = max(0.0, self.blink - dt)
        self.gesture = max(0.0, self.gesture - dt)
        self.line_left = max(0.0, self.line_left - dt)
        if now >= self.blink_at:
            self.blink = 0.14
            self.blink_at = now + random.uniform(1.8, 4.8)

        # Moods are temporary; happy is what a fruit relaxes back into.
        if self.mood_left > 0:
            self.mood_left -= dt
            if self.mood_left <= 0:
                self.emotion = "happy"
        elif control is None and now >= self.mood_at:
            self.feel(random.choice(IDLE_MOODS), random.uniform(2.5, 5.5))
            self.mood_at = now + random.uniform(6.0, 14.0)

        pace = self.mood.pace
        if control is not None:
            if control.length_squared():
                self.velocity += control.normalize() * 1500 * dt
        else:
            self.repath -= dt
            if self.pos.distance_to(self.walk_target) < 12 or self.repath <= 0:
                self._pick_target(scene)
            step = self.walk_target - self.pos
            if step.length_squared():
                self.velocity += step.normalize() * 260 * pace * dt

        self.velocity *= 0.86 ** (dt * 60)
        limit = (PLAYER_SPEED if control is not None else WANDER_SPEED) * pace
        if self.velocity.length() > limit:
            self.velocity.scale_to_length(limit)
        self.velocity.y *= 0.82 if control is not None else 1.0   # gentle top-down feel

        self._move(self.velocity * dt, scene)
        if abs(self.velocity.x) > 6:
            self.facing = 1 if self.velocity.x > 0 else -1

    def _pick_target(self, scene) -> None:
        bounds = scene.bounds
        for _ in range(12):
            target = pygame.Vector2(random.uniform(bounds.left + 48, bounds.right - 48),
                                    random.uniform(bounds.top + 64, bounds.bottom - 48))
            probe = pygame.Rect(0, 0, 28, 14)
            probe.center = (target.x, target.y + FEET)
            if not any(probe.colliderect(s) for s in scene.solids):
                self.walk_target = target
                self.repath = random.uniform(4.0, 9.0)
                return
        self.walk_target = pygame.Vector2(self.pos)
        self.repath = 1.5

    def _move(self, delta: pygame.Vector2, scene) -> None:
        solids = scene.solids
        for axis in (0, 1):
            if not delta[axis]:
                continue
            self.pos[axis] += delta[axis]
            box = self.footprint()
            if any(box.colliderect(s) for s in solids):
                self.pos[axis] -= delta[axis]
                self.velocity[axis] = 0
        b = scene.bounds
        self.pos.x = clamp(self.pos.x, b.left + 20, b.right - 20)
        self.pos.y = clamp(self.pos.y, b.top + 8 - FEET + 40, b.bottom - FEET - 4)

    # -- drawing -------------------------------------------------------------

    def draw(self, surface: pygame.Surface, cam: Camera, now: float, selected: bool,
             show_name: bool = True) -> None:
        mood = self.mood
        k = cam.zoom * ART_SPACE          # old 96 px art space -> screen pixels
        walking = clamp(self.velocity.length() / (PLAYER_SPEED * 0.8), 0, 1)
        cycle = now * (8 + walking * 9) + self.phase

        # Every step is a little hop: `air` peaks mid-stride, so the body lifts,
        # stretches on the way up and squashes as it lands.
        air = abs(math.sin(cycle)) * walking
        idle_bob = math.sin(now * 3.1 + self.phase) * mood.bounce
        bob = -air * BOUNCE + idle_bob - (7 if self.gesture else 0)
        stretch = 1 + (air - 0.4) * 0.16 * walking

        wx, wy = self.pos.x, self.pos.y + bob * ART_SPACE
        x, y = cam.to_screen(wx, wy)
        foot_x, foot_y = cam.to_screen(self.pos.x, self.pos.y + FEET)
        foot_y -= air * 3 * cam.zoom       # toes come off the ground too

        def px(dx: float, dy: float) -> tuple[float, float]:
            return x + dx * k, y + dy * k

        # Shadow (HD -- a crisp ellipse at any zoom). It shrinks as they hop,
        # which is what actually sells the height.
        ground_y = cam.to_screen(0, self.pos.y + FEET)[1]
        tuck = 1 - air * 0.3
        shade = pygame.Surface((max(2, round(74 * k * tuck)), max(2, round(16 * k * tuck))),
                               pygame.SRCALPHA)
        pygame.draw.ellipse(shade, (40, 70, 45, round(90 * tuck)), shade.get_rect())
        surface.blit(shade, shade.get_rect(center=(foot_x, ground_y + 2 * k)))
        if selected:
            pulse = (46 + math.sin(now * 5) * 3) * k
            hd_ellipse(surface, (255, 235, 116),
                       (foot_x - pulse, ground_y - pulse * .34, pulse * 2, pulse * .68), 2 * k)

        # Rubber-hose legs, HD, behind the body.
        stride = math.sin(cycle) * 20 * walking
        lw = max(1.0, 5 * k)
        left_foot = (foot_x + (-16 + stride) * k, foot_y)
        right_foot = (foot_x + (16 - stride) * k, foot_y)
        hd_line(surface, INK, [px(-15, 32), (foot_x + (-17 - stride * .2) * k, foot_y - 10 * k), left_foot], lw)
        hd_line(surface, INK, [px(15, 32), (foot_x + (17 + stride * .2) * k, foot_y - 10 * k), right_foot], lw)
        hd_line(surface, INK, [left_foot, (left_foot[0] - 8 * k * self.facing, foot_y)], lw)
        hd_line(surface, INK, [right_foot, (right_foot[0] - 8 * k * self.facing, foot_y)], lw)

        self._draw_arms(surface, px, k, now, cycle, walking, mood)

        body_h = BODY * stretch
        body_w = BODY / stretch
        blit_art(surface, cam, self.art, wx - body_w / 2, wy + BODY / 2 - body_h, body_w, body_h)

        self._draw_face(surface, px, k, now, walking, mood)
        self._draw_mood_fx(surface, px, k, now)
        self._overlay = (px, k, show_name and not selected)

    def draw_overlay(self, surface: pygame.Surface, cam: Camera, now: float) -> None:
        """Balloons and name tags, drawn after the world so nothing occludes them.

        Uses the same screen-space mapping :meth:`draw` just built, so the tag
        rides the villager's hop instead of trailing a frame behind.
        """
        if self._overlay is None:
            return
        px, k, name_tag = self._overlay
        self._draw_speech(surface, px, k, cam, self.mood, now, name_tag)

    def _draw_arms(self, surface, px, k, now, cycle, walking, mood: Emotion) -> None:
        swing = math.sin(cycle) * 12 * walking
        pose = mood.arms if (self.gesture or mood.arms in ("droop", "fists")) else "swing"
        flap = math.sin(now * 12) * 6

        if pose == "wave":
            left, left_mid = px(-50, 13 - swing), px(-42, 18)
            right, right_mid = px(50, -12 + flap), px(44, -2)
        elif pose == "droop":
            left, left_mid = px(-40, 34), px(-40, 16)
            right, right_mid = px(40, 34), px(40, 16)
        elif pose == "fists":
            shake = math.sin(now * 22) * 2
            left, left_mid = px(-34, -6 + shake), px(-44, 14)
            right, right_mid = px(34, -6 - shake), px(44, 14)
        elif pose == "up":
            left, left_mid = px(-44, -26 + flap * .3), px(-46, -2)
            right, right_mid = px(44, -26 - flap * .3), px(46, -2)
        else:
            left, left_mid = px(-50, 13 - swing), px(-42, 18)
            right, right_mid = px(50, 13 + swing), px(42, 18)

        lw = max(1.0, 5 * k)
        hd_line(surface, INK, [px(-32, 3), left_mid, left], lw)
        hd_line(surface, INK, [px(32, 3), right_mid, right], lw)
        self._draw_hand(surface, left, k, -1, pose == "fists")
        self._draw_hand(surface, right, k, 1, pose == "fists")

    @staticmethod
    def _draw_hand(surface, hand, k, facing: int, fist: bool = False) -> None:
        hx, hy = hand
        hd_circle(surface, INK, hand, (7 if fist else 5) * k)
        if fist:
            return
        for dy in (-5, 0, 5):
            hd_line(surface, INK, [(hx, hy), (hx + facing * 7 * k, hy + dy * k)], max(1.0, 2 * k))

    def _draw_face(self, surface, px, k, now, walking, mood: Emotion) -> None:
        look = math.sin(now * .75 + self.phase) * 1.5 * self.facing
        style = "arc" if (self.blink and mood.eyes != "half") else mood.eyes

        for side in (-1, 1):
            eye = px(side * 12, -5)
            self._draw_eye(surface, eye, k, side, style, look, now)
            inner_dy, outer_dy = mood.brow
            hd_line(surface, INK,
                    [px(side * 12 + side * 6, -18 + inner_dy), px(side * 12 - side * 7, -18 + outer_dy)],
                    max(1.0, 3 * k))

        if mood.blush:
            for side in (-1, 1):
                cx, cy = px(side * 22, 7)
                hd_ellipse(surface, BLUSH, (cx - 8 * k, cy - 4 * k, 16 * k, 9 * k))

        self._draw_mouth(surface, px, k, mood.mouth, walking)

    @staticmethod
    def _draw_eye(surface, eye, k, side, style, look, now) -> None:
        ex, ey = eye
        if style == "arc":       # blink / happy closed-eye curve
            hd_arc(surface, INK, (ex - 7 * k, ey - 4 * k, 14 * k, 12 * k),
                   math.pi * .15, math.pi * .85, 3 * k)
            return
        if style == "half":      # sleepy lids
            hd_ellipse(surface, CREAM, (ex - 6 * k, ey - 4 * k, 12 * k, 11 * k))
            hd_ellipse(surface, INK, (ex - 4 * k, ey - k, 7 * k, 8 * k))
            hd_line(surface, INK, [(ex - 7 * k, ey - 3 * k), (ex + 7 * k, ey - k)], max(1.0, 3 * k))
            return
        if style == "heart":
            for dx in (-3, 3):
                hd_circle(surface, (233, 76, 96), (ex + dx * k, ey - 4 * k), 5 * k)
            pygame.draw.polygon(surface, (233, 76, 96), [
                (ex - 7 * k, ey - 2 * k), (ex + 7 * k, ey - 2 * k), (ex, ey + 8 * k)])
            hd_circle(surface, CREAM, (ex - 3 * k, ey - 5 * k), 2 * k)
            return

        wide = style == "wide"
        w, h = (16, 22) if wide else (12, 18)
        hd_ellipse(surface, CREAM, (ex - w / 2 * k, ey - h / 2 * k, w * k, h * k))
        pw, ph = (6, 9) if wide else (7, 12)
        hd_ellipse(surface, INK, (ex + (look - pw / 2) * k, ey - ph / 2 * k, pw * k, ph * k))
        hd_circle(surface, CREAM, (ex + (look - 1) * k, ey - 3 * k), 2 * k)

        if style == "teary":
            drop = (now * 34 + ex) % 46
            hd_circle(surface, TEAR, (ex + side * 8 * k, ey + (6 + drop) * k), max(1.0, (5 - drop / 12) * k))

    @staticmethod
    def _draw_mouth(surface, px, k, style, walking) -> None:
        mx, my = px(0, 16)
        if style == "grin":
            hd_ellipse(surface, INK, (mx - 9 * k, my - 3 * k, 18 * k, 17 * k))
            hd_arc(surface, (243, 105, 112), (mx - 5 * k, my + 6 * k, 10 * k, 6 * k), math.pi, math.tau, 3 * k)
        elif style == "frown":
            hd_arc(surface, INK, (mx - 8 * k, my + 2 * k, 16 * k, 12 * k), 0, math.pi, 3 * k)
        elif style == "wavy":
            hd_line(surface, INK,
                    [(mx + (-9 + i * 3) * k, my + (4 + math.sin(i * 1.6) * 2.5) * k) for i in range(7)],
                    max(1.0, 3 * k))
        elif style == "round":
            hd_circle(surface, INK, (mx, my + 5 * k), 6 * k)
        elif style == "grimace":
            rect = pygame.Rect(round(mx - 10 * k), round(my), max(2, round(20 * k)), max(2, round(11 * k)))
            pygame.draw.rect(surface, INK, rect, border_radius=max(1, round(3 * k)))
            pygame.draw.line(surface, CREAM, (rect.left + 2, rect.centery), (rect.right - 2, rect.centery),
                             max(1, round(2 * k)))
            for t in (.25, .5, .75):
                tx = rect.left + rect.width * t
                pygame.draw.line(surface, CREAM, (tx, rect.top + 1), (tx, rect.bottom - 1), max(1, round(2 * k)))
        elif style == "smirk":
            hd_arc(surface, INK, (mx - 2 * k, my - 2 * k, 13 * k, 11 * k), math.pi, math.tau, 3 * k)
        else:
            w = 18 if walking > .15 else 16
            hd_arc(surface, INK, (mx - w / 2 * k, my - 5 * k, w * k, 14 * k), math.pi, math.tau, 3 * k)

    def _draw_mood_fx(self, surface, px, k, now) -> None:
        """Little cartoon annotations floating around the head."""
        if self.emotion == "angry":
            for i in range(3):
                a = now * 2.5 + i * 2.1
                spot = px(math.cos(a) * 34, -46 + math.sin(a * 1.3) * 5)
                hd_circle(surface, (250, 233, 233), spot, 6 * k)
                hd_circle(surface, (232, 88, 88), spot, 6 * k, 2 * k)
        elif self.emotion == "love":
            for i in range(3):
                t = (now * .8 + i / 3) % 1.0
                hx, hy = px(30 + math.sin(t * 6 + i) * 8, -30 - t * 46)
                size = 7 * (1 - t * .5) * k
                for dx in (-size / 2, size / 2):
                    hd_circle(surface, (233, 76, 96), (hx + dx, hy), max(1.0, size / 2))
                pygame.draw.polygon(surface, (233, 76, 96), [
                    (hx - size, hy), (hx + size, hy), (hx, hy + size * 1.4)])
        elif self.emotion == "sleepy":
            for i in range(3):
                t = (now * .5 + i / 3) % 1.0
                hd_text(surface, "z", px(26 + t * 18, -44 - t * 40), 26 * k, INK)
        elif self.emotion == "worried":
            hd_circle(surface, TEAR, px(26, -26 + ((now * 2) % 1.0) * 16), 5 * k)
        elif self.emotion == "surprised":
            for i in range(6):
                a = i * math.tau / 6 + now * 1.5
                r0 = 44
                r1 = 53 + math.sin(now * 9) * 3
                hd_line(surface, (255, 236, 130),
                        [px(math.cos(a) * r0, -6 + math.sin(a) * r0),
                         px(math.cos(a) * r1, -6 + math.sin(a) * r1)], max(1.0, 3 * k))

    def _draw_speech(self, surface, px, k, cam: Camera, mood: Emotion, now: float,
                     name_tag: bool) -> None:
        on_screen = cam.safe.collidepoint(px(0, 0))
        if self.thinking and on_screen:
            centre = px(50, -64)
            hd_circle(surface, CREAM, centre, 22 * k)
            hd_circle(surface, INK, centre, 22 * k, 2 * k)
            for i in range(3):                  # three dots, chasing each other
                lift = math.sin(now * 6 - i * 0.9) * 3
                bright = 1 if math.sin(now * 6 - i * 0.9) > 0 else 0.45
                hd_circle(surface, tuple(round(c * bright + 150 * (1 - bright)) for c in INK),
                          (centre[0] + (i - 1) * 11 * k, centre[1] + lift * k), 3.5 * k)
            return
        if self.line_left > 0 and on_screen:
            self._speech_bubble(surface, px, k, cam, self.line)
            return                       # a name tag would sit under the balloon
        if self.gesture:
            centre = px(55, -61)
            hd_circle(surface, CREAM, centre, 22 * k)
            hd_circle(surface, INK, centre, 22 * k, 2 * k)
            hd_text(surface, mood.thought, centre, 29 * k, INK, center=True)
        if name_tag and cam.zoom > 0.7:
            hd_text(surface, self.name, px(0, -76), min(15 * cam.zoom + 9, 34), (250, 250, 245),
                    center=True, shadow=(40, 40, 46))

    def _speech_bubble(self, surface, px, k, cam: Camera, text: str) -> None:
        """HD speech balloon: the text is rasterised at its final size."""
        size = max(13, 22 * k)
        font = hd_font(round(size))
        words, lines, current = text.split(), [], ""
        for word in words:
            probe = f"{current} {word}".strip()
            if font.size(probe)[0] > 230 * k and current:
                lines.append(current)
                current = word
            else:
                current = probe
        lines.append(current)

        widths = [font.size(line)[0] for line in lines]
        pad = max(6, 8 * k)
        w = max(widths) + pad * 2
        h = len(lines) * font.get_linesize() + pad * 2
        anchor_x, anchor_y = px(0, -62)
        rect = pygame.Rect(0, 0, round(w), round(h))
        rect.midbottom = (round(anchor_x), round(anchor_y))
        rect.clamp_ip(cam.safe.inflate(-12, -12))

        pygame.draw.rect(surface, CREAM, rect, border_radius=max(3, round(8 * k)))
        pygame.draw.rect(surface, INK, rect, max(1, round(2 * k)), border_radius=max(3, round(8 * k)))
        tail = [(anchor_x - 7 * k, rect.bottom - 1), (anchor_x + 7 * k, rect.bottom - 1),
                (anchor_x, anchor_y + 10 * k)]
        pygame.draw.polygon(surface, CREAM, tail)
        pygame.draw.polygon(surface, INK, tail, max(1, round(2 * k)))
        for i, line in enumerate(lines):
            hd_text(surface, line, (rect.x + pad, rect.y + pad + i * font.get_linesize()),
                    round(size), INK)
