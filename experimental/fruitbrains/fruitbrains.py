"""Fruit Brains -- a minimal Animal Crossing style prototype.

Run from anywhere with::

    python3 experimental/fruitbrains/fruitbrains.py

A little town built from the free "Cozy Town" and "Interior" 16 px tilesets,
walked around by fruit villagers who have moods.  Pixel art renders at 2x by
default (each asset carries its own scale, so mixed scales are fine), while
faces, limbs, speech bubbles and all text are drawn at screen resolution, and
the whole thing zooms smoothly between 0.5x and 5x.

Controls
    WASD / arrows   walk           mouse wheel or -/=   zoom
    E or ENTER      talk           TAB                  take over another fruit
    1-8             set your mood  SPACE                hop
    click           take over the fruit you clicked     ESC  quit
"""

from __future__ import annotations

import math
import random

import pygame

from assets import Assets
from fruit import EMOTIONS, MOOD_KEYS, Fruit
from render import Camera, clamp, hd_font, hd_text
import world


WIDTH, HEIGHT = 1120, 700
FPS = 60
CREAM = (255, 247, 218)


class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Fruit Brains")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        random.seed(7)

        self.assets = Assets()
        self.scenes = world.build_scenes(self.assets)
        self.scene = self.scenes["town"]
        self.cam = Camera((WIDTH, HEIGHT))
        self.cam.zoom = self.cam.target_zoom = 1.4
        self.cam.safe = pygame.Rect(0, 58, WIDTH, HEIGHT - 58 - 40)   # clear of the HUD bars

        self.fruits = self._spawn()
        self.player_index = 0
        self.fade = 0.0
        self.door_cooldown = 0.0
        self.hint = ""
        self._shade: pygame.Surface | None = None
        self.mood_hotkeys = {getattr(pygame, f"K_{i + 1}"): name for i, name in enumerate(MOOD_KEYS)}

        self.cam.bounds = self.scene.bounds
        self.cam.snap_to(self.player.pos)

    # -- setup ---------------------------------------------------------------

    def _spawn(self) -> list[Fruit]:
        rng = random.Random(4)
        fruits: list[Fruit] = []
        names = list(self.assets.fruits)
        town, house = self.scenes["town"], self.scenes["house"]
        for i, name in enumerate(names):
            indoors = i >= len(names) - 2          # a couple of fruits stay home
            scene = house if indoors else town
            for _ in range(30):
                pos = pygame.Vector2(rng.uniform(96, scene.bounds.width - 96),
                                     rng.uniform(140, scene.bounds.height - 96))
                probe = pygame.Rect(0, 0, 28, 14)
                probe.center = (pos.x, pos.y + 38)
                if not any(probe.colliderect(s) for s in scene.solids):
                    break
            fruit = Fruit(name, self.assets.fruits[name], pos, i, scene.name)
            fruit.walk_target = pygame.Vector2(pos)
            fruits.append(fruit)
        return fruits

    @property
    def player(self) -> Fruit:
        return self.fruits[self.player_index]

    def here(self) -> list[Fruit]:
        return [f for f in self.fruits if f.scene == self.scene.name]

    # -- input ---------------------------------------------------------------

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEWHEEL:
            self.cam.zoom_by(1.12 ** event.y)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            spot = self.cam.to_world(*event.pos)
            local = self.here()
            if local:
                nearest = min(local, key=lambda f: f.pos.distance_to(spot))
                if nearest.pos.distance_to(spot) < 60:
                    self.player_index = self.fruits.index(nearest)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.cam.zoom_by(1.25)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.cam.zoom_by(1 / 1.25)
            elif event.key == pygame.K_TAB:
                local = self.here()
                order = -1 if event.mod & pygame.KMOD_SHIFT else 1
                if len(local) > 1:
                    nxt = local[(local.index(self.player) + order) % len(local)]
                    self.player_index = self.fruits.index(nxt)
            elif event.key in self.mood_hotkeys:
                name = self.mood_hotkeys[event.key]
                self.player.feel(name, 0.9 if name == "happy" else 6.0)
            elif event.key == pygame.K_SPACE:
                self.player.feel("excited", 2.0)
                self.player.velocity.y -= 60
            elif event.key in (pygame.K_e, pygame.K_RETURN):
                self.talk()
        return True

    def talk(self) -> None:
        neighbours = [f for f in self.here() if f is not self.player]
        if not neighbours:
            return
        other = min(neighbours, key=lambda f: f.pos.distance_to(self.player.pos))
        if other.pos.distance_to(self.player.pos) < 110:
            other.greet(self.player)
            self.player.facing = 1 if other.pos.x > self.player.pos.x else -1

    # -- simulation ----------------------------------------------------------

    def update(self, dt: float, now: float) -> None:
        keys = pygame.key.get_pressed()
        control = pygame.Vector2(
            int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - int(keys[pygame.K_LEFT] or keys[pygame.K_a]),
            int(keys[pygame.K_DOWN] or keys[pygame.K_s]) - int(keys[pygame.K_UP] or keys[pygame.K_w]))

        for fruit in self.fruits:
            scene = self.scenes[fruit.scene]
            fruit.update(dt, now, control if fruit is self.player else None, scene)
        for bird in self.scene.birds:
            bird.update(dt, self.player.feet)

        self.door_cooldown = max(0.0, self.door_cooldown - dt)
        self.fade = max(0.0, self.fade - dt * 3)
        self.hint = ""
        box = self.player.footprint()
        for door in self.scene.doors:
            if box.colliderect(door.rect):
                if self.door_cooldown <= 0:
                    self.enter(door)
                break
            if box.inflate(80, 80).colliderect(door.rect):
                self.hint = f"walk in to visit {door.label}"

        neighbours = [f for f in self.here() if f is not self.player]
        if neighbours and not self.hint:
            other = min(neighbours, key=lambda f: f.pos.distance_to(self.player.pos))
            if other.pos.distance_to(self.player.pos) < 110:
                self.hint = f"E -- talk to {other.name}"

        self.cam.update(dt, self.player.pos + pygame.Vector2(0, 10))

    def enter(self, door: world.Door) -> None:
        self.scene = self.scenes[door.target]
        self.player.scene = door.target
        self.player.pos.update(door.spawn)
        self.player.velocity.update(0, 0)
        self.player.walk_target.update(door.spawn)
        self.cam.bounds = self.scene.bounds
        self.cam.snap_to(self.player.pos)
        self.door_cooldown = 1.0
        self.fade = 1.0

    # -- drawing -------------------------------------------------------------

    def draw(self, now: float) -> None:
        screen, cam = self.screen, self.cam
        screen.fill(self.scene.sky)
        self.scene.draw_ground(screen, cam)

        for prop in self.scene.props:
            if prop.flat:
                prop.draw(screen, cam, now)

        actors = [p for p in self.scene.props if not p.flat]
        actors += self.here()
        actors += self.scene.birds
        near = 260 / max(0.35, cam.zoom) + 120        # only tag the neighbours you can reach
        for actor in sorted(actors, key=lambda a: a.depth):
            if isinstance(actor, Fruit):
                actor.draw(screen, cam, now, actor is self.player,
                           actor.pos.distance_to(self.player.pos) < near)
            else:
                actor.draw(screen, cam, now)

        if self.scene.indoors:
            self._vignette()
        if self.fade > 0:
            veil = pygame.Surface((WIDTH, HEIGHT))
            veil.set_alpha(round(255 * min(1.0, self.fade)))
            screen.blit(veil, (0, 0))
        self.hud(now)

    def _vignette(self) -> None:
        """Soft lamp-light falloff indoors, built once and reused."""
        if self._shade is None:
            small = pygame.Surface((64, 40), pygame.SRCALPHA)
            for y in range(40):
                for x in range(64):
                    d = math.hypot((x - 31.5) / 31.5, (y - 19.5) / 19.5)
                    small.set_at((x, y), (26, 18, 32, round(clamp((d - .45) * 210, 0, 150))))
            self._shade = pygame.transform.smoothscale(small, (WIDTH, HEIGHT))
        self.screen.blit(self._shade, (0, 0))

    def hud(self, now: float) -> None:
        screen = self.screen
        bar = pygame.Surface((WIDTH, 54), pygame.SRCALPHA)
        bar.fill((26, 32, 38, 214))
        screen.blit(bar, (0, 0))
        hd_text(screen, "FRUIT BRAINS", (20, 14), 38, (255, 228, 111), bold=True)
        hd_text(screen, self.scene.title, (232, 21), 24, (168, 214, 200))
        hero = self.player
        hd_text(screen, f"{hero.name} is feeling {hero.mood.label}", (380, 21), 24, CREAM)
        hd_text(screen, f"{self.cam.zoom:.2f}x", (WIDTH - 78, 21), 24, (168, 214, 200))

        moods = "  ".join(f"{i + 1} {EMOTIONS[n].label}" for i, n in enumerate(MOOD_KEYS))
        foot = pygame.Surface((WIDTH, 34), pygame.SRCALPHA)
        foot.fill((26, 32, 38, 190))
        screen.blit(foot, (0, HEIGHT - 34))
        hd_text(screen, "WASD move   E talk   TAB swap   wheel/-+ zoom   SPACE hop", (18, HEIGHT - 26), 21, CREAM)
        moods_w = hd_font(21).size(moods)[0]
        hd_text(screen, moods, (WIDTH - moods_w - 18, HEIGHT - 26), 21, (188, 200, 210))

        if self.hint:
            glow = 210 + math.sin(now * 6) * 45
            hd_text(screen, self.hint, (WIDTH // 2, HEIGHT - 78), 26,
                    (255, round(glow), 190), center=True, shadow=(30, 30, 36))

    # -- loop ----------------------------------------------------------------

    def run(self) -> None:
        running = True
        while running:
            dt = min(self.clock.tick(FPS) / 1000.0, .05)
            now = pygame.time.get_ticks() / 1000.0
            for event in pygame.event.get():
                running = self.handle(event) and running
            self.update(dt, now)
            self.draw(now)
            pygame.display.flip()
        pygame.quit()


def main() -> None:
    Game().run()


if __name__ == "__main__":
    main()
