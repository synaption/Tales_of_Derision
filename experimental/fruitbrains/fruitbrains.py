"""Fruit Brains -- a minimal Animal Crossing style prototype.

Run from anywhere with::

    python3 experimental/fruitbrains/fruitbrains.py

A little town built from the free "Cozy Town" and "Interior" 16 px tilesets,
walked around by fruit villagers who have moods.  Pixel art renders at 2x by
default (each asset carries its own scale, so mixed scales are fine), while
faces, limbs, speech bubbles and all text are drawn at screen resolution, and
the whole thing zooms smoothly between 0.5x and 5x.

Villagers answer back: press E to open a conversation, type, and a small local
LLM (Ollama / llama.cpp / KoboldCpp -- see chat.py) writes their reply in
character.  If nothing is serving, the game starts its own `ollama serve` and
stops it again on exit; a server that was already running is used as-is and
left alone.

Options
    --letta       persistent villagers: each one is a Letta agent that
                  remembers you between sessions (see letta_backend.py)
    --forget      delete the Letta agents and let the town meet you fresh
    --canned      scripted lines instead of a model (deterministic, for beats)
    --no-serve    never start a server; fail if one isn't already up
    --no-pull     never download a model
    --help

Controls
    WASD / arrows   walk           mouse wheel or -/=   zoom
    E               talk           TAB                  take over another fruit
    type + ENTER    say something  ESC                  leave the conversation
    1-8             set your mood  SPACE                hop
    click           take over the fruit you clicked     ESC  quit
"""

from __future__ import annotations

import math
import os
import random
import sys

import pygame

from assets import Assets
from chat import ChatService, NoLocalModel
from fruit import EMOTIONS, MOOD_KEYS, Fruit
from render import Camera, clamp, hd_font, hd_text
import world


WIDTH, HEIGHT = 1120, 700
FPS = 60
CREAM = (255, 247, 218)


TALK_RANGE = 130          # world units you can chat across


def mood_from_text(text: str, current: str) -> str:
    """Read the villager's own reply back and let it colour their face."""
    low = text.lower()
    for mood, cues in (("excited", ("!", "yay", "wow", "let's", "party")),
                       ("love", ("love", "adore", "sweet", "heart", "darling")),
                       ("sad", ("sorry", "sad", "miss", "alone", "sigh")),
                       ("angry", ("no!", "rude", "hate", "grr", "jam")),
                       ("surprised", ("?!", "oh!", "really", "goodness")),
                       ("sleepy", ("nap", "sleep", "yawn", "tired", "zzz")),
                       ("worried", ("worry", "rain", "afraid", "hope not", "bruise"))):
        if any(cue in low for cue in cues):
            return mood
    return current if current != "happy" else "happy"


class Game:
    def __init__(self, chat_service: ChatService | None = None) -> None:
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

        # Dialogue: a local model if one is listening, hand-written lines if not.
        self.chat = chat_service if chat_service is not None else ChatService()
        self.talking: Fruit | None = None
        self.typed = ""
        self.transcript: list[tuple[str, str]] = []
        self.chat_error = ""
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
        if self.talking is not None and self._handle_chat(event):
            return True
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

    # -- conversation --------------------------------------------------------

    def nearest(self) -> Fruit | None:
        neighbours = [f for f in self.here() if f is not self.player]
        if not neighbours:
            return None
        other = min(neighbours, key=lambda f: f.pos.distance_to(self.player.pos))
        return other if other.pos.distance_to(self.player.pos) < TALK_RANGE else None

    def talk(self) -> None:
        """Open a conversation with the villager you're standing next to."""
        other = self.nearest()
        if other is None:
            return
        other.greet(self.player)
        self.chat.recall(other.name)      # what do they already know about me?
        self.player.facing = 1 if other.pos.x > self.player.pos.x else -1
        self.talking = other
        self.typed = ""
        self.transcript = [(other.name, other.line)]
        pygame.key.start_text_input()

    def end_chat(self) -> None:
        pygame.key.stop_text_input()
        if self.talking is not None:
            self.chat.cancel(self.talking.name)     # don't leave a reply in flight
            self.talking.chatting = False
            self.talking.thinking = False
        self.player.chatting = False
        self.talking = None
        self.typed = ""

    def _handle_chat(self, event: pygame.event.Event) -> bool:
        """Keyboard belongs to the chat box while a conversation is open."""
        other = self.talking
        if event.type == pygame.TEXTINPUT:
            if len(self.typed) < 120:
                self.typed += event.text
            return True
        if event.type != pygame.KEYDOWN:
            return False
        if event.key == pygame.K_ESCAPE:
            self.end_chat()
        elif event.key == pygame.K_BACKSPACE:
            self.typed = self.typed[:-1]
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            said = self.typed.strip()
            if said and not self.chat.busy(other.name):
                self.transcript.append((self.player.name, said))
                self.chat.ask(other.name, other.emotion, self.player.name, said)
                self.typed = ""
        else:
            return event.key not in (pygame.K_F1,)      # swallow everything else
        return True

    def collect_replies(self) -> None:
        """Pick up whatever the model finished while we were drawing frames."""
        for villager, text, error in self.chat.poll():
            speaker = next((f for f in self.fruits if f.name == villager), None)
            if speaker is None:
                continue
            if error or not text:
                # Say nothing rather than fake it -- a scripted line here would
                # read as a villager ignoring the question.
                self.chat_error = error or "empty reply from the model"
                continue
            self.chat_error = ""
            speaker.say(text, 7.0)
            speaker.feel(mood_from_text(text, speaker.emotion), 5.0)
            self.transcript.append((villager, text))
            del self.transcript[:-6]

    # -- simulation ----------------------------------------------------------

    def update(self, dt: float, now: float) -> None:
        self.collect_replies()
        keys = pygame.key.get_pressed()
        control = pygame.Vector2(0, 0) if self.talking is not None else pygame.Vector2(
            int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - int(keys[pygame.K_LEFT] or keys[pygame.K_a]),
            int(keys[pygame.K_DOWN] or keys[pygame.K_s]) - int(keys[pygame.K_UP] or keys[pygame.K_w]))

        if self.talking is not None:
            if self.talking.scene != self.scene.name:
                self.end_chat()
            else:
                # Both of you hold position and face each other until ESC.
                self.talking.chatting = self.player.chatting = True
                self.talking.facing = 1 if self.player.pos.x > self.talking.pos.x else -1
                self.player.facing = -self.talking.facing

        for fruit in self.fruits:
            scene = self.scenes[fruit.scene]
            fruit.thinking = self.chat.busy(fruit.name)
            if fruit.thinking:
                # Show the words as the model produces them: a half-finished
                # sentence is proof it's working, where dots never are.
                sofar = self.chat.partial(fruit.name)
                if sofar:
                    fruit.thinking = False
                    fruit.say(sofar + " ...", 1.0)
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

        if self.talking is None and not self.hint:
            other = self.nearest()
            if other is not None:
                self.hint = f"E -- talk to {other.name}"

        focus = self.player.pos + pygame.Vector2(0, 10)
        if self.talking is not None:            # frame both of you mid-conversation
            focus = (focus + self.talking.pos) / 2
        self.cam.update(dt, focus)

    def enter(self, door: world.Door) -> None:
        self.end_chat()
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
        for fruit in self.here():
            fruit.draw_overlay(screen, cam, now)      # balloons ride above the world

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
        tint = (150, 208, 160) if self.chat.online else (150, 150, 158)
        hd_text(screen, self.chat.name, (WIDTH - 320, 22), 20, tint)

        moods = "  ".join(f"{i + 1} {EMOTIONS[n].label}" for i, n in enumerate(MOOD_KEYS))
        foot = pygame.Surface((WIDTH, 34), pygame.SRCALPHA)
        foot.fill((26, 32, 38, 190))
        screen.blit(foot, (0, HEIGHT - 34))
        hd_text(screen, "WASD move   E talk   TAB swap   wheel/-+ zoom   SPACE hop", (18, HEIGHT - 26), 21, CREAM)
        moods_w = hd_font(21).size(moods)[0]
        hd_text(screen, moods, (WIDTH - moods_w - 18, HEIGHT - 26), 21, (188, 200, 210))

        if self.talking is not None:
            self.chat_panel(now)
        elif self.hint:
            glow = 210 + math.sin(now * 6) * 45
            hd_text(screen, self.hint, (WIDTH // 2, HEIGHT - 78), 26,
                    (255, round(glow), 190), center=True, shadow=(30, 30, 36))

    def fit_text(self, text: str, pos: tuple[int, int], size: int,
                 colour: tuple[int, int, int], width: int) -> None:
        """HD text trimmed with an ellipsis until it fits `width` pixels."""
        font = hd_font(size)
        while font.size(text)[0] > width and len(text) > 4:
            text = text[:-2] + "…"
        hd_text(self.screen, text, pos, size, colour)

    def chat_panel(self, now: float) -> None:
        """The typed half of the conversation, drawn HD at the foot of the screen."""
        screen = self.screen
        other = self.talking
        panel = pygame.Rect(56, HEIGHT - 214, WIDTH - 112, 168)
        board = pygame.Surface(panel.size, pygame.SRCALPHA)
        board.fill((26, 32, 38, 232))
        screen.blit(board, panel)
        pygame.draw.rect(screen, (120, 168, 150), panel, 2, border_radius=6)

        hd_text(screen, f"talking with {other.name}", (panel.x + 16, panel.y + 12), 23, (255, 228, 111))
        hd_text(screen, "ENTER send    ESC leave", (panel.right - 210, panel.y + 14), 20, (140, 156, 168))
        if self.chat.scripted:
            hd_text(screen, "scripted lines - not reading your input", (panel.x + 232, panel.y + 14),
                    20, (240, 186, 110))
        elif self.chat.stateful:
            # What the villager has actually written down about you -- the whole
            # reason for the agent layer, so it belongs on screen, not in a log.
            note = " ".join(self.chat.memory(other.name).split()) or "nothing about you yet"
            self.fit_text(f"remembers: {note}", (panel.x + 232, panel.y + 15), 19,
                          (128, 150, 170), panel.right - 232 - 226)

        y = panel.y + 44
        for speaker, line in self.transcript[-3:]:
            mine = speaker == self.player.name
            colour = (168, 214, 240) if mine else CREAM
            self.fit_text(f"{speaker}: {line}", (panel.x + 16, y), 21, colour, panel.width - 32)
            y += 26

        prompt = pygame.Rect(panel.x + 12, panel.bottom - 44, panel.width - 24, 32)
        pygame.draw.rect(screen, (16, 20, 24), prompt, border_radius=4)
        if self.chat.busy(other.name):
            waited = self.chat.waited(other.name)
            sofar = self.chat.partial(other.name)
            if sofar:
                status = f"{other.name}: {sofar}"
            else:
                dots = "." * (1 + int(now * 3) % 3)
                loading = "  (loading the model, first reply is the slow one)" if waited > 8 else ""
                status = f"{other.name} is thinking{dots}  {waited:.0f}s{loading}"
            self.fit_text(status, (prompt.x + 10, prompt.y + 7), 21, (150, 208, 160),
                          prompt.width - 100)
            hd_text(screen, "ESC cancels", (prompt.right - 96, prompt.y + 7), 20, (140, 156, 168))
        else:
            caret = "_" if int(now * 2) % 2 else " "
            hd_text(screen, f"> {self.typed}{caret}", (prompt.x + 10, prompt.y + 7), 21, CREAM)
        if self.chat_error:
            hd_text(screen, f"model error: {self.chat_error[:76]}", (panel.x + 16, panel.bottom - 66),
                    19, (238, 132, 132))

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


def main(argv: list[str] | None = None) -> int:
    """Check the model is up *before* opening a window, and say so plainly if not."""
    args = sys.argv[1:] if argv is None else argv
    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0
    if "--canned" in args:
        os.environ["FRUITBRAINS_LLM"] = "canned"
    if "--letta" in args or "--forget" in args:
        os.environ["FRUITBRAINS_LLM"] = "letta"

    try:
        service = ChatService(autostart="--no-serve" not in args, pull="--no-pull" not in args)
    except NoLocalModel as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    if "--forget" in args:
        # Deliberately destructive, so it is its own run: wipe, say what went, exit.
        try:
            gone = service.backend.forget()
            print(f"Fruit Brains: forgot {len(gone)} villager(s): {', '.join(gone) or 'none'}")
        finally:
            service.shutdown()
        return 0

    if service.scripted:
        print("Fruit Brains: scripted-lines mode -- villagers will not answer what you type.")
    else:
        print(f"Fruit Brains: talking to {service.name}"
              + (" -- villagers remember you between sessions" if service.stateful else ""))
    try:
        Game(service).run()
    except KeyboardInterrupt:
        pass
    finally:
        service.shutdown()          # only stops a server this session started
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
