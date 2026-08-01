"""Headless smoke test for Fruit Brains -- no window is ever opened.

    python3 experimental/fruitbrains/headlesstest.py

Drives the simulation and the renderer against a dummy SDL video driver,
asserts the things that are easy to break (mood decay, collision, door
round-trips, zoom limits, mixed pixel scales) and drops a contact sheet at
``fruitbrains_headless.png`` so the art can be eyeballed without launching the
game.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("FRUITBRAINS_LLM", "canned")   # never probe the network in tests
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pygame

import chat as chat_mod
import fruitbrains
from chat import CannedBackend, ChatService, tidy
from fruit import EMOTIONS, Fruit
from render import MAX_ZOOM, MIN_ZOOM


SHOT = Path(__file__).resolve().parent / "fruitbrains_headless.png"


def step(game: fruitbrains.Game, frames: int, t0: float = 0.0) -> float:
    for i in range(frames):
        game.update(1 / 60, t0 + i / 60)
        game.draw(t0 + i / 60)
    return t0 + frames / 60


def test_moods_decay_to_happy(game: fruitbrains.Game) -> None:
    fruit = game.player
    fruit.feel("angry", 0.5)
    assert fruit.emotion == "angry"
    step(game, 60)
    assert fruit.emotion == "happy", f"mood stuck on {fruit.emotion}"


def test_every_mood_draws(game: fruitbrains.Game) -> None:
    for name in EMOTIONS:
        game.player.feel(name, 5.0)
        step(game, 6, 30)
    game.player.feel("happy", 0.1)


def test_solids_block(game: fruitbrains.Game) -> None:
    solid = game.scenes["town"].solids[0]
    fruit = game.player
    fruit.pos.update(solid.centerx, solid.centery - 200)
    for _ in range(240):                      # push straight into the wall
        fruit.velocity.y += 30
        fruit.update(1 / 60, 0, None, game.scenes["town"])
    assert not fruit.footprint().colliderect(solid), "walked through a solid prop"


def test_door_round_trip(game: fruitbrains.Game) -> None:
    door = game.scenes["town"].doors[0]
    game.player.pos.update(door.rect.centerx, door.rect.bottom + 70)
    t = 0.0
    for _ in range(240):
        game.player.velocity.y -= 26
        t = step(game, 1, t)
        if game.scene.name == "house":
            break
    assert game.scene.name == "house", "never got through the front door"
    assert game.scenes["house"].bounds.collidepoint(game.player.pos), "spawned outside the room"

    for _ in range(300):
        game.player.velocity.y += 26
        t = step(game, 1, t)
        if game.scene.name == "town":
            break
    assert game.scene.name == "town", "could not get back outside"


def test_zoom_is_bounded(game: fruitbrains.Game) -> None:
    for _ in range(40):
        game.cam.zoom_by(2.0)
    assert game.cam.target_zoom <= MAX_ZOOM
    for _ in range(80):
        game.cam.zoom_by(0.5)
    assert game.cam.target_zoom >= MIN_ZOOM
    game.cam.target_zoom = game.cam.zoom = 1.4


def test_mixed_pixel_scales(game: fruitbrains.Game) -> None:
    """Every sprite carries its own scale; 16 px tiles land on a 32 unit grid."""
    tile = game.assets.town["tile_path"]
    assert (tile.w, tile.h) == (32, 32), "tileset is not rendering at 2x"
    fruit_art = next(iter(game.assets.fruits.values()))
    assert fruit_art.w == 64 and fruit_art.art_scale == 2
    assert game.assets.flowers[0].art_scale == 2
    # A hand-made 1x sprite has to survive the same pipeline.
    from render import PixelArt, blit_art
    odd = PixelArt(pygame.Surface((9, 5), pygame.SRCALPHA), 1)
    assert (odd.w, odd.h) == (9, 5)
    blit_art(game.screen, game.cam, odd, game.player.pos.x, game.player.pos.y)


def test_talking(game: fruitbrains.Game) -> None:
    others = [f for f in game.here() if f is not game.player]
    other = min(others, key=lambda f: f.pos.distance_to(game.player.pos))
    other.pos.update(game.player.pos + pygame.Vector2(40, 0))
    game.talk()
    assert other.line, "villager said nothing"
    step(game, 10)


class ScriptedBackend(chat_mod.Backend):
    """A stand-in model: deterministic, optionally slow, never touches a socket."""

    name = "scripted"
    online = True

    def __init__(self, delay: float = 0.0, text: str = "Oh! Lovely to see you, neighbour.") -> None:
        self.delay, self.text = delay, text
        self.seen: list[list[dict]] = []

    def reply(self, messages: list[dict], mood: str) -> str:
        self.seen.append(messages)
        if self.delay:
            time.sleep(self.delay)
        return self.text


class BrokenBackend(chat_mod.Backend):
    name = "broken"
    online = True

    def reply(self, messages: list[dict], mood: str) -> str:
        raise ConnectionRefusedError("nothing listening on :11434")


def test_prompt_is_in_character(game: fruitbrains.Game) -> None:
    backend = ScriptedBackend()
    service = ChatService(backend)
    service.ask("Lemon", "angry", "Apple", "how are you?")
    for _ in range(200):
        if service.poll():
            break
        time.sleep(0.005)
    system = backend.seen[0][0]["content"]
    assert "Lemon" in system and "sour" in system, system
    assert "angry" in system, "the villager's mood never reaches the model"
    assert service.conversation("Lemon").history, "history was not kept for the next turn"


def test_dialogue_round_trip(game: fruitbrains.Game) -> None:
    game.chat = ChatService(ScriptedBackend(text="Mind the puddles by the fountain!"))
    other = min((f for f in game.here() if f is not game.player),
                key=lambda f: f.pos.distance_to(game.player.pos))
    other.pos.update(game.player.pos + pygame.Vector2(50, 0))
    game.talk()
    assert game.talking is other, "conversation did not open"

    for ch in "hello there":
        game.handle(pygame.event.Event(pygame.TEXTINPUT, text=ch))
    game.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE, mod=0))
    assert game.typed == "hello ther"
    game.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    assert game.typed == "" and game.chat.busy(other.name)

    for _ in range(240):
        step(game, 1, 40)
        if not game.chat.busy(other.name):
            break
    assert other.line == "Mind the puddles by the fountain!", other.line
    assert game.transcript[-1][0] == other.name
    step(game, 4, 41)                                   # draw the reply balloon

    game.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
    assert game.talking is None, "ESC did not close the conversation"


def test_walking_is_not_blocked_by_the_model(game: fruitbrains.Game) -> None:
    """A slow Pi must cost frames rate, not gameplay."""
    game.chat = ChatService(ScriptedBackend(delay=0.6, text="Sorry, I was daydreaming."))
    other = min((f for f in game.here() if f is not game.player),
                key=lambda f: f.pos.distance_to(game.player.pos))
    other.pos.update(game.player.pos + pygame.Vector2(50, 0))
    game.talk()
    game.chat.ask(other.name, other.emotion, game.player.name, "are you awake?")
    start = time.perf_counter()
    step(game, 12, 50)
    assert time.perf_counter() - start < 0.4, "the frame loop waited on the model"
    assert other.thinking, "no thinking indicator while the model works"
    for _ in range(400):
        step(game, 1, 52)
        if not game.chat.busy(other.name):
            break
    assert other.line == "Sorry, I was daydreaming."
    game.end_chat()


def test_dead_server_falls_back(game: fruitbrains.Game) -> None:
    service = ChatService(BrokenBackend())
    service.ask("Pear", "happy", "Apple", "hello?")
    replies = []
    for _ in range(400):
        replies = service.poll()
        if replies:
            break
        time.sleep(0.005)
    villager, text, error = replies[0]
    assert villager == "Pear" and text, "no fallback line after a backend failure"
    assert error and "ConnectionRefused" in error, error


def test_reply_tidying(game: fruitbrains.Game) -> None:
    assert tidy('  "Hello there!"  ') == "Hello there!"
    assert tidy("Lemon: *shrugs* fine, I suppose.") == "fine, I suppose."
    long = tidy("word " * 200)
    assert len(long) <= chat_mod.REPLY_CHARS + 3
    assert CannedBackend().reply([{"role": "user", "content": "how are you?"}], "sad")


def contact_sheet(game: fruitbrains.Game) -> None:
    """One PNG: town at a distance, a close-up, and the cottage interior."""
    shots = []
    game.cam.target_zoom = game.cam.zoom = 0.8
    game.player.pos.update(544, 560)
    game.cam.snap_to(game.player.pos)
    step(game, 90, 5)
    shots.append(game.screen.copy())

    game.cam.target_zoom = game.cam.zoom = 2.6
    game.cam.snap_to(game.player.pos)
    game.player.feel("love", 6.0)
    step(game, 40, 12)
    shots.append(game.screen.copy())

    game.enter(game.scenes["town"].doors[0])
    game.fade = 0.0
    game.cam.target_zoom = game.cam.zoom = 1.5
    game.cam.snap_to(game.player.pos)
    step(game, 90, 20)
    shots.append(game.screen.copy())

    w, h = shots[0].get_size()
    sheet = pygame.Surface((w, h * len(shots)))
    for i, shot in enumerate(shots):
        sheet.blit(shot, (0, i * h))
    pygame.image.save(sheet, SHOT)


def main() -> int:
    game = fruitbrains.Game()
    checks = (test_moods_decay_to_happy, test_every_mood_draws, test_solids_block,
              test_door_round_trip, test_zoom_is_bounded, test_mixed_pixel_scales, test_talking,
              test_prompt_is_in_character, test_dialogue_round_trip,
              test_walking_is_not_blocked_by_the_model, test_dead_server_falls_back,
              test_reply_tidying)
    for check in checks:
        check(game)
        print(f"  ok  {check.__name__}")
    contact_sheet(game)
    print(f"  ok  contact sheet -> {SHOT.name}")
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
