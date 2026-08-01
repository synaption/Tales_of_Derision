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
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pygame

import fruitbrains
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
              test_door_round_trip, test_zoom_is_bounded, test_mixed_pixel_scales, test_talking)
    for check in checks:
        check(game)
        print(f"  ok  {check.__name__}")
    contact_sheet(game)
    print(f"  ok  contact sheet -> {SHOT.name}")
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
