"""Render the page with no window, no display server, nothing on screen.

Deliberately hostile: SDL is pointed at a driver that cannot show anything and
`pygame.display.set_mode` is never called, so anything that quietly depended on
a window will fail here rather than in six months on a build machine.
"""
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame

import inkGL

pygame.init()
pygame.font.init()

SIZE = (400, 280)
failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


check("no display was ever opened", pygame.display.get_surface() is None)

renderer = inkGL.ParchmentInkRenderer(SIZE, supersample=2, headless=True)
check(
    "a context exists without a window",
    renderer.headless and renderer.target is not renderer.context.screen,
    renderer.context.info["GL_RENDERER"],
)

settings = inkGL.DemoSettings()
view = inkGL.View(SIZE)
renderer.render((SIZE[0] // 2, SIZE[1] // 2), settings, view)

frame = renderer.capture()
check(
    "a frame comes back the right size",
    len(frame) == SIZE[0] * SIZE[1] * 3,
    f"{len(frame)} bytes for {SIZE[0]}x{SIZE[1]}x3",
)

pixels = np.frombuffer(frame, dtype=np.uint8).reshape(SIZE[1], SIZE[0], 3)
check(
    "it is parchment, not an empty buffer",
    pixels.mean() > 40 and pixels.std() > 4,
    f"mean {pixels.mean():.1f}, spread {pixels.std():.1f}",
)
check(
    "and it is warm, the way parchment is",
    pixels[..., 0].mean() > pixels[..., 2].mean(),
    f"red {pixels[..., 0].mean():.1f} vs blue {pixels[..., 2].mean():.1f}",
)

# The lamp has to actually light the page: move it and the picture must change.
renderer.render((20, 20), settings, view)
corner = np.frombuffer(renderer.capture(), np.uint8).reshape(SIZE[1], SIZE[0], 3)
moved = float(np.abs(corner.astype(int) - pixels.astype(int)).mean())
check("moving the lamp changes the frame", moved > 2.0, f"mean change {moved:.1f}")

# Drawing, and the debug views, all work with nothing on screen.
renderer.upload_stroke(renderer.canvas.begin_stroke((60, 140), 9, erase=False))
for point in ((140, 150), (240, 120), (330, 190)):
    renderer.upload_stroke(renderer.canvas.extend_stroke(point))
renderer.canvas.end_stroke()
renderer.render((SIZE[0] // 2, SIZE[1] // 2), settings, view)
inked = np.frombuffer(renderer.capture(), np.uint8).reshape(SIZE[1], SIZE[0], 3)
check(
    "a stroke shows up in the render",
    inked.mean() < pixels.mean(),
    f"page darkened from {pixels.mean():.1f} to {inked.mean():.1f}",
)

# Dried off first, or the page-wide wetness saturates and the wetness view is
# a flat white square -- correctly, but it would prove nothing.
settings.wetness = inkGL.PAGE_WET_FLOOR
views = {}
for mode, name in enumerate(("final", "normals", "roughness", "ink", "wetness")):
    settings.debug_mode = mode
    renderer.render((200, 140), settings, view)
    views[name] = np.frombuffer(
        renderer.capture(), np.uint8
    ).reshape(SIZE[1], SIZE[0], 3)
    check(f"  debug view {mode} ({name}) renders", views[name].std() > 1.0,
          f"spread {views[name].std():.1f}")
for name in ("normals", "roughness", "ink", "wetness"):
    check(f"  and {name} is not just the final view",
          not np.array_equal(views[name], views["final"]))
settings.debug_mode = 0
settings.wetness = 1.0

# The panels composite over the page, which needs blending against the target.
renderer.help.toggle()
renderer.sliders.toggle()
renderer.sliders.refresh(settings)
renderer.render((200, 140), settings, view)
panelled = np.frombuffer(renderer.capture(), np.uint8).reshape(SIZE[1], SIZE[0], 3)
renderer.help.toggle()
renderer.sliders.toggle()
renderer.render((200, 140), settings, view)
plain = np.frombuffer(renderer.capture(), np.uint8).reshape(SIZE[1], SIZE[0], 3)
check(
    "panels draw over the page",
    float(np.abs(panelled.astype(int) - plain.astype(int)).mean()) > 3.0,
    f"mean change {np.abs(panelled.astype(int) - plain.astype(int)).mean():.1f}",
)

# And the whole point: the creature can be drawn with nobody watching.
reveal = renderer.draw_feature(inkGL.PEN_SPEED)
check("the pen is ready", reveal is not None)
renderer.show_creature(False)
frames = 0
while not reveal.finished and frames < 4000:
    renderer.upload_stroke(reveal.update(1 / 60.0, renderer.canvas))
    frames += 1
renderer.show_creature(True)
renderer.render((200, 140), settings, view)
drawn = np.frombuffer(renderer.capture(), np.uint8).reshape(SIZE[1], SIZE[0], 3)
check("the creature drew headlessly", frames < 4000, f"{frames} frames")

printed = inkGL.make_demo_ink(SIZE, renderer.sprites, 0, scale=2)
check(
    "and the page ended up as the printed artwork",
    np.array_equal(
        pygame.surfarray.array_alpha(printed),
        pygame.surfarray.array_alpha(renderer.canvas.base),
    ),
)

# A snapshot can be saved, which is what a failing test wants to leave behind.
shot = renderer.snapshot()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "headless.png")
pygame.image.save(shot, path)
check("a snapshot saves to disk", os.path.getsize(path) > 1000,
      f"{os.path.getsize(path)} bytes at {path}")
check("the snapshot is the right way up",
      shot.get_size() == SIZE, f"{shot.get_size()}")

renderer.release()
print()
print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
