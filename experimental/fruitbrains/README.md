Fruitbrains!
animal crossing, but fruits
runs on local LLMs with SillyTavern

## Running

    python3 experimental/fruitbrains/fruitbrains.py       # the prototype
    python3 experimental/fruitbrains/headlesstest.py      # asserts + contact sheet, no window

## Controls

| key | |
| --- | --- |
| WASD / arrows | walk |
| E or ENTER | talk to the villager next to you |
| TAB / shift-TAB | take over another fruit (click one, too) |
| 1-8 | set your own mood |
| SPACE | hop |
| mouse wheel, `-` / `=` | zoom (0.5x - 5x, smoothed) |
| ESC | quit |

Walk into the cottage door to go inside; walk onto the mat to come back out.

## Layout

| file | |
| --- | --- |
| `fruitbrains.py` | game loop, input, scene switching, HUD |
| `render.py` | camera, mixed-scale pixel-art blitting, HD text/vector helpers |
| `assets.py` | tileset slicing (measured rects) + generated grass and flowers |
| `world.py` | the town and cottage scenes, props, collision, doors |
| `fruit.py` | villagers: emotions, wandering, HD faces and rubber-hose limbs |

## Mixed pixel scales

World units are the shared space: one tile is 32 world units, and the camera's
`zoom` is the only thing that turns world units into screen pixels.  Each
sprite carries its own `art_scale`, so the 16 px "Cozy Town"/"Interior" tiles
and the 32 px fruit art both render at 2x today, but an asset drawn at 1x or 3x
drops in without touching the camera.

Faces, limbs, speech balloons, name tags and all UI text are *not* pixel art --
they are drawn straight into screen space at the size they end up on screen, so
zooming in gives you a finer face and sharper text instead of bigger pixels.

Art: "Cozy Town" and "Interior" free packs in `reference/`, fruit sprites in
`Fruits_Separated/`.
