Fruitbrains!
animal crossing, but fruits
runs on local LLMs with Letta
small enough to run on a raspberry pi

## Running

    python3 experimental/fruitbrains/fruitbrains.py       # the prototype
    python3 experimental/fruitbrains/headlesstest.py      # asserts + contact sheet, no window

## Controls

| key | |
| --- | --- |
| WASD / arrows | walk |
| E | talk to the villager next to you |
| type, ENTER | say something back; ESC leaves the conversation |
| TAB / shift-TAB | take over another fruit (click one, too) |
| 1-8 | set your own mood |
| SPACE | hop |
| mouse wheel, `-` / `=` | zoom (0.5x - 5x, smoothed) |
| ESC | quit |

Walk into the cottage door to go inside; walk onto the mat to come back out.

## Talking to the fruit

Press `E` next to a villager, type, hit ENTER.  Their reply comes from a small
local model; with no server running they answer from hand-written lines instead,
so the game never depends on the model being up.  The backend in use is printed
in the top bar.

    ollama serve                        # then, once:
    ollama pull qwen2.5:0.5b            # ~400 MB, happy on a Pi 4

Auto-detected in this order: `$FRUITBRAINS_LLM_URL`, Ollama on `:11434`, an
OpenAI-compatible server on `:8080` / `:5001` / `:5000` (llama.cpp's
`llama-server`, KoboldCpp, oobabooga, LM Studio), then offline lines.

    FRUITBRAINS_LLM=ollama|openai|canned    pick a backend explicitly
    FRUITBRAINS_LLM_URL=http://pi.local:11434
    FRUITBRAINS_MODEL=smollm2:360m

Pi sizing: a 0.5 B / 360 M instruct model at Q4_K_M on a Pi 4, a 1 B on a Pi 5.
Replies are capped at `MAX_TOKENS = 48` and generated on a worker thread, so a
slow reply costs seconds of waiting, never frames.  Adding another backend
(Letta, a remote box, anything with a persona-shaped API) means one class with a
`reply()` method in `chat.py` -- see `CannedBackend` for the shape.

## Layout

| file | |
| --- | --- |
| `fruitbrains.py` | game loop, input, scene switching, HUD, chat panel |
| `render.py` | camera, mixed-scale pixel-art blitting, HD text/vector helpers |
| `assets.py` | tileset slicing (measured rects) + generated grass and flowers |
| `world.py` | the town and cottage scenes, props, collision, doors |
| `fruit.py` | villagers: emotions, wandering, HD faces and rubber-hose limbs |
| `chat.py` | local-LLM backends, personas, threaded conversation service |

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
