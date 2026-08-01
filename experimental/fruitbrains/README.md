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
local model.

**The game runs its own server.**  If nothing is listening it starts
`ollama serve` itself, pulls the model the first time (a few hundred MB, once),
waits for it to answer, and stops it again when you quit -- including on Ctrl-C
or a crash.  A server that was *already* running is used as-is and never shut
down: the game only stops what it started.

The model is still required.  If it can't get one up -- `ollama` not installed,
the server dying, a pull failing -- it prints why and exits (code 2) rather than
opening a window.  If a request fails mid-game the villager says nothing and the
error shows in the chat panel; a scripted line is never passed off as a reply,
because that just looks like a model ignoring you.

    python3 fruitbrains.py              # that's it -- server and model handled
    python3 fruitbrains.py --no-serve   # use a server I started myself
    python3 fruitbrains.py --no-pull    # never download anything

### Model sizing

The model is picked from what the machine can hold, and the choice is printed at
startup (`NVIDIA GeForce RTX 4080 (16 GB VRAM) -> model qwen2.5:7b`).

| hardware | model |
| --- | --- |
| 22 GB+ VRAM | `qwen2.5:14b` |
| 10 GB+ VRAM | `qwen2.5:7b` |
| 6 GB+ VRAM | `qwen2.5:3b` |
| 3.5 GB+ VRAM | `qwen2.5:1.5b` |
| CPU, 16 GB+ RAM | `qwen2.5:3b` |
| CPU, 8 GB+ RAM | `qwen2.5:1.5b` |
| Pi 4 / 3 GB+ RAM | `qwen2.5:0.5b` |
| anything smaller | `smollm2:360m` |

NVIDIA, AMD and Apple Silicon are detected (`hardware.py`); anything it can't
read falls back to the CPU ladder.  `FRUITBRAINS_MODEL` overrides it entirely.
Weights stay resident for 30 minutes between chats, so only the first reply
after a cold start pays the load cost.

Auto-detected before starting anything: `$FRUITBRAINS_LLM_URL`, Ollama on
`:11434`, an OpenAI-compatible server on `:8080` / `:5001` / `:5000`
(llama.cpp's `llama-server`, KoboldCpp, oobabooga, LM Studio).  Server output
goes to `$TMPDIR/fruitbrains-ollama.log`.

    FRUITBRAINS_LLM=ollama|openai|canned    pick a backend explicitly
    FRUITBRAINS_LLM_URL=http://pi.local:11434
    FRUITBRAINS_MODEL=smollm2:360m
    FRUITBRAINS_DEBUG=1                     print every prompt and reply

Pi sizing: a 0.5 B / 360 M instruct model at Q4_K_M on a Pi 4, a 1 B on a Pi 5.
Replies are capped at `MAX_TOKENS` and generated on a worker thread, so a slow
reply costs seconds of waiting, never frames.  Adding another backend (Letta, a
remote box, anything with a persona-shaped API) means one class with `reply()`
and `check()` in `chat.py`.

### Scripted lines

`--canned` (or `FRUITBRAINS_LLM=canned`) runs dialogue off hand-written lines --
for barks, tutorial beats, shop patter, determinism in tests.  It is a mode you
choose, never a fallback: the chat panel labels it *"scripted lines - not
reading your input"* so it can't be mistaken for a broken model.  The greeting
a villager gives when you walk up is always a scripted bark; only the replies
to what you type come from the model.

## Layout

| file | |
| --- | --- |
| `fruitbrains.py` | game loop, input, scene switching, HUD, chat panel |
| `render.py` | camera, mixed-scale pixel-art blitting, HD text/vector helpers |
| `assets.py` | tileset slicing (measured rects) + generated grass and flowers |
| `world.py` | the town and cottage scenes, props, collision, doors |
| `fruit.py` | villagers: emotions, wandering, HD faces and rubber-hose limbs |
| `chat.py` | local-LLM backends, personas, threaded conversation service |
| `hardware.py` | GPU/RAM detection and the model ladder |

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
