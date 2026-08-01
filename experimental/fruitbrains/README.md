Fruitbrains!
animal crossing, but fruits
runs on local LLMs with Letta
small enough to run on a raspberry pi

## Running

    python3 experimental/fruitbrains/fruitbrains.py       

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
reply costs seconds of waiting, never frames.  Adding another backend means one
class with `reply()` and `check()` -- or `converse()` and `stateful = True` if
it keeps its own memory, the way `letta_backend.py` does.

### Persistent personalities (Letta)

    python3 fruitbrains.py --letta

Without this, a villager's memory is the last six messages and it dies with the
process.  With it, **each villager is a Letta agent** -- `fruitbrains-Peach`,
tagged `fruitbrains`, living in Letta's database -- with a `persona` block for
who they are and a `human` block for what they've worked out about you.  Tell
Peach you keep bees and she still knows next week.

What persists is the conversation itself, held by the agent -- quit, restart,
come back tomorrow, and Peach still knows about the bees.  The chat panel shows
her `human` block next to her name, so what she has written down is on screen
rather than in a log.

**Verified against Letta 0.16.8 with `qwen2.5:7b`**, and one caveat came out of
it: the agents are given Letta's base tools, including core-memory writes, but
whether they *use* them depends on the model.  Ollama's OpenAI-compatible
endpoint returns no native tool calls for qwen2.5 (`tool_calls: null`, even with
tools in the request), so a villager talks well and remembers the conversation
but rarely edits its own memory block -- and when it tries, it writes
`memory_insert(...)` as dialogue.  That never reaches the screen: `spoken()`
strips tool-call syntax and fails the turn rather than have a villager read JSON
at you.  A model with working tool calls gets the self-editing memory too; the
lever is `FRUITBRAINS_MODEL`.

Letta is the agent layer, not the model: it runs inference through the same
local Ollama and the same [model ladder](#model-sizing), plus a small embedding
model (`nomic-embed-text`, ~270 MB) for archival memory.  With nothing running,
`--letta` starts Ollama, pulls both models, starts `letta server` pointed at
Ollama, and stops both on exit.  Anything already listening is used as-is and
never shut down.

Letta's server keeps its state in Postgres, so the docker image (which brings
one) is the path that works with nothing else installed:

    docker run -d -p 8283:8283 --add-host=host.docker.internal:host-gateway \
      -e OLLAMA_BASE_URL=http://host.docker.internal:11434 letta/letta:latest
    python3 fruitbrains.py --letta          # finds it on :8283 and leaves it alone

Ollama binds `127.0.0.1` by default, which the container cannot reach, and the
symptom is Letta listing no models at all -- so it needs
`OLLAMA_HOST=0.0.0.0 ollama serve` (or any address the container can get to).
Startup says so if the handle it wants isn't on offer.

    pip install letta                       # the game starts this one itself, but
    LETTA_PG_URI=postgresql://...           # only with a Postgres to point at
    python3 fruitbrains.py --forget         # delete the agents, meet the town fresh

    FRUITBRAINS_LETTA_URL=http://pi.local:8283   # or an existing/docker Letta
    FRUITBRAINS_EMBEDDING=nomic-embed-text
    LETTA_API_KEY=...                            # sent as a bearer token

A Letta already serving on `:8283` is picked up automatically, `--letta` or
not.  Memory costs latency -- the agent re-reads its blocks each turn -- so a Pi
is happier on the plain Ollama backend, and both are the same game.

The persona block is written once, at creation, and never overwritten: a game
that rewrote it every launch would be a game where nothing persists.  Edit
`PERSONAS` in `chat.py` and existing villagers keep the personality they grew;
`--forget` is how you start them over.

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
| `letta_backend.py` | one Letta agent per villager: memory that outlives the process |
| `serve.py` | HTTP with deadlines, and child servers we start and stop |
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
