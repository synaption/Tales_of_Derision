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
come back tomorrow, and Peach still knows about the bees.  The chat panel says
so next to her name: the facts she has written down if her model writes any,
otherwise how far back the two of you go (`remembers: 9 things you have said
before`).

**Verified against Letta 0.16.8 with `qwen2.5:7b`**, and one caveat came out of
it: agents get `memory_insert` and `memory_replace`, but whether they *use* them
is a property of the model, not of the plumbing.  qwen2.5:7b calls tools
correctly when asked to (native `tool_calls` on both Ollama endpoints) and still
never writes to its memory blocks -- across six turns it made zero tool calls,
and told point-blank to "write this down" it replies "I'll remember that!" and
records nothing.  Deciding to reach for a tool unprompted is an agentic skill
small models lack; see the [model note](#which-model-for-letta).

When such a model *is* nudged into writing memory, it tends to emit
`memory_insert(...)` as dialogue instead of a real call.  That never reaches the
screen: `spoken()` strips tool-call syntax and fails the turn rather than have a
villager read JSON at you.

### Which model for Letta

Conversation memory works with any model -- it is server-side and needs no tool
calls.  Self-editing memory blocks need a model that *chooses* to call tools:

| model | writes memory? | as a neighbour |
| --- | --- | --- |
| `llama3.1:8b` | **yes** -- verified `memory_insert`/`memory_replace` every turn | flat; recites facts back at you |
| `qwen2.5:7b` | no -- zero tool calls in six turns | warm, curious, asks after your bees |
| `qwen3:14b` | untested; the obvious next try on a 16 GB card | -- |
| Pi-sized (≤3 B) | no | conversation memory only -- use the plain Ollama backend |

It is a straight trade, and **the town runs on qwen2.5** -- the whole
[ladder](#model-sizing) is qwen2.5, and being good company matters more here
than filing notes.  Every villager still remembers every conversation; that
lives on the server and needs no tools at all.  Swap in llama3.1 for a session
if you want to watch the blocks fill up:

    FRUITBRAINS_MODEL=llama3.1:8b python3 fruitbrains.py --letta

Letta is the agent layer, not the model: it runs inference through the same
local Ollama and the same [model ladder](#model-sizing), plus a small embedding
model (`nomic-embed-text`, ~270 MB) for archival memory.  With nothing running,
`--letta` starts Ollama, pulls both models, starts `letta server` pointed at
Ollama, and stops both on exit.  Anything already listening is used as-is and
never shut down.

Letta's server keeps its state in Postgres, so the docker image (which brings
one) is the path that works with nothing else installed:

    docker run -d --name fruitbrains-letta --network host \
      --restart unless-stopped \
      -e OLLAMA_BASE_URL=http://127.0.0.1:11434 \
      -v ~/.letta/pgdata:/var/lib/postgresql/data letta/letta:latest
    python3 fruitbrains.py --letta          # finds it on :8283 and leaves it alone

`--network host` is doing real work: Ollama binds `127.0.0.1`, so a bridged
container cannot reach it, and the symptom is not a connection error but Letta
listing *no models at all*.  Sharing the host's network sidesteps that without
having to re-bind Ollama.  The volume is what makes the villagers durable --
their memories live in Letta's Postgres, not in this repo, so pointing that
somewhere temporary means a town that forgets every reboot.

`--letta` needs the server up *first*: without one, the game refuses to open a
window and prints how to start it (exit 2).  That is the same rule as the model
itself -- a villager who has quietly lost their memory is worse than a game
that will not start.

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
