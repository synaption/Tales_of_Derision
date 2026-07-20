# Tales of Derision

GOALS:
- use ecs, esper
- make a roguelike game in pygame-ce.
- start with very basic and go from there
- wasd movement, fully configurable configurable controls, controller support eventually
- lot's of simulation, testing, and prcedural generation
- mods are a first class priority.  anybody should easily be able to add there own files to add or change content.
- seed based
## Run

Install dependencies:

  python3 -m pip install --user esper pygame

  python3 src/main.py

Or bypass the title screen/main menu and load a specific save file:

  python3 src/main.py --save_file src/data/saves/my_run.json

Stress test with cave rats on every walkable map square:

  python3 src/main.py --rat-flood

Move: hold WASD, press Space to take a step. Menu: Esc. Inventory: I.

## Web build (pygbag)

Install pygbag (once):

  python3 -m pip install --user pygbag esper

Build the web bundle:

  bash scripts/build_pygbag.sh

Output is written to:

  build/web

Preview locally (build + serve + open, frees the port and cleans up on exit):

  ./online.sh

Then open `http://localhost:8000` in your browser. `online.sh` serves via
`scripts/serve_coi.py`, which sets cross-origin isolation headers (COOP/COEP); a
plain server also works:

  python3 -m http.server --directory build/web 8000

Use port `8000` for local preview. This pygbag runtime rewrites package fetches
to `http://localhost:8000/cdn/...` while booting; serving on other ports can
leave the page stuck at `Loading, please wait ...`.

If you still see an old screen that does not advance, do a hard refresh
(`Ctrl+Shift+R`) to clear cached web assets.

Running `python3 -m pygbag ...` directly from repo root can regenerate
`build/web` with a different layout while debugging — prefer `./online.sh` or the
static server above.

### Web build architecture (emscripten / pygbag)

Pygbag runs the **entire** game on the browser's single main thread, and audio is
mixed on that same thread (no SharedArrayBuffer / worker threads in this runtime).
Every decision below exists to keep that one thread responsive. Desktop is
unaffected — these paths are gated on `IS_WEB` (`sys.platform == "emscripten"`),
so the native build keeps its original blocking/threaded behavior.

**Cooperative async loop (or the tab freezes).**
- Entry is async: `main.py` -> `src/main.py:main`, run via `asyncio.run`.
- All input waits go through `_await_action`. On web it polls non-blocking and
  `await asyncio.sleep(_WEB_IDLE_POLL_SECONDS)` (~1/60 s, a *real* frame) between
  polls, and yields once before returning each action.
- **Never** add blocking loops (`while True: renderer.poll_action()`,
  `pygame.time.wait`, `time.sleep`) in the input path — on pygbag they never
  yield and freeze the page at `Loading, please wait ...`. New menus/loops must
  be `async def` and use `await _await_action(renderer)`.
- Why a real frame and not `asyncio.sleep(0)`: a zero-delay spin pegs the single
  thread and starves the audio mixer, causing clicks/pops.

**Display sizing & crispness.**
- Browser "fullscreen" isn't a real mode. On web the surface is sized to the
  visible canvas (`window.innerWidth/innerHeight`) so 1 surface px ~= 1 CSS px;
  an oversized surface gets scaled down and everything looks tiny.
- The canvas gets `image-rendering: pixelated` after each `set_mode` so text and
  tiles stay crisp instead of blurred by the browser's default smoothing.

**Audio.**
- Music is a **44.1 kHz OGG** played as a looped in-memory `Sound` on a reserved
  channel — *not* `pygame.mixer.music` (streaming is unreliable in pygbag) and
  *not* mp3 (often won't decode). The OGG matches the mixer rate (44100) so there
  is no resampling crackle. Desktop still streams `mixer.music`.
- The web mixer uses a moderate/large buffer (`WEB_AUDIO_BUFFER`, overridable via
  `web_audio_buffer` in `options.json`) and opens with
  `AUDIO_ALLOW_FREQUENCY_CHANGE` so SDL matches the browser's device sample rate
  instead of resampling every callback.
- Any residual occasional pop is pygbag's main-thread audio jitter. True threaded
  audio needs cross-origin isolation (COOP/COEP), which did not engage in pygbag
  0.9.3; `scripts/serve_coi.py` serves those headers locally and is kept for a
  future runtime that supports it.

**Render cost (keep per-turn work small so it can't starve audio).**
- Field-of-view is memoized per player position; wall autotile masks are cached
  (static map); non-visible cells are skipped (`clear()` already blacks them).
- The map is composited from a cached **world-coordinate** surface: a walking
  step blits the visible FOV box (offset by the camera scroll) plus a few shadow
  fills, instead of re-drawing hundreds of tiles. The blit is **clipped to the
  viewport** so it can't bleed into the sidebar when zoomed/scrolled.
- Menus with a game backdrop snapshot the scene once (`capture_backdrop`) and
  reuse it, instead of re-rendering the whole game every keypress.
- Invariant: keep the caches invalidated correctly if you touch tiles, scale, or
  camera — `invalidate_map_surface` / `invalidate_backdrop` run on `set_mode`,
  and the FOV/map caches assume a static map (walls don't change mid-session).

**Build.**
- `scripts/build_pygbag.sh` bundles the pygame-ce wasm wheel + esper, vendors
  `browserfs.min.js`, and prunes redundant music sources from the web bundle
  (drops `.wav`/`.mp3` when an `.ogg` sibling exists) to shrink the download.

## GitHub Pages deployment

- Workflow file: `.github/workflows/pygbag-pages.yml`
- Trigger: push to `main`/`master` (or run manually from Actions)
- One-time setup: repo Settings -> Pages -> Source -> GitHub Actions
- Optional auto-enable: add repository secret `PAGES_ADMIN_TOKEN` (PAT with repo + pages admin/write permissions)

After the workflow finishes, the deploy job prints the final Pages URL.

## Audio

- On startup, the game now attempts to play the first supported file found in
  `audio/music/` on loop (`.mp3`, `.ogg`, `.wav`, `.flac`, `.m4a`).
- If no audio file is found, `pygame` audio init fails, or the mixer cannot
  open a device,
  the game continues silently.
- `audio_buffer` in `src/data/config/options.json` controls mixer buffer size
  (default `16384` in this build). If audio is still choppy, increase it.

## Tilesets

- Render flow now follows:
  `Game object -> glyph + fg + bg -> tileset lookup -> PNG tile blit`.
- `Renderable` supports optional `fg` and `bg` values.
- `gfx/tilesets/pygame_tileset_config.json` now supports `tile_id` entries
  from `gfx/tilesets/Hexany/tile_index.csv`.
- If a glyph is not found in configured lookup tables, renderer falls back to
  `gfx/tilesets/Bisasam_16x16.png` by default.
- The fallback sheet is configurable and can be replaced with another
  dwarf-fortress style CP437 tilesheet via the config `fallback` section.

## Testing

Tests are headless and do not require a live game window. They use a fake
renderer test double to validate ECS movement + render-loop behavior.

Install pytest (once):

  python3 -m pip install --user pytest

Run tests:

  ./run_tests.sh

This shows per-test status (`PASSED`/`FAILED`) and basic timing metrics.

Run only headless renderer integration tests:

  ./run_tests.sh headless

Run only totally unrendered logic/data tests:

  ./run_tests.sh unrendered

## Menus

- Startup flow: Title Screen -> Main Menu (`Continue`, `New Game`, `Quit`)
- In-game menu: press `Esc` to open Pause Menu (`Options`, `Quit`)
- Pause menu navigation: WASD to move selection, Enter to select, `Esc` to resume game
- Options menu: toggle `Fullscreen` and `Show FPS`; changes are written to working options file

## Saves and options

- The game should save at the end of every turn.
- Default save file: `src/data/saves/default_save.json`
- Default options file: `src/data/config/default_options.json`
- Working options file: `src/data/config/options.json`
- On startup, if `src/data/config/options.json` is missing, it is copied from
  `src/data/config/default_options.json`.
- User save files can live in `src/data/saves/*.json` and can be loaded directly
  with `--save_file`.
- `--save_file` also bypasses title screen and main menu.
- Keybinds are action-based in `src/data/config/options.json` (`move_up`,
  `confirm_action`, `open_pause_menu`, etc.).

## Documentation

Full docs live in the [wiki/](wiki/Home.md) — architecture, ECS model, per-module
reference, how to add a renderer backend, and the roadmap.

## Layout

| file                   | role                                                      |
|------------------------|-----------------------------------------------------------|
| `src/main.py`              | entry point + turn loop                               |
| `src/components.py`        | ECS data: `Position`, `Renderable`, `Player`          |
| `src/game_map.py`          | tile grid (renderer-agnostic)                         |
| `src/systems.py`           | `MovementProcessor`, `RenderProcessor` (esper processors) |
| `src/renderer/base.py`     | `Renderer` interface seam                              |
| `src/renderer/pygame_renderer.py` | pygame implementation of `Renderer`          |

ECS via [esper](https://github.com/benmoran56/esper) (3.x, module-level API).

## Swapping renderers later

Game/system code stays renderer-agnostic. The current runtime uses
`PygameRenderer`, but future backends can still implement the same
`Renderer` interface (`setup/teardown/clear/draw_glyph/draw_text/present/
poll_action`) and be passed to `RenderProcessor`.

## Inspiration
- Caves of Qud
- Lord of the Rings
- DaFluffyPotato
- Minecraft
- Rimworld
- Dwarf Fortress
- Song of Syx
- Infectionator World Dominator
- Earth Defense Force
- chrono trigger
- zelda


## Themes
Fantasy

Time Travel
- [BRAINSTORM] Time is cyclical.  Hyper advanced civilization makes floating islands, destroys the planet, and then inteligently redesigns new planets from their floating society.  These floating societies tend to be sparcely populated by a few super adept NPCs.  These "gods" die and inhabit the same reincarnation loop as you.  i.e. you are a reincanant.  You have of course forgotten this.  

Zombie

MacGuffins/Plot Coupons
- the one ring
- the infinity stones
- dragon balls

Knowledge and Teaching

Music, and Comrodery

Ballence and Equilibrium

Karma and Reincarnation


## Design Goals Big Picture
Thousands of Years Simulations

Economy, Money, Banking, Farming, Hunting, Fishing, Thirst, Hunger

Action Economy:
- There is a certain amount of time in a day.
- There is a turn order.
- Actions take a certain amount of time based on a number of factors, quickness, movement speed, agility, ect.
- Turn order is decided based on when actions are completed.  So everything is in action or it's waiting for it's next turn. 
- The effects of the action are immediate.  i.e. an attack happens, the damage is done immediately, the attacker is in the attack state for a certain amount of time units, and then they are in a wait state until it is there turn.   
- I will try to balance the action economy so that one day of typical gameplay ends up being 1 hour in real life.  
- animations happen either in order, or multiple at the same time, depending on what they are.  

Targets
- desktop fully rendered on windows and linux
- steam
- itch.io via Pygbag
- github pages via Pygbag

All NPCs are playable.

Players and NPCs have the same needs as the player like food and water.

When the player dies they become a random sentiaent NPC somewhere in the world.  It's like "Roy" from Rick and Morty.  

Morrorwind style leveling.  You level up individual skills, when you level up those skills you gain a character level and can upgrade attributes str, dex, con, int, wis, char.  You also get more health and magic if you have magic.  

## Style
Pixel Art

HD text

shader effects, lighting

basic animations, or no animations at all

characters face the direction they are going, either just left or right, or up, down, left, and right, or all 8 directions depending on the sprite.  

Dialogue in a fake gibberish language "##!/$*~# GH01^@"  

speach bubbles, and sims like symbol popups i.e. ++ 

## Characters
Wizards
Great Fairy
NPCs
Mostly Farmers