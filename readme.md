# Tales of Derision

GOALS:
- use ecs, esper
- make a roguelike game in pygame-ce.
- start with very basic and go from there
- wasd movement, fully configurable configurable controls, controller support eventually
- lot's of simulation, testing, and prcedural generation
- mods are a first class priority.  anybody should easily be able to add there own files to add or change content.
- seed based determinism
## Run

Install dependencies:

  python3 -m pip install --user esper pygame

  python3 src/main.py

Or bypass the title screen/main menu and load a specific save file:

  python3 src/main.py --save_file src/data/saves/my_run.json

Stress test with cave rats on every walkable map square:

  python3 src/main.py --rat-flood

Move: hold WASD, press Space to take a step — or press Space with no direction
held to **wait in place and pass a turn**. Player menu: Tab (Inventory/Status/
…). Inventory: I. Status: C. Pause menu: Esc.

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

## Survival (food, water, wood)

The player has **hunger**, **thirst**, and **tiredness** that rise every turn (a
step, or a `Space` wait; shown on the bottom status line). Keep them down or
you'll get escalating warnings:

- **Water** — face the stone **well** (`O`) and press the interact key (Enter) to
  drink and quench your thirst. A `Waterskin` in your inventory can also be drunk
  (from the inventory screen) as portable water.
- **Meat** — kill an enemy, then face its **corpse** and press Enter to open the
  loot menu; every corpse yields meat you can take, named for the creature (a rat
  drops `Rat Meat`, a goblin drops `Goblin Meat`). Raw meat barely fills you —
  cook it first.
- **Wood** — face a **tree** (`T`) and press Enter to chop a piece of `Wood`. The
  tree falls after a few chops.
- **Cooking** — face the iron **stove** (`#`) with `Wood` **and** any raw meat in
  your inventory and press Enter: the wood fuels a fire that cooks the meat into a
  filling `Cooked ... Meat`.
- **Eating/drinking** — open the inventory (`I`), select a food or drink on the
  items side, and press Enter to consume it (everything else equips as before).

Interaction targets the tile you're facing: hold a movement key toward the
well/tree/stove/corpse, then press Enter (the same targeting used for talking to
NPCs).

Every creature (player and NPCs alike) accumulates hunger, thirst, and tiredness
each turn, though only the player's needs are surfaced as on-screen warnings.

## Day/night cycle and sleep

Time flows with your turns. A full **day** runs dawn → day → dusk → night (the
current phase leads the bottom status line), and the world **visibly darkens**
after dusk. Every character must **sleep once a day**: **tiredness** climbs each
turn and climbs **twice as fast at night**, so you can't stay up forever.

- **Sleep** — press `R` to rest. Standing next to your **bed** (`=`, in your camp
  by the well and stove) beds you down **at home**; anywhere else you **set up
  camp** (a campfire `^` appears) and sleep there. You can also face the bed and
  press Enter. Sleeping fast-forwards turns — the whole world keeps simulating
  through the night — until you wake rested, then breaks camp automatically.
- **NPCs sleep too.** When a creature gets tired it drops what it's doing and
  heads for bed, **preferring its home**: villagers walk back to their house,
  while homeless wildlife (and anyone too exhausted to make it home) just camp
  where they stand. Sleepers skip their turn until their tiredness recovers.

The clock, phase boundaries, tiredness rates, and the night multiplier live at
the top of [src/systems.py](src/systems.py) and are easy to tune.

**Calendar.** Turns also drive a calendar: a **year is 4 months, each 4 weeks of
7 days** (112 days a year). The status line leads with the full date and clock
time, e.g. `Y1 M2 W3 D4 13:45 Day` (year, month, week, day-of-week, time, phase).
See `calendar` / `format_datetime` in [src/systems.py](src/systems.py).

## Houses, residents, and building

A **house** is any floor area fully sealed by walls (`#`) and/or windows (`o`)
whose only opening is a **door** (`+`) -- windows block movement but you (and the
light) can see through them, doors you can walk through. Enclosed rooms are found
by a flood-fill in [src/game_map.py](src/game_map.py) (`find_enclosed_rooms`).

Every house is furnished with a **bed**, an **oven** (the cooking stove), a
**chest**, a **table**, a **wardrobe**, and a **bookshelf**
(`furnish_house` in [src/systems.py](src/systems.py)).

**Houses belong to people.** A house is owned via its **bed** and **chest** (an
`Owned` marker records the owner). Your own bed by the well belongs to *you* from
the start, so villagers will never claim it — even if you wall it into a proper
house. A house with no owner (or whose owner no longer exists) is up for grabs.

**Respecting property.** Trying to **sleep in someone else's bed** or **open
someone else's chest** (`n`) pops a *"This belongs to &lt;name&gt;. Are you sure?"*
prompt (defaults to **No**, `Esc` cancels) — your own and unowned property is used
without a fuss. Villager chests hold a bit of loot, so you *can* help yourself if
you're willing to; the game just makes sure it was on purpose.

**Residents move in.** Villagers are *residents*: a villager who owns **no** house
claims the nearest **unowned** one it can walk to (marking it as theirs), then
walks there to sleep and cooks at its oven. Once someone owns a house nobody else
takes it, so two villagers never share one — and a villager who already owns a
home is left alone (it never re-claims or rebuilds).

**Villagers build.** Only a resident that owns **no** house builds: it takes a
preset cabin design, finds a clear spot, gathers **wood**, and raises the walls
and door one piece at a time over many turns (it still stops to eat, drink, and
sleep, so a cabin takes a while). When the last piece goes up the cabin is
furnished and the builder becomes its owner.

**You can build too.** Open the **Craft** tab in the `Tab` menu to craft `Wall`,
`Window`, and `Door` pieces out of `Wood` (walls/windows cost 2, doors 3). The
crafted pieces go into your pack; switch to the **Inventory** tab, select a
piece, and press a **direction** to place it on that adjacent tile -- seal off a
room with four walls and a door and you've built a house of your own. Recipes and
the item/tile mapping live in [src/items.py](src/items.py).

**The world is a living ecosystem.** The map is a **3×3 grid of sections**. You
are only ever in one section at a time: the camera shows just your current
section, and when you walk off its edge you cross into the neighbouring section
("You cross into a new area."). The other eight sections are **still fully
simulated** every turn — deer graze and drink, predators hunt — so the world
keeps living while you're elsewhere.

Two lakes and a river (`~` water) run through it. Wild **deer** (`d`) roam:
when hungry they graze trees, when thirsty they drink from lakes and rivers.

**The forest grows and regrows.** Each day, every open outdoor ground tile has a
small chance of sprouting a **seedling** -- a **tree** (`t` -> `T`, 0.01%/tile) or,
less often, a **berry bush** (`,` -> `%`) -- and every mature tree/bush has a
**0.005% chance** (half that) of dying. A seedling that survives a full **year**
(112 days) matures. So the woods slowly spread across open ground while old
plants fall, and as villagers chop wood and deer graze the stands down, fresh
seedlings keep reseeding -- a living, self-sustaining forest. The per-day odds
and the map's soft plant cap live in `TreeGrowthProcessor` in
[src/systems.py](src/systems.py); growth is evaluated once per in-game day.

**Berry bushes** (`%`) are a renewable food source. A ripe bush shows red; face
it and press the interact key (Enter) to **pick a handful of `Berries`** (eaten
from the inventory like any food). The picked bush turns bare green, and grows a
**fresh crop of berries 7 days later** -- so a patch of bushes feeds you (and
foraging villagers) indefinitely if you don't strip it faster than it regrows.

When hungry, **villagers cook**, just like you: they scavenge meat from a corpse
(or hunt a deer), chop a tree for **wood**, carry both to a **stove**, cook the
raw meat into a meal, and then eat it — meat is never eaten raw. If no game or
meat is reachable in their area, they fall back to **foraging from trees**
(nuts/berries — renewable, unlike deer) rather than starve. Predator monsters
(**goblins**, **cave rats**) are less civilised and eat raw meat on the spot. You
can hunt deer too: walk into one to take it down for `Deer Meat`.

NPCs only pursue food, water, and homes they can actually **walk to**: the map
tracks connected walkable regions (`region_of`/`same_region` in
[src/game_map.py](src/game_map.py)), so a villager won't strand itself trying to
reach a deer or a house on the far bank of a river it can't cross — it forages
what's on its side, and builds a home there if it can't reach an empty one.

**Player menu.** Press `Tab` to open the player menu — a tabbed screen with
**Inventory** and **Status** (with **Map**, **Journal**, and **Skills** stubbed
for later). `Tab` reopens on whichever tab you last used; inside the menu, `Tab`
cycles tabs. Direct keys still jump straight to a tab: `I` opens Inventory, `C`
opens Status (pressing the same key again closes). `Esc` closes.

To read another creature, face it and press the interact key (Enter): friendlies
open a dialogue with a **Status** option (like Trade), while wild or hostile
creatures show a read-only **examine** panel with their disposition, hunger,
thirst, tiredness, and statuses (including whether they're **asleep**).

**Swimming.** Water blocks NPCs and line of sight, but *you* can wade in and
swim across lakes and rivers (walls still block).

**Status identifiers.** A character's tile animates through its own glyph plus an
identifier for each active status, each for a configurable length of time, in a
repeating cycle that plays in real time even while you stand still. Swimming
shows your tile for 1s then a `~` for 0.5s; sleeping shows a `Z`. Statuses
**stack sequentially**: if you were also on fire, the cycle would be your tile
(1s) → `~` (0.5s) → a red `F` (0.5s) → repeat. Durations and identifiers live in
`_STATUS_DISPLAY` /
`_STATUS_BASE_SECONDS` in [src/systems.py](src/systems.py).

Survival item names and their food/water values live in
[src/items.py](src/items.py). Note: survival state is not saved yet — the world
regenerates each session (only map size + player position are persisted).

## Menus

- Startup flow: Title Screen -> Main Menu (`Continue`, `New Game`, `Quit`)
- Player menu: press `Tab` to open the tabbed player menu (`Inventory`, `Craft`,
  `Status`, and stubbed `Map`/`Journal`/`Skills`). `Tab` reopens on the last tab
  and cycles tabs from inside; `I`/`C` jump straight to Inventory/Status (and
  toggle closed). The `Craft` tab builds wall/window/door pieces from wood; place
  them from the `Inventory` tab by selecting one and pressing a direction.
- In-game pause menu: press `Esc` to open Pause Menu (`Save Game`, `Options`, `Quit`)
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