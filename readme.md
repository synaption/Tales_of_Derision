# Tales of Derision

A turn-based, procedurally-generated roguelike **world simulation** in Python
(pygame-ce + [esper](https://github.com/benmoran56/esper) ECS). A living island of
villagers, wildlife, and monsters that eat, drink, sleep, forage, cook, build
houses, form families, and reproduce — all while the whole map keeps simulating
around you, seed-deterministic from top to bottom.

This README is the **high-level design document and the entry point** to the rest
of the project docs. It holds the vision and the running brain-dump of ideas; the
[wiki/](wiki/Home.md) holds the concrete "how it works / how to play / how to
extend it" reference.

---

## Quick start

```bash
./scripts/install.sh

python3 src/main.py                                   # play
python3 src/main.py --save_file src/data/saves/x.json # load a save (skips menus)
python3 src/main.py --rat-flood                        # stress test: a rat on every tile
./run_tests.sh                                         # 236 headless tests, ~1s
```

Move: hold **WASD**, press **Space** to step (or Space with no direction to wait a
turn). **Enter** interacts with the tile you face. **Tab** player menu, **I**
inventory, **C** status, **R** sleep, **Esc** pause. Full controls and the survival
/ building / social loops are in [wiki/Gameplay](wiki/Gameplay.md).

---

## Documentation (the wiki)

| Page | What it covers |
|------|----------------|
| [Home](wiki/Home.md) | Wiki landing page and status |
| [Getting Started](wiki/Getting-Started.md) | Install, run, controls, saves/options |
| [Gameplay](wiki/Gameplay.md) | The player-facing manual: survival, day/night, houses, building, the ecosystem |
| [Architecture](wiki/Architecture.md) | ECS model, the layer DAG, the turn loop, data flow |
| [Components](wiki/Components.md) | Every data component entities are built from |
| [Systems](wiki/Systems.md) | Every processor + free-function subsystem |
| [Game Map](wiki/Game-Map.md) | Ocean/island world, pathfinding, regions, enclosed rooms |
| [Renderers](wiki/Renderers.md) | The renderer seam and how to add a backend |
| [Action Economy](wiki/Action-Economy.md) | Time-unit turn scheduling (speed, action cost) |
| [World Simulation](wiki/World-Simulation.md) | Region scheduler, background catch-up, "living world" |
| [Content & Mods](wiki/Content-and-Mods.md) | Prefabs, kits, effects, items — adding/modding content |
| [Performance](wiki/Performance.md) | Caching invariants and the per-turn cost budget |
| [Autotiling](wiki/47-tile_autotiling.md) · [Gibberish](wiki/Gibberish.md) | The 47-tile wall/water masks; the fake NPC language |
| [Roadmap](wiki/Roadmap.md) | What's done and what's next |

**Reading order of authority** (per `notes4LLMs.md`): `notes4LLMs.md` → this
README → the wiki.


## Windows executable packaging

Build a standalone Windows executable from Linux with Docker:

```bash
./scripts/build_windows.sh
```

The script builds a Wine-based Linux container, installs Windows Python plus the
packaging dependencies inside that container, and writes `dist/TalesOfDerision.exe`.
It only builds the executable; it does not attempt to launch it under Linux. Copy
the resulting executable to Windows to run it.

The builder intentionally uses the version of `pip` bundled with Windows Python.
Do not upgrade it in Wine: newer `pip` versions can call the Windows `CopyFile2`
API, which is not implemented by the Ubuntu Wine version and can leave Wine's
debugger waiting indefinitely. If an older build stopped at a `CopyFile2` error,
press **Ctrl+C**, pull this version of the script, and run the build again.

Dependency installation and PyInstaller run under one virtual X server. Starting
each Wine command with a separate `xvfb-run` can shut down the first display while
Wine is still using it, resulting in `X connection ... broken` followed by a hang.
The image shuts down its setup-time Wine server before it is saved, and the build
prints progress before each Wine command. Dependency installation times out after
15 minutes and packaging after 30 minutes rather than waiting forever.
Before packaging, the builder verifies that Windows Python imports the complete
`pygame` package. The PyInstaller specification explicitly collects pygame's
Python modules, data, and native DLLs because the game imports pygame dynamically.
It likewise collects the complete `content` package because the content loader
discovers core creatures, items, effects, flora, and features by module name at
runtime rather than through static imports.
At runtime, the frozen application resolves tiles and sounds from PyInstaller's
bundle directory. Saves and options—including tile scale—are stored persistently
under `%LOCALAPPDATA%\TalesOfDerision` instead of the temporary bundle directory.

GitHub Actions uses the same Linux build script and uploads the executable artifact
through `.github/workflows/windows-exe.yml` on pull requests, manual dispatches, and
pushes to `dev` or `work`.

***
## Design Goals Big Picture
Thousands of Years Simulations

Economy, Money, Banking, Farming, Hunting, Fishing, Thirst, Hunger

Action Economy:

There is a certain amount of time in a day.
There is a turn order.
Actions take a certain amount of time based on a number of factors, quickness, movement speed, agility, ect.
Turn order is decided based on when actions are completed. So everything is in action or it's waiting for it's next turn.
The effects of the action are immediate. i.e. an attack happens, the damage is done immediately, the attacker is in the attack state for a certain amount of time units, and then they are in a wait state until it is there turn.
I will try to balance the action economy so that one day of typical gameplay ends up being 1 hour in real life.
animations happen either in order, or multiple at the same time, depending on what they are.

All NPCs are playable.

Players and NPCs have the same needs as the player like food and water.

When the player dies they become a random sentiaent NPC somewhere in the world. It's like "Roy" from Rick and Morty.

Morrorwind style leveling. You level up individual skills, when you level up those skills you gain a character level and can upgrade attributes str, dex, con, int, wis, char. You also get more health and magic if you have magic.


### Inspiration
Caves of Qud
Lord of the Rings
DaFluffyPotato
Minecraft
Rimworld
Dwarf Fortress
Song of Syx
Infectionator World Dominator
Earth Defense Force
chrono trigger
zelda
Themes
Fantasy

Time Travel

[BRAINSTORM] Time is cyclical. Hyper advanced civilization makes floating islands, destroys the planet, and then inteligently redesigns new planets from their floating society. These floating societies tend to be sparcely populated by a few super adept NPCs. These "gods" die and inhabit the same reincarnation loop as you. i.e. you are a reincanant. You have of course forgotten this.
Zombie

MacGuffins/Plot Coupons

the one ring
the infinity stones
dragon balls
Knowledge and Teaching

Music, and Comrodery

Ballence and Equilibrium

Karma and Reincarnation

Design Goals Big Picture
Thousands of Years Simulations

Economy, Money, Banking, Farming, Hunting, Fishing, Thirst, Hunger

Action Economy:

There is a certain amount of time in a day.
There is a turn order.
Actions take a certain amount of time based on a number of factors, quickness, movement speed, agility, ect.
Turn order is decided based on when actions are completed. So everything is in action or it's waiting for it's next turn.
The effects of the action are immediate. i.e. an attack happens, the damage is done immediately, the attacker is in the attack state for a certain amount of time units, and then they are in a wait state until it is there turn.
I will try to balance the action economy so that one day of typical gameplay ends up being 1 hour in real life.
animations happen either in order, or multiple at the same time, depending on what they are.
Targets

desktop fully rendered on windows and linux
steam
itch.io via Pygbag
github pages via Pygbag
All NPCs are playable.

Players and NPCs have the same needs as the player like food and water.

When the player dies they become a random sentiaent NPC somewhere in the world. It's like "Roy" from Rick and Morty.

Morrorwind style leveling. You level up individual skills, when you level up those skills you gain a character level and can upgrade attributes str, dex, con, int, wis, char. You also get more health and magic if you have magic.

fire emblem like evolutions, mainly asthetic.  

Style
Pixel Art

HD text

shader effects, lighting

basic animations, or no animations at all

characters face the direction they are going, either just left or right, or up, down, left, and right, or all 8 directions depending on the sprite.

Dialogue in a fake gibberish language "##!/$*~# GH01^@"

speach bubbles, and sims like symbol popups i.e. ++

Characters
Wizards Great Fairy NPCs Mostly Farmers