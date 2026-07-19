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

    python3 main.py

Or bypass the title screen/main menu and load a specific save file:

    python3 main.py --save_file data/saves/my_run.json

Move: wasd (also arrow keys / hjkl). Menu: Esc. Save/Quit are in the menu.
(Needs a real terminal — curses won't run from an IDE output pane.)

## Audio

- On startup, the game now attempts to play the first supported file found in
  `audio/music/` on loop (`.mp3`, `.ogg`, `.wav`, `.flac`, `.m4a`).
- If no audio file is found, `pygame` is not installed, or audio init fails,
  the game continues silently.
- `audio_buffer` in `data/config/options.json` controls mixer buffer size
  (default `16384` in this build). If audio is still choppy, increase it.

Install `pygame` for music playback:

  python3 -m pip install --user pygame

## Testing

Tests are headless and do not require curses. They use a fake renderer test
double to validate ECS movement + render-loop behavior.

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
- In-game menu: press `Esc` to open Pause Menu (`Save Game`, `Options`, `Quit`)
- Pause menu navigation: arrows/WASD to move selection, Enter to select, `Esc` to resume game
- Options menu: toggle `Fullscreen` and `Show FPS`; changes are written to working options file

## Saves and options

- Default save file: `data/saves/default_save.json`
- Default options file: `data/config/default_options.json`
- Working options file: `data/config/options.json`
- On startup, if `data/config/options.json` is missing, it is copied from
  `data/config/default_options.json`.
- User save files can live in `data/saves/*.json` and can be loaded directly
  with `--save_file`.
- `--save_file` also bypasses title screen and main menu.

## Documentation

Full docs live in the [wiki/](wiki/Home.md) — architecture, ECS model, per-module
reference, how to add a renderer backend, and the roadmap.

## Layout

| file                   | role                                                      |
|------------------------|-----------------------------------------------------------|
| `main.py`              | entry point + turn loop                                   |
| `components.py`        | ECS data: `Position`, `Renderable`, `Player`              |
| `game_map.py`          | tile grid (renderer-agnostic)                             |
| `systems.py`           | `MovementProcessor`, `RenderProcessor` (esper processors) |
| `renderer/base.py`     | `Renderer` interface — the seam for swapping backends     |
| `renderer/terminal.py` | curses implementation of `Renderer`                       |

ECS via [esper](https://github.com/benmoran56/esper) (3.x, module-level API).

## Swapping renderers later

Game/system code never imports curses. To add tcod/pygame/raylib, implement the
`Renderer` interface (`setup/teardown/clear/draw_glyph/draw_text/present/
poll_action`) and pass that instance to `RenderProcessor` in `main.py`. Input is
already abstracted to action strings (`move_up`, `quit`, …), so nothing else
changes.

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
- itch.io

All NPCs are playable.

Players and NPCs have the same needs as the player like food and water.

When the player dies they become a random sentiaent NPC somewhere in the world.  It's like "Roy" from Rick and Morty.  

## Style
Pixel Art

HD text

shader effects, lighting

basic animations, or no animations at all

characters face the direction the are going, either just left or right, or up, down, left, and right, or all 8 directions depending on the sprite.  

Dialogue in a fake gibberish language "##!/$*~# GH01^@"  

speach bubbles, and sims like symbol popups i.e. ++ 

## Characters
Wizards
Great Fairy
NPCs
Mostly Farmers