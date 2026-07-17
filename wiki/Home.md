# pyRL2 Wiki

An ECS-based roguelike in Python that runs in the terminal today and is built to
be re-rendered with tcod, pygame, raylib, or OpenGL later — without rewriting the
game.

## Status

- ✅ esper ECS wired up (3.x, module-level API)
- ✅ Bordered map rendered to the terminal via curses
- ✅ Movable `@` with wall collision (arrows / hjkl / wasd)
- ✅ Title screen + main menu + in-game pause menu
- ✅ Save/load flow with default and user save files
- ✅ Default/working options files with auto-bootstrap
- ✅ Renderer fully decoupled behind an interface
- ⬜ Enemies, FOV, dungeon generation, real turn queue (see [Roadmap](Roadmap.md))

## Pages

| Page | What it covers |
|------|----------------|
| [Getting Started](Getting-Started.md) | Requirements, install, run, controls |
| [Architecture](Architecture.md) | ECS model, layering, the turn loop, data flow |
| [Components](Components.md) | The data pieces entities are made of |
| [Systems](Systems.md) | The processors that act on components |
| [Game Map](Game-Map.md) | The tile grid and how it stays renderer-agnostic |
| [Renderers](Renderers.md) | The `Renderer` seam and how to add a backend |
| [Roadmap](Roadmap.md) | What's done and what's next |

## The one idea to take away

Game and system code **never import curses**. Display goes through a `Renderer`
interface and input arrives as abstract action strings (`move_up`, `menu_select`,
`open_pause_menu`, …). Swapping to raylib/pygame means writing one class —
nothing else changes. See
[Renderers](Renderers.md).
