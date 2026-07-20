# Tales of Derision Wiki

An ECS-based roguelike in Python with a pygame runtime and a renderer seam that
keeps game logic backend-agnostic.

## Status

- ✅ esper ECS wired up (3.x, module-level API)
- ✅ Pygame renderer with scalable tile/UI presentation
- ✅ Turn-based movement, collision, combat bump, and NPC movement
- ✅ Title screen + main menu + in-game pause menu
- ✅ Dialogue, inventory/equipment, and trade menus
- ✅ Save/load flow with default and user save files
- ✅ Default/working options files with auto-bootstrap
- ✅ Renderer decoupled behind an interface for headless tests
- ⬜ Expanded content, richer AI, world simulation depth (see [Roadmap](Roadmap.md))

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

Game and system code avoid direct renderer dependencies. Display goes through a
`Renderer` interface and input arrives as abstract action strings (`move_up`,
`menu_select`, `open_pause_menu`, ...). This is what keeps integration tests
headless and makes future backend work isolated to renderer code. See
[Renderers](Renderers.md).
