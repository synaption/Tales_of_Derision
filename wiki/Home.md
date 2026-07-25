# Tales of Derision Wiki

An ECS-based roguelike **world simulation** in Python with a pygame runtime and a
renderer seam that keeps game logic backend-agnostic. Seed-deterministic; built to
scale toward thousands of tiles and NPCs.

The top-level [README](../readme.md) is the high-level design doc and vision dump.
This wiki is the concrete reference.

## Status

- ✅ esper ECS (3.x, module-level API); renderer decoupled behind an interface for
  headless tests
- ✅ Time-based **action economy** (turn order by completion time) + deterministic
  scheduler
- ✅ **Region scheduler**: far sections lag and pay down their simulation debt
  nearest-first, so the whole 120×60 island stays alive cheaply
- ✅ Ocean/island worldgen; A* + flow-field pathfinding; connected-region reachability
- ✅ Survival (hunger/thirst/tiredness), day/night + calendar, sleep/camp
- ✅ NPC AI: forage/graze/hunt/cook, drink, sleep, build, socialize
- ✅ Houses: ownership, residents, shared blueprint→haul→raise construction
- ✅ Social sim: personality traits, friendships, gibberish speech bubbles
- ✅ Family: marriage, mating cooldowns, pregnancy, birth; procedural names (onymancer)
- ✅ Flora ecosystem: trees, berry bushes, saplings, seaweed; grow/regrow/die
- ✅ Fog-of-war tile memory; section camera; status-identifier animation
- ✅ Hybrid content/mod registry: prefabs, kits, effects, items, loader (see
  [Content & Mods](Content-and-Mods.md))
- ⬜ HP/damage/death model, skills/leveling, economy — see [Roadmap](Roadmap.md)

## Pages

| Page | What it covers |
|------|----------------|
| [Getting Started](Getting-Started.md) | Install, run, controls, saves/options |
| [Gameplay](Gameplay.md) | Player manual: survival, day/night, houses, building, ecosystem |
| [Architecture](Architecture.md) | ECS model, the layer DAG, the turn loop, data flow |
| [Components](Components.md) | Every data component entities are built from |
| [Systems](Systems.md) | Every processor + free-function subsystem |
| [Game Map](Game-Map.md) | Ocean/island world, pathfinding, regions, enclosed rooms |
| [Renderers](Renderers.md) | The renderer seam and how to add a backend |
| [Action Economy](Action-Economy.md) | Time-unit turn scheduling |
| [World Simulation](World-Simulation.md) | Region scheduler + background catch-up |
| [Content & Mods](Content-and-Mods.md) | Prefabs, kits, effects, items — adding/modding content |
| [Performance](Performance.md) | Caching invariants and the per-turn cost budget |
| [Autotiling](47-tile_autotiling.md) | 47-tile wall/water neighbour masks |
| [Gibberish](Gibberish.md) | The fake NPC language |
| [Roadmap](Roadmap.md) | What's done and what's next |

## The one idea to take away

Game and system code never import a renderer. Display goes through a `Renderer`
interface and input arrives as abstract action strings (`move_up`, `confirm_action`,
`open_pause_menu`, …). That is what keeps integration tests headless and future
backend work isolated to renderer code. See [Renderers](Renderers.md).
