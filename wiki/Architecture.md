# Architecture

## ECS in one paragraph

Entity-Component-System splits the game into three parts: **entities** are just ids,
**components** are plain data attached to entities, and **systems** (esper calls them
*processors*) hold all the behaviour. The player isn't a `Player` class with position
+ rendering + movement baked in; it's an entity that *has* a `Position`, a
`Renderable`, and a `Player` tag, and separate systems read those components and do
work. **Behaviour composes by adding components, not by subclassing** — a creature is
a carnivore because it has `Diet("carnivore")`, a resident because it has `Resident`,
prey because it has `Deer`. This is what lets content scale: see
[Content & Mods](Content-and-Mods.md).

## esper 3.x model

esper 3.x keeps ECS state in **module-level** globals rather than a `World` object.

| Call | Purpose |
|------|---------|
| `esper.create_entity(*components)` | Make an entity from components |
| `esper.add_processor(proc, priority=0)` | Register a system; higher priority runs first |
| `esper.get_components(A, B)` | Iterate `(entity, (a, b))` for entities having both |
| `esper.component_for_entity(ent, A)` | Fetch one component off one entity |
| `esper.has_component(ent, A)` | Membership test |
| `esper.add_component` / `remove_component` / `delete_entity` | Mutate the database |
| `esper.process(*args)` | Run every processor's `process(*args)` in priority order |

## Layer DAG (imports only point down)

```
L0 data/util    components.py · rng.py · action.py · onymancer.py · config.py · queries.py
L1 map/topology game_map.py · regions.py · schedule.py
L2 content      content/ (registry · kits · effects · items · prefabs · loader)
L3 systems      systems.py (core hub) · render.py (RenderProcessor) · ai.py (creature AI)
L4 logic        interactions.py (renderer-free gameplay logic)
L5 presentation ui.py (screens/menus/widgets) · audio.py
L6 app          main.py (entry point + turn loop) · worldgen.py (world setup)
   persistence.py depends only on components + game_map
```

Nothing in Data, Map, Content, or Systems imports pygame — the arrows only ever point
toward data and the `Renderer` interface. Imports point **down**: `ui` may import
`interactions`, `interactions` may import `systems`, but never the reverse, and
nothing imports `main`. `render.py`/`ai.py` import their helpers from `systems` and
are re-exported through it (import them via `systems`).

## The turn loop

The game is turn-based: it waits for actionable input, runs the systems once, and
repeats. A single call to `esper.process(action)` runs **every** processor in
priority order, so one keypress advances time, AI, needs, and the frame together.

Processor registration and priorities (`main.py`, high runs first):

| Priority | Processor | Role |
|---------:|-----------|------|
| 2 | `TimeProcessor` | Advance the world clock **first** so needs/AI read the right time of day |
| 1 | `MovementProcessor` | Apply the player's move / bump-attack |
| 0 | `HousingProcessor` | Claim/assign homes before the AI acts on them |
| 0 | `NpcAiProcessor` | The per-turn brain for every land NPC |
| 0 | `FishAiProcessor` | The sea's brain (aquatic mirror of the land AI) |
| 0 | `NeedsProcessor` | Advance hunger/thirst/tiredness |
| 0 | `TreeGrowthProcessor` | Daily flora grow/regrow/die |
| 0 | `ReproductionProcessor` | Deliver due pregnancies |
| 0 | `RenderProcessor` | Draw the frame **last** |

Time-advancing systems only tick on a real action (a move or `WAIT_ACTION`); menu
refreshes call `esper.process(None)`, which advances nothing. Move-before-draw (and
Time-before-everything) means a keypress is reflected in the same frame.

The loop also does the work a bare `esper.process` can't: it resolves held-direction
keys into a single action, routes UI actions (menus, look mode, interactions) to
their handlers, and — when the player is idle or only a status animation is playing —
spends the spare time paying down background region-simulation debt
([World Simulation](World-Simulation.md)).

## Data flow of a keypress

1. `PygameRenderer.poll_action()` reads pygame events and emits an abstract action
   string (`move_left`, `menu_select`, `confirm_action`, `sleep`, `look`, …).
2. `main.py` handles UI-state actions (inventory/pause/options/dialogue/look/
   interactions) and routes gameplay actions into `esper.process(...)`.
3. `TimeProcessor` advances the clock; `MovementProcessor` moves the player (or bumps
   an adjacent creature into a melee attack); the AI/needs/flora/reproduction systems
   simulate the world.
4. `RenderProcessor` computes field-of-view, draws visible tiles (+ remembered tiles
   from memory), draws entities, the sidebar/log/status line, and presents.

## Testing headless

Because the renderer is an interface, tests substitute a fake (see
[src/tests/fakes.py](../src/tests/fakes.py)): register it with `RenderProcessor`, call
`esper.process("move_up")`, and assert on the recorded draw calls or the player's
`Position`. No live window required. The whole survival/social/housing/reproduction
sim is driven and asserted this way — 236 tests in ~1s.

## Related pages

[Components](Components.md) · [Systems](Systems.md) · [Game Map](Game-Map.md) ·
[Action Economy](Action-Economy.md) · [World Simulation](World-Simulation.md) ·
[Renderers](Renderers.md)
