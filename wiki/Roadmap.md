# Roadmap

## Done

- [x] esper ECS scaffolding (3.x module-level API)
- [x] `Position`, `Renderable`, `Player` [components](Components.md)
- [x] [`GameMap`](Game-Map.md): bordered room, `is_walkable`
- [x] [`MovementProcessor`](Systems.md) with wall collision
- [x] [`RenderProcessor`](Systems.md): map + entities + status line
- [x] [`Renderer` interface](Renderers.md) + curses `TerminalRenderer`
- [x] Movement on arrows / hjkl / wasd, quit on q/Esc
- [x] Headless verification via a fake renderer

## Next (small steps, in order)

1. **Tile as data.** Turn map tiles into a `Tile` dataclass
   (`walkable`, `transparent`, colours). See [Game Map](Game-Map.md#extending-later).
2. **A second entity.** Add an enemy `@`-style glyph with `Position` + `Renderable`
   (no AI yet) to confirm multi-entity rendering.
3. **Blocking & bump.** Entities that block movement; walking into one is a "bump"
   (attack later).
4. **Turn/action queue.** Replace the direct `esper.process(action)` with an
   action object model so enemies take turns after the player.
5. **Enemy AI.** A simple `AIProcessor` that steps enemies toward the player.
6. **Combat + Health.** `Health` component, damage on bump, death removes the
   entity.
7. **Dungeon generation.** Rooms + corridors producing the `GameMap.tiles` grid.
8. **Field of view.** Visibility pass using tile `transparent`.

## Later / bigger

- Graphical backend (raylib or pygame) implementing [`Renderer`](Renderers.md);
  possibly a real-time `main.py` loop.
- Items, inventory, and a message log.
- Save/load.

Keep [the wiki](Home.md) in step with the code as these land.
