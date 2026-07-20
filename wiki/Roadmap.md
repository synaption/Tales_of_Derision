# Roadmap

## Done

- [x] esper ECS scaffolding (3.x module-level API)
- [x] `Position`, `Renderable`, `Player` [components](Components.md)
- [x] [`GameMap`](Game-Map.md): bordered room, `is_walkable`
- [x] [`MovementProcessor`](Systems.md) with wall collision
- [x] [`RenderProcessor`](Systems.md): map + entities + sidebar/log/status line
- [x] [`Renderer` interface](Renderers.md) + pygame `PygameRenderer`
- [x] Menu flows: title, main, pause, options, inventory, dialogue/trade
- [x] Save/load + options bootstrap and scaling toggles
- [x] Headless verification via a fake renderer

## Next (small steps, in order)

1. **Keybinds from options.** Wire `options.json` keybinds into pygame action mapping.
2. **Scene-state refactor.** Replace nested menu loops with explicit game/UI states.
3. **Combat depth.** Add `Health`, damage tuning, and death handling for actors.
4. **Content growth.** Expand NPC/item pools and encounter variety.
5. **World generation.** Upgrade maps from static layouts to generated regions.
6. **Perception and stealth.** Improve FOV/visibility and NPC awareness rules.

## Later / bigger

- Controller support and richer input remapping UX.
- Animated feedback, transitions, and frame-timed effects.
- Economy/faction simulation goals from the project vision.

Keep [the wiki](Home.md) in step with the code as these land.
