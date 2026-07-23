# Roadmap

## Done

- [x] esper ECS (3.x); `Renderer` interface + pygame backend; headless fake-renderer tests
- [x] Time-based **action economy** + deterministic completion-time scheduler
- [x] **Region scheduler**: background world simulation, nearest-first catch-up, idle pumping
- [x] Seed-deterministic worldgen; ocean/island map; A* + flow-field pathfinding; connected regions
- [x] Survival (hunger/thirst/tiredness); day/night + 112-day calendar; sleep/camp
- [x] NPC AI: forage/graze/hunt/cook, drink, sleep, build, socialize, chase
- [x] Houses: enclosed-room detection, ownership, residents, shared blueprint→haul→raise construction
- [x] Social sim: personality traits, friendships, gibberish speech bubbles
- [x] Family: marriage, mating cooldowns, pregnancy, birth; procedural names (onymancer)
- [x] Flora ecosystem: trees/bushes/saplings/seaweed grow, regrow, and die
- [x] Fog-of-war tile memory; section camera; status-identifier animation
- [x] Combat bump-attack + `slay_entity` corpses/loot; melee/death SFX
- [x] Save/load (map size + player + seed) + options bootstrap; rebindable keybinds

## In progress — the major refactor

Tracked in the plan; each phase keeps the test suite green.

1. [x] **Docs truth-up** — this wiki + the README now reflect reality.
2. [x] **Content/mod foundation** — the hybrid prefab/kit/effect/item registry
   ([Content & Mods](Content-and-Mods.md)); hand-assembled spawns/items/`OnFire`
   migrated onto it.
3. [~] **Split `main.py`** — `audio.py` (music/SFX), `worldgen.py` (setup + spawns),
   and `queries.py` (shared ECS queries) extracted; the `ui/` package and
   `interactions.py` split still to come.
4. [ ] **Split `systems.py`** → a `systems/` package (one module per subsystem) +
   `render/`.
5. [ ] **Perf/scaling passes** — entity spatial index, verified region catch-up on
   entry/sleep, a cProfile stress harness.

## Next — gameplay depth

- **HP / damage / death model** — replace "maxed need → warning" and bump-kill with
  real health, damage tuning, and death handling (the player becoming a random NPC on
  death — the "Roy" premise).
- **Status effects with teeth** — fire that actually burns, poison, buffs/debuffs — via
  the effect registry.
- **Content variety** — many more enemies, NPCs, items, and effects, authored as
  prefabs/data (the payoff of Phase 2).
- **Skills & leveling** — Morrowind-style skill-ups → character levels → attributes.
- **Economy & factions** — money, banking, trade, farming, fishing at scale.
- **Perception & stealth** — richer FOV/awareness rules.

## Later / bigger

- Controller support and richer input-remapping UX.
- Animated feedback, transitions, frame-timed effects; shaders/lighting.
- Persisting full world/sim state across save/load (today only size+player+seed).
- Time-travel / reincarnation systems built on seed determinism.

Keep [the wiki](Home.md) in step with the code as these land.
