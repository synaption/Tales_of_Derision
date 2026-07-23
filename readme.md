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
python3 -m pip install --user esper pygame pytest

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

---

## Design goals — big picture

- **Thousands-of-years simulations.** Full simulation of every map tile, eventually.
  The world should keep living whether or not you're watching, and do *more*
  simulation when little is happening — prioritising the tiles **nearest** the
  player, not the stalest ones. Entering a new region brings that whole region up to
  date; sleeping brings the **whole world** up to date.
- **Memory over disk.** Move everything into memory at startup (sounds, tiles, state)
  wherever it helps.
- **Mods are first-class.** Anybody should be able to drop in their own files to add
  or change content — creatures, items, effects — either as readable data or as
  Python, the same way the core game does it. See [Content & Mods](wiki/Content-and-Mods.md).
- **Seed-based determinism.** One master seed reproduces the entire world (the
  foundation for reproducible saves and, later, time travel). All simulation
  randomness flows through `rng.world_rng().stream(name)`.
- **Everyone is playable.** All NPCs have the same needs as the player. When the
  player dies they become a random sentient NPC somewhere in the world (the "Roy"
  from Rick and Morty premise).
- **Economy & survival.** Money, banking, farming, hunting, fishing; hunger, thirst,
  tiredness.
- **Morrowind-style leveling.** Level individual skills; skill-ups grant character
  levels and attribute points (STR, DEX, CON, INT, WIS, CHA), plus health/magic.

### Action economy
- A day holds a fixed amount of time; there is a turn order.
- Actions take time based on quickness, movement speed, agility, etc.
- Turn order is by **completion time**: everything is either mid-action or waiting
  for its next turn. Effects are immediate — an attack lands now, the attacker sits
  in an "attack" state for some time units, then waits for its next turn.
- Target: one day of typical play ≈ one real-life hour.
- Animations play in order or concurrently depending on what they are.

### Targets
- Desktop, fully rendered, Windows + Linux. Steam.

---

## Style

Pixel art. HD text. Shader effects and lighting. Basic animations, or none at all.
Characters face the way they move (left/right, the four, or all eight directions
depending on the sprite). Dialogue in a fake gibberish language
(`##!/$*~# GH01^@`), speech bubbles, and Sims-like symbol popups (`++`, `--`).

## Characters
Wizards · Great Fairy · NPCs · mostly Farmers.

## Inspiration
Caves of Qud · Lord of the Rings · DaFluffyPotato · Minecraft · Rimworld · Dwarf
Fortress · Song of Syx · Infectionator World Dominator · Earth Defense Force ·
Chrono Trigger · Zelda.

---

## Themes & brainstorms (idea dump)

**Fantasy.**

**Time travel — time is cyclical.** A hyper-advanced civilization builds floating
islands, destroys the planet, then intelligently redesigns new planets from their
floating society. These floating societies are sparsely populated by a few
super-adept NPCs. These "gods" die and enter the same reincarnation loop as you —
i.e. *you are a reincarnant* who has forgotten it.

**Zombie.**

**MacGuffins / plot coupons** — the one ring, the infinity stones, the dragon balls.

**Knowledge and teaching. Music and comradery. Balance and equilibrium. Karma and
reincarnation.**

**Open questions / TODO thoughts** (see also `next.md`):
- Rename to *Seeds of Derision*?
- Make sure people (NPCs) never get permanently stuck.
- Can `tcod` help — pathfinding especially?
