# Project Nightshift

A top-down survival-horror roguelike built with Python and Pygame.

Project Nightshift combines procedurally generated environments, resource scarcity, interconnected puzzles, and an energy-based action economy. Its presentation draws inspiration from the low-resolution textures, limited color palettes, visual distortion, and oppressive atmosphere of PlayStation-era horror games, translated into a readable top-down tile-based format.

## Overview

Each run places the player inside a procedurally assembled horror scenario built from rooms, corridors, locked routes, environmental puzzles, enemies, supplies, and hidden areas.

The player must explore carefully, manage limited energy and resources, unlock new routes, solve puzzles, and determine when to fight, flee, rest, or take a dangerous shortcut.

The game is designed around deliberate actions rather than real-time reflexes. Almost every meaningful action consumes energy, allowing the player to plan several moves ahead while enemies respond according to the same underlying turn system.

## Core Features

* Top-down tile-based survival-horror gameplay
* Procedurally generated levels and progression routes
* Energy-based turn and action economy
* Limited ammunition, healing items, inventory space, and safe areas
* Interconnected locks, keys, shortcuts, and environmental puzzles
* Multiple level themes with distinct layouts and hazards
* Persistent enemy and environmental states
* PS1-inspired low-resolution visual style
* Seeded generation for reproducible runs
* Pygame-based rendering and input
* Data-driven enemies, items, rooms, and puzzle definitions

## Energy-Based Action Economy

The game uses energy instead of a traditional one-action-per-turn system.

Every actor has a current energy value and an energy recovery rate. Actions consume different amounts of energy depending on their complexity, speed, and equipment requirements.

Example action costs:

| Action                 | Energy Cost |
| ---------------------- | ----------: |
| Wait                   |          50 |
| Turn in place          |          25 |
| Move one tile          |         100 |
| Open or close a door   |          75 |
| Pick up an item        |          50 |
| Search an object       |         125 |
| Use a light weapon     |         100 |
| Use a heavy weapon     |         150 |
| Reload                 |         125 |
| Use a healing item     |         150 |
| Push a heavy object    |         200 |
| Interact with a puzzle |         100 |
| Sprint one tile        |         150 |

When the player spends energy, time advances until another actor has enough energy to act.

This allows fast enemies to move more frequently, heavy weapons to feel slower, injuries to reduce efficiency, and certain actions to create meaningful periods of vulnerability.

### Example

The player has 200 energy.

They may choose to:

* Move twice
* Move once and fire a light weapon
* Open a door, enter the room, and pick up an item
* Use a slow healing item while nearby enemies gain several opportunities to move

The energy system should remain visible enough for planning without exposing unnecessary simulation detail.

## Visual Direction

The game uses a top-down tile set inspired by the visual limitations and atmosphere of PlayStation-era horror games.

The intended style includes:

* Low-resolution textures
* Restricted color palettes
* Strong shadows and limited lighting
* Dithered gradients
* Texture warping and subtle affine distortion
* Pixelated environmental details
* Low-frame-count environmental animations
* Deliberate visual noise
* Fog, darkness, rain, steam, and flickering lights
* Chunky interface elements and bitmap typography

Although the camera is top-down, tiles should suggest three-dimensional environments through perspective, shadow, wall height, props, and layered foreground elements.

Recommended internal tile sizes:

* 16×16 for highly abstract graphics
* 24×24 for compact environments
* 32×32 for detailed rooms and readable objects

The game may render to a low-resolution internal surface and scale it to the display using nearest-neighbor scaling.

## Level Types

Each level type defines its own room library, hazards, enemies, puzzles, visual palette, ambient sounds, and generation rules.

### Mansion

A decaying residence built around hallways, bedrooms, studies, dining rooms, galleries, servant passages, and hidden chambers.

Typical features:

* Central entrance hall
* Locked wings
* Portrait and statue puzzles
* Fireplaces and hidden switches
* Libraries and private studies
* Ornamental keys
* Narrow servant corridors
* Outdoor gardens or courtyards

### Castle

A fortified structure containing towers, battlements, prisons, chapels, crypts, armories, and ceremonial halls.

Typical features:

* Vertical layouts
* Portcullises and drawbridge mechanisms
* Heraldic puzzles
* Torch and brazier interactions
* Traps and collapsing floors
* Secret passages
* Catacombs beneath the main structure
* Long sightlines in halls and courtyards

### Industrial

A factory, refinery, power station, warehouse complex, or processing facility.

Typical features:

* Conveyor belts
* Machinery hazards
* Steam and pressure systems
* Power-routing puzzles
* Keycards and maintenance tools
* Loading bays
* Control rooms
* Catwalks and service tunnels

### City

An abandoned or quarantined urban district assembled from streets, apartments, shops, offices, clinics, alleys, and transit spaces.

Typical features:

* Multiple building entrances
* Outdoor and indoor transitions
* Barricaded streets
* Rooftop shortcuts
* Civilian apartments
* Police or emergency facilities
* Vehicle obstacles
* Dynamic visibility caused by weather and lighting

### Sewer

A network of tunnels, drainage channels, maintenance rooms, pumping stations, and flooded passages.

Typical features:

* Water-level controls
* One-way drops
* Narrow walkways
* Toxic or infected water
* Valve puzzles
* Grates and maintenance gates
* Poor visibility
* Ambush-focused enemy placement

### Secret Underground Lab

A concealed research facility with laboratories, observation rooms, containment zones, security checkpoints, medical areas, and power systems.

Typical features:

* Keycard clearance levels
* Backup generators
* Security terminals
* Containment breaches
* Decontamination rooms
* Experimental enemies
* Research notes
* Elevator and ventilation routes

### Rural

An isolated countryside area containing farmhouses, barns, fields, forests, cabins, dirt roads, wells, caves, and abandoned utility buildings.

Typical features:

* Open outdoor spaces
* Limited visibility in crops or trees
* Long travel distances
* Sparse safe areas
* Hunting equipment
* Agricultural machinery
* Natural cave systems
* Weather-driven hazards

## Procedural Generation

Levels are generated from an abstract progression graph before tiles are placed.

The generation pipeline should follow this order:

1. Select a level type.
2. Generate a critical progression path.
3. Add optional branches and resource rooms.
4. Assign locks, keys, puzzles, and clue locations.
5. Add shortcuts and return routes.
6. Convert the progression graph into rooms and corridors.
7. Place enemies, supplies, hazards, and environmental storytelling.
8. Validate that the level can be completed.
9. Reject or repair invalid seeds.
10. Render the final tile map.

A generated level should include:

* One recognizable entrance
* At least one major landmark
* One or more safe or low-threat areas
* Multiple locked routes
* At least one meaningful shortcut
* Optional high-risk resource areas
* A final objective or exit condition
* A valid path through all mandatory dependencies

## Puzzles

Puzzles should be generated from reusable templates rather than fully random answers.

Supported puzzle families may include:

* Symbol ordering
* Numeric codes
* Item combination
* Power routing
* Valve and pressure systems
* Statue positioning
* Light and shadow alignment
* Weight balancing
* Environmental sequence puzzles
* Multi-room state puzzles

Each mandatory puzzle should have sufficient clues placed somewhere reachable before the puzzle must be solved.

Puzzle data should define:

```python
{
    "id": "generator_room_power",
    "type": "routing",
    "difficulty": 2,
    "required_items": ["replacement_fuse"],
    "required_clues": ["maintenance_diagram"],
    "reward_flag": "east_wing_powered"
}
```

## Combat and Avoidance

Combat is intentionally expensive.

Enemies should function as persistent route-management problems rather than disposable obstacles. The player may kill, stun, distract, evade, trap, or temporarily disable enemies depending on available equipment.

Weapons should differ through:

* Energy cost
* Damage
* Accuracy
* Range
* Noise
* Ammunition rarity
* Reload time
* Stopping power
* Inventory size

Noise may attract enemies from nearby rooms or increase local threat levels.

## Inventory

The inventory uses a limited slot system.

Items may occupy one or more slots depending on size.

Item categories include:

* Weapons
* Ammunition
* Healing supplies
* Keys
* Puzzle objects
* Tools
* Documents
* Defensive items
* Light sources

The player may need to leave useful items behind, return to storage, or choose between carrying combat supplies and progression items.

## Safe Rooms

Safe rooms provide temporary relief from exploration pressure.

Possible safe-room features:

* Item storage
* Save functionality
* Limited healing
* Map review
* Crafting or item inspection
* Persistent music cue
* No standard enemy spawns

Safe rooms should not be completely predictable in every run, but their placement must support fair progression.

## Controls

Default controls:

| Input             | Action                  |
| ----------------- | ----------------------- |
| Arrow Keys / WASD | Move or navigate menus  |
| Space             | Wait                    |
| E                 | Interact                |
| F                 | Attack                  |
| R                 | Reload                  |
| I / Tab           | Open inventory          |
| M                 | Open map                |
| Q                 | Quick-use equipped item |
| Escape            | Pause or close menu     |
| F1                | Toggle debug overlay    |

Controls should be configurable through a settings file.

## Project Structure

```text
project-nightshift/
├── main.py
├── requirements.txt
├── README.md
├── assets/
│   ├── audio/
│   ├── fonts/
│   ├── sprites/
│   ├── tilesets/
│   └── ui/
├── data/
│   ├── enemies/
│   ├── items/
│   ├── level_types/
│   ├── puzzles/
│   ├── rooms/
│   └── weapons/
├── game/
│   ├── actions/
│   │   ├── base_action.py
│   │   ├── combat_actions.py
│   │   ├── interaction_actions.py
│   │   └── movement_actions.py
│   ├── actors/
│   │   ├── actor.py
│   │   ├── enemy.py
│   │   └── player.py
│   ├── generation/
│   │   ├── level_generator.py
│   │   ├── progression_graph.py
│   │   ├── puzzle_generator.py
│   │   ├── room_placer.py
│   │   └── validator.py
│   ├── systems/
│   │   ├── ai_system.py
│   │   ├── combat_system.py
│   │   ├── energy_system.py
│   │   ├── inventory_system.py
│   │   ├── lighting_system.py
│   │   └── save_system.py
│   ├── ui/
│   │   ├── hud.py
│   │   ├── inventory_menu.py
│   │   └── map_screen.py
│   ├── world/
│   │   ├── entity.py
│   │   ├── level.py
│   │   ├── room.py
│   │   └── tile.py
│   ├── camera.py
│   ├── config.py
│   └── game.py
└── tests/
    ├── test_energy_system.py
    ├── test_generation.py
    └── test_progression_validation.py
```

## Requirements

* Python 3.11 or newer
* Pygame 2.5 or newer

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Example `requirements.txt`:

```text
pygame>=2.5.0
```

## Running the Game

From the project directory:

```bash
python main.py
```

To start with a specific generation seed:

```bash
python main.py --seed 18421
```

To enable generation debugging:

```bash
python main.py --seed 18421 --debug-generation
```

## Minimal Energy System Example

```python
from dataclasses import dataclass


@dataclass
class Actor:
    name: str
    energy: int = 0
    energy_gain: int = 100

    def gain_energy(self) -> None:
        self.energy += self.energy_gain

    def can_afford(self, cost: int) -> bool:
        return self.energy >= cost

    def spend_energy(self, cost: int) -> None:
        if not self.can_afford(cost):
            raise ValueError(f"{self.name} cannot afford action cost {cost}")

        self.energy -= cost
```

A scheduler may repeatedly grant energy until at least one actor can perform an action:

```python
def advance_time(actors: list[Actor], minimum_cost: int = 100) -> None:
    while not any(actor.can_afford(minimum_cost) for actor in actors):
        for actor in actors:
            actor.gain_energy()
```

The final implementation should allow actors to have different speeds, status effects, and action costs.

## Development Priorities

### Phase 1: Playable Core

* Tile map rendering
* Grid movement
* Energy scheduler
* Basic enemy turns
* Collision
* Doors and interactions
* Inventory
* One weapon
* One healing item
* One small hand-authored level

### Phase 2: Procedural Levels

* Room graph generation
* Room templates
* Corridor placement
* Locks and keys
* Progression validation
* Seed support
* Mansion level type

### Phase 3: Survival Systems

* Ammunition economy
* Persistent enemies
* Safe rooms
* Storage
* Status effects
* Noise
* Lighting and visibility

### Phase 4: Puzzle Generation

* Puzzle templates
* Clue placement
* Multi-room dependencies
* Item combination
* Puzzle validation

### Phase 5: Additional Environments

* Castle
* Industrial
* City
* Sewer
* Secret underground lab
* Rural

### Phase 6: Presentation

* PS1-style rendering effects
* Audio layering
* Screen transitions
* Animated environmental tiles
* UI polish
* Save and run history

## Design Principles

1. Exploration should feel dangerous but understandable.
2. Every required objective must be logically discoverable.
3. Resources should create decisions, not unavoidable failure.
4. Backtracking should reveal changes, threats, or shortcuts.
5. Enemies should affect route planning.
6. Procedural generation should preserve authored pacing.
7. Visual limitations should support atmosphere and readability.
8. Randomness should vary situations without removing player agency.

## Debugging Tools

The development build should include overlays for:

* Actor energy
* Enemy awareness
* Room boundaries
* Progression graph nodes
* Locked connections
* Key and puzzle locations
* Critical path
* Reachable tiles
* Lighting values
* Noise propagation
* Generation seed
* Validation failures

A generated run should always be reproducible from its seed.

## Current Status

This project is in early development.

Initial work should focus on proving the following systems:

* Energy-based movement and combat
* Reliable progression generation
* Puzzle and clue placement
* Survival resource balancing
* A readable top-down PS1-inspired visual style

## License

Choose a license before distributing the project.

Suggested options:

* MIT for an open and permissive project
* GPL-3.0 for a project whose derivatives must remain open
* All Rights Reserved for a closed commercial project

## Acknowledgements

Project Nightshift is inspired by classic survival-horror level design, tile-based roguelikes, low-resolution 3D aesthetics, and procedural dungeon-generation techniques.

It is not affiliated with or endorsed by any existing game studio or franchise.
