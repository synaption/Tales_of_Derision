# Getting Started

## Requirements

- **Python 3.12+** (uses `str | None` syntax in annotations)
- **[esper](https://github.com/benmoran56/esper) 3.x** — the ECS library
- **pygame** — window/input/audio runtime used by the game

## Install

```bash
python3 -m pip install --user esper pygame
```

## Run

```bash
python3 src/main.py
```

Load a specific save and skip title/main menu:

```bash
python3 src/main.py --save_file src/data/saves/my_run.json
```

## Controls

### Gameplay controls

| Action | Keys |
|--------|------|
| Set direction | Hold `W/A/S/D` |
| Take a turn / move / wait | `Space` |
| Interact (faced tile) / menu select | `Enter` |
| Player menu (tabbed) | `Tab` |
| Open inventory | `I` |
| Open status | `C` |
| Sleep / camp | `R` |
| Look mode | (bound in options) |
| Open pause menu | `Esc` |
| Tile scale up/down | `+` / `-` |

Diagonal movement is supported by holding two directions (for example `W` + `D`)
while pressing `Space`. Pressing `Space` with **no** direction held waits a turn in
place. Interaction (`Enter`) targets the tile you're facing — hold a direction toward
a well/tree/stove/corpse/NPC, then press `Enter`. See [Gameplay](Gameplay.md) for the
full survival, building, and social loops.

### Menu controls

| Action | Keys |
|--------|------|
| Move selection | `W/S` |
| Select | `Enter` |
| Cycle tabs (player menu) | `Tab` |
| Back / resume | `Esc` |

The **player menu** (`Tab`) has `Inventory`, `Craft`, and `Status` tabs (`Map`,
`Journal`, `Skills` stubbed). The **pause menu** (`Esc`) has `Save Game`, `Options`,
`Quit`.

Keybinds are **data** under `keybinds` in `src/data/config/options.json` (action name
→ key list) and are fully rebindable. The raw-key mapping lives in
[src/renderer/pygame_renderer.py](../src/renderer/pygame_renderer.py); see
[Renderers](Renderers.md) for backend details.

## Saves and options files

- Default save: `src/data/saves/default_save.json`
- Default options: `src/data/config/default_options.json`
- Working options: `src/data/config/options.json`
- On startup, missing `options.json` is auto-copied from `default_options.json`.

Keybinds live under `keybinds` in `options.json` using action names (for example
`move_up`, `confirm_action`, `open_pause_menu`).

## Verifying headlessly

Because the renderer is decoupled, you can drive the whole game headless with a
fake renderer that records `draw_glyph` calls and feeds actions to
`esper.process(...)`. This is how movement and collision are tested. See
[Architecture](Architecture.md#testing-headless).
