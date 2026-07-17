# Getting Started

## Requirements

- **Python 3.12+** (uses `str | None` syntax in annotations)
- **[esper](https://github.com/benmoran56/esper) 3.x** — the ECS library
- **curses** — for the terminal backend (standard library on Linux/macOS; on
  Windows use `windows-curses` or run under WSL)

## Install

```bash
pip install esper
```

## Run

```bash
cd project-alpha/pyRL2
python3 main.py
```

Load a specific save and skip title/main menu:

```bash
python3 main.py --save_file data/saves/my_run.json
```

> ⚠️ curses needs a **real terminal**. It will not run from an IDE output pane or
> a captured/non-TTY process.

## Controls

| Action | Keys |
|--------|------|
| Move up | `w` · `k` · ↑ |
| Move down | `s` · `j` · ↓ |
| Move left | `a` · `h` · ← |
| Move right | `d` · `l` · → |
| Open pause menu | `Esc` |

### Menu controls

| Action | Keys |
|--------|------|
| Move selection | `w/s` · `k/j` · ↑/↓ |
| Select item | `Enter` |
| Close pause menu (resume) | `Esc` |

Pause menu contains `Save Game`, `Options`, and `Quit`.

Key-to-action mapping lives in [`renderer/terminal.py`](../renderer/terminal.py);
see [Renderers](Renderers.md) to change it.

## Saves and options files

- Default save: `data/saves/default_save.json`
- Default options: `data/config/default_options.json`
- Working options: `data/config/options.json`
- On startup, missing `options.json` is auto-copied from `default_options.json`.

## Verifying without a terminal

Because the renderer is decoupled, you can drive the whole game headless with a
fake renderer that records `draw_glyph` calls and feeds actions to
`esper.process(...)`. This is how movement and collision are tested. See
[Architecture](Architecture.md#testing-headless).
