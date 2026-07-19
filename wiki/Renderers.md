# Renderers

Rendering is isolated behind a small interface so game/systems code stays
backend-agnostic. The shipped runtime is pygame.

## The `Renderer` interface

Defined in [src/renderer/base.py](../src/renderer/base.py) as an ABC:

| Method | Responsibility |
|--------|----------------|
| `setup()` / `teardown()` | Acquire/release the display (called by the `with` block) |
| `clear()` | Erase the frame before drawing |
| `draw_glyph(x, y, glyph)` | Draw one character at map cell `(x, y)` |
| `draw_text(x, y, text)` | Draw a UI string (status line, messages) |
| `present()` | Flush the frame to screen |
| `poll_action()` | Block for input, return an **abstract action string** |

It is a context manager: `with PygameRenderer(...) as renderer:` calls `setup()`
on enter and `teardown()` on exit.

## Abstract actions

`poll_action()` returns backend-independent strings, never raw key codes:

```
"move_up"  "move_down"  "move_left"  "move_right"
"move_up_left" "move_down_right"
"menu_select" "confirm_action" "open_inventory" "open_pause_menu" "quit" None
```

`None` means "key not recognised". This is why swapping backends doesn't touch
[movement](Systems.md) — every backend speaks the same action vocabulary.

## `PygameRenderer`

[src/renderer/pygame_renderer.py](../src/renderer/pygame_renderer.py) implements
the interface with pygame and owns the key map:

| Raw key | Action |
|---------|--------|
| `W` / `A` / `S` / `D` | Direction actions |
| `Space` | `confirm_action` |
| `Enter` | `menu_select` |
| `I` | `open_inventory` |
| `Esc` | `open_pause_menu` |
| Window close | `quit` |

Notes:
- Supports configurable fullscreen and tile/UI scaling.
- Supports sprite tiles via optional tileset config.
- Supports draggable sidebar width and panel-style UI drawing helpers.

## Adding a backend (future)

1. Create `renderer/raylib.py` with `class RaylibRenderer(Renderer)`.
2. Implement the seven methods:
   - `setup`/`teardown` → open/close the window.
   - `draw_glyph(x, y, g)` → draw `g` at `(x*cell, y*cell)` in pixels; pick a
     monospace font.
  - `poll_action()` → read input and return the same action strings.
3. In [src/main.py](../src/main.py), construct your renderer instead of
  `PygameRenderer`. Systems and map code stay unchanged.

```python
# main.py — the only line that names a backend
with RaylibRenderer() as renderer:
    esper.add_processor(RenderProcessor(renderer, game_map))
    ...
```

## Turn loop note

The current [turn loop](Architecture.md#the-turn-loop) is turn-driven and waits
for actionable input. A future animation-heavy backend could shift to a fixed
timestep frame loop while preserving the action vocabulary and ECS processor
contracts.
