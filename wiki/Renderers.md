# Renderers

This is the seam that makes "later render with tcod / pygame / raylib / OpenGL"
cheap. Everything the game needs from a display backend is captured by one
interface; the game never imports a specific backend.

## The `Renderer` interface

Defined in [`renderer/base.py`](../renderer/base.py) as an ABC:

| Method | Responsibility |
|--------|----------------|
| `setup()` / `teardown()` | Acquire/release the display (called by the `with` block) |
| `clear()` | Erase the frame before drawing |
| `draw_glyph(x, y, glyph)` | Draw one character at map cell `(x, y)` |
| `draw_text(x, y, text)` | Draw a UI string (status line, messages) |
| `present()` | Flush the frame to screen |
| `poll_action()` | Block for input, return an **abstract action string** |

It's a context manager: `with TerminalRenderer() as renderer:` calls `setup()` on
enter and `teardown()` on exit, so the terminal is always restored even on error.

## Abstract actions

`poll_action()` returns backend-independent strings, never raw key codes:

```
"move_up"  "move_down"  "move_left"  "move_right"
"menu_select"  "open_pause_menu"  None
```

`None` means "key not recognised". This is why swapping backends doesn't touch
[movement](Systems.md) — every backend speaks the same action vocabulary.

## `TerminalRenderer` (curses)

[`renderer/terminal.py`](../renderer/terminal.py) implements the interface with
curses and owns the key map:

| Raw key | Action |
|---------|--------|
| ↑ / `k` / `w` | `move_up` |
| ↓ / `j` / `s` | `move_down` |
| ← / `h` / `a` | `move_left` |
| → / `l` / `d` | `move_right` |
| `Enter` | `menu_select` |
| `Esc` (27) | `open_pause_menu` |

Notes:
- `setup()` sets `curses.set_escdelay(25)` so Escape opens menus quickly.
- `setup()` calls `initscr`, `noecho`, `cbreak`, hides the cursor, enables
  `keypad` (so arrow keys arrive as `KEY_UP` etc.).
- `draw_glyph`/`draw_text` swallow the `curses.error` raised when writing the
  bottom-right cell.
- `teardown()` reverses all of it and calls `endwin()`.

## Adding a backend (e.g. raylib)

1. Create `renderer/raylib.py` with `class RaylibRenderer(Renderer)`.
2. Implement the seven methods:
   - `setup`/`teardown` → open/close the window.
   - `draw_glyph(x, y, g)` → draw `g` at `(x*cell, y*cell)` in pixels; pick a
     monospace font.
   - `poll_action()` → read keys and return the same action strings. If your
     backend is frame-driven rather than blocking, either block until a key is
     down or restructure `main.py` into a real-time loop (see below).
3. In [`main.py`](../main.py), construct your renderer instead of
   `TerminalRenderer`. Nothing else changes.

```python
# main.py — the only line that names a backend
with RaylibRenderer() as renderer:
    esper.add_processor(RenderProcessor(renderer, game_map))
    ...
```

## Terminal vs. real-time backends

The current [turn loop](Architecture.md#the-turn-loop) *blocks* on
`poll_action()` — perfect for a terminal roguelike. A graphical backend that
wants continuous animation would instead run a fixed-timestep loop that polls
input non-blocking each frame and calls `esper.process(...)` every tick. That's a
`main.py` change; the `Renderer` interface and all systems stay the same.
