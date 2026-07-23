# Renderers

Rendering is isolated behind a small interface so game/systems code stays
backend-agnostic. The shipped runtime is pygame. The interface is
[src/renderer/base.py](../src/renderer/base.py); the implementation is
[src/renderer/pygame_renderer.py](../src/renderer/pygame_renderer.py).

## The `Renderer` interface

An ABC and a context manager (`with PygameRenderer(...) as renderer:` calls `setup()`
on enter, `teardown()` on exit):

| Method | Responsibility |
|--------|----------------|
| `setup()` / `teardown()` | Acquire/release the display |
| `clear()` | Erase the frame before drawing |
| `draw_glyph(x, y, glyph, fg=None, bg=None)` | Draw one character at map cell `(x,y)` with optional colours |
| `draw_glyph_classified(x, y, glyph, classification, fg, bg, force_glyph=False)` | Draw with a semantic class (wall/enemy/…) so a backend can pick a sprite; `force_glyph` forces the literal glyph (status identifiers). Default impl falls back to `draw_glyph` |
| `draw_text(x, y, text)` | Draw a UI string |
| `present()` | Flush the frame |
| `poll_action()` | Return an **abstract action string** (non-blocking variants exist in the loop) |

### Remembered-tile ("fog of war") tone
`base.py` also owns the memory-fade colour math (`memory_color`, `MEMORY_DESATURATE`,
`MEMORY_DIM`) on the neutral seam, so both the backend-agnostic draw logic and the
pygame backend apply the *same* desaturate-then-dim transform to remembered tiles and
sprites — one source of truth. (Desaturate toward each pixel's own luminance and dim
by multiply, so black stays black; a flat grey overlay would wash darks out.)

## Abstract actions

`poll_action()` returns backend-independent strings, never raw key codes — this is
why swapping backends never touches [movement](Systems.md). The vocabulary includes:

```
move_up  move_down  move_left  move_right  (+ diagonals via two held keys)
confirm_action  menu_select  open_menu  open_inventory  open_status
sleep  look  open_pause_menu  tile_scale_up  tile_scale_down
ui_layout_changed  quit   None (unrecognised)
```

Keybinds are **data**: they live under `keybinds` in
`src/data/config/options.json` (action name → key list) and the pygame backend maps
raw keys through them, so controls are fully rebindable without code changes.

## `PygameRenderer`

Implements the interface with pygame and owns the key map, plus:
- Configurable fullscreen and tile/UI scaling; draggable sidebar width.
- **Sprite tiles** via an optional tileset config
  (`gfx/tilesets/pygame_tileset_config.json`), with a CP437 fallback sheet.
- The **cached world-coordinate map surface** (`build_map_surface` /
  `blit_map_region` / `redraw_map_cells`) and the **remembered-layer cache** — the
  performance heart of the renderer. Never transform that whole surface per frame; see
  [Performance](Performance.md).
- Menu-backdrop snapshotting (`capture_backdrop`) so menus don't re-render the scene
  every keypress.

## Adding a backend

1. Create `renderer/raylib.py` with `class RaylibRenderer(Renderer)`.
2. Implement the abstract methods: `setup`/`teardown` open/close the window;
   `draw_glyph` draws `glyph` at `(x*cell, y*cell)`; `poll_action` reads input and
   returns the same action strings.
3. In [src/main.py](../src/main.py) construct your renderer instead of
   `PygameRenderer` — the only line that names a backend. Systems and map code stay
   unchanged.

## Turn loop note

The current [turn loop](Architecture.md#the-turn-loop) is turn-driven and waits for
actionable input (polling on a short timeout so real-time status animations and idle
background simulation still advance). A future animation-heavy backend could shift to
a fixed-timestep frame loop while preserving the action vocabulary and ECS processor
contracts.
