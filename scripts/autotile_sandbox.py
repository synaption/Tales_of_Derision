#!/usr/bin/env python3
"""Quick pygame autotile sandbox for wall/water/ledges adjacency testing.

Controls:
- Left click: place active tile type
- Middle click: erase cell
- Right click on placed tile: open autotile picker for that cell's requested variant
- Arrow keys (picker): move selected tile in the active set sheet
- Enter (picker): assign selected tile to requested mask and save tile_index.csv
- Esc (picker): cancel picker
- 1: select water
- 2: select wall
- 3: select ledges
- C: clear grid
- Esc: quit
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pygame

GRID_W = 8
GRID_H = 8
DIRECTION_ORDER = (1, 2, 3, 4, 5, 6, 7, 8)
DIRECTION_OFFSETS = {
    1: (-1, -1),
    2: (0, -1),
    3: (1, -1),
    4: (1, 0),
    5: (1, 1),
    6: (0, 1),
    7: (-1, 1),
    8: (-1, 0),
}
DIRECTION_MASKS = {
    1: 1,    # NW
    2: 2,    # N
    3: 4,    # NE
    4: 16,   # E
    5: 128,  # SE
    6: 64,   # S
    7: 32,   # SW
    8: 8,    # W
}


@dataclass
class PickerState:
    grid_x: int
    grid_y: int
    base: str
    requested_mask: int
    requested_name: str
    update_mask: int
    update_name: str
    selected_col: int
    selected_row: int


def _in_bounds(x: int, y: int) -> bool:
    return 0 <= x < GRID_W and 0 <= y < GRID_H


def _parse_int(value: object) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _tilesheet_for_base(base_name: str) -> str:
    if base_name == "wall":
        return "autotile_transparent_wall"
    if base_name == "water":
        return "autotile_transparent_water"
    if base_name == "ledges":
        return "autotile_transparent_ledges"
    return base_name


def _base_name_for_tilesheet(sheet_name: str) -> str | None:
    normalized = (sheet_name or "").strip().lower()
    if normalized == "autotile_transparent_wall":
        return "wall"
    if normalized == "autotile_transparent_water":
        return "water"
    if normalized == "autotile_transparent_ledges":
        return "ledges"
    return None


def _mask_from_tile_name(base_name: str, tile_name: str) -> int | None:
    normalized = (tile_name or "").strip().lower()
    prefix = base_name.lower()
    if not normalized.startswith(prefix):
        return None
    suffix = normalized[len(prefix):]
    if not suffix:
        return None
    mask_value = _parse_int(suffix)
    if mask_value is None or not (0 <= mask_value <= 255):
        return None
    return mask_value


def _extract_tile_at(sheet: pygame.Surface, col: int, row: int, tile_size: int) -> pygame.Surface | None:
    cols = sheet.get_width() // tile_size
    rows = sheet.get_height() // tile_size
    if col < 0 or row < 0 or col >= cols or row >= rows:
        return None

    tile = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
    tile.blit(
        sheet,
        (0, 0),
        pygame.Rect(col * tile_size, row * tile_size, tile_size, tile_size),
    )
    return tile


def _grid_cell_from_pos(pos: tuple[int, int], cell_px: int) -> tuple[int, int] | None:
    x, y = pos
    if y < 0 or y >= GRID_H * cell_px:
        return None

    gx = x // cell_px
    gy = y // cell_px
    if not _in_bounds(gx, gy):
        return None
    return gx, gy


def _move_selection(col: int, row: int, dx: int, dy: int, sheet_cols: int, sheet_rows: int) -> tuple[int, int]:
    col = max(0, min(sheet_cols - 1, col + dx))
    row = max(0, min(sheet_rows - 1, row + dy))
    return col, row


def _load_variant_bank(
    tile_index_path: Path,
    sheet_surfaces: Dict[str, pygame.Surface],
    tile_size: int,
) -> tuple[
    Dict[str, Dict[int, pygame.Surface]],
    Dict[str, Dict[int, str]],
    Dict[str, Dict[int, tuple[int, int]]],
]:
    surfaces: Dict[str, Dict[int, pygame.Surface]] = {"water": {}, "wall": {}, "ledges": {}}
    names: Dict[str, Dict[int, str]] = {"water": {}, "wall": {}, "ledges": {}}
    coords: Dict[str, Dict[int, tuple[int, int]]] = {"water": {}, "wall": {}, "ledges": {}}

    with tile_index_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if not isinstance(row, dict):
                continue

            base_name = _base_name_for_tilesheet(str(row.get("tilesheet", "")))
            if base_name is None or base_name not in sheet_surfaces:
                continue

            mask_value = _mask_from_tile_name(base_name, str(row.get("tile_name", "")))
            if mask_value is None:
                continue

            tile_number = _parse_int(row.get("tile_number", ""))
            if tile_number is None or tile_number < 0:
                continue

            sheet = sheet_surfaces[base_name]
            sheet_cols = sheet.get_width() // tile_size
            sheet_rows = sheet.get_height() // tile_size
            if sheet_cols <= 0 or sheet_rows <= 0:
                continue

            col = tile_number % sheet_cols
            row_idx = tile_number // sheet_cols
            if row_idx < 0 or row_idx >= sheet_rows:
                continue

            tile = _extract_tile_at(sheet, col, row_idx, tile_size)
            if tile is None:
                continue

            surfaces[base_name][mask_value] = tile
            names[base_name][mask_value] = f"{base_name}{mask_value}"
            coords[base_name][mask_value] = (col, row_idx)

    return surfaces, names, coords


def _update_tile_index_target(
    tile_index_path: Path,
    base_name: str,
    mask_value: int,
    selected_col: int,
    selected_row: int,
    tile_size: int,
    sheet_surfaces: Dict[str, pygame.Surface],
) -> None:
    sheet = sheet_surfaces.get(base_name)
    if sheet is None:
        raise ValueError(f"Unknown base: {base_name}")

    sheet_cols = sheet.get_width() // tile_size
    sheet_rows = sheet.get_height() // tile_size
    if sheet_cols <= 0 or sheet_rows <= 0:
        raise ValueError(f"Invalid sheet dimensions for base={base_name}")
    if selected_col < 0 or selected_row < 0 or selected_col >= sheet_cols or selected_row >= sheet_rows:
        raise ValueError(
            f"Selected tile out of range for {base_name}: ({selected_col},{selected_row})"
        )

    tile_number = selected_row * sheet_cols + selected_col
    target_sheet = _tilesheet_for_base(base_name)
    target_name = _requested_name(base_name, mask_value)

    rows: list[dict[str, str]] = []
    with tile_index_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if not isinstance(row, dict):
                continue
            rows.append({k: str(v) for k, v in row.items()})

    target_row: dict[str, str] | None = None
    for row in rows:
        row_sheet = str(row.get("tilesheet", "")).strip().lower()
        row_name = str(row.get("tile_name", "")).strip().lower()
        if row_sheet == target_sheet and row_name == target_name:
            target_row = row
            break

    if not fieldnames:
        fieldnames = ["tilesheet", "tile_number", "tile_name"]

    # If the mask row is missing, create one so picker changes are still persisted.
    if target_row is None:
        target_row = {k: "" for k in fieldnames}
        rows.append(target_row)

    target_row["tilesheet"] = target_sheet
    target_row["tile_name"] = target_name
    target_row["tile_number"] = str(tile_number)

    with tile_index_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _neighbor_mask(grid: list[list[str | None]], x: int, y: int) -> int:
    base = grid[y][x]
    if base is None:
        return 0

    connected: Dict[int, bool] = {}
    for direction in DIRECTION_ORDER:
        dx, dy = DIRECTION_OFFSETS[direction]
        nx = x + dx
        ny = y + dy
        connected[direction] = _in_bounds(nx, ny) and grid[ny][nx] == base

    mask_value = 0
    # Cardinals
    if connected.get(2, False):
        mask_value |= DIRECTION_MASKS[2]  # N
    if connected.get(4, False):
        mask_value |= DIRECTION_MASKS[4]  # E
    if connected.get(6, False):
        mask_value |= DIRECTION_MASKS[6]  # S
    if connected.get(8, False):
        mask_value |= DIRECTION_MASKS[8]  # W

    # 47-tile corner rule: diagonals require both adjacent cardinals.
    if connected.get(1, False) and connected.get(2, False) and connected.get(8, False):
        mask_value |= DIRECTION_MASKS[1]  # NW
    if connected.get(3, False) and connected.get(2, False) and connected.get(4, False):
        mask_value |= DIRECTION_MASKS[3]  # NE
    if connected.get(7, False) and connected.get(6, False) and connected.get(8, False):
        mask_value |= DIRECTION_MASKS[7]  # SW
    if connected.get(5, False) and connected.get(6, False) and connected.get(4, False):
        mask_value |= DIRECTION_MASKS[5]  # SE

    return mask_value


def _requested_name(base_name: str, mask_value: int) -> str:
    return f"{base_name}{mask_value}"


def _resolve_variant(
    base_name: str,
    mask_value: int,
    variant_surfaces: Dict[str, Dict[int, pygame.Surface]],
    variant_names: Dict[str, Dict[int, str]],
    variant_coords: Dict[str, Dict[int, tuple[int, int]]],
) -> tuple[pygame.Surface | None, str, tuple[int, int] | None, int | None]:
    bank = variant_surfaces.get(base_name, {})
    names = variant_names.get(base_name, {})
    coords = variant_coords.get(base_name, {})
    requested = _requested_name(base_name, mask_value)

    if mask_value in bank:
        return bank[mask_value], names.get(mask_value, requested), coords.get(mask_value), mask_value

    if 0 in bank:
        return bank[0], names.get(0, _requested_name(base_name, 0)), coords.get(0), 0

    if not bank:
        return None, requested, None, None

    best_key = min(
        bank.keys(),
        key=lambda cand: (
            (mask_value & ~cand).bit_count(),
            (cand & ~mask_value).bit_count(),
            -(mask_value & cand).bit_count(),
            cand,
        ),
    )
    return bank[best_key], names.get(best_key, requested), coords.get(best_key), best_key


def _draw_text(screen: pygame.Surface, font: pygame.font.Font, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    image = font.render(text, True, color)
    screen.blit(image, (x, y))


def main() -> int:
    parser = argparse.ArgumentParser(description="8x8 autotile sandbox for wall/water/ledges morphing.")
    parser.add_argument(
        "--walls-sheet",
        default="gfx/tilesets/Hexany/autotile_transparent_wall.png",
        help="Walls sheet path.",
    )
    parser.add_argument(
        "--water-sheet",
        default="gfx/tilesets/Hexany/autotile_transparent_water.png",
        help="Water sheet path.",
    )
    parser.add_argument(
        "--ledges-sheet",
        default="gfx/tilesets/Hexany/autotile_transparent_ledges.png",
        help="Ledges sheet path.",
    )
    parser.add_argument(
        "--tile-index",
        default="gfx/tilesets/Hexany/tile_index.csv",
        help="Tile index CSV path.",
    )
    parser.add_argument("--tile-size", type=int, default=16, help="Source tile size in pixels.")
    parser.add_argument("--scale", type=int, default=3, help="Screen scale factor for each grid cell.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Auto-exit after N frames (0 keeps running). Useful for smoke tests.",
    )
    args = parser.parse_args()

    walls_sheet_path = Path(args.walls_sheet)
    water_sheet_path = Path(args.water_sheet)
    ledges_sheet_path = Path(args.ledges_sheet)
    tile_index_path = Path(args.tile_index)
    tile_size = max(1, int(args.tile_size))
    scale = max(1, int(args.scale))

    for required_path in (walls_sheet_path, water_sheet_path, ledges_sheet_path, tile_index_path):
        if not required_path.exists():
            print(f"Required file not found: {required_path}")
            return 1

    pygame.init()

    try:
        loaded_wall_sheet = pygame.image.load(str(walls_sheet_path))
        loaded_water_sheet = pygame.image.load(str(water_sheet_path))
        loaded_ledges_sheet = pygame.image.load(str(ledges_sheet_path))
    except Exception as exc:
        print(f"Failed to load sheet: {exc}")
        return 1

    cell_px = tile_size * scale
    hud_height = 128
    panel_pad = 6
    panel_gap = 16
    grid_width = GRID_W * cell_px
    grid_height = GRID_H * cell_px

    panel_sheet_width = max(
        loaded_wall_sheet.get_width(),
        loaded_water_sheet.get_width(),
        loaded_ledges_sheet.get_width(),
    )
    panel_sheet_height = max(
        loaded_wall_sheet.get_height(),
        loaded_water_sheet.get_height(),
        loaded_ledges_sheet.get_height(),
    )

    panel_width = panel_sheet_width + panel_pad * 2

    width = grid_width + panel_gap + panel_width + panel_gap
    height = max(grid_height + hud_height, panel_sheet_height + panel_pad * 2 + 80)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Autotile Sandbox (Wall/Water/Ledges)")

    sheet_surfaces: Dict[str, pygame.Surface] = {
        "wall": loaded_wall_sheet.convert_alpha(),
        "water": loaded_water_sheet.convert_alpha(),
        "ledges": loaded_ledges_sheet.convert_alpha(),
    }

    panel_x = grid_width + panel_gap
    panel_y = 8

    variant_surfaces, variant_names, variant_coords = _load_variant_bank(
        tile_index_path,
        sheet_surfaces,
        tile_size,
    )

    for base_name in ("wall", "water", "ledges"):
        if not variant_surfaces.get(base_name):
            print(f"Warning: no variants loaded for {base_name}.")

    font = pygame.font.Font(None, 22)
    small_font = pygame.font.Font(None, 20)
    clock = pygame.time.Clock()

    grid: list[list[str | None]] = [[None for _ in range(GRID_W)] for _ in range(GRID_H)]
    active = "water"
    frame_count = 0
    running = True
    picker_state: PickerState | None = None
    status_message = ""

    def paint_at_mouse(buttons: tuple[bool, bool, bool]) -> None:
        if picker_state is not None:
            return

        cell = _grid_cell_from_pos(pygame.mouse.get_pos(), cell_px)
        if cell is None:
            return

        gx, gy = cell
        if buttons[0]:
            grid[gy][gx] = active
        if buttons[1]:
            grid[gy][gx] = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if picker_state is not None:
                    picker_sheet = sheet_surfaces[picker_state.base]
                    picker_cols = picker_sheet.get_width() // tile_size
                    picker_rows = picker_sheet.get_height() // tile_size

                    if event.key == pygame.K_ESCAPE:
                        status_message = "Picker canceled."
                        picker_state = None
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        try:
                            _update_tile_index_target(
                                tile_index_path,
                                picker_state.base,
                                picker_state.update_mask,
                                picker_state.selected_col,
                                picker_state.selected_row,
                                tile_size,
                                sheet_surfaces,
                            )
                            variant_surfaces, variant_names, variant_coords = _load_variant_bank(
                                tile_index_path,
                                sheet_surfaces,
                                tile_size,
                            )
                            status_message = (
                                f"Assigned {picker_state.update_name} -> "
                                f"({picker_state.selected_col},{picker_state.selected_row}); "
                                "saved in tile_index.csv"
                            )
                        except Exception as exc:
                            status_message = (
                                "Failed to update tile_index: "
                                f"sheet={_tilesheet_for_base(picker_state.base)} "
                                f"requested_mask={picker_state.requested_mask} "
                                f"update_mask={picker_state.update_mask} "
                                f"error={exc}"
                            )
                        picker_state = None
                    elif event.key == pygame.K_LEFT:
                        picker_state.selected_col, picker_state.selected_row = _move_selection(
                            picker_state.selected_col,
                            picker_state.selected_row,
                            dx=-1,
                            dy=0,
                            sheet_cols=picker_cols,
                            sheet_rows=picker_rows,
                        )
                    elif event.key == pygame.K_RIGHT:
                        picker_state.selected_col, picker_state.selected_row = _move_selection(
                            picker_state.selected_col,
                            picker_state.selected_row,
                            dx=1,
                            dy=0,
                            sheet_cols=picker_cols,
                            sheet_rows=picker_rows,
                        )
                    elif event.key == pygame.K_UP:
                        picker_state.selected_col, picker_state.selected_row = _move_selection(
                            picker_state.selected_col,
                            picker_state.selected_row,
                            dx=0,
                            dy=-1,
                            sheet_cols=picker_cols,
                            sheet_rows=picker_rows,
                        )
                    elif event.key == pygame.K_DOWN:
                        picker_state.selected_col, picker_state.selected_row = _move_selection(
                            picker_state.selected_col,
                            picker_state.selected_row,
                            dx=0,
                            dy=1,
                            sheet_cols=picker_cols,
                            sheet_rows=picker_rows,
                        )
                else:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_1:
                        active = "water"
                    elif event.key == pygame.K_2:
                        active = "wall"
                    elif event.key == pygame.K_3:
                        active = "ledges"
                    elif event.key == pygame.K_c:
                        grid = [[None for _ in range(GRID_W)] for _ in range(GRID_H)]
                    elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                        cell = _grid_cell_from_pos(pygame.mouse.get_pos(), cell_px)
                        if cell is not None:
                            gx, gy = cell
                            grid[gy][gx] = None
            elif event.type == pygame.MOUSEBUTTONDOWN and picker_state is None:
                cell = _grid_cell_from_pos(event.pos, cell_px)
                if cell is None:
                    continue
                gx, gy = cell

                if event.button == 1:
                    grid[gy][gx] = active
                elif event.button == 2:
                    grid[gy][gx] = None
                elif event.button == 3:
                    base = grid[gy][gx]
                    if base not in sheet_surfaces:
                        status_message = "Right-click picker needs a placed wall/water/ledges tile."
                        continue

                    mask_value = _neighbor_mask(grid, gx, gy)
                    requested_name = _requested_name(base, mask_value)
                    _, resolved_name, chosen_coord, resolved_mask = _resolve_variant(
                        base,
                        mask_value,
                        variant_surfaces,
                        variant_names,
                        variant_coords,
                    )

                    if chosen_coord is None:
                        chosen_coord = (0, 0)
                    if resolved_mask is None:
                        resolved_mask = mask_value
                    update_name = _requested_name(base, resolved_mask)

                    picker_state = PickerState(
                        grid_x=gx,
                        grid_y=gy,
                        base=base,
                        requested_mask=mask_value,
                        requested_name=requested_name,
                        update_mask=resolved_mask,
                        update_name=update_name,
                        selected_col=chosen_coord[0],
                        selected_row=chosen_coord[1],
                    )
                    if resolved_mask == mask_value:
                        status_message = (
                            f"Picker open for {requested_name}. "
                            "Use arrow keys, Enter to apply, Esc to cancel."
                        )
                    else:
                        status_message = (
                            f"Picker open: requested {requested_name}, editing existing {resolved_name}. "
                            "Use arrow keys, Enter to apply, Esc to cancel."
                        )

        paint_at_mouse(pygame.mouse.get_pressed(3))

        screen.fill((17, 19, 24))

        hovered_cell: tuple[int, int] | None = _grid_cell_from_pos(pygame.mouse.get_pos(), cell_px)

        hovered_info = ""
        for y in range(GRID_H):
            for x in range(GRID_W):
                cell_rect = pygame.Rect(x * cell_px, y * cell_px, cell_px, cell_px)
                base = grid[y][x]

                requested = ""
                chosen_name = ""

                if base is None:
                    pygame.draw.rect(screen, (28, 31, 38), cell_rect)
                else:
                    mask_value = _neighbor_mask(grid, x, y)
                    requested = _requested_name(base, mask_value)

                    if picker_state is not None and picker_state.grid_x == x and picker_state.grid_y == y:
                        selected_tile = _extract_tile_at(
                            sheet_surfaces[picker_state.base],
                            picker_state.selected_col,
                            picker_state.selected_row,
                            tile_size,
                        )
                        chosen_name = f"{picker_state.base}@({picker_state.selected_col},{picker_state.selected_row})"
                        tile = selected_tile
                    else:
                        tile, chosen_name, _, _ = _resolve_variant(
                            base,
                            mask_value,
                            variant_surfaces,
                            variant_names,
                            variant_coords,
                        )

                    if tile is not None:
                        tile_scaled = pygame.transform.scale(tile, (cell_px, cell_px))
                        screen.blit(tile_scaled, cell_rect.topleft)
                    elif base == "water":
                        pygame.draw.rect(screen, (52, 117, 199), cell_rect)
                    elif base == "ledges":
                        pygame.draw.rect(screen, (93, 140, 112), cell_rect)
                    else:
                        pygame.draw.rect(screen, (110, 82, 56), cell_rect)

                if hovered_cell == (x, y):
                    if base is None:
                        hovered_info = f"Cell ({x},{y}) tile=empty"
                    else:
                        hovered_info = f"Cell ({x},{y}) tile={chosen_name} req={requested}"

                pygame.draw.rect(screen, (55, 60, 72), cell_rect, 1)

        panel_base = picker_state.base if picker_state is not None else active
        panel_sheet = sheet_surfaces[panel_base]

        panel_rect = pygame.Rect(
            panel_x,
            panel_y,
            panel_width,
            panel_sheet.get_height() + panel_pad * 2,
        )
        pygame.draw.rect(screen, (14, 17, 22), panel_rect)
        pygame.draw.rect(screen, (180, 184, 195), panel_rect, 1)

        sheet_pos = (panel_rect.x + panel_pad, panel_rect.y + panel_pad)
        screen.blit(panel_sheet, sheet_pos)

        if picker_state is not None:
            selection_rect = pygame.Rect(
                sheet_pos[0] + picker_state.selected_col * tile_size,
                sheet_pos[1] + picker_state.selected_row * tile_size,
                tile_size,
                tile_size,
            )
            pygame.draw.rect(screen, (255, 255, 255), selection_rect, 2)

            _draw_text(
                screen,
                small_font,
                f"Picker target: {picker_state.requested_name}",
                panel_rect.x,
                panel_rect.bottom + 8,
                (235, 235, 235),
            )
            if picker_state.update_name != picker_state.requested_name:
                _draw_text(
                    screen,
                    small_font,
                    f"Updating row: {picker_state.update_name}",
                    panel_rect.x,
                    panel_rect.bottom + 26,
                    (226, 210, 168),
                )
                selected_line_y = panel_rect.bottom + 44
            else:
                selected_line_y = panel_rect.bottom + 26
            _draw_text(
                screen,
                small_font,
                f"Selected tile: ({picker_state.selected_col},{picker_state.selected_row})",
                panel_rect.x,
                selected_line_y,
                (210, 218, 235),
            )
        else:
            _draw_text(
                screen,
                small_font,
                "Right-click a placed tile to pick from this set sheet.",
                panel_rect.x,
                panel_rect.bottom + 8,
                (160, 165, 180),
            )

        hud_rect = pygame.Rect(0, GRID_H * cell_px, width, hud_height)
        pygame.draw.rect(screen, (11, 13, 18), hud_rect)
        _draw_text(
            screen,
            font,
            "LMB place  MMB erase  RMB picker  |  1 water  2 wall  3 ledges  |  C clear  Esc quit",
            10,
            GRID_H * cell_px + 8,
            (220, 220, 220),
        )
        _draw_text(screen, font, f"Active: {active}  Panel: {panel_base}", 10, GRID_H * cell_px + 32, (190, 215, 255))
        _draw_text(
            screen,
            small_font,
            "Mask bits: NW=1 N=2 NE=4 W=8 E=16 SW=32 S=64 SE=128 (47-tile corner rule)",
            10,
            GRID_H * cell_px + 54,
            (160, 165, 180),
        )
        if hovered_info:
            _draw_text(screen, small_font, hovered_info, 10, GRID_H * cell_px + 76, (210, 210, 210))
        if status_message:
            _draw_text(screen, small_font, status_message, 10, GRID_H * cell_px + 98, (188, 212, 180))

        pygame.display.flip()
        clock.tick(60)

        frame_count += 1
        if args.max_frames > 0 and frame_count >= args.max_frames:
            running = False

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
