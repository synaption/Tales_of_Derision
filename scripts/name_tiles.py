#!/usr/bin/env python3
"""Interactive tile naming helper.

Shows one tile at a time by writing it to identify.png, then prompts for a name.
Outputs a simple CSV schema: tilesheet,tile_number,tile_name.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import pygame

_FIELDNAMES = ["tilesheet", "tile_number", "tile_name"]
_MASK_TO_GRID = {
    1: (0, 0),
    2: (1, 0),
    4: (2, 0),
    16: (2, 1),
    128: (2, 2),
    64: (1, 2),
    32: (0, 2),
    8: (0, 1),
}
_LEGACY_DIRECTION_MASKS = {
    "1": 1,    # NW
    "2": 2,    # N
    "3": 4,    # NE
    "4": 16,   # E
    "5": 128,  # SE
    "6": 64,   # S
    "7": 32,   # SW
    "8": 8,    # W
}


def _infer_tilesheet_name(sheet_path: Path) -> str:
    name = sheet_path.stem
    suffixes = ("_transparent", "_sheet", "_tileset")
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _load_existing(path: Path) -> Dict[Tuple[str, int], str]:
    entries: Dict[Tuple[str, int], str] = {}
    if not path.exists():
        return entries

    try:
        with path.open("r", encoding="utf-8", newline="") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                tilesheet = str(row.get("tilesheet", "")).strip()
                tile_number_raw = row.get("tile_number")
                tile_name = str(row.get("tile_name", "")).strip()
                if not tilesheet:
                    continue
                try:
                    tile_number = int(str(tile_number_raw).strip())
                except (TypeError, ValueError):
                    continue
                entries[(tilesheet, tile_number)] = tile_name
    except Exception:
        return entries

    return entries


def _write_csv(path: Path, entries: Dict[Tuple[str, int], str]) -> None:
    rows = [
        {
            "tilesheet": tilesheet,
            "tile_number": str(tile_number),
            "tile_name": tile_name,
        }
        for (tilesheet, tile_number), tile_name in sorted(entries.items(), key=lambda item: (item[0][0], item[0][1]))
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _parse_autotile_name(name: str) -> tuple[str, int | None]:
    normalized = (name or "").strip().lower()
    if not normalized:
        return "", None

    if normalized.startswith("wall"):
        base_name = "wall"
        suffix = normalized[4:]
    elif normalized.startswith("water"):
        base_name = "water"
        suffix = normalized[5:]
    else:
        return normalized, None

    if suffix == "":
        return base_name, 0

    # Preferred format: base + decimal 0..255 mask.
    if suffix.isdigit():
        value = int(suffix)
        if 0 <= value <= 255:
            return base_name, value

    # Legacy fallback: base + direction digits 1..8.
    if all(ch in _LEGACY_DIRECTION_MASKS for ch in suffix):
        mask_value = 0
        for ch in suffix:
            mask_value |= _LEGACY_DIRECTION_MASKS[ch]
        return base_name, mask_value

    return base_name, None


def _extract_tile_surface(
    sheet: pygame.Surface,
    tile_number: int,
    cols: int,
    rows: int,
    tile_size: int,
) -> pygame.Surface | None:
    if tile_number < 0 or tile_number >= cols * rows:
        return None

    tile_x = tile_number % cols
    tile_y = tile_number // cols
    tile_surface = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
    src_rect = pygame.Rect(tile_x * tile_size, tile_y * tile_size, tile_size, tile_size)
    tile_surface.blit(sheet, (0, 0), src_rect)
    return tile_surface


def _render_identify_preview(
    sheet: pygame.Surface,
    tile_number: int,
    cols: int,
    rows: int,
    tile_size: int,
    identify_path: Path,
    preview_scale: int,
) -> None:
    display_tile = tile_size * max(1, preview_scale)
    preview = pygame.Surface((display_tile * 3, display_tile * 3), pygame.SRCALPHA)
    preview.fill((10, 10, 10, 255))

    try:
        font = pygame.font.Font(None, max(14, int(display_tile * 0.7)))
    except Exception:
        font = None

    def blit_cell(cell_x: int, cell_y: int, source_tile: pygame.Surface) -> None:
        if display_tile != tile_size:
            source_tile = pygame.transform.scale(source_tile, (display_tile, display_tile))
        preview.blit(source_tile, (cell_x * display_tile, cell_y * display_tile))

    for cell_y in range(3):
        for cell_x in range(3):
            cell_rect = pygame.Rect(cell_x * display_tile, cell_y * display_tile, display_tile, display_tile)
            pygame.draw.rect(preview, (28, 28, 28), cell_rect, 1)

    center_tile = _extract_tile_surface(sheet, tile_number, cols, rows, tile_size)
    if center_tile is not None:
        blit_cell(1, 1, center_tile)

    for mask_value, (cell_x, cell_y) in _MASK_TO_GRID.items():
        cell_rect = pygame.Rect(cell_x * display_tile, cell_y * display_tile, display_tile, display_tile)
        pygame.draw.rect(preview, (20, 20, 20), cell_rect)
        pygame.draw.rect(preview, (45, 45, 45), cell_rect, 1)
        if font is not None:
            label = font.render(str(mask_value), True, (225, 225, 225))
            label_rect = label.get_rect(center=cell_rect.center)
            preview.blit(label, label_rect)

    identify_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(preview, str(identify_path))


def _prompt_name(tile_number: int, row: int, col: int, existing: str, context_base: str) -> tuple[str, bool]:
    print(f"Tile #{tile_number} (row={row}, col={col})")
    if existing:
        print(f"Current name: {existing}")
    if context_base:
        print(f"Direction preview base: {context_base}")

    while True:
        raw = input("Name (Enter=keep, q=quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return existing, True
        if raw == "":
            return existing, False
        return raw, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively name tiles in a sheet.")
    parser.add_argument("--sheet", required=True, help="Path to the source tilesheet image.")
    parser.add_argument(
        "--sheet-name",
        help="Tilesheet name stored in CSV. Defaults to image stem (without common suffixes).",
    )
    parser.add_argument(
        "--output",
        default="gfx/tilesets/Hexany/tile_index.csv",
        help="CSV output path (default: gfx/tilesets/Hexany/tile_index.csv).",
    )
    parser.add_argument(
        "--identify",
        default="identify.png",
        help="Path where current tile preview is written (default: identify.png).",
    )
    parser.add_argument("--tile-size", type=int, default=16, help="Tile size in pixels (default: 16).")
    parser.add_argument(
        "--preview-scale",
        type=int,
        default=2,
        help="Scale factor for identify.png preview cells (default: 2).",
    )
    parser.add_argument(
        "--context-base",
        default="",
        help="Optional naming context shown in prompt output.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting tile number (row-major) for prompting (default: 0).",
    )
    parser.add_argument(
        "--skip-labeled",
        action="store_true",
        help="Skip tiles that already have names in the CSV.",
    )
    args = parser.parse_args()

    sheet_path = Path(args.sheet)
    output_path = Path(args.output)
    identify_path = Path(args.identify)
    tile_size = max(1, int(args.tile_size))
    preview_scale = max(1, int(args.preview_scale))
    context_base_arg = (str(args.context_base) if args.context_base is not None else "").strip().lower()

    if not sheet_path.exists():
        print(f"Sheet not found: {sheet_path}")
        return 1

    tilesheet_name = args.sheet_name.strip() if isinstance(args.sheet_name, str) and args.sheet_name.strip() else _infer_tilesheet_name(sheet_path)

    pygame.init()
    try:
        sheet = pygame.image.load(str(sheet_path))
    except Exception as exc:
        print(f"Failed to load sheet: {exc}")
        return 1

    width, height = sheet.get_size()
    cols = width // tile_size
    rows = height // tile_size
    if cols <= 0 or rows <= 0:
        print("Invalid sheet dimensions for the given tile size.")
        return 1

    total = cols * rows
    start_tile = max(0, min(total - 1, int(args.start)))

    entries = _load_existing(output_path)

    print(f"Tilesheet: {tilesheet_name}")
    print(f"Grid: {cols}x{rows} ({total} tiles)")
    print(f"Preview file: {identify_path}")
    print("Mask bits: NW=1, N=2, NE=4, W=8, E=16, SW=32, S=64, SE=128")
    print("Use names like wall0, wall255, water64, etc.")

    active_base = context_base_arg
    quit_requested = False
    for tile_number in range(start_tile, total):
        key = (tilesheet_name, tile_number)
        existing = entries.get(key, "")
        if existing and args.skip_labeled:
            continue

        tile_x = tile_number % cols
        tile_y = tile_number // cols

        existing_base, _ = _parse_autotile_name(existing)
        context_base = active_base or existing_base
        _render_identify_preview(
            sheet=sheet,
            tile_number=tile_number,
            cols=cols,
            rows=rows,
            tile_size=tile_size,
            identify_path=identify_path,
            preview_scale=preview_scale,
        )

        name, should_quit = _prompt_name(tile_number, tile_y, tile_x, existing, context_base)
        entries[key] = name
        _write_csv(output_path, entries)

        if not context_base_arg:
            entered_base, _ = _parse_autotile_name(name)
            if entered_base:
                active_base = entered_base

        if should_quit:
            quit_requested = True
            break

    if quit_requested:
        print("Stopped by user. Progress saved.")
    else:
        print("Finished labeling all tiles in the sheet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
