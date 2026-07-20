#!/usr/bin/env python3
"""Audit autotile wall/water variants and report missing entries.

This script reads the simplified tile index CSV and analyzes autotile entries
using 8-bit neighbor-mask names:

- wall<mask>
- water<mask>

where <mask> is decimal 0..255 with bits:
NW=1, N=2, NE=4, W=8, E=16, SW=32, S=64, SE=128.

Primary report:
- wall variants missing where water counterpart exists
- water variants missing where wall counterpart exists

Secondary report:
- malformed wall/water names (typos, invalid masks, legacy directional suffixes)
- duplicate canonical variants
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

BASES = ("wall", "water")
LEGACY_DIRECTION_MASKS = {
    "1": 1,    # NW
    "2": 2,    # N
    "3": 4,    # NE
    "4": 16,   # E
    "5": 128,  # SE
    "6": 64,   # S
    "7": 32,   # SW
    "8": 8,    # W
}


@dataclass(frozen=True)
class ParsedVariant:
    line_no: int
    tile_number: int | None
    raw_name: str
    base: str
    suffix: str
    mask: int


@dataclass(frozen=True)
class NameIssue:
    line_no: int
    tile_number: int | None
    raw_name: str
    reason: str
    suggestion: str | None = None


def _canonical_name(base: str, mask: int) -> str:
    return f"{base}{mask}"


def _legacy_suffix_to_mask(suffix: str) -> int | None:
    if not suffix:
        return None
    if not all(char in LEGACY_DIRECTION_MASKS for char in suffix):
        return None

    value = 0
    for char in suffix:
        value |= LEGACY_DIRECTION_MASKS[char]
    return value


def _parse_wall_or_water_name(name: str, line_no: int, tile_number: int | None) -> tuple[ParsedVariant | None, list[NameIssue]]:
    issues: list[NameIssue] = []
    normalized = (name or "").strip().lower()
    for base in BASES:
        if not normalized.startswith(base):
            continue

        suffix = normalized[len(base) :]
        if suffix == "":
            parsed = ParsedVariant(
                line_no=line_no,
                tile_number=tile_number,
                raw_name=name,
                base=base,
                suffix=suffix,
                mask=0,
            )
            issues.append(
                NameIssue(
                    line_no=line_no,
                    tile_number=tile_number,
                    raw_name=name,
                    reason="missing numeric mask suffix",
                    suggestion=_canonical_name(base, 0),
                )
            )
            return parsed, issues

        if suffix.isdigit():
            value = int(suffix)
            if 0 <= value <= 255:
                parsed = ParsedVariant(
                    line_no=line_no,
                    tile_number=tile_number,
                    raw_name=name,
                    base=base,
                    suffix=suffix,
                    mask=value,
                )
                canonical = _canonical_name(base, value)
                if normalized != canonical:
                    issues.append(
                        NameIssue(
                            line_no=line_no,
                            tile_number=tile_number,
                            raw_name=name,
                            reason="non-canonical numeric mask formatting",
                            suggestion=canonical,
                        )
                    )
                return parsed, issues

            issues.append(
                NameIssue(
                    line_no=line_no,
                    tile_number=tile_number,
                    raw_name=name,
                    reason="mask out of range (expected 0..255)",
                )
            )
            return None, issues

        legacy_mask = _legacy_suffix_to_mask(suffix)
        if legacy_mask is not None:
            parsed = ParsedVariant(
                line_no=line_no,
                tile_number=tile_number,
                raw_name=name,
                base=base,
                suffix=suffix,
                mask=legacy_mask,
            )
            issues.append(
                NameIssue(
                    line_no=line_no,
                    tile_number=tile_number,
                    raw_name=name,
                    reason="legacy directional suffix format",
                    suggestion=_canonical_name(base, legacy_mask),
                )
            )
            return parsed, issues

        issues.append(
            NameIssue(
                line_no=line_no,
                tile_number=tile_number,
                raw_name=name,
                reason="invalid suffix format",
            )
        )
        return None, issues

    return None, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Find missing wall/water autotile variants.")
    parser.add_argument(
        "--index",
        default="gfx/tilesets/Hexany/tile_index.csv",
        help="Path to tile index CSV (default: gfx/tilesets/Hexany/tile_index.csv)",
    )
    parser.add_argument(
        "--sheet",
        default="autotile",
        help="Tilesheet name to inspect (default: autotile)",
    )
    parser.add_argument(
        "--full-combos",
        action="store_true",
        help="Also report missing variants against all 256 direction combinations.",
    )
    args = parser.parse_args()

    index_path = Path(args.index)
    sheet_name = args.sheet.strip().lower()

    if not index_path.exists():
        print(f"Index not found: {index_path}")
        return 1

    variants_by_base: Dict[str, Dict[int, list[ParsedVariant]]] = {base: {} for base in BASES}
    issues: list[NameIssue] = []
    autotile_rows = 0

    with index_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for line_no, row in enumerate(reader, start=2):
            if not isinstance(row, dict):
                continue

            row_sheet = str(row.get("tilesheet", "")).strip().lower()
            if row_sheet != sheet_name:
                continue

            autotile_rows += 1
            raw_name = str(row.get("tile_name", "")).strip()
            tile_number_raw = row.get("tile_number")

            tile_number: int | None = None
            try:
                tile_number = int(str(tile_number_raw).strip())
            except (TypeError, ValueError):
                tile_number = None

            normalized = raw_name.lower()
            if not normalized or normalized == "blank":
                continue

            parsed, parse_issues = _parse_wall_or_water_name(raw_name, line_no, tile_number)
            issues.extend(parse_issues)

            if parsed is None:
                if normalized.startswith("wal") or normalized.startswith("wate"):
                    suggestion = None
                    if normalized.startswith("wal") and not normalized.startswith("wall"):
                        suggestion = "wall" + normalized[3:]
                    elif normalized.startswith("wate") and not normalized.startswith("water"):
                        suggestion = "water" + normalized[4:]
                    issues.append(
                        NameIssue(
                            line_no=line_no,
                            tile_number=tile_number,
                            raw_name=raw_name,
                            reason="name does not match wall*/water* convention",
                            suggestion=suggestion,
                        )
                    )
                continue

            variants_by_base[parsed.base].setdefault(parsed.mask, []).append(parsed)

    wall_masks = set(variants_by_base["wall"].keys())
    water_masks = set(variants_by_base["water"].keys())

    missing_wall = sorted(water_masks - wall_masks)
    missing_water = sorted(wall_masks - water_masks)

    duplicate_lines: list[str] = []
    for base in BASES:
        for mask, rows in variants_by_base[base].items():
            if len(rows) <= 1:
                continue
            canonical = _canonical_name(base, mask)
            where = ", ".join(
                f"line {entry.line_no} (tile {entry.tile_number if entry.tile_number is not None else '?'})"
                for entry in rows
            )
            duplicate_lines.append(f"- {canonical}: {where}")

    print(f"Autotile audit for sheet={sheet_name!r}")
    print(f"Rows scanned: {autotile_rows}")
    print(f"Unique wall variants: {len(wall_masks)}")
    print(f"Unique water variants: {len(water_masks)}")
    print()

    print("Missing wall variants (water counterpart exists):")
    if missing_wall:
        for mask in missing_wall:
            print(f"- {_canonical_name('wall', mask)}")
    else:
        print("- none")
    print()

    print("Missing water variants (wall counterpart exists):")
    if missing_water:
        for mask in missing_water:
            print(f"- {_canonical_name('water', mask)}")
    else:
        print("- none")
    print()

    print("Potential naming issues:")
    if issues:
        for issue in issues:
            location = f"line {issue.line_no}"
            if issue.tile_number is not None:
                location += f", tile {issue.tile_number}"
            message = f"- {location}: {issue.raw_name!r} -> {issue.reason}"
            if issue.suggestion:
                message += f" (suggested: {issue.suggestion})"
            print(message)
    else:
        print("- none")
    print()

    print("Duplicate canonical variants:")
    if duplicate_lines:
        for line in duplicate_lines:
            print(line)
    else:
        print("- none")

    if args.full_combos:
        all_masks = set(range(256))
        missing_wall_full = sorted(all_masks - wall_masks)
        missing_water_full = sorted(all_masks - water_masks)
        print()
        print("Full 256-combo coverage:")
        print(f"- Missing wall combos: {len(missing_wall_full)}")
        print(f"- Missing water combos: {len(missing_water_full)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
