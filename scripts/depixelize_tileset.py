#!/usr/bin/env python3
"""Upscale a tilesheet with the Kopf-Lischinski depixelizing algorithm.

Splits a sheet into its tiles, vectorises each one on its own, and reassembles
the results into a larger sheet. Tiles are traced separately on purpose: the
algorithm works on a connectivity graph over neighbouring pixels, and running
it across a whole sheet would happily join one glyph to the next.

The algorithm itself is not implemented here -- it comes from the `depixelizer`
package, which follows Kopf and Lischinski's "Depixelizing Pixel Art"
(SIGGRAPH 2011): a similarity graph over the pixels, heuristics to resolve
crossing diagonals, a reshaped cell graph, and quadratic B-splines through the
resulting boundaries. What this script adds is everything around it: cutting
the sheet up, turning an alpha-keyed sprite into something the algorithm can
read, rasterising the splines back to pixels, and putting the sheet together.

    python3 scripts/depixelize_tileset.py gfx/tilesets/dwarf_fortress/hack_square_64x64.png

Installing the library
----------------------
`pip install depixelizer` fails: the published source distribution forgets to
include its own VERSION file. Until that is fixed upstream::

    pip install pypng networkx svgwrite
    pip download --no-deps --no-binary :all: depixelizer
    tar xzf depixelizer-0.0.2.tar.gz && cd depixelizer-0.0.2
    echo 0.0.2 > VERSION && pip install .

Note that `depixelizer` is GPL-3. Using it to prepare artwork offline is fine;
shipping it inside the game would put the game under the same licence.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from depixelizer.depixelize.pixel_data import PixelData
    from depixelizer.geometry import smoothen
except ImportError:  # pragma: no cover - a message is more use than a stack trace
    print(__doc__.split("Installing the library")[1], file=sys.stderr)
    raise SystemExit("depixelizer is not installed; see above.")


# The spline optimiser is a random walk: each control point takes a number of
# small random steps and keeps whichever lowered its energy. The defaults take
# 20 steps of at most 0.05 pixels, which is both slow and barely enough to move
# anything. Fewer, longer steps reach the same place in a third of the time --
# measured on this tileset, the two are indistinguishable side by side.
SMOOTHER_GUESS_OFFSET = 0.6
SMOOTHER_ITERATIONS = 8

# How finely each quadratic Bezier is sampled when filling it, and how much the
# fill is supersampled before being scaled down. Together these are what make
# the traced edge land smoothly rather than as a staircase of its own.
CURVE_SAMPLES = 8
FILL_SUPERSAMPLE = 4

# A quarter of the cores, by default. Memory is not the constraint -- a worker
# peaks around 80MB -- but a tile takes the better part of a minute of solid
# computation, and a run is hundreds of tiles. Filling every core with that for
# ten minutes starves whatever the machine is really for; under WSL it starves
# the Windows host the virtual machine depends on, and the desktop goes down
# with it. Take a quarter, run it politely, and let the user ask for more.
DEFAULT_WORKER_SHARE = 4

# Workers run at the lowest priority the scheduler offers, so that however many
# are asked for, they only ever get the time nothing else wants.
WORKER_NICENESS = 19


@dataclass(frozen=True)
class Job:
    """One tile to trace: where it is on the sheet and what it looks like."""

    column: int
    row: int
    # Alpha only. These sheets are white glyphs on transparent, so the shape is
    # entirely in the alpha channel and the colour never varies.
    alpha: np.ndarray

    @property
    def blank(self) -> bool:
        return not (self.alpha > 128).any()

    @property
    def solid(self) -> bool:
        return (self.alpha > 128).all()


def quantise(alpha: np.ndarray, levels: int) -> np.ndarray:
    """Turn an alpha channel into the greyscale image the tracer reads.

    Inverted, so the glyph is dark on a light background: `depixelizer` treats
    every distinct colour as its own region, and it is easier to reason about
    the result when the ink is the thing that is not the background.

    `levels` of 2 is what the algorithm is designed for -- a hard edge, one
    region per colour. Higher values keep some of the source's anti-aliasing at
    the cost of splitting each soft edge into a stack of thin regions, which is
    both slower and, on artwork like this, not obviously better. That is the
    comparison this argument exists to let you make.

    Above about six levels the library starts failing: it only handles two of
    the three ways a 2x2 block can be connected when both its diagonals are
    present, and asserts on the third. Soft edges produce that third case
    readily, hard ones essentially never. `trace` catches it, so a sheet still
    finishes, but a high setting will quietly fall back on many of its tiles.
    """
    if levels <= 2:
        return np.where(alpha > 128, 0, 255).astype(np.uint8)
    step = 256 // levels
    band = np.minimum(alpha // step, levels - 1)
    return (255 - band * (255 // (levels - 1))).astype(np.uint8)


def curve_points(spline, scale: float) -> list[tuple[float, float]]:
    """A quadratic B-spline as a flat polygon, sampled fine enough to fill.

    The library hands out its splines as a run of quadratic Beziers sharing
    endpoints; each is walked at `CURVE_SAMPLES` steps and the whole run is
    filled as one polygon, which closes it implicitly.
    """
    points: list[tuple[float, float]] = []
    for start, control, end in spline.Quadratic_Bezier_Fit():
        for step in range(CURVE_SAMPLES):
            t = step / CURVE_SAMPLES
            u = 1.0 - t
            points.append(
                (
                    (u * u * start[0] + 2 * u * t * control[0] + t * t * end[0]) * scale,
                    (u * u * start[1] + 2 * u * t * control[1] + t * t * end[1]) * scale,
                )
            )
    return points


def rasterise(shapes, scale: int, size: int, smoothed: bool = True) -> np.ndarray:
    """Fill the traced shapes back into an alpha channel.

    Each shape is filled with the alpha its grey level stands for, and its
    inner paths -- the holes, the counters of an 'A' or the eyes of a face --
    are punched back out. Largest first, so a small detail sitting inside a
    bigger region is drawn after the region it sits in rather than under it.

    The fill happens at `FILL_SUPERSAMPLE` times the wanted size and is scaled
    down afterwards, which is what gives the traced edge its anti-aliasing.

    `smoothed` picks the optimised splines over the raw ones. The raw ones are
    still the algorithm's curves through the reshaped cell graph -- only the
    final energy minimisation is missing -- so they are a far better fallback
    than giving up on the trace.
    """
    big = size * FILL_SUPERSAMPLE
    canvas = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(canvas)
    step = scale * FILL_SUPERSAMPLE

    for shape in sorted(shapes, key=lambda s: -len(s.pixels)):
        # Grey came from inverted alpha, so this puts it back.
        opacity = 255 - int(shape.value[0])
        if opacity == 0:
            continue
        splines = shape.smooth_splines if smoothed else shape.splines
        draw.polygon(curve_points(splines[0], step), fill=opacity)
        for hole in splines[1:]:
            draw.polygon(curve_points(hole, step), fill=0)

    return np.asarray(canvas.resize((size, size), Image.LANCZOS))


def smooth_resize(alpha: np.ndarray, size: int) -> np.ndarray:
    """A plain filtered enlargement, for tiles the tracer could not handle."""
    return np.asarray(
        Image.fromarray(alpha, "L").resize((size, size), Image.LANCZOS)
    )


# What happened to a tile, worst last, so a run can report how it went.
TRACED, UNSMOOTHED, RESIZED = "traced", "unsmoothed", "resized"


def trace(job: Job, levels: int, scale: int) -> tuple[int, int, np.ndarray, str]:
    """Upscale one tile. Returns where it goes, its alpha, and how it went.

    The library is fragile on input it was not written for, in two ways seen on
    real sheets: it asserts on a 2x2 block topology it does not handle, and its
    spline evaluation leaves a variable unbound for a degenerate curve. Neither
    should cost a run that takes tens of minutes, so each stage falls back to
    the last one that worked:

        the traced curves, optimised   -- what the algorithm is for
        the traced curves, unoptimised -- still Kopf-Lischinski, just blockier
        a filtered enlargement         -- only if the trace itself failed
    """
    size = job.alpha.shape[0] * scale

    # Nothing to trace, and no point paying for it.
    if job.blank:
        return job.column, job.row, np.zeros((size, size), np.uint8), TRACED
    if job.solid and levels <= 2:
        return job.column, job.row, np.full((size, size), 255, np.uint8), TRACED

    smoothen.SplineSmoother.GUESS_OFFSET = SMOOTHER_GUESS_OFFSET
    smoothen.SplineSmoother.ITERATIONS = SMOOTHER_ITERATIONS

    grey = quantise(job.alpha, levels)
    rows = [[(int(v), int(v), int(v)) for v in row] for row in grey]

    try:
        data = PixelData(rows)
        # The library narrates every stage on stdout, which is unhelpful when
        # there are hundreds of tiles going at once.
        with contextlib.redirect_stdout(io.StringIO()):
            data.create_pixel_graph()
            data.remove_diagonals()
            data.create_grid_graph()
            data.deform_grid()
            data.create_shapes()
            data.get_boundaries()
            data.add_shape_boundaries()
    except Exception:
        return job.column, job.row, smooth_resize(job.alpha, size), RESIZED

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            data.smooth_splines()
        return job.column, job.row, rasterise(data.shapes, scale, size), TRACED
    except Exception:
        return (
            job.column,
            job.row,
            rasterise(data.shapes, scale, size, smoothed=False),
            UNSMOOTHED,
        )


def _trace_star(arguments):
    """`Pool.imap_unordered` takes one argument; this unpacks it."""
    return trace(*arguments)


def _be_polite() -> None:
    """Drop a worker to the back of the scheduler's queue.

    Run once per worker as it starts. This is what makes the parallelism safe
    to turn up: the work is pure computation with no deadline, so it should
    never be the reason something else on the machine stops responding.
    """
    try:
        os.nice(WORKER_NICENESS)
    except (OSError, AttributeError):  # pragma: no cover - not every platform
        pass


def cut_into_tiles(sheet: Image.Image, tile: int) -> list[Job]:
    """Every tile of the sheet, in reading order."""
    alpha = np.asarray(sheet)[..., 3]
    across, down = sheet.width // tile, sheet.height // tile
    return [
        Job(column, row, alpha[row * tile:(row + 1) * tile,
                               column * tile:(column + 1) * tile])
        for row in range(down)
        for column in range(across)
    ]


def upscale_sheet(
    sheet: Image.Image,
    tile: int,
    scale: int,
    levels: int,
    workers: int,
    label: str,
) -> Image.Image:
    """Trace every tile and reassemble them into one larger sheet.

    Tiles are independent, so they go out to a process pool. This is the whole
    reason the job is tractable: a single 64-pixel tile takes the better part
    of a minute, and there are usually a couple of hundred of them.
    """
    jobs = cut_into_tiles(sheet, tile)
    out_tile = tile * scale
    canvas = np.zeros(
        (sheet.height * scale, sheet.width * scale, 4), dtype=np.uint8
    )
    # White glyphs: only the alpha carries the shape, so the colour is uniform.
    canvas[..., :3] = 255

    worth_tracing = sum(1 for job in jobs if not job.blank)
    print(
        f"{label}: {len(jobs)} tiles ({worth_tracing} with something in them), "
        f"{tile}px -> {out_tile}px, {workers} of {os.cpu_count()} cores "
        f"at nice {WORKER_NICENESS}",
        flush=True,
    )

    started = time.perf_counter()
    done = 0
    outcomes: dict[str, int] = {}
    with Pool(workers, initializer=_be_polite) as pool:
        for column, row, alpha, outcome in pool.imap_unordered(
            _trace_star, [(job, levels, scale) for job in jobs], chunksize=1
        ):
            top, left = row * out_tile, column * out_tile
            canvas[top:top + out_tile, left:left + out_tile, 3] = alpha
            done += 1
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if done % 16 == 0 or done == len(jobs):
                elapsed = time.perf_counter() - started
                rate = done / max(elapsed, 1e-6)
                print(
                    f"  {done}/{len(jobs)} tiles, {elapsed:.0f}s elapsed, "
                    f"{(len(jobs) - done) / max(rate, 1e-6):.0f}s left",
                    flush=True,
                )

    for outcome in (UNSMOOTHED, RESIZED):
        if outcomes.get(outcome):
            print(f"  {outcomes[outcome]} tile(s) fell back to: {outcome}", flush=True)
    return Image.fromarray(canvas, "RGBA")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sheet", type=Path, help="the tilesheet to upscale")
    parser.add_argument(
        "--tile", type=int, default=64, help="tile size in the source (default 64)"
    )
    parser.add_argument(
        "--scale", type=int, default=8, help="how much bigger to make it (default 8)"
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[2, 6],
        help="alpha levels to trace at: 2 is a hard edge, which is what the "
        "algorithm is designed for; higher keeps some of the source's "
        "anti-aliasing, but above about six the library fails on many tiles "
        "and they fall back to a filter. Give several to write one sheet per "
        "setting (default: 2 6)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // DEFAULT_WORKER_SHARE),
        help="processes to trace with. They run at the lowest scheduling "
        "priority, but each still occupies a core solidly for the length of "
        "the run (default: a quarter of the cores)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write, defaulting to beside the source",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    if not args.sheet.exists():
        print(f"No such sheet: {args.sheet}", file=sys.stderr)
        return 1

    sheet = Image.open(args.sheet).convert("RGBA")
    if sheet.width % args.tile or sheet.height % args.tile:
        print(
            f"{sheet.width}x{sheet.height} is not a whole number of "
            f"{args.tile}px tiles",
            file=sys.stderr,
        )
        return 1

    out_dir = args.output_dir or args.sheet.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tile = args.tile * args.scale

    for levels in args.levels:
        kind = "hard" if levels <= 2 else f"soft{levels}"
        label = f"{kind} ({levels} alpha level{'s' if levels > 1 else ''})"
        result = upscale_sheet(
            sheet, args.tile, args.scale, levels, args.workers, label
        )
        name = f"{args.sheet.stem}_x{args.scale}_{kind}.png"
        destination = out_dir / name
        result.save(destination)
        print(
            f"  wrote {destination} "
            f"({result.width}x{result.height}, {out_tile}px tiles)\n",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
