#!/usr/bin/env python3
"""
Apply a parchment-and-canvas shader-like effect to a JPG or PNG.

Dependencies:
    pip install pillow numpy

Examples:
    python parchment_canvas.py input.png output.png
    python parchment_canvas.py input.jpg output.jpg --strength 0.75
    python parchment_canvas.py input.png output.png --seed 42 --weave-strength 0.10
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def hex_rgb(value: str) -> np.ndarray:
    """Convert #RRGGBB to a float RGB array in the 0..1 range."""
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError("Color must be in #RRGGBB format.")
    try:
        return np.array(
            [int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)],
            dtype=np.float32,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color must be in #RRGGBB format.") from exc


def smooth_noise(
    width: int,
    height: int,
    cells_x: float,
    cells_y: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create smooth value noise by resizing a small random grid.

    cells_x/cells_y roughly control the number of broad noise regions.
    """
    grid_w = max(2, int(math.ceil(cells_x)) + 2)
    grid_h = max(2, int(math.ceil(cells_y)) + 2)

    grid = rng.random((grid_h, grid_w), dtype=np.float32)
    grid_img = Image.fromarray(np.uint8(np.clip(grid, 0.0, 1.0) * 255), mode="L")
    resized = grid_img.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def fbm(
    width: int,
    height: int,
    base_cells: float,
    octaves: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simple fractal Brownian motion made from several smooth-noise layers."""
    value = np.zeros((height, width), dtype=np.float32)
    amplitude = 0.5
    total = 0.0
    aspect = height / max(width, 1)

    for octave in range(octaves):
        cells_x = base_cells * (2.03**octave)
        cells_y = max(2.0, cells_x * aspect)
        value += smooth_noise(width, height, cells_x, cells_y, rng) * amplitude
        total += amplitude
        amplitude *= 0.5

    return value / max(total, 1e-6)


def soft_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Photoshop-like soft-light blend. Inputs are 0..1 float arrays."""
    low = base - (1.0 - 2.0 * blend) * base * (1.0 - base)
    d = np.where(
        base <= 0.25,
        ((16.0 * base - 12.0) * base + 4.0) * base,
        np.sqrt(np.clip(base, 0.0, 1.0)),
    )
    high = base + (2.0 * blend - 1.0) * (d - base)
    return np.where(blend <= 0.5, low, high)


def apply_parchment_canvas(
    image: Image.Image,
    *,
    seed: int = 7,
    strength: float = 0.72,
    color_strength: float = 0.36,
    weave_frequency: float = 105.0,
    weave_strength: float = 0.10,
    aging_strength: float = 0.34,
    edge_darkness: float = 0.58,
    parchment_color: np.ndarray = np.array([0.72, 0.55, 0.31], dtype=np.float32),
    highlight_color: np.ndarray = np.array([0.95, 0.84, 0.60], dtype=np.float32),
    aged_color: np.ndarray = np.array([0.25, 0.12, 0.05], dtype=np.float32),
) -> Image.Image:
    """Return a new RGBA image with a parchment/canvas treatment."""
    src_rgba = image.convert("RGBA")
    rgba = np.asarray(src_rgba, dtype=np.float32) / 255.0
    src = rgba[..., :3]
    alpha = rgba[..., 3:4]

    height, width = src.shape[:2]
    rng = np.random.default_rng(seed)

    # Broad paper discoloration and fine fiber variation.
    stains = fbm(width, height, base_cells=5.0, octaves=5, rng=rng)
    fibers = fbm(width, height, base_cells=42.0, octaves=3, rng=rng)

    # Coordinates and low-frequency weave distortion.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    uv_x = xx / max(width - 1, 1)
    uv_y = yy / max(height - 1, 1)

    warp_x = smooth_noise(width, height, 12.0, 12.0 * height / max(width, 1), rng)
    warp_y = smooth_noise(width, height, 12.0, 12.0 * height / max(width, 1), rng)
    warped_x = uv_x + (warp_x - 0.5) * 0.016
    warped_y = uv_y + (warp_y - 0.5) * 0.016

    # Crossed canvas threads. The high exponent keeps strands narrow.
    horizontal = np.power(
        0.5 + 0.5 * np.sin(warped_y * weave_frequency * 2.0 * np.pi),
        8.0,
    )
    vertical = np.power(
        0.5
        + 0.5
        * np.sin(warped_x * weave_frequency * 0.92 * 2.0 * np.pi + 0.7),
        8.0,
    )
    weave = horizontal + vertical + horizontal * vertical * 0.5
    weave = weave - np.mean(weave)

    # Irregular aged border.
    nearest_edge = np.minimum.reduce([uv_x, 1.0 - uv_x, uv_y, 1.0 - uv_y])
    edge_noise = fbm(width, height, base_cells=10.0, octaves=4, rng=rng)
    irregular_edge = nearest_edge + (edge_noise - 0.5) * 0.05
    edge_mask = 1.0 - np.clip(irregular_edge / 0.18, 0.0, 1.0)
    edge_mask = edge_mask * edge_mask * (3.0 - 2.0 * edge_mask)

    # Build the parchment surface itself.
    paper_value = np.clip(stains * aging_strength + fibers * 0.12, 0.0, 1.0)
    paper = (
        parchment_color[None, None, :] * (1.0 - paper_value[..., None])
        + highlight_color[None, None, :] * paper_value[..., None]
    )

    dark_stains = np.power(np.clip(1.0 - stains, 0.0, 1.0), 3.0) * 0.18
    paper = (
        paper * (1.0 - dark_stains[..., None])
        + aged_color[None, None, :] * dark_stains[..., None]
    )
    paper *= 1.0 + weave[..., None] * weave_strength
    paper = (
        paper * (1.0 - edge_mask[..., None] * edge_darkness)
        + aged_color[None, None, :] * edge_mask[..., None] * edge_darkness
    )
    paper = np.clip(paper, 0.0, 1.0)

    # Treat the source as pigment printed/painted onto the textured surface.
    textured_source = soft_light(src, np.clip(paper, 0.0, 1.0))
    toned_source = src * (1.0 - color_strength) + textured_source * color_strength

    # Multiply fine paper texture into the image while keeping recognizable colors.
    surface_luma = np.mean(paper, axis=2, keepdims=True)
    detail = (fibers[..., None] - 0.5) * 0.10 + weave[..., None] * weave_strength
    surfaced = toned_source * (0.84 + 0.22 * surface_luma) + detail

    # Warm the image slightly toward parchment, especially in highlights.
    luminance = (
        src[..., 0:1] * 0.2126
        + src[..., 1:2] * 0.7152
        + src[..., 2:3] * 0.0722
    )
    warm_mix = np.clip(0.18 + luminance * 0.22, 0.0, 0.45) * color_strength
    surfaced = surfaced * (1.0 - warm_mix) + paper * warm_mix

    result = src * (1.0 - strength) + surfaced * strength
    result = np.clip(result, 0.0, 1.0)

    out = np.concatenate([result, alpha], axis=2)
    return Image.fromarray(np.uint8(out * 255.0 + 0.5), mode="RGBA")


def save_image(image: Image.Image, output_path: Path, jpeg_quality: int) -> None:
    """Save using the output extension, preserving alpha for PNG."""
    suffix = output_path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        # JPEG has no alpha. Composite transparent pixels over parchment beige.
        background = Image.new("RGB", image.size, (184, 140, 79))
        background.paste(image.convert("RGBA"), mask=image.getchannel("A"))
        background.save(
            output_path,
            format="JPEG",
            quality=jpeg_quality,
            subsampling=0,
            optimize=True,
        )
    elif suffix == ".png":
        image.save(output_path, format="PNG", optimize=True)
    else:
        raise ValueError("Output filename must end in .jpg, .jpeg, or .png.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a procedural parchment-and-canvas effect to a JPG or PNG."
    )
    parser.add_argument("input", type=Path, help="Input .jpg/.jpeg/.png image")
    parser.add_argument("output", type=Path, help="Output .jpg/.jpeg/.png image")
    parser.add_argument("--seed", type=int, default=7, help="Texture seed")
    parser.add_argument(
        "--strength",
        type=float,
        default=0.72,
        help="Overall effect strength, usually 0..1",
    )
    parser.add_argument(
        "--color-strength",
        type=float,
        default=0.36,
        help="Amount of parchment color tint, usually 0..1",
    )
    parser.add_argument(
        "--weave-frequency",
        type=float,
        default=105.0,
        help="Number of canvas-thread cycles across the image",
    )
    parser.add_argument(
        "--weave-strength",
        type=float,
        default=0.10,
        help="Visibility of the canvas weave",
    )
    parser.add_argument(
        "--aging-strength",
        type=float,
        default=0.34,
        help="Strength of stains and uneven paper color",
    )
    parser.add_argument(
        "--edge-darkness",
        type=float,
        default=0.58,
        help="Darkening near image edges",
    )
    parser.add_argument(
        "--parchment-color",
        type=hex_rgb,
        default=hex_rgb("#B88C4F"),
        metavar="#RRGGBB",
    )
    parser.add_argument(
        "--highlight-color",
        type=hex_rgb,
        default=hex_rgb("#F2D699"),
        metavar="#RRGGBB",
    )
    parser.add_argument(
        "--aged-color",
        type=hex_rgb,
        default=hex_rgb("#40200D"),
        metavar="#RRGGBB",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality, 1..100",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.output.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise SystemExit("Output filename must end in .jpg, .jpeg, or .png.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(args.input) as image:
        result = apply_parchment_canvas(
            image,
            seed=args.seed,
            strength=float(np.clip(args.strength, 0.0, 1.0)),
            color_strength=float(np.clip(args.color_strength, 0.0, 1.0)),
            weave_frequency=max(1.0, args.weave_frequency),
            weave_strength=max(0.0, args.weave_strength),
            aging_strength=max(0.0, args.aging_strength),
            edge_darkness=float(np.clip(args.edge_darkness, 0.0, 1.0)),
            parchment_color=args.parchment_color,
            highlight_color=args.highlight_color,
            aged_color=args.aged_color,
        )

    save_image(
        result,
        args.output,
        jpeg_quality=int(np.clip(args.jpeg_quality, 1, 100)),
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()