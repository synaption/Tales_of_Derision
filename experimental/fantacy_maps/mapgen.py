"""Procedural generator for Tolkien-style hand-drawn fantasy maps.

Produces a parchment-textured island/continent map with hachured
mountain ranges drawn in profile, pine-cluster forests, rivers,
an inked coastline, a compass rose and a title banner - in the
spirit of the old Middle-earth maps.
"""
import argparse
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Palette (ink on aged parchment)
# ---------------------------------------------------------------------------
PARCHMENT = (222, 199, 152)
PARCHMENT_DARK = (196, 168, 118)
INK = (73, 52, 34)
INK_LIGHT = (110, 82, 56)
FOREST_INK = (66, 74, 42)
WATER_TINT = (198, 176, 132)


def lerp_color(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ---------------------------------------------------------------------------
# Heightmap
# ---------------------------------------------------------------------------
def diamond_square(power, roughness, rng):
    size = 2 ** power + 1
    grid = np.zeros((size, size), dtype=np.float64)
    grid[0, 0] = rng.uniform(-1, 1)
    grid[0, -1] = rng.uniform(-1, 1)
    grid[-1, 0] = rng.uniform(-1, 1)
    grid[-1, -1] = rng.uniform(-1, 1)

    step = size - 1
    scale = 1.0
    while step > 1:
        half = step // 2

        # diamond step
        for y in range(half, size, step):
            for x in range(half, size, step):
                avg = (grid[y - half, x - half] + grid[y - half, x + half] +
                       grid[y + half, x - half] + grid[y + half, x + half]) / 4.0
                grid[y, x] = avg + rng.uniform(-1, 1) * scale

        # square step
        for y in range(0, size, half):
            for x in range((y // half + 1) % 2 * half, size, step):
                total, count = 0.0, 0
                if y - half >= 0:
                    total += grid[y - half, x]
                    count += 1
                if y + half < size:
                    total += grid[y + half, x]
                    count += 1
                if x - half >= 0:
                    total += grid[y, x - half]
                    count += 1
                if x + half < size:
                    total += grid[y, x + half]
                    count += 1
                grid[y, x] = total / count + rng.uniform(-1, 1) * scale

        step = half
        scale *= roughness

    grid -= grid.min()
    grid /= grid.max()
    return grid


def make_island_heightmap(w, h, rng, power=8, roughness=0.55, land_fraction=0.55):
    """Returns (elevation, land_mask). Elevation is normalized 0..1 across
    land only (0 = coastline, 1 = highest peak); sea cells are 0."""
    raw = diamond_square(power, roughness, rng)
    img = Image.fromarray((raw * 255).astype(np.uint8))
    img = img.resize((w, h), Image.BICUBIC)
    noise = np.asarray(img, dtype=np.float64) / 255.0

    # Radial falloff (allowed to go negative outside the inscribed ellipse)
    # tapers the landmass into sea near the canvas edges/corners without
    # flattening the interior into a single smooth plateau - the raw
    # noise's ridges and valleys are preserved.
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((xx - cx) / (w * 0.5)) ** 2 + ((yy - cy) / (h * 0.5)) ** 2)
    falloff = 1.0 - dist ** 2

    combined = noise * 0.65 + falloff * 0.35
    sea_level = np.percentile(combined, 100 * (1 - land_fraction))
    land_mask = combined > sea_level

    elevation = np.zeros_like(combined)
    span = combined.max() - sea_level
    if span > 1e-6:
        elevation[land_mask] = (combined[land_mask] - sea_level) / span
    return elevation, land_mask


# ---------------------------------------------------------------------------
# Parchment background
# ---------------------------------------------------------------------------
def paint_parchment(size, rng):
    w, h = size
    base = Image.new("RGB", (w, h), PARCHMENT)

    # low-frequency mottling
    mottle = diamond_square(7, 0.6, rng)
    mottle_img = Image.fromarray((mottle * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    mottle_arr = np.asarray(mottle_img, dtype=np.float64) / 255.0

    arr = np.asarray(base, dtype=np.float64)
    shade = (mottle_arr - 0.5) * 40.0
    arr += shade[..., None]

    # vignette
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((xx - cx) / (w * 0.55)) ** 2 + ((yy - cy) / (h * 0.55)) ** 2)
    vig = np.clip(1.0 - (dist - 0.6) * 0.6, 0.55, 1.0)
    arr *= vig[..., None]

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # fine grain
    grain = (rng.random((h, w)) * 24 - 12).astype(np.int16)
    arr = np.asarray(img, dtype=np.int16) + grain[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def add_age_stains(img, rng, count=10):
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r = int(min(w, h) * 0.1)
    for _ in range(count):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(int(min(w, h) * 0.05), int(min(w, h) * 0.18))
        alpha = rng.integers(8, 22)
        color = PARCHMENT_DARK + (int(alpha),)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    overlay = overlay.filter(ImageFilter.GaussianBlur(max(1, int(r) // 2)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"), (0, 0))
    return img


# ---------------------------------------------------------------------------
# Border (deckled edge + ink frame)
# ---------------------------------------------------------------------------
def draw_deckled_border(img, rng, margin=26):
    w, h = img.size

    # darker backdrop outside the deckled edge
    backdrop = Image.new("RGB", (w, h), (74, 58, 40))
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)

    steps = 140
    top, bottom, left, right = [], [], [], []
    jitter = margin * 0.35
    for i in range(steps + 1):
        t = i / steps
        top.append((t * w, margin + rng.uniform(-jitter, jitter)))
        bottom.append((t * w, h - margin + rng.uniform(-jitter, jitter)))
        left.append((margin + rng.uniform(-jitter, jitter), t * h))
        right.append((w - margin + rng.uniform(-jitter, jitter), t * h))

    poly = top + right + list(reversed(bottom)) + list(reversed(left))
    mdraw.polygon(poly, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))

    img.paste(backdrop, (0, 0), Image.eval(mask, lambda v: 255 - v))

    draw = ImageDraw.Draw(img)
    draw.line(poly + [poly[0]], fill=INK, width=3, joint="curve")

    inset = 10
    draw.rectangle((margin + inset, margin + inset, w - margin - inset, h - margin - inset),
                    outline=INK_LIGHT, width=1)

    # tick marks along the inner frame like an old chart's scale bar
    for x in range(int(margin + inset), int(w - margin - inset), 14):
        draw.line((x, margin + inset, x, margin + inset + 5), fill=INK_LIGHT, width=1)
        draw.line((x, h - margin - inset - 5, x, h - margin - inset), fill=INK_LIGHT, width=1)
    for y in range(int(margin + inset), int(h - margin - inset), 14):
        draw.line((margin + inset, y, margin + inset + 5, y), fill=INK_LIGHT, width=1)
        draw.line((w - margin - inset - 5, y, w - margin - inset, y), fill=INK_LIGHT, width=1)

    return margin + inset + 6


# ---------------------------------------------------------------------------
# Coastline
# ---------------------------------------------------------------------------
def draw_coastline(draw, land_mask, rng):
    h, w = land_mask.shape
    boundary = (
        (land_mask & ~np.roll(land_mask, 1, axis=0))
        | (land_mask & ~np.roll(land_mask, -1, axis=0))
        | (land_mask & ~np.roll(land_mask, 1, axis=1))
        | (land_mask & ~np.roll(land_mask, -1, axis=1))
    )
    ys, xs = np.nonzero(boundary)
    grad_y, grad_x = np.gradient(land_mask.astype(np.float64))

    for x, y in zip(xs.tolist(), ys.tolist()):
        r = 1.6
        draw.ellipse((x - r, y - r, x + r, y + r), fill=INK)
        if rng.random() < 0.35:
            nx, ny = grad_x[y, x], grad_y[y, x]
            norm = math.hypot(nx, ny)
            if norm > 1e-6:
                nx, ny = nx / norm, ny / norm
                length = rng.uniform(3, 7)
                draw.line((x, y, x + nx * length, y + ny * length), fill=INK_LIGHT, width=1)


# ---------------------------------------------------------------------------
# Mountains (drawn in profile, "caterpillar" ranges)
# ---------------------------------------------------------------------------
def draw_peak(draw, x, y, size, rng):
    left = (x - size, y + size * 0.55)
    right = (x + size, y + size * 0.55)
    top = (x + rng.uniform(-size * 0.15, size * 0.15), y - size)
    draw.polygon([left, top, right], fill=PARCHMENT, outline=INK)
    draw.line([left, top], fill=INK, width=2)
    draw.line([top, right], fill=INK, width=2)
    # shaded flank strokes
    for i in range(1, 4):
        t = i / 4.0
        sx = top[0] + (right[0] - top[0]) * t
        sy = top[1] + (right[1] - top[1]) * t
        ex = sx - size * 0.28
        ey = sy + size * 0.12
        draw.line((sx, sy, ex, ey), fill=INK_LIGHT, width=1)
    # snow-cap hint near summit
    cap_l = (top[0] - size * 0.18, top[1] + size * 0.32)
    cap_r = (top[0] + size * 0.18, top[1] + size * 0.32)
    draw.line([cap_l, top, cap_r], fill=INK, width=1)


def draw_mountain_range(draw, cells, rng, cell_size):
    if not cells:
        return
    # draw farther-back (smaller y) peaks first so nearer ones overlap them,
    # producing the layered "caterpillar ridge" silhouette of hand-drawn maps.
    # Cells are thinned by minimum spacing so peaks read as distinct summits
    # rather than fusing into one solid mass.
    ordered = sorted(cells, key=lambda c: c[1])
    min_gap = cell_size * 1.8
    chosen = []
    last_pos = None
    for cx, cy in ordered:
        px, py = cx * cell_size, cy * cell_size
        if last_pos is not None and math.hypot(px - last_pos[0], py - last_pos[1]) < min_gap:
            continue
        chosen.append((px, py))
        last_pos = (px, py)

    for px, py in chosen:
        px += rng.uniform(-cell_size * 0.35, cell_size * 0.35)
        py += rng.uniform(-cell_size * 0.2, cell_size * 0.2)
        size = cell_size * rng.uniform(1.0, 1.5)
        draw_peak(draw, px, py, size, rng)


def make_ridge_field(w, h, rng):
    raw = diamond_square(8, 0.55, rng)
    img = Image.fromarray((raw * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    field = np.asarray(img, dtype=np.float64) / 255.0
    # fold around the midpoint so values peak into thin sinuous ridges
    # instead of the broad rounded blobs plain noise produces
    return 1.0 - np.abs(field * 2.0 - 1.0)


def place_mountains(draw, height, ridge, land_mask, rng, cell_size=14, coverage=0.06):
    h, w = height.shape
    combined = ridge * height
    gh, gw = h // cell_size, w // cell_size
    block_score = np.zeros((gh, gw), dtype=np.float64)
    block_land_frac = np.zeros((gh, gw), dtype=np.float64)
    for gy in range(gh):
        for gx in range(gw):
            cblock = combined[gy * cell_size:(gy + 1) * cell_size, gx * cell_size:(gx + 1) * cell_size]
            lblock = land_mask[gy * cell_size:(gy + 1) * cell_size, gx * cell_size:(gx + 1) * cell_size]
            block_score[gy, gx] = cblock.mean()
            block_land_frac[gy, gx] = lblock.mean()

    eligible = block_land_frac > 0.6
    if eligible.any():
        cutoff = np.percentile(block_score[eligible], 100 * (1 - coverage))
    else:
        cutoff = 1.0
    grid_high = eligible & (block_score > cutoff)

    try:
        from scipy import ndimage
        labels, n = ndimage.label(grid_high, structure=np.ones((3, 3)))
    except ImportError:
        labels, n = grid_high.astype(int), int(grid_high.any())

    ranges = []
    for label_id in range(1, n + 1):
        ys, xs = np.nonzero(labels == label_id)
        if len(xs) == 0:
            continue
        cells = list(zip(xs.tolist(), ys.tolist()))
        ranges.append(cells)

    for cells in ranges:
        draw_mountain_range(draw, cells, rng, cell_size)

    return grid_high, cell_size


# ---------------------------------------------------------------------------
# Forests (pine clusters)
# ---------------------------------------------------------------------------
def draw_tree(draw, x, y, size, rng):
    half_w = size * rng.uniform(0.4, 0.55)
    apex = (x, y - size)
    base_l = (x - half_w, y)
    base_r = (x + half_w, y)
    draw.polygon([base_l, apex, base_r], fill=FOREST_INK, outline=INK)
    trunk_h = size * 0.22
    draw.line((x, y, x, y + trunk_h), fill=FOREST_INK, width=max(1, int(size * 0.12)))


def place_forests(draw, height, land_mask, mountain_mask, rng, w, h, cell_size=14, coverage=0.4):
    forest_noise = diamond_square(7, 0.6, rng)
    fn_img = Image.fromarray((forest_noise * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    forest_field = np.asarray(fn_img, dtype=np.float64) / 255.0

    gh, gw = mountain_mask.shape
    step = max(6, int(cell_size * 0.75))

    coords, scores = [], []
    for gy in range(0, h, step):
        for gx in range(0, w, step):
            y, x = min(gy, h - 1), min(gx, w - 1)
            if not land_mask[y, x]:
                continue
            local_h = height[y, x]
            if local_h > 0.68 or local_h < 0.15:
                continue
            mgy, mgx = min(y // cell_size, gh - 1), min(x // cell_size, gw - 1)
            if mountain_mask[mgy, mgx]:
                continue
            coords.append((gx, gy))
            scores.append(forest_field[y, x])

    if not coords:
        return

    # threshold within the eligible (land, mid-elevation, non-mountain) area
    # itself, so forest cover shows up reliably regardless of how the
    # independent forest-noise field happens to line up with the coastline
    cutoff = np.percentile(scores, 100 * (1 - coverage))
    for (gx, gy), score in zip(coords, scores):
        if score >= cutoff and rng.random() < 0.85:
            jx = gx + rng.uniform(-step * 0.4, step * 0.4)
            jy = gy + rng.uniform(-step * 0.4, step * 0.4)
            size = rng.uniform(step * 0.5, step * 0.85)
            draw_tree(draw, jx, jy, size, rng)


# ---------------------------------------------------------------------------
# Rivers
# ---------------------------------------------------------------------------
def trace_river(height, land_mask, start, rng, max_steps=4000):
    h, w = height.shape
    x, y = start
    path = [(x, y)]
    for _ in range(max_steps):
        xi, yi = int(x), int(y)
        if xi <= 1 or xi >= w - 2 or yi <= 1 or yi >= h - 2:
            break
        if not land_mask[yi, xi]:
            break
        best = None
        best_h = height[yi, xi]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = xi + dx, yi + dy
                nh = height[ny, nx] - rng.uniform(0, 0.01)
                if nh < best_h:
                    best_h = nh
                    best = (nx + rng.uniform(-0.3, 0.3), ny + rng.uniform(-0.3, 0.3))
        if best is None:
            break
        x, y = best
        path.append((x, y))
        if len(path) > 6 and math.hypot(path[-1][0] - path[-6][0], path[-1][1] - path[-6][1]) < 0.5:
            break
    return path


def draw_rivers(draw, height, land_mask, rng, count=4):
    from scipy.ndimage import gaussian_filter
    smooth_height = gaussian_filter(height, sigma=6)

    h, w = height.shape
    high_pts = np.argwhere(height > 0.75)
    if len(high_pts) == 0:
        return
    rng.shuffle(high_pts)
    drawn = 0
    for py, px in high_pts:
        if drawn >= count:
            break
        if not land_mask[py, px]:
            continue
        path = trace_river(smooth_height, land_mask, (float(px), float(py)), rng)
        if len(path) < 20:
            continue
        draw.line(path, fill=INK_LIGHT, width=2, joint="curve")
        drawn += 1


# ---------------------------------------------------------------------------
# Compass rose
# ---------------------------------------------------------------------------
def draw_compass_rose(draw, cx, cy, r):
    for ang_deg in range(0, 360, 45):
        a = math.radians(ang_deg - 90)
        length = r if ang_deg % 90 == 0 else r * 0.55
        x2, y2 = cx + math.cos(a) * length, cy + math.sin(a) * length
        draw.line((cx, cy, x2, y2), fill=INK, width=2 if ang_deg % 90 == 0 else 1)

    for ang_deg in range(0, 360, 45):
        a = math.radians(ang_deg - 90)
        pa = math.radians(ang_deg - 90 - 8)
        pb = math.radians(ang_deg - 90 + 8)
        length = r if ang_deg % 90 == 0 else r * 0.55
        tip = (cx + math.cos(a) * length, cy + math.sin(a) * length)
        base_a = (cx + math.cos(pa) * length * 0.18, cy + math.sin(pa) * length * 0.18)
        base_b = (cx + math.cos(pb) * length * 0.18, cy + math.sin(pb) * length * 0.18)
        draw.polygon([tip, base_a, base_b], outline=INK, fill=PARCHMENT if ang_deg % 90 else INK_LIGHT)

    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=INK, width=1)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", int(r * 0.28))
    except OSError:
        font = ImageFont.load_default()
    labels = {"N": (cx, cy - r * 1.28), "S": (cx, cy + r * 1.1), "E": (cx + r * 1.28, cy), "W": (cx - r * 1.28, cy)}
    for text, (lx, ly) in labels.items():
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text((lx - (bbox[2] - bbox[0]) / 2, ly - (bbox[3] - bbox[1]) / 2), text, fill=INK, font=font)


# ---------------------------------------------------------------------------
# Title banner
# ---------------------------------------------------------------------------
def draw_title_banner(img, draw, title, cx, cy, w_hint):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 30)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x, pad_y = 34, 14
    bw = max(w_hint, text_w + pad_x * 2)
    bh = text_h + pad_y * 2

    left, right = cx - bw / 2, cx + bw / 2
    top, bottom = cy - bh / 2, cy + bh / 2
    tail = bh * 0.5

    ribbon = [
        (left - tail, top), (right + tail, top),
        (right, cy), (right + tail, bottom),
        (left - tail, bottom), (left, cy),
    ]
    draw.polygon(ribbon, fill=PARCHMENT_DARK, outline=INK)
    draw.rectangle((left, top, right, bottom), fill=lerp_color(PARCHMENT, PARCHMENT_DARK, 0.25), outline=INK, width=2)

    draw.text((cx - text_w / 2 - bbox[0], cy - text_h / 2 - bbox[1]), title, fill=INK, font=font)


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------
def generate_map(width, height, seed, title):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)

    img = paint_parchment((width, height), nprng)
    img = add_age_stains(img, nprng, count=8)

    hmap, land_mask = make_island_heightmap(width, height, nprng, power=8, roughness=0.55)

    # tint the sea very slightly so the coastline reads clearly
    sea_arr = np.zeros((height, width, 4), dtype=np.uint8)
    sea_arr[~land_mask] = (*WATER_TINT, 60)
    sea_overlay = Image.fromarray(sea_arr, "RGBA")
    img.paste(Image.alpha_composite(img.convert("RGBA"), sea_overlay).convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(img)

    draw_coastline(draw, land_mask, nprng)
    ridge_field = make_ridge_field(width, height, nprng)
    mountain_mask, cell_size = place_mountains(draw, hmap, ridge_field, land_mask, rng)
    place_forests(draw, hmap, land_mask, mountain_mask, nprng, width, height, cell_size=cell_size)
    draw_rivers(draw, hmap, land_mask, nprng, count=7)

    margin = draw_deckled_border(img, nprng, margin=26)
    draw = ImageDraw.Draw(img)

    draw_compass_rose(draw, width - margin - 70, height - margin - 70, 48)
    draw_title_banner(img, draw, title, width / 2, margin + 46, width * 0.28)

    return img


def main():
    parser = argparse.ArgumentParser(description="Generate a Tolkien-style fantasy map.")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--title", type=str, default="Mind'enalap")
    parser.add_argument("--out", type=str, default="map.png")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    print(f"Generating map '{args.title}' with seed {seed}...")
    img = generate_map(args.width, args.height, seed, args.title)
    img.save(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
