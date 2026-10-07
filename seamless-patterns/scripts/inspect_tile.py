#!/usr/bin/env python3
"""Build and verify exact 3x3 tile copies; never judge visual seamlessness.

Requires local Python 3.9+ and Pillow. No network or generation calls.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

try:
    from PIL import Image, PngImagePlugin
except ImportError:
    raise SystemExit(
        "Pillow is unavailable in this Python environment. Use a local Python "
        "with Pillow (for example the Codex bundled runtime). No preview was built."
    )


PNG_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "I;16"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def png_options(tile):
    """Keep relevant color/alpha metadata without applying a color transform."""
    options = {
        key: tile.info[key]
        for key in ("icc_profile", "transparency", "dpi")
        if key in tile.info
    }
    chunks = PngImagePlugin.PngInfo()
    if "gamma" in tile.info:
        chunks.add(b"gAMA", struct.pack(">I", round(tile.info["gamma"] * 100000)))
    if "srgb" in tile.info:
        chunks.add(b"sRGB", bytes([tile.info["srgb"]]))
    if "chromaticity" in tile.info:
        chunks.add(b"cHRM", struct.pack(
            ">8I", *(round(value * 100000) for value in tile.info["chromaticity"])
        ))
    options["pnginfo"] = chunks
    return options


def verify_saved(path, expected, metadata):
    """Compare native samples, including RGB values under transparent pixels."""
    with Image.open(path) as actual:
        actual.load()
        if actual.size != expected.size or actual.mode != expected.mode:
            raise RuntimeError(f"Size or mode changed while saving {path.name}")
        if actual.tobytes() != expected.tobytes():
            raise RuntimeError(f"Pixel samples changed while saving {path.name}")
        if actual.getpalette() != expected.getpalette():
            raise RuntimeError(f"Palette changed while saving {path.name}")
        for key in ("transparency", "icc_profile", "gamma", "srgb", "chromaticity"):
            if actual.info.get(key) != metadata.get(key):
                raise RuntimeError(f"{key} changed while saving {path.name}")


def full_runs(values, full_value):
    """Zero-based [start, end) runs; a description, never a pass/fail test."""
    runs = []
    start = None
    for index, value in enumerate(list(values) + [None]):
        if value == full_value:
            if start is None:
                start = index
        elif start is not None:
            runs.append([start, index])
            start = None
    return runs


def background_distribution(tile, background):
    """Exact-background fractions expose empty bands without a quality threshold.

    Near-background colors, gradients, motif rhythm and intended negative space
    still require visual inspection. Fully transparent pixels ignore hidden RGB.
    """
    rgba = tile.convert("RGBA")
    samples = rgba.tobytes()
    width, height = rgba.size
    rows, columns = [0] * height, [0] * width
    target = bytes(background)
    for y in range(height):
        for x in range(width):
            offset = 4 * (y * width + x)
            match = samples[offset + 3] == 0 if background[3] == 0 else samples[offset:offset + 4] == target
            if match:
                rows[y] += 1
                columns[x] += 1
    return {
        "background_rgba": list(background), "comparison": "exact decoded RGBA; alpha-zero ignores hidden RGB",
        "background_fraction_by_row": [value / width for value in rows],
        "background_fraction_by_column": [value / height for value in columns],
        "fully_background_row_runs_y": full_runs(rows, width),
        "fully_background_column_runs_x": full_runs(columns, height),
        "verdict": "not_assessed",
        "limitations": "Exact matching misses near-background bands. Empty runs can be intentional. No universal threshold or automatic verdict.",
    }


def cyclic_offset(tile):
    """Integer half-period pixel permutation, for diagnosis only, not repair."""
    width, height = tile.size
    dx, dy = width // 2, height // 2
    offset = Image.new(tile.mode, tile.size)
    if tile.palette is not None:
        offset.putpalette(tile.getpalette(), rawmode="RGB")
    xs, ys = (0, width - dx, width), (0, height - dy, height)
    for left, right in zip(xs, xs[1:]):
        for top, bottom in zip(ys, ys[1:]):
            if left < right and top < bottom:
                offset.paste(tile.crop((left, top, right, bottom)), ((left + dx) % width, (top + dy) % height))
    return offset, [dx, dy]


def inspect_tile(source, output_dir, context=64, background=None):
    source = Path(source).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if context < 1:
        raise ValueError("--context must be a positive crop width in pixels")
    if background is not None:
        if not isinstance(background, (list, tuple)) or len(background) != 4 or any(
            type(channel) is not int or not 0 <= channel <= 255 for channel in background
        ):
            raise ValueError("Diagnostic background requires four RGBA integer channels (0..255)")
    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a new directory")

    source_digest = sha256(source)
    with Image.open(source) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Multi-frame input: export the final static tile first")
        if image.getexif().get(274, 1) not in (None, 1):
            raise ValueError(
                "Input requires an EXIF orientation transform; export an oriented "
                "static tile explicitly, then check that final file"
            )
        if image.mode not in PNG_MODES:
            raise ValueError(
                f"Mode {image.mode} cannot be preserved by this PNG helper. "
                "Export a supported final tile explicitly, then check it again. "
                f"Supported modes: {', '.join(sorted(PNG_MODES))}"
            )
        image.load()
        tile = image.copy()
        metadata = dict(image.info)

    width, height = tile.size
    radius = min(context, width, height)
    preview = Image.new(tile.mode, (3 * width, 3 * height))
    if tile.palette is not None:
        # putpalette updates both Pillow's metadata and its native image core.
        preview.putpalette(tile.getpalette(), rawmode="RGB")
    # Integer placement; no mask: alpha and even hidden RGB samples are copied.
    for row in range(3):
        for column in range(3):
            preview.paste(tile, (column * width, row * height))

    options = png_options(tile)
    output_dir.mkdir(parents=True, exist_ok=False)
    preview_path = output_dir / "preview-3x3.png"
    preview.save(preview_path, format="PNG", **options)
    verify_saved(preview_path, preview, metadata)

    # Verify every copy from the encoded-and-reopened PNG, not just in memory.
    with Image.open(preview_path) as saved:
        saved.load()
        for row in range(3):
            for column in range(3):
                left, top = column * width, row * height
                copy = saved.crop((left, top, left + width, top + height))
                if copy.mode != tile.mode or copy.tobytes() != tile.tobytes():
                    raise RuntimeError(f"Saved copy at ({column}, {row}) is not exact")

    crops = []

    def save_crop(name, box, kind, boundary):
        crop = preview.crop(box)
        path = output_dir / name
        crop.save(path, format="PNG", **options)
        verify_saved(path, crop, metadata)
        crops.append({
            "file": name,
            "kind": kind,
            "box_xyxy": list(box),
            "boundary_in_preview": boundary,
            "sha256": sha256(path),
        })

    for index in (1, 2):
        x, y = index * width, index * height
        save_crop(
            f"seam-vertical-{index}.png",
            (x - radius, 0, x + radius, 3 * height),
            "vertical_seam", {"x": x},
        )
        save_crop(
            f"seam-horizontal-{index}.png",
            (0, y - radius, 3 * width, y + radius),
            "horizontal_seam", {"y": y},
        )
    for row in (1, 2):
        for column in (1, 2):
            x, y = column * width, row * height
            save_crop(
                f"junction-{column}-{row}.png",
                (x - radius, y - radius, x + radius, y + radius),
                "four_tile_junction", {"x": x, "y": y},
            )

    shifted, shift_xy = cyclic_offset(tile)
    shifted_path = output_dir / "diagnostic-offset.png"
    shifted.save(shifted_path, format="PNG", **options)
    verify_saved(shifted_path, shifted, metadata)
    diagnostics = {
        "cyclic_offset": {
            "file": shifted_path.name, "sha256": sha256(shifted_path),
            "shift_xy": shift_xy, "size": list(tile.size),
            "purpose": "Moves the original edges into the interior. Pixel permutation only; does not repair the tile or replace exact 3x3 QA.",
        },
    }
    if background is not None:
        diagnostics["background_distribution"] = background_distribution(tile, background)

    if sha256(source) != source_digest:
        raise RuntimeError("Source changed during inspection; check the final file again")
    report = {
        "source": {
            "path": str(source), "sha256": source_digest,
            "size": [width, height], "mode": tile.mode,
        },
        "assembly": {
            "file": preview_path.name,
            "sha256": sha256(preview_path),
            "size": [3 * width, 3 * height],
            "grid": [3, 3], "gap_pixels": 0,
            "scale": 1, "pixel_exact_copies_verified": True,
            "copy_count_verified": 9,
            "color_and_transparency_metadata_verified": True,
        },
        "geometry": {
            "coordinate_system": "zero-based pixel boundaries in preview",
            "vertical_seams_x": [width, 2 * width],
            "horizontal_seams_y": [height, 2 * height],
            "four_tile_junctions_xy": [
                [width, height], [2 * width, height],
                [width, 2 * height], [2 * width, 2 * height],
            ],
            "crop_context_pixels": radius,
        },
        "crops": crops,
        "diagnostics": diagnostics,
        "visual_review": {
            "status": "pending", "vertical_seams": "pending",
            "horizontal_seams": "pending", "four_tile_junctions": "pending",
            "repeat_distribution": "pending",
            "viewing_scale": None, "defects": [],
            "limitations": [
                "This helper verifies exact copies, not visual seamlessness. "
                "Open the preview, all seam crops and all four junction crops."
            ],
        },
    }
    report_path = output_dir / "inspection.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tile", type=Path, help="Final static tile file")
    parser.add_argument("--output-dir", required=True, type=Path, help="New result directory")
    parser.add_argument(
        "--context", type=int, default=64,
        help="Pixels on each side of a seam crop; not a quality threshold",
    )
    parser.add_argument(
        "--background", help="Optional diagnostic background: #RRGGBB or transparent. No quality threshold.",
    )
    args = parser.parse_args()
    try:
        background = None
        if args.background == "transparent":
            background = (0, 0, 0, 0)
        elif args.background is not None:
            value = args.background
            if len(value) != 7 or not value.startswith("#"):
                raise ValueError("--background must be #RRGGBB or transparent")
            background = tuple(int(value[index:index + 2], 16) for index in (1, 3, 5)) + (255,)
        report = inspect_tile(args.tile, args.output_dir, args.context, background)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Inspection failed: {error}", file=sys.stderr)
        return 1
    print(f"Exact-copy assembly verified: {report}")
    print("Visual seamlessness: NOT CHECKED. Inspect the preview and all eight crops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
