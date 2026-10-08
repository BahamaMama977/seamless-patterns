#!/usr/bin/env python3
"""Compose local transparent motifs on a torus and build exact-copy QA.

Original implementation for seamless-patterns. Python 3.9+ and Pillow;
no generator, network, service, API key, or third-party JavaScript runtime.
"""

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageOps, __version__ as PILLOW_VERSION
except ImportError:
    raise SystemExit("Pillow is unavailable. Use a local Python with Pillow.")

from inspect_tile import inspect_tile, sha256


RENDERER = "seamless-patterns-pillow-v3"


def integer(value, label, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def number(value, label, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid {label}")
    return value


def only_keys(mapping, allowed, label):
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")


def background_rgba(value):
    if isinstance(value, str) and len(value) == 7 and value.startswith("#"):
        try:
            return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5)) + (255,)
        except ValueError:
            pass
    if isinstance(value, list) and len(value) == 4:
        if all(type(channel) is int and 0 <= channel <= 255 for channel in value):
            return tuple(value)
    raise ValueError("background must be #RRGGBB or four integer RGBA channels (0..255)")


def load_assets(specs, base):
    if not isinstance(specs, list) or not specs:
        raise ValueError("assets must be a non-empty list of local transparent motifs")
    assets = {}
    profiles = set()
    for spec in specs:
        only_keys(spec, ("id", "path", "sha256"), "asset")
        asset_id = spec.get("id")
        if not isinstance(asset_id, str) or not asset_id or asset_id in assets:
            raise ValueError("Asset IDs must be unique non-empty strings")
        if not isinstance(spec.get("path"), str) or "://" in spec["path"]:
            raise ValueError("Asset path must be a local file path")
        path = (base / spec["path"]).resolve(strict=True)
        digest = sha256(path)
        if spec.get("sha256", digest) != digest:
            raise ValueError(f"Asset hash mismatch: {asset_id}")
        with Image.open(path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError(f"Export a static motif first: {asset_id}")
            if source.getexif().get(274, 1) not in (None, 1):
                raise ValueError(f"Export the oriented motif first: {asset_id}")
            if "A" not in source.getbands() and "transparency" not in source.info:
                raise ValueError(f"Real alpha transparency is required: {asset_id}")
            source.load()
            profile = source.info.get("icc_profile")
            profiles.add(profile)
            rgba = source.convert("RGBA")
        alpha = rgba.getchannel("A")
        bounds = alpha.getbbox()
        if bounds is None or alpha.getextrema()[0] != 0:
            raise ValueError(f"Motif needs visible pixels and a transparent exterior: {asset_id}")
        # A complete cutout must not be clipped by its own source canvas.
        if bounds[0] == 0 or bounds[1] == 0 or bounds[2] == rgba.width or bounds[3] == rgba.height:
            raise ValueError(f"Motif touches its source edge; inspect/export a complete cutout: {asset_id}")
        assets[asset_id] = {
            "image": rgba.crop(bounds), "path": path, "sha256": digest,
            "source_size": list(rgba.size), "alpha_bbox_xyxy": list(bounds),
            "icc_profile": profile,
        }
    if len(profiles) > 1:
        raise ValueError("Motifs have different color profiles. Normalize them explicitly before composition.")
    return assets


def value_range(value, label, integral=False):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be [minimum, maximum]")
    validator = integer if integral else number
    lower = validator(value[0], label, 1 if integral else None)
    upper = validator(value[1], label, 1 if integral else None)
    if lower > upper:
        raise ValueError(f"Reversed {label} range")
    return lower, upper


def generate_placements(layout, seed, width, height, assets, object_spacing=None):
    only_keys(layout, ("count", "size_px", "rotation_deg", "min_center_distance_px", "max_attempts"), "layout")
    count = integer(layout.get("count"), "layout.count", 1)
    size_min, size_max = value_range(layout.get("size_px"), "layout.size_px", integral=True)
    angle_min, angle_max = value_range(layout.get("rotation_deg", [0, 0]), "layout.rotation_deg")
    spacing = number(layout.get("min_center_distance_px", 0), "layout.min_center_distance_px", 0)
    attempts = integer(layout.get("max_attempts", count * 200), "layout.max_attempts", 1)
    rng = random.Random(seed)
    ids = list(assets)
    placements = []
    clearance = ObjectSpacing(width, height, object_spacing) if object_spacing is not None else None
    for _ in range(attempts):
        x, y = rng.randrange(width), rng.randrange(height)
        # Minimum distance on the torus: neighbors across an edge count too.
        if any(
            min(abs(x - item["x"]), width - abs(x - item["x"])) ** 2
            + min(abs(y - item["y"]), height - abs(y - item["y"])) ** 2 < spacing ** 2
            for item in placements
        ):
            continue
        item = {
            "asset_id": rng.choice(ids), "x": x, "y": y,
            "size_px": rng.randint(size_min, size_max),
            "rotation_deg": rng.uniform(angle_min, angle_max),
        }
        if clearance is not None:
            motif = prepare_motif(assets[item["asset_id"]]["image"], item["size_px"], item["rotation_deg"])
            if not clearance.try_add(motif, x, y):
                continue
        placements.append(item)
        if len(placements) == count:
            return placements
    raise ValueError(
        f"Placed only {len(placements)} of {count} motifs within {attempts} attempts. "
        "Reduce spacing/object gap/count or sizes, or increase the tile size/attempt budget explicitly."
    )


def prepare_motif(image, size, angle):
    """Transform once, then reuse identical raster pixels for every wrapped copy."""
    factor = size / max(image.size)
    target = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    transformed = image.resize(target, Image.Resampling.LANCZOS)
    if angle % 360:
        transformed = transformed.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    return transformed


def wrapped_origins(left, top, motif_width, motif_height, width, height):
    """All integer-period translations intersecting the tile, including corners.

    Supports motifs larger than one period; no hard-coded +/-1 copy limit.
    Strict interval intersections avoid duplicate compositing at a touching edge.
    """
    for row in range((-top - motif_height) // height + 1, (height - 1 - top) // height + 1):
        for column in range((-left - motif_width) // width + 1, (width - 1 - left) // width + 1):
            yield left + column * width, top + row * height


def masks_intersect(first, first_xy, second, second_xy):
    """Test actual nonzero mask pixels in the overlapping rectangles only."""
    ax, ay = first_xy
    bx, by = second_xy
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + first.width, bx + second.width), min(ay + first.height, by + second.height)
    if left >= right or top >= bottom:
        return False
    a = first.crop((left - ax, top - ay, right - ax, bottom - ay))
    b = second.crop((left - bx, top - by, right - bx, bottom - by))
    return ImageChops.multiply(a, b).getbbox() is not None


def expand_square_mask(mask, gap):
    """Exact square dilation with logarithmic shifts rather than a huge kernel.

    The zero border holds every intermediate shift; offset wraparound therefore
    transfers only zero pixels. Each pass extends a contiguous covered interval.
    """
    expanded = ImageOps.expand(mask, gap, fill=0)
    for axis in ((1, 0), (0, 1)):
        covered = 0
        while covered < gap:
            step = min(gap - covered, covered + 1)
            positive = ImageChops.offset(expanded, axis[0] * step, axis[1] * step)
            negative = ImageChops.offset(expanded, -axis[0] * step, -axis[1] * step)
            expanded = ImageChops.lighter(ImageChops.lighter(expanded, positive), negative)
            covered += step
    return expanded


class ObjectSpacing:
    """Optional pixel-silhouette clearance, including periodic neighbors.

    Square dilation is conservative for diagonal distances. It enforces at
    least min_gap_px empty pixel columns/rows between the selected alpha
    silhouettes. No restriction is applied when object_spacing is absent.
    """

    def __init__(self, width, height, spec):
        only_keys(spec, ("min_gap_px", "alpha_threshold"), "object_spacing")
        self.gap = integer(spec.get("min_gap_px"), "object_spacing.min_gap_px", 0)
        self.threshold = integer(spec.get("alpha_threshold", 1), "object_spacing.alpha_threshold", 1)
        if self.threshold > 255:
            raise ValueError("object_spacing.alpha_threshold must be <= 255")
        if self.gap >= min(width, height):
            raise ValueError("Object gap cannot fit between periodic copies of any visible pixel")
        self.width, self.height = width, height
        self.occupied = Image.new("L", (width, height), 0)

    def fragments(self, mask, left, top):
        for ox, oy in wrapped_origins(left, top, mask.width, mask.height, self.width, self.height):
            box = (max(0, ox), max(0, oy), min(self.width, ox + mask.width), min(self.height, oy + mask.height))
            part = mask.crop((box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy))
            yield part, box

    def try_add(self, motif, x, y):
        mask = motif.getchannel("A").point(lambda value: 255 if value >= self.threshold else 0)
        expanded = mask
        if self.gap:
            expanded = expand_square_mask(mask, self.gap)
        # A single large motif can also collide with its own translated copies.
        cols = math.ceil((mask.width + self.gap) / self.width)
        rows = math.ceil((mask.height + self.gap) / self.height)
        for row in range(-rows, rows + 1):
            for col in range(-cols, cols + 1):
                if (row or col) and masks_intersect(
                    expanded, (-self.gap, -self.gap), mask, (col * self.width, row * self.height)
                ):
                    return False
        left, top = x % self.width - motif.width // 2, y % self.height - motif.height // 2
        for part, box in self.fragments(expanded, left - self.gap, top - self.gap):
            if ImageChops.multiply(part, self.occupied.crop(box)).getbbox() is not None:
                return False
        # Only accepted silhouettes are committed, without expanding both sides.
        for part, box in self.fragments(mask, left, top):
            self.occupied.paste(255, box, part)
        return True


def render_placements(assets, placements, width, height, background, object_spacing=None):
    tile = Image.new("RGBA", (width, height), background)
    resolved = []
    if not isinstance(placements, list) or not placements:
        raise ValueError("placements must be a non-empty list")
    clearance = ObjectSpacing(width, height, object_spacing) if object_spacing is not None else None
    # Preserve layer order: all wrapped copies of one motif precede the next motif.
    for item in placements:
        only_keys(item, ("asset_id", "x", "y", "size_px", "rotation_deg", "opacity"), "placement")
        asset_id = item.get("asset_id")
        if not isinstance(asset_id, str) or asset_id not in assets:
            raise ValueError(f"Unknown placement asset: {asset_id}")
        x = integer(item.get("x"), "placement.x") % width
        y = integer(item.get("y"), "placement.y") % height
        size = integer(item.get("size_px"), "placement.size_px", 1)
        angle = number(item.get("rotation_deg", 0), "placement.rotation_deg")
        opacity = number(item.get("opacity", 1), "placement.opacity", 0)
        if opacity > 1:
            raise ValueError("placement.opacity must be between 0 and 1")
        motif = prepare_motif(assets[asset_id]["image"], size, angle)
        if opacity != 1:
            # Scale existing alpha once; preserve RGB, including white details.
            motif.putalpha(motif.getchannel("A").point([
                int(value * opacity + 0.5) for value in range(256)
            ]))
        if clearance is not None and not clearance.try_add(motif, x, y):
            raise ValueError(f"Object spacing violated at placement {len(resolved)}: check sizes, rotation, gap and periodic neighbors")
        left, top = x - motif.width // 2, y - motif.height // 2
        origins = list(wrapped_origins(left, top, motif.width, motif.height, width, height))
        for origin in origins:
            # Source-over only in the COMPOSITION; never in the QA preview.
            tile.alpha_composite(motif, dest=origin)
        resolved.append({
            "asset_id": asset_id, "x": x, "y": y, "size_px": size,
            "rotation_deg": angle, "opacity": opacity, "raster_size": list(motif.size),
            "raster_sha256": hashlib.sha256(motif.tobytes()).hexdigest(),
            "origins_xy": [list(origin) for origin in origins],
        })
    return tile, resolved


def compose_pattern(config_path, output_dir):
    config_path = Path(config_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    only_keys(config, ("schema_version", "width", "height", "background", "seed", "assets", "layout", "placements", "object_spacing"), "config")
    if config.get("schema_version") != 1:
        raise ValueError("Only schema_version 1 is supported")
    width, height = integer(config.get("width"), "width", 1), integer(config.get("height"), "height", 1)
    background = background_rgba(config.get("background"))
    seed = integer(config.get("seed"), "seed")
    if ("layout" in config) == ("placements" in config):
        raise ValueError("Provide exactly one of layout or placements")
    assets = load_assets(config.get("assets"), config_path.parent)
    object_spacing = config.get("object_spacing")
    if "object_spacing" in config and object_spacing is None:
        raise ValueError("object_spacing must be an object; omit it to allow overlaps")
    placements = config.get("placements") if "placements" in config else generate_placements(
        config["layout"], seed, width, height, assets, object_spacing
    )
    tile, resolved = render_placements(assets, placements, width, height, background, object_spacing)
    # Persist concrete placements and pinned asset hashes, not just a seed.
    replay = {
        "schema_version": 1, "width": width, "height": height,
        "background": list(background), "seed": seed,
        "assets": [
            {"id": key, "path": str(asset["path"]), "sha256": asset["sha256"]}
            for key, asset in assets.items()
        ],
        "placements": [
            {key: item[key] for key in ("asset_id", "x", "y", "size_px", "rotation_deg", "opacity")}
            for item in resolved
        ],
    }
    if object_spacing is not None:
        replay["object_spacing"] = object_spacing
    output_dir.mkdir(parents=True, exist_ok=False)
    tile_path = output_dir / "tile.png"
    options = {}
    profile = next(iter(assets.values()))["icc_profile"]
    if profile is not None:
        options["icc_profile"] = profile
    tile.save(tile_path, **options)
    with Image.open(tile_path) as saved:
        saved.load()
        if saved.mode != tile.mode or saved.size != tile.size or saved.tobytes() != tile.tobytes():
            raise RuntimeError("Export changed the composed tile pixels")
    for asset in assets.values():
        if sha256(asset["path"]) != asset["sha256"]:
            raise RuntimeError("Asset changed during composition; rebuild from the final asset")
    replay_path = output_dir / "replay-config.json"
    replay_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    check = inspect_tile(tile_path, output_dir / "check", background=background)
    manifest = {
        "renderer": {
            "id": RENDERER, "script_sha256": sha256(Path(__file__)),
            "python": platform.python_version(), "pillow": PILLOW_VERSION,
            "transforms": "trim transparent exterior; LANCZOS resize; BICUBIC expanded rotation; rounded alpha multiplication by placement opacity; integer source-over wrapping",
        },
        "request_config": config, "replay_config": replay_path.name,
        "assets": [
            {"id": key, **{field: str(value) if field == "path" else value
                           for field, value in asset.items() if field not in ("image", "icc_profile")}}
            for key, asset in assets.items()
        ],
        "placements": resolved,
        "output": {"file": tile_path.name, "size": list(tile.size), "mode": tile.mode, "sha256": sha256(tile_path)},
        "inspection": str(check.relative_to(output_dir)),
        "visual_review_status": "pending",
        "limitations": [
            "Periodic placement does not prove visually even distribution or attractive repetition.",
            "Opacity controls compositing, not artistic style or reference similarity.",
            "Seed controls local placement only, not image generation.",
            "Pixel reproducibility also requires the recorded renderer/runtime and unchanged assets.",
        ],
    }
    if object_spacing is not None:
        manifest["object_spacing_verification"] = {
            "config": object_spacing, "transformed_alpha_masks_verified": True,
            "periodic_neighbors_and_self_copies_checked": True,
            "distance_rule": "square pixel dilation; conservative for diagonals",
            "limitations": "Checks alpha at the selected threshold, not optical spacing or visual rhythm.",
        }
    manifest_path = output_dir / "composition.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Local JSON configuration")
    parser.add_argument("--output-dir", required=True, type=Path, help="New result directory")
    args = parser.parse_args()
    try:
        result = compose_pattern(args.config, args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Composition failed: {error}", file=sys.stderr)
        return 1
    print(f"Periodic composition and exact-copy preview saved: {result}")
    print("Visual seamlessness and repeat distribution: NOT CHECKED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
