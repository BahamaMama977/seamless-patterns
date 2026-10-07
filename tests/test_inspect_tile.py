"""Behavioral checks outside the distributable skill; no network/generation."""

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageCms, PngImagePlugin

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT.parent / "seamless-patterns" / "scripts" / "inspect_tile.py"
spec = importlib.util.spec_from_file_location("inspect_tile", SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class ExactCopyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tile-tests-", dir=ROOT)
        self.work = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def check_image(self, tile, name="tile.png", **save_options):
        source = self.work / name
        tile.save(source, **save_options)
        source_bytes = source.read_bytes()
        output = self.work / (source.stem + "-check")
        report_path = helper.inspect_tile(source, output, context=3)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(source_bytes, source.read_bytes())
        self.assertEqual(report["source"]["sha256"], helper.sha256(source))
        self.assertTrue(report["assembly"]["pixel_exact_copies_verified"])
        self.assertEqual(report["visual_review"]["status"], "pending")
        self.assertEqual(report["visual_review"]["repeat_distribution"], "pending")
        self.assertEqual(report["visual_review"]["defects"], [])
        self.assertEqual(len(report["crops"]), 8)
        width, height = tile.size
        self.assertEqual(report["geometry"]["vertical_seams_x"], [width, 2 * width])
        self.assertEqual(report["geometry"]["horizontal_seams_y"], [height, 2 * height])
        self.assertEqual(report["geometry"]["four_tile_junctions_xy"], [
            [width, height], [2 * width, height],
            [width, 2 * height], [2 * width, 2 * height],
        ])
        with Image.open(source) as decoded, Image.open(output / "preview-3x3.png") as preview:
            self.assertEqual(preview.size, (3 * width, 3 * height))
            self.assertEqual(preview.mode, decoded.mode)
            self.assertEqual(preview.getpalette(), decoded.getpalette())
            for key in ("transparency", "icc_profile", "gamma", "srgb", "chromaticity"):
                self.assertEqual(preview.info.get(key), decoded.info.get(key))
            for row in range(3):
                for column in range(3):
                    block = preview.crop((column * width, row * height,
                                          (column + 1) * width, (row + 1) * height))
                    self.assertEqual(block.tobytes(), decoded.tobytes())
            for crop_report in report["crops"]:
                with Image.open(output / crop_report["file"]) as crop:
                    expected = preview.crop(crop_report["box_xyxy"])
                    self.assertEqual(crop.size, expected.size)
                    self.assertEqual(crop.tobytes(), expected.tobytes())
                    self.assertEqual(crop_report["sha256"], helper.sha256(output / crop_report["file"]))
            offset_info = report["diagnostics"]["cyclic_offset"]
            dx, dy = offset_info["shift_xy"]
            with Image.open(output / offset_info["file"]) as offset:
                self.assertEqual(offset.mode, decoded.mode)
                self.assertEqual(offset.getpalette(), decoded.getpalette())
                for y in range(height):
                    for x in range(width):
                        self.assertEqual(offset.getpixel((x, y)), decoded.getpixel(((x - dx) % width, (y - dy) % height)))
        return source, output

    def test_rgb_nonsquare_with_obvious_discontinuities(self):
        tile = Image.new("RGB", (7, 5))
        tile.putdata([(x * 37 % 256, y * 59 % 256, (x + 3 * y) * 23 % 256)
                      for y in range(5) for x in range(7)])
        self.check_image(tile)

    def test_rgba_including_hidden_rgb_and_partial_alpha(self):
        tile = Image.new("RGBA", (9, 7))
        tile.putdata([(x * 29 % 256, y * 43 % 256, 193, (x + y) * 51 % 256)
                      for y in range(7) for x in range(9)])
        tile.putpixel((0, 0), (123, 87, 45, 0))
        self.check_image(tile)

    def test_palette_transparency(self):
        for alpha in (2, bytes([0, 128, 255, 75])):
            with self.subTest(alpha=alpha):
                tile = Image.new("P", (7, 5))
                tile.putpalette([component for index in range(256)
                                 for component in (index, 255 - index, index * 3 % 256)])
                tile.putdata([(x + y) % 4 for y in range(5) for x in range(7)])
                self.check_image(tile, name=f"palette-{type(alpha).__name__}.png", transparency=alpha)

    def test_bilevel_grayscale_and_la(self):
        for mode in ("1", "L", "LA"):
            with self.subTest(mode=mode):
                tile = Image.new(mode, (7, 5))
                values = [((x + y) % 2 * 255) for y in range(5) for x in range(7)]
                tile.putdata([(value, 255 - value) for value in values] if mode == "LA" else values)
                self.check_image(tile, name=f"mode-{mode}.png")

    def test_16_bit_native_samples(self):
        samples = [((index * 2003) % 65536) for index in range(35)]
        tile = Image.frombytes("I;16", (7, 5), struct.pack("<35H", *samples))
        self.check_image(tile)

    def test_small_tiles_and_context_clamping(self):
        for size in ((1, 1), (1, 4), (5, 1)):
            with self.subTest(size=size):
                self.check_image(Image.new("RGBA", size, (31, 57, 89, 17)),
                                 name=f"tiny-{size[0]}-{size[1]}.png")

    def test_color_profile_and_png_color_chunks(self):
        tile = Image.new("RGB", (8, 6), (95, 113, 155))
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        chunks = PngImagePlugin.PngInfo()
        chunks.add(b"gAMA", struct.pack(">I", 45455))
        chunks.add(b"cHRM", struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000))
        self.check_image(tile, icc_profile=profile, pnginfo=chunks)
        chunks = PngImagePlugin.PngInfo()
        chunks.add(b"sRGB", bytes([0]))
        self.check_image(tile, name="srgb.png", pnginfo=chunks)

    def test_jpeg_means_exact_decoded_pixels(self):
        tile = Image.new("RGB", (11, 7))
        tile.putdata([(x * 23, y * 31, 133) for y in range(7) for x in range(11)])
        self.check_image(tile, name="tile.jpg", quality=83)

    def test_existing_destination_is_not_overwritten(self):
        source, output = self.check_image(Image.new("RGB", (3, 2), (19, 31, 47)))
        before = {file.name: file.read_bytes() for file in output.iterdir()}
        with self.assertRaisesRegex(ValueError, "already exists"):
            helper.inspect_tile(source, output)
        self.assertEqual(before, {file.name: file.read_bytes() for file in output.iterdir()})

    def test_unsupported_mode_animation_and_orientation_fail_without_output(self):
        cmyk = self.work / "cmyk.tif"
        Image.new("CMYK", (4, 3)).save(cmyk)
        animated = self.work / "animated.gif"
        Image.new("RGB", (4, 3), "red").save(animated, save_all=True,
                                             append_images=[Image.new("RGB", (4, 3), "blue")])
        oriented = self.work / "oriented.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (4, 3)).save(oriented, exif=exif)
        for source in (cmyk, animated, oriented):
            with self.subTest(file=source.name):
                output = self.work / (source.stem + "-check")
                original = source.read_bytes()
                with self.assertRaises(ValueError):
                    helper.inspect_tile(source, output)
                self.assertFalse(output.exists())
                self.assertEqual(source.read_bytes(), original)

    def test_cli_reports_unchecked_visual_state_and_fails_cleanly(self):
        source = self.work / "tile.png"
        Image.new("RGB", (5, 3), "green").save(source)
        output = self.work / "cli-check"
        result = subprocess.run([sys.executable, str(SCRIPT), str(source),
                                 "--output-dir", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NOT CHECKED", result.stdout)
        result = subprocess.run([sys.executable, str(SCRIPT), str(source),
                                 "--output-dir", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("already exists", result.stderr)
        result = subprocess.run([sys.executable, str(SCRIPT), str(source),
                                 "--output-dir", str(self.work / "invalid"), "--context", "0"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.work / "invalid").exists())

    def test_white_bands_are_described_without_automatic_visual_verdict(self):
        tile = Image.new("RGBA", (10, 8), "white")
        for y in range(2, 6):
            for x in range(3, 7):
                tile.putpixel((x, y), (200, 10, 20, 255))
        source = self.work / "white-edges.png"
        tile.save(source)
        report = json.loads(helper.inspect_tile(source, self.work / "bands", background=(255, 255, 255, 255)).read_text())
        distribution = report["diagnostics"]["background_distribution"]
        self.assertEqual(distribution["fully_background_row_runs_y"], [[0, 2], [6, 8]])
        self.assertEqual(distribution["fully_background_column_runs_x"], [[0, 3], [7, 10]])
        self.assertEqual(distribution["background_fraction_by_row"][3], 0.6)
        self.assertEqual(distribution["verdict"], "not_assessed")
        self.assertEqual(report["visual_review"]["status"], "pending")

    def test_transparent_distribution_ignores_hidden_rgb(self):
        tile = Image.new("RGBA", (4, 3), (100, 60, 30, 0))
        tile.putpixel((1, 1), (100, 60, 30, 1))
        distribution = helper.background_distribution(tile, (0, 0, 0, 0))
        self.assertEqual(distribution["fully_background_row_runs_y"], [[0, 1], [2, 3]])
        self.assertEqual(distribution["background_fraction_by_row"], [1, 0.75, 1])

    def test_invalid_background_does_not_create_an_output_directory(self):
        source = self.work / "tile.png"
        Image.new("RGB", (4, 3), "white").save(source)
        for index, background in enumerate(((0, 0, 0), (256, 0, 0, 255), "white")):
            output = self.work / f"bad-background-{index}"
            with self.assertRaisesRegex(ValueError, "Diagnostic background"):
                helper.inspect_tile(source, output, background=background)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
